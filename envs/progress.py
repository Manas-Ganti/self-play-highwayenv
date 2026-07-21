"""Lap-progress tracking on a circuit road network.

The reward's dominant term is lap progress, and the head-to-head win condition is
"most progress at timeout", so this is the most load-bearing measurement in the
project. highway-env gives us only a lane-local longitudinal coordinate, so we
lift it to a monotone distance-travelled signal:

1. Walk the road-network graph from node ``a`` to recover the segment cycle.
2. Accumulate segment lengths to get each segment's start offset around the lap.
3. Arc position ``s = offset[segment] + longitudinal`` lies in [0, lap_length).
4. Track wrap-around across steps to convert ``s`` into unbounded distance.

Wrap detection uses a half-lap heuristic: a backwards jump of more than half a
lap in one step is a lap completion, not reverse driving. A policy would have to
travel >174 m backwards in a single 0.2 s step to fool it, which is impossible at
these speeds.
"""

from __future__ import annotations

import numpy as np


class TrackGeometry:
    """Cached arc-length parameterisation of a circuit road network."""

    def __init__(self, network, start_node: str = "a") -> None:
        self.network = network
        self.segment_offsets: dict[tuple[str, str], float] = {}

        node = start_node
        offset = 0.0
        visited: set[str] = set()
        while node not in visited:
            visited.add(node)
            successors = list(network.graph[node].keys())
            if len(successors) != 1:
                raise ValueError(
                    f"node '{node}' has {len(successors)} successors; "
                    "lap progress assumes a simple circuit"
                )
            nxt = successors[0]
            self.segment_offsets[(node, nxt)] = offset
            # Lanes of a segment are parallel; lane 0's length defines the segment.
            offset += network.graph[node][nxt][0].length
            node = nxt

        if node != start_node:
            raise ValueError("road network is not a closed circuit")
        self.lap_length = offset

    def arc_position(self, vehicle) -> float:
        """Arc position of ``vehicle`` around the lap, in [0, lap_length)."""

        lane_index = vehicle.lane_index
        segment = (lane_index[0], lane_index[1])
        offset = self.segment_offsets.get(segment)
        if offset is None:  # pragma: no cover - vehicle off the known circuit
            return 0.0
        lane = self.network.get_lane(lane_index)
        longitudinal, _ = lane.local_coordinates(vehicle.position)
        longitudinal = float(np.clip(longitudinal, 0.0, lane.length))
        return (offset + longitudinal) % self.lap_length


class ProgressTracker:
    """Unwraps a vehicle's arc position into cumulative distance travelled."""

    def __init__(self, geometry: TrackGeometry) -> None:
        self.geometry = geometry
        self._prev_arc: float | None = None
        self._laps = 0

    def reset(self, vehicle) -> None:
        """Re-anchor the tracker at the vehicle's spawn position."""

        self._prev_arc = self.geometry.arc_position(vehicle)
        self._laps = 0

    def update(self, vehicle) -> float:
        """Advance the tracker one step; return cumulative distance in metres."""

        arc = self.geometry.arc_position(vehicle)
        lap_length = self.geometry.lap_length
        if self._prev_arc is not None:
            delta = arc - self._prev_arc
            if delta < -lap_length / 2:
                self._laps += 1
            elif delta > lap_length / 2:
                # Symmetric case: crossed the start line backwards.
                self._laps -= 1
        self._prev_arc = arc
        return self.distance

    @property
    def distance(self) -> float:
        """Cumulative signed distance travelled since reset, in metres."""

        if self._prev_arc is None:
            return 0.0
        return self._laps * self.geometry.lap_length + self._prev_arc

    @property
    def laps(self) -> int:
        """Completed laps (can go negative if the vehicle drives backwards)."""

        return self._laps
