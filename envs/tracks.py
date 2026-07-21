"""Track geometry and deterministic vehicle placement.

Track A is the training + primary-evaluation circuit (highway-env's stock
``racetrack-v0`` geometry). Track C is held-out geometry, used *only* for the
Phase 7 transfer ablation -- no policy trains on it.

This module also owns the racetrack subclass that fixes two behaviours of stock
``RacetrackEnv._make_vehicles`` which are incompatible with the protocol:

1. Stock spawns ``rng.integers(other_vehicles)`` traffic vehicles -- a *random*
   count in [0, N). CLAUDE.md §3 requires fixed density (4), identical across
   every training and evaluation condition, or traffic density becomes an
   uncontrolled confound between PPO and GRPO runs.
2. Stock rejection-samples spawn positions and silently *drops* any vehicle that
   lands too close to another, so even the random count is not honoured. We
   resample until the count is met.

It also places controlled vehicles at explicit, swappable grid slots so the
head-to-head harness can run paired episodes with starting positions exchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.config import TrackName

# Longitudinal offsets (metres along the start straight) of the two head-to-head
# grid slots. Swapping which agent gets which slot is how paired episodes control
# for start-position advantage (CLAUDE.md §5, Phase 5).
GRID_SLOTS: tuple[tuple[str, str, int, float], ...] = (
    ("a", "b", 0, 30.0),
    ("a", "b", 1, 20.0),
)

_MIN_SPAWN_SEPARATION = 15.0
_TRAFFIC_SPEED_RANGE = (6.0, 9.0)


@dataclass(frozen=True)
class TrackSpec:
    """One circuit's identity and highway-env overrides."""

    name: TrackName
    base_env_id: str
    highway_overrides: dict[str, Any] = field(default_factory=dict)


_SPECS: dict[TrackName, TrackSpec] = {
    # Stock racetrack-v0 geometry: 9 segments (a..i), 2 lanes, ~348 m lap.
    TrackName.A: TrackSpec(name=TrackName.A, base_env_id="racetrack-v0"),
    # Held-out geometry: longer straights, different curvature. Transfer only.
    TrackName.C: TrackSpec(name=TrackName.C, base_env_id="racetrack-large-v0"),
}


def track_spec(track: TrackName) -> TrackSpec:
    """Return the spec for ``track``."""

    return _SPECS[TrackName(track)]


_ENV_CLASS_CACHE: dict[TrackName, type] = {}


def racetrack_class(track: TrackName) -> type:
    """Build (and cache) the racetrack subclass for ``track``.

    Lazy so this module stays importable without highway-env installed.
    """

    track = TrackName(track)
    if track in _ENV_CLASS_CACHE:
        return _ENV_CLASS_CACHE[track]

    from highway_env.envs.racetrack_env import RacetrackEnv, RacetrackEnvLarge
    from highway_env.vehicle.behavior import IDMVehicle

    base = {TrackName.A: RacetrackEnv, TrackName.C: RacetrackEnvLarge}[track]

    class FixedTrafficRacetrack(base):  # type: ignore[valid-type,misc]
        """Racetrack with deterministic grid starts and fixed traffic density."""

        def _make_vehicles(self) -> None:
            rng = self.np_random
            n_controlled = int(self.config["controlled_vehicles"])
            n_traffic = int(self.config["other_vehicles"])

            self.controlled_vehicles = []
            for i in range(n_controlled):
                if n_controlled > 1:
                    # Explicit, swappable grid slots for head-to-head racing.
                    from_, to_, lane, longitudinal = GRID_SLOTS[i % len(GRID_SLOTS)]
                    lane_index = (from_, to_, lane)
                else:
                    # Solo: randomised start along the opening straight, so the
                    # policy sees the whole track rather than memorising one pose.
                    lane_index = ("a", "b", int(rng.integers(2)))
                    longitudinal = float(rng.uniform(20, 50))

                vehicle = self.action_type.vehicle_class.make_on_lane(
                    self.road, lane_index, speed=None, longitudinal=longitudinal
                )
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

            self._spawn_traffic(rng, n_traffic)

        def _spawn_traffic(self, rng, n_traffic: int) -> None:
            """Spawn exactly ``n_traffic`` IDM vehicles, seeded deterministically.

            Resamples rejected positions rather than dropping the vehicle, so the
            density the config asks for is the density every episode gets.
            """

            placed = 0
            attempts = 0
            max_attempts = 200 * max(n_traffic, 1)

            while placed < n_traffic and attempts < max_attempts:
                attempts += 1
                lane_index = self.road.network.random_lane_index(rng)
                lane = self.road.network.get_lane(lane_index)
                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    lane_index,
                    longitudinal=float(rng.uniform(low=0.0, high=lane.length)),
                    speed=float(rng.uniform(*_TRAFFIC_SPEED_RANGE)),
                )
                if all(
                    np.linalg.norm(vehicle.position - other.position)
                    >= _MIN_SPAWN_SEPARATION
                    for other in self.road.vehicles
                ):
                    self.road.vehicles.append(vehicle)
                    placed += 1

            if placed < n_traffic:  # pragma: no cover - geometry is roomy enough
                raise RuntimeError(
                    f"could only place {placed}/{n_traffic} traffic vehicles on "
                    f"track {track.value}; loosen _MIN_SPAWN_SEPARATION"
                )

    FixedTrafficRacetrack.__name__ = f"FixedTrafficRacetrack{track.value.upper()}"
    _ENV_CLASS_CACHE[track] = FixedTrafficRacetrack
    return FixedTrafficRacetrack
