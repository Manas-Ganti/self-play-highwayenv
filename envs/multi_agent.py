"""Two-agent head-to-head racetrack (PettingZoo Parallel API).

Used **only** for evaluation (Phase 5) and the C3 self-play baseline (Phase 7).
No policy trains here in the main experiment -- that is the whole point of the
distribution shift.

The observation each agent receives is produced by the same rival-agnostic
kinematics config used in solo training, so the rival enters the observation as
just another nearby vehicle. Nothing marks it as an agent (CLAUDE.md §3).

Win condition
-------------
Progress-based: first to ``laps_to_finish`` laps, else most distance travelled at
timeout. Agent-traffic collision is terminal for that agent only (it loses unless
the rival also fails). Agent-agent collision is terminal for both and the
*aggressor* is scored the loss -- this rule exists specifically to make ramming a
losing strategy, and Phase 5 requires reviewing videos to confirm it worked.

Aggressor attribution
---------------------
Closing speed ``dot(v_i - v_j, r_hat_ij)`` is symmetric -- it is a property of the
pair, not of either vehicle -- so it cannot by itself name a rammer. We instead
compare each agent's own velocity component *toward* the other,
``toward_i = dot(v_i, r_hat_ij)``. These sum to the closing speed, and the agent
contributing more of the approach is the aggressor: a rear-ender or side-impactor
is driving into a rival who is not driving into it. If the pair was not actually
closing, no aggressor is assigned and the collision is scored a draw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo import ParallelEnv

from envs.config import EnvConfig
from envs.progress import ProgressTracker, TrackGeometry
from envs.tracks import racetrack_class

# Results recorded per agent in the terminal info dict.
WIN, LOSS, DRAW = "win", "loss", "draw"


def attribute_aggressor(
    positions: list[np.ndarray], velocities: list[np.ndarray]
) -> int | None:
    """Return the index of the agent that initiated a collision, or ``None``.

    ``None`` means the vehicles were not closing on each other (a glancing or
    incidental contact), which is scored a draw rather than blamed on either side.
    """

    delta = np.asarray(positions[1]) - np.asarray(positions[0])
    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        return None
    unit = delta / norm

    toward_0 = float(np.dot(velocities[0], unit))
    toward_1 = float(np.dot(velocities[1], -unit))

    if toward_0 + toward_1 <= 0.0:  # not closing
        return None
    if np.isclose(toward_0, toward_1):  # symmetric head-on: nobody is more at fault
        return None
    return 0 if toward_0 > toward_1 else 1


class HeadToHeadRacetrackEnv(ParallelEnv):
    """Two policy-controlled vehicles racing amid identical IDM traffic."""

    metadata = {"name": "h2h_racetrack_v0", "is_parallel": True, "render_modes": ["rgb_array"]}

    def __init__(self, config: EnvConfig, render_mode: str | None = None) -> None:
        if config.n_agents != 2:
            raise ValueError("HeadToHeadRacetrackEnv requires n_agents=2")

        self.config = config
        self.render_mode = render_mode
        self.possible_agents = list(config.agent_ids)
        self.agents: list[str] = []

        env_cls = racetrack_class(config.track)
        hw_config = config.highway_config()
        if render_mode == "rgb_array":
            # Draw to an in-memory surface; never open a window. This is what makes
            # headless nodes render at all -- see arc/arc_env.sh ("Headless rendering").
            hw_config["offscreen_rendering"] = True
        self._env = env_cls(config=hw_config, render_mode=render_mode)

        # Homogeneous spaces; highway-env returns a tuple, one entry per agent.
        self._obs_space = self._env.observation_space[0]
        self._act_space = self._env.action_space[0]

        # agent_ids[i] drives controlled_vehicles[self._slots[i]]. Swapping this
        # permutation swaps starting grid slots without touching traffic seeding,
        # which is what makes paired episodes paired.
        self._slots: list[int] = [0, 1]
        self._trackers: list[ProgressTracker] = []
        self._steps = 0

    def vehicle_index(self, agent: str) -> int:
        """Which controlled vehicle ``agent`` is currently driving.

        Depends on the current ``swap_starts``, so scripted policies must call this
        after ``reset`` rather than assuming ``agent_i`` drives vehicle ``i``.
        """

        return self._slots[self.possible_agents.index(agent)]

    def observation_space(self, agent: str):  # noqa: D102 - PettingZoo API
        return self._obs_space

    def action_space(self, agent: str):  # noqa: D102 - PettingZoo API
        return self._act_space

    # -- PettingZoo API --------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the race.

        Options
        -------
        swap_starts : bool
            Exchange the two agents' starting grid slots. The traffic seed is
            unaffected, so a (seed, swap) pair defines one paired episode.
        """

        options = options or {}
        self._slots = [1, 0] if options.get("swap_starts", False) else [0, 1]

        seed = self.config.seed if seed is None else seed
        obs, _ = self._env.reset(seed=seed)

        geometry = TrackGeometry(self._env.unwrapped.road.network)
        self._trackers = []
        for vehicle in self._env.unwrapped.controlled_vehicles:
            tracker = ProgressTracker(geometry)
            tracker.reset(vehicle)
            self._trackers.append(tracker)

        self.agents = list(self.possible_agents)
        self._steps = 0

        observations = {a: obs[self._slots[i]] for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions: dict[str, Any]):
        """Apply simultaneous actions; return the parallel-API 5-tuple.

        Rewards are the zero-sum race outcome (+1 win / -1 loss / 0 draw) applied
        at termination. Head-to-head is evaluation-only, so no shaping is needed;
        the C3 self-play baseline consumes this same signal.
        """

        joint = tuple(
            np.asarray(actions[self.agents[self._slots.index(v)]], dtype=np.float32)
            for v in range(2)
        )
        obs, _, terminated, truncated, _ = self._env.step(joint)
        self._steps += 1

        vehicles = self._env.unwrapped.controlled_vehicles
        for i, vehicle in enumerate(vehicles):
            self._trackers[i].update(vehicle)

        outcome = self._resolve(vehicles)
        done = outcome is not None or bool(truncated) or self._steps >= self.config.max_steps
        if done and outcome is None:
            outcome = self._resolve_on_timeout(vehicles)

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for i, agent in enumerate(self.agents):
            v_idx = self._slots[i]
            vehicle = vehicles[v_idx]
            result = outcome[v_idx] if outcome else None

            observations[agent] = obs[v_idx]
            rewards[agent] = {WIN: 1.0, LOSS: -1.0, DRAW: 0.0}.get(result, 0.0)
            terminations[agent] = bool(done and not truncated)
            truncations[agent] = bool(truncated or self._steps >= self.config.max_steps)
            infos[agent] = {
                "result": result,
                "distance": self._trackers[v_idx].distance,
                "laps": self._trackers[v_idx].laps,
                "speed": float(vehicle.speed),
                "crashed": bool(vehicle.crashed),
                "off_road": not bool(vehicle.on_road),
                "position": np.array(vehicle.position, dtype=np.float64),
                "velocity": np.array(vehicle.velocity, dtype=np.float64),
                "steps": self._steps,
                "start_slot": v_idx,
            }

        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    # -- outcome resolution ----------------------------------------------------
    def _agent_agent_collision(self, vehicles) -> bool:
        """True if the two controlled vehicles crashed into *each other*.

        highway-env sets ``vehicle.crashed`` without recording the counterparty,
        so we confirm the rival is the crash partner by proximity: both crashed
        and they are within a vehicle length of one another.
        """

        if not (vehicles[0].crashed and vehicles[1].crashed):
            return False
        separation = float(np.linalg.norm(vehicles[0].position - vehicles[1].position))
        return separation <= vehicles[0].LENGTH + 1.0

    def _resolve(self, vehicles) -> list[str] | None:
        """Terminal outcome per *vehicle index*, or ``None`` if the race is live."""

        laps = [t.laps for t in self._trackers]
        finished = [lap >= self.config.laps_to_finish for lap in laps]
        if any(finished):
            if all(finished):  # same step: fall back to distance
                return self._by_distance()
            winner = 0 if finished[0] else 1
            return self._verdict(winner)

        if self._agent_agent_collision(vehicles):
            aggressor = attribute_aggressor(
                [v.position for v in vehicles], [v.velocity for v in vehicles]
            )
            if aggressor is None:
                return [DRAW, DRAW]
            return self._verdict(1 - aggressor)

        failed = [v.crashed or not v.on_road for v in vehicles]
        if all(failed):
            return [DRAW, DRAW]
        if any(failed):
            # Agent-traffic collision (or off-track): terminal for that agent only.
            # The survivor takes the win -- it is still racing and its rival is out.
            return self._verdict(0 if failed[1] else 1)
        return None

    def _resolve_on_timeout(self, vehicles) -> list[str]:
        return self._by_distance()

    def _by_distance(self) -> list[str]:
        d0, d1 = (t.distance for t in self._trackers)
        if np.isclose(d0, d1):
            return [DRAW, DRAW]
        return self._verdict(0 if d0 > d1 else 1)

    @staticmethod
    def _verdict(winner: int) -> list[str]:
        return [WIN if i == winner else LOSS for i in range(2)]

    @property
    def highway(self):
        """The underlying highway-env instance (see :meth:`SoloRacetrackEnv.highway`)."""

        return self._env.unwrapped

    @property
    def unwrapped(self):
        """PettingZoo envs are not Gym-wrapped; this makes `.unwrapped` uniform."""

        return self

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()
