"""Solo racetrack environment -- the Gymnasium env both PPO and GRPO train on.

One controlled vehicle, four IDM traffic vehicles, continuous control over
``[throttle, steering]``, and the dense lap-progress reward. This is the *only* environment used for training; the
rival agent is never present here (CLAUDE.md §0).

Reproducibility contract, relied upon by the GRPO group collector: ``reset(seed=s)``
is a pure function of ``s``. Two envs reset with the same seed produce byte-identical
initial observations, identical traffic placement, and identical agent pose. The
group collector's whole premise -- rollouts that differ *only* by policy
stochasticity -- rests on this, so it is asserted in
``tests/test_group_collector.py``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from envs.config import EnvConfig
from envs.progress import ProgressTracker, TrackGeometry
from envs.reward import compute_reward
from envs.tracks import racetrack_class


class SoloRacetrackEnv(gym.Env):
    """Single-agent racetrack with IDM traffic and the frozen dense reward."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 5}

    def __init__(self, config: EnvConfig, render_mode: str | None = None) -> None:
        if config.n_agents != 1:
            raise ValueError("SoloRacetrackEnv requires n_agents=1; use make_h2h_env")

        self.config = config
        self.render_mode = render_mode
        self._dt = 1.0 / config.policy_frequency

        env_cls = racetrack_class(config.track)
        self._env = env_cls(config=config.highway_config(), render_mode=render_mode)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self._geometry: TrackGeometry | None = None
        self._tracker: ProgressTracker | None = None
        self._prev_distance = 0.0
        self._prev_laps = 0
        self._steps = 0

    # -- gym API ---------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seed = self.config.seed if seed is None else seed
        obs, info = self._env.reset(seed=seed, options=options)

        self._geometry = TrackGeometry(self._env.unwrapped.road.network)
        self._tracker = ProgressTracker(self._geometry)
        self._tracker.reset(self._vehicle)
        self._prev_distance = 0.0
        self._prev_laps = 0
        self._steps = 0
        return obs, dict(info)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self._env.step(action)
        self._steps += 1

        assert self._tracker is not None, "step() before reset()"
        distance = self._tracker.update(self._vehicle)
        delta = distance - self._prev_distance
        laps = self._tracker.laps
        lap_completed = laps > self._prev_laps

        vehicle = self._vehicle
        crashed = bool(vehicle.crashed)
        off_road = not bool(vehicle.on_road)
        finished = laps >= self.config.laps_to_finish

        terms = compute_reward(
            self.config.reward,
            delta_distance=delta,
            speed=float(vehicle.speed),
            action=np.asarray(action, dtype=np.float64),
            crashed=crashed,
            off_road=off_road,
            lap_completed=lap_completed,
            dt=self._dt,
        )

        self._prev_distance = distance
        self._prev_laps = laps

        terminated = bool(terminated) or crashed or off_road or finished
        truncated = bool(truncated) or self._steps >= self.config.max_steps

        info = dict(info)
        info.update(terms.as_dict())
        info.update(
            {
                "distance": distance,
                "laps": laps,
                "speed": float(vehicle.speed),
                "crashed": crashed,
                "off_road": off_road,
                "finished": finished,
                "steps": self._steps,
            }
        )
        return obs, terms.total, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    # -- helpers ---------------------------------------------------------------
    @property
    def _vehicle(self):
        return self._env.unwrapped.controlled_vehicles[0]

    @property
    def highway(self):
        """The underlying highway-env instance.

        Rendering, video export, and scripted baselines read the simulator through
        this. Learned policies must never touch it -- they see only the observation.
        """

        return self._env.unwrapped
