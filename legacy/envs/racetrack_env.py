"""Competitive racetrack environment as a PettingZoo parallel environment.

Wraps highway-env's ``racetrack-v0`` with two (configurable) policy-controlled
vehicles and a randomised number of NPCs. Both agents observe the same physics
simulation and act simultaneously.

This module is deliberately *flexible*: every quantity that an experiment might
perturb — physics, geometry, agent count, action space, game type — is read
from a typed :class:`~envs.config.EnvConfig`, never hardcoded. It is still only
ever constructed via :func:`envs.env_factory.make_env`.

The observation handed to policies is a fixed, ordered, normalised feature
vector (see :data:`FEATURE_ORDER`). The same features are also exposed as a
dict to the reward function, which keeps reward code readable and the policy
input compact.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from pettingzoo import ParallelEnv
except ModuleNotFoundError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    ParallelEnv = object  # type: ignore[assignment,misc]

from envs.config import ActionType, EnvConfig, GameType
from envs.vehicle_models import patch_vehicle_physics
from rewards.base import BaseReward

logger = logging.getLogger("racetrack_rl")

# Ordered, normalised observation features. Position is normalised by track
# length and heading is encoded as (cos, sin) — never a raw angle, and never a
# raw timestep — so policies can transfer across track scales (see CLAUDE.md).
FEATURE_ORDER: tuple[str, ...] = (
    "x",
    "y",
    "vx",
    "vy",
    "cos_h",
    "sin_h",
    "opponent_x",
    "opponent_y",
    "opponent_vx",
    "opponent_vy",
    "lap_progress",
    "opponent_lap_progress",
)

# Discrete meta-action labels, indices matching highway-env's DiscreteMetaAction.
DISCRETE_ACTIONS: tuple[str, ...] = (
    "LANE_LEFT",
    "IDLE",
    "LANE_RIGHT",
    "FASTER",
    "SLOWER",
)


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Flatten a feature dict into the fixed-order policy observation vector."""

    return np.array([features[k] for k in FEATURE_ORDER], dtype=np.float32)


_HIGHWAY_ENV_CLASS = None


def _competitive_highway_env_class():
    """Lazily build and cache the RacetrackEnv subclass used under the hood.

    Defining the subclass lazily keeps this module importable without
    highway-env installed (the config and reward layers stay testable). The
    subclass exists solely to fix stock racetrack-v0's controlled-vehicle
    creation, which assumes a continuous action type whose ``vehicle_class`` is
    a plain class; under ``DiscreteMetaAction`` it is a ``functools.partial`` and
    ``make_on_lane`` is unreachable.
    """

    global _HIGHWAY_ENV_CLASS
    if _HIGHWAY_ENV_CLASS is not None:
        return _HIGHWAY_ENV_CLASS

    import functools

    from highway_env.envs.racetrack_env import RacetrackEnv
    from highway_env.vehicle.behavior import IDMVehicle

    class CompetitiveRacetrackHighwayEnv(RacetrackEnv):
        """RacetrackEnv whose controlled vehicles work for any action type."""

        @staticmethod
        def _resolve_vehicle(vehicle_class, road, lane_index, longitudinal, speed):
            """Construct a controlled vehicle, unwrapping a ``partial`` factory.

            ``DiscreteMetaAction.vehicle_class`` is ``partial(MDPVehicle,
            target_speeds=...)``; the underlying class carries ``make_on_lane``,
            and the partial's keywords (e.g. ``target_speeds``) are re-applied
            after construction so the action mapping stays consistent.
            """

            if isinstance(vehicle_class, functools.partial):
                base_cls = vehicle_class.func
                extra_kwargs = dict(vehicle_class.keywords)
            else:
                base_cls = vehicle_class
                extra_kwargs = {}

            vehicle = base_cls.make_on_lane(
                road, lane_index, longitudinal=longitudinal, speed=speed
            )
            for key, value in extra_kwargs.items():
                setattr(vehicle, key, value)
            return vehicle

        def _make_vehicles(self) -> None:
            rng = self.np_random
            vehicle_class = self.action_type.vehicle_class

            self.controlled_vehicles = []
            for i in range(self.config["controlled_vehicles"]):
                lane_index = (
                    ("a", "b", rng.integers(2))
                    if i == 0
                    else self.road.network.random_lane_index(rng)
                )
                vehicle = self._resolve_vehicle(
                    vehicle_class,
                    self.road,
                    lane_index,
                    longitudinal=rng.uniform(20, 50),
                    speed=None,
                )
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

            if self.config["other_vehicles"] > 0:
                front = IDMVehicle.make_on_lane(
                    self.road,
                    ("b", "c", lane_index[-1]),
                    longitudinal=rng.uniform(
                        low=0.0,
                        high=self.road.network.get_lane(("b", "c", 0)).length,
                    ),
                    speed=6.0 + rng.uniform(high=3.0),
                )
                self.road.vehicles.append(front)

                for _ in range(rng.integers(self.config["other_vehicles"])):
                    rand_lane_index = self.road.network.random_lane_index(rng)
                    npc = IDMVehicle.make_on_lane(
                        self.road,
                        rand_lane_index,
                        longitudinal=rng.uniform(
                            low=0.0,
                            high=self.road.network.get_lane(rand_lane_index).length,
                        ),
                        speed=6.0 + rng.uniform(high=3.0),
                    )
                    # Prevent early collisions with existing vehicles.
                    if all(
                        np.linalg.norm(npc.position - v.position) >= 20
                        for v in self.road.vehicles
                    ):
                        self.road.vehicles.append(npc)

    _HIGHWAY_ENV_CLASS = CompetitiveRacetrackHighwayEnv
    return _HIGHWAY_ENV_CLASS


class CompetitiveRacetrackEnv(ParallelEnv):  # type: ignore[misc]
    """Two-agent competitive racetrack, PettingZoo parallel API.

    Parameters
    ----------
    config : EnvConfig
        Validated environment configuration.
    reward_fn : BaseReward
        Reward callable injected by the factory; applied per agent each step.

    Notes
    -----
    Construct only through :func:`envs.env_factory.make_env`.
    """

    metadata = {"name": "competitive_racetrack_v0", "is_parallel": True}

    def __init__(self, config: EnvConfig, reward_fn: BaseReward) -> None:
        if gym is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "highway-env / gymnasium / pettingzoo are required to run the "
                "environment. Install requirements.txt."
            )
        self.config = config
        self.reward_fn = reward_fn

        self.possible_agents: list[str] = list(config.agent_ids)
        self.agents: list[str] = []

        self._np_random = np.random.default_rng(config.seed)
        self._highway = self._build_highway_env()

        # Per-agent observation/action spaces (homogeneous across agents).
        obs_dim = len(FEATURE_ORDER)
        self._observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        if config.action_type is ActionType.DISCRETE:
            self._action_space: Any = gym.spaces.Discrete(len(DISCRETE_ACTIONS))
        else:
            self._action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

        self._step_count = 0
        self._max_steps = config.duration * config.policy_frequency
        self._prev_progress: dict[str, float] = {}

    # -- PettingZoo required space accessors ------------------------------------
    def observation_space(self, agent: str):  # noqa: D102 - PettingZoo API
        return self._observation_space

    def action_space(self, agent: str):  # noqa: D102 - PettingZoo API
        return self._action_space

    # -- highway-env construction ----------------------------------------------
    def _build_highway_env(self, render_mode: str | None = None):
        """Create and configure the underlying highway-env racetrack instance.

        Uses :class:`CompetitiveRacetrackHighwayEnv` — a thin RacetrackEnv
        subclass that builds controlled vehicles in an action-type-agnostic way
        so the discrete meta-action + multi-agent combination this project
        requires works (stock racetrack-v0 only supports continuous control).

        All flexibility lives in the config dict below; physics is applied to
        the controlled vehicles after reset via :func:`patch_vehicle_physics`.
        """

        cfg = self.config
        n_npc = int(self._np_random.integers(cfg.npc_min, cfg.npc_max + 1))

        action_cfg = (
            {"type": "DiscreteMetaAction"}
            if cfg.action_type is ActionType.DISCRETE
            else {"type": "ContinuousAction"}
        )

        env_cls = _competitive_highway_env_class()
        env = env_cls(
            config={
                "controlled_vehicles": cfg.n_agents,
                "other_vehicles": n_npc,
                "duration": cfg.duration,
                "policy_frequency": cfg.policy_frequency,
                "lane_width": cfg.track_width,
                # Per-agent observation/action wrappers.
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                        "absolute": False,
                        "normalize": True,
                    },
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": action_cfg,
                },
            },
            render_mode=render_mode,
        )
        return env

    # -- helpers ----------------------------------------------------------------
    def _controlled_vehicles(self) -> list:
        return self._highway.unwrapped.controlled_vehicles

    def _apply_physics(self) -> None:
        """Patch drag/mass onto each controlled vehicle (supports asymmetry)."""

        for idx, vehicle in enumerate(self._controlled_vehicles()):
            vc = self.config.vehicle_for(idx)
            patch_vehicle_physics(vehicle, drag=vc.drag, mass_scale=vc.mass_scale)

    def _lap_progress(self, vehicle) -> float:
        """Fraction of the race completed, in [0, 1], independent of track scale.

        Uses cumulative longitudinal distance over the road network divided by
        the total race distance (lap length * lap_count). Falls back gracefully
        if highway-env does not expose the needed attributes.
        """

        try:
            lane = vehicle.lane
            long_pos, _ = lane.local_coordinates(vehicle.position)
            lap_len = self.config.road_length
            total = lap_len * self.config.lap_count
            travelled = getattr(vehicle, "_travelled", 0.0) + long_pos
            return float(np.clip(travelled / total, 0.0, 1.0))
        except Exception:  # pragma: no cover - defensive, geometry varies
            return 0.0

    def _build_features(self, idx: int) -> dict[str, float]:
        """Assemble the normalised feature dict for agent ``idx``."""

        vehicles = self._controlled_vehicles()
        me = vehicles[idx]
        opp = vehicles[1 - idx] if len(vehicles) > 1 else me

        scale = self.config.road_length
        norm_speed = float(np.clip(me.speed / max(me.MAX_SPEED, 1e-6), -1.0, 1.0)) \
            if hasattr(me, "MAX_SPEED") else float(np.tanh(me.speed / 30.0))

        feats = {
            "x": float(me.position[0] / scale),
            "y": float(me.position[1] / scale),
            "vx": float(np.tanh(me.velocity[0] / 30.0)),
            "vy": float(np.tanh(me.velocity[1] / 30.0)),
            "cos_h": float(np.cos(me.heading)),
            "sin_h": float(np.sin(me.heading)),
            "opponent_x": float((opp.position[0] - me.position[0]) / scale),
            "opponent_y": float((opp.position[1] - me.position[1]) / scale),
            "opponent_vx": float(np.tanh((opp.velocity[0] - me.velocity[0]) / 30.0)),
            "opponent_vy": float(np.tanh((opp.velocity[1] - me.velocity[1]) / 30.0)),
            "lap_progress": self._lap_progress(me),
            "opponent_lap_progress": self._lap_progress(opp),
        }
        feats["normalised_speed"] = norm_speed
        return feats

    def _winner(self) -> str | None:
        """Return the winning agent id, or ``None`` if the race is undecided.

        A crash by one agent hands the win to the other. Otherwise the agent
        with greater lap progress at truncation wins.
        """

        vehicles = self._controlled_vehicles()
        crashed = [getattr(v, "crashed", False) for v in vehicles]
        if any(crashed) and not all(crashed):
            survivor = crashed.index(False)
            return self.possible_agents[survivor]
        if self._step_count >= self._max_steps:
            progress = [self._lap_progress(v) for v in vehicles]
            return self.possible_agents[int(np.argmax(progress))]
        return None

    # -- PettingZoo API ---------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the race and return initial observations and infos."""

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._highway = self._build_highway_env()
        self._highway.reset(seed=seed if seed is not None else self.config.seed)
        self._apply_physics()

        self.agents = list(self.possible_agents)
        self._step_count = 0

        features = {a: self._build_features(i) for i, a in enumerate(self.agents)}
        self._prev_features = features
        observations = {a: features_to_vector(features[a]) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Apply simultaneous actions and return the parallel-API 5-tuple."""

        joint_action = tuple(actions[a] for a in self.agents)
        _, _, terminated, truncated, hw_info = self._highway.step(joint_action)
        self._step_count += 1

        next_features = {a: self._build_features(i) for i, a in enumerate(self.agents)}
        vehicles = self._controlled_vehicles()
        winner = self._winner()
        episode_over = bool(terminated or truncated) or winner is not None

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for i, a in enumerate(self.agents):
            crashed = bool(getattr(vehicles[i], "crashed", False))
            won = winner == a
            info = {
                "crashed": crashed,
                "won": won,
                "terminal": episode_over,
                "normalised_speed": next_features[a].get("normalised_speed", 0.0),
                "lap_progress": next_features[a]["lap_progress"],
            }
            reward = self.reward_fn(
                a, self._prev_features[a], actions[a], next_features[a], info
            )
            if self.config.game_type is GameType.COOPERATIVE:
                # Shared reward: averaged across agents (computed below).
                info["_raw_reward"] = reward

            observations[a] = features_to_vector(next_features[a])
            rewards[a] = float(reward)
            terminations[a] = episode_over
            truncations[a] = bool(truncated)
            infos[a] = info

        if self.config.game_type is GameType.COOPERATIVE:
            shared = float(np.mean(list(rewards.values())))
            rewards = {a: shared for a in self.agents}

        self._prev_features = next_features
        if episode_over:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):  # noqa: D102 - PettingZoo API
        return self._highway.render()

    def close(self) -> None:  # noqa: D102 - PettingZoo API
        self._highway.close()
