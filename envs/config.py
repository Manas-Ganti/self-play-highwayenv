"""Typed environment configuration.

One config object drives every environment in the project: solo training, solo
evaluation, and the two-agent head-to-head harness. Nothing else constructs a
highway-env instance -- see :func:`envs.make_solo_env` / :func:`envs.make_h2h_env`.

The load-bearing invariant (CLAUDE.md §3): the observation carries **no feature
distinguishing IDM traffic from a rival agent**. At evaluation the rival must
look like an unusually capable traffic vehicle, so the distribution shift is
behavioural rather than structural. There is deliberately no knob here that
could turn that off.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent


class TrackName(StrEnum):
    """Track geometry. A is train + primary eval; C is held-out transfer only."""

    A = "a"
    C = "c"


class ObsType(StrEnum):
    """Observation parameterisation. Both are rival-agnostic by construction."""

    KINEMATICS = "kinematics"
    OCCUPANCY = "occupancy"


class RewardConfig(BaseModel):
    """Weights for the dense solo reward. Frozen after the Phase 1 gate.

    Loaded from ``configs/reward.yaml``; changing it later is a protocol
    amendment and must be recorded in ``report/log.md`` (CLAUDE.md §8).
    """

    lambda_progress: float = 1.0
    lambda_speed: float = 0.1
    target_speed: float = 10.0
    collision_penalty: float = 5.0
    offtrack_penalty: float = 0.5
    action_cost: float = 0.0
    lap_bonus: float = 0.0

    @classmethod
    def load(cls, path: str | Path | None = None) -> RewardConfig:
        """Load the frozen reward weights from ``configs/reward.yaml``."""

        path = Path(path) if path else REPO_ROOT / "configs" / "reward.yaml"
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)


class EnvConfig(BaseModel):
    """Single source of truth for one environment instance.

    Parameters
    ----------
    track : TrackName
        Geometry. Track A for all training and primary evaluation.
    n_traffic : int
        IDM traffic vehicles. Fixed at 4 across every training and evaluation
        condition (CLAUDE.md §3) -- traffic density is never an experimental axis.
    n_agents : int
        Policy-controlled vehicles. 1 for solo, 2 for head-to-head.
    obs_type : ObsType
        Observation family. Neither variant exposes rival identity.
    vehicles_count : int
        Nearby vehicles included in a kinematics observation (excluding ego).
    grid_x, grid_y, grid_step : occupancy-grid extent, metres, in the ego frame
        (x = ahead, y = lateral). The defaults are the original symmetric ±18 m
        grid, so runs saved before these fields existed replay unchanged. The
        forward extent is the policy's look-ahead: at speed v it must cover the
        stopping distance to slower traffic (pilot #3, report/log.md).
    speed_range : (min, max) m/s for the *controlled* car(s), or None for
        highway-env's default of (-40, 40), which also allows reversing. Traffic
        is unaffected. Both h2h agents share the same cap.
    laps_to_finish : int
        Laps that constitute a finished race.
    duration : int
        Episode horizon in simulated seconds.
    seed : int
        Episode seed; traffic init is a deterministic function of it.
    """

    track: TrackName = TrackName.A
    n_traffic: int = Field(default=4, ge=0)
    n_agents: int = Field(default=1, ge=1, le=2)

    obs_type: ObsType = ObsType.KINEMATICS
    vehicles_count: int = Field(default=5, ge=1)
    grid_x: tuple[float, float] = (-18.0, 18.0)
    grid_y: tuple[float, float] = (-18.0, 18.0)
    grid_step: float = Field(default=3.0, gt=0)
    speed_range: tuple[float, float] | None = None

    laps_to_finish: int = Field(default=1, ge=1)
    duration: int = Field(default=60, ge=1)
    policy_frequency: int = Field(default=5, ge=1)
    simulation_frequency: int = Field(default=15, ge=1)

    reward: RewardConfig = Field(default_factory=RewardConfig)
    seed: int = 0

    @property
    def max_steps(self) -> int:
        """Episode horizon in policy steps."""

        return self.duration * self.policy_frequency

    @property
    def agent_ids(self) -> list[str]:
        """Stable PettingZoo agent identifiers."""

        return [f"agent_{i}" for i in range(self.n_agents)]

    def highway_config(self) -> dict[str, Any]:
        """Render the highway-env config dict for this environment.

        Continuous control with **both** steering and throttle: stock
        racetrack-v0 ships ``longitudinal: False`` (steering only), which would
        make lap-time a non-decision and gut the racing problem.
        """

        from envs.tracks import track_spec

        spec = track_spec(self.track)
        action = {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "target_speeds": [0, 5, 10],
        }
        if self.speed_range is not None:
            action["speed_range"] = list(self.speed_range)
        if self.obs_type is ObsType.KINEMATICS:
            observation: dict[str, Any] = {
                "type": "Kinematics",
                # No rival-identity feature. `presence` marks real-vs-padded
                # rows, not traffic-vs-agent, and is required for a fixed-width
                # observation.
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "vehicles_count": self.vehicles_count + 1,
                "absolute": False,
                "normalize": True,
                "see_behind": True,
                "order": "sorted",
            }
        else:
            observation = {
                "type": "OccupancyGrid",
                "features": ["presence", "vx", "vy", "on_road"],
                "grid_size": [list(self.grid_x), list(self.grid_y)],
                "grid_step": [self.grid_step, self.grid_step],
                "as_image": False,
                "align_to_vehicle_axes": True,
            }

        cfg: dict[str, Any] = {
            "observation": observation,
            "action": action,
            "controlled_vehicles": self.n_agents,
            "other_vehicles": self.n_traffic,
            "duration": self.duration,
            "policy_frequency": self.policy_frequency,
            "simulation_frequency": self.simulation_frequency,
            "screen_width": 800,
            "screen_height": 800,
            "centering_position": [0.5, 0.5],
        }
        cfg.update(spec.highway_overrides)

        if self.n_agents > 1:
            cfg["observation"] = {
                "type": "MultiAgentObservation",
                "observation_config": observation,
            }
            cfg["action"] = {"type": "MultiAgentAction", "action_config": action}
        return cfg
