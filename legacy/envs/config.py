"""Typed configuration objects for the competitive racetrack environment.

All environment construction flows through :class:`EnvConfig`. Experiment code
passes a plain ``dict`` to :func:`envs.env_factory.make_env`, which validates it
into one of these models. Beyond that boundary, only typed config objects are
passed between modules (see CLAUDE.md — "no raw dict passing between modules").
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GameType(StrEnum):
    """Whether agents are zero-sum competitors or share a cooperative reward."""

    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"


class ActionType(StrEnum):
    """Action parameterisation. Discrete is the research-phase default."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class VehicleConfig(BaseModel):
    """Per-vehicle physics parameters.

    Parameters
    ----------
    drag : float
        Aerodynamic drag coefficient. Higher values decelerate faster at speed.
    mass_scale : float
        Multiplier on base vehicle mass. Affects acceleration response.
    """

    drag: float = Field(default=0.4, ge=0.0)
    mass_scale: float = Field(default=1.0, gt=0.0)


class EnvConfig(BaseModel):
    """Single source of truth for one environment instance.

    Every parameter the research axes perturb lives here. The factory reads
    nothing else. Defaults match ``experiments/configs/base.yaml``.

    Parameters
    ----------
    track_width : float
        Lane width of the racetrack.
    road_length : float
        Nominal length of the circuit; observations normalise position by this
        so policies can transfer across track scales.
    lap_count : int
        Number of laps that constitute a race.
    n_agents : int
        Number of policy-controlled agents. The research uses two.
    npc_min, npc_max : int
        Inclusive bounds for the randomised number of uncontrolled NPC vehicles.
    vehicle : VehicleConfig
        Default physics applied to all controlled vehicles.
    agent_vehicles : list[VehicleConfig] | None
        Optional per-agent physics override for asymmetric-agent experiments.
        When set, its length must equal ``n_agents``.
    game_type : GameType
        Competitive (zero-sum) or cooperative (shared reward).
    action_type : ActionType
        Discrete meta-actions (research default) or continuous control.
    reward : str
        Key into ``rewards.registry.REGISTRY``.
    reward_params : dict
        Keyword arguments forwarded to the reward constructor (e.g. dense lambdas).
    duration : int
        Episode horizon in environment seconds.
    policy_frequency : int
        Decisions per simulated second.
    seed : int
        Master seed for the environment RNG.
    """

    track_width: float = Field(default=10.0, gt=0.0)
    road_length: float = Field(default=300.0, gt=0.0)
    lap_count: int = Field(default=2, ge=1)

    n_agents: int = Field(default=2, ge=1)
    npc_min: int = Field(default=0, ge=0)
    npc_max: int = Field(default=3, ge=0)

    vehicle: VehicleConfig = Field(default_factory=VehicleConfig)
    agent_vehicles: list[VehicleConfig] | None = None

    game_type: GameType = GameType.COMPETITIVE
    action_type: ActionType = ActionType.DISCRETE

    reward: str = "sparse"
    reward_params: dict = Field(default_factory=dict)

    duration: int = Field(default=60, ge=1)
    policy_frequency: int = Field(default=5, ge=1)

    seed: int = 0

    @model_validator(mode="after")
    def _check_consistency(self) -> EnvConfig:
        if self.npc_max < self.npc_min:
            raise ValueError("npc_max must be >= npc_min")
        if self.agent_vehicles is not None and len(self.agent_vehicles) != self.n_agents:
            raise ValueError(
                f"agent_vehicles has {len(self.agent_vehicles)} entries "
                f"but n_agents is {self.n_agents}"
            )
        return self

    def vehicle_for(self, index: int) -> VehicleConfig:
        """Return the physics config for the agent at ``index``.

        Falls back to the shared ``vehicle`` config when no asymmetric override
        is provided.
        """

        if self.agent_vehicles is not None:
            return self.agent_vehicles[index]
        return self.vehicle

    @property
    def agent_ids(self) -> list[str]:
        """Stable PettingZoo agent identifiers, e.g. ``["agent_0", "agent_1"]``."""

        return [f"agent_{i}" for i in range(self.n_agents)]


# Reward type keys recognised by the registry. Kept here so configs and the
# registry validate against a single list.
RewardKey = Literal["sparse", "dense", "hybrid"]
