"""Typed schema and loader for experiment YAML configs.

An experiment config is the top-level object passed to ``run_experiment.py``. It
bundles the environment, reward, training, and evaluation settings plus an
optional ``sweep`` block. YAML files may inherit from another via ``base_config``;
this loader resolves that chain and validates the result into pydantic models so
the rest of the code never touches raw dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from envs.config import ActionType, EnvConfig, GameType, VehicleConfig

logger = logging.getLogger("racetrack_rl")

CONFIG_DIR = Path(__file__).parent / "configs"


class DenseParams(BaseModel):
    """Dense-reward lambda weights."""

    lambda_progress: float = 1.0
    lambda_speed: float = 0.3
    lambda_crash: float = 5.0
    lambda_competitive: float = 0.5


class RewardConfig(BaseModel):
    """Reward selection and its parameters."""

    type: str = "sparse"
    dense: DenseParams = Field(default_factory=DenseParams)


class EnvYaml(BaseModel):
    """The ``env`` block of an experiment config (flat convenience keys)."""

    track_width: float = 10.0
    road_length: float = 300.0
    vehicle_drag: float = 0.4
    vehicle_mass_scale: float = 1.0
    lap_count: int = 2
    game_type: GameType = GameType.COMPETITIVE
    action_type: ActionType = ActionType.DISCRETE
    seed: int = 0


class PPOParams(BaseModel):
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3.0e-4
    clip_range: float = 0.2


class SACParams(BaseModel):
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.005
    learning_rate: float = 3.0e-4


class TrainingConfig(BaseModel):
    """Training loop and algorithm hyperparameters."""

    algorithm: str = "sac"
    total_timesteps: int = 1_000_000
    checkpoint_interval: int = 50_000
    opponent_pool_max_size: int = 50
    pool_temperature: float = 200.0
    device: str = "mps"
    seed: int = 0
    ppo: PPOParams = Field(default_factory=PPOParams)
    sac: SACParams = Field(default_factory=SACParams)


class EvaluationConfig(BaseModel):
    """Evaluation settings."""

    episodes: int = 50
    record_trajectories: bool = True
    metrics: list[str] = Field(
        default_factory=lambda: [
            "win_rate",
            "elo",
            "mean_lap_time",
            "crash_rate",
            "overtake_count",
        ]
    )


class SweepAxis(BaseModel):
    param: str
    values: list[Any]


class SweepConfig(BaseModel):
    axes: list[SweepAxis]
    base_config: str | None = None
    train_on: Any = None
    eval_on: Any = "all_combinations"


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    experiment_name: str = "unnamed"
    env: EnvYaml = Field(default_factory=EnvYaml)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    sweep: SweepConfig | None = None

    def to_env_config(self, seed: int | None = None) -> EnvConfig:
        """Build a runtime :class:`EnvConfig` from this experiment config.

        Parameters
        ----------
        seed : int, optional
            Overrides the env seed (e.g. for per-episode randomisation).
        """

        reward_params: dict[str, Any] = {}
        if self.reward.type in ("dense", "hybrid"):
            reward_params = self.reward.dense.model_dump()

        return EnvConfig(
            track_width=self.env.track_width,
            road_length=self.env.road_length,
            lap_count=self.env.lap_count,
            vehicle=VehicleConfig(
                drag=self.env.vehicle_drag,
                mass_scale=self.env.vehicle_mass_scale,
            ),
            game_type=self.env.game_type,
            action_type=self.env.action_type,
            reward=self.reward.type,
            reward_params=reward_params,
            seed=seed if seed is not None else self.env.seed,
        )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""

    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML, resolving ``base_config`` inheritance.

    Parameters
    ----------
    path : str or Path
        Path to the experiment YAML (absolute, or relative to ``configs/``).

    Returns
    -------
    ExperimentConfig
        The fully-merged, validated experiment configuration.
    """

    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path

    raw = yaml.safe_load(path.read_text()) or {}

    base_name = raw.get("base_config")
    if base_name:
        base_raw = yaml.safe_load((CONFIG_DIR / base_name).read_text()) or {}
        # Don't let a child's base_config pointer propagate into the merged dict.
        raw = {k: v for k, v in raw.items() if k != "base_config"}
        raw = _deep_merge(base_raw, raw)

    config = ExperimentConfig.model_validate(raw)
    logger.info("Loaded experiment config '%s' from %s", config.experiment_name, path)
    return config
