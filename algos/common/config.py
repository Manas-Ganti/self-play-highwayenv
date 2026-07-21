"""Typed run configuration for both algorithms.

``SHARED_HPARAMS`` names the hyperparameters that mean the same thing in PPO and
GRPO. The Phase 3 hyperparameter search must cover these identically for both --
CLAUDE.md §1 calls matched search budgets "a validity requirement, not a
suggestion", and ``scripts/hp_search.py`` asserts it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from envs.config import EnvConfig, RewardConfig

# Hyperparameters shared by both algorithms; searched over the same grid.
SHARED_HPARAMS: tuple[str, ...] = (
    "learning_rate",
    "n_epochs",
    "batch_size",
    "clip_range",
    "entropy_coef",
    "max_grad_norm",
    "hidden_sizes",
    "log_std_init",
)


class PPOConfig(BaseModel):
    """PPO hyperparameters: shared block plus the critic/GAE machinery."""

    learning_rate: float = 3.0e-4
    n_epochs: int = 10
    batch_size: int = 256
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    log_std_init: float = -0.5

    # PPO-only: everything below exists because there is a critic.
    n_steps: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    normalize_advantage: bool = True


class GRPOConfig(BaseModel):
    """GRPO hyperparameters: the shared block plus group structure.

    Note what is *absent*: no ``gamma`` (advantages come from undiscounted
    episodic returns), no ``gae_lambda``, no ``vf_coef``. Adding a learned
    baseline here would destroy the research question (CLAUDE.md §8).
    """

    learning_rate: float = 3.0e-4
    n_epochs: int = 10
    batch_size: int = 256
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    log_std_init: float = -0.5

    # GRPO-only: group structure.
    group_size: int = 8  # G in CLAUDE.md §4.2
    n_groups: int = 4  # groups per update; n_envs = group_size * n_groups
    advantage_eps: float = 1e-8
    per_timestep_advantages: bool = False  # the GRPO-t diagnostic variant (§4)


class RunConfig(BaseModel):
    """A complete training run: environment, algorithm, budget, bookkeeping."""

    name: str = "ppo_seed0"
    algo: Literal["ppo", "grpo"] = "ppo"
    seed: int = 0
    total_steps: int = 2_000_000
    n_envs: int = 16
    device: str = "auto"

    eval_every: int = 100_000
    eval_episodes: int = 20
    checkpoint_every: int = 250_000

    use_wandb: bool = True
    results_dir: str = "results"
    group: str | None = None

    env: EnvConfig = Field(default_factory=EnvConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)

    @property
    def algo_config(self) -> PPOConfig | GRPOConfig:
        return self.ppo if self.algo == "ppo" else self.grpo

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Load a run config, resolving the frozen reward weights.

        The reward is deliberately *not* inlined per-run: both algorithms read
        ``configs/reward.yaml`` so a stray per-run override cannot make the two
        conditions incomparable.
        """

        with open(path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        env_data = data.setdefault("env", {})
        reward_path = data.pop("reward_config", None)
        env_data["reward"] = RewardConfig.load(reward_path).model_dump()

        cfg = cls.model_validate(data)
        if cfg.algo == "grpo":
            expected = cfg.grpo.group_size * cfg.grpo.n_groups
            if cfg.n_envs != expected:
                # n_envs is a *derived* quantity for GRPO: the group layout owns it.
                cfg = cfg.model_copy(update={"n_envs": expected})
        return cfg
