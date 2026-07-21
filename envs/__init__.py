"""Environment package -- the single entry point for env construction.

Nothing outside this package instantiates highway-env directly.

    from envs import make_solo_env, make_vec_env, make_h2h_env
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from envs.config import EnvConfig, ObsType, RewardConfig, TrackName
from envs.multi_agent import HeadToHeadRacetrackEnv
from envs.progress import ProgressTracker, TrackGeometry
from envs.reward import RewardTerms, compute_reward
from envs.solo import SoloRacetrackEnv

__all__ = [
    "EnvConfig",
    "HeadToHeadRacetrackEnv",
    "ObsType",
    "ProgressTracker",
    "RewardConfig",
    "RewardTerms",
    "SoloRacetrackEnv",
    "TrackGeometry",
    "TrackName",
    "compute_reward",
    "make_h2h_env",
    "make_solo_env",
    "make_vec_env",
    "solo_env_fn",
]


def _coerce(config: dict[str, Any] | EnvConfig | None) -> EnvConfig:
    if isinstance(config, EnvConfig):
        return config
    return EnvConfig.model_validate(config or {})


def make_solo_env(
    config: dict[str, Any] | EnvConfig | None = None,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Build the single-agent training environment, with a flat observation."""

    cfg = _coerce(config)
    env: gym.Env = SoloRacetrackEnv(cfg, render_mode=render_mode)
    return gym.wrappers.FlattenObservation(env)


def solo_env_fn(
    config: dict[str, Any] | EnvConfig | None = None,
) -> Callable[[], gym.Env]:
    """Return a picklable thunk for vector-env worker processes."""

    cfg = _coerce(config)

    def _thunk() -> gym.Env:
        return make_solo_env(cfg)

    return _thunk


def make_vec_env(
    config: dict[str, Any] | EnvConfig | None = None,
    *,
    n_envs: int = 16,
    asynchronous: bool = True,
) -> gym.vector.VectorEnv:
    """Build a vectorised stack of solo envs.

    highway-env stepping is pure-Python and CPU-bound, so throughput comes from
    process-level parallelism here, not from the GPU (CLAUDE.md §1).

    ``SAME_STEP`` autoreset (rather than Gymnasium 1.x's ``NEXT_STEP`` default)
    keeps every transition in the buffer a real one, so the rollout collectors
    never have to mask out placeholder steps.
    """

    cfg = _coerce(config)
    fns = [solo_env_fn(cfg) for _ in range(n_envs)]
    mode = gym.vector.AutoresetMode.SAME_STEP
    if asynchronous and n_envs > 1:
        return gym.vector.AsyncVectorEnv(fns, autoreset_mode=mode)
    return gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)


def make_h2h_env(
    config: dict[str, Any] | EnvConfig | None = None,
    *,
    render_mode: str | None = None,
) -> HeadToHeadRacetrackEnv:
    """Build the two-agent head-to-head evaluation environment."""

    cfg = _coerce(config)
    if cfg.n_agents != 2:
        cfg = cfg.model_copy(update={"n_agents": 2})
    return HeadToHeadRacetrackEnv(cfg, render_mode=render_mode)
