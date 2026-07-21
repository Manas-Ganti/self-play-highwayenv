"""Shared machinery: nets, buffers, collectors, logging, config, utilities.

PPO and GRPO import everything from here, so a change to the rollout path affects
both identically and cannot bias the comparison.
"""

from algos.common.buffer import (
    Batch,
    GroupBuffer,
    RolloutBuffer,
    group_relative_advantages,
)
from algos.common.collectors import EpisodeStats, GroupCollector, VecCollector
from algos.common.config import SHARED_HPARAMS, GRPOConfig, PPOConfig, RunConfig
from algos.common.logger import Logger
from algos.common.nets import ActorCritic, Critic, GaussianActor
from algos.common.utils import (
    GradientVarianceTracker,
    explained_variance,
    global_grad_norm,
    resolve_device,
    set_seed,
)

__all__ = [
    "SHARED_HPARAMS",
    "ActorCritic",
    "Batch",
    "Critic",
    "EpisodeStats",
    "GRPOConfig",
    "GaussianActor",
    "GradientVarianceTracker",
    "GroupBuffer",
    "GroupCollector",
    "Logger",
    "PPOConfig",
    "RolloutBuffer",
    "RunConfig",
    "VecCollector",
    "explained_variance",
    "global_grad_norm",
    "group_relative_advantages",
    "resolve_device",
    "set_seed",
]
