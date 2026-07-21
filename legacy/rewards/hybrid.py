"""Hybrid reward: dense shaping during the episode + sparse terminal signal.

Tests whether potential-style shaping helps or hurts generalisation relative to
pure sparse. On terminal steps the sparse win/crash signal is added on top of
(or, optionally, replaces) the dense shaping.
"""

from __future__ import annotations

from typing import Any

from rewards.base import BaseReward
from rewards.dense import DenseReward
from rewards.sparse import SparseReward


class HybridReward(BaseReward):
    """Sum of dense shaping and the sparse terminal signal.

    Parameters
    ----------
    lambda_progress, lambda_speed, lambda_crash, lambda_competitive : float
        Forwarded to the underlying :class:`~rewards.dense.DenseReward`.
    terminal_weight : float
        Multiplier on the sparse terminal signal, letting the experiment trade
        off shaping against the true objective.
    """

    def __init__(
        self,
        lambda_progress: float = 1.0,
        lambda_speed: float = 0.3,
        lambda_crash: float = 5.0,
        lambda_competitive: float = 0.5,
        terminal_weight: float = 1.0,
    ) -> None:
        self._dense = DenseReward(
            lambda_progress=lambda_progress,
            lambda_speed=lambda_speed,
            lambda_crash=lambda_crash,
            lambda_competitive=lambda_competitive,
        )
        self._sparse = SparseReward()
        self.terminal_weight = terminal_weight

    def __call__(
        self,
        agent_id: str,
        obs: dict[str, Any],
        action: int,
        next_obs: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        shaped = self._dense(agent_id, obs, action, next_obs, info)
        if info.get("terminal", False) or info.get("crashed", False) or info.get("won", False):
            shaped += self.terminal_weight * self._sparse(
                agent_id, obs, action, next_obs, info
            )
        return shaped
