"""Sparse reward: terminal signal only.

This is the primary research condition. With no shaping, any blocking or
overtaking the agent learns is emergent from the win/lose/crash signal rather
than prescribed by the reward.
"""

from __future__ import annotations

from typing import Any

from rewards.base import BaseReward


class SparseReward(BaseReward):
    """+1 win, -1 crash, 0 otherwise.

    The environment is expected to set ``info["won"]`` and ``info["crashed"]``
    on the terminal step. Crashing takes precedence over a simultaneous win.
    """

    def __call__(
        self,
        agent_id: str,
        obs: dict[str, Any],
        action: int,
        next_obs: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        if info.get("crashed", False):
            return -1.0
        if info.get("won", False):
            return 1.0
        return 0.0
