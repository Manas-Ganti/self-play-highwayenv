"""Dense reward: per-step shaping.

    r = lambda_progress * delta_lap_progress
      + lambda_speed    * normalised_speed
      - lambda_crash    * crash_indicator
      - lambda_competitive * (opponent_progress - agent_progress)

Each lambda is swept independently in ``reward_ablation.yaml`` to isolate its
contribution. The reward is bounded provided the environment supplies
normalised speed in [0, 1] and lap-progress fractions in [0, 1].
"""

from __future__ import annotations

from typing import Any

from rewards.base import BaseReward


class DenseReward(BaseReward):
    """Weighted sum of progress, speed, crash, and competitive-gap terms.

    Parameters
    ----------
    lambda_progress : float
        Weight on per-step lap-progress increase.
    lambda_speed : float
        Weight on normalised forward speed.
    lambda_crash : float
        Penalty weight applied on a crash.
    lambda_competitive : float
        Penalty weight on the opponent's progress lead over this agent.
    """

    def __init__(
        self,
        lambda_progress: float = 1.0,
        lambda_speed: float = 0.3,
        lambda_crash: float = 5.0,
        lambda_competitive: float = 0.5,
    ) -> None:
        self.lambda_progress = lambda_progress
        self.lambda_speed = lambda_speed
        self.lambda_crash = lambda_crash
        self.lambda_competitive = lambda_competitive

    def __call__(
        self,
        agent_id: str,
        obs: dict[str, Any],
        action: int,
        next_obs: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        delta_progress = float(
            next_obs.get("lap_progress", 0.0) - obs.get("lap_progress", 0.0)
        )
        speed = float(next_obs.get("normalised_speed", info.get("normalised_speed", 0.0)))
        crash = 1.0 if info.get("crashed", False) else 0.0
        gap = float(
            next_obs.get("opponent_lap_progress", 0.0)
            - next_obs.get("lap_progress", 0.0)
        )

        return (
            self.lambda_progress * delta_progress
            + self.lambda_speed * speed
            - self.lambda_crash * crash
            - self.lambda_competitive * gap
        )
