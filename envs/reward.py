"""The dense solo reward -- identical for PPO and GRPO.

    r_t = lambda_progress * (delta_distance / (target_speed * dt))   # lap progress
        + lambda_speed    * clip(speed / target_speed, 0, 1)         # speed shaping
        - action_cost     * ||a||^2                                  # control effort
        - collision_penalty * 1[crashed]                             # terminal
        - offtrack_penalty  * 1[off_road]                            # terminal
        + lap_bonus       * 1[lap completed this step]

The progress term is normalised by the distance covered in one step at target
speed, so a lap-record pace earns ~1.0/step and the reward stays O(1) regardless
of track length or control frequency -- which matters because GRPO normalises
*episodic* returns within a group, and an unbounded reward scale would make the
group std the dominant source of variance.

Weights live in ``configs/reward.yaml`` and are frozen at the Phase 1 gate. Any
later change is a protocol amendment and must be logged in ``report/log.md``.
Both algorithms read the same file: an asymmetric reward would invalidate every
PPO-vs-GRPO comparison downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.config import RewardConfig


@dataclass
class RewardTerms:
    """Per-step breakdown, logged for diagnostics and reward-hacking checks."""

    progress: float = 0.0
    speed: float = 0.0
    action_cost: float = 0.0
    collision: float = 0.0
    offtrack: float = 0.0
    lap_bonus: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.progress
            + self.speed
            - self.action_cost
            - self.collision
            - self.offtrack
            + self.lap_bonus
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "reward/progress": self.progress,
            "reward/speed": self.speed,
            "reward/action_cost": self.action_cost,
            "reward/collision": self.collision,
            "reward/offtrack": self.offtrack,
            "reward/lap_bonus": self.lap_bonus,
        }


def compute_reward(
    cfg: RewardConfig,
    *,
    delta_distance: float,
    speed: float,
    action: np.ndarray | None,
    crashed: bool,
    off_road: bool,
    lap_completed: bool,
    dt: float,
) -> RewardTerms:
    """Evaluate the dense reward for one transition.

    Parameters
    ----------
    delta_distance : float
        Metres of lap progress since the previous step (may be negative).
    speed : float
        Current forward speed, m/s.
    action : np.ndarray or None
        The continuous action applied, for the control-effort cost.
    crashed, off_road, lap_completed : bool
        Terminal / event flags from the environment step.
    dt : float
        Seconds per policy step (``1 / policy_frequency``).
    """

    step_distance_at_target = max(cfg.target_speed * dt, 1e-8)
    terms = RewardTerms(
        progress=cfg.lambda_progress * (delta_distance / step_distance_at_target),
        speed=cfg.lambda_speed * float(np.clip(speed / max(cfg.target_speed, 1e-8), 0.0, 1.0)),
        collision=cfg.collision_penalty if crashed else 0.0,
        offtrack=cfg.offtrack_penalty if off_road else 0.0,
        lap_bonus=cfg.lap_bonus if lap_completed else 0.0,
    )
    if action is not None and cfg.action_cost:
        terms.action_cost = cfg.action_cost * float(np.sum(np.square(action)))
    return terms
