"""Baseline opponents and policies for evaluation.

Every evaluation must include these reference points so a win rate means
something. They share the SB3 ``predict(obs, deterministic=...) -> (action, state)``
interface, so they drop into the opponent slot or the learner slot unchanged.

``solo_trained`` and the ``self_play_*`` baselines are not defined here — they
are trained models loaded from disk in ``evaluate.py``. This module holds only
the non-learned references.
"""

from __future__ import annotations

import logging

import numpy as np

from envs.racetrack_env import DISCRETE_ACTIONS, FEATURE_ORDER

logger = logging.getLogger("racetrack_rl")

_IDX = {name: i for i, name in enumerate(FEATURE_ORDER)}
_FASTER = DISCRETE_ACTIONS.index("FASTER")
_LANE_LEFT = DISCRETE_ACTIONS.index("LANE_LEFT")


class RandomPolicy:
    """Uniform random discrete action each step.

    Parameters
    ----------
    n_actions : int
        Size of the discrete action space.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, n_actions: int = len(DISCRETE_ACTIONS), seed: int = 0) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic: bool = True):
        return int(self._rng.integers(self.n_actions)), None


class RuleBasedPolicy:
    """Always FASTER; LANE_LEFT if the opponent is ahead in the same lane.

    "Ahead in the same lane" is approximated from the relative-opponent features
    (``opponent_x`` positive and ``opponent_y`` near zero in track-normalised
    coordinates).
    """

    def __init__(self, lane_eps: float = 0.02) -> None:
        self.lane_eps = lane_eps

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float32)
        opp_ahead = obs[_IDX["opponent_x"]] > 0.0
        same_lane = abs(obs[_IDX["opponent_y"]]) < self.lane_eps
        if opp_ahead and same_lane:
            return _LANE_LEFT, None
        return _FASTER, None


REGISTRY = {
    "random": RandomPolicy,
    "rule_based": RuleBasedPolicy,
}
