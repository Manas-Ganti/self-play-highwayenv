"""Evaluation: solo suite, head-to-head harness, ELO, and the statistics rules."""

from eval.elo import fit_elo, games_from_matchups
from eval.h2h import EpisodeRecord, MatchupResult, run_matchup
from eval.policies import IDMPolicy, Policy, TorchPolicy
from eval.solo import SoloEvalResult, evaluate_solo
from eval.stats import (
    Interval,
    bootstrap_ci,
    degradation,
    iqm,
    iqm_ci,
    paired_permutation_test,
    win_rate_ci,
)

__all__ = [
    "EpisodeRecord",
    "IDMPolicy",
    "Interval",
    "MatchupResult",
    "Policy",
    "SoloEvalResult",
    "TorchPolicy",
    "bootstrap_ci",
    "degradation",
    "evaluate_solo",
    "fit_elo",
    "games_from_matchups",
    "iqm",
    "iqm_ci",
    "paired_permutation_test",
    "run_matchup",
    "win_rate_ci",
]
