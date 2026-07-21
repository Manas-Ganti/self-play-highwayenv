"""Statistical machinery -- CLAUDE.md §7 is non-negotiable, so it is tested.

These functions decide what the project *claims*. A permutation test that
silently breaks the pairing, or an IQM that is really a mean, would not crash
anything -- it would just produce a confident wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.elo import expected_score, fit_elo
from eval.stats import (
    bootstrap_ci,
    degradation,
    iqm,
    paired_permutation_test,
    win_rate_ci,
)


class TestIQM:
    def test_iqm_of_hand_computed_sample(self):
        # Sorted: [1..8]. The middle 50% is [3, 4, 5, 6], whose mean is 4.5.
        assert iqm(np.array([5, 3, 8, 1, 6, 2, 7, 4])) == pytest.approx(4.5)

    def test_iqm_ignores_an_outlying_seed(self):
        """The reason the protocol mandates IQM over the mean for 5-seed claims."""

        clean = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        with_outlier = np.array([10.0, 11.0, 12.0, 13.0, 1000.0])

        assert iqm(clean) == pytest.approx(iqm(with_outlier))
        assert np.mean(clean) != pytest.approx(np.mean(with_outlier))


class TestBootstrap:
    def test_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=5.0, scale=1.0, size=20)
        ci = bootstrap_ci(x, statistic=np.mean, n_resamples=2000, rng=rng)

        assert ci.low < ci.point < ci.high
        assert ci.low < 5.0 < ci.high

    def test_wider_spread_gives_a_wider_interval(self):
        rng = np.random.default_rng(1)
        tight = bootstrap_ci(rng.normal(0, 0.1, 20), np.mean, 2000, rng=rng)
        loose = bootstrap_ci(rng.normal(0, 5.0, 20), np.mean, 2000, rng=rng)
        assert (loose.high - loose.low) > (tight.high - tight.low)


class TestPairedPermutation:
    def test_identical_inputs_give_p_of_one(self):
        x = np.array([1.0, 0.0, 1.0, 0.5])
        observed, p = paired_permutation_test(x, x.copy(), n_resamples=1000)
        assert observed == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_a_consistent_winner_is_significant(self):
        # A beats B on every one of 40 pairs; no sign-flip assignment can match it.
        a = np.ones(40)
        b = np.zeros(40)
        observed, p = paired_permutation_test(a, b, n_resamples=2000)
        assert observed == pytest.approx(1.0)
        assert p < 0.01

    def test_a_coin_flip_is_not_significant(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, size=60).astype(float)
        b = 1.0 - a
        _, p = paired_permutation_test(a, b, n_resamples=2000, rng=rng)
        assert p > 0.05

    def test_pairing_is_enforced_by_length(self):
        with pytest.raises(ValueError, match="equal lengths"):
            paired_permutation_test(np.ones(5), np.ones(4))


class TestWinRate:
    def test_draws_count_as_half(self):
        # Two wins, two draws, two losses -> a 50% win rate, not 33%.
        outcomes = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
        assert win_rate_ci(outcomes, n_resamples=500).point == pytest.approx(0.5)


class TestDegradation:
    def test_relative_change_against_the_agents_own_baseline(self):
        # Collision rate doubles from solo to competition: +100% degradation.
        assert degradation(solo=0.10, competitive=0.20) == pytest.approx(1.0)
        # Unchanged.
        assert degradation(solo=0.10, competitive=0.10) == pytest.approx(0.0)
        # Improved.
        assert degradation(solo=0.20, competitive=0.10) == pytest.approx(-0.5)

    def test_zero_baseline_is_nan_not_infinity(self):
        assert np.isnan(degradation(solo=0.0, competitive=0.1))


class TestElo:
    def test_expected_score_is_half_for_equal_ratings(self):
        assert expected_score(1000, 1000) == pytest.approx(0.5)

    def test_a_dominant_agent_outranks_the_others(self):
        games = [("a", "b", 1.0)] * 20 + [("a", "c", 1.0)] * 20 + [("b", "c", 0.5)] * 20
        ratings = fit_elo(games)
        assert ratings["a"] > ratings["b"]
        assert ratings["a"] > ratings["c"]
        assert ratings["b"] == pytest.approx(ratings["c"], abs=5.0)

    def test_fit_is_independent_of_game_order(self):
        """Sequential ELO depends on order; a round robin's rating must not."""

        rng = np.random.default_rng(0)
        games = [("a", "b", 1.0)] * 10 + [("b", "c", 1.0)] * 10 + [("a", "c", 0.5)] * 10

        first = fit_elo(games)
        shuffled = list(games)
        rng.shuffle(shuffled)
        second = fit_elo(shuffled)

        for name in first:
            assert first[name] == pytest.approx(second[name], abs=1e-6)
