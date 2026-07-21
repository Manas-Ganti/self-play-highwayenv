"""Reward function tests.

Covers, per CLAUDE.md, at least: crash case, win case, neutral step, and that
the reward is bounded. These run without highway-env / SB3.
"""

from __future__ import annotations

import pytest

from rewards.registry import REGISTRY, make_reward


def _obs(lap_progress=0.0, opponent_lap_progress=0.0, normalised_speed=0.0):
    return {
        "lap_progress": lap_progress,
        "opponent_lap_progress": opponent_lap_progress,
        "normalised_speed": normalised_speed,
    }


# -- sparse ------------------------------------------------------------------
def test_sparse_win():
    r = make_reward("sparse")
    assert r("agent_0", _obs(), 3, _obs(), {"won": True}) == 1.0


def test_sparse_crash():
    r = make_reward("sparse")
    assert r("agent_0", _obs(), 3, _obs(), {"crashed": True}) == -1.0


def test_sparse_crash_precedence_over_win():
    r = make_reward("sparse")
    assert r("agent_0", _obs(), 3, _obs(), {"won": True, "crashed": True}) == -1.0


def test_sparse_neutral():
    r = make_reward("sparse")
    assert r("agent_0", _obs(), 1, _obs(), {}) == 0.0


# -- dense -------------------------------------------------------------------
def test_dense_progress_term():
    r = make_reward("dense", lambda_speed=0.0, lambda_competitive=0.0, lambda_crash=0.0)
    reward = r("agent_0", _obs(lap_progress=0.1), 3, _obs(lap_progress=0.2), {})
    assert reward == pytest.approx(0.1)  # lambda_progress * 0.1


def test_dense_crash_penalty():
    r = make_reward("dense")
    reward = r("agent_0", _obs(), 4, _obs(), {"crashed": True})
    assert reward <= -5.0 + 1e-6


def test_dense_competitive_gap():
    r = make_reward("dense", lambda_progress=0.0, lambda_speed=0.0, lambda_crash=0.0)
    # Opponent ahead by 0.4 -> penalty of lambda_competitive * 0.4.
    reward = r(
        "agent_0", _obs(), 1,
        _obs(lap_progress=0.1, opponent_lap_progress=0.5), {},
    )
    assert reward == pytest.approx(-0.5 * 0.4)


def test_dense_bounded():
    r = make_reward("dense")
    # Extreme but in-range inputs stay finite and bounded.
    reward = r(
        "agent_0",
        _obs(lap_progress=0.0),
        3,
        _obs(lap_progress=1.0, opponent_lap_progress=1.0, normalised_speed=1.0),
        {},
    )
    assert -20.0 < reward < 20.0


# -- hybrid ------------------------------------------------------------------
def test_hybrid_adds_terminal_signal():
    dense = make_reward("dense")
    hybrid = make_reward("hybrid")
    obs, nxt = _obs(), _obs()
    info_win = {"won": True, "terminal": True}
    assert hybrid("agent_0", obs, 3, nxt, info_win) == pytest.approx(
        dense("agent_0", obs, 3, nxt, info_win) + 1.0
    )


def test_hybrid_no_terminal_equals_dense():
    dense = make_reward("dense")
    hybrid = make_reward("hybrid")
    obs, nxt = _obs(lap_progress=0.0), _obs(lap_progress=0.1)
    assert hybrid("agent_0", obs, 3, nxt, {}) == pytest.approx(
        dense("agent_0", obs, 3, nxt, {})
    )


# -- registry ----------------------------------------------------------------
def test_registry_keys():
    assert set(REGISTRY) == {"sparse", "dense", "hybrid"}


def test_make_reward_unknown():
    with pytest.raises(KeyError):
        make_reward("does_not_exist")
