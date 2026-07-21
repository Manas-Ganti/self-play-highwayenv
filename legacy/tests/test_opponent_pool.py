"""Opponent pool and ELO tests.

These run without highway-env / SB3 by using a trivial stand-in policy.
"""

from __future__ import annotations

from agents.opponent_pool import EloTracker, FrozenPolicy, OpponentPool


class _DummyPolicy:
    """Minimal policy with the SB3 predict signature."""

    def __init__(self, action: int = 0) -> None:
        self.action = action

    def predict(self, observation, deterministic: bool = True):
        return self.action, None


def test_pool_seeds_initial_policy():
    pool = OpponentPool(initial_policy=_DummyPolicy(), max_size=5)
    assert len(pool) == 1
    assert pool.mean_elo == 1000.0


def test_pool_empty_when_no_initial():
    pool = OpponentPool(initial_policy=None)
    assert len(pool) == 0


def test_pool_eviction_oldest_first():
    pool = OpponentPool(initial_policy=_DummyPolicy(0), max_size=2)
    pool.add_checkpoint(_DummyPolicy(1), step=10)
    pool.add_checkpoint(_DummyPolicy(2), step=20)  # evicts slot 0
    assert len(pool) == 2
    # The surviving oldest should be the step=10 checkpoint.
    steps = [p.checkpoint_step for p in pool._pool]
    assert steps == [10, 20]


def test_sample_opponent_returns_frozen():
    pool = OpponentPool(initial_policy=_DummyPolicy(), max_size=5, seed=1)
    pool.add_checkpoint(_DummyPolicy(), elo=1100.0, step=5)
    opp = pool.sample_opponent(learner_elo=1000.0)
    assert isinstance(opp, FrozenPolicy)


def test_sampling_is_reproducible():
    def draw():
        pool = OpponentPool(initial_policy=_DummyPolicy(), max_size=10, seed=42)
        for i in range(5):
            pool.add_checkpoint(_DummyPolicy(), elo=1000.0 + 50 * i, step=i)
        return [
            pool.sample_opponent(1000.0).checkpoint_step for _ in range(10)
        ]

    assert draw() == draw()


def test_elo_update_zero_sum():
    pool = OpponentPool(initial_policy=_DummyPolicy(), max_size=5)
    opp = pool.sample_opponent(1000.0)
    tracker = EloTracker(learner_elo=1000.0, k=32.0)
    before = tracker.learner_elo + opp.elo
    tracker.update(opp, learner_won=True)
    after = tracker.learner_elo + opp.elo
    assert abs(before - after) < 1e-9  # symmetric transfer conserves total
    assert tracker.learner_elo > 1000.0


def test_elo_loss_decreases_rating():
    pool = OpponentPool(initial_policy=_DummyPolicy(), max_size=5)
    opp = pool.sample_opponent(1000.0)
    tracker = EloTracker(learner_elo=1000.0)
    tracker.update(opp, learner_won=False)
    assert tracker.learner_elo < 1000.0


def test_expected_score_symmetry():
    assert OpponentPool.expected_score(1000, 1000) == 0.5
    assert OpponentPool.expected_score(1200, 1000) > 0.5
