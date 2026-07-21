"""Frozen-checkpoint opponent pool with ELO tracking — the self-play core.

Rules (from CLAUDE.md):

* both agents start from the same initial policy;
* every ``checkpoint_interval`` steps the current policy is snapshotted into the
  pool with ELO 1000;
* each episode samples an opponent by skill proximity to the learner;
* the opponent is always frozen — its weights never change mid-episode;
* ELO updates happen at *episode end*, never per step.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("racetrack_rl")


@dataclass
class FrozenPolicy:
    """An immutable snapshot of a policy plus its current ELO.

    Parameters
    ----------
    policy : Any
        A frozen, inference-only policy object (e.g. an SB3 ``policy`` deep-copy
        or a loaded model). Its weights are never mutated by the pool.
    elo : float
        Skill estimate, updated only between episodes.
    checkpoint_step : int
        Training step at which the snapshot was taken (for logging/eviction).
    """

    policy: object
    elo: float = 1000.0
    checkpoint_step: int = 0

    def predict(self, observation, deterministic: bool = True):
        """Delegate action selection to the wrapped policy (inference only)."""

        return self.policy.predict(observation, deterministic=deterministic)


class OpponentPool:
    """Maintains a population of frozen past policies for self-play training.

    Parameters
    ----------
    initial_policy : Any, optional
        The policy used to initialise pool slot 0. May be ``None`` to create an
        empty pool that is seeded later with :meth:`add_checkpoint` — useful when
        the learner's policy object only exists after the SB3 model is built.
    max_size : int
        Maximum number of checkpoints to retain. Oldest are evicted first.
    temperature : float
        Controls spread of the skill-proximity sampling distribution. Larger
        temperature flattens the distribution toward uniform.
    seed : int
        Seed for the opponent sampler RNG (kept reproducible per CLAUDE.md).
    """

    def __init__(
        self,
        initial_policy: object | None = None,
        max_size: int = 50,
        temperature: float = 200.0,
        seed: int = 0,
    ) -> None:
        self.max_size = max_size
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)
        self._pool: list[FrozenPolicy] = []
        if initial_policy is not None:
            self.add_checkpoint(initial_policy, elo=1000.0, step=0)

    def __len__(self) -> int:
        return len(self._pool)

    def add_checkpoint(
        self, policy: object, elo: float = 1000.0, step: int = 0
    ) -> None:
        """Deep-copy ``policy`` into the pool, evicting the oldest if full."""

        frozen = FrozenPolicy(policy=copy.deepcopy(policy), elo=elo, checkpoint_step=step)
        self._pool.append(frozen)
        if len(self._pool) > self.max_size:
            evicted = self._pool.pop(0)
            logger.debug("OpponentPool evicted checkpoint @%d", evicted.checkpoint_step)
        logger.info(
            "OpponentPool: added checkpoint @%d (size=%d)", step, len(self._pool)
        )

    def sample_opponent(self, learner_elo: float) -> FrozenPolicy:
        """Sample a frozen opponent weighted by skill proximity to the learner.

        ``weight = 1 / (1 + exp(|learner_elo - opp_elo| / temperature))``.
        """

        weights = np.array(
            [
                1.0 / (1.0 + math.exp(abs(learner_elo - p.elo) / self.temperature))
                for p in self._pool
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        idx = int(self._rng.choice(len(self._pool), p=weights))
        return self._pool[idx]

    @property
    def mean_elo(self) -> float:
        """Mean ELO across the pool (logged as ``train/pool_mean_elo``)."""

        return float(np.mean([p.elo for p in self._pool])) if self._pool else 0.0

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Standard ELO expected score for A against B."""

        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


@dataclass
class EloTracker:
    """Tracks the learner's ELO against frozen opponents.

    Separated from the pool so the learner's rating is a single authoritative
    value updated once per episode at episode end.

    Parameters
    ----------
    learner_elo : float
        The learner's current rating.
    k : float
        ELO K-factor (update step size).
    """

    learner_elo: float = 1000.0
    k: float = 32.0
    history: list[float] = field(default_factory=list)

    def update(self, opponent: FrozenPolicy, learner_won: bool) -> None:
        """Update the learner's ELO from one finished episode's result.

        Parameters
        ----------
        opponent : FrozenPolicy
            The frozen opponent that was played; its ELO is updated symmetrically.
        learner_won : bool
            Whether the learner won the episode.
        """

        expected = OpponentPool.expected_score(self.learner_elo, opponent.elo)
        score = 1.0 if learner_won else 0.0
        delta = self.k * (score - expected)
        self.learner_elo += delta
        opponent.elo -= delta
        self.history.append(self.learner_elo)
        logger.debug(
            "ELO update: learner=%.1f opp=%.1f (won=%s)",
            self.learner_elo,
            opponent.elo,
            learner_won,
        )
