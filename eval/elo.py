"""ELO over the head-to-head round robin.

ELO is a *secondary* metric (win rate with a bootstrap CI is primary, CLAUDE.md
§5): it compresses the whole round robin into one number per agent, which is
useful for ranking but hides the pairwise structure and has no honest CI.

Because the round robin is a fixed set of games rather than a stream, ratings are
fitted by iterating over the game list to convergence, and the result is
order-independent -- unlike sequential ELO, where whoever plays last moves most.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

INITIAL_RATING = 1000.0
K_FACTOR = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Logistic expectation that A beats B."""

    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def fit_elo(
    games: list[tuple[str, str, float]],
    n_iterations: int = 200,
    k: float = K_FACTOR,
    initial: float = INITIAL_RATING,
) -> dict[str, float]:
    """Fit ratings to a round robin.

    Parameters
    ----------
    games : list of (name_a, name_b, score_a)
        ``score_a`` is 1.0 win / 0.5 draw / 0.0 loss for ``name_a``.
    n_iterations : int
        Sweeps over the game list. Ratings are re-fit from the same games each
        sweep with a decaying step, which converges to an order-independent
        maximum-likelihood-like fit.
    """

    names = sorted({n for game in games for n in game[:2]})
    ratings = dict.fromkeys(names, initial)
    if not games:
        return ratings

    for iteration in range(n_iterations):
        step = k / (1.0 + iteration)
        deltas: dict[str, float] = defaultdict(float)
        for name_a, name_b, score_a in games:
            exp_a = expected_score(ratings[name_a], ratings[name_b])
            residual = score_a - exp_a
            deltas[name_a] += step * residual
            deltas[name_b] -= step * residual
        for name in names:
            ratings[name] += deltas[name] / max(len(games) / len(names), 1.0)

    # ELO is only defined up to an additive constant; centre it on the initial
    # rating so numbers are comparable across round robins.
    mean = float(np.mean(list(ratings.values())))
    return {name: rating - mean + initial for name, rating in ratings.items()}


def games_from_matchups(matchups) -> list[tuple[str, str, float]]:
    """Flatten :class:`eval.h2h.MatchupResult` objects into an ELO game list."""

    games: list[tuple[str, str, float]] = []
    for matchup in matchups:
        for episode in matchup.episodes:
            games.append((matchup.name_a, matchup.name_b, episode.score))
    return games
