"""Reward registry — the single import point for reward construction.

Usage::

    from rewards.registry import make_reward
    reward_fn = make_reward("sparse")
    reward_fn = make_reward("dense", lambda_speed=0.5)

Register a new reward by adding it to ``REGISTRY``. Never construct reward
classes directly in training scripts.
"""

from __future__ import annotations

from collections.abc import Callable

from rewards.base import BaseReward
from rewards.dense import DenseReward
from rewards.hybrid import HybridReward
from rewards.sparse import SparseReward

REGISTRY: dict[str, Callable[..., BaseReward]] = {
    "sparse": SparseReward,
    "dense": DenseReward,
    "hybrid": HybridReward,
}


def make_reward(name: str, **params) -> BaseReward:
    """Instantiate the reward registered under ``name``.

    Parameters
    ----------
    name : str
        Registry key (``"sparse"``, ``"dense"``, ``"hybrid"``).
    **params
        Constructor keyword arguments (e.g. dense lambdas).

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """

    if name not in REGISTRY:
        raise KeyError(
            f"unknown reward '{name}'. Registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**params)
