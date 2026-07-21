"""SB3 PPO wrapper for self-play training.

Thin, opinionated factory around ``stable_baselines3.PPO`` that wires in the
device-safe selection (MPS with CPU fallback) and the hyperparameters from the
training config. The self-play loop itself lives in ``experiments/train.py``;
this module only constructs and persists the model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.device import resolve_device

logger = logging.getLogger("racetrack_rl")


class PPOAgent:
    """Construct and manage an SB3 PPO learner.

    Parameters
    ----------
    env : gymnasium.Env
        The (single-agent) training environment, e.g. a
        :class:`~agents.self_play.SelfPlaySingleAgentEnv`.
    hyperparams : dict
        PPO hyperparameters (``n_steps``, ``batch_size``, ``n_epochs``,
        ``learning_rate``, ``clip_range``).
    device : str
        Requested torch device; resolved safely via :func:`resolve_device`.
    seed : int
        Seed forwarded to the SB3 model for reproducibility.
    tensorboard_log : str or Path or None
        Directory for TensorBoard logs.
    """

    def __init__(
        self,
        env: Any,
        hyperparams: dict[str, Any] | None = None,
        device: str = "mps",
        seed: int = 0,
        tensorboard_log: str | Path | None = None,
    ) -> None:
        from stable_baselines3 import PPO

        hp = hyperparams or {}
        self.device = resolve_device(device)
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            device=self.device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            n_steps=hp.get("n_steps", 2048),
            batch_size=hp.get("batch_size", 64),
            n_epochs=hp.get("n_epochs", 10),
            learning_rate=hp.get("learning_rate", 3.0e-4),
            clip_range=hp.get("clip_range", 0.2),
            verbose=1,
        )
        logger.info("PPOAgent initialised on device=%s", self.device)

    @property
    def policy(self):
        """The underlying SB3 policy (deep-copied into the opponent pool)."""

        return self.model.policy

    def predict(self, observation, deterministic: bool = True):
        return self.model.predict(observation, deterministic=deterministic)

    def learn(self, total_timesteps: int, **kwargs) -> None:
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))
        logger.info("PPOAgent saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu"):
        from stable_baselines3 import PPO

        return PPO.load(str(path), device=resolve_device(device))
