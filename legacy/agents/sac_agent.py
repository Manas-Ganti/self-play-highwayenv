"""SB3 SAC wrapper for self-play training.

SAC is the default algorithm in ``base.yaml``. Note SAC expects a continuous
action space, so pair it with ``action_type: continuous`` in the env config;
for the discrete research default use :class:`~agents.ppo_agent.PPOAgent` (or
SB3's discrete-SAC variants). The wrapper mirrors :class:`PPOAgent` so the
training loop is algorithm-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.device import resolve_device

logger = logging.getLogger("racetrack_rl")


class SACAgent:
    """Construct and manage an SB3 SAC learner.

    Parameters
    ----------
    env : gymnasium.Env
        The (single-agent) training environment.
    hyperparams : dict
        SAC hyperparameters (``buffer_size``, ``learning_starts``,
        ``batch_size``, ``tau``, ``learning_rate``).
    device : str
        Requested torch device; resolved safely via :func:`resolve_device`.
    seed : int
        Seed forwarded to the SB3 model.
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
        from stable_baselines3 import SAC

        hp = hyperparams or {}
        self.device = resolve_device(device)
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            device=self.device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            buffer_size=hp.get("buffer_size", 200_000),
            learning_starts=hp.get("learning_starts", 5_000),
            batch_size=hp.get("batch_size", 256),
            tau=hp.get("tau", 0.005),
            learning_rate=hp.get("learning_rate", 3.0e-4),
            verbose=1,
        )
        logger.info("SACAgent initialised on device=%s", self.device)

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
        logger.info("SACAgent saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu"):
        from stable_baselines3 import SAC

        return SAC.load(str(path), device=resolve_device(device))
