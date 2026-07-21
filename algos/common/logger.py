"""Experiment logging: Weights & Biases, degrading gracefully to TensorBoard.

CLAUDE.md §2: log to W&B (project ``racing-grpo``); if ``WANDB_API_KEY`` is absent
or the import fails, fall back to TensorBoard rather than crashing a training run
that may already be hours in.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("racing-grpo")

WANDB_PROJECT = "racing-grpo"


class Logger:
    """Thin metric sink over W&B or TensorBoard.

    Parameters
    ----------
    run_name : str
        Unique name for this run (e.g. ``ppo_seed0``).
    config : dict
        Full run config, recorded for reproducibility.
    log_dir : Path
        Local directory for TensorBoard events and checkpoints.
    use_wandb : bool
        Request W&B. Ignored (with a warning) when no API key is present.
    """

    def __init__(
        self,
        run_name: str,
        config: dict[str, Any],
        log_dir: str | Path = "results",
        use_wandb: bool = True,
        group: str | None = None,
    ) -> None:
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        self._tb = None

        if use_wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=WANDB_PROJECT,
                    name=run_name,
                    group=group,
                    config=config,
                    dir=str(self.log_dir),
                    reinit=True,
                )
                logger.info("logging to W&B project '%s' as '%s'", WANDB_PROJECT, run_name)
            except Exception as exc:  # pragma: no cover - network / auth failures
                logger.warning("W&B unavailable (%s); falling back to TensorBoard", exc)
                self._wandb = None

        if self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.log_dir))
            reason = "no WANDB_API_KEY" if use_wandb else "W&B disabled"
            logger.info("logging to TensorBoard at %s (%s)", self.log_dir, reason)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Record a metric dict at ``step`` (env steps, not gradient updates)."""

        if not metrics:
            return
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        elif self._tb is not None:
            for key, value in metrics.items():
                self._tb.add_scalar(key, value, step)

    def log_video(self, path: str | Path, key: str = "video", step: int = 0) -> None:
        """Attach a rendered episode video, when the backend supports it."""

        if self._wandb is not None:
            self._wandb.log({key: self._wandb.Video(str(path))}, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.close()
