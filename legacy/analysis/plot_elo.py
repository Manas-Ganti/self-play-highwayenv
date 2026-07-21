"""Plot ELO curves over training from TensorBoard logs.

    python analysis/plot_elo.py --logdir results/

Reads the ``train/learner_elo`` and ``train/pool_mean_elo`` scalars logged by
the self-play callback and overlays them, one figure per run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger("racetrack_rl")


def _read_scalar(event_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    """Read a scalar series from a TensorBoard event directory."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(event_dir))
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return [], []
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def plot_elo(logdir: str | Path, out_path: str | Path | None = None) -> Path:
    """Plot learner vs pool-mean ELO for every run under ``logdir``.

    Parameters
    ----------
    logdir : str or Path
        Directory containing TensorBoard event files (searched recursively).
    out_path : str or Path, optional
        Where to save the figure. Defaults to ``<logdir>/elo_curve.png``.

    Returns
    -------
    Path
        The saved figure path.
    """

    logdir = Path(logdir)
    event_dirs = {p.parent for p in logdir.rglob("events.out.tfevents.*")}

    fig, ax = plt.subplots(figsize=(8, 5))
    for event_dir in sorted(event_dirs):
        steps, learner = _read_scalar(event_dir, "train/learner_elo")
        _, pool = _read_scalar(event_dir, "train/pool_mean_elo")
        label = event_dir.relative_to(logdir)
        if steps:
            ax.plot(steps, learner, label=f"{label} learner")
        if pool:
            ax.plot(steps, pool, "--", label=f"{label} pool mean", alpha=0.6)

    ax.set_xlabel("training step")
    ax.set_ylabel("ELO")
    ax.set_title("Self-play ELO progression")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path) if out_path else logdir / "elo_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved ELO curve to %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Plot ELO curves over training.")
    p.add_argument("--logdir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    plot_elo(args.logdir, args.out)


if __name__ == "__main__":
    main()
