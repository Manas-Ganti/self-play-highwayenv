"""Transfer heatmap: env parameter vs win-rate degradation.

    python analysis/plot_transfer.py --results results/physics_sweep/

Reads a sweep's ``results.csv`` (two env axes + win_rate) and renders a heatmap
of win-rate degradation relative to the training condition. Works for both the
physics sweep (drag x mass) and the geometry sweep (width x length).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("racetrack_rl")


def _latest_results_csv(results_dir: Path) -> Path:
    """Find the most recent ``results.csv`` under ``results_dir``."""

    candidates = sorted(results_dir.rglob("results.csv"))
    if not candidates:
        raise FileNotFoundError(f"no results.csv under {results_dir}")
    return candidates[-1]


def plot_transfer(
    results_dir: str | Path,
    opponent: str = "rule_based",
    out_path: str | Path | None = None,
) -> Path:
    """Render a win-rate-degradation heatmap for a two-axis sweep.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing (recursively) a sweep ``results.csv``.
    opponent : str
        Which opponent's win rate to plot.
    out_path : str or Path, optional
        Output figure path.

    Returns
    -------
    Path
        The saved figure path.
    """

    results_dir = Path(results_dir)
    df = pd.read_csv(_latest_results_csv(results_dir))
    df = df[df["opponent"] == opponent]

    # The two non-metric columns are the swept env axes.
    metric_cols = {
        "opponent", "episodes", "win_rate", "crash_rate",
        "mean_steps", "mean_lap_progress", "mean_overtakes",
    }
    axes = [c for c in df.columns if c not in metric_cols]
    if len(axes) != 2:
        raise ValueError(f"expected exactly 2 sweep axes, found {axes}")
    row_axis, col_axis = axes

    pivot = df.pivot_table(index=row_axis, columns=col_axis, values="win_rate")
    baseline = pivot.to_numpy().max()
    degradation = baseline - pivot

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(degradation.to_numpy(), cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
    ax.set_xlabel(col_axis)
    ax.set_ylabel(row_axis)
    ax.set_title(f"Win-rate degradation vs {opponent}\n(baseline best = {baseline:.2f})")
    for i in range(degradation.shape[0]):
        for j in range(degradation.shape[1]):
            ax.text(
                j, i, f"{degradation.to_numpy()[i, j]:.2f}",
                ha="center", va="center", color="w", fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="win-rate drop")

    out_path = Path(out_path) if out_path else results_dir / "transfer_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved transfer heatmap to %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Plot transfer heatmap.")
    p.add_argument("--results", required=True)
    p.add_argument("--opponent", default="rule_based")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    plot_transfer(args.results, args.opponent, args.out)


if __name__ == "__main__":
    main()
