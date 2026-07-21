"""Generate a LaTeX/markdown table from a reward-ablation sweep.

    python analysis/ablation_table.py --results results/reward_ablation/ --format markdown

Aggregates win rate (and other metrics) per reward variant into a table suitable
for the writeup.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("racetrack_rl")

_DISPLAY_COLS = ["opponent", "win_rate", "crash_rate", "mean_overtakes"]


def _latest_results_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.rglob("results.csv"))
    if not candidates:
        raise FileNotFoundError(f"no results.csv under {results_dir}")
    return candidates[-1]


def build_table(results_dir: str | Path, fmt: str = "markdown") -> str:
    """Build an ablation table string in the requested format.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing a sweep ``results.csv``.
    fmt : str
        ``"markdown"`` or ``"latex"``.

    Returns
    -------
    str
        The rendered table.
    """

    df = pd.read_csv(_latest_results_csv(Path(results_dir)))
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    table = df[cols].round(3)

    if fmt == "latex":
        return table.to_latex(index=False)
    return table.to_markdown(index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Generate ablation table.")
    p.add_argument("--results", required=True)
    p.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rendered = build_table(args.results, args.format)
    if args.out:
        Path(args.out).write_text(rendered)
        logger.info("Wrote table to %s", args.out)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
