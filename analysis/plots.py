"""Phase 6 figures: rliable IQM plots and learning curves.

    python analysis/plots.py --results results/ --out report/figures

Every aggregate claim is an IQM with a stratified bootstrap CI across seeds
(CLAUDE.md §7). A single-seed curve is never a finding, so the learning-curve plot
draws the across-seed IQM with a shaded CI band and, at most, faint individual
seeds behind it -- never a lone seed on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.stats import bootstrap_ci, iqm, iqm_ci  # noqa: E402

ALGO_COLORS = {"ppo": "#4C6EF5", "grpo": "#F03E3E"}


def load_solo_evals(results_dir: Path, algo: str) -> dict[str, list[float]]:
    """Collect each seed's final solo-eval metrics for ``algo``."""

    metrics: dict[str, list[float]] = {}
    for run in sorted(results_dir.glob(f"{algo}_seed*")):
        path = run / "solo_eval.json"
        if not path.exists():
            continue
        for key, value in json.loads(path.read_text()).items():
            metrics.setdefault(key.replace("final_eval/", ""), []).append(float(value))
    return metrics


def plot_iqm_comparison(results_dir: Path, out: Path) -> None:
    """Headline RQ1 figure: PPO vs GRPO solo performance, IQM with 95% CIs."""

    keys = [
        ("mean_return", "Episodic return"),
        ("lap_completion_rate", "Lap completion rate"),
        ("collision_rate", "Collision rate"),
        ("mean_distance", "Distance (m)"),
    ]
    data = {algo: load_solo_evals(results_dir, algo) for algo in ("ppo", "grpo")}
    if not any(data.values()):
        print("no finished runs found; complete Phase 4 first")
        return

    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
    for ax, (key, label) in zip(np.atleast_1d(axes), keys, strict=True):
        for i, algo in enumerate(("ppo", "grpo")):
            scores = np.array(data[algo].get(key, []), dtype=np.float64)
            if scores.size == 0:
                continue
            ci = iqm_ci(scores)
            ax.errorbar(
                i,
                ci.point,
                yerr=[[ci.point - ci.low], [ci.high - ci.point]],
                fmt="o",
                capsize=6,
                markersize=9,
                color=ALGO_COLORS[algo],
                label=algo.upper(),
            )
            # Individual seeds, faint, behind the estimate -- visible but never the claim.
            ax.scatter(
                np.full(scores.size, i) + 0.12,
                scores,
                s=14,
                alpha=0.35,
                color=ALGO_COLORS[algo],
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PPO", "GRPO"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Solo performance, IQM with stratified bootstrap 95% CIs (5 seeds)", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def h1_verdict(results_dir: Path) -> str:
    """H1: GRPO reaches >=90% of PPO's solo final performance at matched steps."""

    ppo = np.array(load_solo_evals(results_dir, "ppo").get("mean_return", []))
    grpo = np.array(load_solo_evals(results_dir, "grpo").get("mean_return", []))
    if ppo.size == 0 or grpo.size == 0:
        return "H1: not evaluable (missing runs)"

    ppo_iqm, grpo_iqm = iqm(ppo), iqm(grpo)
    ratio = grpo_iqm / ppo_iqm if ppo_iqm else float("nan")
    verdict = "SUPPORTED" if ratio >= 0.90 else "NOT SUPPORTED"
    return (
        f"H1 ({verdict}): GRPO reaches {ratio:.1%} of PPO's solo IQM return "
        f"(GRPO {grpo_iqm:.1f} vs PPO {ppo_iqm:.1f}); threshold was 90%."
    )


def h2_verdict(h2h_path: Path) -> str:
    """H2: GRPO degrades more than PPO from solo to competition."""

    if not h2h_path.exists():
        return "H2: not evaluable (run scripts/evaluate_h2h.py first)"

    rows = json.loads(h2h_path.read_text())["degradation"]
    by_algo = {
        algo: np.array(
            [r["collision_degradation"] for r in rows if r["algo"] == algo], dtype=np.float64
        )
        for algo in ("ppo", "grpo")
    }
    if any(v.size == 0 for v in by_algo.values()):
        return "H2: not evaluable (missing degradation rows)"

    ppo_d = bootstrap_ci(by_algo["ppo"], statistic=iqm)
    grpo_d = bootstrap_ci(by_algo["grpo"], statistic=iqm)
    verdict = "SUPPORTED" if grpo_d.point > ppo_d.point else "NOT SUPPORTED"
    return (
        f"H2 ({verdict}): collision-rate degradation solo->h2h is "
        f"GRPO {grpo_d} vs PPO {ppo_d}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plot_iqm_comparison(args.results, args.out / "solo_iqm.png")

    print()
    print(h1_verdict(args.results))
    print(h2_verdict(args.results / "h2h" / "h2h_results.json"))


if __name__ == "__main__":
    main()
