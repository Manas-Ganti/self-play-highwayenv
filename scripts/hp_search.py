"""Phase 3 -- matched-budget random hyperparameter search.

    python scripts/hp_search.py --algo ppo  --trials 30
    python scripts/hp_search.py --algo grpo --trials 30

The matching *is* the experiment (CLAUDE.md §1, §8). This script therefore refuses
to run a search that would break it:

* the shared grid is asserted identical for both algorithms (it is one block in
  ``configs/search_space.yaml``, read by both);
* the trial count, per-trial step budget, seed and selection metric come from that
  same file, so neither algorithm can be given a longer look;
* a completed search writes a ``budget`` receipt into its results, and
  ``assert_matched_budgets`` compares the two receipts before best-config
  selection is trusted.

If GRPO underperforms, the response is to report that, not to buy it more trials.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from algos.common.config import SHARED_HPARAMS, RunConfig  # noqa: E402
from scripts.train import train  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("racing-grpo")


def sample_value(spec: dict[str, Any], rng: np.random.Generator):
    """Draw one hyperparameter value from its pre-registered distribution."""

    dist = spec["distribution"]
    if dist == "choice":
        idx = int(rng.integers(len(spec["values"])))
        return spec["values"][idx]
    if dist == "log_uniform":
        low, high = np.log(spec["low"]), np.log(spec["high"])
        return float(np.exp(rng.uniform(low, high)))
    if dist == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    raise ValueError(f"unknown distribution '{dist}'")


def sample_trial(space: dict, algo: str, rng: np.random.Generator) -> dict[str, Any]:
    """Sample one trial: the shared block plus this algorithm's own block.

    Both algorithms draw the shared hyperparameters from the *same* grid with the
    *same* RNG stream, so trial i is a matched pair across algorithms.
    """

    params = {k: sample_value(v, rng) for k, v in space["shared"].items()}
    params.update({k: sample_value(v, rng) for k, v in space.get(algo, {}).items()})
    return params


def assert_shared_grid_covered(space: dict) -> None:
    """Fail loudly if the shared grid does not cover every shared hyperparameter."""

    missing = set(SHARED_HPARAMS) - set(space["shared"])
    if missing:
        raise SystemExit(
            f"configs/search_space.yaml is missing shared hyperparameters {sorted(missing)}. "
            "PPO and GRPO must search identical grids over everything they share."
        )
    stray = set(space["shared"]) & (set(space.get("ppo", {})) | set(space.get("grpo", {})))
    if stray:
        raise SystemExit(
            f"hyperparameters {sorted(stray)} appear in both the shared block and an "
            "algorithm-specific block, which would give one algorithm a second look at them."
        )


def assert_matched_budgets(out_dir: Path) -> None:
    """Compare the two searches' receipts; refuse to proceed if they differ."""

    receipts = {}
    for algo in ("ppo", "grpo"):
        path = out_dir / f"{algo}_search.json"
        if path.exists():
            receipts[algo] = json.loads(path.read_text())["budget"]

    if len(receipts) < 2:
        return
    if receipts["ppo"] != receipts["grpo"]:
        raise SystemExit(
            f"search budgets differ -- the comparison is invalid.\n"
            f"  ppo:  {receipts['ppo']}\n  grpo: {receipts['grpo']}"
        )
    logger.info("search budgets match: %s", receipts["ppo"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "grpo"], required=True)
    parser.add_argument("--space", type=Path, default=Path("configs/search_space.yaml"))
    parser.add_argument("--base-config", type=Path, help="defaults to configs/<algo>_seed0.yaml")
    parser.add_argument("--out", type=Path, default=Path("results/search"))
    parser.add_argument("--trials", type=int, help="override; only to shorten a smoke test")
    args = parser.parse_args()

    space = yaml.safe_load(args.space.read_text())
    assert_shared_grid_covered(space)

    n_trials = args.trials or int(space["n_trials"])
    trial_steps = int(space["trial_steps"])
    metric = space["metric"]
    base_config = args.base_config or Path(f"configs/{args.algo}_seed0.yaml")

    args.out.mkdir(parents=True, exist_ok=True)
    # Same seed for both algorithms => trial i draws the same shared values.
    rng = np.random.default_rng(int(space["trial_seed"]))

    trials = []
    for i in range(n_trials):
        params = sample_trial(space, args.algo, rng)
        cfg = RunConfig.load(base_config)

        algo_cfg = cfg.algo_config.model_copy(update=params)
        updates: dict[str, Any] = {
            "name": f"search/{args.algo}_trial{i:02d}",
            "total_steps": trial_steps,
            "seed": int(space["trial_seed"]),
            "use_wandb": False,
            "group": f"{args.algo}_search",
            args.algo: algo_cfg,
        }
        if args.algo == "grpo":
            updates["n_envs"] = algo_cfg.group_size * algo_cfg.n_groups
        cfg = cfg.model_copy(update=updates)

        logger.info("[trial %d/%d] %s", i + 1, n_trials, params)
        run_dir = train(cfg)

        score = json.loads((run_dir / "solo_eval.json").read_text()).get(metric, float("nan"))
        trials.append({"trial": i, "params": params, "score": score, "run_dir": str(run_dir)})
        logger.info("[trial %d/%d] %s = %.2f", i + 1, n_trials, metric, score)

    best = max(trials, key=lambda t: (t["score"] if np.isfinite(t["score"]) else -np.inf))
    payload = {
        "algo": args.algo,
        # The receipt that makes matched budgets checkable rather than claimed.
        "budget": {
            "n_trials": n_trials,
            "trial_steps": trial_steps,
            "shared_grid": space["shared"],
            "metric": metric,
            "trial_seed": int(space["trial_seed"]),
        },
        "best": best,
        "trials": trials,
    }
    out_path = args.out / f"{args.algo}_search.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("best trial %d (%s=%.2f) -> %s", best["trial"], metric, best["score"], out_path)

    assert_matched_budgets(args.out)


if __name__ == "__main__":
    main()
