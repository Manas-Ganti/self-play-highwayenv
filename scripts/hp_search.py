"""Phase 3 -- matched-budget random hyperparameter search.

    python scripts/hp_search.py --algo ppo              # all trials, sequentially
    python scripts/hp_search.py --algo ppo --trial-index 7   # one trial (a SLURM array task)
    python scripts/hp_search.py --algo ppo --collect         # aggregate the trial files

On ARC each trial is one array task (arc/search.slurm). Trials are pre-sampled in
order from one seeded stream, so trial i has identical parameters whether it ran
in a loop or as array task i -- parallelism cannot change what was searched.

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


def sample_trial(
    space: dict, algo: str, shared_rng: np.random.Generator, algo_rng: np.random.Generator
) -> dict[str, Any]:
    """Sample one trial: the shared block plus this algorithm's own block.

    The shared block and the algorithm block draw from *separate* streams. With one
    stream, PPO (4 algorithm-specific draws per trial) and GRPO (1) would consume it
    at different rates, so from trial 1 onward their "shared" values would diverge.
    Separate streams keep trial i a true matched pair across algorithms.
    """

    params = {k: sample_value(v, shared_rng) for k, v in space["shared"].items()}
    params.update({k: sample_value(v, algo_rng) for k, v in space.get(algo, {}).items()})
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


def planned_trials(space: dict, algo: str, n_trials: int) -> list[dict[str, Any]]:
    """Every trial's parameters, drawn in order from the pre-registered seed.

    Same seed for both algorithms => trial i draws the same shared values. Drawing
    the whole plan up front makes trial i independent of which trials ran before it
    in this process, which is what lets the search run as a SLURM array.
    """

    seed = int(space["trial_seed"])
    shared_rng = np.random.default_rng([seed, 0])
    algo_rng = np.random.default_rng([seed, 1])
    return [sample_trial(space, algo, shared_rng, algo_rng) for _ in range(n_trials)]


def run_trial(
    space: dict, algo: str, index: int, params: dict[str, Any], base_config: Path, out: Path
) -> dict[str, Any]:
    """Train one trial at the search budget and write its result file."""

    cfg = RunConfig.load(base_config)
    algo_cfg = cfg.algo_config.model_copy(update=params)
    updates: dict[str, Any] = {
        "name": f"search/{algo}_trial{index:02d}",
        "total_steps": int(space["trial_steps"]),
        "seed": int(space["trial_seed"]),
        "use_wandb": False,
        "group": f"{algo}_search",
        algo: algo_cfg,
    }
    if algo == "grpo":
        updates["n_envs"] = algo_cfg.group_size * algo_cfg.n_groups
    cfg = cfg.model_copy(update=updates)

    logger.info("[trial %d] %s", index, params)
    run_dir = train(cfg)

    metric = space["metric"]
    score = json.loads((run_dir / "solo_eval.json").read_text()).get(metric, float("nan"))
    trial = {"trial": index, "params": params, "score": score, "run_dir": str(run_dir)}
    trial_path(out, algo, index).write_text(json.dumps(trial, indent=2, default=float))
    logger.info("[trial %d] %s = %.2f", index, metric, score)
    return trial


def trial_path(out: Path, algo: str, index: int) -> Path:
    return out / "trials" / f"{algo}_trial{index:02d}.json"


def collect(space: dict, algo: str, n_trials: int, out: Path) -> Path:
    """Aggregate per-trial files into the search result + budget receipt.

    Refuses to pick a winner from an incomplete search: best-of-27 is a smaller
    budget than best-of-30, and silently comparing it to the other algorithm's
    full search is exactly the mismatch §8 forbids.
    """

    missing = [i for i in range(n_trials) if not trial_path(out, algo, i).exists()]
    if missing:
        raise SystemExit(
            f"{algo}: {len(missing)}/{n_trials} trials have no result file: {missing}. "
            "Re-run those indices (e.g. --array=" + ",".join(map(str, missing)) + ")."
        )
    trials = [json.loads(trial_path(out, algo, i).read_text()) for i in range(n_trials)]

    metric = space["metric"]
    best = max(trials, key=lambda t: (t["score"] if np.isfinite(t["score"]) else -np.inf))
    payload = {
        "algo": algo,
        # The receipt that makes matched budgets checkable rather than claimed.
        "budget": {
            "n_trials": n_trials,
            "trial_steps": int(space["trial_steps"]),
            "shared_grid": space["shared"],
            "metric": metric,
            "trial_seed": int(space["trial_seed"]),
        },
        "best": best,
        "trials": trials,
    }
    out_path = out / f"{algo}_search.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("best trial %d (%s=%.2f) -> %s", best["trial"], metric, best["score"], out_path)

    assert_matched_budgets(out)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "grpo"], required=True)
    parser.add_argument("--space", type=Path, default=Path("configs/search_space.yaml"))
    parser.add_argument("--base-config", type=Path, help="defaults to configs/<algo>_seed0.yaml")
    parser.add_argument("--out", type=Path, default=Path("results/search"))
    parser.add_argument("--trials", type=int, help="override; only to shorten a smoke test")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trial-index", type=int, help="run only this trial (SLURM array task)")
    mode.add_argument("--collect", action="store_true", help="aggregate finished trial files")
    args = parser.parse_args()

    space = yaml.safe_load(args.space.read_text())
    assert_shared_grid_covered(space)

    n_trials = args.trials or int(space["n_trials"])
    base_config = args.base_config or Path(f"configs/{args.algo}_seed0.yaml")
    (args.out / "trials").mkdir(parents=True, exist_ok=True)

    if args.collect:
        collect(space, args.algo, n_trials, args.out)
        return

    plan = planned_trials(space, args.algo, n_trials)
    if args.trial_index is not None:
        if not 0 <= args.trial_index < n_trials:
            raise SystemExit(f"--trial-index must be in [0, {n_trials}), got {args.trial_index}")
        run_trial(space, args.algo, args.trial_index, plan[args.trial_index], base_config, args.out)
        return

    for i, params in enumerate(plan):
        run_trial(space, args.algo, i, params, base_config, args.out)
    collect(space, args.algo, n_trials, args.out)


if __name__ == "__main__":
    main()
