"""Experiment entry point: train once, evaluate on every test condition.

    python experiments/run_experiment.py --config experiments/configs/base.yaml

For a sweep config, the model is trained a single time on the ``train_on``
condition and then evaluated, without retraining, on every combination produced
by the sweep axes. Results land in ``results/<experiment_name>/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.config_schema import ExperimentConfig, load_experiment_config
from experiments.evaluate import evaluate
from experiments.train import train

logger = logging.getLogger("racetrack_rl")

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def _sweep_combinations(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Expand the sweep axes into the list of evaluation env-override dicts.

    Only physics/geometry env params are mapped to env overrides here; reward
    sweeps are handled as separate training variants by the caller.
    """

    if config.sweep is None:
        return [{}]

    env_axes = [ax for ax in config.sweep.axes if ax.param.startswith("env.")]
    if not env_axes:
        return [{}]

    names = [ax.param.split(".", 1)[1] for ax in env_axes]
    combos = []
    for values in itertools.product(*[ax.values for ax in env_axes]):
        combos.append(dict(zip(names, values, strict=True)))
    return combos


def _train_overrides(config: ExperimentConfig) -> dict[str, Any]:
    """The single training condition for the sweep (``train_on`` block)."""

    if config.sweep is None or not isinstance(config.sweep.train_on, dict):
        return {}
    return dict(config.sweep.train_on)


def run(config_path: str | Path, seed: int = 0) -> Path:
    """Run a full experiment and return its output directory.

    Parameters
    ----------
    config_path : str or Path
        Path to the experiment YAML.
    seed : int
        Master seed, forwarded to training, env, and samplers.

    Returns
    -------
    Path
        The timestamped results directory.
    """

    config = load_experiment_config(config_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_ROOT / config.experiment_name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config for reproducibility.
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))

    # 1. Train once on the training condition.
    train_overrides = _train_overrides(config)
    for k, v in train_overrides.items():
        setattr(config.env, k, v)
    logger.info("Training condition: %s", train_overrides or "(base env)")
    model_path = train(config, output_dir=out_dir, seed=seed)

    # 2. Evaluate on every sweep combination without retraining.
    rows: list[dict[str, Any]] = []
    for combo in _sweep_combinations(config):
        metrics = evaluate(
            model_path=model_path,
            env_overrides=combo,
            episodes=config.evaluation.episodes,
            record_trajectories=config.evaluation.record_trajectories,
            seed=seed,
        )
        for m in metrics:
            rows.append({**combo, **m.as_dict()})

    _write_results(out_dir, rows)
    logger.info("Experiment complete: %s", out_dir)
    return out_dir


def _write_results(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write evaluation rows as both CSV and JSON."""

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    if rows:
        fieldnames = sorted({k for row in rows for k in row})
        with (out_dir / "results.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a named racetrack experiment.")
    p.add_argument("--config", required=True, help="Path to experiment YAML.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    run(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
