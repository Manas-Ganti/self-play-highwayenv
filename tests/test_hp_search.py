"""The HP search must search the same thing whether it runs as a loop or a SLURM array."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.hp_search import collect, planned_trials, trial_path

SPACE = yaml.safe_load(Path("configs/search_space.yaml").read_text())


@pytest.mark.parametrize("algo", ["ppo", "grpo"])
def test_array_task_i_gets_the_same_params_as_loop_iteration_i(algo):
    plan = planned_trials(SPACE, algo, SPACE["n_trials"])
    # An array task re-plans from scratch in a fresh process; it must get the same list.
    assert planned_trials(SPACE, algo, SPACE["n_trials"]) == plan
    # A shortened smoke search is a prefix of the full one, not a different search.
    assert planned_trials(SPACE, algo, 5) == plan[:5]


def test_both_algorithms_draw_identical_shared_values_on_every_trial():
    ppo = planned_trials(SPACE, "ppo", SPACE["n_trials"])
    grpo = planned_trials(SPACE, "grpo", SPACE["n_trials"])
    for p, g in zip(ppo, grpo, strict=True):
        assert {k: p[k] for k in SPACE["shared"]} == {k: g[k] for k in SPACE["shared"]}


def _write_trials(out: Path, algo: str, scores: list[float]) -> None:
    (out / "trials").mkdir(parents=True, exist_ok=True)
    for i, score in enumerate(scores):
        trial = {"trial": i, "params": {}, "score": score, "run_dir": f"r{i}"}
        trial_path(out, algo, i).write_text(json.dumps(trial))


def test_collect_refuses_an_incomplete_search(tmp_path):
    _write_trials(tmp_path, "ppo", [1.0, 2.0])
    with pytest.raises(SystemExit, match="1/3 trials"):
        collect(SPACE, "ppo", 3, tmp_path)


def test_collect_picks_the_best_finite_score_and_writes_the_receipt(tmp_path):
    _write_trials(tmp_path, "ppo", [1.0, float("nan"), 5.0])
    payload = json.loads(collect(SPACE, "ppo", 3, tmp_path).read_text())
    assert payload["best"]["trial"] == 2
    assert payload["budget"]["n_trials"] == 3


def test_collect_rejects_mismatched_budgets(tmp_path):
    _write_trials(tmp_path, "ppo", [1.0, 2.0, 3.0])
    _write_trials(tmp_path, "grpo", [1.0, 2.0])
    collect(SPACE, "ppo", 3, tmp_path)  # a search is fine on its own ...
    with pytest.raises(SystemExit, match="budgets differ"):
        collect(SPACE, "grpo", 2, tmp_path)  # ... but not next to a larger one
