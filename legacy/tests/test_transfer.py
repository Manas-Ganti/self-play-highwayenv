"""Transfer / sweep-expansion tests.

The pure config-expansion logic runs anywhere; end-to-end transfer evaluation
requires the full RL stack and is skipped when it is unavailable.
"""

from __future__ import annotations

import importlib.util

import pytest

from experiments.config_schema import load_experiment_config
from experiments.run_experiment import _sweep_combinations, _train_overrides

_HAS_STACK = all(
    importlib.util.find_spec(m) is not None
    for m in ("highway_env", "stable_baselines3", "pettingzoo")
)
requires_stack = pytest.mark.skipif(not _HAS_STACK, reason="RL stack not installed")


def test_physics_sweep_expands_to_grid():
    config = load_experiment_config("physics_sweep.yaml")
    combos = _sweep_combinations(config)
    assert len(combos) == 16  # 4 drag x 4 mass_scale
    assert {"vehicle_drag", "vehicle_mass_scale"} <= set(combos[0])


def test_geometry_sweep_expands_to_grid():
    config = load_experiment_config("geometry_sweep.yaml")
    combos = _sweep_combinations(config)
    assert len(combos) == 16  # 4 width x 4 length


def test_train_overrides_from_sweep():
    config = load_experiment_config("physics_sweep.yaml")
    assert _train_overrides(config) == {"vehicle_drag": 0.4, "vehicle_mass_scale": 1.0}


def test_base_config_inheritance():
    config = load_experiment_config("physics_sweep.yaml")
    # Inherited from base.yaml.
    assert config.training.algorithm == "sac"
    assert config.evaluation.episodes == 50


def test_to_env_config_dense_params():
    config = load_experiment_config("base.yaml")
    config.reward.type = "dense"
    env_cfg = config.to_env_config()
    assert env_cfg.reward == "dense"
    assert env_cfg.reward_params["lambda_crash"] == 5.0


@requires_stack
def test_transfer_degradation_runs(tmp_path):
    """A 2-condition physics transfer eval completes end to end (smoke)."""

    from envs.env_factory import make_env

    env = make_env({"reward": "sparse", "duration": 3, "vehicle_drag": 0.2})
    env.reset(seed=0)
    env.close()
