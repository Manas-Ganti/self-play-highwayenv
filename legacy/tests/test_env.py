"""Environment and config tests.

The config-layer tests run anywhere; the live-env tests require highway-env and
are skipped when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.config import EnvConfig, VehicleConfig
from envs.env_factory import _coerce_config
from envs.racetrack_env import FEATURE_ORDER, features_to_vector

_HAS_HIGHWAY = __import__("importlib").util.find_spec("highway_env") is not None
requires_highway = pytest.mark.skipif(
    not _HAS_HIGHWAY, reason="highway-env not installed"
)


# -- config layer (no heavy deps) -------------------------------------------
def test_flat_config_coercion():
    cfg = _coerce_config({"vehicle_drag": 0.7, "vehicle_mass_scale": 1.5, "track_width": 8})
    assert cfg.vehicle.drag == 0.7
    assert cfg.vehicle.mass_scale == 1.5
    assert cfg.track_width == 8


def test_nested_config_coercion():
    cfg = _coerce_config({"vehicle": {"drag": 0.3}})
    assert cfg.vehicle.drag == 0.3


def test_agent_ids():
    assert EnvConfig(n_agents=2).agent_ids == ["agent_0", "agent_1"]


def test_asymmetric_vehicle_validation():
    with pytest.raises(ValueError):
        EnvConfig(n_agents=2, agent_vehicles=[VehicleConfig()])  # wrong length


def test_vehicle_for_falls_back_to_shared():
    cfg = EnvConfig(vehicle=VehicleConfig(drag=0.9))
    assert cfg.vehicle_for(0).drag == 0.9


def test_npc_bounds_validation():
    with pytest.raises(ValueError):
        EnvConfig(npc_min=5, npc_max=1)


def test_feature_vector_order():
    feats = {k: float(i) for i, k in enumerate(FEATURE_ORDER)}
    vec = features_to_vector(feats)
    assert np.allclose(vec, np.arange(len(FEATURE_ORDER)))


# -- live environment (needs highway-env) -----------------------------------
@requires_highway
def test_make_env_reset_step():
    from envs.env_factory import make_env

    env = make_env({"reward": "sparse", "duration": 5, "seed": 0})
    obs, infos = env.reset(seed=0)
    assert set(obs) == {"agent_0", "agent_1"}
    for a in obs:
        assert obs[a].shape == (len(FEATURE_ORDER),)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs2, rewards, term, trunc, infos = env.step(actions)
    assert set(rewards) == {"agent_0", "agent_1"}
    env.close()
