"""Algorithm smoke tests.

These do not check that PPO and GRPO *learn well* -- that is what Phases 2-4 are
for, with seeds and confidence intervals. They check that each loop runs, that the
gradients are finite, and above all that the structural differences the research
question depends on are really there: GRPO has no critic, and both share the rest.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algos.common.collectors import GroupCollector, VecCollector
from algos.common.config import SHARED_HPARAMS, GRPOConfig, PPOConfig
from algos.common.utils import GradientVarianceTracker
from algos.grpo import GRPO
from algos.ppo import PPO
from envs import make_vec_env
from envs.config import EnvConfig, RewardConfig

GROUP_SIZE = 2
N_GROUPS = 2


@pytest.fixture(scope="module")
def env_config() -> EnvConfig:
    return EnvConfig(reward=RewardConfig.load(), duration=6)


@pytest.fixture
def dims(env_config):
    env = make_vec_env(env_config, n_envs=1, asynchronous=False)
    obs_dim = int(env.single_observation_space.shape[0])
    act_dim = int(env.single_action_space.shape[0])
    env.close()
    return obs_dim, act_dim


class TestStructuralDifferences:
    """The differences that ARE the experiment."""

    def test_grpo_has_no_critic(self, dims):
        obs_dim, act_dim = dims
        grpo = GRPO(obs_dim, act_dim, GRPOConfig(), torch.device("cpu"), n_envs=4)

        assert grpo.model.critic is None
        assert not grpo.model.has_critic
        # And no value parameters reached the optimiser.
        names = [n for n, _ in grpo.model.named_parameters()]
        assert not any(n.startswith("critic") for n in names)

    def test_ppo_has_a_critic(self, dims):
        obs_dim, act_dim = dims
        ppo = PPO(obs_dim, act_dim, PPOConfig(), torch.device("cpu"), n_envs=1)

        assert ppo.model.critic is not None
        assert any(n.startswith("critic") for n, _ in ppo.model.named_parameters())

    def test_grpo_config_has_no_value_hyperparameters(self):
        """No gamma, no gae_lambda, no vf_coef -- there is nothing to bootstrap."""

        fields = set(GRPOConfig.model_fields)
        assert not (fields & {"gamma", "gae_lambda", "vf_coef"})

    def test_both_configs_expose_every_shared_hyperparameter(self):
        """The Phase 3 search can only be matched if both sides have the same knobs."""

        for cfg_cls in (PPOConfig, GRPOConfig):
            missing = set(SHARED_HPARAMS) - set(cfg_cls.model_fields)
            assert not missing, f"{cfg_cls.__name__} is missing {missing}"

    def test_grpo_t_is_not_silently_enabled(self, dims):
        obs_dim, act_dim = dims
        with pytest.raises(NotImplementedError, match="GRPO-t"):
            GRPO(
                obs_dim,
                act_dim,
                GRPOConfig(per_timestep_advantages=True),
                torch.device("cpu"),
                n_envs=4,
            )


class TestTrainingStep:
    def test_ppo_update_runs_and_produces_finite_gradients(self, env_config, dims):
        obs_dim, act_dim = dims
        torch.manual_seed(0)

        env = make_vec_env(env_config, n_envs=2, asynchronous=False)
        cfg = PPOConfig(n_steps=16, batch_size=8, n_epochs=2)
        ppo = PPO(obs_dim, act_dim, cfg, torch.device("cpu"), n_envs=2)
        collector = VecCollector(env, torch.device("cpu"), seed=0)

        before = [p.clone() for p in ppo.model.parameters()]
        ppo.collect(collector)
        metrics = ppo.update()
        env.close()

        assert np.isfinite(metrics["train/loss"])
        assert np.isfinite(metrics["train/value_loss"])
        assert metrics["train/grad_variance"] >= 0.0
        assert collector.total_steps == 16 * 2
        # The policy actually moved.
        after = list(ppo.model.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))

    def test_grpo_update_runs_and_produces_finite_gradients(self, env_config, dims):
        obs_dim, act_dim = dims
        torch.manual_seed(0)

        n_envs = GROUP_SIZE * N_GROUPS
        env = make_vec_env(env_config, n_envs=n_envs, asynchronous=False)
        cfg = GRPOConfig(group_size=GROUP_SIZE, n_groups=N_GROUPS, batch_size=8, n_epochs=2)
        grpo = GRPO(obs_dim, act_dim, cfg, torch.device("cpu"), n_envs=n_envs)
        collector = GroupCollector(env, torch.device("cpu"), group_size=GROUP_SIZE, seed=0)

        before = [p.clone() for p in grpo.model.parameters()]
        grpo.collect(collector, max_steps=env_config.max_steps + 1)
        metrics = grpo.update()
        env.close()

        assert np.isfinite(metrics["train/loss"])
        assert "train/value_loss" not in metrics  # there is no value loss to report
        assert metrics["grpo/n_groups"] == N_GROUPS
        assert metrics["grpo/n_episodes"] == n_envs
        assert 0.0 <= metrics["grpo/degenerate_group_frac"] <= 1.0

        after = list(grpo.model.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))


class TestGradientVariance:
    """Both algorithms log this, so RQ1 gets an explanation and not just a verdict."""

    def test_identical_gradients_have_zero_variance(self):
        tracker = GradientVarianceTracker()
        param = torch.nn.Parameter(torch.zeros(3))

        for _ in range(4):
            param.grad = torch.tensor([1.0, 2.0, 3.0])
            tracker.update([param])

        assert tracker.variance == pytest.approx(0.0, abs=1e-9)
        assert tracker.gradient_norm == pytest.approx(np.sqrt(14.0), rel=1e-5)

    def test_noisy_gradients_have_positive_variance(self):
        tracker = GradientVarianceTracker()
        param = torch.nn.Parameter(torch.zeros(2))

        for grad in ([1.0, 0.0], [-1.0, 0.0], [3.0, 0.0], [-3.0, 0.0]):
            param.grad = torch.tensor(grad)
            tracker.update([param])

        # Coordinate 0 has values [1,-1,3,-3]: mean 0, sample variance 20/3.
        # Coordinate 1 is constant 0. Mean per-coordinate variance = 10/3.
        assert tracker.variance == pytest.approx(10.0 / 3.0, rel=1e-6)
        # Zero mean gradient, non-zero variance -> pure noise, no signal.
        assert tracker.snr == pytest.approx(0.0, abs=1e-9)
