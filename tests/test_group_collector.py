"""Group collector determinism -- the assertion GRPO's validity rests on.

GRPO uses the group's mean return as a value baseline. That is only legitimate if
every rollout in the group starts from *the same state*: then the group mean
estimates that state's value, which is exactly what a critic would have supplied.

If the resets silently diverged -- a different traffic layout per member, say --
the group mean would be an average over different states, the baseline would be
biased, and nothing in the training curves would look wrong. Hence CLAUDE.md §4.2
requires this to be built and tested in isolation, and hence these tests:

* members of a group start from identical observations,
* different groups start from different observations,
* members diverge once they start acting stochastically.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algos.common.buffer import GroupBuffer
from algos.common.collectors import GroupCollector
from algos.common.nets import ActorCritic
from envs import make_vec_env
from envs.config import EnvConfig, RewardConfig

GROUP_SIZE = 4
N_GROUPS = 2


@pytest.fixture(scope="module")
def env_config() -> EnvConfig:
    # Short episodes keep the collector tests fast; the machinery under test is
    # unaffected by the horizon.
    return EnvConfig(reward=RewardConfig.load(), duration=6)


@pytest.fixture
def vec_env(env_config):
    env = make_vec_env(env_config, n_envs=GROUP_SIZE * N_GROUPS, asynchronous=False)
    yield env
    env.close()


def test_group_members_share_an_initial_observation(vec_env):
    collector = GroupCollector(vec_env, torch.device("cpu"), group_size=GROUP_SIZE, seed=0)
    obs, _ = vec_env.reset(seed=collector._group_seeds())

    for gid in range(N_GROUPS):
        members = obs[gid * GROUP_SIZE : (gid + 1) * GROUP_SIZE]
        for k in range(1, GROUP_SIZE):
            np.testing.assert_array_equal(
                members[0],
                members[k],
                err_msg=f"group {gid} member {k} did not start from the group's state",
            )


def test_different_groups_start_from_different_states(vec_env):
    collector = GroupCollector(vec_env, torch.device("cpu"), group_size=GROUP_SIZE, seed=0)
    obs, _ = vec_env.reset(seed=collector._group_seeds())

    first = obs[0]
    second = obs[GROUP_SIZE]
    assert not np.array_equal(first, second), (
        "two groups drew the same reset state; the batch carries less information "
        "than it appears to"
    )


def test_members_diverge_after_stochastic_actions(vec_env):
    torch.manual_seed(0)
    collector = GroupCollector(vec_env, torch.device("cpu"), group_size=GROUP_SIZE, seed=0)
    obs, _ = vec_env.reset(seed=collector._group_seeds())

    model = ActorCritic(
        int(vec_env.single_observation_space.shape[0]),
        int(vec_env.single_action_space.shape[0]),
        with_critic=False,
    )
    with torch.no_grad():
        action, _, _ = model.act(torch.as_tensor(obs, dtype=torch.float32))
    action_np = action.numpy()

    # Same state + same weights, but sampled actions -> the members must differ.
    group_0 = action_np[:GROUP_SIZE]
    assert not np.allclose(group_0[0], group_0[1]), (
        "group members sampled identical actions; without action noise the group "
        "has no within-group spread and GRPO has no learning signal"
    )

    obs, *_ = vec_env.step(action_np)
    after = obs[:GROUP_SIZE]
    assert not np.allclose(after[0], after[1]), "members did not diverge after acting"


def test_group_ids_are_assigned_group_major(vec_env, env_config):
    """Episode i must be tagged with the group whose reset state it actually used."""

    torch.manual_seed(0)
    collector = GroupCollector(vec_env, torch.device("cpu"), group_size=GROUP_SIZE, seed=1)
    buffer = GroupBuffer(device=torch.device("cpu"))
    model = ActorCritic(
        int(vec_env.single_observation_space.shape[0]),
        int(vec_env.single_action_space.shape[0]),
        with_critic=False,
    )

    stats = collector.collect(model, buffer, max_steps=env_config.max_steps + 1)

    assert buffer.n_episodes == GROUP_SIZE * N_GROUPS, "every env must yield one episode"
    counts = np.bincount(buffer.group_ids, minlength=N_GROUPS)
    np.testing.assert_array_equal(counts, np.full(N_GROUPS, GROUP_SIZE))
    assert len(stats.returns) == GROUP_SIZE * N_GROUPS
    assert collector.total_steps > 0


def test_collect_produces_finite_advantages(vec_env, env_config):
    torch.manual_seed(0)
    collector = GroupCollector(vec_env, torch.device("cpu"), group_size=GROUP_SIZE, seed=2)
    buffer = GroupBuffer(device=torch.device("cpu"))
    model = ActorCritic(
        int(vec_env.single_observation_space.shape[0]),
        int(vec_env.single_action_space.shape[0]),
        with_critic=False,
    )

    collector.collect(model, buffer, max_steps=env_config.max_steps + 1)
    advantages = buffer.compute_advantages()

    assert np.isfinite(advantages).all()
    assert advantages.shape == (GROUP_SIZE * N_GROUPS,)


def test_group_size_must_divide_n_envs(vec_env):
    with pytest.raises(ValueError, match="multiple of group_size"):
        GroupCollector(vec_env, torch.device("cpu"), group_size=5, seed=0)
