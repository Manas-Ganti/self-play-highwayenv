"""Advantage math, checked against hand-computed values.

The group-relative advantage is the single most important function in the project
(CLAUDE.md §8 calls it out by name): if it is subtly wrong, GRPO still trains,
still produces smooth curves, and still yields a publishable-looking number that
means nothing. So it is tested against arithmetic done by hand, not against
whatever the implementation happens to return.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algos.common.buffer import GroupBuffer, RolloutBuffer, group_relative_advantages


class TestGroupRelativeAdvantages:
    def test_hand_computed_single_group(self):
        # returns [1, 2, 3, 4] -> mean 2.5, population std sqrt(1.25) = 1.118034
        returns = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array([0, 0, 0, 0])

        std = np.sqrt(1.25)
        expected = np.array([-1.5, -0.5, 0.5, 1.5]) / (std + 1e-8)

        got = group_relative_advantages(returns, groups)
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_groups_are_normalised_independently(self):
        # Group 1's returns are group 0's plus 100. Because each group is
        # standardised against its own mean, both groups must yield *identical*
        # advantages -- a group-relative advantage carries no information about
        # how good the group's starting state was, only about within-group rank.
        returns = np.array([1.0, 2.0, 3.0, 101.0, 102.0, 103.0])
        groups = np.array([0, 0, 0, 1, 1, 1])

        adv = group_relative_advantages(returns, groups)
        np.testing.assert_allclose(adv[:3], adv[3:], rtol=1e-6)

    def test_zero_mean_within_each_group(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(size=24)
        groups = np.repeat(np.arange(3), 8)

        adv = group_relative_advantages(returns, groups)
        for gid in range(3):
            assert abs(adv[groups == gid].mean()) < 1e-6
            assert abs(adv[groups == gid].std() - 1.0) < 1e-5

    def test_degenerate_group_yields_zero_not_nan(self):
        # All returns identical -> std 0. This must produce zero advantages (no
        # gradient) rather than a division blow-up that would poison the update.
        adv = group_relative_advantages(np.array([5.0, 5.0, 5.0]), np.array([0, 0, 0]))
        np.testing.assert_allclose(adv, np.zeros(3), atol=1e-6)
        assert np.isfinite(adv).all()

    def test_uses_population_std_not_sample_std(self):
        # Population std of [0, 2] is 1.0; the sample (ddof=1) std is sqrt(2).
        # The GRPO literature standardises with the population convention, and
        # the difference is a systematic ~sqrt(G/(G-1)) rescaling of every
        # advantage, i.e. a silent change to the effective learning rate.
        adv = group_relative_advantages(np.array([0.0, 2.0]), np.array([0, 0]))
        np.testing.assert_allclose(adv, np.array([-1.0, 1.0]), rtol=1e-6)


class TestGroupBufferBroadcast:
    def test_advantage_is_broadcast_to_every_timestep(self):
        buffer = GroupBuffer(device=torch.device("cpu"))
        # Two episodes in one group, of different lengths (3 and 2 steps).
        buffer.add_episode(np.zeros((3, 4)), np.zeros((3, 2)), np.zeros(3), 10.0, group_id=0)
        buffer.add_episode(np.zeros((2, 4)), np.zeros((2, 2)), np.zeros(2), 20.0, group_id=0)

        episode_adv = buffer.compute_advantages()
        # Returns [10, 20]: mean 15, population std 5 -> advantages [-1, +1].
        np.testing.assert_allclose(episode_adv, np.array([-1.0, 1.0]), rtol=1e-5)

        batch = next(buffer.batches(batch_size=100))
        advantages = batch.advantages.numpy()
        assert advantages.shape == (5,)
        # Three timesteps carry -1, two carry +1 -- one scalar per episode, repeated.
        assert sorted(np.round(advantages, 5)) == sorted([-1.0, -1.0, -1.0, 1.0, 1.0])

    def test_transitions_and_episode_counts(self):
        buffer = GroupBuffer(device=torch.device("cpu"))
        buffer.add_episode(np.zeros((7, 4)), np.zeros((7, 2)), np.zeros(7), 1.0, group_id=0)
        buffer.add_episode(np.zeros((3, 4)), np.zeros((3, 2)), np.zeros(3), 2.0, group_id=1)
        assert buffer.n_episodes == 2
        assert buffer.n_transitions == 10


class TestGAE:
    def _buffer(self, gamma=0.99, lam=0.95, n_steps=3):
        return RolloutBuffer(
            n_steps=n_steps,
            n_envs=1,
            obs_dim=2,
            act_dim=1,
            device=torch.device("cpu"),
            gamma=gamma,
            gae_lambda=lam,
        )

    def _add(self, buf, reward, value, terminated=0.0, truncated=0.0, final_value=0.0):
        buf.add(
            obs=np.zeros((1, 2), dtype=np.float32),
            action=np.zeros((1, 1), dtype=np.float32),
            log_prob=np.zeros(1, dtype=np.float32),
            reward=np.array([reward], dtype=np.float32),
            value=np.array([value], dtype=np.float32),
            terminated=np.array([terminated], dtype=np.float32),
            truncated=np.array([truncated], dtype=np.float32),
            final_value=np.array([final_value], dtype=np.float32),
        )

    def test_hand_computed_gae_no_termination(self):
        gamma, lam = 0.99, 0.95
        buf = self._buffer(gamma, lam)
        # rewards 1, 1, 1 with values 0.5, 0.5, 0.5 and a bootstrap V(s_3) = 0.5
        for _ in range(3):
            self._add(buf, reward=1.0, value=0.5)
        buf.compute_gae(last_value=np.array([0.5], dtype=np.float32))

        delta = 1.0 + gamma * 0.5 - 0.5  # identical at every step = 0.995
        a2 = delta
        a1 = delta + gamma * lam * a2
        a0 = delta + gamma * lam * a1

        np.testing.assert_allclose(
            buf.advantages[:3, 0], np.array([a0, a1, a2]), rtol=1e-5
        )
        # returns = advantages + values
        np.testing.assert_allclose(
            buf.returns[:3, 0], np.array([a0, a1, a2]) + 0.5, rtol=1e-5
        )

    def test_termination_zeroes_the_bootstrap(self):
        gamma, lam = 0.99, 0.95
        buf = self._buffer(gamma, lam, n_steps=2)
        self._add(buf, reward=1.0, value=0.5)
        self._add(buf, reward=-5.0, value=0.5, terminated=1.0)  # crashed
        buf.compute_gae(last_value=np.array([0.5], dtype=np.float32))

        # Terminal step: the future is worth exactly 0, so the bootstrap term
        # vanishes and delta = r - V. `last_value` must be ignored here.
        a1 = -5.0 + gamma * 0.0 - 0.5
        # Step 0 is in the *same episode* as the crash, so the crash absolutely
        # does propagate back to it -- that is how PPO learns the step before the
        # collision was a bad one. GAE is only cut at an episode *boundary*.
        a0 = (1.0 + gamma * 0.5 - 0.5) + gamma * lam * a1

        np.testing.assert_allclose(buf.advantages[:2, 0], np.array([a0, a1]), rtol=1e-5)
        assert a0 < 0.0, "the crash should make the preceding step look bad"

    def test_gae_does_not_flow_across_an_episode_boundary(self):
        """The step after a terminal one belongs to a new episode; GAE must cut."""

        gamma, lam = 0.99, 0.95
        buf = self._buffer(gamma, lam, n_steps=2)
        self._add(buf, reward=-5.0, value=0.5, terminated=1.0)  # episode A ends
        self._add(buf, reward=1.0, value=0.5)  # episode B, step 0
        buf.compute_gae(last_value=np.array([0.5], dtype=np.float32))

        # Episode A's terminal advantage must not inherit anything from episode B.
        a0 = -5.0 - 0.5
        np.testing.assert_allclose(buf.advantages[0, 0], a0, rtol=1e-5)

    def test_truncation_bootstraps_off_final_value(self):
        gamma, lam = 0.99, 0.95
        buf = self._buffer(gamma, lam, n_steps=1)
        # Time limit hit, not a real terminal state: the future is worth V(final_obs).
        self._add(buf, reward=1.0, value=0.5, truncated=1.0, final_value=2.0)
        buf.compute_gae(last_value=np.array([99.0], dtype=np.float32))  # must be ignored

        expected = 1.0 + gamma * 2.0 - 0.5
        np.testing.assert_allclose(buf.advantages[0, 0], expected, rtol=1e-5)

    def test_truncation_and_termination_differ(self):
        """A truncated step must not be treated as terminal.

        Collapsing the two is the classic silent PPO bug: it teaches the agent
        that the world ends at the time limit, biasing every value estimate low.
        """

        gamma = 0.99
        term = self._buffer(gamma, 0.95, n_steps=1)
        self._add(term, reward=1.0, value=0.5, terminated=1.0, final_value=0.0)
        term.compute_gae(last_value=np.array([0.0], dtype=np.float32))

        trunc = self._buffer(gamma, 0.95, n_steps=1)
        self._add(trunc, reward=1.0, value=0.5, truncated=1.0, final_value=2.0)
        trunc.compute_gae(last_value=np.array([0.0], dtype=np.float32))

        assert not np.isclose(term.advantages[0, 0], trunc.advantages[0, 0])
        np.testing.assert_allclose(
            trunc.advantages[0, 0] - term.advantages[0, 0], gamma * 2.0, rtol=1e-5
        )


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_advantages_sum_to_zero_within_group(group_size):
    rng = np.random.default_rng(1)
    n_groups = 3
    returns = rng.normal(size=n_groups * group_size)
    groups = np.repeat(np.arange(n_groups), group_size)

    adv = group_relative_advantages(returns, groups)
    for gid in range(n_groups):
        assert abs(adv[groups == gid].sum()) < 1e-5
