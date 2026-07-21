"""Rollout storage.

Two buffers, because the two algorithms need structurally different things:

* :class:`RolloutBuffer` -- fixed-horizon, time-major, GAE over bootstrapped
  values. PPO's.
* :class:`GroupBuffer` -- variable-length whole episodes tagged by group id, with
  one scalar advantage broadcast to every timestep of an episode. GRPO's.

The GRPO advantage math lives in :func:`group_relative_advantages` and is unit
tested against hand-computed values (``tests/test_advantages.py``); it is the
single most important function in the project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    """One minibatch of transitions, on-device."""

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor | None = None  # PPO only; GRPO has no value target


class RolloutBuffer:
    """Fixed-horizon on-policy buffer with GAE(lambda). Used by PPO."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps, self.n_envs = n_steps, n_envs
        self.device = device
        self.gamma, self.gae_lambda = gamma, gae_lambda

        shape = (n_steps, n_envs)
        self.obs = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        # Terminations and truncations are stored separately and treated
        # differently: a terminal state has value 0, a *truncated* one does not.
        # Collapsing them would silently turn this time-limited task into a
        # shorter-horizon one and bias every value estimate low.
        self.terminated = np.zeros(shape, dtype=np.float32)
        self.truncated = np.zeros(shape, dtype=np.float32)
        # V(final_obs) for envs truncated at step t; 0 otherwise.
        self.final_values = np.zeros(shape, dtype=np.float32)

        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        final_value: np.ndarray,
    ) -> None:
        """Record one vectorised step."""

        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.final_values[i] = final_value
        self.ptr += 1

    def compute_gae(self, last_value: np.ndarray) -> None:
        """Fill ``advantages`` and ``returns`` with GAE(lambda).

        ``last_value`` is V(s_T) for the observation following the final stored
        step, used to bootstrap the tail of an unfinished episode.
        """

        adv = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.ptr)):
            done = np.maximum(self.terminated[t], self.truncated[t])
            value_after = last_value if t == self.ptr - 1 else self.values[t + 1]

            # Bootstrap value of the state reached by this step: zero if the
            # episode truly ended, the stored pre-reset value if it was cut off
            # by the time limit, else the next step's value.
            next_value = np.where(
                self.terminated[t] > 0,
                0.0,
                np.where(self.truncated[t] > 0, self.final_values[t], value_after),
            )
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            # GAE does not flow across an episode boundary.
            adv = delta + self.gamma * self.gae_lambda * (1.0 - done) * adv
            self.advantages[t] = adv
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def batches(self, batch_size: int, normalize_adv: bool = True):
        """Yield shuffled minibatches over the flattened rollout."""

        n = self.ptr * self.n_envs
        flat = lambda x: torch.as_tensor(  # noqa: E731
            x[: self.ptr].reshape(n, *x.shape[2:]), device=self.device
        )
        obs, actions = flat(self.obs), flat(self.actions)
        log_probs, advantages = flat(self.log_probs), flat(self.advantages)
        returns = flat(self.returns)

        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for idx in torch.randperm(n, device=self.device).split(batch_size):
            yield Batch(
                obs=obs[idx],
                actions=actions[idx],
                log_probs=log_probs[idx],
                advantages=advantages[idx],
                returns=returns[idx],
            )


def group_relative_advantages(
    returns: np.ndarray, group_ids: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """GRPO's advantage: standardise episodic returns *within* each group.

        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    This is the critic's replacement. Because every rollout in a group starts
    from an identical state, ``mean(R_group)`` is an unbiased estimate of that
    state's value -- which is exactly what a critic would have supplied, only
    computed from samples instead of learned.

    ``std`` uses the population (biased, ddof=0) convention, matching the GRPO
    literature. A degenerate group -- all returns identical, so std 0 -- yields
    all-zero advantages, contributing no gradient rather than exploding.
    """

    returns = np.asarray(returns, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    advantages = np.zeros_like(returns)

    for gid in np.unique(group_ids):
        mask = group_ids == gid
        group = returns[mask]
        advantages[mask] = (group - group.mean()) / (group.std() + eps)
    return advantages.astype(np.float32)


class GroupBuffer:
    """Whole-episode storage for GRPO, tagged by group id.

    Each episode contributes one scalar advantage, broadcast to all of its
    timesteps. There is no value target and no bootstrapping anywhere.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.clear()

    def clear(self) -> None:
        self._obs: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._log_probs: list[np.ndarray] = []
        self._returns: list[float] = []
        self._group_ids: list[int] = []
        self._lengths: list[int] = []

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        episodic_return: float,
        group_id: int,
    ) -> None:
        """Record one complete rollout and the return it earned."""

        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(np.asarray(actions, dtype=np.float32))
        self._log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self._returns.append(float(episodic_return))
        self._group_ids.append(int(group_id))
        self._lengths.append(len(obs))

    @property
    def n_episodes(self) -> int:
        return len(self._returns)

    @property
    def n_transitions(self) -> int:
        return int(sum(self._lengths))

    @property
    def returns(self) -> np.ndarray:
        return np.asarray(self._returns, dtype=np.float64)

    @property
    def group_ids(self) -> np.ndarray:
        return np.asarray(self._group_ids, dtype=np.int64)

    def compute_advantages(self, eps: float = 1e-8) -> np.ndarray:
        """Per-episode group-relative advantages (one scalar per episode)."""

        return group_relative_advantages(self.returns, self.group_ids, eps=eps)

    def batches(self, batch_size: int, eps: float = 1e-8):
        """Yield shuffled minibatches with advantages already broadcast."""

        episode_adv = self.compute_advantages(eps=eps)
        obs = torch.as_tensor(np.concatenate(self._obs), device=self.device)
        actions = torch.as_tensor(np.concatenate(self._actions), device=self.device)
        log_probs = torch.as_tensor(np.concatenate(self._log_probs), device=self.device)
        advantages = torch.as_tensor(
            np.repeat(episode_adv, self._lengths), device=self.device
        )

        n = obs.shape[0]
        for idx in torch.randperm(n, device=self.device).split(batch_size):
            yield Batch(
                obs=obs[idx],
                actions=actions[idx],
                log_probs=log_probs[idx],
                advantages=advantages[idx],
                returns=None,
            )
