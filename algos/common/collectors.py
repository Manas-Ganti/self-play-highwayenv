"""Rollout collectors.

:class:`VecCollector` -- fixed-horizon collection across a vectorised env. PPO's.

:class:`GroupCollector` -- GRPO's, and the fiddliest engineering in the project.
It launches G rollouts from an *identical reset state* (same track, same traffic
init, same agent pose), so that they differ only by policy stochasticity. That is
what licenses using the group mean as a value baseline: the rollouts share a
state, so they share a true value. If the resets silently diverged -- different
traffic seeds, say -- the group mean would be an average over *different* states,
the baseline would be biased, and GRPO would be quietly broken in a way no
training curve would reveal. Hence ``tests/test_group_collector.py`` asserts
identical initial observations within a group and divergence after the first
stochastic action.

Both collectors report per-episode statistics for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from algos.common.buffer import GroupBuffer, RolloutBuffer
from algos.common.nets import ActorCritic


@dataclass
class EpisodeStats:
    """Aggregated statistics over the episodes finished during one collection."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    crashes: list[bool] = field(default_factory=list)
    off_roads: list[bool] = field(default_factory=list)
    finished: list[bool] = field(default_factory=list)

    def summary(self, prefix: str = "rollout") -> dict[str, float]:
        if not self.returns:
            return {}
        return {
            f"{prefix}/episodic_return": float(np.mean(self.returns)),
            f"{prefix}/episodic_return_std": float(np.std(self.returns)),
            f"{prefix}/episode_length": float(np.mean(self.lengths)),
            f"{prefix}/distance": float(np.mean(self.distances)),
            f"{prefix}/collision_rate": float(np.mean(self.crashes)),
            f"{prefix}/offtrack_rate": float(np.mean(self.off_roads)),
            f"{prefix}/lap_completion_rate": float(np.mean(self.finished)),
            f"{prefix}/n_episodes": float(len(self.returns)),
        }


def _episode_end_info(final_info: dict, i: int) -> dict:
    """Pull env ``i``'s terminal info out of the vector env's masked dict.

    The vector env stacks each scalar info key into an array and adds a ``_key``
    mask. highway-env also emits a nested ``rewards`` dict, which stacks into a
    dict rather than an array -- skip those; the episode stats only need scalars.
    """

    out = {}
    for key, value in final_info.items():
        if key.startswith("_") or isinstance(value, dict):
            continue
        if isinstance(value, (np.ndarray, list, tuple)) and i < len(value):
            out[key] = value[i]
    return out


class VecCollector:
    """Fixed-horizon on-policy collection over a vectorised env. Used by PPO."""

    def __init__(self, env: gym.vector.VectorEnv, device: torch.device, seed: int = 0) -> None:
        self.env = env
        self.device = device
        self.n_envs = env.num_envs
        # Distinct seeds per worker: identical resets across the stack would make
        # 16 envs collect 1 env's worth of information.
        self._obs, _ = env.reset(seed=[seed + i for i in range(self.n_envs)])
        self._ep_return = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_length = np.zeros(self.n_envs, dtype=np.int64)
        self.total_steps = 0

    @torch.no_grad()
    def collect(self, model: ActorCritic, buffer: RolloutBuffer) -> EpisodeStats:
        """Fill ``buffer`` with ``buffer.n_steps`` vectorised steps."""

        buffer.reset()
        stats = EpisodeStats()

        for _ in range(buffer.n_steps):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            action, log_prob, value = model.act(obs_t)
            action_np = action.cpu().numpy()

            next_obs, reward, terminated, truncated, infos = self.env.step(action_np)
            self.total_steps += self.n_envs

            # Truncated envs need V(final_obs) to bootstrap; SAME_STEP autoreset
            # has already replaced next_obs with the *reset* observation.
            final_value = np.zeros(self.n_envs, dtype=np.float32)
            if truncated.any():
                final_obs = np.stack(
                    [
                        infos["final_obs"][i] if truncated[i] else self._obs[i]
                        for i in range(self.n_envs)
                    ]
                ).astype(np.float32)
                v = model.critic(torch.as_tensor(final_obs, device=self.device))
                final_value = np.where(truncated, v.cpu().numpy(), 0.0).astype(np.float32)

            buffer.add(
                obs=self._obs,
                action=action_np,
                log_prob=log_prob.cpu().numpy(),
                reward=reward,
                value=value.cpu().numpy(),
                terminated=terminated.astype(np.float32),
                truncated=truncated.astype(np.float32),
                final_value=final_value,
            )

            self._ep_return += reward
            self._ep_length += 1
            done = np.logical_or(terminated, truncated)
            for i in np.flatnonzero(done):
                info_i = _episode_end_info(infos.get("final_info", {}), i)
                stats.returns.append(float(self._ep_return[i]))
                stats.lengths.append(int(self._ep_length[i]))
                stats.distances.append(float(info_i.get("distance", 0.0)))
                stats.crashes.append(bool(info_i.get("crashed", False)))
                stats.off_roads.append(bool(info_i.get("off_road", False)))
                stats.finished.append(bool(info_i.get("finished", False)))
                self._ep_return[i] = 0.0
                self._ep_length[i] = 0

            self._obs = next_obs

        last_value = model.critic(
            torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
        )
        buffer.compute_gae(last_value.cpu().numpy())
        return stats


class GroupCollector:
    """Collect ``n_groups`` x ``group_size`` episodes; each group shares a reset state.

    Envs are laid out group-major: env index ``g * G + k`` is member ``k`` of
    group ``g``. All G members of a group are reset with the same seed, so they
    begin from a byte-identical state and diverge only through action sampling.

    Episodes within a group finish at different times. Rather than complicate the
    stepping loop, finished envs keep stepping (the vector env has auto-reset
    them) and their transitions are simply not recorded -- ``active`` masks them
    out. This wastes some CPU on the tail of a batch, which is the honest price of
    a collector that is easy to verify; correctness here is worth more than the
    last few percent of throughput.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        device: torch.device,
        group_size: int,
        seed: int = 0,
    ) -> None:
        if env.num_envs % group_size != 0:
            raise ValueError(
                f"n_envs ({env.num_envs}) must be a multiple of group_size ({group_size})"
            )
        self.env = env
        self.device = device
        self.group_size = group_size
        self.n_groups = env.num_envs // group_size
        self.n_envs = env.num_envs
        self._rng = np.random.default_rng(seed)
        self.total_steps = 0

    def _group_seeds(self) -> list[int]:
        """One fresh seed per group, repeated across that group's G members."""

        seeds = self._rng.integers(0, 2**31 - 1, size=self.n_groups)
        return [int(s) for s in np.repeat(seeds, self.group_size)]

    @torch.no_grad()
    def collect(
        self, model: ActorCritic, buffer: GroupBuffer, max_steps: int = 1000
    ) -> EpisodeStats:
        """Run one full batch of grouped episodes into ``buffer``."""

        buffer.clear()
        stats = EpisodeStats()

        obs, _ = self.env.reset(seed=self._group_seeds())
        active = np.ones(self.n_envs, dtype=bool)

        ep_obs: list[list[np.ndarray]] = [[] for _ in range(self.n_envs)]
        ep_actions: list[list[np.ndarray]] = [[] for _ in range(self.n_envs)]
        ep_log_probs: list[list[float]] = [[] for _ in range(self.n_envs)]
        ep_return = np.zeros(self.n_envs, dtype=np.float64)
        ep_length = np.zeros(self.n_envs, dtype=np.int64)

        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob, _ = model.act(obs_t)
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()

            next_obs, reward, terminated, truncated, infos = self.env.step(action_np)
            self.total_steps += int(active.sum())

            for i in np.flatnonzero(active):
                ep_obs[i].append(obs[i].copy())
                ep_actions[i].append(action_np[i].copy())
                ep_log_probs[i].append(float(log_prob_np[i]))
                ep_return[i] += reward[i]
                ep_length[i] += 1

            done = np.logical_or(terminated, truncated)
            for i in np.flatnonzero(done & active):
                info_i = _episode_end_info(infos.get("final_info", {}), i)
                buffer.add_episode(
                    obs=np.asarray(ep_obs[i], dtype=np.float32),
                    actions=np.asarray(ep_actions[i], dtype=np.float32),
                    log_probs=np.asarray(ep_log_probs[i], dtype=np.float32),
                    episodic_return=float(ep_return[i]),
                    group_id=i // self.group_size,
                )
                stats.returns.append(float(ep_return[i]))
                stats.lengths.append(int(ep_length[i]))
                stats.distances.append(float(info_i.get("distance", 0.0)))
                stats.crashes.append(bool(info_i.get("crashed", False)))
                stats.off_roads.append(bool(info_i.get("off_road", False)))
                stats.finished.append(bool(info_i.get("finished", False)))
                active[i] = False

            obs = next_obs
            if not active.any():
                break

        # An env still active at max_steps hit the collector's cap rather than the
        # env's own time limit; its return is a partial one. The env's max_steps is
        # the binding limit in practice, so this is a safety valve, not a code path
        # we expect to take.
        for i in np.flatnonzero(active):
            if ep_length[i] > 0:
                buffer.add_episode(
                    obs=np.asarray(ep_obs[i], dtype=np.float32),
                    actions=np.asarray(ep_actions[i], dtype=np.float32),
                    log_probs=np.asarray(ep_log_probs[i], dtype=np.float32),
                    episodic_return=float(ep_return[i]),
                    group_id=i // self.group_size,
                )
                stats.returns.append(float(ep_return[i]))
                stats.lengths.append(int(ep_length[i]))
                stats.distances.append(0.0)
                stats.crashes.append(False)
                stats.off_roads.append(False)
                stats.finished.append(False)

        return stats
