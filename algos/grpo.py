"""GRPO for continuous control -- the core contribution.

Group Relative Policy Optimisation, ported from LLM fine-tuning into dense-reward
continuous control. G rollouts are launched from an *identical* reset state and
differ only by policy stochasticity; the group's mean return serves as the value
baseline a critic would otherwise have learned:

    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

broadcast to every timestep of rollout i, then optimised with PPO's clipped
surrogate plus an entropy bonus.

What is deliberately absent
---------------------------
* **No value network, no value loss.** Critic-freeness is the research question.
* **No KL-to-reference term.** GRPO in LLM fine-tuning regularises toward the
  pre-trained reference policy. There *is* no reference policy in from-scratch
  RL, so the term is meaningless here; clipping is the trust region. This is not
  an oversight -- do not add one back.

Honest positioning (goes in the report): this variant is close to RLOO with
clipping and multi-sample groups, evaluated in dense-reward continuous control.
The claim is empirical, not that the estimator is new.

GRPO-t (``per_timestep_advantages=True``) is the Phase 4 diagnostic variant, built
only if base GRPO misses H1: it standardises discounted returns-to-go across the
group *at each timestep*, which separates "credit assignment over a long episode
failed" from "having no critic failed".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.common.buffer import GroupBuffer
from algos.common.collectors import EpisodeStats, GroupCollector
from algos.common.config import GRPOConfig
from algos.common.nets import ActorCritic
from algos.common.utils import GradientVarianceTracker


class GRPO:
    """Critic-free policy optimisation with group-relative advantages."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: GRPOConfig,
        device: torch.device,
        n_envs: int | None = None,
    ) -> None:
        if config.per_timestep_advantages:
            raise NotImplementedError(
                "GRPO-t is a Phase 4 diagnostic, built only if base GRPO misses H1 "
                "(CLAUDE.md §4). Implement it in algos/grpo_t.py -- not by mutating "
                "this class -- so the two variants stay independently testable."
            )

        self.config = config
        self.device = device
        self.n_envs = n_envs or config.group_size * config.n_groups

        self.model = ActorCritic(
            obs_dim,
            act_dim,
            hidden=tuple(config.hidden_sizes),
            log_std_init=config.log_std_init,
            with_critic=False,  # the entire point
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer = GroupBuffer(device=device)

    def collect(self, collector: GroupCollector, max_steps: int = 1000) -> EpisodeStats:
        return collector.collect(self.model, self.buffer, max_steps=max_steps)

    def update(self) -> dict[str, float]:
        """One GRPO update over the collected groups. Returns diagnostics."""

        cfg = self.config
        if self.buffer.n_episodes == 0:  # pragma: no cover - collector guarantees > 0
            return {}

        grad_tracker = GradientVarianceTracker()
        policy_params = list(self.model.actor.parameters())

        losses, policy_losses, entropies, clip_fractions, kls = [], [], [], [], []

        for _ in range(cfg.n_epochs):
            for batch in self.buffer.batches(cfg.batch_size, eps=cfg.advantage_eps):
                log_prob, entropy, _ = self.model.evaluate(batch.obs, batch.actions)
                ratio = torch.exp(log_prob - batch.log_probs)

                surrogate_1 = ratio * batch.advantages
                surrogate_2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    * batch.advantages
                )
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                entropy_loss = -entropy.mean()

                # No value term. The loss is the surrogate and the entropy bonus.
                loss = policy_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_tracker.update(policy_params)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    losses.append(loss.item())
                    policy_losses.append(policy_loss.item())
                    entropies.append(entropy.mean().item())
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()
                    )
                    kls.append(((ratio - 1.0) - (log_prob - batch.log_probs)).mean().item())

        return {**self._loss_metrics(losses, policy_losses, entropies, clip_fractions, kls),
                **self._group_metrics(),
                "train/grad_variance": grad_tracker.variance,
                "train/grad_norm": grad_tracker.gradient_norm,
                "train/grad_snr": grad_tracker.snr,
                "train/log_std": float(self.model.actor.log_std.mean().item())}

    @staticmethod
    def _loss_metrics(losses, policy_losses, entropies, clip_fractions, kls) -> dict[str, float]:
        return {
            "train/loss": float(np.mean(losses)),
            "train/policy_loss": float(np.mean(policy_losses)),
            "train/entropy": float(np.mean(entropies)),
            "train/clip_fraction": float(np.mean(clip_fractions)),
            "train/approx_kl": float(np.mean(kls)),
        }

    def _group_metrics(self) -> dict[str, float]:
        """Diagnostics specific to the group baseline.

        ``degenerate_group_frac`` is the one to watch: a group whose returns are
        all identical has zero within-group spread, so every advantage in it is
        zero and it contributes *no gradient at all*. If that fraction climbs, the
        policy has collapsed to determinism and GRPO is silently learning from a
        shrinking slice of its data -- a failure mode PPO simply does not have.
        """

        returns = self.buffer.returns
        group_ids = self.buffer.group_ids
        advantages = self.buffer.compute_advantages(eps=self.config.advantage_eps)

        stds = np.array(
            [returns[group_ids == gid].std() for gid in np.unique(group_ids)], dtype=np.float64
        )
        return {
            "grpo/group_return_std": float(stds.mean()),
            "grpo/degenerate_group_frac": float((stds < 1e-6).mean()),
            "grpo/advantage_abs_mean": float(np.abs(advantages).mean()),
            "grpo/n_groups": float(len(stds)),
            "grpo/n_episodes": float(self.buffer.n_episodes),
            "grpo/transitions_per_update": float(self.buffer.n_transitions),
        }

    # -- checkpointing ---------------------------------------------------------
    def save(self, path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.model_dump(),
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
