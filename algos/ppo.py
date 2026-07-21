"""Custom PPO -- the reference implementation every later comparison inherits.

Standard PPO: GAE(lambda), clipped surrogate, clipped-free value loss, entropy
bonus. Nothing exotic. Phase 2 requires this loop to be statistically
indistinguishable from SB3 PPO before GRPO is built, because if this loop is
subtly wrong, the PPO-vs-GRPO comparison measures the bug rather than the
algorithms.

Shares ``algos/common/`` (nets, buffer, collector) with GRPO, so the *only*
differences between the two are the advantage estimator and the value loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algos.common.buffer import RolloutBuffer
from algos.common.collectors import EpisodeStats, VecCollector
from algos.common.config import PPOConfig
from algos.common.nets import ActorCritic
from algos.common.utils import GradientVarianceTracker, explained_variance


class PPO:
    """Proximal Policy Optimisation with a learned value baseline."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: PPOConfig,
        device: torch.device,
        n_envs: int,
    ) -> None:
        self.config = config
        self.device = device
        self.n_envs = n_envs

        self.model = ActorCritic(
            obs_dim,
            act_dim,
            hidden=tuple(config.hidden_sizes),
            log_std_init=config.log_std_init,
            with_critic=True,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer(
            n_steps=config.n_steps,
            n_envs=n_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

    @property
    def steps_per_update(self) -> int:
        return self.config.n_steps * self.n_envs

    def collect(self, collector: VecCollector) -> EpisodeStats:
        return collector.collect(self.model, self.buffer)

    def update(self) -> dict[str, float]:
        """One PPO update over the collected rollout. Returns diagnostics."""

        cfg = self.config
        grad_tracker = GradientVarianceTracker()
        policy_params = list(self.model.actor.parameters())

        losses, policy_losses, value_losses, entropies, clip_fractions, kls = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in range(cfg.n_epochs):
            for batch in self.buffer.batches(
                cfg.batch_size, normalize_adv=cfg.normalize_advantage
            ):
                log_prob, entropy, value = self.model.evaluate(batch.obs, batch.actions)
                ratio = torch.exp(log_prob - batch.log_probs)

                surrogate_1 = ratio * batch.advantages
                surrogate_2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    * batch.advantages
                )
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = nn.functional.mse_loss(value, batch.returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Record the *policy* gradient before clipping: this is the number
                # compared against GRPO's, so it must not be a clipped one.
                grad_tracker.update(policy_params)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    losses.append(loss.item())
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.mean().item())
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()
                    )
                    # Schulman's low-variance KL estimator.
                    kls.append(((ratio - 1.0) - (log_prob - batch.log_probs)).mean().item())

        ev = explained_variance(
            self.buffer.values[: self.buffer.ptr].ravel(),
            self.buffer.returns[: self.buffer.ptr].ravel(),
        )
        return {
            "train/loss": float(np.mean(losses)),
            "train/policy_loss": float(np.mean(policy_losses)),
            "train/value_loss": float(np.mean(value_losses)),
            "train/entropy": float(np.mean(entropies)),
            "train/clip_fraction": float(np.mean(clip_fractions)),
            "train/approx_kl": float(np.mean(kls)),
            "train/explained_variance": ev,
            "train/log_std": float(self.model.actor.log_std.mean().item()),
            "train/grad_variance": grad_tracker.variance,
            "train/grad_norm": grad_tracker.gradient_norm,
            "train/grad_snr": grad_tracker.snr,
            "train/advantage_std": float(self.buffer.advantages[: self.buffer.ptr].std()),
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
