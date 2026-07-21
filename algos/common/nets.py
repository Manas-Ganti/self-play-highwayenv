"""Policy and value networks.

Shared verbatim by PPO and GRPO. The *only* difference between the two
algorithms' models is that GRPO does not build a critic -- that absence is the
research question, so it must not leak into the architecture anywhere else.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """Tanh MLP with orthogonal init; small final-layer gain for a calm start."""

    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [_layer(last, h, gain=np.sqrt(2)), nn.Tanh()]
        last = h
    layers.append(_layer(last, out_dim, gain=0.01))
    return nn.Sequential(*layers)


def _layer(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy over continuous ``[throttle, steering]`` actions.

    ``log_std`` is a state-independent parameter -- the standard choice for
    continuous-control PPO. It matters more than usual here: it is the sole
    source of the within-group diversity GRPO's advantage estimator needs, so an
    entropy collapse degrades GRPO's *learning signal*, not just its exploration.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.mean_net = mlp(obs_dim, hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, entropy)``, summed over action dimensions."""

        dist = self.distribution(obs)
        if action is None:
            action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy


class Critic(nn.Module):
    """State-value network. PPO only -- GRPO never constructs one."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor plus an optional critic.

    ``with_critic=False`` yields the GRPO model: no value head, no value
    parameters in the optimiser, nothing to bootstrap from.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
        with_critic: bool = True,
    ) -> None:
        super().__init__()
        self.actor = GaussianActor(obs_dim, act_dim, hidden, log_std_init)
        self.critic = Critic(obs_dim, hidden) if with_critic else None

    @property
    def has_critic(self) -> bool:
        return self.critic is not None

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample an action for rollout collection: ``(action, log_prob, value)``."""

        action, log_prob, _ = self.actor(obs, deterministic=deterministic)
        value = self.critic(obs) if self.critic is not None else None
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Re-evaluate stored actions under the current policy (the PPO ratio)."""

        _, log_prob, entropy = self.actor(obs, action=action)
        value = self.critic(obs) if self.critic is not None else None
        return log_prob, entropy, value
