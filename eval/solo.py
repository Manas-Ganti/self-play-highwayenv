"""Solo evaluation suite.

Produces each agent's *in-distribution* baseline: the numbers the head-to-head
degradation in RQ2 is measured against. Run with a deterministic policy (the
distribution's mean action) over a fixed set of episode seeds, so PPO and GRPO
are scored on exactly the same episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from algos.common.nets import ActorCritic
from envs import make_solo_env
from envs.config import EnvConfig


@dataclass
class SoloEvalResult:
    """Per-condition summary of a solo evaluation."""

    n_episodes: int
    mean_return: float
    mean_distance: float
    lap_completion_rate: float
    collision_rate: float
    offtrack_rate: float
    mean_lap_time: float  # seconds, over episodes that completed a lap; nan if none
    mean_speed: float

    def as_dict(self, prefix: str = "eval") -> dict[str, float]:
        return {f"{prefix}/{k}": v for k, v in asdict(self).items()}


@torch.no_grad()
def evaluate_solo(
    model: ActorCritic,
    env_config: EnvConfig,
    device: torch.device,
    n_episodes: int = 50,
    seed_offset: int = 10_000,
    deterministic: bool = True,
) -> SoloEvalResult:
    """Evaluate ``model`` on ``n_episodes`` fixed-seed solo episodes.

    Episode seeds are ``seed_offset + i``, disjoint from training seeds and
    identical across algorithms and training seeds -- every agent is scored on the
    same traffic.
    """

    env = make_solo_env(env_config)
    dt = 1.0 / env_config.policy_frequency

    returns, distances, laps, crashes, offtracks, lap_times, speeds = [], [], [], [], [], [], []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        ep_return, ep_steps, ep_speeds = 0.0, 0, []
        info: dict = {}

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = model.act(obs_t, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            ep_return += float(reward)
            ep_steps += 1
            ep_speeds.append(float(info.get("speed", 0.0)))
            done = bool(terminated or truncated)

        returns.append(ep_return)
        distances.append(float(info.get("distance", 0.0)))
        finished = bool(info.get("finished", False))
        laps.append(finished)
        crashes.append(bool(info.get("crashed", False)))
        offtracks.append(bool(info.get("off_road", False)))
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        if finished:
            lap_times.append(ep_steps * dt / env_config.laps_to_finish)

    env.close()
    return SoloEvalResult(
        n_episodes=n_episodes,
        mean_return=float(np.mean(returns)),
        mean_distance=float(np.mean(distances)),
        lap_completion_rate=float(np.mean(laps)),
        collision_rate=float(np.mean(crashes)),
        offtrack_rate=float(np.mean(offtracks)),
        mean_lap_time=float(np.mean(lap_times)) if lap_times else float("nan"),
        mean_speed=float(np.mean(speeds)),
    )
