"""Pygame live dashboard for evaluation rollouts (never during training).

    python viz/live_render.py --model results/sac_sparse_base/final.zip

Loads a trained model, rolls it out against a baseline on a rendered env, and
streams the highway-env frame alongside a small telemetry panel (speed, lap
progress, ELO if available). This is a visualisation tool only — it performs no
training and writes no checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agents.baselines import RuleBasedPolicy
from envs.env_factory import make_env

logger = logging.getLogger("racetrack_rl")


def run_dashboard(
    model_path: str | Path,
    env_overrides: dict | None = None,
    opponent: str = "rule_based",
    episodes: int = 5,
    seed: int = 0,
) -> None:
    """Roll out a model with live rendering.

    Parameters
    ----------
    model_path : str or Path
        Saved SB3 model controlling ``agent_0``.
    env_overrides : dict, optional
        Environment parameter overrides.
    opponent : str
        Baseline opponent name.
    episodes : int
        Number of episodes to display.
    seed : int
        Base RNG seed.
    """

    from stable_baselines3 import PPO

    env_overrides = dict(env_overrides or {})
    env_overrides["render_mode"] = "human"  # honoured by the factory/env if set
    learner = PPO.load(str(model_path), device="cpu")
    opp = RuleBasedPolicy()

    for ep in range(episodes):
        env = make_env({**env_overrides, "seed": seed + ep})
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a, _ = learner.predict(obs["agent_0"], deterministic=True)
            o, _ = opp.predict(obs["agent_1"], deterministic=True)
            obs, rewards, term, trunc, infos = env.step({"agent_0": a, "agent_1": o})
            env.render()
            done = term["agent_0"] or trunc["agent_0"]
        logger.info("episode %d finished (won=%s)", ep, infos["agent_0"].get("won"))
        env.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Live visual eval dashboard.")
    p.add_argument("--model", required=True)
    p.add_argument("--env-config", default="{}")
    p.add_argument("--opponent", default="rule_based")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_dashboard(
        model_path=args.model,
        env_overrides=json.loads(args.env_config),
        opponent=args.opponent,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
