"""Rollout evaluation: win rate, ELO, trajectory logging.

Pure evaluation — loads a trained model, rolls it out against baselines on a
specified environment variant, and writes metrics/trajectories. No training,
no weight updates. CLI matches the command in CLAUDE.md::

    python experiments/evaluate.py \
        --model results/sac_sparse_base/final.zip \
        --env-config '{"vehicle_drag": 0.7, "track_width": 8}' \
        --episodes 50
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.baselines import RandomPolicy, RuleBasedPolicy
from envs.env_factory import make_env
from envs.racetrack_env import FEATURE_ORDER

logger = logging.getLogger("racetrack_rl")

# Index of opponent_x within the observation vector. opponent_x < 0 means the
# opponent is behind us (we lead); a sign flip in our favour is an overtake.
FEATURE_LEAD_IDX = FEATURE_ORDER.index("opponent_x")


@dataclass
class EpisodeResult:
    """Per-episode evaluation record."""

    won: bool
    crashed: bool
    steps: int
    lap_progress: float
    overtakes: int
    trajectory: list[list[float]] = field(default_factory=list)


@dataclass
class EvalMetrics:
    """Aggregated metrics over an evaluation run, against one opponent."""

    opponent: str
    episodes: int
    win_rate: float
    crash_rate: float
    mean_steps: float
    mean_lap_progress: float
    mean_overtakes: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_policy(spec: str, n_actions: int, seed: int):
    """Resolve an opponent spec to a policy with a ``predict`` method.

    ``spec`` is one of ``"random"``, ``"rule_based"``, or a path to a saved
    SB3 model.
    """

    if spec == "random":
        return RandomPolicy(n_actions=n_actions, seed=seed)
    if spec == "rule_based":
        return RuleBasedPolicy()
    # Otherwise treat as a saved model path; PPO/SAC share the load API.
    from stable_baselines3 import PPO

    return PPO.load(spec, device="cpu")


def _count_overtake(prev_lead: float, lead: float) -> bool:
    """Detect a lead sign-flip in our favour (an overtake)."""

    return prev_lead <= 0.0 < lead


def evaluate(
    model_path: str | Path,
    env_overrides: dict[str, Any],
    episodes: int = 50,
    opponents: list[str] | None = None,
    record_trajectories: bool = True,
    seed: int = 0,
) -> list[EvalMetrics]:
    """Evaluate a trained model against a set of opponents on one env variant.

    Parameters
    ----------
    model_path : str or Path
        Saved SB3 model to evaluate (controls ``agent_0``).
    env_overrides : dict
        Environment parameter overrides merged onto defaults (e.g.
        ``{"vehicle_drag": 0.7, "track_width": 8}``).
    episodes : int
        Episodes per opponent.
    opponents : list of str, optional
        Opponent specs (baseline names or model paths). Defaults to
        ``["random", "rule_based"]``.
    record_trajectories : bool
        Whether to store per-step positions for trajectory plots.
    seed : int
        Base seed; each episode uses ``seed + episode_index``.

    Returns
    -------
    list of EvalMetrics
        One aggregated record per opponent.
    """

    from stable_baselines3 import PPO

    opponents = opponents or ["random", "rule_based"]
    learner = PPO.load(str(model_path), device="cpu")

    all_metrics: list[EvalMetrics] = []
    for opp_spec in opponents:
        results: list[EpisodeResult] = []
        for ep in range(episodes):
            env = make_env({**env_overrides, "seed": seed + ep})
            n_actions = env.action_space("agent_0").n
            opponent = _load_policy(opp_spec, n_actions, seed + ep)

            obs, _ = env.reset(seed=seed + ep)
            done = False
            steps, overtakes = 0, 0
            prev_lead = 0.0
            traj: list[list[float]] = []
            info_a: dict[str, Any] = {}
            while not done:
                a_action, _ = learner.predict(obs["agent_0"], deterministic=True)
                o_action, _ = opponent.predict(obs["agent_1"], deterministic=True)
                obs, _, term, trunc, infos = env.step(
                    {"agent_0": a_action, "agent_1": o_action}
                )
                info_a = infos["agent_0"]
                done = term["agent_0"] or trunc["agent_0"]
                steps += 1
                if obs.get("agent_0") is not None:
                    feat = obs["agent_0"]
                    lead = float(-feat[FEATURE_LEAD_IDX])  # opponent_x: <0 => we lead
                    if _count_overtake(prev_lead, lead):
                        overtakes += 1
                    prev_lead = lead
                    if record_trajectories:
                        traj.append([float(feat[0]), float(feat[1])])
            results.append(
                EpisodeResult(
                    won=bool(info_a.get("won", False)),
                    crashed=bool(info_a.get("crashed", False)),
                    steps=steps,
                    lap_progress=float(info_a.get("lap_progress", 0.0)),
                    overtakes=overtakes,
                    trajectory=traj,
                )
            )
            env.close()

        metrics = _aggregate(opp_spec, results)
        logger.info(
            "vs %-12s win_rate=%.3f crash_rate=%.3f",
            opp_spec,
            metrics.win_rate,
            metrics.crash_rate,
        )
        all_metrics.append(metrics)
    return all_metrics


def _aggregate(opponent: str, results: list[EpisodeResult]) -> EvalMetrics:
    n = len(results)
    return EvalMetrics(
        opponent=opponent,
        episodes=n,
        win_rate=float(np.mean([r.won for r in results])) if n else 0.0,
        crash_rate=float(np.mean([r.crashed for r in results])) if n else 0.0,
        mean_steps=float(np.mean([r.steps for r in results])) if n else 0.0,
        mean_lap_progress=float(np.mean([r.lap_progress for r in results])) if n else 0.0,
        mean_overtakes=float(np.mean([r.overtakes for r in results])) if n else 0.0,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained racetrack model.")
    p.add_argument("--model", required=True, help="Path to saved SB3 model.")
    p.add_argument(
        "--env-config",
        default="{}",
        help='JSON env overrides, e.g. \'{"vehicle_drag": 0.7, "track_width": 8}\'',
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--opponents", nargs="*", default=["random", "rule_based"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    env_overrides = json.loads(args.env_config)
    metrics = evaluate(
        model_path=args.model,
        env_overrides=env_overrides,
        episodes=args.episodes,
        opponents=args.opponents,
        seed=args.seed,
    )
    payload = [m.as_dict() for m in metrics]
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote metrics to %s", args.out)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
