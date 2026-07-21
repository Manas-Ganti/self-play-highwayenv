"""Training entry point for both algorithms.

    python scripts/train.py --config configs/ppo_seed0.yaml

The two algorithms run through the *same* loop: same env, same budget accounting,
same evaluation cadence, same logging. Only the collector, the buffer and the
update differ. Keeping the harness common is what makes "matched compute" mean
something -- env steps are counted identically for both, so neither can win by
quietly stepping the environment more.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algos.common.collectors import GroupCollector, VecCollector  # noqa: E402
from algos.common.config import RunConfig  # noqa: E402
from algos.common.logger import Logger  # noqa: E402
from algos.common.utils import resolve_device, set_seed  # noqa: E402
from algos.grpo import GRPO  # noqa: E402
from algos.ppo import PPO  # noqa: E402
from envs import make_vec_env  # noqa: E402
from eval.solo import evaluate_solo  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("racing-grpo")


def build(cfg: RunConfig):
    """Construct (algorithm, collector, vec_env) for the configured algorithm."""

    device = resolve_device(cfg.device)
    env = make_vec_env(cfg.env, n_envs=cfg.n_envs, asynchronous=cfg.n_envs > 1)
    obs_dim = int(env.single_observation_space.shape[0])
    act_dim = int(env.single_action_space.shape[0])

    if cfg.algo == "ppo":
        algo = PPO(obs_dim, act_dim, cfg.ppo, device, n_envs=cfg.n_envs)
        collector = VecCollector(env, device, seed=cfg.seed)
    else:
        algo = GRPO(obs_dim, act_dim, cfg.grpo, device, n_envs=cfg.n_envs)
        collector = GroupCollector(
            env, device, group_size=cfg.grpo.group_size, seed=cfg.seed
        )
    logger.info(
        "%s | device=%s obs_dim=%d act_dim=%d n_envs=%d", cfg.algo, device, obs_dim, act_dim, cfg.n_envs
    )
    return algo, collector, env, device


def train(cfg: RunConfig) -> Path:
    """Run one training job to completion. Returns the run directory."""

    set_seed(cfg.seed)
    algo, collector, env, device = build(cfg)

    run_logger = Logger(
        run_name=cfg.name,
        config=cfg.model_dump(mode="json"),
        log_dir=cfg.results_dir,
        use_wandb=cfg.use_wandb,
        group=cfg.group or cfg.algo,
    )
    run_dir = run_logger.log_dir
    (run_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))

    next_eval = cfg.eval_every
    next_ckpt = cfg.checkpoint_every
    updates = 0
    start = time.time()

    while collector.total_steps < cfg.total_steps:
        if cfg.algo == "ppo":
            stats = algo.collect(collector)
        else:
            stats = algo.collect(collector, max_steps=cfg.env.max_steps + 1)
        metrics = algo.update()

        steps = collector.total_steps
        updates += 1
        elapsed = time.time() - start
        metrics.update(stats.summary())
        metrics["time/steps_per_sec"] = steps / max(elapsed, 1e-8)
        metrics["time/updates"] = updates
        run_logger.log(metrics, step=steps)

        if steps >= next_eval:
            result = evaluate_solo(
                algo.model, cfg.env, device, n_episodes=cfg.eval_episodes
            )
            run_logger.log(result.as_dict(), step=steps)
            logger.info(
                "[%s] %d steps | return %.1f | lap rate %.2f | collisions %.2f",
                cfg.name,
                steps,
                result.mean_return,
                result.lap_completion_rate,
                result.collision_rate,
            )
            next_eval += cfg.eval_every

        if steps >= next_ckpt:
            algo.save(run_dir / f"checkpoint_{steps}.pt")
            next_ckpt += cfg.checkpoint_every

    algo.save(run_dir / "final.pt")

    # Final solo eval: the Phase 4 gate artefact and the RQ2 degradation baseline.
    final = evaluate_solo(algo.model, cfg.env, device, n_episodes=50)
    run_logger.log(final.as_dict("final_eval"), step=collector.total_steps)
    (run_dir / "solo_eval.json").write_text(json.dumps(final.as_dict("final_eval"), indent=2))
    logger.info("[%s] done in %.1f min -> %s", cfg.name, (time.time() - start) / 60, run_dir)

    run_logger.close()
    env.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, help="override the config's seed")
    parser.add_argument("--total-steps", type=int, help="override the step budget")
    parser.add_argument("--name", type=str, help="override the run name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = RunConfig.load(args.config)
    updates: dict = {}
    if args.seed is not None:
        updates["seed"] = args.seed
        updates["name"] = args.name or f"{cfg.algo}_seed{args.seed}"
        updates["env"] = cfg.env.model_copy(update={"seed": args.seed})
    if args.total_steps is not None:
        updates["total_steps"] = args.total_steps
    if args.name is not None:
        updates["name"] = args.name
    if args.no_wandb:
        updates["use_wandb"] = False
    if updates:
        cfg = cfg.model_copy(update=updates)

    train(cfg)


if __name__ == "__main__":
    main()
