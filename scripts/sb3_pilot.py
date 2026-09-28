"""Phase 1 -- env validation via an SB3 PPO pilot.

    python scripts/sb3_pilot.py                      # track A, 2M steps, seed 0
    python scripts/sb3_pilot.py --total-steps 200000 --name pilot_smoke
    python scripts/sb3_pilot.py --obs-type occupancy   # road-aware observation (§3 allows either)

The question this answers is about the *environment*, not the algorithm: can an
off-the-shelf, known-correct PPO learn to lap track A in traffic? If it cannot,
nothing later means anything -- a GRPO "failure" on an unlearnable env is not a
finding. SB3 is used here and in the Phase 2 parity check only (CLAUDE.md §4.1).

Gate (CLAUDE.md §5): >=90% of deterministic eval episodes complete a lap without
collision within 2M steps. An episode terminates on the first crash, off-track
or finish, so ``lap_completion_rate`` is exactly "lapped without colliding".

Scored by the same ``evaluate_solo`` harness, on the same fixed eval seeds, as
the custom PPO and GRPO runs -- so the Phase 2 comparison is like-for-like.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from algos.common.config import RunConfig  # noqa: E402
from algos.common.logger import Logger  # noqa: E402
from algos.common.utils import resolve_device, set_seed  # noqa: E402
from envs import solo_env_fn  # noqa: E402
from envs.config import ObsType  # noqa: E402
from eval.solo import SoloEvalResult, evaluate_solo  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("racing-grpo")

GATE_LAP_RATE = 0.90


class SB3Actor:
    """Adapts an SB3 policy to the ``model.act`` interface ``evaluate_solo`` expects."""

    def __init__(self, model: PPO) -> None:
        self.model = model

    def act(self, obs: torch.Tensor, deterministic: bool = True):
        action, _ = self.model.predict(obs.cpu().numpy(), deterministic=deterministic)
        return torch.as_tensor(action), None, None


class EvalCallback(BaseCallback):
    """Periodic solo eval through the project harness, logged on env steps."""

    def __init__(self, cfg: RunConfig, run_logger: Logger) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_logger = run_logger
        self.next_eval = cfg.eval_every
        self.start = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            result = evaluate_solo(
                SB3Actor(self.model), self.cfg.env, torch.device("cpu"), self.cfg.eval_episodes
            )
            metrics = result.as_dict()
            metrics["time/steps_per_sec"] = self.num_timesteps / max(time.time() - self.start, 1e-8)
            self.run_logger.log(metrics, step=self.num_timesteps)
            logger.info(
                "[pilot] %d steps | return %.1f | lap rate %.2f | collisions %.2f | off-track %.2f",
                self.num_timesteps,
                result.mean_return,
                result.lap_completion_rate,
                result.collision_rate,
                result.offtrack_rate,
            )
            self.next_eval += self.cfg.eval_every
        return True


def build_sb3(cfg: RunConfig, env, device: torch.device, tb_dir: Path) -> PPO:
    """SB3 PPO with the custom PPO's hyperparameters, mapped one-to-one."""

    hp = cfg.ppo
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=hp.learning_rate,
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        n_epochs=hp.n_epochs,
        gamma=hp.gamma,
        gae_lambda=hp.gae_lambda,
        clip_range=hp.clip_range,
        ent_coef=hp.entropy_coef,
        vf_coef=hp.vf_coef,
        max_grad_norm=hp.max_grad_norm,
        normalize_advantage=hp.normalize_advantage,
        policy_kwargs={
            "net_arch": list(hp.hidden_sizes),
            "log_std_init": hp.log_std_init,
            "activation_fn": torch.nn.Tanh,
        },
        tensorboard_log=str(tb_dir),
        seed=cfg.seed,
        device=device,
        verbose=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_seed0.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, help="override the 2M budget (smoke tests)")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--obs-type",
        choices=[o.value for o in ObsType],
        help="override the config's observation family (Phase 1 env iteration only)",
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = RunConfig.load(args.config)
    obs_type = ObsType(args.obs_type) if args.obs_type else cfg.env.obs_type
    tag = "" if obs_type is ObsType.KINEMATICS else f"{obs_type.value}_"
    name = args.name or f"phase1/sb3_ppo_{tag}seed{args.seed}"
    cfg = cfg.model_copy(
        update={
            "name": name,
            "seed": args.seed,
            "env": cfg.env.model_copy(update={"seed": args.seed, "obs_type": obs_type}),
            "total_steps": args.total_steps or cfg.total_steps,
            "group": "phase1_sb3",
            "use_wandb": cfg.use_wandb and not args.no_wandb,
        }
    )
    finished = Path(cfg.results_dir) / cfg.name / "solo_eval.json"
    if finished.exists():
        # Env/reward iteration reuses seeds, so default names collide across pilots.
        # Overwriting would silently destroy the earlier pilot's evidence.
        raise SystemExit(f"{finished} already exists -- pass --name (e.g. phase1/p3_occupancy_seed0)")
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_logger = Logger(
        run_name=cfg.name,
        config={**cfg.model_dump(mode="json"), "implementation": "sb3"},
        log_dir=cfg.results_dir,
        use_wandb=cfg.use_wandb,
        group=cfg.group,
    )
    run_dir = run_logger.log_dir
    (run_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))

    env = SubprocVecEnv([solo_env_fn(cfg.env) for _ in range(cfg.n_envs)])
    model = build_sb3(cfg, env, device, run_dir / "sb3_tb")
    logger.info(
        "SB3 PPO pilot | obs=%s device=%s n_envs=%d steps=%d",
        obs_type.value, device, cfg.n_envs, cfg.total_steps,
    )

    start = time.time()
    model.learn(total_timesteps=cfg.total_steps, callback=EvalCallback(cfg, run_logger))
    model.save(run_dir / "sb3_final")
    env.close()

    final: SoloEvalResult = evaluate_solo(SB3Actor(model), cfg.env, torch.device("cpu"), 50)
    passed = bool(final.lap_completion_rate >= GATE_LAP_RATE)
    verdict = {
        **final.as_dict("final_eval"),
        "gate/threshold": GATE_LAP_RATE,
        "gate/passed": passed,
        "total_steps": int(model.num_timesteps),
        "wall_minutes": (time.time() - start) / 60,
    }
    run_logger.log({k: float(v) for k, v in verdict.items()}, step=int(model.num_timesteps))
    (run_dir / "solo_eval.json").write_text(json.dumps(verdict, indent=2, default=float))
    run_logger.close()

    logger.info(
        "PHASE 1 GATE %s: lap rate %.2f (need >= %.2f), collisions %.2f, off-track %.2f -> %s",
        "PASSED" if passed else "FAILED",
        final.lap_completion_rate,
        GATE_LAP_RATE,
        final.collision_rate,
        final.offtrack_rate,
        run_dir / "solo_eval.json",
    )
    if not np.isfinite(final.mean_return):
        sys.exit("non-finite eval return -- inspect the run before trusting anything")


if __name__ == "__main__":
    main()
