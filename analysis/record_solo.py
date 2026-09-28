"""Record solo episodes of a trained run, to *watch* what the policy does.

    python analysis/record_solo.py --run results/phase1/sb3_ppo_seed0            # 3 eval episodes
    python analysis/record_solo.py --run results/ppo_seed0 --n 5 --wandb        # + upload to W&B
    python analysis/record_solo.py --run results/phase1/sb3_ppo_seed0 --policy straight

Replays the same fixed eval seeds ``evaluate_solo`` scores (``10_000 + i``), so the
videos are the episodes behind the logged numbers -- not a fresh, luckier draw.
The env is rebuilt from the run's own ``config.json``, so a run trained on the
occupancy observation is replayed on the occupancy observation.

Works for SB3 pilot runs (``sb3_final.zip``) and the custom PPO/GRPO loop
(``final.pt``). ``--policy straight`` replays a scripted zero-steer, fixed-throttle
controller on the same seeds: the reference that diagnosed pilot #1.

Each frame is stamped with step, speed, lap distance and (on the last frame) the
outcome, and the file name carries the outcome too, so a folder listing already
reads as a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.video import write_video  # noqa: E402
from envs import make_solo_env  # noqa: E402
from envs.config import EnvConfig, RewardConfig  # noqa: E402

EVAL_SEED_OFFSET = 10_000  # must match eval/solo.py::evaluate_solo


def load_env_config(run: Path) -> EnvConfig:
    """The env exactly as the run trained on it (observation family included)."""

    path = run / "config.json"
    if not path.exists():
        print(f"  {path} missing; falling back to the default env config")
        return EnvConfig(reward=RewardConfig.load())
    return EnvConfig.model_validate(json.loads(path.read_text())["env"])


def load_policy(run: Path, policy: str, env):
    """Return ``obs -> action`` for the requested policy."""

    if policy == "straight":
        # [throttle, steering]: gentle throttle, no steering (pilot #1's reference).
        return lambda obs: np.array([0.3, 0.0], dtype=np.float32)

    if (run / "sb3_final.zip").exists():
        from stable_baselines3 import PPO

        model = PPO.load(run / "sb3_final.zip", device="cpu")
        return lambda obs: model.predict(obs, deterministic=True)[0]

    if (run / "final.pt").exists():
        from eval.policies import TorchPolicy

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(env.action_space.shape[0])
        pol = TorchPolicy.load(run / "final.pt", obs_dim, act_dim, torch.device("cpu"))
        return pol.action

    raise SystemExit(f"no sb3_final.zip or final.pt in {run} (pull with --with-ckpt?)")


def stamp(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a small text overlay in the top-left corner."""

    # Pillow, not OpenCV: it is always present (matplotlib depends on it).
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((10, 10 + 18 * i), text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def outcome(info: dict) -> str:
    if info.get("finished"):
        return "LAP"
    if info.get("crashed"):
        return "CRASH"
    if info.get("off_road"):
        return "OFFTRACK"
    return "TIMEOUT"


def record(env, act, seed: int, label: str) -> tuple[list[np.ndarray], dict]:
    obs, _ = env.reset(seed=seed)
    frames, info, step, done = [], {}, 0, False
    raw_peak = 0  # brightest raw pixel seen, measured *before* the text overlay
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        step += 1
        done = bool(terminated or truncated)
        frame = env.render()
        if frame is None:
            continue
        lines = [
            label,
            f"step {step}  speed {info['speed']:.1f} m/s  dist {info['distance']:.0f} m",
        ]
        if done:
            lines.append(f"END: {outcome(info)}")
        raw_peak = max(raw_peak, int(np.asarray(frame).max()))
        frames.append(stamp(np.asarray(frame), lines))
    # A broken SDL setup does not raise -- it renders black. Refuse to write a
    # "video" that shows nothing, rather than let an empty file pass as evidence.
    if frames and raw_peak == 0:
        raise SystemExit(
            "rendered frames are all black -- the SDL video driver is not drawing. "
            f"SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER')!r}; it must not be 'dummy' "
            "(highway-env disables drawing under it). Try 'offscreen'."
        )
    # Hold the final frame ~1.5 s so the ending is actually visible.
    frames += frames[-1:] * 7
    return frames, {"seed": seed, "outcome": outcome(info), "distance": float(info["distance"]),
                    "steps": step, "speed": float(info["speed"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run dir (results/...)")
    parser.add_argument("--n", type=int, default=3, help="episodes (eval seeds 10000..)")
    parser.add_argument("--policy", choices=["trained", "straight"], default="trained")
    parser.add_argument("--out", type=Path, default=Path("report/videos/solo"))
    parser.add_argument("--wandb", action="store_true", help="also upload to W&B (racing-grpo)")
    args = parser.parse_args()

    # highway-env draws nothing when SDL_VIDEODRIVER == "dummy" (its EnvViewer
    # disables itself), so replace dummy -- which old shells and launchers set --
    # with SDL's `offscreen` driver, which renders headless on Linux and macOS.
    if os.environ.get("SDL_VIDEODRIVER", "dummy") == "dummy":
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
    env_cfg = load_env_config(args.run)
    env = make_solo_env(env_cfg, render_mode="rgb_array")
    act = load_policy(args.run, args.policy, env)

    run_name = args.run.name if args.policy == "trained" else f"{args.run.name}_straight"
    out_dir = args.out / run_name
    label = f"{run_name}  obs={env_cfg.obs_type.value}"

    results = []
    for i in range(args.n):
        seed = EVAL_SEED_OFFSET + i
        frames, res = record(env, act, seed, label)
        path = out_dir / f"ep{i:02d}_seed{seed}_{res['outcome']}_{res['distance']:.0f}m.mp4"
        write_video(frames, path)
        res["path"] = str(path)
        results.append(res)
        print(f"  [{i + 1}/{args.n}] {res['outcome']:8s} {res['distance']:6.1f} m  "
              f"{res['steps']:3d} steps  -> {path}")
    env.close()

    if args.wandb and os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb.init(project="racing-grpo", name=f"video/{run_name}", job_type="video",
                   config={"run": str(args.run), "policy": args.policy, "seeds": [r["seed"] for r in results]})
        for i, r in enumerate(results):
            key = f"episodes/ep{i:02d}_{r['outcome']}_{r['distance']:.0f}m"
            wandb.log({key: wandb.Video(r["path"], format="mp4")})
        print(f"uploaded to W&B: {wandb.run.url}")
        wandb.finish()
    elif args.wandb:
        print("--wandb set but no WANDB_API_KEY; videos are only on disk")


if __name__ == "__main__":
    main()
