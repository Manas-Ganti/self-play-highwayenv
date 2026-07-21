"""Video export for head-to-head matchups.

    python analysis/video.py --a results/ppo_seed0 --b results/grpo_seed0 --n 20

Phase 5 requires 20 videos per headline matchup and an explicit review for
**ramming**. That review is not decoration: the win condition was designed to make
ramming lose, and the only way to know it worked is to watch. If the videos show
agents winning by driving into rivals, the aggressor attribution needs tightening
and the matchups get re-run (CLAUDE.md §5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from algos.common.utils import resolve_device  # noqa: E402
from envs import make_h2h_env, make_solo_env  # noqa: E402
from envs.config import EnvConfig, RewardConfig  # noqa: E402
from eval.h2h import time_to_collision  # noqa: E402
from eval.policies import IDMPolicy, TorchPolicy  # noqa: E402


def write_video(frames: list[np.ndarray], path: Path, fps: int = 5) -> None:
    """Write frames to an mp4, falling back to a GIF when no encoder is present."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
    except (ImportError, ValueError) as exc:
        gif = path.with_suffix(".gif")
        print(f"  mp4 encoder unavailable ({exc}); writing {gif.name}")
        import imageio.v2 as imageio

        imageio.mimwrite(gif, frames, duration=1.0 / fps)


def record_episode(policy_a, policy_b, env, seed: int, swapped: bool):
    """Play one episode with rendering on; return (frames, verdict dict)."""

    obs, _ = env.reset(seed=seed, options={"swap_starts": swapped})
    policy_a.reset(env, env.vehicle_index("agent_0"))
    policy_b.reset(env, env.vehicle_index("agent_1"))

    frames, min_ttc, info = [], float("inf"), {}
    while env.agents:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        actions = {
            "agent_0": policy_a.action(obs["agent_0"]),
            "agent_1": policy_b.action(obs["agent_1"]),
        }
        obs, _, _, _, info = env.step(actions)

        a, b = info["agent_0"], info["agent_1"]
        min_ttc = min(
            min_ttc, time_to_collision(a["position"], a["velocity"], b["position"], b["velocity"])
        )

    a, b = info["agent_0"], info["agent_1"]
    contact = a["crashed"] and b["crashed"]
    return frames, {
        "seed": seed,
        "swapped": swapped,
        "result_a": a["result"],
        "agent_agent_contact": bool(contact),
        "min_ttc": float(min_ttc),
        # A win taken immediately after contact is the signature of ramming. It is
        # flagged, not auto-judged -- the videos are the evidence.
        "ramming_suspect": bool(contact and a["result"] == "win"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="run dir for agent A")
    parser.add_argument("--b", type=Path, help="run dir for agent B; omit to race the IDM floor")
    parser.add_argument("--out", type=Path, default=Path("report/videos"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    reward = RewardConfig.load()

    probe = make_solo_env(EnvConfig(reward=reward))
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.shape[0])
    probe.close()

    policy_a = TorchPolicy.load(args.a / "final.pt", obs_dim, act_dim, device, name=args.a.name)
    policy_b = (
        TorchPolicy.load(args.b / "final.pt", obs_dim, act_dim, device, name=args.b.name)
        if args.b
        else IDMPolicy()
    )

    env = make_h2h_env(EnvConfig(n_agents=2, reward=reward), render_mode="rgb_array")
    out_dir = args.out / f"{policy_a.name}_vs_{policy_b.name}"

    verdicts = []
    for i in range(args.n):
        seed = args.seed_offset + i
        swapped = bool(i % 2)
        frames, verdict = record_episode(policy_a, policy_b, env, seed, swapped)
        verdicts.append(verdict)

        tag = "RAMMING?" if verdict["ramming_suspect"] else verdict["result_a"]
        name = f"ep{i:02d}_seed{seed}_{'swap' if swapped else 'norm'}_{tag}.mp4"
        if frames:
            write_video(frames, out_dir / name)
        print(f"  [{i + 1}/{args.n}] {name}")
    env.close()

    suspects = [v for v in verdicts if v["ramming_suspect"]]
    print(f"\nwrote {len(verdicts)} videos to {out_dir}")
    print(f"agent-agent contact in {sum(v['agent_agent_contact'] for v in verdicts)} episodes")
    if suspects:
        print(
            f"\n!! {len(suspects)} episodes were WON immediately after agent-agent contact.\n"
            "   Watch them. If the winner initiated the contact, the aggressor\n"
            "   attribution in envs/multi_agent.py is too loose and Phase 5 must be\n"
            "   re-run after tightening it (CLAUDE.md §5)."
        )
    else:
        print("no ramming suspects: no episode was won straight off an agent-agent collision.")


if __name__ == "__main__":
    main()
