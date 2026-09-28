"""Throughput profile -- the Phase 0 gate artefact.

    python scripts/profile_env.py --out report/throughput.md

highway-env stepping is pure-Python and CPU-bound while the policy is a small
MLP, so the A100 is *not* the bottleneck (CLAUDE.md §1). This script measures
where the ceiling actually is, so the number of concurrent runs per node is
chosen from data rather than from hope.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from algos.common.nets import ActorCritic  # noqa: E402
from algos.common.utils import resolve_device  # noqa: E402
from envs import make_vec_env  # noqa: E402
from envs.config import EnvConfig, RewardConfig  # noqa: E402


def profile(n_envs: int, n_steps: int, device: torch.device, asynchronous: bool) -> float:
    """Return steps/sec for a policy-in-the-loop rollout at this env count."""

    cfg = EnvConfig(reward=RewardConfig.load())
    env = make_vec_env(cfg, n_envs=n_envs, asynchronous=asynchronous)
    obs_dim = int(env.single_observation_space.shape[0])
    act_dim = int(env.single_action_space.shape[0])
    model = ActorCritic(obs_dim, act_dim).to(device)

    obs, _ = env.reset(seed=list(range(n_envs)))
    # Warm up: the first steps pay for worker startup and CUDA context creation.
    for _ in range(10):
        with torch.no_grad():
            action, _, _ = model.act(torch.as_tensor(obs, dtype=torch.float32, device=device))
        obs, *_ = env.step(action.cpu().numpy())

    start = time.perf_counter()
    for _ in range(n_steps):
        with torch.no_grad():
            action, _, _ = model.act(torch.as_tensor(obs, dtype=torch.float32, device=device))
        obs, *_ = env.step(action.cpu().numpy())
    elapsed = time.perf_counter() - start

    env.close()
    return n_steps * n_envs / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-counts", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--steps", type=int, default=200, help="steps per env")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("report/throughput.md"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    # Cores this process may use: under SLURM that is the job's allocation, not
    # the node's os.cpu_count() (96 on an OWL node regardless of --cpus).
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    node = os.environ.get("SLURMD_NODENAME", "local")
    print(f"device={device}  cpu_cores={cores} (allocated)  node={node}")

    rows = []
    for n_envs in args.env_counts:
        sps = profile(n_envs, args.steps, device, asynchronous=n_envs > 1)
        rows.append((n_envs, sps))
        print(f"  n_envs={n_envs:3d}  {sps:8.0f} steps/sec")

    best_envs, best_sps = max(rows, key=lambda r: r[1])
    hours_2m = 2_000_000 / best_sps / 3600

    lines = [
        "# Environment throughput",
        "",
        f"- device: `{device}`",
        f"- cpu cores: {cores} (allocated to this process)",
        f"- node: {node}",
        f"- torch: {torch.__version__}",
        f"- measured: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Policy-in-the-loop rollout (small MLP forward + vectorised env step).",
        "",
        "| n_envs | steps/sec | speedup vs 1 env |",
        "|---:|---:|---:|",
    ]
    baseline = rows[0][1]
    for n_envs, sps in rows:
        lines.append(f"| {n_envs} | {sps:,.0f} | {sps / baseline:.1f}x |")

    lines += [
        "",
        f"**Best: {best_envs} envs at {best_sps:,.0f} steps/sec** "
        f"-> 2M steps of *collection alone* take ~{hours_2m:.1f} h. A real run also pays",
        "for periodic eval (single env) and policy updates; see `arc/README.md` for sizing.",
        "",
        "## Reading this",
        "",
        "Env stepping is CPU-bound pure Python and the policy is a small MLP, so the",
        "resource being rationed is CPU cores, not accelerators. Once the speedup curve",
        "flattens, more envs per run stop paying; spend the remaining cores on more",
        "concurrent runs instead -- see `scripts/launch_condition.sh`.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
