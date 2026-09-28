# Environment throughput

- device: `cpu`
- cpu cores: 96
- torch: 2.14.0+cu130
- measured: 2026-09-28 13:42

Policy-in-the-loop rollout (small MLP forward + vectorised env step).

| n_envs | steps/sec | speedup vs 1 env |
|---:|---:|---:|
| 1 | 173 | 1.0x |
| 8 | 980 | 5.7x |
| 16 | 1,082 | 6.2x |
| 32 | 1,621 | 9.4x |

**Best: 32 envs at 1,621 steps/sec** -> a 2M-step run takes ~0.3 h.

## Reading this

Env stepping is CPU-bound pure Python; the policy MLP barely touches the GPU.
So the way to exploit an A100 is to run many seeds *concurrently* (each takes a
sliver of GPU memory) rather than to make one run faster. Scale n_envs per run
until the speedup curve flattens, then spend the remaining cores on more
concurrent runs -- see `scripts/launch_condition.sh`.
