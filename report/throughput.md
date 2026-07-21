# Environment throughput

- device: `mps`
- cpu cores: 8
- torch: 2.12.0
- measured: 2026-07-13 16:20

Policy-in-the-loop rollout (small MLP forward + vectorised env step).

| n_envs | steps/sec | speedup vs 1 env |
|---:|---:|---:|
| 1 | 109 | 1.0x |
| 4 | 346 | 3.2x |
| 8 | 384 | 3.5x |
| 16 | 381 | 3.5x |
| 32 | 388 | 3.5x |

**Best: 32 envs at 388 steps/sec** -> a 2M-step run takes ~1.4 h.

## Reading this

Env stepping is CPU-bound pure Python; the policy MLP barely touches the GPU.
So the way to exploit an A100 is to run many seeds *concurrently* (each takes a
sliver of GPU memory) rather than to make one run faster. Scale n_envs per run
until the speedup curve flattens, then spend the remaining cores on more
concurrent runs -- see `scripts/launch_condition.sh`.
