# Running on VT ARC

How this project runs on VT ARC. **Platform: OWL CPU nodes** (`normal_q`,
the project account). Only one partition's resources can be used, and this
project's work is CPU work, so everything runs there.
It is distilled from the runbooks of two earlier projects on the same cluster. Every
rule below cost one of them at least one wasted allocation. Account names and mail
addresses are deliberately absent, because this repo is public.

## Why OWL CPU nodes

highway-env stepping is pure Python and CPU-bound, and the policies are 64–256-unit
MLPs (≤ ~150k parameters). **Nothing in this project needs a GPU:**

| work | runs on |
|---|---|
| env stepping (most of the wall time) | CPU, one core per env |
| policy forward + PPO/GRPO updates | CPU (tiny MLPs; `device: auto` picks CPU when no GPU is allocated) |
| eval, head-to-head, statistics | CPU |
| Phase 5 videos | CPU (pygame software rendering; `SDL_VIDEODRIVER=offscreen`, never `dummy`: highway-env draws nothing under `dummy`) |

Given one partition, OWL `normal_q` wins on everything this workload uses:

- **cores per job.** A full 96-core node. GPU partitions give about 16 cores per
  GPU (128 per 8 GPUs, for both the Tinkercliffs A100s and the OWL B200s).
- **clock speed.** Genoa at 3.8 GHz sustained. Each env steps in a single Python
  thread, so clock speed feeds straight into steps/s.
- **cost.** A GPU allocation would pay for a GPU that sits idle.

`submit.sh` defaults to OWL. The `--gpu a100|h200|l40s` paths still exist in
case the project ever moves partitions, but they are not a second pool to mix
with OWL.

**Sizing rule:** give each concurrent run `n_envs + 1` cores and never exceed 96
per job. PPO runs 16 envs, so a condition is 5 × 18 = 90 cores, one job. GRPO runs
32 envs, so a condition is 5 × 33 = 165 cores and must be split into two jobs:
seeds 0–2 (`grpo 3 2000000 0`) and seeds 3–4 (`grpo 2 2000000 3`). More envs than
cores just slows every run down together.

## One-time setup (login node)

```bash
# 1. Checkout. The Mac and ARC checkouts only meet through git.
cd ~/ondemand/data
git clone git@github.com:Manas-Ganti/self-play-highwayenv.git
cd self-play-highwayenv

# 2. Dedicated env. Never reuse vrr / vrr-train / rtn.
arc/setup_env.sh              # ~/miniconda3/envs/racing-grpo, python 3.11, torch (TORCH_CUDA)
                              # runs pytest at the end; writes arc/requirements.lock.txt

# 3. Identity, outside the repo
mkdir -p ~/.config/racing-grpo
cat > ~/.config/racing-grpo/arc.env <<'EOF'
RG_ACCOUNT=<slurm account>            # one account for every partition, CPU and GPU
RG_MAIL_USER=<pid>@vt.edu
EOF

```

**OWL QOS** (partition `normal_q`, confirmed 2026-09-28):

| QOS | priority | max wall | UsageFactor |
|---|---|---|---|
| `owl_normal_short` | 1500 | 1 day | **2** (bills double) |
| `owl_normal_base` | 1000 | 7 days | 1 |
| `owl_normal_long` | 500 | 14 days | 1 |

`submit.sh` defaults to **`owl_normal_base`**, or `long` past 7 days.
On Tinkercliffs, "short" is simply the best tier. On OWL it costs twice the
allocation, and these jobs hold dozens of cores for hours. Pass
`--qos owl_normal_short` only when getting the job started sooner is worth
double the cost. `preemptable_q` is free (UsageFactor 0), but it can evict a job,
and `train.py` cannot resume yet, so it is not used.

W&B reads `WANDB_API_KEY` from `~/.config/vrr/secrets.env`, the same file the
other projects use (override it with `SECRETS_ENV=`). If no key is set, the run
logs to TensorBoard under `results/`.

## Submitting

Always go through `arc/submit.sh`. It derives partition, QOS, gres and `--mem` as
one set, and it reads the account from `~/.config/racing-grpo/arc.env`. Settings
go **after** the `.slurm` file as argparse flags, where a typo fails loudly.

| option | default | notes |
|---|---|---|
| `--gpu` | `owl` | OWL CPU nodes. (`a100`/`h200`/`l40s` exist only for a future partition move) |
| `--partition` / `--qos` | derived | e.g. `--qos owl_normal_short` for priority at 2× cost |
| `--time` | `04:00:00` | ≤ 7 days → `owl_normal_base`; longer → `owl_normal_long` |
| `--cpus` | `32` | **the real resource.** Max 96 (one node). Size it from `report/throughput_owl.md` |
| `--mem` | `2G × cpus` | never `0`: a job asking for the whole node cannot backfill |
| `--array` | — | for `arc/search.slurm` |
| `--dry-run` | — | print the `sbatch` line only |

### The phase sequence on ARC

```bash
PY=~/miniconda3/envs/racing-grpo/bin/python

# Phase 0 follow-up: profile on OWL. DONE 2026-09-28 (job 968475) -> report/throughput_owl.md
arc/submit.sh --cpus 32 --time 00:30:00 arc/job.slurm scripts/profile_env.py \
    --device cpu --env-counts 1 8 16 32 --out report/throughput_owl.md

# Phase 1: SB3 pilot. Gate: lap rate >= 0.90 (printed as "PHASE 1 GATE ...")
arc/submit.sh --cpus 17 --time 03:00:00 arc/job.slurm scripts/sb3_pilot.py
#   -> results/phase1/sb3_ppo_seed0/solo_eval.json ; then freeze configs/reward.yaml

# Phase 2: custom PPO parity. 3 seeds each; SB3 seeds via --seed
arc/submit.sh --cpus 51 --time 04:00:00 arc/condition.slurm ppo 3 2000000
for s in 0 1 2; do arc/submit.sh --cpus 17 --time 03:00:00 arc/job.slurm scripts/sb3_pilot.py --seed $s; done

# Phase 3: matched HP search. One trial per array task, same indices for both.
# GRPO trials run up to 64 envs (n_groups=8), hence the larger request.
arc/submit.sh --array 0-29 --cpus 17 --time 01:30:00 arc/search.slurm ppo
arc/submit.sh --array 0-29 --cpus 65 --time 01:30:00 arc/search.slurm grpo
$PY scripts/hp_search.py --algo ppo --collect && $PY scripts/hp_search.py --algo grpo --collect

# Phase 4: 5 seeds x {PPO, GRPO}. PPO fits one node; GRPO is split across two jobs.
arc/submit.sh --cpus 85 --time 04:00:00 arc/condition.slurm ppo
arc/submit.sh --cpus 96 --time 04:00:00 arc/condition.slurm grpo 3 2000000 0
arc/submit.sh --cpus 66 --time 04:00:00 arc/condition.slurm grpo 2 2000000 3

# Phase 5: head-to-head round robin
arc/submit.sh --cpus 32 --time 12:00:00 arc/job.slurm scripts/evaluate_h2h.py --results results/
```

**Sizing basis** (`report/throughput_owl.md`, OWL job 968475, 32 cores):

| envs | steps/s | vs 1 env |
|---:|---:|---:|
| 1 | 173 | 1.0× |
| 8 | 980 | 5.7× |
| 16 | 1,082 | 6.2× |
| 32 | 1,621 | 9.4× |

That puts a 2M-step PPO run (16 envs) at about 31 min of collection. Periodic
eval adds about 12 min (20 episodes every 100k steps, on a single env at ~173
steps/s), and updates add a few more, so about 1 h in total. The `--time` values
above are about 3–4× that. **Scaling flattens after 8 envs** (8 → 16 is +10%
with cores to spare). The bottleneck is the main process, which runs the policy
and waits on the slowest env each step, not the core count. `n_envs` stays as
configured because it is an algorithm setting; the core counts below are
conservative. Rules of thumb:

- Give a run `n_envs + 1` cores. PPO uses 16 envs. GRPO uses
  `group_size × n_groups`, which is 32 at the defaults and up to 64 in the search.
- `condition.slurm` splits the job's cores evenly across seeds. It uses the job's
  real allocation (`sched_getaffinity`), not the node's core count.
- Check a few of each search's array tasks and resubmit only the missing indices.
  `--collect` names them, and it refuses to choose a winner from an incomplete search.

## Watching

```bash
squeue -u $USER -o "%.10i %.14j %.9T %.11M %.11L %.22R %N"
tail -f logs/slurm/rg-condition-<jobid>.out
tail -f results/launch_logs/ppo_seed0.log      # per-seed logs inside a condition job
seff <jobid>                                   # CPU efficiency: low means --cpus is too high
```

## Getting results to the Mac

`results/` is gitignored. Pull it over rsync (on the Mac):

```bash
arc/pull_results.sh               # json/tensorboard only
arc/pull_results.sh --with-ckpt   # plus checkpoints, for local video export
```

## Why the launchers look like this

| rule | the failure it prevents |
|---|---|
| absolute interpreter path, no `conda activate` | the cluster's Miniforge cannot see `~/miniconda3`, and `activate` "succeeds" without switching; the job then dies on an import |
| `$PY -V` asserted at job start | a 0-byte interpreter makes every job "succeed" in 2 s with an empty log |
| import check for torch/gymnasium/highway_env/sb3 | a wrong env fails in second 1 instead of minute 20 |
| `OMP_NUM_THREADS=1` | each vector-env worker's torch spawning one thread per node core (hundreds of processes on a 128-core DGX) |
| `SLURM_SUBMIT_DIR`, not `dirname $0` | sbatch runs a spool copy of the script, so relative paths resolve wrong |
| settings as argparse passthrough | `VAR=x sbatch` silently drops names the launcher does not read |
| explicit `--mem`, never `0` | `--mem=0` means "whole node" and never backfills |
| `*_short` QOS under 24 h | priority 2000 vs 1000. "short" means up to a full day |
| `logs/slurm/` tracked with `.gitkeep` | sbatch cannot open `--output` in a missing dir |
| warning on uncommitted changes | `.slurm` is copied at **submit** time and Python is read at **run** time. Edit on the Mac, push, `git pull` on ARC, then submit |
| `WANDB_DIR=$HOME/wandb` | `/projects/$USER` is not writable |

One difference from the other projects: the Mac `.venv` is Python 3.12 and the
ARC env is 3.11 (the spec's version). The code runs on both, and the tests pass
on 3.12 locally.
