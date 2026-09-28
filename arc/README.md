# Running on VT ARC

How this project runs on VT ARC: the OWL CPU nodes, and the A100 / H200
(Tinkercliffs) and L40S (Falcon) GPU nodes.
It is distilled from the runbooks of two earlier projects on the same cluster. Every
rule below cost one of them at least one wasted allocation. Account names and mail
addresses are deliberately absent, because this repo is public.

## What this workload is (and why it changes the GPU choice)

highway-env stepping is pure-Python and CPU-bound, and the policies are 64–256-unit
MLPs. **The GPU is almost idle. The allocation that matters is `--cpus-per-task`.**
Consequences:

- One GPU per job, always. Nothing here is multi-GPU, so NCCL (the H200 `ib0` trap)
  never comes up.
- Any of the three GPU types works. Pick whichever queue moves fastest: `squeue -p
  a100_normal_q`, `h200_normal_q` and (on Falcon) `l40s_normal_q`. The H200's
  141 GB is wasted on this workload, so a100 is the default.
- Rendering (Phase 5 videos) is pygame in software, so **no RT cores are needed.**
  The Isaac Sim "L40S only" rule does not apply. `arc_env.sh` sets
  `SDL_VIDEODRIVER=dummy`.
- **OWL is probably the best fit.** This workload needs many fast cores, and OWL
  has them. Its Genoa nodes have 96 cores each and hold a 3.8 GHz boost clock,
  while Tinkercliffs base nodes run at 2.0 GHz. highway-env steps each env in a
  single Python thread, so clock speed feeds straight into steps/s. The
  expectation is that one OWL node runs a PPO condition (5 × 17 cores) and uses no
  GPU allocation. The profile jobs below test this before anything relies on it.

## Choosing where to run

| | OWL (`--gpu owl`) | A100 / H200 / L40S |
|---|---|---|
| resource | up to 96 cores/node, 768 GB, no GPU | 1 GPU + whatever cores you request |
| policy device | `cpu` (automatic: `device: auto` finds no CUDA) | `cuda` |
| submit from | an OWL login node | Tinkercliffs (A100/H200), Falcon (L40S) |
| best for | pilot, conditions, HP-search arrays, h2h eval | only if profiling shows the GPU speeds up updates |

The call comes from data. Run the profile job on both OWL and an A100, and use
whichever reaches the higher steps/s. At the current MLP sizes (64–256 units) the
GPU is expected to add nothing. If that holds, run everything on OWL.

A GRPO condition needs 5 × 33 cores, which is more than one OWL node's 96. Split
it: submit seeds 0–2 and 3–4 as separate jobs, or give each run fewer envs than
cores. More envs than cores just slows every run down together.

## One-time setup (login node)

```bash
# 1. Checkout. The Mac and ARC checkouts only meet through git.
cd ~/ondemand/data
git clone git@github.com:Manas-Ganti/self-play-highwayenv.git
cd self-play-highwayenv

# 2. Dedicated env. Never reuse vrr / vrr-train / rtn.
arc/setup_env.sh              # ~/miniconda3/envs/racing-grpo, python 3.11, torch cu121
                              # runs pytest at the end; writes arc/requirements.lock.txt

# 3. Identity, outside the repo
mkdir -p ~/.config/racing-grpo
cat > ~/.config/racing-grpo/arc.env <<'EOF'
RG_ACCOUNT=<gpu slurm account>        # either ECE account works
RG_CPU_ACCOUNT=<cpu allocation>       # used for --gpu owl; falls back to RG_ACCOUNT
RG_MAIL_USER=<pid>@vt.edu
RG_OWL_PARTITION=normal_q             # confirm in step 4
EOF

# 4. Once, on an OWL login node: confirm partition + QOS names
sinfo -s
sacctmgr show qos format=name%28,priority,maxwall
```

> The OWL partition name (`normal_q`) follows the ARC convention but **has not
> been confirmed on OWL**, and `submit.sh` passes no OWL QOS. If `sacctmgr` lists
> short/base tiers there, pass `--qos <name>` or set `RG_OWL_PARTITION`. The
> first OWL submit will show whether this is right; a wrong name is rejected at
> submit time, so it costs no queue wait.

W&B reads `WANDB_API_KEY` from `~/.config/vrr/secrets.env`, the same file the
other projects use (override it with `SECRETS_ENV=`). If no key is set, the run
logs to TensorBoard under `results/`.

## Submitting

Always go through `arc/submit.sh`. It derives partition, QOS, gres and `--mem` as
one set, and it reads the account from `~/.config/racing-grpo/arc.env`. Settings
go **after** the `.slurm` file as argparse flags, where a typo fails loudly.

| option | default | notes |
|---|---|---|
| `--gpu` | `a100` | `owl` → CPU-only OWL; `a100` / `h200` → Tinkercliffs; `l40s` → Falcon (submit from `falcon1`/`falcon2`) |
| `--partition` / `--qos` | derived | override when a cluster's names differ |
| `--time` | `04:00:00` | < 24 h → `*_short` QOS (highest priority); ≥ 24 h → `*_base` |
| `--cpus` | `32` | **the real resource.** Size it from `report/throughput.md` |
| `--mem` | `2G × cpus` | never `0`: a job asking for the whole node cannot backfill |
| `--array` | — | for `arc/search.slurm` |
| `--dry-run` | — | print the `sbatch` line only |

### The phase sequence on ARC

```bash
PY=~/miniconda3/envs/racing-grpo/bin/python

# Phase 0 follow-up: profile OWL vs A100 (log.md asks for a re-profile). Decides where
# everything else runs.
arc/submit.sh --gpu owl --cpus 32 --time 00:30:00 arc/job.slurm scripts/profile_env.py \
    --device cpu --env-counts 1 8 16 32 --out report/throughput_owl.md          # from owl1
arc/submit.sh --cpus 32 --time 00:30:00 arc/job.slurm scripts/profile_env.py \
    --device cuda --env-counts 1 8 16 32 --out report/throughput_a100.md        # from tinkercliffs

# Each command below also works with `--gpu owl` (submit from an OWL login node).

# Phase 1: SB3 pilot. Gate: lap rate >= 0.90 (printed as "PHASE 1 GATE ...")
arc/submit.sh --cpus 20 --time 08:00:00 arc/job.slurm scripts/sb3_pilot.py
#   -> results/phase1/sb3_ppo_seed0/solo_eval.json ; then freeze configs/reward.yaml

# Phase 2: custom PPO parity. 3 seeds each; SB3 seeds via --seed
arc/submit.sh --cpus 60 --time 08:00:00 arc/condition.slurm ppo 3 2000000
for s in 0 1 2; do arc/submit.sh --cpus 20 --time 08:00:00 arc/job.slurm scripts/sb3_pilot.py --seed $s; done

# Phase 3: matched HP search. One trial per array task, same indices for both
arc/submit.sh --array 0-29 --cpus 16 --time 02:00:00 arc/search.slurm ppo
arc/submit.sh --array 0-29 --cpus 36 --time 02:00:00 arc/search.slurm grpo
$PY scripts/hp_search.py --algo ppo --collect && $PY scripts/hp_search.py --algo grpo --collect

# Phase 4: 5 seeds x {PPO, GRPO}, concurrently on one GPU each
arc/submit.sh --cpus 96 --time 12:00:00 arc/condition.slurm ppo
arc/submit.sh --cpus 96 --time 12:00:00 arc/condition.slurm grpo

# Phase 5: head-to-head round robin
arc/submit.sh --cpus 32 --time 12:00:00 arc/job.slurm scripts/evaluate_h2h.py --results results/
```

The `--cpus` and `--time` values above are **estimates** from the laptop profile
(~390 steps/s at 8 cores). Replace them with the ARC profile numbers. Rules of
thumb:

- Give a run about `n_envs + 1` cores. PPO uses 16 envs. GRPO uses
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

Two things behave differently from the other projects:

- The Mac `.venv` is Python 3.12 and the ARC env is 3.11 (the spec's version).
  The code runs on both, and the tests pass on 3.12 locally.
- On macOS `/bin/bash` is 3.2, and `launch_condition.sh` needs bash ≥ 4.3.
  ARC is fine. On the Mac, run it with Homebrew bash.
