#!/usr/bin/env bash
# Launch the seeds of one condition concurrently on a single node.
# On VT ARC, run it inside a job: arc/submit.sh ... arc/condition.slurm <algo>
#
#   ./scripts/launch_condition.sh ppo                  # seeds 0-4, 2M steps each
#   ./scripts/launch_condition.sh grpo 3 2000000       # seeds 0-2
#   ./scripts/launch_condition.sh grpo 2 2000000 3     # seeds 3-4
#
# Why concurrent seeds rather than one fast run (CLAUDE.md §1): highway-env
# stepping is pure-Python and CPU-bound, and the policy is a small MLP, so one
# run cannot use a whole node. The CPU is the resource being rationed. When a
# condition needs more cores than one node has (GRPO: 5 x 33), split it into
# two jobs with FIRST_SEED.

set -euo pipefail

ALGO="${1:?usage: launch_condition.sh <ppo|grpo> [n_seeds] [total_steps] [first_seed]}"
N_SEEDS="${2:-5}"
TOTAL_STEPS="${3:-2000000}"
FIRST_SEED="${4:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="results/launch_logs"
mkdir -p "$LOG_DIR"

PY="${PY:-python}"

# The cores this process may actually use. Under SLURM that is the job's
# allocation, NOT os.cpu_count(): on a shared 128-core ARC node cpu_count() is
# 128 while the job may own 32 arbitrary cores, and pinning to "0-31" would land
# on cores the job does not hold (taskset fails, or every run piles onto one).
ALLOWED=($("$PY" -c 'import os
cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
print(" ".join(map(str, cores)))'))
CORES=${#ALLOWED[@]}
# Each run owns a disjoint slice of those cores; its vector-env workers live
# there. Without this the runs' worker pools fight over the same cores and every
# run slows down together.
CORES_PER_RUN=$(( CORES / N_SEEDS ))
if (( CORES_PER_RUN < 1 )); then CORES_PER_RUN=1; fi

echo "launching $N_SEEDS x $ALGO (seeds ${FIRST_SEED}-$(( FIRST_SEED + N_SEEDS - 1 ))) @ ${TOTAL_STEPS} steps"
echo "  $CORES cores total, ~$CORES_PER_RUN per run"

PIDS=()
for (( SEED=FIRST_SEED; SEED<FIRST_SEED+N_SEEDS; SEED++ )); do
  CONFIG="configs/${ALGO}_seed${SEED}.yaml"
  [[ -f "$CONFIG" ]] || { echo "missing $CONFIG"; exit 1; }

  LOG="${LOG_DIR}/${ALGO}_seed${SEED}.log"
  START=$(( (SEED - FIRST_SEED) * CORES_PER_RUN ))
  CPU_LIST="$(IFS=,; echo "${ALLOWED[*]:START:CORES_PER_RUN}")"

  CMD=("$PY" scripts/train.py --config "$CONFIG" --seed "$SEED" --total-steps "$TOTAL_STEPS")

  # Pin CPU affinity when taskset is available (Linux) and every run gets its own
  # cores. Under SLURM, CUDA_VISIBLE_DEVICES is already the one allocated GPU and
  # is inherited as-is; outside SLURM default to GPU 0. All runs share it.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  if command -v taskset >/dev/null 2>&1 && (( CORES >= N_SEEDS )); then
    taskset -c "$CPU_LIST" "${CMD[@]}" > "$LOG" 2>&1 &
  else
    CPU_LIST="unpinned"
    "${CMD[@]}" > "$LOG" 2>&1 &
  fi

  PID=$!
  PIDS+=("$PID")
  echo "  seed $SEED -> pid ${PID}, cores ${CPU_LIST}, log $LOG"
done

echo "waiting for ${#PIDS[@]} runs..."
FAILED=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || { echo "run with pid $PID FAILED"; FAILED=1; }
done

if (( FAILED )); then
  echo "at least one run failed; check $LOG_DIR"
  exit 1
fi
echo "all $N_SEEDS $ALGO runs complete"
