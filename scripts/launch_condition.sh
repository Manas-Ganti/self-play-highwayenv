#!/usr/bin/env bash
# Launch all seeds of one condition concurrently on a single A100 node.
#
#   ./scripts/launch_condition.sh ppo            # seeds 0-4, 2M steps each
#   ./scripts/launch_condition.sh grpo 5 2000000
#
# Why concurrent seeds rather than one fast run (CLAUDE.md §1): highway-env
# stepping is pure-Python and CPU-bound, and the policy is a small MLP. One run
# cannot saturate an A100 no matter how hard it tries, but five runs share it
# trivially -- each takes a sliver of GPU memory. So the GPU is exploited by
# *width*, and the CPU is the resource actually being rationed here.

set -euo pipefail

ALGO="${1:?usage: launch_condition.sh <ppo|grpo> [n_seeds] [total_steps]}"
N_SEEDS="${2:-5}"
TOTAL_STEPS="${3:-2000000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="results/launch_logs"
mkdir -p "$LOG_DIR"

CORES="$(python -c 'import os; print(os.cpu_count() or 1)')"
# Each run owns a contiguous slice of cores; its vector-env workers live there.
# Without this the runs' worker pools fight over the same cores and every run
# slows down together.
CORES_PER_RUN=$(( CORES / N_SEEDS ))
if (( CORES_PER_RUN < 1 )); then CORES_PER_RUN=1; fi

echo "launching $N_SEEDS x $ALGO @ ${TOTAL_STEPS} steps"
echo "  $CORES cores total, ~$CORES_PER_RUN per run"

PIDS=()
for (( SEED=0; SEED<N_SEEDS; SEED++ )); do
  CONFIG="configs/${ALGO}_seed${SEED}.yaml"
  [[ -f "$CONFIG" ]] || { echo "missing $CONFIG"; exit 1; }

  LOG="${LOG_DIR}/${ALGO}_seed${SEED}.log"
  START=$(( SEED * CORES_PER_RUN ))
  END=$(( START + CORES_PER_RUN - 1 ))

  CMD=(python scripts/train.py --config "$CONFIG" --seed "$SEED" --total-steps "$TOTAL_STEPS")

  # Pin CPU affinity when taskset is available (Linux); the A100 is shared by all.
  if command -v taskset >/dev/null 2>&1 && (( CORES_PER_RUN > 0 )); then
    CUDA_VISIBLE_DEVICES=0 taskset -c "${START}-${END}" "${CMD[@]}" > "$LOG" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=0 "${CMD[@]}" > "$LOG" 2>&1 &
  fi

  PIDS+=($!)
  echo "  seed $SEED -> pid ${PIDS[-1]}, cores ${START}-${END}, log $LOG"
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
