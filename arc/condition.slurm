#!/usr/bin/env bash
#SBATCH --job-name=rg-condition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Phase 4: all seeds of one condition, concurrently, on ONE GPU (CLAUDE.md §1).
# Env stepping is CPU-bound, so give it cores, not GPUs:
#
#   arc/submit.sh --cpus 96 --time 08:00:00 arc/condition.slurm ppo
#   arc/submit.sh --cpus 96 --time 08:00:00 arc/condition.slurm grpo 5 2000000
#
# Size --cpus from report/throughput.md measured on the node type you submit to.

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

./scripts/launch_condition.sh "$@"
