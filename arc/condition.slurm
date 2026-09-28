#!/usr/bin/env bash
#SBATCH --job-name=rg-condition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Phase 4: the seeds of one condition, concurrently, on one OWL node.
# Env stepping is CPU-bound, so this is sized in cores:
#
#   arc/submit.sh --cpus 90 --time 12:00:00 arc/condition.slurm ppo                 # 5 x 18
#   arc/submit.sh --cpus 96 --time 12:00:00 arc/condition.slurm grpo 3 2000000 0    # 3 x 32
#   arc/submit.sh --cpus 64 --time 12:00:00 arc/condition.slurm grpo 2 2000000 3    # 2 x 32
#
# Size --cpus from report/throughput.md measured on the node type you submit to.

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

./scripts/launch_condition.sh "$@"
