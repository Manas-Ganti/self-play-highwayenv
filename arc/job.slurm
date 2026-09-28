#!/usr/bin/env bash
#SBATCH --job-name=rg-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%j.out
#
# Generic single-process job: runs `$PY <args>` in the project env.
# Submit through arc/submit.sh, which supplies partition/QOS/gres/account:
#
#   arc/submit.sh --time 00:30:00 arc/job.slurm scripts/profile_env.py --device cuda
#   arc/submit.sh --time 06:00:00 arc/job.slurm scripts/sb3_pilot.py
#   arc/submit.sh --time 04:00:00 arc/job.slurm scripts/train.py --config configs/ppo_seed0.yaml
#   arc/submit.sh --cpus 64 --time 12:00:00 arc/job.slurm scripts/evaluate_h2h.py --results results/

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

[[ $# -gt 0 ]] || { echo "usage: arc/job.slurm <script.py> [args...]" >&2; exit 2; }
echo "[job] $PY $*"
"$PY" "$@"
