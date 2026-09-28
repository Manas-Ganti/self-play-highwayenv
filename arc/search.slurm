#!/usr/bin/env bash
#SBATCH --job-name=rg-search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x-%A_%a.out
#
# Phase 3: one hyperparameter-search trial per array task. The trial index is the
# array index, so both algorithms get the same trials by construction:
#
#   arc/submit.sh --array 0-29 --cpus 16 --time 02:00:00 arc/search.slurm ppo
#   arc/submit.sh --array 0-29 --cpus 16 --time 02:00:00 arc/search.slurm grpo
#
# then, once every task of BOTH arrays has finished (login node, no GPU):
#
#   $PY scripts/hp_search.py --algo ppo  --collect
#   $PY scripts/hp_search.py --algo grpo --collect

source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"

ALGO="${1:?usage: arc/search.slurm <ppo|grpo> [extra hp_search args]}"
shift
: "${SLURM_ARRAY_TASK_ID:?submit with --array (e.g. --array 0-29)}"

"$PY" scripts/hp_search.py --algo "$ALGO" --trial-index "$SLURM_ARRAY_TASK_ID" "$@"
