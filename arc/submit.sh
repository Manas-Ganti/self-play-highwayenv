#!/usr/bin/env bash
# The one way to submit a job for this project on VT ARC.
#
#   arc/submit.sh [--gpu owl|a100|h200|l40s] [--time HH:MM:SS] [--cpus N] [--mem 64G]
#                 [--partition P] [--qos Q] [--array 0-29] [--dry-run]
#                 <arc/job.slurm> [passthrough args...]
#
# This project runs on OWL CPU nodes (the default, `--gpu owl`): the workload is
# CPU-bound and only one partition's resources are available to it at a time
# (arc/README.md). The GPU types remain as an alternative, not a second pool.
#
#   arc/submit.sh --time 00:30:00 arc/job.slurm scripts/profile_env.py --device cpu
#   arc/submit.sh --cpus 20 --time 08:00:00 arc/job.slurm scripts/sb3_pilot.py
#   arc/submit.sh --cpus 90 --time 12:00:00 arc/condition.slurm ppo
#   arc/submit.sh --array 0-29 --cpus 18 --time 03:00:00 arc/search.slurm ppo
#
# Why a wrapper instead of bare sbatch: partition, QOS and gres come as a set that
# differs per GPU type and per wall time, and getting one wrong costs a queue
# wait (a job on the wrong QOS pends for a day; `--mem=0` never backfills). This
# derives them together. Everything after the .slurm file is passed straight to
# the Python entry point, where argparse fails loudly on a typo -- unlike
# `VAR=x sbatch`, which silently drops any name the launcher does not read.
#
# Account and mail address are read from ~/.config/racing-grpo/arc.env, never
# from this public repo:
#     RG_ACCOUNT=<slurm account>      # one account for every partition (CPU and GPU)
#     RG_MAIL_USER=<pid>@vt.edu

set -euo pipefail

GPU=owl
TIME=04:00:00
CPUS=32
MEM=""
ARRAY=""
PART_OVERRIDE=""
QOS_OVERRIDE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --array) ARRAY="$2"; shift 2 ;;
    --partition) PART_OVERRIDE="$2"; shift 2 ;;
    --qos) QOS_OVERRIDE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n 2,32p "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

SCRIPT="${1:?usage: arc/submit.sh [options] <arc/*.slurm> [args...]}"
shift
[[ -f "$SCRIPT" ]] || { echo "no such launcher: $SCRIPT" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONF="${RG_ARC_CONF:-$HOME/.config/racing-grpo/arc.env}"
# shellcheck disable=SC1090
[[ -f "$CONF" ]] && source "$CONF"

# Wall time in hours, to pick the QOS: "short" is the highest priority on ARC and
# caps at a full day, so everything under 24 h belongs there.
HOURS="$(awk -F'[-:]' '{ if (NF==4) print $1*24+$2; else print $1+0 }' <<<"$TIME")"
if (( HOURS >= 24 )); then TIER=base; else TIER=short; fi

case "$GPU" in
  a100)
    # Tinkercliffs. NCCL is irrelevant here: every job is single-process.
    PART=a100_normal_q; QOS="tc_a100_normal_${TIER}"; GRES=gpu:a100:1 ;;
  h200)
    PART=h200_normal_q; QOS="tc_h200_normal_${TIER}"; GRES=gpu:h200:1 ;;
  l40s)
    # Falcon: submit from falcon1/falcon2 (/home is shared). No QOS flag.
    PART=l40s_normal_q; QOS=""; GRES=gpu:l40s:1
    if [[ "$(hostname)" != falcon* ]]; then
      echo "warning: l40s lives on Falcon; submit from falcon1/falcon2, not $(hostname)" >&2
    fi ;;
  owl|cpu)
    # OWL, CPU only: no gres. QOS tiers on normal_q (sacctmgr, 2026-09-28):
    #   owl_normal_short  prio 1500  1 day   UsageFactor 2  <- bills DOUBLE
    #   owl_normal_base   prio 1000  7 days  UsageFactor 1
    #   owl_normal_long   prio  500  14 days UsageFactor 1
    # Unlike Tinkercliffs, "short" costs 2x allocation here, and these jobs hold
    # dozens of cores for hours -- so default to base, and opt into short with
    # --qos owl_normal_short only when queue time matters more than budget.
    PART=normal_q; GRES=""
    if (( HOURS >= 168 )); then QOS=owl_normal_long; else QOS=owl_normal_base; fi
    if [[ "$(hostname)" != owl* ]]; then
      echo "warning: OWL jobs must be submitted from an OWL login node, not $(hostname)" >&2
    fi
    if (( CPUS > 96 )); then echo "OWL nodes have 96 cores; --cpus $CPUS cannot fit" >&2; exit 2; fi ;;
  *) echo "--gpu must be a100, h200, l40s or owl" >&2; exit 2 ;;
esac
# One account covers every partition, CPU and GPU alike.
ACCOUNT="${RG_ACCOUNT:?set RG_ACCOUNT in $CONF}"
[[ -n "$PART_OVERRIDE" ]] && PART="$PART_OVERRIDE"
[[ -n "$QOS_OVERRIDE" ]] && QOS="$QOS_OVERRIDE"

# Never --mem=0 (whole node: can only start on an idle node, never backfills).
# 2 GB per core covers a highway-env worker with plenty of headroom.
MEM="${MEM:-$(( CPUS * 2 ))G}"

ARGS=(
  --account="$ACCOUNT"
  --partition="$PART"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME"
)
[[ -n "$GRES" ]] && ARGS+=(--gres="$GRES")
[[ -n "$QOS" ]] && ARGS+=(--qos="$QOS")
[[ -n "$ARRAY" ]] && ARGS+=(--array="$ARRAY")
[[ -n "${RG_MAIL_USER:-}" ]] && ARGS+=(--mail-user="$RG_MAIL_USER" --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80)

# SLURM opens the --output path before the job starts; a missing dir fails the job.
mkdir -p logs/slurm

# .slurm files are copied at SUBMIT time, Python at RUN time. Warn if what is
# about to run differs from the pushed branch -- the Mac and ARC checkouts only
# meet through git.
if [[ -n "$(git status --porcelain -- arc scripts algos envs eval configs 2>/dev/null)" ]]; then
  echo "warning: uncommitted changes in this checkout -- is this the code you meant to run?" >&2
fi

echo "sbatch ${ARGS[*]} $SCRIPT $*"
if (( DRY_RUN )); then exit 0; fi
sbatch "${ARGS[@]}" "$SCRIPT" "$@"
