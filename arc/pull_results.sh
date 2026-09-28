#!/usr/bin/env bash
# Run on the Mac: copy results from the ARC checkout into this one.
#
#   arc/pull_results.sh                  # everything under results/ (no checkpoints)
#   arc/pull_results.sh --with-ckpt      # also *.pt (needed for local h2h / video)
#
# results/ is gitignored -- runs are reproduced from config + seed, never
# committed -- so rsync is the only path from ARC to the Mac. Nothing from a
# pulled run goes on the website or resume until its solo_eval.json / h2h output
# is actually on disk here (the portfolio's proof rule).

set -euo pipefail

ARC_HOST="${ARC_HOST:-tinkercliffs1.arc.vt.edu}"
ARC_USER="${ARC_USER:-$USER}"
ARC_REPO="${ARC_REPO:-/home/$ARC_USER/ondemand/data/self-play-highwayenv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXCLUDES=(--exclude 'wandb/' --exclude 'launch_logs/')
[[ "${1:-}" == "--with-ckpt" ]] || EXCLUDES+=(--exclude '*.pt' --exclude '*.zip')

rsync -avz "${EXCLUDES[@]}" "${ARC_USER}@${ARC_HOST}:${ARC_REPO}/results/" "$REPO_ROOT/results/"
rsync -avz "${ARC_USER}@${ARC_HOST}:${ARC_REPO}/logs/slurm/" "$REPO_ROOT/logs/slurm/"
