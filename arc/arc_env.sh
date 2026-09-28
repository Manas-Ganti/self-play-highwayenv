# Sourced by every arc/*.slurm launcher. Not executable on its own.
#
# Each block below exists because the same failure cost a GPU allocation on
# another project on this cluster (see arc/README.md, "Why the launchers look
# like this"). The rule throughout: fail loudly in the first second of the job,
# never silently twenty minutes in.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# --- Interpreter: absolute path, never `conda activate` -----------------------
# `module load Miniforge3` swaps in the cluster's conda, which cannot see a
# personal ~/miniconda3 root; `source activate` can then report success without
# switching interpreters. So: no activation at all, just an absolute path.
RG_ENV="${RG_ENV:-$HOME/miniconda3/envs/racing-grpo}"
export PY="$RG_ENV/bin/python"
export PATH="$RG_ENV/bin:$PATH"

# A zero-byte interpreter "runs" every script as an empty shell script and exits
# 0 -- a job that completes in 2 s with an empty .err. A real CPython always
# prints a version, so this is the check that cannot be fooled.
PY_VERSION="$("$PY" -V 2>&1 || true)"
if [[ "$PY_VERSION" != Python\ 3.* ]]; then
  echo "[arc_env] FATAL: '$PY -V' printed '${PY_VERSION}'. Broken or missing env." >&2
  echo "[arc_env] run arc/setup_env.sh on a login node, or set RG_ENV." >&2
  exit 3
fi
"$PY" - <<'EOF'
import importlib.util, sys
missing = [m for m in ("torch", "gymnasium", "highway_env", "stable_baselines3", "pydantic")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"[arc_env] FATAL: {sys.executable} is missing {missing}")
EOF
echo "[arc_env] python=$PY ($PY_VERSION)"

# --- GPU jobs must actually get the GPU -----------------------------------------
# `device: auto` falls back to CPU when CUDA is unusable, and a torch build newer
# than the node's driver (cu130 needs driver >= 580) is unusable *silently*: the
# job runs, just on the CPU, and a "GPU" profile measures the wrong thing. So a
# job that was allocated a GPU asserts torch can use it.
if [[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]]; then
  "$PY" - <<'EOF'
import sys, torch
if not torch.cuda.is_available():
    import subprocess
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        drv = "unknown (no nvidia-smi)"
    sys.exit(f"[arc_env] FATAL: GPU allocated but torch {torch.__version__} (CUDA {torch.version.cuda}) "
             f"cannot use it; node driver {drv}. Rebuild the env with a matching TORCH_CUDA "
             "(arc/setup_env.sh header).")
print(f"[arc_env] cuda ok: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
EOF
fi

# --- Threads: the env workers own the cores ------------------------------------
# Every AsyncVectorEnv worker is its own process, and torch in each of them
# would otherwise spawn one thread per *node* core. On a 128-core DGX that is
# thousands of threads fighting over the handful of cores SLURM gave us.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# --- Headless rendering --------------------------------------------------------
# highway-env renders through pygame in software. No RT cores are involved (the
# Isaac Sim constraint does not apply here), so video export works on any node.
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MPLBACKEND=Agg

# --- Logging -------------------------------------------------------------------
# /projects/$USER is not writable on ARC; W&B's scratch must live in $HOME.
export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"
mkdir -p "$WANDB_DIR" logs/slurm results
# Secrets (WANDB_API_KEY, optional TELEGRAM_*) live outside the repo. Without a
# W&B key the Logger falls back to TensorBoard under results/.
SECRETS_ENV="${SECRETS_ENV:-$HOME/.config/vrr/secrets.env}"
# shellcheck disable=SC1090
[[ -f "$SECRETS_ENV" ]] && source "$SECRETS_ENV"
if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "[arc_env] W&B on"; else echo "[arc_env] W&B off -> TensorBoard"; fi

echo "[arc_env] job=${SLURM_JOB_ID:-local} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} gpu=${CUDA_VISIBLE_DEVICES:-none} commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"

# --- Optional Telegram ping on exit --------------------------------------------
# No-op without a token; every curl ends in `|| true` so a notification failure
# can never fail a run; the trap re-returns the job's real exit code.
arc_notify() {
  [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]] || return 0
  curl -s -m 10 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" >/dev/null || true
}
arc_finish() {
  local rc=$?
  arc_notify "racing-grpo ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-} on $(hostname): exit ${rc}"
  return $rc
}
trap arc_finish EXIT
