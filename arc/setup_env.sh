#!/usr/bin/env bash
# One-time install of the project env on a VT ARC login node (no GPU needed).
#
#   arc/setup_env.sh            # create + install + verify
#   arc/setup_env.sh --verify   # verify only
#
# A dedicated env: never reuse or merge vrr / vrr-train / rtn. Their torch and
# numpy pins are not this project's, and crossing envs fails at import, deep
# inside a job. /home is shared between Tinkercliffs (A100/H200) and Falcon
# (L40S), so this one env serves all three GPU types.

set -euo pipefail

RG_ENV="${RG_ENV:-$HOME/miniconda3/envs/racing-grpo}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
PY="$RG_ENV/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

verify() {
  echo "== verify $RG_ENV"
  # A real CPython always prints its version; a 0-byte binary prints nothing.
  "$PY" -V || { echo "FATAL: interpreter broken"; exit 3; }
  "$PY" - <<'EOF'
import gymnasium, highway_env, numpy, pettingzoo, stable_baselines3, torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("gymnasium", gymnasium.__version__, "| highway-env", highway_env.__version__,
      "| sb3", stable_baselines3.__version__, "| numpy", numpy.__version__)
# torch.cuda.is_available() is False on a login node; that is expected.
EOF
  "$PY" -m pip check || echo "(pip check reported conflicts above -- read them)"
  # Pure logic, no GPU: collector determinism, advantage math, env invariants.
  SDL_VIDEODRIVER=dummy MPLBACKEND=Agg "$PY" -m pytest -q
}

if [[ "${1:-}" == "--verify" ]]; then verify; exit 0; fi

[[ -x "$CONDA" ]] || { echo "no conda at $CONDA; set CONDA=/path/to/bin/conda" >&2; exit 2; }

# Login-node /tmp is ~20 GB and the torch wheel unpacks large; build in $HOME.
export TMPDIR="$HOME/tmp/racing-grpo-install"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

if [[ ! -x "$PY" ]]; then
  "$CONDA" create -y -p "$RG_ENV" python=3.11
fi

# torch first, from the CUDA 12.1 index, so the later requirements install keeps
# it instead of pulling a CPU or mismatched-CUDA wheel from PyPI.
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install -r requirements.txt

# CLAUDE.md §2: pin exact versions after the first successful install. Commit
# this file from the ARC checkout so the Mac and ARC agree on what ran.
"$PY" -m pip freeze --exclude-editable > arc/requirements.lock.txt
echo "wrote arc/requirements.lock.txt -- commit it"

verify
