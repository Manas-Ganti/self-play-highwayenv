# `legacy/` — archived prior project

This directory holds the previous incarnation of this repo: **"Competitive
Self-Play Racetrack RL"**, which asked whether competitive self-play produces
policies that generalise across physics and geometry changes, and whether reward
density moderates that generalisation.

The current `CLAUDE.md` asks a different question (critic-free RL under
competitive distribution shift: PPO vs GRPO), and its §3/§4 requirements are
incompatible with this code — most sharply, the old observation vector carried
explicit `opponent_x` / `opponent_vx` / `opponent_lap_progress` features, and the
new spec forbids any rival-identity feature outright.

Nothing here is imported by the live project, and it is excluded from linting and
the test suite. It is kept rather than deleted because the repo has no git history
to recover it from. **Delete it freely once you are confident nothing is wanted.**

## Contents

| Path | What it was |
|---|---|
| `envs/racetrack_env.py` | PettingZoo two-agent wrapper with opponent-aware observations |
| `envs/config.py` | `EnvConfig` with physics (drag/mass) and geometry knobs |
| `envs/env_factory.py` | `make_env` single entry point |
| `envs/vehicle_models.py` | Configurable drag/mass physics patching |
| `envs/domain_randomiser.py` | Sampled env configs for transfer sweeps |
| `rewards/` | sparse / dense / hybrid reward registry |
| `agents/` | SB3 PPO + SAC wrappers, opponent pool with ELO, self-play loop |
| `experiments/` | Config schema, train/evaluate split, experiment runner |
| `analysis/` | ELO curves, transfer heatmaps, ablation tables |
| `viz/` | Pygame live-render dashboard |
| `tests/` | Tests for the above |

## Ideas worth stealing back

* `agents/opponent_pool.py` — frozen-checkpoint pool with ELO-weighted sampling.
  Directly relevant to the **Phase 7 C3 self-play baseline** (RQ3).
* `viz/live_render.py` — live pygame dashboard; a nicer debugging surface than
  `analysis/video.py`'s offline mp4 export.
* `analysis/plot_transfer.py` — transfer heatmap, if the Phase 7 track-C ablation
  grows past a single number.
