# Project log

Chronological record of phase gates, decisions, and protocol amendments.
Every phase gate gets an entry here and a commit (CLAUDE.md §5, §8).

---

## 2026-07-13 — Repository restructured onto the new CLAUDE.md spec

The repo previously implemented a **different project**: "Competitive Self-Play
Racetrack RL" (SB3 SAC/PPO, self-play from step one, discrete meta-actions,
domain-randomised physics, geometry/physics transfer sweeps). The new spec asks a
different research question, so the code was rebuilt to §6's layout rather than
patched.

The old modules are archived under `legacy/` (not deleted — this repo has no git
history to recover them from). See `legacy/README.md` for what each one was.

### Why almost nothing was reusable

| Old design | New spec | Consequence |
|---|---|---|
| Observation had explicit `opponent_x`, `opponent_vx`, `opponent_lap_progress` features | "Never add a rival-identity feature" (§3) | Observation rebuilt from scratch; this one is load-bearing, not cosmetic |
| Self-play from step one | Solo training, rival only at eval | Training env is now single-agent; PettingZoo is eval-only |
| SB3 SAC + PPO wrappers | Custom PyTorch PPO + GRPO sharing `algos/common/` | New algorithm layer; SB3 retained only for Phase 1/2 validation |
| Discrete meta-actions | `ContinuousAction` (steering + throttle) | New action space |
| Reward registry (sparse/dense/hybrid), swept as a research axis | One frozen dense reward in `configs/reward.yaml` | Registry dropped; reward is now a constant of the experiment, not a variable |
| Physics/geometry domain randomisation | Fixed traffic (4), fixed track A | Randomisers dropped; track C is a Phase 7 ablation only |

### Environment decisions worth flagging (§0 asks for disagreements to be noted)

1. **Traffic density.** Stock `RacetrackEnv._make_vehicles` spawns
   `rng.integers(other_vehicles)` traffic cars — a *random* count in `[0, N)` — and
   then silently *drops* any that fail its spawn-separation check. Both would make
   traffic density an uncontrolled confound between PPO and GRPO runs. Overridden
   in `envs/tracks.py`: exactly 4 vehicles, resampled rather than dropped.
2. **Throttle.** Stock `racetrack-v0` ships `ContinuousAction(longitudinal=False)`
   — steering only, speed fixed. That would make lap time a non-decision and gut
   the racing problem. Enabled both axes.
3. **Track C** is highway-env's `racetrack-large-v0` rather than bespoke geometry:
   held-out and genuinely different, with no new geometry code to get wrong.
4. **`laps_to_finish` = 1, not 2.** Track A's lap is 348 m and the speed limit is
   10 m/s, so two laps needs ~70 s against a 60 s horizon — `finished` could never
   fire, `lap_completion_rate` would be pinned at 0, and the Phase 1 gate would be
   unpassable for reasons unrelated to the agent. Guarded by
   `tests/test_env.py::test_the_race_is_actually_finishable_within_the_horizon`.
5. **Aggressor attribution.** Closing *speed* is symmetric between two vehicles —
   it is a property of the pair — so it cannot name a rammer on its own. Attribution
   compares each agent's own velocity component *toward* the other; the one
   contributing more of the approach is the aggressor. A pair that was not closing
   is a draw, not a coin-flip loss. To be validated against video in Phase 5.

### Bugs caught while wiring this up

- **Action order.** highway-env's `ContinuousAction` is `[throttle, steering]`, not
  `[steering, throttle]`. A learned policy is indifferent (it learns whatever
  mapping it is handed), so this was invisible until the scripted IDM floor drove
  off the track within 40 m on every episode. The floor now completes a clean lap.
- **GAE across truncation.** Terminations and truncations are stored separately;
  a time-limit cutoff bootstraps off `V(final_obs)` instead of being treated as a
  terminal state with value 0. Tested in `tests/test_advantages.py`.

## 2026-09-28 — Moved execution to VT ARC; Phase 1 script added

Training moves from "a single A100 server" to VT ARC (A100/H200 on Tinkercliffs,
L40S on Falcon), one GPU per job. `arc/` holds the launchers; `arc/README.md` is
the runbook. No training has run yet.

- **`scripts/sb3_pilot.py` (new).** Phase 1 had no script. It runs SB3 PPO with the
  custom PPO's hyperparameters mapped one-to-one, and scores the result with the
  same `evaluate_solo` harness and eval seeds, so the Phase 2 parity comparison is
  like-for-like. It prints and writes the gate verdict. A local 8k-step smoke run
  confirmed the pipeline works end to end. That run was not a result, and its
  output was deleted.
- **`launch_condition.sh` core pinning.** The script read `os.cpu_count()`, which
  under SLURM is the *node's* core count (128 on a DGX), not the job's allocation,
  so `taskset` would have pinned runs to cores the job does not own. It now splits
  `sched_getaffinity(0)` and inherits the job's `CUDA_VISIBLE_DEVICES`.
- **HP search as a SLURM array.** `--trial-index i` runs one trial, and `--collect`
  aggregates the results and writes the budget receipt. `--collect` refuses to run
  while any trial is missing.
- **Bug fixed in `hp_search.sample_trial` (before any search ran).** Shared and
  algorithm-specific hyperparameters were drawn from one RNG stream. PPO makes 4
  algorithm-specific draws per trial and GRPO makes 1, so from trial 1 onward the
  two algorithms got *different* shared values. The docstring's "trial i is a
  matched pair" was therefore false. The grid and trial count were still matched,
  but the pairing was not. The two blocks now draw from separate streams, and
  `tests/test_hp_search.py` asserts identical shared values on every trial.
  No protocol amendment is needed because no search had run.
- **OWL CPU nodes added** (`arc/submit.sh --gpu owl`). The workload is CPU-bound,
  so 3.8 GHz Genoa cores may beat any GPU node. The OWL-vs-A100 profile decides.
  OWL uses partition `normal_q`, and jobs default to QOS `owl_normal_base`.
  `owl_normal_short` has UsageFactor 2, so it bills double (see `arc/README.md`).

## 2026-09-28 — OWL throughput profile (Phase 0 re-profile)

OWL job 968475, node owl017, 32 cores, CPU policy. Full table in
`report/throughput_owl.md`.

| envs | steps/s | vs 1 env |
|---:|---:|---:|
| 1 | 173 | 1.0× |
| 8 | 980 | 5.7× |
| 16 | 1,082 | 6.2× |
| 32 | 1,621 | 9.4× |

- One OWL core is about 1.6× faster than the laptop (173 vs 109 steps/s), and the
  16-env rate is about 2.8× the laptop's.
- Scaling flattens after 8 envs, and the job still had spare cores, so the limit
  is the main process (policy forward plus waiting on the slowest env each step),
  not the core count. `n_envs` is not changed, because it sets the rollout batch
  shape and is an algorithm setting. Revisit only if wall time becomes a problem.
- Estimate: a 2M-step PPO run takes about 1 h (31 min collection, about 12 min
  periodic eval, plus updates). `arc/README.md` sizes jobs at about 3–4× that.
- The report's `cpu cores: 96` line was wrong: it showed the node total, not the
  job's 32. `profile_env.py` now reports the allocation.

---

## Phase status

- [x] **Phase 0 — Scaffold.** (Re-profiled on OWL 2026-09-28; see above.) Repo at §6 layout; 61 tests green; throughput
      profiled (`report/throughput.md`).
      *Gate: PASSED* — `pytest` green on collector determinism + advantage math;
      `scripts/profile_env.py` reports ~390 steps/s (8 envs, 8-core laptop).
      Re-profile on the ARC node type before Phase 4 (`arc/README.md`).
- [ ] **Phase 1 — Env validation via SB3 PPO pilot.** Script ready
      (`scripts/sb3_pilot.py`); not yet run.
      *Gate:* ≥90% of eval episodes complete a lap without collision within 2M steps.
      **`configs/reward.yaml` is provisional until this gate passes, then frozen.**
- [ ] **Phase 2 — Custom PPO parity vs SB3.** Not started. Do not build GRPO until
      this passes.
- [ ] **Phase 3 — GRPO + matched HP search.** Code is in place
      (`algos/grpo.py`, `scripts/hp_search.py`); the search has not been run.
- [ ] **Phase 4 — Full solo runs.** Not started.
- [ ] **Phase 5 — Head-to-head evaluation.** Harness in place; needs Phase 4 runs.
- [ ] **Phase 6 — Analysis + report.** Not started.
- [ ] **Phase 7 — Stretch.** Not started.

## Protocol amendments

- **2026-09-28: compute platform is OWL CPU-only (amends §1).** §1 assumed one
  A100 with the policy on GPU. Only one partition's resources are available, and
  env stepping dominates the wall time, so every run uses OWL `normal_q` CPU
  nodes with the policy on CPU. **No effect on validity:** PPO and GRPO run on
  identical hardware and devices, and step budgets are counted in env steps, not
  wall time. Max 96 cores per job; a GRPO condition (5 × 33 cores) runs as two
  jobs split by seed (`launch_condition.sh <algo> <n> <steps> <first_seed>`).

*(Reward: none yet. `configs/reward.yaml` has not been frozen — that happens at the Phase 1
gate. Any change to it after that point must be recorded here.)*
