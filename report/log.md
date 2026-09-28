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

## 2026-09-28 — Phase 1 pilot #1: FAILED (kinematics observation)

SB3 PPO, seed 0, 2M steps, track A, `obs_type: kinematics`. OWL, 35 min wall.
W&B run `qqdbcyk0`; `results/phase1/sb3_ppo_seed0/solo_eval.json` on ARC.

| final eval (50 ep) | value |
|---|---|
| lap completion | **0.00** (gate ≥ 0.90) |
| off-track | 0.98 |
| collision | 0.02 |
| mean speed | ~15.8 m/s (during training evals) |
| mean distance | ~65 m of a 348 m lap |

**Diagnosis: the policy learned "full throttle, straight ahead".** A scripted
throttle-0.3, zero-steer controller leaves the road at 77.8 m at 15.4 m/s. That
is the same place and speed as the trained agent. The lap opens with a 58 m
straight (`a→b`) and then a curve (`b→c`); the agent never takes the first corner.

**Root cause: the kinematics observation carries no road information.** The
ego row is `[presence, x, y, vx, vy, cos_h, sin_h]` with x/y absolute and
normalised by ±200 m, so 1 m of lateral drift moves the input by 0.005. There is
no lane offset, no heading-to-lane error, and no upcoming curvature. Steering
would require memorising the track from a vanishing position signal. Speed rose
through training (evals: 10 → 15.8 m/s) while distance did not, because progress
reward is maximised by driving the straight as fast as possible.

**Next:** `obs_type: occupancy`, which §3 already allows. It is highway-env's own
racetrack observation: 4 layers (presence, vx, vy, **on_road**) on a 12×12 grid
of ±18 m, aligned to the vehicle. It is rival-agnostic (no identity layer) and
has the same flattened shape in the solo and h2h envs (576). The invariant tests
now run for both observation families. Env iteration before the gate is within
protocol (§5 Phase 1: "simplify... and re-run"). Neither algorithm has trained
yet, so this is not a protocol amendment. The reward is unchanged.

### Watching episodes

`analysis/record_solo.py --run <run dir>` replays a run on its own eval seeds and
writes stamped mp4s, uploading them to W&B with `--wandb`. `--policy straight`
replays the zero-steer reference on the same seeds.

**Rendering bug, found by the black-frame guard on ARC:** highway-env's
`EnvViewer` sets `enabled = False` whenever `SDL_VIDEODRIVER == "dummy"`, and then
draws nothing. Every frame comes out black, on Linux and macOS alike, with no
error. `arc_env.sh` set `dummy` for every job, so all headless video (including
the Phase 5 ramming review) would have been blank. Fix: SDL's `offscreen` driver,
plus highway-env `offscreen_rendering: True` whenever an env is built with
`render_mode="rgb_array"`. `tests/test_env.py::TestRendering` asserts non-black
frames for the solo and h2h envs. The recorder still refuses to write all-black
frames.

Open item: `analysis/video.py` (Phase 5 h2h) builds its env from the *default*
config (kinematics). If the gate passes on occupancy, it must read the run's
`config.json` the way `record_solo.py` does, or h2h will feed occupancy-trained
policies the wrong observation.

## 2026-09-28 — Phase 1 pilots #2a/#2b: FAILED, but occupancy fixed steering

SB3 PPO, 2M steps, `obs_type: occupancy`, two seeds, reward unchanged from #1.
W&B runs `1t62mcey` (seed 0) and `zykjm5kt` (seed 1).

| final eval (50 ep) | kinematics #1 | occupancy s0 | occupancy s1 | IDM floor |
|---|---|---|---|---|
| lap completion | 0.00 | 0.48 | 0.72 | **0.98** (49/50) |
| collision | 0.02 | 0.30 | 0.26 | 0.02 |
| off-track | 0.98 | 0.22 | 0.04 | 0.00 |

The IDM floor was run locally on the same 50 eval seeds (10000–10049) with
highway-env 1.11. ARC has 1.12.1, so treat it as indicative. Its one crash was
into a vehicle ahead at 5 m/s.

**Reading.** The occupancy observation fixed the problem: the policy now takes
corners (pilot #1 took none). The remaining failure is collisions, and the IDM
result shows they are avoidable (90% is reachable). The policy drives about 16.6
m/s (seed 0 final eval) through traffic at 6–9 m/s.

**Cause: the reward made that trade correct.** A lap's progress sums to about
174 whatever the speed (348 m / (10 m/s × 0.2 s)). The crash penalty was 5, there
was no finish reward, and γ = 0.99 discounting favours reaching reward sooner.
A crash at 290 m forfeited only ~34 points, while speed raised the value of every
point. The eval curves oscillated with no trend (lap rate 0.3–0.8 across
training evals), consistent with a weak, noisy signal about crashing.

**Change (pre-freeze reward iteration, within §5 Phase 1):**
`collision_penalty` 5 → 20, `offtrack_penalty` 0.5 → 20, `lap_bonus` 0 → 100.
The rationale is in `configs/reward.yaml`. The penalties stay below the ~32
points earned reaching the first corner, so "stand still" never beats driving
early in training. PPO and GRPO both read this one file, so the comparison stays
symmetric. Next: pilot #3, occupancy, same two seeds.

## 2026-09-28 — Phase 1 pilot #3: FAILED; the reward change did not help

SB3 PPO, occupancy (±18 m grid), revised reward (collision/offtrack 20,
lap_bonus 100). W&B `dcw4fmkd` (seed 0) and `dokh7c42` (seed 1).

| final eval (50 ep) | #2 s0 / s1 | #3 s0 / s1 |
|---|---|---|
| lap completion | 0.48 / 0.72 | 0.58 / 0.60 |
| collision | 0.30 / 0.26 | 0.30 / 0.36 |
| off-track | 0.22 / 0.04 | 0.12 / 0.06 |
| mean speed, last training eval | 16.6 / 12.4 | 18.2 / 14.1 |

A crash now costs at least 120 (20 penalty plus the forfeited 100 bonus), about
25× the old cost, yet speed rose and the collision rate did not fall. A policy
that could see crashes coming would respond to that price, so this one probably
cannot see them.

**Field-of-view check.** The grid sees 18 m ahead. Braking at 5 m/s² from 18 m/s
to traffic speed (~7 m/s) needs about 19 m of gap (v²/2a, plus one policy step,
plus one car length). New tool: `analysis/crash_diagnosis.py` replays eval
episodes and, for each crash, records the first-visible gap against the stopping
gap needed. A deliberately reckless lane-keeping driver (IDM, 18 m/s target, no
time gap) was run for 20 seeds per grid, locally. **This measures how far each
grid can see on track A, not the trained policy:**

| grid (ahead × lateral) | obs dim | median first-visible gap (rear-ends) | seen too late |
|---|---|---|---|
| ±18 × ±18 | 576 | 17.3 m | 14/16 |
| −12…48 × ±18 | 960 | 22.6 m | 2/16 |
| −12…48 × ±30 | 1600 | 28.4 m | 2/16 |
| −18…54 × ±36 | 2304 | 41.7 m | 2/16 |

Forward range alone gains less than expected. Track A's bends are tight, so a
car further along the road sits off to the side in the ego frame, and lateral
range matters as much as forward range.

**Next.** (1) Run `crash_diagnosis.py` on the two #3 policies (ARC) to confirm
that the *trained* agent's crashes are seen-too-late. (2) In parallel, pilot #4:
`grid_x=[-12, 48]`, `grid_y=[-30, 30]`, reward unchanged from #3, so the grid is
the only difference. If (1) shows the crashes were seen in time, #4 is
cancelled, and the problem is control or reward rather than perception.
`EnvConfig` grid fields default to the old ±18 m grid, so saved runs replay
unchanged. The h2h env uses the same `highway_config`, so solo and h2h stay
identical.

---

## Phase status

- [x] **Phase 0 — Scaffold.** (Re-profiled on OWL 2026-09-28; see above.) Repo at §6 layout; 61 tests green; throughput
      profiled (`report/throughput.md`).
      *Gate: PASSED* — `pytest` green on collector determinism + advantage math;
      `scripts/profile_env.py` reports ~390 steps/s (8 envs, 8-core laptop).
      Re-profile on the ARC node type before Phase 4 (`arc/README.md`).
- [ ] **Phase 1 — Env validation via SB3 PPO pilot.** Pilot #1 (kinematics)
      0.00; #2 (occupancy) 0.48 / 0.72; #3 (+ revised reward) 0.58 / 0.60; #4 (wider grid) pending.
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
