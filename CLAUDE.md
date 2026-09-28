# CLAUDE.md — Critic-Free RL Under Competitive Distribution Shift
## PPO vs GRPO on highway-env Racetracks

This file is the authoritative spec for this project. Read it fully before writing code. Work phase by phase; do not start a phase until the previous phase's acceptance criteria pass. When a design decision here conflicts with something you'd prefer, follow this document and flag the disagreement in a comment.

---

## 0. Project Summary

Train PPO and GRPO agents **independently** on solo racetrack driving (with IDM traffic), then evaluate them **head-to-head** on a shared track neither experienced as multi-agent. GRPO here means group-relative advantage estimation without a critic, ported from LLM fine-tuning into dense-reward continuous control — nearly unpopulated territory empirically.

**RQ1:** Does GRPO match PPO in-distribution (solo) at matched compute?
**RQ2:** Which algorithm degrades more gracefully under competitive distribution shift?
**RQ3 (stretch):** How much head-to-head performance comes from opponent exposure vs algorithm? (PPO-selfplay upper baseline.)

Pre-registered hypotheses:
- H1: GRPO reaches ≥90% of PPO solo final performance at matched env steps.
- H2: GRPO shows larger solo→competition degradation than PPO.
- H3: PPO-selfplay beats both solo agents head-to-head.

Negative results are results. If GRPO fails, the deliverable pivots to diagnosing *why* (see GRPO-t variant, §4).

---

## 1. Hardware & Execution Model

Target machine: VT ARC **OWL CPU nodes** (`normal_q`: 96 Genoa cores at 3.8 GHz, 768 GB per node). Only one partition's resources are available, and this workload is CPU-bound, so no GPU is used (protocol amendment 2026-09-28 in `report/log.md`). Max 96 cores per job; a condition needing more is split across jobs by seed. All submission goes through `arc/submit.sh`; `arc/README.md` is the operational runbook. Nothing here is multi-GPU.

**Critical performance fact:** highway-env stepping is pure-Python and CPU-bound. The policy networks are small MLPs. The A100 is therefore *not* the bottleneck — CPU env throughput is. Exploit the GPU by running **many training runs concurrently**, not by making one run faster:

- Use `AsyncVectorEnv` (Gymnasium) or SB3 `SubprocVecEnv` with 16–32 parallel envs per run, tuned to available cores.
- Run all 5 seeds of a condition as concurrent processes sharing one GPU (each uses a sliver of GPU memory; small MLPs coexist trivially). `scripts/launch_condition.sh` spawns them on the job's GPU with disjoint CPU affinity drawn from the SLURM allocation; on ARC it runs inside `arc/condition.slurm`. Size `--cpus` at ~`n_envs + 1` per concurrent run.
- The HP search runs one trial per SLURM array task (`arc/search.slurm`); trials are pre-sampled from the pre-registered seed, so the array and a sequential loop search identical configs.
- Policy tensor ops run on CPU (`device: auto` with no GPU allocated); env stepping on CPU workers. Profile on OWL and record steps/sec in `report/throughput_owl.md`.

Budgets (upgraded for this hardware):
- 2M env steps per solo run (pilot decides whether curves plateau; extend to 3M only if not).
- 5 seeds per condition.
- Hyperparameter search: 30 trials random search per algorithm on seed 0, identical search budget for PPO and GRPO (this matching is a validity requirement, not a suggestion).

---

## 2. Environment Setup

```bash
conda create -n racing-grpo python=3.11 -y
conda activate racing-grpo
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install highway-env gymnasium pettingzoo stable-baselines3 \
            numpy pandas matplotlib seaborn rliable wandb \
            moviepy imageio opencv-python-headless pyyaml pytest
```

Pin exact versions in `requirements.txt` after first successful install. Log everything to Weights & Biases (project `racing-grpo`); fall back to TensorBoard if no W&B key is available — check for `WANDB_API_KEY` and degrade gracefully.

---

## 3. Environment Design (do not deviate without flagging)

**Base:** highway-env `racetrack-v0`, continuous actions (`ContinuousAction`: steering, throttle).

**Observation:** kinematics-list or occupancy-grid observation including all nearby vehicles, with **no feature distinguishing traffic from a rival agent**. This is the load-bearing design decision: at eval time the rival must appear as "an unusually capable traffic vehicle," so the distribution shift is behavioral, not structural. Never add a rival-identity feature.

**Traffic:** IDM-controlled, fixed density (4 vehicles), identical across all training and evaluation. Seeded deterministically per episode.

**Tracks:** Track A = training and primary eval. Track C = held-out geometry for the transfer ablation only. Both algorithms train on the same track A.

**Reward (solo, identical for PPO and GRPO):** dense lap-progress + speed shaping − terminal collision penalty − off-track penalty. Weights live in `configs/reward.yaml` and are frozen after Phase 1.

**Multi-agent wrapper:** PettingZoo Parallel API over highway-env with two controlled vehicles. Needed only for evaluation (Phase 5) and the C3 self-play baseline.

**Win condition (head-to-head):** progress-based — most lap progress at timeout or first to finish N laps. Agent–agent collision is terminal for both; the initiating agent (closing-velocity attribution: rear-end or side-impact with positive closing speed) is scored the loss. Agent–traffic collision is terminal for that agent only. This exists specifically to kill ramming strategies — verify with videos in Phase 5.

---

## 4. Algorithms

### 4.1 Custom PPO (`algos/ppo.py`)
Standard PPO with GAE, clipped surrogate, value loss, entropy bonus — written as a custom PyTorch loop sharing `algos/common/` (rollout buffer, nets, vectorized collector) with GRPO. SB3 PPO is used once, in Phase 1, purely to validate this custom loop.

### 4.2 GRPO for continuous control (`algos/grpo.py`) — the core contribution

**Group:** G = 8 rollouts launched from the **same reset state** (same track segment, same traffic init, same agent pose), differing only in policy stochasticity. Implement via a vectorized collector where G env copies share a reset seed per group. This same-seed reset machinery is the fiddliest engineering in the project — build and test it in isolation first (`tests/test_group_collector.py`: assert identical initial observations within a group, divergence after first stochastic action).

**Advantage:** episodic return R_i per rollout; A_i = (R_i − mean_group) / (std_group + 1e-8); broadcast A_i to every timestep of rollout i.

**Loss:** PPO clipped surrogate with broadcast A_i + entropy bonus. **No value network, no value loss, no KL-to-reference term** (clipping is the trust region; there is no reference policy in from-scratch RL — leave a comment saying so).

**Honest positioning (put this in the report):** this variant ≈ RLOO with clipping and multi-sample groups, evaluated in dense-reward continuous control. Log per-update gradient variance for both algorithms to quantify the group-baseline vs GAE variance gap.

**Diagnostic variant GRPO-t (build only if base GRPO underperforms H1):** per-timestep group-relative advantages on discounted returns-to-go, normalized across the group at each t. Isolates credit assignment failure from critic-absence failure.

---

## 5. Phase Plan with Acceptance Gates

Work strictly in order. Each phase ends with its acceptance check passing and a short entry in `report/log.md`.

**Phase 0 — Scaffold.** Repo structure (§6), configs, W&B wiring, `pytest` running, throughput profile script.
✅ *Gate:* `pytest` green on collector unit tests; `python scripts/profile_env.py` reports steps/sec.

**Phase 1 — Env validation via SB3 PPO pilot.** One SB3 PPO run on track A with traffic.
✅ *Gate:* reliable lap completion (≥90% of eval episodes complete a lap without collision) within 2M steps. If this fails after reasonable tuning, simplify the track/traffic and re-run — do not proceed on an unlearnable env. Freeze `configs/reward.yaml` at gate pass.

**Phase 2 — Custom PPO parity.** Custom loop vs SB3, 3 seeds each, same HPs.
✅ *Gate:* final-performance IQMs overlap within bootstrap CIs and learning curves are visually indistinguishable. **Do not build GRPO until this passes** — every later comparison inherits this loop's correctness.

**Phase 3 — GRPO + matched HP search.** Group collector, GRPO loss, then 30-trial random search for *each* algorithm on seed 0 over the pre-registered grid in `configs/search_space.yaml`.
✅ *Gate:* GRPO trains without divergence on best HPs; search budgets provably identical (same trial count, same grid dimensions where shared).

**Phase 4 — Full solo runs.** 5 seeds × {PPO, GRPO}, 2M steps, best HPs. Launch concurrently per §1.
✅ *Gate:* all 10 runs complete; solo eval suite (50 episodes/seed, deterministic policy) logged.

**Phase 5 — Head-to-head evaluation.** All (PPO seed i, GRPO seed j) pairs = 25 matchups; plus PPO-vs-PPO and GRPO-vs-GRPO cross-seed matchups; plus each vs the IDM rule-based floor. Each matchup: 100 paired episodes, 50 per starting position, traffic seeds identical across the position swap.
Metrics: win rate + bootstrap CI (primary); ELO over the full round robin; **solo→competition degradation** in lap time / collision rate / off-track rate relative to each agent's own Phase 4 solo eval (this is the RQ2 metric); overtakes, blocking events, min time-to-collision distributions. Record 20 videos per headline matchup.
✅ *Gate:* paired-episode permutation test computed; degradation table produced; videos reviewed for ramming (if present, tighten aggressor attribution and re-run).

**Phase 6 — Analysis + report.** rliable IQM plots, hypothesis verdicts, behavioral analysis (emergent blocking/defense — neither agent saw a rival in training, so any such behavior is repurposed traffic avoidance), 6–8 page preprint-style writeup in `report/`.

**Phase 7 (stretch, cut freely) —** C3 self-play baseline for RQ3; ablations in priority order: group size G ∈ {4, 8, 16}; GRPO-t if triggered; collision-penalty Pareto sweep; track-C geometry transfer.

---

## 6. Repository Layout

```
racing-grpo/
├── CLAUDE.md                  # this file
├── requirements.txt
├── envs/                      # gym wrapper, PettingZoo wrapper, track configs, traffic seeding
├── algos/
│   ├── common/                # nets, rollout buffer, vectorized + group collectors
│   ├── ppo.py
│   └── grpo.py
├── configs/                   # reward.yaml, search_space.yaml, one YAML per condition/seed
├── arc/                       # VT ARC: submit.sh, arc_env.sh, *.slurm, setup_env.sh, README.md
├── scripts/
│   ├── profile_env.py
│   ├── sb3_pilot.py           # Phase 1 gate: SB3 PPO env validation
│   ├── train.py               # entry: python scripts/train.py --config configs/ppo_seed0.yaml
│   ├── launch_condition.sh    # spawn all seeds concurrently
│   └── evaluate_h2h.py
├── eval/                      # h2h harness, ELO, paired permutation test
├── analysis/                  # rliable plots, behavioral tagging, video export
├── tests/                     # collector determinism, advantage math, env wrapper
└── report/                    # log.md, throughput.md, preprint draft
```

---

## 7. Statistical Rules (non-negotiable)

All aggregate claims: IQM + stratified bootstrap 95% CIs across seeds (rliable). Win rates: paired-episode permutation tests. Never report single-seed curves as findings. Never compare PPO and GRPO at unequal step or HP-search budgets.

---

## 8. Guardrails for Claude Code

- Do not swap in a different GRPO formulation (e.g., adding a learned baseline) — that destroys the research question. Variants go in separate files behind config flags.
- Do not "fix" GRPO underperformance by giving it extra tuning trials; the matched budget *is* the experiment.
- Do not add rival-identity features to observations, ever.
- Do not change `configs/reward.yaml` after Phase 1 without recording it as a protocol amendment in `report/log.md`.
- Prefer boring, testable code over clever abstractions; every algorithmic function gets a unit test with hand-computed expected values (especially the group advantage normalization).
- Commit at every phase gate with the gate result in the commit message.