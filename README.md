# Critic-Free RL Under Competitive Distribution Shift
## PPO vs GRPO on highway-env racetracks

Train PPO and GRPO **independently** on solo racetrack driving (with IDM traffic),
then evaluate them **head-to-head** on a track neither experienced as multi-agent.
GRPO here means group-relative advantage estimation *without a critic*, ported from
LLM fine-tuning into dense-reward continuous control — nearly unpopulated territory
empirically.

- **RQ1** Does GRPO match PPO in-distribution (solo) at matched compute?
- **RQ2** Which algorithm degrades more gracefully under competitive distribution shift?
- **RQ3** How much head-to-head performance comes from opponent exposure vs algorithm?

[`CLAUDE.md`](CLAUDE.md) is the authoritative spec — this README is a map, not a
substitute. [`report/log.md`](report/log.md) tracks phase gates and amendments.

---

## The load-bearing design decision

The observation contains **no feature distinguishing IDM traffic from a rival
agent**. At evaluation the rival must appear as "an unusually capable traffic
vehicle", so the distribution shift is *behavioural*, not *structural*. Everything
downstream depends on this, so it is enforced by test
(`tests/test_env.py::TestObservationInvariants`) and there is no config knob that
can turn it off.

## Quickstart

```bash
conda create -n racing-grpo python=3.11 -y && conda activate racing-grpo
pip install torch --index-url https://download.pytorch.org/whl/cu121   # A100 node
pip install -r requirements.txt

pytest                                          # 61 tests
python scripts/profile_env.py                   # -> report/throughput.md

# Train one run (W&B if WANDB_API_KEY is set, else TensorBoard)
python scripts/train.py --config configs/ppo_seed0.yaml
python scripts/train.py --config configs/grpo_seed0.yaml

# All 5 seeds of a condition, concurrently on one GPU
./scripts/launch_condition.sh ppo
./scripts/launch_condition.sh grpo

# Matched-budget hyperparameter search (Phase 3)
python scripts/hp_search.py --algo ppo
python scripts/hp_search.py --algo grpo

# Head-to-head round robin + degradation table (Phase 5)
python scripts/evaluate_h2h.py --results results/
python analysis/video.py --a results/ppo_seed0 --b results/grpo_seed0 --n 20

# Figures + hypothesis verdicts (Phase 6)
python analysis/plots.py --results results/
```

## Architecture

| Layer | Module | Role |
|---|---|---|
| Env | `envs/config.py` | Typed `EnvConfig`; renders the highway-env config |
| | `envs/solo.py` | **Training env.** 1 agent + 4 IDM cars, dense reward |
| | `envs/multi_agent.py` | **Eval only.** PettingZoo 2-agent race, win condition, aggressor attribution |
| | `envs/tracks.py` | Track A / C; fixed traffic density; swappable grid slots |
| | `envs/progress.py` | Lap progress via arc-length on the road graph |
| | `envs/reward.py` | The dense reward. Weights frozen in `configs/reward.yaml` |
| Algos | `algos/common/` | Nets, buffers, collectors, logging — **shared by both algorithms** |
| | `algos/ppo.py` | Custom PPO: GAE, clipped surrogate, value loss |
| | `algos/grpo.py` | **Custom GRPO: group-relative advantages, no critic** |
| Eval | `eval/h2h.py` | Paired-episode head-to-head harness |
| | `eval/stats.py` | IQM, bootstrap CIs, paired permutation tests (§7) |
| | `eval/policies.py` | `Policy` protocol: learned nets and the IDM floor |
| | `eval/elo.py` | Order-independent round-robin ELO |
| Analysis | `analysis/plots.py` | rliable IQM figures, H1/H2 verdicts |
| | `analysis/video.py` | Video export + the Phase 5 **ramming** review |

## The two algorithms differ in exactly two places

Both share `algos/common/` — the same nets, the same rollout path, the same env,
the same reward, the same step accounting. The differences are:

1. **The advantage estimator.** PPO: GAE over a learned value function. GRPO:
   `A_i = (R_i − mean(R_group)) / (std(R_group) + ε)`, one scalar per episode
   broadcast to all its timesteps, where the group's G rollouts start from an
   *identical* reset state and differ only by policy stochasticity.
2. **The value loss.** PPO has one. GRPO has no critic at all — no value network,
   no value loss, and no KL-to-reference term (there is no reference policy in
   from-scratch RL; clipping is the trust region).

Anything else that differed between them would be a confound.

## Invariants (CLAUDE.md §8)

- Never add a rival-identity feature to the observation.
- Never give GRPO a learned baseline — that *is* the research question. Variants
  (e.g. GRPO-t) go in separate files behind config flags.
- Never give one algorithm more HP-search trials than the other; the matched
  budget is the experiment, and `scripts/hp_search.py` refuses to violate it.
- Never change `configs/reward.yaml` after the Phase 1 gate without logging a
  protocol amendment in `report/log.md`.
- Never report a single-seed curve as a finding: IQM + bootstrap CI across 5 seeds.

## Status

Phase 0 complete (scaffold, tests, throughput profile). Phases 1–7 not started —
`report/log.md` has the gate checklist and the design decisions worth arguing with.

`legacy/` holds the archived prior project ("Competitive Self-Play Racetrack RL", a
different research question). Nothing live imports it; delete it when you're sure.
