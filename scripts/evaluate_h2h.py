"""Phase 5 -- head-to-head round robin.

    python scripts/evaluate_h2h.py --results results/ --out results/h2h

Runs every matchup the protocol asks for (CLAUDE.md §5):

* all 25 (PPO seed i, GRPO seed j) cross-algorithm pairs -- the headline
* PPO-vs-PPO and GRPO-vs-GRPO cross-seed pairs -- the within-algorithm control
* every agent against the IDM rule-based floor

and emits win rates with bootstrap CIs, a paired permutation test, round-robin
ELO, and the solo->competition degradation table that answers RQ2.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from algos.common.utils import resolve_device  # noqa: E402
from envs import make_h2h_env, make_solo_env  # noqa: E402
from envs.config import EnvConfig, RewardConfig  # noqa: E402
from eval.elo import fit_elo, games_from_matchups  # noqa: E402
from eval.h2h import run_matchup  # noqa: E402
from eval.policies import IDMPolicy, TorchPolicy  # noqa: E402
from eval.stats import degradation, iqm_ci, paired_permutation_test  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("racing-grpo")


def discover_runs(results_dir: Path, algo: str) -> list[Path]:
    """Find finished runs for ``algo`` (directories holding a ``final.pt``)."""

    return sorted(p for p in results_dir.glob(f"{algo}_seed*") if (p / "final.pt").exists())


def load_solo_baselines(run_dirs: list[Path]) -> dict[str, dict]:
    """Read each run's Phase 4 solo eval -- the denominator of the RQ2 metric."""

    baselines = {}
    for run in run_dirs:
        path = run / "solo_eval.json"
        if path.exists():
            baselines[run.name] = json.loads(path.read_text())
        else:
            logger.warning("%s has no solo_eval.json; degradation will be nan", run.name)
    return baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/h2h"))
    parser.add_argument("--pairs", type=int, default=50, help="paired episodes per matchup (x2)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig(n_agents=2, reward=RewardConfig.load())
    probe = make_solo_env(EnvConfig(reward=RewardConfig.load()))
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.shape[0])
    probe.close()

    ppo_runs = discover_runs(args.results, "ppo")
    grpo_runs = discover_runs(args.results, "grpo")
    logger.info("found %d PPO and %d GRPO runs", len(ppo_runs), len(grpo_runs))
    if not ppo_runs or not grpo_runs:
        raise SystemExit("need finished PPO and GRPO runs; complete Phase 4 first")

    policies = {}
    for run in ppo_runs + grpo_runs:
        policies[run.name] = TorchPolicy.load(
            run / "final.pt", obs_dim, act_dim, device, name=run.name
        )
    policies["idm"] = IDMPolicy()

    ppo_names = [r.name for r in ppo_runs]
    grpo_names = [r.name for r in grpo_runs]

    # The matchup list, exactly as the protocol specifies.
    matchup_pairs: list[tuple[str, str]] = []
    matchup_pairs += list(itertools.product(ppo_names, grpo_names))        # 25 headline
    matchup_pairs += list(itertools.combinations(ppo_names, 2))            # PPO cross-seed
    matchup_pairs += list(itertools.combinations(grpo_names, 2))           # GRPO cross-seed
    matchup_pairs += [(n, "idm") for n in ppo_names + grpo_names]          # vs the floor

    env = make_h2h_env(env_cfg)
    results = []
    for i, (name_a, name_b) in enumerate(matchup_pairs, 1):
        matchup = run_matchup(
            policies[name_a], policies[name_b], env_cfg, n_pairs=args.pairs, env=env
        )
        results.append(matchup)
        wr = matchup.win_rate
        logger.info(
            "[%d/%d] %s vs %s | win rate %s", i, len(matchup_pairs), name_a, name_b, wr
        )
    env.close()

    # --- headline: PPO vs GRPO, pooled over the 25 cross-algorithm matchups ---
    cross = [m for m in results if m.name_a in ppo_names and m.name_b in grpo_names]
    ppo_scores = np.concatenate([m.scores for m in cross])

    # Paired test. The unit is one traffic seed within one matchup, and PPO's score
    # on it is averaged over both starting orientations -- so start-position
    # advantage cancels *inside* the unit rather than being averaged over across
    # units. GRPO's score is the complement, since the game is zero-sum.
    normal, swapped = (
        np.concatenate(x) for x in zip(*(m.paired_scores() for m in cross), strict=True)
    )
    ppo_paired = (normal + swapped) / 2.0
    grpo_paired = 1.0 - ppo_paired
    observed, p_value = paired_permutation_test(ppo_paired, grpo_paired)

    per_matchup_wr = np.array([m.win_rate.point for m in cross])
    headline = {
        "ppo_win_rate_vs_grpo": float(ppo_scores.mean()),
        "ppo_win_rate_iqm_over_matchups": str(iqm_ci(per_matchup_wr)),
        "paired_permutation_observed": observed,
        "paired_permutation_p_value": p_value,
        "n_episodes": int(ppo_scores.size),
    }

    # --- RQ2: solo -> competition degradation, each agent vs its own solo eval ---
    baselines = load_solo_baselines(ppo_runs + grpo_runs)
    degradation_rows = []
    for name in ppo_names + grpo_names:
        solo = baselines.get(name, {})
        played = [m for m in results if name in (m.name_a, m.name_b)]
        if not played or not solo:
            continue

        def side(m, name=name):
            return "a" if m.name_a == name else "b"

        collisions = float(
            np.mean([m.summary()[f"{side(m)}_collision_rate"] for m in played])
        )
        offtrack = float(np.mean([m.summary()[f"{side(m)}_offtrack_rate"] for m in played]))
        degradation_rows.append(
            {
                "agent": name,
                "algo": "ppo" if name.startswith("ppo") else "grpo",
                "solo_collision_rate": solo.get("final_eval/collision_rate", float("nan")),
                "h2h_collision_rate": collisions,
                "collision_degradation": degradation(
                    solo.get("final_eval/collision_rate", float("nan")), collisions
                ),
                "solo_offtrack_rate": solo.get("final_eval/offtrack_rate", float("nan")),
                "h2h_offtrack_rate": offtrack,
                "offtrack_degradation": degradation(
                    solo.get("final_eval/offtrack_rate", float("nan")), offtrack
                ),
                "solo_lap_completion": solo.get("final_eval/lap_completion_rate", float("nan")),
            }
        )

    elo = fit_elo(games_from_matchups(results))

    payload = {
        "headline": headline,
        "elo": dict(sorted(elo.items(), key=lambda kv: -kv[1])),
        "degradation": degradation_rows,
        "matchups": [
            {"a": m.name_a, "b": m.name_b, **m.summary()} for m in results
        ],
    }
    (args.out / "h2h_results.json").write_text(json.dumps(payload, indent=2, default=float))

    print("\n=== Headline: PPO vs GRPO ===")
    print(f"PPO win rate: {headline['ppo_win_rate_vs_grpo']:.3f} over {headline['n_episodes']} episodes")
    print(f"paired permutation p = {p_value:.4f}")
    print("\n=== ELO ===")
    for name, rating in payload["elo"].items():
        print(f"  {name:16s} {rating:7.1f}")
    print(f"\nwrote {args.out / 'h2h_results.json'}")


if __name__ == "__main__":
    main()
