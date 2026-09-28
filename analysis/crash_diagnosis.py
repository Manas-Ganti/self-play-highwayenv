"""Why does a trained policy crash? Replay its eval episodes and classify every crash.

    python analysis/crash_diagnosis.py --run results/phase1/p3_occupancy_seed0 --n 50

Pilot #3 raised the cost of a crash roughly 25x and the policy drove *faster* and
crashed as often. That is the signature of a policy that cannot see a crash
coming, not one that is indifferent to crashing. This script tests that directly.

For every crash it records, from the simulator (the policy never sees any of it):

* **type** -- ego ran into a car ahead (rear-end), was hit from behind, or side contact;
* **ego / other speed** at impact;
* **first-visible gap** -- how far ahead the other car was when it first entered
  the occupancy grid (the policy's field of view), and the ego's speed then;
* **stopping gap needed** at that moment: closing speed²/(2·5 m/s²) + one policy
  step of reaction + one car length. highway-env's ContinuousAction brakes at
  most 5 m/s².

If most rear-end crashes have *first-visible gap < stopping gap needed*, the
crash was unavoidable from the observation at that speed: the fix is the field
of view (or a slower policy), not the reward. Replays the same fixed eval seeds
``evaluate_solo`` scores, with the run's own env config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from analysis.record_solo import EVAL_SEED_OFFSET, load_env_config, load_policy  # noqa: E402
from envs import make_solo_env  # noqa: E402
from envs.config import ObsType  # noqa: E402

MAX_BRAKE = 5.0  # m/s^2, highway-env ContinuousAction acceleration_range
CAR_LENGTH = 5.0  # m, highway-env Vehicle.LENGTH
LANE_HALF = 2.5  # m; |lateral| below this counts as "same lane"


def ego_frame(ego, other) -> tuple[float, float]:
    """Other vehicle's position in the ego frame: (ahead, left) in metres."""

    rel = np.asarray(other.position) - np.asarray(ego.position)
    c, s = np.cos(ego.heading), np.sin(ego.heading)
    return float(rel @ np.array([c, s])), float(rel @ np.array([-s, c]))


def stopping_gap(v_ego: float, v_other: float, dt: float) -> float:
    closing = max(v_ego - v_other, 0.0)
    return closing**2 / (2 * MAX_BRAKE) + closing * dt + CAR_LENGTH


def diagnose_episode(env, act, seed: int, grid_x, grid_y, dt: float) -> dict | None:
    """Play one episode; return a crash record, or None if it did not crash."""

    obs, _ = env.reset(seed=seed)
    hw = env.unwrapped.highway
    ego = hw.controlled_vehicles[0]
    first_seen: dict[int, dict] = {}
    info, done, step = {}, False, 0

    while not done:
        # Field of view *before* acting: what the policy could see when it chose.
        for v in hw.road.vehicles:
            if v is ego or id(v) in first_seen:
                continue
            ahead, left = ego_frame(ego, v)
            if grid_x[0] <= ahead <= grid_x[1] and grid_y[0] <= left <= grid_y[1]:
                first_seen[id(v)] = {"step": step, "ahead": ahead, "left": left,
                                     "v_ego": float(ego.speed), "v_other": float(v.speed)}
        obs, _, terminated, truncated, info = env.step(act(obs))
        step += 1
        done = bool(terminated or truncated)

    if not info.get("crashed"):
        return None

    crashed = [v for v in hw.road.vehicles if v is not ego and v.crashed]
    if not crashed:
        return {"seed": seed, "type": "unknown", "distance": float(info["distance"])}
    other = min(crashed, key=lambda v: np.linalg.norm(np.asarray(v.position) - ego.position))
    ahead, left = ego_frame(ego, other)
    if abs(left) < LANE_HALF:
        kind = "rear-end (ego hit car ahead)" if ahead > 0 else "hit from behind"
    else:
        kind = "side contact"

    seen = first_seen.get(id(other))
    rec = {
        "seed": seed,
        "type": kind,
        "distance": round(float(info["distance"]), 1),
        "v_ego_impact": round(float(ego.speed), 1),
        "v_other_impact": round(float(other.speed), 1),
    }
    if seen is not None:
        need = stopping_gap(seen["v_ego"], seen["v_other"], dt)
        rec.update({
            "first_visible_ahead": round(seen["ahead"], 1),
            "v_ego_when_seen": round(seen["v_ego"], 1),
            "stopping_gap_needed": round(need, 1),
            "steps_visible_before_crash": step - seen["step"],
            # Visible only once already inside stopping distance: unavoidable
            # from the observation at that speed.
            "seen_too_late": bool(seen["ahead"] > 0 and seen["ahead"] < need),
        })
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50, help="eval episodes (seeds 10000..)")
    args = parser.parse_args()

    env_cfg = load_env_config(args.run)
    if env_cfg.obs_type is not ObsType.OCCUPANCY:
        print("note: kinematics run -- 'field of view' below uses the grid extents, "
              "which a kinematics policy does not have; read the crash types only")
    env = make_solo_env(env_cfg)
    act = load_policy(args.run, "trained", env)
    dt = 1.0 / env_cfg.policy_frequency

    records = []
    for i in range(args.n):
        rec = diagnose_episode(env, act, EVAL_SEED_OFFSET + i, env_cfg.grid_x, env_cfg.grid_y, dt)
        if rec is not None:
            records.append(rec)
    env.close()

    print(f"\n{args.run}  grid_x={env_cfg.grid_x}  crashes: {len(records)}/{args.n}\n")
    for r in records:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))

    if records:
        kinds: dict[str, int] = {}
        for r in records:
            kinds[r["type"]] = kinds.get(r["type"], 0) + 1
        rear = [r for r in records if r["type"].startswith("rear-end") and "seen_too_late" in r]
        late = sum(r["seen_too_late"] for r in rear)
        print("\nsummary")
        print("  types:", kinds)
        print(f"  mean ego speed at impact {np.mean([r.get('v_ego_impact', np.nan) for r in records]):.1f} m/s"
              f" vs other {np.mean([r.get('v_other_impact', np.nan) for r in records]):.1f} m/s")
        if rear:
            print(f"  rear-end crashes first seen INSIDE stopping distance: {late}/{len(rear)}")
            print("  -> mostly 'too late': perception-limited (widen grid_x / slow down)"
                  if late > len(rear) / 2 else
                  "  -> mostly seen in time: the policy saw the car and still hit it (control/reward)")

    out = args.run / "crash_diagnosis.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
