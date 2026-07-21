"""Head-to-head evaluation harness (Phase 5).

A *matchup* is 100 paired episodes between two policies. Paired means: for each
traffic seed, the episode is played twice with the agents' starting grid slots
swapped, and the traffic is identical across the swap. Fifty seeds x two
orientations = 100 episodes, and any start-position advantage cancels within a
pair rather than being averaged over and hoped away.

Behavioural metrics (overtakes, blocking, minimum time-to-collision) are measured
here rather than in the env, because they are analyst-defined and must not be able
to influence the reward or the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from envs import make_h2h_env
from envs.config import EnvConfig
from envs.multi_agent import DRAW, LOSS, WIN
from eval.policies import Policy
from eval.stats import Interval, win_rate_ci

# A rival closer than this, for a sustained stretch, counts as an interaction.
INTERACTION_DISTANCE = 25.0
# TTC below this is a near-miss.
NEAR_MISS_TTC = 1.5


@dataclass
class EpisodeRecord:
    """One head-to-head episode, from agent A's point of view."""

    seed: int
    swapped: bool
    result: str  # WIN / LOSS / DRAW for agent A
    a_distance: float
    b_distance: float
    a_crashed: bool
    b_crashed: bool
    a_off_road: bool
    b_off_road: bool
    steps: int
    min_ttc: float
    overtakes_a: int
    overtakes_b: int
    blocking_steps_a: int

    @property
    def score(self) -> float:
        """1.0 win / 0.5 draw / 0.0 loss, for agent A."""

        return {WIN: 1.0, DRAW: 0.5, LOSS: 0.0}[self.result]


@dataclass
class MatchupResult:
    """Aggregated result of one matchup."""

    name_a: str
    name_b: str
    episodes: list[EpisodeRecord] = field(default_factory=list)

    @property
    def scores(self) -> np.ndarray:
        return np.array([e.score for e in self.episodes], dtype=np.float64)

    @property
    def win_rate(self) -> Interval:
        return win_rate_ci(self.scores)

    def paired_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Scores split by orientation, aligned by seed, for the paired test.

        Returns ``(scores_unswapped, scores_swapped)`` over the seeds present in
        both orientations, in a consistent order.
        """

        normal = {e.seed: e.score for e in self.episodes if not e.swapped}
        swapped = {e.seed: e.score for e in self.episodes if e.swapped}
        seeds = sorted(set(normal) & set(swapped))
        return (
            np.array([normal[s] for s in seeds]),
            np.array([swapped[s] for s in seeds]),
        )

    def summary(self) -> dict[str, float]:
        eps = self.episodes
        if not eps:
            return {}
        return {
            "win_rate": self.win_rate.point,
            "win_rate_low": self.win_rate.low,
            "win_rate_high": self.win_rate.high,
            "n_episodes": float(len(eps)),
            "a_collision_rate": float(np.mean([e.a_crashed for e in eps])),
            "b_collision_rate": float(np.mean([e.b_crashed for e in eps])),
            "a_offtrack_rate": float(np.mean([e.a_off_road for e in eps])),
            "b_offtrack_rate": float(np.mean([e.b_off_road for e in eps])),
            "a_mean_distance": float(np.mean([e.a_distance for e in eps])),
            "b_mean_distance": float(np.mean([e.b_distance for e in eps])),
            "mean_min_ttc": float(np.mean([e.min_ttc for e in eps if np.isfinite(e.min_ttc)]))
            if any(np.isfinite(e.min_ttc) for e in eps)
            else float("nan"),
            "near_miss_rate": float(np.mean([e.min_ttc < NEAR_MISS_TTC for e in eps])),
            "overtakes_a": float(np.mean([e.overtakes_a for e in eps])),
            "overtakes_b": float(np.mean([e.overtakes_b for e in eps])),
            "blocking_steps_a": float(np.mean([e.blocking_steps_a for e in eps])),
        }


def time_to_collision(
    pos_a: np.ndarray, vel_a: np.ndarray, pos_b: np.ndarray, vel_b: np.ndarray
) -> float:
    """Time until two point vehicles reach closest approach, if they are closing.

    ``inf`` when they are separating -- the pair poses no imminent collision.
    """

    rel_pos = np.asarray(pos_b) - np.asarray(pos_a)
    rel_vel = np.asarray(vel_b) - np.asarray(vel_a)
    closing = -float(np.dot(rel_pos, rel_vel))
    if closing <= 1e-8:
        return float("inf")
    return float(np.dot(rel_pos, rel_pos) / closing)


def run_episode(
    policy_a: Policy,
    policy_b: Policy,
    env,
    seed: int,
    swapped: bool,
) -> EpisodeRecord:
    """Play one head-to-head episode. ``agent_0`` is A, ``agent_1`` is B."""

    obs, _ = env.reset(seed=seed, options={"swap_starts": swapped})
    # Scripted policies bind to the vehicle they are driving. `swapped` already
    # moved the agent->vehicle mapping, so ask the env which vehicle is whose.
    policy_a.reset(env, env.vehicle_index("agent_0"))
    policy_b.reset(env, env.vehicle_index("agent_1"))

    min_ttc = float("inf")
    overtakes_a = overtakes_b = 0
    blocking_steps_a = 0
    prev_lead: int | None = None
    info: dict = {}
    steps = 0

    while env.agents:
        actions = {
            "agent_0": policy_a.action(obs["agent_0"]),
            "agent_1": policy_b.action(obs["agent_1"]),
        }
        obs, _, _, _, info = env.step(actions)
        steps += 1

        a, b = info["agent_0"], info["agent_1"]
        gap = float(np.linalg.norm(a["position"] - b["position"]))
        ttc = time_to_collision(a["position"], a["velocity"], b["position"], b["velocity"])
        min_ttc = min(min_ttc, ttc)

        # Overtake: the lead (by distance travelled) changes hands while the two
        # are close enough for it to be an actual on-track pass rather than an
        # artefact of one agent lapping alone.
        lead = 0 if a["distance"] > b["distance"] else 1
        if prev_lead is not None and lead != prev_lead and gap < INTERACTION_DISTANCE:
            if lead == 0:
                overtakes_a += 1
            else:
                overtakes_b += 1
        prev_lead = lead

        # Blocking: A leads, B is close behind, and A is slower than B -- i.e. A is
        # in B's way and not simply driving off into the distance.
        if lead == 0 and gap < INTERACTION_DISTANCE and a["speed"] < b["speed"]:
            blocking_steps_a += 1

    a, b = info["agent_0"], info["agent_1"]
    return EpisodeRecord(
        seed=seed,
        swapped=swapped,
        result=a["result"] or DRAW,
        a_distance=a["distance"],
        b_distance=b["distance"],
        a_crashed=a["crashed"],
        b_crashed=b["crashed"],
        a_off_road=a["off_road"],
        b_off_road=b["off_road"],
        steps=steps,
        min_ttc=min_ttc,
        overtakes_a=overtakes_a,
        overtakes_b=overtakes_b,
        blocking_steps_a=blocking_steps_a,
    )


def run_matchup(
    policy_a: Policy,
    policy_b: Policy,
    env_config: EnvConfig,
    n_pairs: int = 50,
    seed_offset: int = 50_000,
    env=None,
) -> MatchupResult:
    """Run ``n_pairs`` paired episodes (2 x n_pairs total) between two policies.

    Traffic seeds are shared across every matchup in the round robin, so every
    pair of agents is compared on the same set of races.
    """

    owned = env is None
    env = env or make_h2h_env(env_config)
    result = MatchupResult(name_a=policy_a.name, name_b=policy_b.name)

    for i in range(n_pairs):
        seed = seed_offset + i
        for swapped in (False, True):
            result.episodes.append(run_episode(policy_a, policy_b, env, seed, swapped))

    if owned:
        env.close()
    return result
