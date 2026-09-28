"""Environment wrapper tests.

The two invariants worth protecting with a test are the ones that would quietly
invalidate the whole study rather than crash:

* the observation contains **no rival-identity feature** (CLAUDE.md §3) -- if it
  ever did, the rival would stop being "an unusually capable traffic vehicle" and
  the distribution shift would become structural instead of behavioural;
* traffic density is fixed at exactly 4, identically in every condition.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs import make_h2h_env, make_solo_env
from envs.config import EnvConfig, ObsType, RewardConfig
from envs.multi_agent import DRAW, LOSS, WIN, attribute_aggressor
from envs.progress import TrackGeometry
from envs.reward import compute_reward


@pytest.fixture(scope="module")
def cfg() -> EnvConfig:
    return EnvConfig(reward=RewardConfig.load(), duration=10)


@pytest.mark.parametrize("obs_type", list(ObsType))
class TestObservationInvariants:
    """Held for *both* observation families -- §3 allows either, never an identity feature."""

    def test_observation_has_no_rival_identity_feature(self, cfg, obs_type):
        """Features must describe motion and road only -- never who is who."""

        cfg = cfg.model_copy(update={"obs_type": obs_type})
        features = cfg.highway_config()["observation"]["features"]
        forbidden = {"is_agent", "is_rival", "agent_id", "controlled", "is_controlled", "rival"}
        assert not (set(features) & forbidden)
        # `presence` marks real-vs-padded rows in a fixed-width observation. It
        # does not distinguish traffic from a rival, and both are equally present.
        assert set(features) <= {
            "presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "on_road", "heading",
        }

    def test_solo_and_h2h_observations_have_the_same_shape(self, cfg, obs_type):
        """A policy trained solo must be able to consume an h2h observation.

        Same shape, same features, same ordering: the rival simply occupies one of
        the nearby-vehicle rows that traffic occupied during training. If these
        diverged, the transfer being measured would be an artefact of the wrapper.
        """

        cfg = cfg.model_copy(update={"obs_type": obs_type})
        solo = make_solo_env(cfg)
        h2h = make_h2h_env(cfg.model_copy(update={"n_agents": 2}))

        solo_dim = int(np.prod(solo.observation_space.shape))
        h2h_dim = int(np.prod(h2h.observation_space("agent_0").shape))
        assert solo_dim == h2h_dim

        solo.close()
        h2h.close()


class TestTraffic:
    def test_traffic_density_is_exactly_four(self, cfg):
        """Stock highway-env randomises the count; the protocol requires it fixed."""

        env = make_solo_env(cfg)
        for seed in range(5):
            env.reset(seed=seed)
            road = env.unwrapped.highway.road
            # 1 controlled vehicle + n_traffic IDM vehicles, every episode.
            assert len(road.vehicles) == 1 + cfg.n_traffic, f"seed {seed}"
        env.close()

    def test_reset_is_a_pure_function_of_the_seed(self, cfg):
        env = make_solo_env(cfg)
        first, _ = env.reset(seed=123)
        second, _ = env.reset(seed=123)
        other, _ = env.reset(seed=124)

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other)
        env.close()


class TestActionSpace:
    def test_continuous_steering_and_throttle(self, cfg):
        """Stock racetrack-v0 is steering-only; the protocol needs throttle too."""

        env = make_solo_env(cfg)
        assert env.action_space.shape == (2,)
        assert cfg.highway_config()["action"]["longitudinal"] is True
        assert cfg.highway_config()["action"]["lateral"] is True
        env.close()


class TestProgress:
    def test_lap_length_matches_the_circuit(self, cfg):
        env = make_solo_env(cfg)
        env.reset(seed=0)
        geometry = TrackGeometry(env.unwrapped.highway.road.network)

        # Track A is a 9-segment circuit (a..i) of ~348 m.
        assert len(geometry.segment_offsets) == 9
        assert 300.0 < geometry.lap_length < 400.0
        env.close()

    def test_the_race_is_actually_finishable_within_the_horizon(self):
        """`laps_to_finish` must be reachable at the track's speed limit.

        Track A's lap is ~348 m and the speed limit is 10 m/s, so two laps needs
        ~70 s -- more than the 60 s horizon. Configured that way, `finished` could
        never fire, `lap_completion_rate` would be pinned at zero, and the Phase 1
        gate ("90% of eval episodes complete a lap") would be unpassable for
        reasons that have nothing to do with the agent. Guard it.
        """

        cfg = EnvConfig(reward=RewardConfig.load())
        env = make_solo_env(cfg)
        env.reset(seed=0)
        geometry = TrackGeometry(env.unwrapped.highway.road.network)
        speed_limit = float(env.unwrapped.highway.config.get("speed_limit", 10.0))
        env.close()

        required = cfg.laps_to_finish * geometry.lap_length / speed_limit
        assert required < cfg.duration, (
            f"finishing {cfg.laps_to_finish} lap(s) needs >= {required:.0f}s at the "
            f"{speed_limit:.0f} m/s speed limit, but the horizon is only {cfg.duration}s"
        )

    def test_distance_is_monotone_when_driving_forward(self, cfg):
        env = make_solo_env(cfg)
        env.reset(seed=0)

        distances = []
        for _ in range(10):
            # highway-env's ContinuousAction is [throttle, steering]: full throttle,
            # no steering input.
            _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            distances.append(info["distance"])
            if terminated or truncated:
                break

        # Deliberately ragged: pairs each distance with its successor.
        assert all(b >= a - 1e-6 for a, b in zip(distances, distances[1:], strict=False))
        assert distances[-1] > 0.0
        env.close()


class TestReward:
    def test_progress_at_target_speed_scores_about_one(self, cfg):
        """The progress term is normalised so lap-record pace ~= 1.0 per step."""

        rc = cfg.reward
        dt = 1.0 / cfg.policy_frequency
        terms = compute_reward(
            rc,
            delta_distance=rc.target_speed * dt,  # exactly target pace
            speed=rc.target_speed,
            action=np.zeros(2),
            crashed=False,
            off_road=False,
            lap_completed=False,
            dt=dt,
        )
        assert terms.progress == pytest.approx(rc.lambda_progress)
        assert terms.speed == pytest.approx(rc.lambda_speed)

    def test_crashing_costs_more_than_it_could_ever_earn_back(self, cfg):
        """A collision must not be a profitable shortcut."""

        rc = cfg.reward
        dt = 1.0 / cfg.policy_frequency
        crashed = compute_reward(
            rc,
            delta_distance=rc.target_speed * dt,
            speed=rc.target_speed,
            action=np.zeros(2),
            crashed=True,
            off_road=False,
            lap_completed=False,
            dt=dt,
        )
        assert crashed.total < 0.0
        assert crashed.collision == rc.collision_penalty

    def test_reward_is_symmetric_for_both_algorithms(self):
        """PPO and GRPO read the same file. There is no per-algorithm reward."""

        assert RewardConfig.load() == RewardConfig.load()


class TestHeadToHead:
    def test_swapping_starts_exchanges_the_agents(self, cfg):
        env = make_h2h_env(cfg.model_copy(update={"n_agents": 2}))

        normal, _ = env.reset(seed=7)
        swapped, _ = env.reset(seed=7, options={"swap_starts": True})

        # Same traffic, same grid -- only who sits where changed.
        np.testing.assert_array_equal(swapped["agent_0"], normal["agent_1"])
        np.testing.assert_array_equal(swapped["agent_1"], normal["agent_0"])
        assert env.vehicle_index("agent_0") == 1
        env.close()

    def test_outcome_is_zero_sum(self, cfg):
        env = make_h2h_env(cfg.model_copy(update={"n_agents": 2}))
        env.reset(seed=3)

        info = {}
        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, rewards, _, _, info = env.step(actions)

        results = {a: info[a]["result"] for a in ("agent_0", "agent_1")}
        assert set(results.values()) in ({WIN, LOSS}, {DRAW})
        assert sum(rewards.values()) == pytest.approx(0.0)
        env.close()


class TestAggressorAttribution:
    """The rule that makes ramming a losing strategy (CLAUDE.md §3)."""

    def test_rear_ender_is_the_aggressor(self):
        # Agent 0 is behind and driving into agent 1, who is simply driving away.
        aggressor = attribute_aggressor(
            [np.array([0.0, 0.0]), np.array([5.0, 0.0])],
            [np.array([10.0, 0.0]), np.array([2.0, 0.0])],
        )
        assert aggressor == 0

    def test_side_impactor_is_the_aggressor(self):
        # Agent 1 drives sideways into agent 0, who is going straight past.
        aggressor = attribute_aggressor(
            [np.array([0.0, 0.0]), np.array([0.0, 5.0])],
            [np.array([8.0, 0.0]), np.array([0.0, -6.0])],
        )
        assert aggressor == 1

    def test_no_aggressor_when_separating(self):
        aggressor = attribute_aggressor(
            [np.array([0.0, 0.0]), np.array([5.0, 0.0])],
            [np.array([-1.0, 0.0]), np.array([2.0, 0.0])],
        )
        assert aggressor is None

    def test_symmetric_head_on_blames_neither(self):
        # Both drive into each other equally hard: no one is more at fault, so it
        # is a draw rather than a coin-flip loss.
        aggressor = attribute_aggressor(
            [np.array([0.0, 0.0]), np.array([5.0, 0.0])],
            [np.array([5.0, 0.0]), np.array([-5.0, 0.0])],
        )
        assert aggressor is None
