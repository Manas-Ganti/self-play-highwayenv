"""Single-agent SB3 view over the two-agent parallel environment.

SB3 trains one policy against a Gymnasium ``Env``; our environment is a
PettingZoo parallel env with two simultaneous agents. This adapter exposes the
*learner's* slot as a standard single-agent env while a frozen opponent drawn
from the :class:`~agents.opponent_pool.OpponentPool` drives the other slot.

The opponent is selected once per episode (at ``reset``) and never updated mid
episode, satisfying the self-play rules in CLAUDE.md. ELO is updated by the
caller at episode end using the ``info["won"]`` / ``info["opponent"]`` it reads
back from the terminal step.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover
    gym = None  # type: ignore[assignment]

from agents.opponent_pool import FrozenPolicy, OpponentPool
from envs.racetrack_env import CompetitiveRacetrackEnv

logger = logging.getLogger("racetrack_rl")


class SelfPlaySingleAgentEnv(gym.Env if gym else object):  # type: ignore[misc]
    """Wrap a parallel racetrack env as a single-agent env for SB3.

    Parameters
    ----------
    parallel_env : CompetitiveRacetrackEnv
        The two-agent environment to drive.
    pool : OpponentPool
        Source of frozen opponents, sampled once per episode.
    learner_id : str
        Which agent slot the learning policy controls (default ``"agent_0"``).
    learner_elo_fn : callable
        Returns the learner's current ELO, used for skill-proximity sampling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        parallel_env: CompetitiveRacetrackEnv,
        pool: OpponentPool,
        learner_id: str = "agent_0",
        learner_elo_fn=lambda: 1000.0,
    ) -> None:
        self.env = parallel_env
        self.pool = pool
        self.learner_id = learner_id
        self.opponent_id = next(
            a for a in parallel_env.possible_agents if a != learner_id
        )
        self.learner_elo_fn = learner_elo_fn

        self.observation_space = parallel_env.observation_space(learner_id)
        self.action_space = parallel_env.action_space(learner_id)

        self._current_opponent: FrozenPolicy | None = None
        self._last_obs: dict[str, np.ndarray] = {}

    @property
    def current_opponent(self) -> FrozenPolicy | None:
        """The frozen opponent active for the current episode (for ELO updates)."""

        return self._current_opponent

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._current_opponent = self.pool.sample_opponent(self.learner_elo_fn())
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_obs = observations
        return observations[self.learner_id], infos[self.learner_id]

    def step(self, action: Any):
        opp_obs = self._last_obs[self.opponent_id]
        assert self._current_opponent is not None
        opp_action, _ = self._current_opponent.predict(opp_obs, deterministic=True)

        joint = {self.learner_id: action, self.opponent_id: opp_action}
        observations, rewards, terminations, truncations, infos = self.env.step(joint)

        done = terminations.get(self.learner_id, False)
        truncated = truncations.get(self.learner_id, False)
        if not done and not truncated:
            self._last_obs = observations
            obs = observations[self.learner_id]
        else:
            # Episode finished; agents list is cleared, reuse the last features.
            obs = self._last_obs.get(
                self.learner_id, np.zeros(self.observation_space.shape, np.float32)
            )

        info = dict(infos.get(self.learner_id, {}))
        info["opponent_elo"] = self._current_opponent.elo
        return obs, rewards.get(self.learner_id, 0.0), done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
