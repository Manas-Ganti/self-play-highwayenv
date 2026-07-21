"""Reward function interface.

Every reward implements :class:`BaseReward`. Rewards are pure functions of a
single agent's transition; they never touch the simulator, write files, or hold
training state. This keeps them swappable via the registry and trivially
testable.
"""

from __future__ import annotations

from typing import Any


class BaseReward:
    """Abstract reward callable.

    A reward maps one agent's transition to a scalar. Implementations should be
    bounded and side-effect free.
    """

    def __call__(
        self,
        agent_id: str,
        obs: dict[str, Any],
        action: int,
        next_obs: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """Return the scalar reward for ``agent_id``'s transition.

        Parameters
        ----------
        agent_id : str
            The agent the reward is computed for.
        obs : dict
            Observation before the action (per-agent feature dict).
        action : int
            Discrete action taken.
        next_obs : dict
            Observation after the action.
        info : dict
            Auxiliary signals from the environment step (``crashed``, ``won``,
            ``lap_progress``, ``opponent_lap_progress``, ``speed``, ...).
        """

        raise NotImplementedError
