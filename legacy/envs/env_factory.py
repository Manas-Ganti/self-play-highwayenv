"""The single entry point for environment construction.

Every environment in the project — training, evaluation, sweeps, viz — is built
here via :func:`make_env`. Nothing else instantiates ``racetrack-v0``,
``CompetitiveRacetrackEnv``, ``ParametrisedVehicle``, or a reward class. This
function owns config validation, reward injection, and (optionally) wrapping the
parallel env for single-agent training against a fixed opponent.
"""

from __future__ import annotations

import logging
from typing import Any

from envs.config import EnvConfig
from envs.racetrack_env import CompetitiveRacetrackEnv
from rewards.registry import make_reward

logger = logging.getLogger("racetrack_rl")


def make_env(config: dict[str, Any] | EnvConfig) -> CompetitiveRacetrackEnv:
    """Build a competitive racetrack environment from a config.

    Parameters
    ----------
    config : dict or EnvConfig
        Raw config (validated into :class:`~envs.config.EnvConfig`) or an
        already-validated config object.

    Returns
    -------
    CompetitiveRacetrackEnv
        A PettingZoo parallel environment with the reward function injected.

    Examples
    --------
    >>> env = make_env({"vehicle_drag": 0.7, "track_width": 8, "reward": "sparse"})
    """

    cfg = _coerce_config(config)
    reward_fn = make_reward(cfg.reward, **cfg.reward_params)
    logger.info(
        "make_env: reward=%s drag=%.2f mass=%.2f track_width=%.1f "
        "road_length=%.0f game=%s action=%s seed=%d",
        cfg.reward,
        cfg.vehicle.drag,
        cfg.vehicle.mass_scale,
        cfg.track_width,
        cfg.road_length,
        cfg.game_type.value,
        cfg.action_type.value,
        cfg.seed,
    )
    return CompetitiveRacetrackEnv(cfg, reward_fn)


def _coerce_config(config: dict[str, Any] | EnvConfig) -> EnvConfig:
    """Validate a raw config into :class:`EnvConfig`.

    Accepts both the flat ``{"vehicle_drag": ...}`` form used in CLI overrides
    and the nested ``{"vehicle": {"drag": ...}}`` form, so callers can use
    whichever is convenient.
    """

    if isinstance(config, EnvConfig):
        return config

    data = dict(config)
    # Flatten the convenience keys used in CLAUDE.md examples into the nested
    # schema EnvConfig expects.
    vehicle = dict(data.pop("vehicle", {}) or {})
    if "vehicle_drag" in data:
        vehicle["drag"] = data.pop("vehicle_drag")
    if "vehicle_mass_scale" in data:
        vehicle["mass_scale"] = data.pop("vehicle_mass_scale")
    if vehicle:
        data["vehicle"] = vehicle

    return EnvConfig.model_validate(data)
