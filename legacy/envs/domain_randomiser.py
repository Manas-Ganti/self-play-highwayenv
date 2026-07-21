"""Samples environment configs from a parameter distribution.

Supports the "domain randomisation as training strategy" research direction:
instead of training on one fixed environment, draw a fresh :class:`EnvConfig`
each episode from a declared distribution. The same machinery underpins a
curriculum scheduler (subclass and widen the ranges over training).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from envs.config import EnvConfig, VehicleConfig

logger = logging.getLogger("racetrack_rl")


@dataclass
class ParamRange:
    """Inclusive sampling range for one continuous parameter."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class DomainRandomiser:
    """Draws randomised :class:`EnvConfig` instances around a base config.

    Parameters
    ----------
    base : EnvConfig
        Config providing the values for any parameter not being randomised.
    drag : ParamRange | None
        Range for ``vehicle.drag``; ``None`` leaves it fixed at the base value.
    mass_scale : ParamRange | None
        Range for ``vehicle.mass_scale``.
    track_width : ParamRange | None
        Range for ``track_width``.
    road_length : ParamRange | None
        Range for ``road_length``.
    seed : int
        Seed for the sampler RNG (kept independent of the env seed).
    """

    base: EnvConfig
    drag: ParamRange | None = None
    mass_scale: ParamRange | None = None
    track_width: ParamRange | None = None
    road_length: ParamRange | None = None
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(self) -> EnvConfig:
        """Return a new :class:`EnvConfig` with randomised parameters."""

        vehicle = VehicleConfig(
            drag=self.drag.sample(self._rng) if self.drag else self.base.vehicle.drag,
            mass_scale=(
                self.mass_scale.sample(self._rng)
                if self.mass_scale
                else self.base.vehicle.mass_scale
            ),
        )
        overrides = {
            "vehicle": vehicle,
            "track_width": (
                self.track_width.sample(self._rng)
                if self.track_width
                else self.base.track_width
            ),
            "road_length": (
                self.road_length.sample(self._rng)
                if self.road_length
                else self.base.road_length
            ),
            # New independent episode seed.
            "seed": int(self._rng.integers(0, 2**31 - 1)),
        }
        cfg = self.base.model_copy(update=overrides)
        logger.debug("DomainRandomiser sampled: %s", overrides)
        return cfg
