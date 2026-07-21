"""Vehicle subclasses exposing configurable physics.

These are the *only* place the physics axes (drag, mass) are implemented. Every
new physics parameter (tire grip, downforce, ...) is added here and surfaced
through :class:`envs.config.VehicleConfig` and the YAML — never hardcoded in
experiment code, and never instantiated directly outside the env factory.
"""

from __future__ import annotations

import logging

try:  # highway-env is an optional import so config/reward code stays testable.
    from highway_env.vehicle.kinematics import Vehicle
except ModuleNotFoundError:  # pragma: no cover - exercised only without the dep
    Vehicle = object  # type: ignore[assignment,misc]

logger = logging.getLogger("racetrack_rl")


class ParametrisedVehicle(Vehicle):  # type: ignore[misc]
    """Vehicle with configurable drag coefficient and mass scaling.

    Parameters
    ----------
    road : highway_env.road.road.Road
        The road the vehicle is placed on.
    position : numpy.ndarray
        Initial ``[x, y]`` position.
    heading : float
        Initial heading in radians.
    speed : float
        Initial speed.
    drag : float
        Aerodynamic drag coefficient. Higher values decelerate the vehicle
        faster at speed. Default 0.4.
    mass_scale : float
        Multiplier on the base vehicle mass. Affects acceleration response.
        Default 1.0.
    """

    def __init__(
        self,
        road,
        position,
        heading: float = 0.0,
        speed: float = 0.0,
        drag: float = 0.4,
        mass_scale: float = 1.0,
    ) -> None:
        super().__init__(road, position, heading, speed)
        self.drag = drag
        self.mass_scale = mass_scale

    def step(self, dt: float) -> None:
        """Advance the vehicle by ``dt`` seconds applying speed-squared drag.

        The drag deceleration scales inversely with mass, so heavier vehicles
        coast further. The base class then integrates the modified acceleration.
        """

        drag_force = self.drag * self.speed**2
        self.action["acceleration"] -= drag_force / self.mass_scale
        super().step(dt)


def patch_vehicle_physics(vehicle, drag: float, mass_scale: float) -> None:
    """Apply parametrised physics to an *already constructed* highway-env vehicle.

    highway-env builds its controlled vehicles internally (via the road network
    and behaviour config), so subclassing alone does not guarantee our class is
    used. This helper rebinds the drag/mass attributes and the ``step`` method
    onto an existing instance, giving the factory a flexible second path that
    works regardless of how the vehicle was created.

    Parameters
    ----------
    vehicle : highway_env.vehicle.kinematics.Vehicle
        The vehicle instance to parametrise in place.
    drag : float
        Drag coefficient to apply.
    mass_scale : float
        Mass multiplier to apply.
    """

    import types

    vehicle.drag = drag
    vehicle.mass_scale = mass_scale
    base_step = vehicle.step

    def _drag_step(self, dt: float) -> None:
        drag_force = self.drag * self.speed**2
        self.action["acceleration"] -= drag_force / self.mass_scale
        base_step(dt)

    vehicle.step = types.MethodType(_drag_step, vehicle)
