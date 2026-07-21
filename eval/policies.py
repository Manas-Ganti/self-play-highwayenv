"""Policy adapters for the evaluation harness.

The head-to-head round robin has to pit learned networks against each other *and*
against a scripted rule-based floor, so it talks to a single :class:`Policy`
protocol rather than to ``ActorCritic`` directly. Everything the harness races --
PPO, GRPO, self-play, IDM -- implements the same two-method interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from algos.common.nets import ActorCritic


@runtime_checkable
class Policy(Protocol):
    """Anything that can drive a car in the evaluation harness."""

    name: str

    def reset(self, env, agent_index: int) -> None:
        """Called at episode start; scripted policies bind to the env here."""

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Map one observation to a continuous ``[throttle, steering]`` action."""


class TorchPolicy:
    """A trained PPO or GRPO network, wrapped for evaluation."""

    def __init__(
        self,
        model: ActorCritic,
        device: torch.device,
        name: str = "policy",
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.name = name
        self.deterministic = deterministic

    def reset(self, env, agent_index: int) -> None:  # noqa: D102 - stateless
        return None

    @torch.no_grad()
    def action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32).ravel(), device=self.device
        ).unsqueeze(0)
        action, _, _ = self.model.act(obs_t, deterministic=self.deterministic)
        return action.squeeze(0).cpu().numpy()

    @classmethod
    def load(
        cls,
        path: str | Path,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        name: str | None = None,
        deterministic: bool = True,
    ) -> TorchPolicy:
        """Load a checkpoint written by ``PPO.save`` / ``GRPO.save``.

        The critic is not rebuilt: evaluation only ever needs the actor, and a
        GRPO checkpoint has no critic weights to rebuild it from anyway.
        """

        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = ActorCritic(
            obs_dim,
            act_dim,
            hidden=tuple(cfg.get("hidden_sizes", (64, 64))),
            log_std_init=float(cfg.get("log_std_init", -0.5)),
            with_critic=any(k.startswith("critic.") for k in ckpt["model"]),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return cls(model, device, name=name or Path(path).stem, deterministic=deterministic)


# highway-env's ContinuousAction is [throttle, steering], both in [-1, 1], mapped
# linearly onto these physical ranges. A scripted controller has to emit normalised
# values, so it needs the ranges to divide by. (A *learned* policy is indifferent to
# the convention -- it simply learns whatever mapping it is given -- which is exactly
# why getting this backwards is invisible until a hand-written controller drives
# straight off the track.)
ACCELERATION_RANGE = 5.0  # m/s^2 at |throttle| = 1
STEERING_RANGE = np.pi / 4  # rad at |steering| = 1

# Lane-following gains, ported from highway_env.vehicle.controller.ControlledVehicle.
TAU_HEADING = 0.2
TAU_LATERAL = 0.6
TAU_PURSUIT = 0.5 * TAU_HEADING
KP_HEADING = 1.0 / TAU_HEADING
KP_LATERAL = 1.0 / TAU_LATERAL
MAX_STEERING_ANGLE = np.pi / 3


class IDMPolicy:
    """Rule-based floor: follow the lane centre at a target speed, yield to the car ahead.

    This is the "can either agent even beat a scripted driver" reference, so it is
    deliberately competent-but-dumb: it drives the racing line and keeps a
    following distance, and it never overtakes, blocks, or defends.

    It ignores the observation and reads the simulator directly, which is exactly
    why it is a *floor* rather than a competitor -- it is not solving the
    partially-observed problem the learned policies are.

    The steering law is ported from highway-env's own ``ControlledVehicle``
    (lateral-offset -> heading reference -> heading-rate -> steering angle) rather
    than hand-tuned, so the floor's competence reflects the simulator's own
    controller and not this file's guesswork.
    """

    def __init__(
        self,
        target_speed: float = 9.0,
        name: str = "idm",
        throttle_gain: float = 1.0,
        safe_time_gap: float = 1.5,
    ) -> None:
        self.name = name
        self.target_speed = target_speed
        self.throttle_gain = throttle_gain
        self.safe_time_gap = safe_time_gap
        self._vehicle = None
        self._road = None

    def reset(self, env, agent_index: int) -> None:
        # Both the solo and head-to-head envs expose `.highway`; scripted policies
        # are allowed to read the simulator, learned ones are not.
        highway = env.unwrapped.highway
        self._vehicle = highway.controlled_vehicles[agent_index]
        self._road = highway.road

    def action(self, obs: np.ndarray) -> np.ndarray:
        vehicle = self._vehicle
        if vehicle is None:
            raise RuntimeError("IDMPolicy.reset(env, agent_index) must be called first")

        steering_angle = self._steering_control(vehicle)
        acceleration = self.throttle_gain * (self._safe_speed(vehicle) - vehicle.speed)

        # Normalise into the [-1, 1] action space, in [throttle, steering] order.
        return np.clip(
            [acceleration / ACCELERATION_RANGE, steering_angle / STEERING_RANGE],
            -1.0,
            1.0,
        ).astype(np.float32)

    def _steering_control(self, vehicle) -> float:
        """Steering angle [rad] that steers the vehicle back to its lane centre."""

        lane = vehicle.lane
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        speed = _not_zero(vehicle.speed)

        # Aim at where the lane will be, not where the vehicle is: pure pursuit.
        lane_future_heading = lane.heading_at(longitudinal + vehicle.speed * TAU_PURSUIT)

        lateral_speed_command = -KP_LATERAL * lateral
        heading_command = np.arcsin(np.clip(lateral_speed_command / speed, -1.0, 1.0))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)

        heading_rate = KP_HEADING * _wrap_angle(heading_ref - vehicle.heading)
        slip_angle = np.arcsin(
            np.clip(vehicle.LENGTH / 2 / speed * heading_rate, -1.0, 1.0)
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        return float(np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))

    def _safe_speed(self, vehicle) -> float:
        """Target speed, cut back to hold a time gap to the vehicle ahead."""

        front, _ = self._road.neighbour_vehicles(vehicle, vehicle.lane_index)
        if front is None:
            return self.target_speed

        gap = float(np.linalg.norm(front.position - vehicle.position)) - vehicle.LENGTH
        desired_gap = max(self.safe_time_gap * vehicle.speed, 5.0)
        if gap < desired_gap:
            return float(np.clip(front.speed * gap / desired_gap, 0.0, self.target_speed))
        return self.target_speed


def _not_zero(x: float, eps: float = 1e-2) -> float:
    """Guard a divisor that goes to zero when the vehicle is stopped."""

    return x if abs(x) > eps else eps


def _wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi] so a heading error never takes the long way round."""

    return float((angle + np.pi) % (2 * np.pi) - np.pi)
