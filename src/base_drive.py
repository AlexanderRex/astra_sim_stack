"""Development point-to-point base control; no simulator imports or extra dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class Goal:
    x: float
    y: float
    yaw: float
    name: str = "point"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ValueError("Goal coordinates and yaw must be finite")


@dataclass(frozen=True)
class DriveConfig:
    position_gain: float = 0.8
    yaw_gain: float = 1.2
    position_integral_gain: float = 0.5
    yaw_integral_gain: float = 0.8
    integral_speed_limit: float = 0.03
    integral_yaw_rate_limit: float = 0.08
    max_speed: float = 0.2
    max_yaw_rate: float = 0.35
    max_acceleration: float = 0.3
    max_yaw_acceleration: float = 0.6
    position_tolerance: float = 0.005
    yaw_tolerance: float = math.radians(0.5)
    stopped_speed: float = 0.015
    stopped_yaw_rate: float = 0.025
    settle_seconds: float = 0.6
    waypoint_timeout: float = 60.0
    stall_seconds: float = 8.0
    progress_distance: float = 0.005
    brake_timeout: float = 3.0
    wall_timeout: float = 180.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("Controller settings must be positive and finite")
        if self.waypoint_timeout <= self.settle_seconds:
            raise ValueError("waypoint_timeout must exceed settle_seconds")


@dataclass(frozen=True)
class BaseState:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float
    seconds: float

    def __post_init__(self):
        for name, value in vars(self).items():
            object.__setattr__(self, name, float(value))
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Base telemetry must be finite")
        if self.seconds < 0:
            raise ValueError("Simulation time must be nonnegative")

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)


class DriveToPoint:
    """Planar feedback controller. All deadlines/dwell times use measured sim time."""

    def __init__(self, goals: list[Goal], config: DriveConfig | None = None):
        if not goals:
            raise ValueError("At least one waypoint is required")
        self.goals = goals
        self.cfg = config or DriveConfig()
        self.index = 0
        self.status = "DRIVE"
        self.reason = ""
        self.reached: list[dict] = []
        self.previous_time: float | None = None
        self.started: float | None = None
        self.progress_time: float | None = None
        self.best_error = math.inf
        self.settling_since: float | None = None
        self.command = np.zeros(3, dtype=np.float64)
        self.integral = np.zeros(3, dtype=np.float64)

    @property
    def goal(self):
        return self.goals[min(self.index, len(self.goals) - 1)]

    def fail(self, reason: str):
        self.status, self.reason = "FAILED", reason
        self.command[:] = 0

    def errors(self, state: BaseState):
        return (
            math.hypot(self.goal.x - state.x, self.goal.y - state.y),
            abs(wrap_angle(self.goal.yaw - state.yaw)),
        )

    def stopped(self, state: BaseState):
        return state.speed <= self.cfg.stopped_speed and abs(state.wz) <= self.cfg.stopped_yaw_rate

    def update(self, state: BaseState) -> np.ndarray:
        if self.status in ("DONE", "FAILED"):
            return np.zeros(3)
        cfg = self.cfg
        if self.previous_time is not None and state.seconds <= self.previous_time:
            self.fail("nonadvancing_simulation_time")
            return self.command.copy()
        dt = 0.0 if self.previous_time is None else state.seconds - self.previous_time
        self.previous_time = state.seconds
        if self.started is None:
            self.started = self.progress_time = state.seconds
        distance, yaw_error = self.errors(state)
        # Convert angular progress to a distance-like metric for stall detection.
        metric = distance + 0.25 * yaw_error
        progress_distance = min(cfg.progress_distance, cfg.position_tolerance * 0.1,
                                0.025 * cfg.yaw_tolerance)
        if metric < self.best_error - progress_distance:
            self.best_error, self.progress_time = metric, state.seconds
        # Enter braking well inside the acceptance region. Keep braking while
        # inside the full tolerance, avoiding chatter at its outer boundary.
        capture = 1.0 if self.status == "SETTLE" else 0.4
        inside = (distance <= capture * cfg.position_tolerance
                  and yaw_error <= capture * cfg.yaw_tolerance)
        # A slow, stalled approach already within acceptance may brake without
        # reaching the preferred inner capture margin. Dwell still requires
        # measured stopping and remaining inside both outer tolerances.
        accepted = (distance <= 0.8 * cfg.position_tolerance
                    and yaw_error <= 0.8 * cfg.yaw_tolerance)
        if (accepted and self.stopped(state)
                and state.seconds - self.progress_time >= cfg.settle_seconds):
            inside = True
        if state.seconds - self.started >= cfg.waypoint_timeout:
            self.fail("waypoint_timeout")
        elif not inside and state.seconds - self.progress_time >= cfg.stall_seconds:
            self.fail("no_measured_progress")
        if self.status == "FAILED":
            return self.command.copy()

        if inside:
            self.status = "SETTLE"
            # Brake immediately at the command boundary; the env still applies its smoothing.
            self.command[:] = 0
            if self.stopped(state):
                if self.settling_since is None:
                    self.settling_since = state.seconds
                if state.seconds - self.settling_since >= cfg.settle_seconds:
                    self.reached.append({"name": self.goal.name, "seconds": state.seconds,
                                         "position_error": distance, "yaw_error": yaw_error})
                    self.index += 1
                    self.integral[:] = 0
                    self.settling_since = None
                    self.started = self.progress_time = state.seconds
                    self.best_error = math.inf
                    self.status = "DONE" if self.index == len(self.goals) else "DRIVE"
            else:
                self.settling_since = None
            return self.command.copy()

        self.status, self.settling_since = "DRIVE", None
        position_error = np.array([self.goal.x - state.x, self.goal.y - state.y])
        signed_yaw_error = wrap_angle(self.goal.yaw - state.yaw)
        if distance <= 0.4 * cfg.position_tolerance:
            self.integral[:2] = 0
        elif distance < 0.08:
            self.integral[:2] += cfg.position_integral_gain * position_error * dt
            magnitude = np.linalg.norm(self.integral[:2])
            if magnitude > cfg.integral_speed_limit:
                self.integral[:2] *= cfg.integral_speed_limit / magnitude
        else:
            self.integral[:2] = 0
        if yaw_error <= 0.4 * cfg.yaw_tolerance:
            self.integral[2] = 0
        elif yaw_error < 0.2:
            self.integral[2] = np.clip(self.integral[2] + cfg.yaw_integral_gain * signed_yaw_error * dt,
                                       -cfg.integral_yaw_rate_limit, cfg.integral_yaw_rate_limit)
        else:
            self.integral[2] = 0
        world = cfg.position_gain * position_error + self.integral[:2]
        norm = np.linalg.norm(world)
        if norm > cfg.max_speed:
            world *= cfg.max_speed / norm
        if distance <= 0.4 * cfg.position_tolerance:
            world[:] = 0
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        desired = np.array([c * world[0] + s * world[1], -s * world[0] + c * world[1],
                            np.clip(cfg.yaw_gain * signed_yaw_error + self.integral[2],
                                    -cfg.max_yaw_rate, cfg.max_yaw_rate)])
        if yaw_error <= 0.4 * cfg.yaw_tolerance:
            desired[2] = 0
        delta = desired[:2] - self.command[:2]
        length = np.linalg.norm(delta)
        if length > cfg.max_acceleration * dt and length > 0:
            delta *= cfg.max_acceleration * dt / length
        self.command[:2] += delta
        self.command[2] += np.clip(desired[2] - self.command[2],
                                   -cfg.max_yaw_acceleration * dt, cfg.max_yaw_acceleration * dt)
        return self.command.copy()


def telemetry(observation: dict) -> tuple[BaseState, np.ndarray]:
    """Read the opt-in (1,28) packet: root pose/velocity/time, then two base-frame EEF poses."""
    try:
        value = observation["policy"]["base_drive_state"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing base_drive_state; use scripts/run_base_drive.sh to enable dev telemetry") from exc
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    packet = np.asarray(value, dtype=np.float64)
    if packet.shape != (1, 28) or not np.isfinite(packet).all():
        raise ValueError("base_drive_state must be a finite (1,28) packet")
    row = packet[0]
    q = row[3:7]
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("Invalid root quaternion in dev telemetry")
    x, y, z, w = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    hold = row[14:28].reshape(2, 7).copy()
    if not np.allclose(np.linalg.norm(hold[:, 3:7], axis=1), 1, atol=1e-3):
        raise ValueError("Invalid EEF quaternion in dev telemetry")
    return BaseState(row[0], row[1], yaw, row[7], row[8], row[12], row[13]), hold


def resolve_goals(items: list[dict], frame: str, initial: BaseState) -> list[Goal]:
    if frame not in ("world", "relative_to_start"):
        raise ValueError("waypoint_frame must be world or relative_to_start")
    goals = []
    for item in items:
        goal = Goal(**item)
        if frame == "relative_to_start":
            c, s = math.cos(initial.yaw), math.sin(initial.yaw)
            goal = Goal(initial.x + c * goal.x - s * goal.y,
                        initial.y + s * goal.x + c * goal.y,
                        wrap_angle(initial.yaw + goal.yaw), goal.name)
        goals.append(goal)
    return goals


def hold_action(hold: np.ndarray, grippers: tuple[float, float], velocity) -> np.ndarray:
    return np.concatenate((hold.reshape(14), grippers, velocity)).astype(np.float32)[None, :]


def run_episode(step: Callable, observation: dict, settings: dict, clock=time.monotonic) -> dict:
    """Run the same feedback loop against a real IPC client or a test plant. No hidden env access."""
    cfg = DriveConfig(**settings.get("controller", {}))
    initial, hold = telemetry(observation)  # Fail before sending anything if the contract is absent.
    goals = resolve_goals(settings.get("waypoints", []), settings.get("waypoint_frame", "world"), initial)
    controller = DriveToPoint(goals, cfg)
    grippers = tuple(settings.get("gripper_commands", [1.0, 1.0]))
    if len(grippers) != 2 or not np.isfinite(grippers).all():
        raise ValueError("gripper_commands must contain two finite binary-command values")
    start_wall = clock()
    steps = 0
    trace = []
    braking_since = None
    brake_wall = None
    stopped_since = None
    previous_state = None
    state = initial
    reason = ""
    stopped = False
    error = None
    stop_error = None

    def request_stop():
        try:
            step(hold_action(hold, grippers, np.zeros(3)))
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            return str(exc)
        return None

    while True:
        try:
            state, _ = telemetry(observation)
            if previous_state is not None and state.seconds <= previous_state.seconds:
                raise ValueError("Nonadvancing simulation time")
        except ValueError as exc:
            # Last validated arm hold remains available. Do not fabricate feedback or success.
            error = str(exc)
            stop_error = request_stop()
            reason = "invalid_or_stale_telemetry"
            break
        if controller.status != "FAILED" and clock() - start_wall >= cfg.wall_timeout:
            controller.fail("wall_timeout")
        command = controller.update(state)
        if not trace or state.seconds - trace[-1]["seconds"] >= 0.2 or trace[-1]["state"] != controller.status:
            distance, yaw_error = controller.errors(state)
            trace.append({"seconds": state.seconds, "state": controller.status,
                          "waypoint": controller.goal.name, "x": state.x, "y": state.y,
                          "yaw": state.yaw, "speed": state.speed, "yaw_rate": state.wz,
                          "position_error": distance, "yaw_error": yaw_error})
        if controller.status == "DONE":
            stopped = True
            reason = "route_completed"
            break
        if controller.status == "FAILED":
            if braking_since is None:
                braking_since, brake_wall = state.seconds, clock()
            command[:] = 0
            if controller.stopped(state):
                if stopped_since is None:
                    stopped_since = state.seconds
                stopped = state.seconds - stopped_since >= cfg.settle_seconds
            else:
                stopped_since = None
            if stopped or state.seconds - braking_since >= cfg.brake_timeout or clock() - brake_wall >= cfg.brake_timeout:
                reason = controller.reason
                break
        previous_state = state
        try:
            transition = step(hold_action(hold, grippers, command))
            steps += 1
            if len(transition) == 3:
                observation, terminated, truncated = transition
            elif len(transition) == 5:
                observation, _, terminated, truncated, _ = transition
            else:
                raise ValueError("Unsupported environment step result")
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            # A stop request is best effort when transport/protocol itself has failed.
            error = str(exc)
            stop_error = request_stop()
            reason = "step_error"
            break
        if hasattr(terminated, "detach"):
            terminated = terminated.detach().cpu().numpy()
        if hasattr(truncated, "detach"):
            truncated = truncated.detach().cpu().numpy()
        if np.asarray(terminated).any() or np.asarray(truncated).any():
            reason = "environment_truncated" if np.asarray(truncated).any() else "environment_terminated"
            break

    return {"success": reason == "route_completed", "reason": reason, "stopped": stopped,
            "steps": steps, "simulation_seconds": state.seconds - initial.seconds,
            "wall_seconds": clock() - start_wall, "waypoints_reached": controller.reached,
            "final_pose": [state.x, state.y, state.yaw], "final_speed": state.speed,
            "final_yaw_rate": state.wz, "error": error, "stop_error": stop_error, "trace": trace}
