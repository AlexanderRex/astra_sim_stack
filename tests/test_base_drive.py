"""Offline controller/IPC-contract tests. The test plant is not a physics simulator."""

import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from base_drive import (  # noqa: E402 -- run directly against the source checkout
    BaseState, DriveConfig, DriveToPoint, Goal, resolve_goals, run_episode, telemetry,
)


HOLD = np.array([[0.4, 0.25, 0.8, 0, 0, 0, 1], [0.4, -0.25, 0.8, 0, 0, 0, 1]])


def packet(state, hold=HOLD):
    q = [0, 0, math.sin(state.yaw / 2), math.cos(state.yaw / 2)]
    values = [state.x, state.y, 0, *q, state.vx, state.vy, 0, 0, 0, state.wz, state.seconds]
    return {"policy": {"base_drive_state": np.array([values + hold.reshape(-1).tolist()])}}


class Plant:
    def __init__(self, *, legacy=False, jam=False, done_at=None, truncated=False):
        self.state = BaseState(0, 0, 1.1, 0, 0, 0, 0)
        self.filtered = np.zeros(3)
        self.commands = []
        self.legacy, self.jam = legacy, jam
        self.done_at, self.truncated = done_at, truncated

    def step(self, action):
        assert action.shape == (1, 19) and action.dtype == np.float32
        assert np.isfinite(action).all()
        np.testing.assert_allclose(action[0, :14], HOLD.reshape(-1), atol=1e-7)
        np.testing.assert_array_equal(action[0, 14:16], [1, 1])
        command = action[0, 16:].copy()
        self.commands.append(command)
        self.filtered += 0.1 * (command - self.filtered)
        vx, vy, wz = self.filtered if not self.jam else np.zeros(3)
        old = self.state
        c, s = math.cos(old.yaw), math.sin(old.yaw)
        wx, wy = c * vx - s * vy, s * vx + c * vy
        self.state = BaseState(old.x + wx * 0.02, old.y + wy * 0.02,
                               old.yaw + wz * 0.02, wx, wy, wz, old.seconds + 0.02)
        done = self.done_at is not None and len(self.commands) >= self.done_at
        term, trunc = np.array([done and not self.truncated]), np.array([done and self.truncated])
        observation = packet(self.state)
        return (observation, 0.0, term, trunc, {}) if self.legacy else (observation, term, trunc)


def settings(**controller):
    return {"waypoint_frame": "world", "waypoints": [
        {"name": "first", "x": 0.35, "y": 0.15, "yaw": math.pi / 2},
        {"name": "second", "x": 0.05, "y": -0.1, "yaw": -0.1}],
        "controller": controller}


@pytest.mark.parametrize("legacy", [False, True])
def test_reaches_both_points_and_measured_stop_with_smoothed_actuation(legacy):
    plant = Plant(legacy=legacy)
    report = run_episode(plant.step, packet(plant.state), settings(), clock=lambda: 0)
    assert report["success"] and report["stopped"]
    assert [g["name"] for g in report["waypoints_reached"]] == ["first", "second"]
    assert np.linalg.norm(np.array(report["final_pose"][:2]) - [0.05, -0.1]) <= 0.04
    assert abs(report["final_pose"][2] + 0.1) <= 0.06
    assert report["final_speed"] <= 0.015 and abs(report["final_yaw_rate"]) <= 0.025
    commands = np.array(plant.commands)
    assert np.linalg.norm(commands[:, :2], axis=1).max() <= 0.200001
    assert np.abs(commands[:, 2]).max() <= 0.350001
    np.testing.assert_array_equal(commands[-1], [0, 0, 0])
    json.dumps(report, allow_nan=False)


def test_position_match_at_high_velocity_is_not_success():
    controller = DriveToPoint([Goal(0, 0, 0)])
    for i in range(60):
        command = controller.update(BaseState(0, 0, 0, 0.1, 0, 0, i * 0.02))
        np.testing.assert_array_equal(command, [0, 0, 0])
        assert controller.status == "SETTLE"
    assert not controller.reached
    for i in range(60, 100):
        controller.update(BaseState(0, 0, 0, 0, 0, 0, i * 0.02))
    assert controller.status == "DONE"


def test_leaving_tolerance_restarts_stationary_dwell():
    controller = DriveToPoint([Goal(0, 0, 0)], DriveConfig(settle_seconds=0.5))
    for t in [0, 0.1, 0.2, 0.3]:
        controller.update(BaseState(0, 0, 0, 0, 0, 0, t))
    controller.update(BaseState(0.1, 0, 0, 0, 0, 0, 0.4))
    for t in [0.5, 0.6, 0.7, 0.8]:
        controller.update(BaseState(0, 0, 0, 0, 0, 0, t))
        assert controller.status == "SETTLE"
    controller.update(BaseState(0, 0, 0, 0, 0, 0, 1.1))
    assert controller.status == "DONE"


def test_world_to_body_rotation_and_shortest_yaw():
    controller = DriveToPoint([Goal(1, 0, math.pi / 2)])
    controller.update(BaseState(0, 0, math.pi / 2, 0, 0, 0, 0))
    velocity = controller.update(BaseState(0, 0, math.pi / 2, 0, 0, 0, 0.1))
    assert abs(velocity[0]) < 1e-8 and velocity[1] < 0
    controller = DriveToPoint([Goal(0, 0, -math.pi + 0.1)])
    controller.update(BaseState(0, 0, math.pi - 0.1, 0, 0, 0, 0))
    assert controller.update(BaseState(0, 0, math.pi - 0.1, 0, 0, 0, 0.1))[2] > 0


def test_relative_waypoints_are_relative_to_initial_heading_and_position():
    initial = BaseState(2, 3, math.pi / 2, 0, 0, 0, 0)
    goal = resolve_goals([{"x": 1, "y": 0, "yaw": 0.2}], "relative_to_start", initial)[0]
    np.testing.assert_allclose([goal.x, goal.y, goal.yaw], [2, 4, math.pi / 2 + 0.2])


def test_stalled_plant_fails_and_zeroes_commands():
    plant = Plant(jam=True)
    report = run_episode(plant.step, packet(plant.state), settings(stall_seconds=0.5), clock=lambda: 0)
    assert not report["success"] and report["reason"] == "no_measured_progress"
    assert report["stopped"]
    assert not report["waypoints_reached"]
    np.testing.assert_array_equal(plant.commands[-1], [0, 0, 0])
    json.dumps(report, allow_nan=False)


def test_waypoint_timeout_is_failure():
    plant = Plant(jam=True)
    report = run_episode(plant.step, packet(plant.state), settings(
        waypoint_timeout=0.8, settle_seconds=0.1, stall_seconds=10), clock=lambda: 0)
    assert not report["success"] and report["reason"] == "waypoint_timeout"


@pytest.mark.parametrize("truncated", [False, True])
def test_episode_end_is_not_navigation_success(truncated):
    plant = Plant(done_at=10, truncated=truncated)
    report = run_episode(plant.step, packet(plant.state), settings(), clock=lambda: 0)
    assert not report["success"] and not report["stopped"]
    assert report["reason"] == ("environment_truncated" if truncated else "environment_terminated")
    assert len(plant.commands) == 10


def test_initial_missing_telemetry_sends_no_action():
    plant = Plant()
    with pytest.raises(ValueError, match="run_base_drive"):
        run_episode(plant.step, {"policy": {}}, settings())
    assert not plant.commands


def test_stale_telemetry_requests_stop_and_never_succeeds():
    observation = packet(BaseState(0, 0, 0, 0, 0, 0, 0))
    commands = []

    def step(action):
        commands.append(action.copy())
        return observation, False, False

    report = run_episode(step, observation, settings(), clock=lambda: 0)
    assert report["reason"] == "invalid_or_stale_telemetry" and not report["success"]
    np.testing.assert_array_equal(commands[-1][0, 16:], [0, 0, 0])


@pytest.mark.parametrize("transport_error", [True, False])
def test_broken_step_protocol_requests_stop_without_claiming_stationary(transport_error):
    commands = []

    def step(action):
        commands.append(action.copy())
        if transport_error:
            raise OSError("unavailable")
        return (None,)

    report = run_episode(step, packet(BaseState(0, 0, 0, 0, 0, 0, 0)), settings(), clock=lambda: 0)
    assert report["reason"] == "step_error" and not report["success"] and not report["stopped"]
    assert len(commands) == 2
    np.testing.assert_array_equal(commands[-1][0, 16:], [0, 0, 0])


def test_wall_watchdog_requests_braking_and_reports_failure():
    plant = Plant()
    calls = iter(np.arange(0, 100, 0.1))
    report = run_episode(plant.step, packet(plant.state), settings(wall_timeout=0.5),
                         clock=lambda: float(next(calls)))
    assert not report["success"] and report["reason"] == "wall_timeout"
    np.testing.assert_array_equal(plant.commands[-1], [0, 0, 0])


@pytest.mark.parametrize("change", ["nan", "zero_quaternion", "wrong_shape"])
def test_invalid_packets_are_rejected(change):
    observation = packet(BaseState(0, 0, 0, 0, 0, 0, 0))
    value = observation["policy"]["base_drive_state"]
    if change == "nan":
        value[0, 0] = math.nan
    elif change == "zero_quaternion":
        value[0, 3:7] = 0
    else:
        observation["policy"]["base_drive_state"] = value[0]
    with pytest.raises(ValueError):
        telemetry(observation)


@pytest.mark.parametrize("setting", [{"max_speed": 0}, {"position_gain": math.nan},
                                     {"max_acceleration": -1}, {"waypoint_timeout": 0.1}])
def test_invalid_controller_config_is_rejected(setting):
    with pytest.raises(ValueError):
        DriveConfig(**setting)


def test_reset_means_new_controller_and_no_old_dwell_or_targets():
    state = BaseState(0, 0, 0, 0, 0, 0, 0)
    first = DriveToPoint([Goal(0, 0, 0)])
    first.update(state)
    first.update(BaseState(0, 0, 0, 0, 0, 0, 1))
    assert first.status == "DONE"
    second = DriveToPoint([Goal(1, 0, 0)])
    second.update(state)
    assert second.status == "DRIVE" and not second.reached


def test_telemetry_patch_applies_and_inverse_transform_preserves_arm_hold():
    relative = Path("robofinals/modules/robots/galbot/galbot.py")
    original = ROOT / "third_party/task_repo" / relative
    if not original.exists():
        pytest.skip("Initialize the pinned task source to check the optional dev patch")
    patch = ROOT / "docker/patches/task-repo-base-drive-observations.patch"
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(original.read_bytes())
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=directory, check=True, capture_output=True)
        subprocess.run(["git", "apply", str(patch)], cwd=directory, check=True, capture_output=True)
        tree = ast.parse(target.read_text(encoding="utf-8"))
    group = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GalbotPolicyObservationsCfg")
    field = next(n.value for n in group.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "base_drive_state" for t in n.targets))
    expression = compile(ast.Expression(field), "telemetry_flag", "eval")
    for enabled in [False, True]:
        environment = {"ASTRA_BASE_DRIVE_TELEMETRY": "1"} if enabled else {}
        value = eval(expression, {"os": SimpleNamespace(environ=environment),
                                  "ObsTerm": lambda **kw: kw, "_base_drive_state": "helper"})
        assert value == ({"func": "helper"} if enabled else None)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_base_drive_state")
    torch_math = SimpleNamespace(cat=lambda xs, dim: np.concatenate(xs, axis=dim),
                                 sum=lambda x, dim, keepdim: np.sum(x, axis=dim, keepdims=keepdim),
                                 linalg=SimpleNamespace(cross=np.cross))
    namespace = {"torch": torch_math, "as_torch": lambda x: x,
                 "sim_quat_raw_to_xyzw_torch": lambda x: x}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "telemetry", "exec"), namespace)
    # Root rotated +90 degrees, nonzero world/environment origins; local hold must recover exactly.
    quat = np.array([[0, 0, math.sqrt(0.5), math.sqrt(0.5)]])
    origin = np.array([[10, 20, 0]])
    root = origin + [2, 3, 0]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    world_pos = (HOLD[:, :3] @ rotation.T)[None, :, :] + root[:, None, :]
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=root, root_quat_w=quat,
                              root_lin_vel_w=np.array([[0.1, 0, 0]]), root_ang_vel_w=np.zeros((1, 3))))
    frame = SimpleNamespace(data=SimpleNamespace(target_pos_w=world_pos,
                                                 target_quat_w=np.repeat(quat[:, None, :], 2, axis=1)))

    class Scene(dict):
        articulations = {"robot": robot}
        env_origins = origin

    env = SimpleNamespace(scene=Scene(ee_frame=frame), num_envs=1,
                          episode_length_buf=np.array([25]), step_dt=0.02)
    value = namespace["_base_drive_state"](env)
    state, hold = telemetry({"policy": {"base_drive_state": value}})
    np.testing.assert_allclose([state.x, state.y, state.yaw, state.seconds], [2, 3, math.pi / 2, 0.5])
    np.testing.assert_allclose(hold, HOLD, atol=1e-7)


def test_precision_capture_has_margin_and_preserves_outer_acceptance():
    cfg = DriveConfig(position_tolerance=0.005, yaw_tolerance=math.radians(0.5))
    controller = DriveToPoint([Goal(0, 0, 0)], cfg)
    controller.update(BaseState(0.0049, 0, 0, 0, 0, 0, 0))
    assert controller.status == "DRIVE"
    assert controller.update(BaseState(0.0049, 0, 0, 0, 0, 0, 0.02))[0] < 0
    controller.update(BaseState(0.001, 0, 0, 0, 0, 0, 0.04))
    assert controller.status == "SETTLE"
    controller.update(BaseState(0.004, 0, 0, 0, 0, 0, 0.2))
    assert controller.status == "SETTLE"
    controller.update(BaseState(0.0051, 0, 0, 0, 0, 0, 0.3))
    assert controller.status == "DRIVE" and not controller.reached


def test_precision_route_reaches_both_tolerances_with_smoothed_actuation():
    plant = Plant()
    report = run_episode(plant.step, packet(plant.state), settings(
        position_tolerance=0.005, yaw_tolerance=math.radians(0.5)), clock=lambda: 0)
    assert report["success"] and report["stopped"]
    assert len(report["waypoints_reached"]) == 2
    for point in report["waypoints_reached"]:
        assert point["position_error"] <= 0.005
        assert point["yaw_error"] <= math.radians(0.5)


def test_local_integral_is_bounded_and_clears_in_capture_zone():
    cfg = DriveConfig(stall_seconds=100, waypoint_timeout=120)
    controller = DriveToPoint([Goal(0, 0, 0)], cfg)
    for i in range(300):
        controller.update(BaseState(0.04, 0, 0.1, 0, 0, 0, i * 0.02))
    assert 0 < np.linalg.norm(controller.integral[:2]) <= cfg.integral_speed_limit
    assert 0 < abs(controller.integral[2]) <= cfg.integral_yaw_rate_limit
    # Position captured while heading still needs correction: no stored push.
    controller.update(BaseState(0, 0, 0.1, 0, 0, 0, 6.0))
    np.testing.assert_array_equal(controller.integral[:2], [0, 0])
    # Heading captured while translation still needs correction.
    controller.update(BaseState(0.04, 0, 0, 0, 0, 0, 6.02))
    assert controller.integral[2] == 0


def test_slow_approach_within_acceptance_can_finish_without_inner_capture():
    controller = DriveToPoint([Goal(0, 0, 0)])
    for i in range(100):
        controller.update(BaseState(0.003, 0, math.radians(0.3), 0, 0, 0, i * 0.02))
    assert controller.status == "DONE"
    assert controller.reached[0]["position_error"] <= 0.005
    assert controller.reached[0]["yaw_error"] <= math.radians(0.5)
