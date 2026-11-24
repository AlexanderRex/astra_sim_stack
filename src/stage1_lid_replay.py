"""Track a recorded Stage 1 drive trajectory from the live fixture pose."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from base_drive import hold_action, telemetry, wrap_angle


def load_reference(name):
    path = Path(__file__).parent / name
    data = json.loads(path.read_text())
    frames = np.asarray(data["frames"], dtype=float)
    joints = np.asarray(data["initial_joints"], dtype=float)
    observe_steps = data["observe_steps"]
    approach_frame = data["approach_frame"]
    push_frame = data["push_frame"]
    retract_frame = data["retract_frame"]
    if (frames.ndim != 2 or frames.shape[1] != 22
            or not 0 < approach_frame < push_frame < retract_frame < len(frames) - observe_steps
            or observe_steps != 150
            or data["dt"] != .02 or joints.shape != (14,)
            or not np.isfinite(frames).all() or not np.isfinite(joints).all()):
        raise ValueError("Invalid Stage 1 drive reference")
    return frames, joints, approach_frame, push_frame, retract_frame, observe_steps


def follow_reference(step, observation, reference, chamber_pose):
    """Actuate both arms and track the recorded base path in the fixture frame."""
    frames, initial_joints, approach_frame, push_frame, retract_frame, observe_steps = reference
    fixture_rotation = Rotation.from_quat(chamber_pose[3:7])
    fixture_forward = fixture_rotation.apply([1., 0., 0.])
    fixture_yaw = math.atan2(fixture_forward[1], fixture_forward[0])
    rotation = Rotation.from_euler("z", fixture_yaw)
    base_targets = rotation.apply(np.c_[frames[:, :2], np.zeros(len(frames))])[:, :2]
    base_targets += chamber_pose[:2]
    base_yaws = frames[:, 2] + fixture_yaw
    peak = 0.
    done = False
    trace = []
    observed_angles = []
    observed_contact = []
    step_count = 0

    def take_step(target_joints, index, velocity, phase):
        nonlocal observation, peak, done, step_count
        state, arm_hold = telemetry(observation)
        xy_error = base_targets[index] - [state.x, state.y]
        xy_fix = Rotation.from_euler("z", -state.yaw).apply([*xy_error, 0.])[:2]
        yaw_error = wrap_angle(base_yaws[index] - state.yaw)
        correction = np.r_[np.clip(1.8 * xy_fix, -.06, .06),
                           np.clip(1.8 * yaw_error, -.12, .12)]
        # Recorded velocity is expressed in the recorded base frame.
        xy_velocity = Rotation.from_euler("z", yaw_error).apply([*velocity[:2], 0.])[:2]
        action = hold_action(arm_hold, frames[index, 6:8], np.r_[xy_velocity, velocity[2]] + correction)
        transition = step(action, target_joints)
        obs, term, trunc = transition if len(transition) == 3 else (
            transition[0], transition[2], transition[3]
        )
        observation = obs
        done = bool(np.asarray(term).any() or np.asarray(trunc).any())
        step_count += 1
        lid = observation["policy"]["lid_push_state"][0]
        opening = math.degrees(-float(lid[7]))
        peak = max(peak, opening)
        fingers = np.asarray(observation["policy"]["lid_push_fingers"][0])
        if phase == "observe":
            observed_angles.append(opening)
            observed_contact.append(float(np.max(fingers[12:16])))
        if step_count % 10 == 0 or not trace or trace[-1]["phase"] != phase:
            q = np.asarray(observation["policy"]["lid_push_arm_joints"][0, :14])
            trace.append(dict(step=step_count, phase=phase, frame=index, angle_deg=opening,
                              base_error_m=float(np.linalg.norm(xy_error)),
                              yaw_error_rad=yaw_error, base_xy=[state.x, state.y],
                              joint_error_rad=float(np.max(np.abs(q - target_joints))),
                              arm_joints_rad=q.tolist(),
                              fingertips=fingers[:12].reshape(4, 3).tolist(),
                              lid_contact_force_n=fingers[12:16].tolist(),
                              chamber_contact_force_n=fingers[16:18].tolist()))

    start_joints = np.asarray(observation["policy"]["lid_push_arm_joints"][0, :14])
    count = max(1, math.ceil(np.max(np.abs(initial_joints - start_joints)) / .35 / .02))
    for target in np.linspace(start_joints, initial_joints, count + 1)[1:]:
        if done:
            break
        take_step(target, 0, np.zeros(3), "replay_prepare")

    settled = False
    for _ in range(200):
        if done:
            break
        take_step(initial_joints, 0, np.zeros(3), "replay_prepare")
        state, _ = telemetry(observation)
        q = np.asarray(observation["policy"]["lid_push_arm_joints"][0, :14])
        if (np.max(np.abs(q - initial_joints)) <= .025
                and np.linalg.norm(base_targets[0] - [state.x, state.y]) <= .015
                and abs(wrap_angle(base_yaws[0] - state.yaw)) <= .035
                and state.speed <= .02 and abs(state.wz) <= .04):
            settled = True
            break

    if settled:
        for index, row in enumerate(frames):
            if done:
                break
            if index < approach_frame:
                phase = "configure_arms"
            elif index < push_frame:
                phase = "approach"
            elif index < retract_frame:
                phase = "push"
            elif index < len(frames) - observe_steps:
                phase = "retract"
            else:
                phase = "observe"
            take_step(row[8:], index, row[3:6], phase)

    lid = observation["policy"]["lid_push_state"][0]
    score = int(lid[10])
    stayed_open = len(observed_angles) == observe_steps and min(observed_angles) >= 36.
    success = bool(lid[11]) and score == 5 and stayed_open
    if not settled:
        reason = "replay_preparation_failed"
    elif not lid[11] or score != 5:
        reason = "threshold_not_reached"
    elif len(observed_angles) < observe_steps:
        reason = "observation_interrupted"
    else:
        reason = "scored" if stayed_open else "lid_closed_after_retract"
    return dict(success=success, reason=reason, stage1=bool(lid[11]), score=score,
                stayed_open=stayed_open, peak_deg=peak, final_deg=math.degrees(-float(lid[7])),
                min_after_retract_deg=min(observed_angles, default=None),
                max_finger_force_after_retract_n=max(observed_contact, default=None),
                observation_steps=len(observed_angles), steps=step_count, trace=trace)
