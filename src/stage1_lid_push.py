"""Stage 1 simulator-state baseline. The hand pushes against the lid handle."""
import json
import math
from pathlib import Path

import numpy as np
from base_drive import DriveConfig, DriveToPoint, Goal, hold_action, run_episode, telemetry
from policy.base import BasePolicy
from scipy.spatial.transform import Rotation, Slerp


HINGE = np.array([0.0, 0.35923016, 0.16500795])
HANDLE_Y = -0.285
# The EEF command is for link7. Its -X axis points toward the fingertips.
TIP_OFFSET = 0.25


def packet(obs):
    robot = np.asarray(obs["policy"]["base_drive_state"][0], dtype=float)
    chamber = np.asarray(obs["policy"]["lid_push_state"][0], dtype=float)
    if robot.shape != (28,) or chamber.shape != (12,) or not (
        np.isfinite(robot).all() and np.isfinite(chamber).all()
    ):
        raise ValueError("Invalid Stage 1 telemetry")
    return robot, chamber


def finger_packet(obs):
    packet = np.asarray(obs["policy"]["lid_push_fingers"], dtype=float)
    if packet.shape != (1, 16) or not np.isfinite(packet).all():
        raise ValueError("Invalid finger pose or contact-force telemetry")
    return packet[0]


class Stage1LidPush(BasePolicy):
    def get_model(self, usr_args):
        return None

    def reset_model(self):
        pass

    def get_action(self):
        raise RuntimeError("Stage1LidPush owns the rollout in eval()")

    def eval(self, task_env, observation, usr_args, video_writer):
        if usr_args["actions_dim"] != 19:
            raise ValueError("Stage 1 expects Galbot-Mobile's 19-D action")
        cfg = usr_args["lid_push"]
        hand = {"left": 0, "right": 1}[cfg["side"]]
        robot, chamber = packet(observation)
        initial = robot[:7].tolist()
        navigation = None
        if cfg.get("navigate"):
            chamber_rotation = Rotation.from_quat(chamber[3:7])
            aisle = chamber[:3] + chamber_rotation.apply([-0.0976, -0.9916, 0])
            state, _ = telemetry(observation)
            # Clear the table before moving along the aisle. The side approach
            # continues around the west corner with the base facing the lid.
            waypoints = [dict(name="clear_table", x=state.x, y=aisle[1], yaw=state.yaw)]
            if cfg.get("approach_from_left"):
                staging = chamber[:3] + chamber_rotation.apply([-0.9076, -0.0416, 0])
                heading = chamber_rotation.apply([1., 0., 0.])
                yaw = math.atan2(heading[1], heading[0])
                waypoints.extend([
                    dict(name="west_corner", x=staging[0], y=aisle[1], yaw=yaw),
                    dict(name="face_chamber", x=staging[0], y=staging[1], yaw=yaw),
                ])
            else:
                heading = chamber_rotation.apply([0., 1., 0.])
                yaw = math.atan2(heading[1], heading[0])
                waypoints.append(dict(name="face_chamber", x=aisle[0], y=aisle[1], yaw=yaw))

            def navigation_step(action):
                nonlocal observation
                transition = task_env.step(action)
                observation = transition[0]
                return transition

            _, travel_hold = telemetry(observation)
            initial_hold = travel_hold.copy()
            # Keep both fingertips inside the base envelope on the west aisle.
            travel_hold[:, 0] = np.minimum(travel_hold[:, 0], 0.12)
            travel_hold[:, 2] += 0.04
            for fraction in np.linspace(0., 1., 95):
                poses = initial_hold + fraction * (travel_hold - initial_hold)
                transition = navigation_step(hold_action(poses, (-1., -1.), (0., 0., 0.)))
                term, trunc = transition[-2:] if len(transition) == 3 else transition[2:4]
                if np.asarray(term).any() or np.asarray(trunc).any():
                    break

            navigation = run_episode(navigation_step, observation, {
                "waypoints": waypoints,
                "gripper_commands": [-1., -1.],
                "controller": {
                    "position_tolerance": 0.025, "yaw_tolerance": 0.055,
                    "stopped_yaw_rate": 0.06, "stall_seconds": 12.,
                },
            })
            robot, chamber = packet(observation)
        hold = robot[14:].reshape(2, 7).copy()
        fingers = finger_packet(observation)
        trace = []
        step_count = 0
        finished = False
        opening = -float(chamber[7])
        peak = opening
        observed_angles = []
        recoveries = 0
        observe_steps = cfg.get("observe_steps", 30)

        def world_hand():
            base = Rotation.from_quat(robot[3:7])
            current = robot[14:].reshape(2, 7)[hand]
            return robot[:3] + base.apply(current[:3]), base * Rotation.from_quat(current[3:])

        def step(position, orientation, phase, velocity=(0., 0., 0.)):
            nonlocal robot, chamber, fingers, finished, step_count, peak
            base = Rotation.from_quat(robot[3:7])
            poses = hold.copy()
            poses[hand, :3] = base.inv().apply(position - robot[:3])
            poses[hand, 3:] = (base.inv() * orientation).as_quat()
            grippers = np.ones(2)
            grippers[hand] = -1.0  # Closed fingers form the pusher.
            action = hold_action(poses, grippers, velocity)
            transition = task_env.step(action)
            obs, term, trunc = transition if len(transition) == 3 else (
                transition[0], transition[2], transition[3]
            )
            robot, chamber = packet(obs)
            fingers = finger_packet(obs)
            step_count += 1
            finished = bool(np.asarray(term).any() or np.asarray(trunc).any())
            peak = max(peak, -float(chamber[7]))
            if step_count % 10 == 0 or not trace or trace[-1]["phase"] != phase:
                trace.append(dict(
                    step=step_count, phase=phase, angle_deg=math.degrees(-chamber[7]),
                    score=int(chamber[10]), eef=world_hand()[0].tolist(), target=position.tolist(),
                    base=robot[:7].tolist(),
                    fingertips=fingers[:12].reshape(4, 3)[2 * hand:2 * hand + 2].tolist(),
                    lid_contact_force_n=fingers[12:][2 * hand:2 * hand + 2].tolist()
                ))

        def move(position, orientation, phase, speed, base_velocity=lambda: (0., 0., 0.)):
            start, start_rot = world_hand()
            duration = max(np.linalg.norm(position - start) / speed,
                           (start_rot.inv() * orientation).magnitude() / 0.9, 0.2)
            count = math.ceil(duration / 0.02)
            interp = Slerp([0, 1], Rotation.concatenate([start_rot, orientation]))
            for i in range(1, count + 1):
                if finished:
                    break
                fraction = i / count
                step(start + fraction * (position - start), interp(fraction), phase, base_velocity())

        chamber_rot = Rotation.from_quat(chamber[3:7])
        chamber_pos = chamber[:3].copy()
        side_approach = cfg.get("approach_from_left", False)
        forward = np.array([1., 0., 0.] if side_approach else [0., 1., 0.])
        pitch_axis = np.cross(forward, [0., 0., 1.])
        # Keep the gripper roll close to the default Galbot stance.
        # Link7's -X points into the chamber and its +Z points down.
        orientation = (chamber_rot * Rotation.from_euler("z", math.pi if side_approach else -math.pi / 2)
                       * Rotation.from_euler("x", math.pi))
        contact = np.array([cfg["handle_x"], cfg.get("handle_y", HANDLE_Y), cfg.get("tip_z", 0.202)])

        def wrist_for_tip(tip, wrist_orientation=orientation):
            return (chamber_pos + chamber_rot.apply(tip)
                    - wrist_orientation.apply([-TIP_OFFSET, 0, 0]))

        if navigation is not None and not navigation["success"]:
            reason = "navigation_failed"
        elif opening > 0.05 or chamber[11]:
            reason = "lid_not_closed_at_reset"
        else:
            current, current_rot = world_hand()
            move(current + chamber_rot.apply(-0.06 * forward), current_rot,
                 "clear", cfg["speed"])
            prepare = contact - 0.20 * forward
            move(wrist_for_tip(prepare), orientation, "approach", cfg["speed"])
            # Tuck the other arm before bringing the base to the table.
            other = 1 - hand
            start = hold[other, 0]
            for x in np.linspace(start, min(start, 0.10), 100):
                if finished:
                    break
                hold[other, 0] = x
                step(wrist_for_tip(prepare), orientation, "tuck")
            precontact_error = float(np.linalg.norm(world_hand()[0] - wrist_for_tip(prepare)))
            if cfg.get("recover_arm_pose") and not finished and precontact_error > 0.025:
                # Reconfigure in free space before advancing the base. The extra
                # waypoint moves the elbow away from the extended travel pose.
                for roll in (0.30, -0.30):
                    back = wrist_for_tip(prepare - .10 * forward)
                    near = back + chamber_rot.apply([0., .07, .06])
                    relaxed = orientation * Rotation.from_rotvec([roll, 0., 0.])
                    move(near, relaxed, "reseed_arm_clearance", cfg["speed"])
                    move(wrist_for_tip(prepare), orientation, "reseed_arm_return", cfg["speed"])
                    for _ in range(45):
                        if finished:
                            break
                        step(wrist_for_tip(prepare), orientation, "reseed_arm_settle")
                    precontact_error = float(np.linalg.norm(world_hand()[0] - wrist_for_tip(prepare)))
                    if finished or precontact_error <= 0.025:
                        break
            drive_ok = not finished and (
                not cfg.get("recover_arm_pose") or precontact_error <= 0.025
            )
            if drive_ok:
                state, _ = telemetry({"policy": {"base_drive_state": robot[None]}})
                base_goal = robot[:3] + chamber_rot.apply(cfg["base_advance"] * forward)
                drive = DriveToPoint([Goal(*base_goal[:2], state.yaw)], DriveConfig(
                    max_speed=0.08, position_tolerance=cfg.get("align_tolerance", 0.015),
                    yaw_tolerance=cfg.get("align_yaw_tolerance", 0.04),
                    stopped_yaw_rate=cfg.get("align_stopped_yaw_rate", 0.025),
                    waypoint_timeout=12.0,
                ))
                for _ in range(650):
                    state, _ = telemetry({"policy": {"base_drive_state": robot[None]}})
                    velocity = drive.update(state)
                    if finished or drive.status in ("DONE", "FAILED"):
                        break
                    advance = np.clip(cfg["base_advance"] - np.linalg.norm(robot[:2] - base_goal[:2]),
                                      0.0, cfg["base_advance"])
                    tip = prepare + 0.65 * advance * forward
                    step(wrist_for_tip(tip), orientation, "align_base", velocity)
                drive_ok = not finished and drive.status == "DONE"
            anchor, _ = telemetry({"policy": {"base_drive_state": robot[None]}})

            def hold_base(follow=0.):
                if not cfg.get("hold_base_during_push"):
                    return (0., 0., 0.)
                state, _ = telemetry({"policy": {"base_drive_state": robot[None]}})
                offset = chamber_rot.apply([0., follow, 0.])
                error = Rotation.from_euler("z", -state.yaw).apply(
                    [anchor.x + offset[0] - state.x, anchor.y + offset[1] - state.y, 0.]
                )[:2]
                yaw_error = math.remainder(anchor.yaw - state.yaw, 2 * math.pi)
                return (*np.clip(2.5 * error, -.08, .08),
                        float(np.clip(2.0 * yaw_error, -.15, .15)))

            if drive_ok:
                move(wrist_for_tip(contact), orientation, "insert", cfg["speed"] * 0.5, hold_base)
            contact_feedback = cfg.get("contact_feedback", False)
            pad = 2 * hand + 1
            # Pad position measured at the start of a successful side push.
            # Calibrate in the fixture frame so small spawn offsets do not enter the target.
            pad_entry = np.array([-.295, -.277, .199])
            wrist_bias = np.zeros(3)
            if drive_ok and contact_feedback:
                nominal = wrist_for_tip(contact)
                for _ in range(55):
                    if finished:
                        break
                    measured = chamber_rot.inv().apply(fingers[:12].reshape(4, 3)[pad] - chamber_pos)
                    error = pad_entry - measured
                    # Leave insertion depth to its existing trajectory. Correct the two
                    # axes across it, where a few millimetres determine which face contacts.
                    error[0] = 0.
                    wrist_bias = np.clip(wrist_bias + np.clip(error * 0.10, -.001, .001),
                                         -.035, .035)
                    step(nominal + chamber_rot.apply(wrist_bias), orientation,
                         "seat_pad", hold_base())
            # A commanded arc follows the handle's trajectory as the lid opens.
            # The tangent points upward at the closed position, then rearward.
            arc_step = math.radians(cfg.get("degrees_per_step", 0.2))
            travel = math.radians(cfg.get("arc_degrees", 48))
            release_angle = math.radians(cfg.get("release_angle_deg", 36))
            angle = 0.
            lead = math.radians(cfg.get("arc_lead_deg", 180))
            push_budget = cfg.get("push_steps", math.ceil(travel / arc_step))
            best_opening = max(0., -float(chamber[7]))
            last_progress = 0
            last_contact = 0
            for tick in range(push_budget):
                if finished or not drive_ok or (chamber[11] and -chamber[7] >= release_angle):
                    break
                opening_now = max(0., -float(chamber[7]))
                if opening_now > best_opening + math.radians(.35):
                    best_opening, last_progress = opening_now, tick
                if np.max(fingers[12 + 2 * hand:14 + 2 * hand]) >= 5.:
                    last_contact = tick
                if contact_feedback and tick - max(last_progress, last_contact) >= 65:
                    if recoveries >= 2:
                        break
                    recoveries += 1
                    last_progress = last_contact = tick
                    best_opening = opening_now
                    angle = opening_now
                    # Return below the current handle before re-entering. A small
                    # forward-lateral correction changes which face of the pad seats.
                    wrist_bias[1] = np.clip(wrist_bias[1] - .008, -.035, .035)
                    wrist_bias[2] = np.clip(wrist_bias[2] - .006, -.035, .035)
                    retry_tip = HINGE + Rotation.from_rotvec([-angle, 0, 0]).apply(contact - HINGE)
                    retry_wrist = wrist_for_tip(retry_tip) + chamber_rot.apply(wrist_bias)
                    clearance = chamber_rot.apply(-.055 * forward + [0., 0., -.022])
                    move(retry_wrist + clearance, orientation, "recontact_clear", cfg["speed"], hold_base)
                    move(retry_wrist, orientation, "recontact_insert", cfg["speed"] * .5, hold_base)
                    continue
                angle = min(angle + arc_step, travel, opening_now + lead)
                tip = HINGE + Rotation.from_rotvec([-angle, 0, 0]).apply(contact - HINGE)
                # Pitch up as the handle rises to keep the wrist below it.
                pitch_start = math.radians(cfg.get("pitch_start_deg", 0))
                pitch = min(max(0., angle - pitch_start) * cfg.get("pitch_ratio", 0),
                            math.radians(45))
                wrist_orientation = chamber_rot * Rotation.from_rotvec(pitch * pitch_axis) * (
                    chamber_rot.inv() * orientation
                )
                follow = cfg.get("base_follow_distance", 0.) * np.clip(
                    (angle - math.radians(15)) / math.radians(50), 0., 1.
                )
                step(wrist_for_tip(tip, wrist_orientation) + chamber_rot.apply(wrist_bias),
                     wrist_orientation, "push",
                     hold_base(follow))
            if not finished:
                current, current_rot = world_hand()
                move(current + chamber_rot.apply(-0.12 * forward), current_rot, "retract", cfg["speed"])
                current, current_rot = world_hand()
                for _ in range(observe_steps):
                    if finished:
                        break
                    step(current, current_rot, "observe")
                    observed_angles.append(math.degrees(-chamber[7]))
            if chamber[11] and int(chamber[10]) == 5:
                reason = "scored"
            else:
                if drive_ok:
                    reason = "threshold_not_reached"
                elif cfg.get("recover_arm_pose") and precontact_error > .025:
                    reason = "precontact_pose_unreachable"
                else:
                    reason = "base_alignment_failed"

        min_after_retract = min(observed_angles, default=None)
        stayed_open = (len(observed_angles) == observe_steps
                       and min_after_retract is not None and min_after_retract >= 36.0)
        scored = reason == "scored"
        if scored and cfg.get("require_stable") and not stayed_open:
            reason = "lid_closed_after_retract"
        report = dict(
            success=reason == "scored", reason=reason, score=int(chamber[10]),
            stayed_open=stayed_open, min_after_retract_deg=min_after_retract,
            stage1=bool(chamber[11]), peak_deg=math.degrees(peak),
            final_deg=math.degrees(-chamber[7]), initial_base=initial,
            chamber_pose=chamber[:7].tolist(), joint_limits_deg=np.degrees(chamber[8:10]).tolist(),
            steps=step_count, contact_recoveries=recoveries, trace=trace, navigation=navigation,
        )
        output = Path(usr_args["save_path"]) / "stage1_lid_push.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("Stage 1:", json.dumps({k: v for k, v in report.items() if k not in ("trace", "navigation")}), flush=True)
        return report["success"]
