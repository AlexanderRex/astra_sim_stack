"""Stage 1 simulator-state baseline. The hand pushes against the lid handle."""
import json
import math
from pathlib import Path

import numpy as np
from base_drive import DriveConfig, DriveToPoint, Goal, hold_action, telemetry
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
        hold = robot[14:].reshape(2, 7).copy()
        initial = robot[:7].tolist()
        trace = []
        step_count = 0
        finished = False
        opening = -float(chamber[7])
        peak = opening

        def world_hand():
            base = Rotation.from_quat(robot[3:7])
            current = robot[14:].reshape(2, 7)[hand]
            return robot[:3] + base.apply(current[:3]), base * Rotation.from_quat(current[3:])

        def step(position, orientation, phase, velocity=(0., 0., 0.)):
            nonlocal robot, chamber, finished, step_count, peak
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
            step_count += 1
            finished = bool(np.asarray(term).any() or np.asarray(trunc).any())
            peak = max(peak, -float(chamber[7]))
            if step_count % 10 == 0 or not trace or trace[-1]["phase"] != phase:
                trace.append(dict(
                    step=step_count, phase=phase, angle_deg=math.degrees(-chamber[7]),
                    score=int(chamber[10]), eef=world_hand()[0].tolist(), target=position.tolist(),
                    base=robot[:7].tolist()
                ))

        def move(position, orientation, phase, speed):
            start, start_rot = world_hand()
            duration = max(np.linalg.norm(position - start) / speed,
                           (start_rot.inv() * orientation).magnitude() / 0.9, 0.2)
            count = math.ceil(duration / 0.02)
            interp = Slerp([0, 1], Rotation.concatenate([start_rot, orientation]))
            for i in range(1, count + 1):
                if finished:
                    break
                fraction = i / count
                step(start + fraction * (position - start), interp(fraction), phase)

        chamber_rot = Rotation.from_quat(chamber[3:7])
        chamber_pos = chamber[:3].copy()
        # Keep the gripper roll close to the default Galbot stance.
        # Link7's -X points into the chamber and its +Z points down.
        orientation = (chamber_rot * Rotation.from_euler("z", -math.pi / 2)
                       * Rotation.from_euler("x", math.pi))
        contact = np.array([cfg["handle_x"], HANDLE_Y, cfg.get("tip_z", 0.202)])
        tip_offset = orientation.apply([-TIP_OFFSET, 0, 0])

        def wrist_for_tip(tip):
            return chamber_pos + chamber_rot.apply(tip) - tip_offset

        if opening > 0.05 or chamber[11]:
            reason = "lid_not_closed_at_reset"
        else:
            current, current_rot = world_hand()
            move(current + chamber_rot.apply([0, -0.06, 0]), current_rot,
                 "clear", cfg["speed"])
            prepare = contact.copy()
            prepare[1] -= 0.20
            move(wrist_for_tip(prepare), orientation, "approach", cfg["speed"])
            # Tuck the other arm before bringing the base to the table.
            other = 1 - hand
            start = hold[other, 0]
            for x in np.linspace(start, min(start, 0.10), 100):
                if finished:
                    break
                hold[other, 0] = x
                step(wrist_for_tip(prepare), orientation, "tuck")
            drive_ok = not finished
            if drive_ok:
                state, _ = telemetry({"policy": {"base_drive_state": robot[None]}})
                base_goal = robot[:3] + chamber_rot.apply([0, cfg["base_advance"], 0])
                drive = DriveToPoint([Goal(*base_goal[:2], state.yaw)], DriveConfig(
                    max_speed=0.08, position_tolerance=0.015, yaw_tolerance=0.04,
                    waypoint_timeout=12.0,
                ))
                for _ in range(650):
                    state, _ = telemetry({"policy": {"base_drive_state": robot[None]}})
                    velocity = drive.update(state)
                    if finished or drive.status in ("DONE", "FAILED"):
                        break
                    advance = np.clip(cfg["base_advance"] - np.linalg.norm(robot[:2] - base_goal[:2]),
                                      0.0, cfg["base_advance"])
                    tip = prepare + [0, 0.65 * advance, 0]
                    step(wrist_for_tip(tip), orientation, "align_base", velocity)
                drive_ok = not finished and drive.status == "DONE"
            if drive_ok:
                move(wrist_for_tip(contact), orientation, "insert", cfg["speed"] * 0.5)
            # A commanded arc follows the handle's trajectory as the lid opens.
            # The tangent points upward at the closed position, then rearward.
            arc_step = math.radians(cfg.get("degrees_per_step", 0.2))
            travel = math.radians(cfg.get("arc_degrees", 48))
            for angle in np.arange(arc_step, travel + arc_step, arc_step):
                if finished or not drive_ok or chamber[11]:
                    break
                tip = HINGE + Rotation.from_rotvec([-angle, 0, 0]).apply(contact - HINGE)
                step(wrist_for_tip(tip), orientation, "push")
            if not finished:
                current, current_rot = world_hand()
                move(current + chamber_rot.apply([0, -0.12, 0]), current_rot, "retract", cfg["speed"])
                current, current_rot = world_hand()
                for _ in range(30):
                    if finished:
                        break
                    step(current, current_rot, "observe")
            if chamber[11] and int(chamber[10]) == 5:
                reason = "scored"
            else:
                reason = "threshold_not_reached" if drive_ok else "base_alignment_failed"

        report = dict(
            success=reason == "scored", reason=reason, score=int(chamber[10]),
            stage1=bool(chamber[11]), peak_deg=math.degrees(peak),
            final_deg=math.degrees(-chamber[7]), initial_base=initial,
            chamber_pose=chamber[:7].tolist(), joint_limits_deg=np.degrees(chamber[8:10]).tolist(),
            steps=step_count, trace=trace,
        )
        output = Path(usr_args["save_path"]) / "stage1_lid_push.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("Stage 1:", json.dumps({k: v for k, v in report.items() if k != "trace"}), flush=True)
        return report["success"]
