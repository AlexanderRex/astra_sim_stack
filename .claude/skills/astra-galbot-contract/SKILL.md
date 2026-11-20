---
name: astra-galbot-contract
description: Inspect Galbot action terms, coordinate conventions, camera timing, and differences between teleoperation and policy control.
license: Apache-2.0
---

# Galbot interface

Read `third_party/task_repo/robofinals/modules/robots/galbot/galbot.py`, its sibling `assets_cfg.py`, and `third_party/task_repo/policy/base.py`. The facts below describe task repo revision `6e3a66c1344e926ef66b3a804058918d146cd2d0`; check the active config and runtime metadata before applying them elsewhere.

- `Galbot-Mobile` declares 20 action values: left/right absolute EEF poses (seven each), two binary gripper commands, and four wheel velocity commands. The fixed-base variant declares 16. EEF poses are not seven joint targets; get the active action ordering and dimension from server metadata.
- Negative gripper actions close; nonnegative actions open. Wheel actions use `GALBOT_WHEEL_JOINTS` order and scale `0.01`; distinguish the input value from the applied joint velocity.
- Torso/leg joints hold their configured pose and are not independent action terms in this config. A joint appearing in observations does not imply a corresponding control input.
- `joint_pos_rel` and `joint_vel_rel` observations are relative to defaults; check the definition before treating them as absolute state.
- Frame-transformer offsets use `wxyz`; DiffIK offsets use `xyzw`. Verify TCP offsets, reference frame, and quaternion ordering at each API boundary.
- Teleoperation also calls `_write_galbot_base_velocity` to write root velocity, in addition to generating wheel actions. Verify measured base movement through policy `step()`; replaying only the wheel vector may behave differently.
- Camera streams are `first_person_camera`, `left_hand_camera`, and `right_hand_camera`. Resolve the actual observation keys explicitly rather than selecting the first prefix match.
- The head camera is configured at 30 Hz, wrists at 20 Hz, and physics at `dt=1/200` with decimation 4 (50 Hz control before overrides). Repeated frames are expected; use capture timestamps when available to distinguish old frames from stalled sensors.
- Wrist intrinsics are constructed for a 400x224 source size while requested output is 224x224. Check effective calibration and output before assuming an intended crop. Inspect actual RGB frames and dtype/channel order.
- An all-zero EEF vector has invalid quaternions. A hold command needs current valid poses and an intentional gripper state. Check frames, gripper sign, action scale, and device/batch shape with small measured movements.

Adapted from NVIDIA USD articulation, physics, and camera guidance by Renato Gasoto under Apache-2.0; see `LICENSE` at the repository root. Upstream references: [articulation](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/usd-articulation/SKILL.md), [physics](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/physics-simulation/SKILL.md), [camera](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-camera/SKILL.md).
