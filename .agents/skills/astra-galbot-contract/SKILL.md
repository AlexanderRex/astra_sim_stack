---
name: astra-galbot-contract
description: Inspect or change Galbot observations, camera timing, ACT and GR00T model views, action semantics, frames, grippers, and base adapters in Astra.
license: Apache-2.0
---

# Galbot interface contract

Adapted from NVIDIA USD articulation, physics, and camera workflows by Renato Gasoto, under Apache-2.0. See `LICENSE.skills.md` at the repository root. Project-specific model-view guidance is added here. Architecture diagrams are proposals; the pinned task code and runtime metadata define today's executable interface.

## Inspect before mapping

Read `third_party/task_repo/robofinals/modules/robots/galbot/{galbot,assets_cfg}.py`, `policy/base.py`, the active config, action manager metadata, and one reset/step sample. Record action term names, order, dimensions, units, scale/offset, frame, quaternion layout, and validity constraints. Distinguish observations relative to default joint state from absolute joint positions.

At task commit `6e3a66c1344e926ef66b3a804058918d146cd2d0`, `Galbot-Mobile` declares:

| Terms | Configured meaning |
| --- | --- |
| Left arm, right arm | Two absolute EEF poses for differential IK, seven numbers each; not seven joint targets |
| Left hand, right hand | One binary command each; negative closes, nonnegative opens |
| Wheels | Four actuated wheel velocities in `GALBOT_WHEEL_JOINTS` order; action scale `0.01` |

The configured total is 20, but obtain it from server metadata. The fixed-base variant declares 16. Torso/leg joints are held in a standing pose and are not exposed as independent policy action terms in this config. Head/torso joints can appear in observations without being controllable action dimensions.

The proposed `[base twist | torso | 7+7 joints | grippers]` is a different contract even if its vector has the same length. Do not reinterpret it by reshaping. Either keep the current EEF/wheel contract for both policies or implement an explicit, tested canonical-to-environment adapter. Exposing new controls requires a separate supported environment API change; do not silently change the shared competition simulator.

## Frames and control behavior

Verify EEF reference frame, TCP offsets, quaternion ordering, and conversions against installed code. This task mixes `wxyz` frame-transformer offsets with `xyzw` DiffIK offsets. Do not apply a global quaternion convention across APIs.

Teleoperation also writes base root velocity through `_write_galbot_base_velocity`; its wheel vector alone may not reproduce base motion through the policy action manager. Check measured base displacement from policy `step()` commands. Do not copy teleop root-state writes into a submitted policy as an assumed allowed control channel.

Separate physical wheel speed from the pre-scale action value. Calibrate any body-twist conversion using wheel geometry, ordering, scaling, and measured odometry; a sum/difference formula in teleop is not proof of a physical velocity contract.

## Cameras and model views

Resolve actual observation keys into explicit head/left-wrist/right-wrist mappings. Inspect captured frames, dtype, channel order, effective intrinsics, and camera timestamps. The inspected config requests head capture at 30 Hz and wrists at 20 Hz, while physics `dt=1/200` and decimation 4 suggest 50 Hz control before overrides. Repeated frames are therefore expected. Record frame age and skew; do not infer capture time solely from the current control tick. Verify the wrists' intrinsic source size versus output resolution before applying crop/resize transforms.

Keep a canonical schema of permitted raw-unit fields and separate model views. ACT gets its configured camera tensor and ACT checkpoint preprocessing/statistics; GR00T gets the raw images/state expected by its own modality config and processor. Do not apply ACT normalization to GR00T or invert processor transforms twice. Derive image layouts, action representation, and chunk widths from the selected interfaces rather than diagram constants. Decode both policies into the same physical action semantics before environment mapping; document how relative actions are anchored and how rotations are represented.

## Evidence

Create a machine-readable schema with a semantic version/hash and per-field names, ordering, representation, units, frame, valid ranges, transforms, and source revision. Unknown entries stay explicitly unresolved. A zero vector contains invalid pose quaternions and must not be called a hold action. Build hold targets from current valid poses with deliberate gripper state.

Validate identity orientation, named-axis translations, gripper sign, wheel ordering/scaling, device/batch shape, and measured motion. Check that a dimension permutation or a joint/EEF schema mismatch is rejected even when tensor shapes match. Record runtime probes separately from static deductions.

Check camera ordering, RGB range, missing/stale streams, transform round trips, and processor parity on a recorded sample. Keep evaluator-only or privileged expert state outside deployable model inputs unless the participant API explicitly permits it.

## Sources

- [Galbot task implementation](https://github.com/rocochallenge/ecm2027_sim_track/blob/6e3a66c1344e926ef66b3a804058918d146cd2d0/robofinals/modules/robots/galbot/galbot.py)
- [NVIDIA USD articulation](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/usd-articulation/SKILL.md)
- [NVIDIA physics](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/physics-simulation/SKILL.md)
- [NVIDIA camera](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-camera/SKILL.md)
