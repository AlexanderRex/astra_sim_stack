# astra

Runtime policy stack for the SGEAI ChamberServiceTask (Galbot-Mobile in the Lightwheel / Isaac Lab
sim). It covers layers 2 to 6 of the ACT / GR00T N1.7 switchable system:

1. Observation adapter
2. Task orchestrator
3. Router and policies
4. Action post-processor
5. Sim action interface

The offline training pipeline is not here yet.

**Status: skeleton.** Classes set up their state from the config. Every behaviour method raises
`NotImplementedError`, and its docstring says what it must do and what goes in and out.

## Data flow (one 50 Hz control tick)

```text
env server (layer 1, provided)
  │ raw obs: joint_pos (77), joint_vel, eef pose (world frame), 3 RGB cameras, last action
  ▼
adapter/        2  raw obs → CanonicalObs (34-D state, base-frame EEF from FK) → ACT / GR00T views
  ▼
orchestrator/   3  stage 0-5, phase, prompt, event (advance / retry / script)
  ▼
policies/       4  router picks ACT | GR00T | scripted wipe | hold → ActionChunk (T, 24) @ 30 fps
  ▼
postprocess/    5  chunk executor (30 fps → 50 Hz) → safety filter → codec → SimAction (20)
  ▼
control/        6  task_env.step(); DiffIK, binary grippers and wheel velocity run in the env
```

`policy.py` holds `AstraPolicy(BasePolicy)`, which is the entry point the task repo's evaluation
harness loads. `eval()` runs the loop above, and its docstring has the full per-tick pseudocode.

## Layout

| Path | Contents |
|---|---|
| `types.py` | Vector layouts, enums (`Stage`, `Phase`, `PolicySource`, `OrchestratorEvent`) and the data passed between layers |
| `schema.py` | Loads `config/schema.yaml` and turns joint and camera names into indices |
| `config/schema.yaml` | Single source of truth: joint names, default joint positions, cameras, state / action layouts, quaternion order, rates |
| `config/runtime.yaml` | Checkpoint paths and the tuning parameters for each layer |
| `config/astra_eval.yml` | Config for the evaluation harness (`policy_name: astra.policy:AstraPolicy`) |
| `adapter/` | `ObservationAdapter`, `ForwardKinematics`, `HistoryBuffer`, view builders, rotation conversions |
| `orchestrator/` | `TaskOrchestrator`, `EventDetectors`, `StageClassifier`, `STAGE_SPECS` |
| `policies/` | `PolicyRouter`, `PolicyBackend`, `ActPolicy`, `Gr00tPolicyBackend`, `ScriptedWipe` |
| `postprocess/` | `ChunkExecutor`, `SafetyFilter`, `ActionCodec` |
| `control/` | `SimActionInterface` |
| `policy.py` | `AstraPolicy`, the harness entry point |
| `diagrams/make_diagram.py` | Generates `astra_policy_diagram.pdf` (system diagram, v1.0) |

## Vector layouts

| Vector | Layout |
|---|---|
| Sim action (20) | `[L EEF pose 7 (xyz + quat, base frame, absolute) \| R EEF pose 7 \| L grip \| R grip \| wheel1-4 vel]` |
| Model action (24) | `[left_eef 9 (xyz + rot6d) \| right_eef 9 \| gripper 2 (0 open, 1 closed) \| wheel_vel 4]` |
| State (34) | `[left_eef_9d \| right_eef_9d \| left_arm_q 7 \| right_arm_q 7 \| gripper 2]` |

ACT and GR00T both use the model action space. `postprocess/action_codec.py` is the only place that
maps it onto the sim action.

## Sim facts the design depends on

These were checked in `rocochallenge/ecm2027_sim_track` and in the dataset.

- **Grippers.** `BinaryJointPositionAction` closes when the value is **< 0**. Send −1 to close and
  +1 to open, never 0 for close.
- **Arm commands.** DiffIK takes **absolute** EEF poses in the robot root frame.
- **EEF observations are in the world frame.** `eef_pos` / `eef_quat` must not be used as state.
  The adapter computes base-frame poses with forward kinematics instead.
- **Gripper observations.** `gripper_pos` reads "the last two joints". Read the gripper joints by
  name instead.
- **Noisy joints.** `joint_pos` has 77 entries, about 40 of them passive rollers that are pure
  noise. Build the state by joint name only.
- **Rates.** Control runs at 50 Hz (physics 200 Hz, decimation 4). The data is 30 fps. Episodes are
  limited to 300 s.
- **Stage flags are chained.** Each of the 6 flags requires the previous one, so the orchestrator
  retries and never skips.
- **The lid is a hinge, not a loose part.** It is `Lid001_joint` on the chamber, range −90° to 0°,
  normalized so 0 = closed and 1 = open. Stage 1 needs it at ≥ 0.4 (about 36°). Stage 6 needs it at
  ≤ 0.005 (about 0.45°). The policy can't observe the lid angle, so the orchestrator estimates it:
  from the hand's rotation while holding the handle (kinematics), and from vision after the release.
- **Wiping.** The sim needs at least 3 separate sponge–chamber contacts of at least 0.5 s each,
  counted only after stage 3. This is why `policies/scripted_wipe.py` exists.
- **Synchronous stepping.** The env steps in lockstep with the policy, so inference time does not
  cost sim time.
- **Frozen sample.** The root-level LeRobot export of `ecm2027_sim_data` has a frozen
  `observation.state`. Use the raw HDF5 episodes instead.

## Running

```bash
./scripts/build_policy.sh                                  # after changing src/requirements.txt
./scripts/start_sim.sh --headless                          # terminal 1: env server
./scripts/start_policy.sh src/astra/config/astra_eval.yml  # terminal 2: one evaluation run
```

Until the stubs are implemented, `AstraPolicy` raises `NotImplementedError` as soon as the harness
constructs it. `policy.py` imports `policy.base` from the task repo, which is on `PYTHONPATH`
inside the policy container.

Regenerate the system diagram:

```bash
python3 src/astra/diagrams/make_diagram.py [OUTPUT_DIR]    # writes astra_policy_diagram.pdf / .png
```

## Before first run

- [ ] Fill `schema.yaml: joint_names` (dump `robot.joint_names` once from `start_sim.sh --shell`)
      and the arm entries of `default_joint_pos` (from `assets_cfg.py`).
- [ ] Export a URDF for forward kinematics and set `kinematics.urdf_path`.
- [ ] Verify `action_quat_order`: command the current FK pose and check that the arm stays still.
- [ ] Set checkpoint paths in `runtime.yaml`.

## Suggested implementation order

1. `schema.py`, `adapter/rotations.py`, `adapter/kinematics.py`. Validate FK against
   `obs["actions"]` with the arm still.
2. `postprocess/action_codec.py`, `postprocess/safety.py`, `control/sim_interface.py`, plus a hold
   policy. Check end to end that the robot holds still.
3. `adapter/observation_adapter.py` and `views.py`, `postprocess/chunk_executor.py`,
   `policies/act_policy.py`. The orchestrator stays disabled (`orchestrator.enabled: false`).
4. `orchestrator/event_detectors.py`, then `stage_specs.py` / `task_orchestrator.py`, then
   `stage_classifier.py`.
5. `policies/scripted_wipe.py`, `policies/groot_policy.py`, `policies/router.py` switching.

## Conventions

- **No hard-coded indices.** Everything goes through `schema.yaml` / `Schema`.
- **Normalization belongs to the models.** Views are never normalized; each checkpoint's own
  processor does it.
- **Shared with the offline builder.** The future dataset builder must import the same code: the
  adapter, `rotations.py` and `ActionCodec.encode`. That keeps training and runtime inputs identical.
- **rot6d** means the first two **rows** of the rotation matrix, as in GR00T N1.7.
- **Time is sim time** (`tick / control_hz`), never wall-clock time.
