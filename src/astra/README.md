# astra

Runtime policy stack for the ChamberServiceTask (Galbot-Mobile in the Isaac Lab sim). It covers
layers 2 to 6 of the ACT / GR00T N1.7 switchable system:

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
policies/       4  router picks ACT | GR00T | scripted wipe | hold → ActionChunk (T, 23) @ 50 Hz
  ▼
postprocess/    5  chunk executor (chunk time → control ticks) → safety filter → codec → SimAction (19)
  ▼
control/        6  task_env.step(); DiffIK, binary grippers and base velocity run in the env
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
| `diagrams/make_diagram.py` | Renders the system diagram (`astra_policy_diagram.pdf` / `.png`, not tracked in git) |

## Vector layouts

| Vector | Layout |
|---|---|
| Sim action (19) | `[L EEF pose 7 (xyz + quat, base frame, absolute) \| R EEF pose 7 \| L grip \| R grip \| base vx, vy, wz]` |
| Model action (23) | `[left_eef 9 (xyz + rot6d) \| right_eef 9 \| gripper 2 (0 open, 1 closed) \| base_vel 3]` |
| State (34) | `[left_eef_9d \| right_eef_9d \| left_arm_q 7 \| right_arm_q 7 \| gripper 2]` |

ACT and GR00T both use the model action space. `postprocess/action_codec.py` is the only place that
maps it onto the sim action. The env also takes the legacy 20-value form of the older demos (four
wheel values in place of the base velocity); the codec can emit either (`sim_action_format`).

## Sim facts the design depends on

These were checked in the task repo (`third_party/task_repo`) and in the dataset. The action layout,
the episode limit and the wipe count follow the upstream branch
`opensource/chamber-service-teleop-eval`, which is not merged or pinned here yet. The pinned revision
still has 20-value wheel actions, a 300 s limit and 3 wipe contacts, and its `step` returns five
values instead of three.

- **Grippers.** `BinaryJointPositionAction` closes when the value is **< 0**. Send −1 to close and
  +1 to open, never 0 for close.
- **Arm commands.** DiffIK takes **absolute** EEF poses in the robot root frame.
- **Base commands.** Body-frame `[vx, vy, wz]` in m/s, m/s, rad/s. The env smooths them (alpha 0.1
  per control step). Legacy wheel values are projected to the same three and not smoothed again.
- **Start pose.** The base is placed at random on reset: ±0.10 m in x and y, ±0.5 rad in yaw.
- **EEF observations are in the world frame.** `eef_pos` / `eef_quat` must not be used as state.
  The adapter computes base-frame poses with forward kinematics instead.
  Their frame is also turned about 180° from the frame the DiffIK command uses (a constant offset,
  measured on the demos), so an observed orientation never equals a commanded one.
- **Gripper observations.** `gripper_pos` reads "the last two joints", which are right gripper
  fingers: it stays at zero when only the left gripper closes. Read the gripper joints from
  `joint_pos` instead. Measured in the sim: the left gripper drives joints 65, 66, 67, 74 (to
  +1.703) and 68, 73 (to −1.703); the right one 69, 70, 71, 76 and 72, 75.
- **Joint positions are absolute.** Both observation groups carry a `joint_pos`: relative to the
  default pose in `policy`, absolute in `embodiment_general_obs`. Merged the way `BasePolicy`
  merges them, the absolute one wins.
- **Noisy joints.** `joint_pos` has 77 entries, about 40 of them passive rollers that are pure
  noise. Build the state by joint name only.
- **Rates.** Control runs at 50 Hz (physics 200 Hz, decimation 4). The demonstrations hold one row
  per control step (measured on three episodes: 19.4 ms per step), so they are 50 Hz too; their
  LeRobot export is labelled 30 fps, which is wrong. Episodes are limited to 900 s of sim time.
- **Stage flags are chained.** Each of the 6 flags requires the previous one, so the orchestrator
  retries and never skips.
- **Nothing comes back but observations.** `step` returns `(observation, terminated, truncated)`:
  no reward, stage flags or score. The policy has to track its own progress.
- **The lid is a hinge, not a loose part.** It is `Lid001_joint` on the chamber, range −90° to 0°,
  normalized so 0 = closed and 1 = open. Stage 1 needs it at ≥ 0.4 (about 36°). Stage 6 needs it at
  ≤ 0.005 (about 0.45°). The policy can't observe the lid angle, so the orchestrator estimates it:
  from the hand's rotation while holding the handle (kinematics), and from vision after the release.
- **Wiping.** The sim needs at least 5 separate sponge–chamber contacts of at least 0.5 s each,
  counted only after stage 3; each one scores on its own. This is why `policies/scripted_wipe.py` exists.
- **Synchronous stepping.** The env steps in lockstep with the policy, so inference time does not
  cost sim time.
- **Frozen sample.** The root-level LeRobot export of the dataset has a frozen
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

Render the system diagram (needs matplotlib; the output is gitignored):

```bash
python3 src/astra/diagrams/make_diagram.py [OUTPUT_DIR]    # writes astra_policy_diagram.pdf / .png
```

## Before first run

- [ ] Fill `schema.yaml: joint_names` (dump `robot.joint_names` once from `start_sim.sh --shell`).
- [ ] Export a URDF for forward kinematics and set `kinematics.urdf_path`.
- [ ] Verify `action_quat_order`: command the current FK pose and check that the arm stays still.
- [ ] Set checkpoint paths in `runtime.yaml`.

## Open questions

- **Base-frame calibration.** Region centres, the hinge axis and the wipe waypoints are fixed in
  the base frame, but the demos drive the base (in the three episodes we checked it ends 1.6 to 1.9 m
  from where it started). They hold only while the base stands where it stood at calibration. Otherwise they need
  the chamber pose relative to the base, which this stack does not estimate yet. The random start
  pose of the base (±0.5 rad yaw) makes this necessary from the first tick.
- **Base smoothing in old and new demos.** Demos recorded as 20 values store the smoothed base
  command; 19-value recordings store the raw one. Mixing them in one training set needs one
  convention (smooth the new ones offline, or decode as `wheels20`).

## Suggested implementation order

Before any of it, replay one demo's recorded actions open loop with the task repo's
`scripts/run_replay_eval.sh DATASET.hdf5 action` (from `scripts/start_sim.sh --shell`). That checks
the action contract, the base motion and the data rate without a model or FK.

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
- **Stage labels come from the orchestrator.** Label the demos with the orchestrator's own stage
  boundaries (the same detectors run over the recorded episodes), not with the sim flags. Otherwise
  the stage a policy is conditioned on at runtime differs from the one it was trained with.
- **rot6d** means the first two **rows** of the rotation matrix, as in GR00T N1.7.
- **Time is sim time** (`tick / control_hz`), never wall-clock time.
