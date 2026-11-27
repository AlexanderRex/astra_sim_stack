# astra

Runtime policy stack for the ChamberServiceTask (Galbot-Mobile in the Isaac Lab sim). It covers
layers 2 to 6 of the ACT / GR00T N1.7 switchable system:

1. Observation adapter
2. Task orchestrator
3. Router and policies
4. Action post-processor
5. Sim action interface

The offline training pipeline is not here yet.

The standalone RGB workbench localizer runs through `src/camera_localization.yml`.
See [camera localization](../../docs/camera-localization.md) for the sensor contract and runtime check.

**Status: implemented, not yet run against the sim.** Every layer is implemented and covered by
unit tests and fake-env episodes (`src/tests`). The ACT and GR00T wrappers are written against
LeRobot 0.5.1 and the Isaac-GR00T `Gr00tPolicy` API, but no checkpoint has been loaded yet.

## Data flow (one 50 Hz control tick)

```text
env server (layer 1, provided)
  │ raw obs: joint_pos (77), joint_vel, eef pose (world frame), 3 RGB cameras, last action
  ▼
adapter/        2  raw obs → CanonicalObs (34-D state: arm joints, grippers, world EEF) → ACT / GR00T views
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
| `../tests/` | Unit tests and fake-env episodes (`python3 -m pytest src/tests`) |
| `diagrams/make_diagram.py` | Renders the system diagram (`astra_policy_diagram.pdf` / `.png`, not tracked in git) |

## Vector layouts

| Vector | Layout |
|---|---|
| Sim action (19) | `[L EEF pose 7 (xyz + quat, base frame, absolute) \| R EEF pose 7 \| L grip \| R grip \| base vx, vy, wz]` |
| Model action (23) | `[left_eef 9 (xyz + rot6d) \| right_eef 9 \| gripper 2 (0 open, 1 closed) \| base_vel 3]` |
| State (34) | `[left_arm_q 7 \| right_arm_q 7 \| gripper 2 \| left_eef_world 9 \| right_eef_world 9]` |

`scripts/convert_dataset.py` writes the training data in exactly these two model layouts, and its
`--self-test` compares them with `types.py`.

ACT and GR00T both use the model action space. `postprocess/action_codec.py` is the only place that
maps it onto the sim action. The env also takes the legacy 20-value form of the older demos (four
wheel values in place of the base velocity); the codec can emit either (`sim_action_format`).

## Sim facts the design depends on

These were checked in the task repo (`third_party/task_repo`, pinned to upstream `main`) and in the dataset.

- **Grippers.** `BinaryJointPositionAction` closes when the value is **< 0**. Send −1 to close and
  +1 to open, never 0 for close.
- **Arm commands.** DiffIK takes **absolute** EEF poses in the robot root frame.
- **Base commands.** Body-frame `[vx, vy, wz]` in m/s, m/s, rad/s. The env smooths them (alpha 0.1
  per control step). Legacy wheel values are projected to the same three and not smoothed again.
- **Start pose.** The base is reset around the pose the demonstrations start from, (0.40, −0.85)
  with 90° yaw: ±0.10 m in x and y, ±10° in yaw.
- **EEF observations are in the world frame.** They do not say where the hands are relative to the
  robot; the arm joint angles do, and those are in the state. The world poses go into the state as
  two separate groups, because they are the only hint of where the robot stands in the room.
  Their frame is also turned about 180° from the frame the DiffIK command uses (a constant offset,
  measured on the demos), so an observed orientation never equals a commanded one.
- **Gripper observations.** `gripper_pos` holds the left and right driven gripper joints, absolute:
  0 open, 1.703 closed. They are joints 65 and 69 of `joint_pos`.
- **Joint positions are relative.** The service sends `joint_pos` minus the default pose. The
  adapter adds `JOINT_DEFAULT_POS` from the task repo's `robofinals/distributed/chamber_schema.py`,
  so everything after it works with absolute angles, as the dataset does. That file also lists the
  77 joint names.
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
- **Wiping.** One point for picking the sponge up, then one for each of the five cavity faces
  (four walls and the floor) that the sponge touches for at least 0.5 s, counted only after
  stage 3. This is why `policies/scripted_wipe.py` exists.
- **Score.** 13 points: lid 2, extract 1, pack 1, wipe 6, replacement 2, close lid 1.
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

`policy.py` imports `policy.base` from the task repo, which is on `PYTHONPATH` inside the policy
container. Each episode writes a summary (stage times, retries, clamps, holds, events) to
`eval_result/test_<n>/astra_episode.json`. That is the policy's own estimate; the simulator's metrics
are the authority.

LeRobot (for ACT) and Isaac-GR00T are not in the policy image yet: add them to
`src/requirements.txt` (the task repo pins `lerobot==0.5.1` as an extra) before setting a checkpoint.
Isaac-GR00T cannot share the policy image's Python environment (it pins numpy 1.26.4 and torch 2.9;
the task repo needs numpy 2).

Until then, `scripts/groot_load_test.sh [CHECKPOINT_DIR]` checks a GR00T checkpoint against this
runtime in Isaac-GR00T's own image, with networking off: weights, a prediction through
`ObservationAdapter` and `Gr00tPolicyBackend`, output ranges, latency and GPU memory. The versions it
uses are pinned in `docker/versions.env`; it needs Cosmos-Reason2-2B in a local Hugging Face cache
and prints the download command if it is missing. The images are synthetic, so it proves loading
and interfaces, not task performance.

Tests run without the sim, Isaac or a GPU (needs numpy, PyYAML, OpenCV, torch and pytest):

```bash
python3 -m pytest src/tests
```

If the task repo's `policy.base` cannot be imported (submodule missing, or mediapy not installed),
the tests register a minimal stand-in for `BasePolicy`.

Render the system diagram (needs matplotlib; the output is gitignored):

```bash
python3 src/astra/diagrams/make_diagram.py [OUTPUT_DIR]    # writes astra_policy_diagram.pdf / .png
```

## Before first run

- [ ] Fill `schema.yaml: joint_names` (dump `robot.joint_names` once from `start_sim.sh --shell`),
      or list the leg joints of the FK chain in `kinematics.joint_ids`.
- [ ] Export a URDF for forward kinematics and set `kinematics.urdf_path` (needed by the detectors
      and the hold action, not by the model input). FK reads the URDF with its own small parser
      (fixed, revolute, continuous and prismatic joints).
- [ ] Verify `action_quat_order` and `obs_quat_order`: command the current FK pose and check that
      the arm stays still.
- [ ] Set checkpoint paths in `runtime.yaml` (relative paths resolve against that file).
- [ ] Optional calibration files (formats in the module docstrings): `detectors.regions_calibration`
      (`event_detectors.py`), `detectors.hinge_calibration`, `scripted_wipe.waypoints`
      (`scripted_wipe.py`). The stage classifier is a TorchScript file
      (`checkpoints.stage_classifier`, contract in `stage_classifier.py`).

## What runs with what is configured

The stack degrades rather than stopping when parts are missing:

| Missing | Effect |
|---|---|
| ACT and GR00T checkpoints | The router stays on HOLD: the robot holds its pose (needs FK). Without FK as well, `eval` refuses to start. |
| URDF (FK) | Base-frame EEF poses repeat the last command. No contact proxy, so wipe contacts are not detected (the scripted wipe's completed presses count instead), and there is no hold pose before the first command. |
| Stage classifier | Vision terms in the stage predicates are skipped, stability windows count ticks since the release, stalls are not detected (budget and failure retries remain). |
| Region calibration | Region tests pass; EXTRACT_PART uses a 10 cm lift. WIPE hands over to the script on contact (or after holding the sponge 2 s without FK). |
| Hinge calibration | No slip check on the kinematic lid estimate. |
| Waypoint file | The scripted wipe presses a grid around the hand-over position, assuming a horizontal surface there. |

Retries come from the stage budget, classifier stalls and failure predicates (empty grasp, drop,
lid not open after release, part released over the wrong region). After `max_retries_per_stage` the
orchestrator keeps going without flushing.

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
