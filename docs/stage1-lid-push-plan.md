# Lid-opening validation

This simulator baseline uses the enclosure pose from the scene. The replay mode tests the base approach, lid opening, and the observation period after hand retraction. The [recorded trajectory section](#replaying-a-successful-joint-trajectory) gives the command and results for the accepted run.

## File map

| File | Role |
|---|---|
| `src/stage1_lid_push.py` | Policy entry point, base approach, observations, and reporting |
| `src/stage1_lid_replay.py` | Trajectory replay with base feedback and lid-angle validation |
| `src/stage1_lid_reference.json` | Commands and phase boundaries from the validated trajectory |
| `src/stage1_lid_push_replay.yml` | Five-episode validation with standard starts |
| `src/stage1_lid_push.yml` | Front-push experiment with a prescribed initial pose |
| `src/stage1_lid_push_side.yml` | Side-push experiment with a prescribed initial pose |
| `src/stage1_lid_push_full.yml` | Side-push experiment after an independent base approach |
| `scripts/run_lid_push.sh`, `docker/lid-push.compose.yml` | Simulator launch with RGB disabled and result export |
| `scripts/export_lid_push_reference.py` | Export of a trajectory from a successful run |
| `docker/patches/task-repo-lid-push-observations.patch` | Development observations for the lid, joints, and contact |
| `docker/patches/task-repo-lid-posture-action.patch` | Extended simulator joint commands |

## Acceptance criteria

The validation series uses five standard resets. At least four episodes must reach the simulator's opening threshold, receive a score of 5, and hold the required angle after the hand retracts. Record the simulator score and the angle during the observation window separately.

## Simulator contract

- The simulator awards 5 points when the lid reaches 40% of its hinge range.
- `_complete_stage(0, self._lid_open(env))` latches the stage flag and score when the lid opens. They persist if the lid closes later.
- `core/models/fixtures/chamber.py` documents `Lid001_joint` with a range of −90° to 0°. The closed angle is 0°, and opening moves in the negative direction. These limits put the threshold at q ≤ −36°. Check the runtime limits before a run.
- The fixture is `Chamber001_01`. The original plan called for identifying the lid body, hinge axis, contact surface, and handle prim from USD and runtime data.
- The base starts with roughly ±10 cm of XY variation and ±0.5 rad of yaw variation. Use a fixture-relative trajectory to account for these starts.
- The native action has 19 values for two absolute end-effector poses in the root frame, two gripper commands, and body-frame base velocity.
- The scripted policy follows the standalone entry point in `src/base_drive_policy.py`. `src/astra/orchestrator/stage_specs.py` provides the wider task structure.

## Original design

One hand pushes an accessible lid surface while the other stays in a clear pose. The base settles before contact. Closed gripper fingers act as a pusher where their collision geometry provides a suitable surface.

Choose a point with leverage about the hinge axis. This expression gives the opening torque for a force applied at the contact point.

`tau_open = axis_open · ((p_contact - p_hinge) × F)`.

Choose another approach side, hand, or contact point if the reachable surface produces closing torque or the hand loses contact along the arc. The experiment isolates opening by pushing.

### 1. Mechanics

- Read the world pose of the hinge axis, q and qdot, limits, damping, friction, drive targets, lid mass, and center of mass from USD or the runtime.
- Use the convex collision geometry to identify an accessible contact surface or edge.
- Check reachability and pusher orientation, including clearance between both hands and the lid and enclosure.
- Estimate the required force after reading the model. Compare `m*g*r_COM` with the contact lever arm and account for the drive and friction.

The output is a contact point in the lid frame, a reachable pre-contact pose, and an opening direction and arc.

### 2. Observations and scoring

- Enable development telemetry for the base pose, both end effectors, enclosure and hinge pose, q and qdot, simulation time, score, and stage flags.
- Record measured contact force or contact pairs when the simulator exposes them. Treat end-effector tracking error as a separate measurement.
- The baseline reads lid state from the simulator. During policy execution, opening, scoring, and hinge motion come from normal simulation and physical contact.
- Express targets in the enclosure or lid frame, then transform the end-effector target into the root frame using the current base pose.
- Start with RGB disabled. Use a separate visual run if contact needs inspection.

### 3. Scripted control with feedback

`RESET → ALIGN_BASE → PRECONTACT → CONTACT → PUSH_ARC → RETRACT → VERIFY`.

- `RESET` checks that the lid is closed and validates quaternions, action metadata, and telemetry.
- `ALIGN_BASE` takes the base to a reachable pose relative to the enclosure and brings it to a stop. The first direct route had failed from some starts.
- `PRECONTACT` moves the selected hand through a clear waypoint and holds the other hand and base.
- `CONTACT` approaches the surface in small bounded steps and checks for contact or a change in q. It monitors tracking error.
- `PUSH_ARC` follows the opening arc using the current hinge angle and limits speed, step size, and penetration. Choose an opening target with margin above 36° after checking stability and collisions.
- `RETRACT` withdraws the pusher through free space and holds the base and other hand.
- `VERIFY` checks the stage flag, score of 5, and angle after release. Full task success is outside this baseline's acceptance criteria.
- At each step, check for stale telemetry, timeouts, lack of hinge movement, large tracking error, rotation in the wrong direction, and episode termination. Limit retries and record each attempt.

### 4. Development runs

Start from one fixed pose to establish that a physical push can open the lid. Then test other starting poses and tune the contact point, hand, base pose, speed, and angle margin. Tune on a separate set of runs before the final five episodes.

Keep parameters in YAML and test against the pinned simulation environment. Record a separate rationale for changes to the solver, friction, mass, or limits.

### 5. Final five episodes

- Record the stack and submodule commits, configuration, seed and reset sequence, policy parameters, and physics backend.
- Run five standard randomized resets with the same tuning throughout.
- Save the score and stage flags for each episode before its reset. The final server metric describes the end of the series.
- Record the episode ID, start poses, end-effector path, q and qdot, contacts when available, time of crossing the threshold, retraction, and any failure reason.
- Count all five attempts. At least four episodes must score 5 and set the opening flag. Review any higher score for unintended progression into later parts of the task.
- Report the per-episode score and post-release stability alongside the logs and configuration.

The first planned experiment was to measure the hinge mechanics, select two or three contact candidates, and check the direction of their torque and hand reachability. The single fixed-start push test used simulation-state feedback with RGB disabled.

## Front-push prototype

The policy observes the enclosure pose and hinge angle. Galbot opens the lid through hand contact. From the prepared front pose, the closed fingers push upward from below the left handle bar.

The USD hinge rotates about the X axis and has local center `(0, 0.3592, 0.1650)` m. Negative rotation opens the lid. The bar is around y = −0.285 m. An upward push from below has a lever arm of about 0.644 m and produces torque in the opening direction. For an 8 kg lid and an assumed center-of-mass lever arm of 0.35 m, the initial component of force against gravity is about 43 N. The estimate uses an assumed center of mass and omits measured friction and drive forces.

The following commands use the built simulator and policy images.

```bash
SIM_CPUSET=0-3 scripts/run_lid_push.sh
SIM_CPUSET=0-3 scripts/run_lid_push.sh src/stage1_lid_push.yml --overrides --test_num 5
```

Each run saves the simulator log and a JSON report with the trace, angle, and score for each episode in `lid-push-results.*`. This configuration starts near the front edge of the enclosure to isolate the mechanics. The standard scene starts the robot at the box and requires a base approach around the table.

## Front-push results

This run used Isaac Sim 6.0.1, PhysX on CPU, Galbot-Mobile, and `scene04_1.usd` with RGB disabled. It ran five resets from the fixed base pose `(−0.62, −1.25, 0.03)` m at 90° yaw with seed 42. The policy prepares the active arm, tucks the other arm, moves the base about 19 cm, inserts the pusher under the bar, and follows the hinge arc until the check passes.

| Episode | Checker score | Peak angle, ° | Angle after retraction, ° |
|---|---:|---:|---:|
| 1 | 5 | 36.01 | 0.00 |
| 2 | 5 | 36.11 | 26.41 |
| 3 | 5 | 36.07 | 26.33 |
| 4 | 5 | 36.12 | 26.73 |
| 5 | 5 | 36.13 | 26.25 |

All five episodes received 5 points from the prepared front pose. The lid dropped below 36° after retraction in every episode. This series established repeatability of the opening motion from the prepared pose. The later replay validation covers approach from the box and a sustained opening after retraction. The local traces and final simulator metric are in `lid-push-results.JbTY1r/`.

## Approach and contact experiments

The standard route starts at the box. The robot backs away from the front edge of the table, follows it, and turns toward the enclosure. In one run, the base stopped with an error of 10.8 mm and 0.70°. Raising the insertion point and holding the base brought the lid to 35.76° from that start, with 25.91° remaining after retraction.

| Experiment | Peak angle, ° | Angle after retraction, ° |
|---|---:|---:|
| Extended front arc from a prepared base pose | 39.15 | 23.65 |
| Side push with a tilted wrist | 37.92 | 0.00 |
| Side push with base hold and limited angular lead | 23.06 | 0.00 |
| Standard start with raised insertion and base hold | 35.76 | 25.91 |
| Late wrist tilt from a prepared base pose | 36.15 | 0.91 |
| Side push with base tracking from a prepared start | 48.00 | 39.89 |

A separate physics diagnostic raised the lid to prescribed angles by changing the hinge state in small increments. After release, the simulation ran without further hinge writes while the robot stayed by the box. This diagnostic measured passive mechanics and was excluded from policy acceptance scoring.

| Release angle, ° | Angle after 3.02 s, ° |
|---|---:|
| 36 | 30.36 |
| 50 | 47.34 |
| 65 | 64.17 |
| 80 | 79.91 |
| 89.8 | 89.8 |

With a 36° scoring threshold, about 50° was a reasonable initial target for short access after release. The lid moved very little near its upper limit. The diagnostic measured three seconds of hold time.

The tested front trajectories lost contact around 39°. By 80° opening, the horizontal distance from the base to the commanded wrist pose reached about 0.95 m. Tilting the wrist lowered its target height. A side approach reduced the forward reach and required the base to follow the handle along the table while maintaining contact.

`src/stage1_lid_push_full.yml` tests the route from a standard start. `src/stage1_lid_push_side.yml` uses a prepared side pose for contact tuning. With `require_stable` and `observe_steps` configured, a pass requires the checker score and an angle of at least 36° throughout the post-retraction window.

The side push with base tracking received 5 points and held at least 39.89° during the three seconds after retraction. The local trace is `lid-push-results.KjtVmJ/test_0/stage1_lid_push.json`.

On the full side route, tucking both hands let the base travel along the table and stop at the work position. The push from that pose lost contact after peaking at 24.20°. The local report is `lid-push-results.T2peFu/test_0/stage1_lid_push.json`.

An insertion adjustment of −9 mm vertically and −13 mm along Y produced no opening from a standard side start. That experiment used a fixed offset from `link7`. The result prompted the later work on measured fingertip pose, contact detection, and retries under the handle. The report is `lid-push-results.bnUOiK/test_0/stage1_lid_push.json`.

## Fingertip feedback

Development telemetry reports the position of a point on each distal fingertip pad and the contact force magnitude between that finger and the lid. The contact filter covers the entire lid, including the handle. In a successful trace, the measured force was sometimes zero while the lid continued to rise. A retry therefore requires both an absence of measured contact and an angle that has stopped increasing.

The side policy corrects the position of the finger that loaded the lid along two transverse axes in the enclosure frame. The correction is capped at 35 mm. If the angle stalls and the measured load drops to zero, it withdraws the wrist and retries insertion up to twice. The calibration pose came from a successful side push in the same scene.

Use this command to check the full route, fingertip poses, and three-second hold.

```bash
SIM_CPUSET=0-3 scripts/run_lid_push.sh src/stage1_lid_push_full.yml
```

One standard-start episode peaked at 48.03°, held at least 43.75° after retraction, and received 5 points. The trace is `lid-push-results.zFETeM/test_0/stage1_lid_push.json`.

A standard-start series with feedback stopped early after two failures, yielding one success in three completed episodes. The base reached its waypoint in all three. In the failed episodes, the wrist stopped about 57–58 mm short of the pre-contact target, and the measured lid contact force remained zero through both retries. The reports are in `lid-push-results.fHqrNG/`.

A free-space detour tested on the second start (`lid_push:recover_arm_pose=True`) reduced wrist position error to about 2 mm. During the hold, however, the fingertip pad was at about 1.00 m instead of the target height of about 1.19 m. After the base advance, the finger was again out of position for insertion under the handle. The run scored 0. The trace is `lid-push-results.ouHBml/test_0/stage1_lid_push.json`.

These runs exposed a problem before the push. Wrist orientation and measured fingertip positions had to agree with the intended preparation pose, and the approach needed to respect the joint limits. The next experiment used the joint angles from a successful trace to evaluate reachable preparation poses. The free-space detour remains available as a parameter for reproducing the experiment.

## Wrist coordinate convention

A successful push with the original FrameTransformer reported a wrist orientation error of 179.81° and a position error of 2.7 mm. The Isaac Lab version in use takes the offset in `xyzw` order. Galbot passed `(1, 0, 0, 0)`, which is the identity in `wxyz` and a 180° rotation about X in `xyzw`. The policy used the rotated measured pose for holding and preparing the arms.

The development mode passes the identity offset in `xyzw` order. Replaying the problematic start reduced the preparation-pose error to 0.49° in orientation and 3 mm in position. The finger made contact, and the lid opened to 16.26°. With the existing insertion path, the finger stopped at the handle edge. The measured load exceeded 400 N while the lid barely moved. This motivated testing retries when the angle stalls under load as well.

The local reports include joint positions and measured orientations in `lid-push-results.UnsrP8/` for the previous offset and `lid-push-results.B0rG8H/` for the corrected offset.

A run from a previously successful start with the corrected offset reached 6.12° and scored 0. `ASTRA_LID_WRIST_XYZW=1` therefore serves as an experimental switch. The trace verifies the coordinate conversion. The original contact path was tuned with the previous offset and required a fresh validation with the corrected one.

| Experiment with corrected orientation | Peak lid angle, ° | Score |
|---|---:|---:|
| Replayed failed start with original trajectory | 16.26 | 0 |
| Same start with shifted final insertion and a retry under sustained load | 0.19 | 0 |
| Previously successful standard start with original trajectory | 6.12 | 0 |
| Failed start with a detour around the support before insertion | ≈0 | 0 |

The last experiment cleared the measured lid contact, but the wrist still fell about 9 cm short of its target. This prompted separate contact measurements between the hand links and the fixed enclosure. The detour report is `lid-push-results.sdkJms/test_0/stage1_lid_push.json`.

### Joint limit under load

During the support detour, the wrist fell 8–15 cm short of the target with no registered load from the lid or fixed enclosure on the measured hand links. The second right-arm joint repeatedly reached its upper limit of **1.608 rad**. It measured **1.104 rad** at the corresponding preparation phase of the successful trace. The report with separate enclosure-contact measurements is `lid-push-results.OFsSIF/test_0/stage1_lid_push.json`.

The opt-in `ASTRA_LID_POSTURE=1` mode adds commands for the seven right-arm joint angles during preparation. The drives respect the standard limits and limit how far the target can lead the measured position. The policy returns to ordinary wrist-pose control after preparation.

Run the following to reproduce the problematic start.

```bash
ASTRA_LID_WRIST_XYZW=1 ASTRA_LID_POSTURE=1 SIM_CPUSET=0-3 \
  scripts/run_lid_push.sh src/stage1_lid_push_full.yml --overrides --test_num 1 \
  --env_cfg:init_robot_base_pos '[1.14079392,-0.74224222,0.0]' \
  --env_cfg:init_robot_base_ori '[0,0,1.893558]' \
  --lid_push:precontact_joints '[-1.45274,1.10430,0.42425,2.32005,0.40488,-0.26232,0.37374]'
```

Preparation brought the second joint to 1.108 rad and the wrist within 7.8 mm and 1.23° of its target. After the base approached, the wrist error fell to 2.7 mm and the second joint was at 1.115 rad. During the push it again reached 1.608 rad. One fingertip-to-lid force sample exceeded 1700 N, the peak angle was 8.02°, and the run scored 0. The report is `lid-push-results.b1FxRQ/test_0/stage1_lid_push.json`.

These tests showed that the joint limit mattered throughout the contact arc. Setting the right-arm preparation pose alone yielded 0 points on the problematic start.

## Replaying a successful joint trajectory

The trajectory records targets for both arms, finger commands, base velocity, and measured base pose from `lid-push-results.q6RAyi/test_0/`. It contains 1,120 steps at 20 ms after the robot's independent approach. The export retains the push, retraction, and observation phases. Base poses are relative to the enclosure.

`ASTRA_LID_POSTURE=replay` commands joint angles for both arms and follows the recorded base path with pose feedback. After the approach, the policy waits for the arm joints and base to reach their tolerances before replaying the push. The joint drives respect their normal limits and bounds on the difference between commanded and measured angles. Physical contact moves the lid and the simulator checker awards the score. The run uses Galbot-Mobile in `scene04_1.usd` with simulator telemetry and a 34-value development action.

Export the trajectory from the source recording with the following command.

```bash
python3 scripts/export_lid_push_reference.py lid-push-results.q6RAyi/test_0 src/stage1_lid_reference.json
```

Run the five standard resets with the replay configuration.

```bash
ASTRA_LID_WRIST_XYZW=1 ASTRA_LID_POSTURE=replay SIM_CPUSET=0-3 \
  scripts/run_lid_push.sh src/stage1_lid_push_replay.yml
```

### Validation from standard starts

The series used five standard resets with seed 42 and RGB disabled. The robot started by the box in a different pose each time and drove to the enclosure independently. Four of five episodes passed. In each passing episode, the lid stayed above 36° for all 150 steps, or three seconds, after retraction.

| Episode | Initial base X, Y, and heading | Checker score | Peak angle, ° | Minimum over 3 s, ° |
|---|---|---:|---:|---:|
| 1 | 1.1400 m, −0.7356 m, 68.69° | 5 | 47.98 | 43.69 |
| 2 | 1.1408 m, −0.7422 m, 108.49° | 0 | 0.00 | 0.00 |
| 3 | 1.1348 m, −0.7183 m, 108.77° | 5 | 47.88 | 43.57 |
| 4 | 1.1284 m, −0.8474 m, 112.52° | 5 | 47.78 | 43.45 |
| 5 | 1.1328 m, −0.7356 m, 70.19° | 5 | 48.06 | 43.79 |

The summary and traces are in `lid-push-results.hPu6Qm/`. A separate run using the second episode's start pose received 5 points and held at least 44.06° after retraction. The trace in `lid-push-results.qBcCPm/` shows how contact dynamics can affect the outcome even with the same initial pose.

### Approach experiments

In the failed episode, contact during insertion turned the base by 7.45°. At the start of the push, the finger was about 7 cm deeper and 2 cm higher than in a passing episode. Additional heading alignment reduced the heading error. The wrist remained outside the required contact pose. That variant scored 5 points in two out of five episodes and stopped at alignment three times. The branch uses the commands from the version that passed four out of five. The experiment traces are in `lid-push-results.5F6RAH/`.

A direct transition from the travel pose to the push pose left a joint error of 0.103 rad and rotated the base by 3.22°. The readiness check stopped the run before lid contact. Approaching with the wrist held in place scored 5 points on the first episode and raised peak contact force during approach from 401 to 835 N. The fingers touched the lid again at the start of the observation window. Both experiments were stopped after their first episode. Their reports are in `lid-push-results.RXClfE/` and `lid-push-results.EDsrnH/`.
