# Galbot workbench localization from RGB

`WorkbenchLocalizer` reports the positions of the tabletop, enclosure, and sponge in meters relative to `base_link`. It takes RGB frames from the head and wrist cameras, absolute joint angles, and calibration. Galbot forward kinematics gives the pose of each camera's optical frame. `read_sensors()` lists the observations the localizer reads and adds the default pose to the joint angles, which the evaluation service sends relative to it.

## File map

| File | Role |
|---|---|
| [`src/camera_localization.yml`](../src/camera_localization.yml) | Camera, scene, and viewpoint search settings |
| [`src/camera_localization_policy.py`](../src/camera_localization_policy.py) | Policy entry point, target confirmation, and result export |
| [`src/astra/perception/__init__.py`](../src/astra/perception/__init__.py) | Perception package |
| [`src/astra/perception/galbot.py`](../src/astra/perception/galbot.py) | RGB and encoder input, forward kinematics, and camera projection |
| [`src/astra/perception/galbot_calibration.json`](../src/astra/perception/galbot_calibration.json) | Galbot camera parameters and mounting transforms |
| [`src/astra/perception/localization.py`](../src/astra/perception/localization.py) | Tabletop, enclosure, and sponge detectors in the robot base frame |
| [`src/astra/perception/view_search.py`](../src/astra/perception/view_search.py) | Viewpoint search using RGB and the estimated tabletop pose |
| [`tests/test_camera_localization.py`](../tests/test_camera_localization.py) | Projection, recognition, and input contract tests |

## Running the check

Run the following with the Astra stack built and RGB cameras enabled.

```bash
SIM_CPUSET=0,2,4,6 scripts/run_headless.sh src/camera_localization.yml
```

The policy holds the arms using encoder feedback and analyzes images every ten steps. It rotates the base in place to find the table. Once the table is visible, the robot faces it and can move along its visible edge while maintaining clearance. Translation stops if the table is lost. Success requires detection of all three targets together in two consecutive observations, with a zero base-velocity command between them.

`third_party/task_repo/eval_result/test_0/localization.json` contains measurements and velocity commands. The adjacent PNG files show the detected outlines. For control, use targets from a single `samples` entry so their coordinates refer to the same base pose. In a successful episode, `best` contains the last observation in which all three targets were confirmed.

Use the localizer directly with HWC `uint8` RGB arrays and a vector of absolute joint angles in radians.

```python
from astra.perception.localization import WorkbenchLocalizer

localizer = WorkbenchLocalizer(table_height_m=0.72)
objects = localizer.locate({"head": head_rgb, "right_wrist": wrist_rgb}, joint_angles_rad)
if "chamber" in objects:
    xyz_m = objects["chamber"].base_xyz_m
```

`from_observation()` accepts the standard Galbot observation. For a different camera set, pass `GalbotCalibration(path=...)` with measured intrinsics, distortion, and mounting transforms. Match the encoder angles to the frame capture time. For a physical table, measure the work surface height relative to the base in the robot's operating stance.

## Estimates

| Object | Reference point in `base_xyz_m` | Additional output |
|---|---|---|
| `table` | Center of the tabletop | Heading of the long axis with a 180° period |
| `chamber` | Origin of the enclosure coordinate frame | Silhouette heading with a 90° period |
| `sponge` | Center of the visible top surface | Source camera and outline in pixels |

The estimator uses the known geometry and appearance of the workspace in `scene04_1.usd`. It tests straight tabletop edges and the surface inside the candidate outline against the expected dimensions of 1.80 × 1.20 m. It locates the enclosure by projecting its dimensions into the RGB image and comparing the lid outline and the appearance of light and dark panels. It locates the sponge by color and physical size. The calibrated tabletop height is 0.72 m relative to the base. Z estimates use that plane and known object dimensions. The `score` measures how closely a hypothesis matches the image. A target with insufficient visible evidence is omitted. The enclosure is reported only when at least 90% of its outline is in the frame; a view that cuts off more does not fix its position, so the target is omitted until the robot turns toward it.

The nearly square enclosure has four plausible headings from its silhouette. Localization provides a position for approaching the workspace. A precise insertion under the handle requires a separate visual estimate of the handle side.

## Initial validation before viewpoint search

| RGB source | Observations | Table | Enclosure | Sponge |
|---|---:|---:|---:|---:|
| Live run from a standard start | 47 | 22 | 14 | 17 |
| Recording from another standard start, seed 137 | 43 | 23 | 14 | 18 |

Both series include frames in which the workspace completely left the field of view while the robot was turning. At step 200 of the live run, all three targets were estimated in the same observation. The head camera gave the table position `(1.265, −0.466, 0.720)` m and the enclosure position `(1.753, −0.132, 0.952)` m. The right wrist camera gave the sponge position `(0.874, −0.073, 0.731)` m.

For frames with simultaneous detection of the table and enclosure, we measured how much the enclosure position varied in the estimated table frame. The median deviations from the center of the series were 20 and 19 mm, with maximum deviations of 45 and 43 mm. These figures measure repeatability.

Absolute accuracy was measured on head-camera frames from two recordings. In the first the base turned in place through about 55° from a standard start. The second began from another start and added 0.4 m of backward and 0.4 m of sideways travel. The reference positions were computed offline from the world-frame hand poses and forward kinematics, and are themselves good to 1–2 cm.

| Recording | Frames | Table: median, largest error | Enclosure: frames with an estimate, median, largest error |
|---|---:|---|---|
| Turn in place | 60 | 0.9 cm, 1.7 cm | 17: 1.7 cm, 5.4 cm |
| Other start, turn and travel | 126 | 1.2 cm, 7.8 cm | 54: 1.9 cm, 4.0 cm |

The table was found in every frame. From a standard start less than half of the enclosure is in the head camera's view, so it is reported only after the base has turned about 10° toward it. One call takes about 0.25 s for the head camera on a laptop CPU.

The joint chain in `galbot_calibration.json` gives the same head and wrist link poses as the manufacturer's public URDF (`galbot_one_golf_description`, Apache-2.0), to numerical precision at three joint configurations.

`tests/test_camera_localization.py` covers projection, metric reconstruction, and the input contract.

## Four-sided validation

The test harness places Galbot in twenty distinct starting poses. Each side is tested at headings of −65°, −30°, 0°, +30°, and +65° relative to the approximate direction of the table. The starting position varies with the heading. The test harness alone uses the spawn parameters. The policy receives RGB frames, encoders, and calibration.

To pass, the table, enclosure, and sponge must appear together in two consecutive observations. The policy has up to 620 search steps after RGB refreshes following a reset. Saved frames are used to review the outline overlays. Checking the material inside the projected lid rejects hypotheses that include part of the floor in the enclosure.

| Side | All three targets on the first observation | All three confirmed with viewpoint search | Maximum search steps |
|---|---:|---:|---:|
| West | 0/5 | 5/5 | 460 |
| East | 2/5 | 5/5 | 100 |
| South | 2/5 | 5/5 | 120 |
| North | 2/5 | 5/5 | 110 |
| Total | 6/20 | 20/20 | 460 |

From the west, the robot needs to move along the table to see the sponge behind the enclosure. The longest search took 9.2 seconds of simulation time.
