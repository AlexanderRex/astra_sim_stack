# astra_sim_stack

Isaac Sim 6.0.1 and Isaac Lab in Docker for the task repo
([`third_party/task_repo`](third_party/task_repo)).

## Requirements

- Linux x86_64 with an NVIDIA GPU and driver 570.169 or newer (`nvidia-smi` works).
- Docker with Buildx and Compose v2.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), registered as a Docker runtime.
- Git LFS (scene and robot assets).
- An X server and `xhost` (package `x11-xserver-utils`) for the Kit window.
- About 100 GB free disk for the build. The base image `nvcr.io/nvidia/isaac-sim:6.0.1` is public on NGC; no login is needed.

## Usage

```bash
./scripts/build_sim.sh   # submodules, LFS assets, then the images
./scripts/start_sim.sh   # opens Kit with the scene, then serves policies on 127.0.0.1:50000
```

`./scripts/start_sim.sh --headless` starts the server without the Kit window: about 2.5 times more
steps per second, so use it for evaluation runs and keep the window for debugging.
`./scripts/start_sim.sh --shell` opens a shell in the server's container, or in a new one.

`third_party/task_repo` is mounted into the container at `/workspace/task`, so edits to it
need no rebuild. Recordings and eval results are written into that directory.
Rebuild with `./scripts/build_sim.sh` after changing its dependencies, `docker/`, or the submodule commits.
Versions and image names are in [`docker/versions.env`](docker/versions.env).

## Dataset conversion

### Prepare dataset conversion metadata

Prepare dataset metadata by using `compute_stage_flags.py` script. The script goes through all recorded robot demos and works out, for every moment in each one, how much of the task the robot has finished so far. The task has six steps:

- open the lid
- lift the old part out
- put it in the box
- wipe the inside of the chamber with the sponge
- put the new part in
- close the lid

The script can't rerun the simulator, so it looks at where the lid, the two parts and the sponge are in each frame. From that it decides when each step counts as done, using the same rules the simulator's task uses (for example, "lid open and not moving for 1 second").
It saves those per-frame progress labels to files. Dataset converter then attaches them to the training data, so the model knows which stage of the task it's in.
One thing it has to guess: the recordings don't say whether the sponge actually touched a wall. So it counts a wall as wiped if the sponge was close enough to it for long enough.

```bash
uv run scripts/compute_stage_flags.py <ORIGINAL_DS_PATH> <CONVERTED_DS_PATH>.work --workers 8
```
GR00T fine-tuning needs the demonstrations as a LeRobot v2.1 dataset. `scripts/convert_dataset.py`
builds one from the folder that holds the batch folders (needs [uv](https://docs.astral.sh/uv/),
which installs the script's own dependencies):

Self test the converter's own logic without any data:
```bash
uv run scripts/convert_dataset.py --self-test
```
Convert data
```bash
uv run scripts/convert_dataset.py <ORIGINAL_DS_PATH> <CONVERTED_DS_PATH> --workers 4
```

Put the final dataset together from the cases already converted, adding the step-by-step labels (which part of the task each moment belongs to).
```bash
uv run scripts/convert_dataset.py <ORIGINAL_DS_PATH <CONVERTED_DS_PATH> --merge-only
```

Check the result:
```bash
uv run scripts/validate_lerobot_v21.py <CONVERTED_DS_PATH>

```
## Policy

The policy runs in its own light container (`astra/policy`: Python, CUDA torch, the task repo's
dependencies, no Isaac). It is a gRPC client of the task repo's evaluation service, which listens
on `127.0.0.1:50000`; this is the interface submissions use.

```bash
./scripts/build_policy.sh                 # once, and after changing src/requirements.txt
./scripts/start_sim.sh [--headless]       # terminal 1: evaluation service
./scripts/start_policy.sh [src/CONFIG.yml] # terminal 2: one evaluation run
```

`./scripts/run_headless.sh [src/CONFIG.yml]` does the last two steps in one terminal: it starts the
service headless in the background, runs the policy against it and stops the service afterwards.
At the end it prints the score the service recorded; the result the policy prints is not
authoritative. The service keeps each episode's score, events and camera videos in
`third_party/task_repo/eval_result/episodes/`.

Policy code and configs live in [`src/`](src), mounted at `/workspace/src`. A config names the
policy class (`policy_name: module:Class`) and how many episodes to run (`test_num`); see
[`src/smoke.yml`](src/smoke.yml), the default, which sends zero actions for 30 s to check the
connection. Classes implement the interface in `third_party/task_repo/policy/base.py`;
[`src/grpc_runner.py`](src/grpc_runner.py) connects them to the service. Extra packages go in
`src/requirements.txt`.

The service owns the env: task, scene, number of envs and seed are set in
[`docker/eval_server.json`](docker/eval_server.json), and a policy cannot change or reset them.

## CPU pinning

The env server keeps 9 to 16 cores busy with spinning worker threads, and that slows its own
simulation step. Pinning the sim container to a few cores makes it faster. On an i9-14900HX laptop
(8 P-cores, 16 E-cores; headless, three cameras, smoke policy) the server ran 7.8 to 11.4 steps/s on
all 24 cores and 14.5 to 15.8 steps/s on four cores (four P-cores, or two P-cores and two E-cores).
On two cores Kit did not finish starting.

```bash
./scripts/bench_sim.sh all 0-3 0-7   # steps/s for each CPU set; 1 to 2 minutes per set
export SIM_CPUSET=0-3                 # start_sim.sh and run_headless.sh then pin the server
```

`SIM_CPUSET` takes Docker's `--cpuset-cpus` syntax; unset, the server uses all CPUs.
`lscpu -e=CPU,CORE,MAXMHZ` lists the CPUs; on Intel hybrid CPUs the P-cores have the higher MAXMHZ.
These numbers come from one laptop, so measure your machine with `bench_sim.sh` before choosing a set.

## Known issues

- **First build is long.** It pulls Isaac Sim (tens of GB) and builds three images. Later runs reuse
  `astra/isaac-lab-slim` until `docker/isaac-lab-slim.Dockerfile`, the Isaac Sim version or base
  image in `docker/versions.env`, the task repo's `runtime-constraints.txt` or the Isaac Lab commit
  change. The full base image is deleted
  after the slim one is built; `docker builder prune --all` frees its build cache.
- **First start is slow.** Kit compiles RTX
  shaders on the CPU for 5 to 15 minutes, logging little besides `carb.tasking is likely stuck`; that is
  not a hang. The caches persist in Docker volumes. On laptops, switch the CPU profile from powersave.
- **Host user must be uid 1000.** Both containers run as uid 1000: the Isaac Lab image fixes it for
  `isaaclab`, and the policy matches it. Both write into the mounted
  `third_party/task_repo`; with another host uid, they cannot write recordings or results there.
- **The task repo is patched locally.** `build_sim.sh` applies
  [`docker/patches/task-repo-17159e5-import-fixes.patch`](docker/patches/task-repo-17159e5-import-fixes.patch) to the
  submodule's working tree to fix import errors, so `git status` there shows it modified.
- **`xhost +local:`** lets every local user's process open windows on your X display.

## Development base controller

`src/base_drive.yml` configures a standalone point-to-point base policy. Set its
waypoints and reference frame, then run `./scripts/run_base_drive.sh`. The default
only holds the initial position. The launcher requires existing images/assets,
does not install dependencies or pull/build images, and applies an opt-in observation patch to the task
submodule. Normal runs keep the extra telemetry disabled.

The policy uses measured base pose and velocity, preserves base-frame arm hold
targets, and requires a stationary dwell at each waypoint. It writes per-run
results and a sampled trace to `eval_result/test_*/base_drive.json` in the task
submodule. Timeouts, lack of progress, stale telemetry, and environment termination
are failures. Controller defaults and world/relative waypoints are documented in
`src/base_drive.yml`; units are metres and radians. This is a development
simulation-state baseline, not camera localization or obstacle avoidance.

Offline checks: `python -m pytest tests/test_base_drive.py -q`.

The wall-time deadline is checked between steps; it cannot interrupt a blocked
IPC call. Runtime startup may still request uncached simulator resources; this
launcher is not a network-isolation mechanism.

Precision control defaults are 5 mm and 0.5 degrees, measured from development
simulation-state telemetry. The controller captures inside 40% of each tolerance
and requires the full stationary dwell within the outer tolerances. A slow
approach with stalled progress can capture inside 80% of the tolerances. Bounded local
integral correction compensates steady bias; stall detection scales to the requested
precision. This does not establish camera localization or end-effector accuracy.

Validation: three camera-free retreat/return/turn episodes passed both limits
(see `tests/base_drive_validation.json` and `src/base_drive_precision_clearance.yml`).
Earlier forward approaches failed at some starts; arbitrary approach routes remain unvalidated.

### Base controller files

| File | Role |
|---|---|
| `src/base_drive.py` | Waypoint controller, movement limits, settling checks, and rollout logic |
| `src/base_drive_policy.py` | Evaluator entry point that saves per-episode reports |
| `src/base_drive.yml` | Hold-position check and default controller settings |
| `src/base_drive_precision_clearance.yml` | Three-episode retreat, return, and turn validation |
| `scripts/run_base_drive.sh` | Headless launcher that enables development telemetry |
| `docker/patches/task-repo-base-drive-observations.patch` | Optional base and arm-hold observations for the simulator |
| `tests/test_base_drive.py` | Controller and action/observation contract tests |
| `tests/base_drive_validation.json` | Measured position and heading errors from the validation run |
