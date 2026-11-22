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
./scripts/start_sim.sh   # env server on 127.0.0.1:50000; Kit opens its window when a policy connects
```

`./scripts/start_sim.sh --headless` starts the server without the Kit window: about 2.5 times more
steps per second, so use it for evaluation runs and keep the window for debugging.
`./scripts/start_sim.sh --shell` opens a shell in the server's container, or in a new one.

`third_party/task_repo` is mounted into the container at `/workspace/task`, so edits to it
need no rebuild. Recordings and eval results are written into that directory.
Rebuild with `./scripts/build_sim.sh` after changing its dependencies, `docker/`, or the submodule commits.
Versions and image names are in [`docker/versions.env`](docker/versions.env).

## Dataset conversion

`scripts/convert_dataset.py` exports demonstrations as a LeRobot v2.1 dataset
from the folder that holds the batch folders (needs [uv](https://docs.astral.sh/uv/),
which installs the script's own dependencies):

```bash
uv run scripts/convert_dataset.py --workers 4 <ORIGINAL_DS_PATH> <CONVERTED_DS_PATH>
```

Check the result, and the converter's own logic without any data:

```bash
uv run scripts/validate_lerobot_v21.py <CONVERTED_DS_PATH>
uv run scripts/convert_dataset.py --self-test
```
## Policy

The policy runs in its own light container (`astra/policy`: Python, CUDA torch, the task repo's
dependencies, no Isaac) and talks to the env server over `127.0.0.1:50000`.

```bash
./scripts/build_policy.sh                 # once, and after changing src/requirements.txt
./scripts/start_sim.sh [--headless]       # terminal 1: env server
./scripts/start_policy.sh [src/CONFIG.yml] # terminal 2: one evaluation run
```

`./scripts/run_headless.sh [src/CONFIG.yml]` does the last two steps in one terminal: it starts the
env server headless in the background, runs the policy against it and stops the server afterwards.
At the end it prints the simulator's own score of the run
(`third_party/task_repo/eval_result/chamber_service_metrics.json`); the result the policy prints
is not authoritative.

Policy code and configs live in [`src/`](src), mounted at `/workspace/src`. A config names the
policy class (`policy_name: module:Class`) and the env; see
[`src/smoke.yml`](src/smoke.yml), the default, which sends zero actions for 30 s to check the
connection. Classes implement the interface in `third_party/task_repo/policy/base.py`; extra
packages go in `src/requirements.txt`. Results are written to `third_party/task_repo/eval_result/`.
Each env server serves one run and then exits, so restart `start_sim.sh` before the next one.

The server hands tensors to the client through Unix sockets in Python's temp dir. Both containers
set `TEMP=/tmp/ipc` on a shared `ipc` volume, so the sockets are reachable across containers.

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
- **First start is slow.** The env server starts Kit only when a policy connects. Kit then compiles RTX
  shaders on the CPU for 5 to 15 minutes, logging little besides `carb.tasking is likely stuck`; that is
  not a hang. The caches persist in Docker volumes. On laptops, switch the CPU profile from powersave.
- **Host user must be uid 1000.** Both containers run as uid 1000: the Isaac Lab image fixes it for
  `isaaclab`, and the policy matches it to share the tensor sockets. Both write into the mounted
  `third_party/task_repo`; with another host uid, they cannot write recordings or results there.
- **The task repo is patched locally.** `build_sim.sh` applies
  [`docker/patches/task-repo-1fd0e3e-import-fixes.patch`](docker/patches/task-repo-1fd0e3e-import-fixes.patch) to the
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
