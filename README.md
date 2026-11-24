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

GR00T fine-tuning needs the demonstrations as a LeRobot v2.1 dataset. `scripts/convert_dataset.py`
builds one from the folder that holds the batch folders (needs [uv](https://docs.astral.sh/uv/),
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
