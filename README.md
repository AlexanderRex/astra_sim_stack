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

## Policy

The policy runs in its own light container (`astra/policy`: Python, CUDA torch, the task repo's
dependencies, no Isaac) and talks to the env server over `127.0.0.1:50000`.

```bash
./scripts/build_policy.sh                 # once, and after changing src/requirements.txt
./scripts/start_sim.sh [--headless]       # terminal 1: env server
./scripts/start_policy.sh [src/CONFIG.yml] # terminal 2: one evaluation run
```

Policy code and configs live in [`src/`](src), mounted at `/workspace/src`. A config names the
policy class (`policy_name: module:Class`) and the env; see
[`src/smoke.yml`](src/smoke.yml), the default, which sends zero actions for 30 s to check the
connection. Classes implement the interface in `third_party/task_repo/policy/base.py`; extra
packages go in `src/requirements.txt`. Results are written to `third_party/task_repo/eval_result/`.
Each env server serves one run and then exits, so restart `start_sim.sh` before the next one.

The server hands tensors to the client through Unix sockets in Python's temp dir. Both containers
set `TEMP=/tmp/ipc` on a shared `ipc` volume, so the sockets are reachable across containers.

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
  [`docker/patches/task-repo-6e3a66c-import-fixes.patch`](docker/patches/task-repo-6e3a66c-import-fixes.patch) to the
  submodule's working tree to fix import errors, so `git status` there shows it modified.
- **`xhost +local:`** lets every local user's process open windows on your X display.
