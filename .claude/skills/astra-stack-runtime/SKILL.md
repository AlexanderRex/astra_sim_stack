---
name: astra-stack-runtime
description: Bring up or diagnose the Astra Isaac Sim Docker stack, including GPU, submodules, IPC, startup stalls, and headless evaluation.
license: Apache-2.0
---

# Astra runtime

Adapted from NVIDIA Isaac Sim installation guidance by `ropardeshi` and headless/troubleshooting guidance by Renato Gasoto, under Apache-2.0. See `LICENSE.skills.md` at the repository root. These are project-specific adaptations, not unmodified NVIDIA skills.

## Establish the actual runtime

Read `docker/versions.env`, `.gitmodules`, `docker/docker-compose.yml`, the relevant launcher, and `git submodule status`. The inspected stack pins Isaac Sim 6.0.1 and Isaac Lab commit `28a37cecdd433c22d9eabd6a5954add9f13a8951`; verify the checkout before applying advice from newer upstream docs.

The supported development path is Linux x86_64 with Docker, Compose v2, Buildx, NVIDIA Container Toolkit, Git LFS, and an RTX GPU. A Windows IDE or WSL path does not establish that this runtime is installed. Identify which Linux environment owns Docker and the GPU. Check `nvidia-smi`, Docker GPU access, disk capacity, and materialized LFS assets. Inspect current compatibility requirements rather than treating a driver's reported CUDA version as an installed toolkit.

## Use the existing launchers

- Build through `scripts/build_sim.sh` and `scripts/build_policy.sh` when needed. First builds are large; do not rebuild for ordinary bind-mounted source edits.
- Run a local connection check with `scripts/run_headless.sh src/smoke.yml`. The smoke policy sends zero actions and returns failure; a connected rollout is transport evidence, not task success or a safe hold controller.
- The server listens at `127.0.0.1:50000`, creates the scene after connection, serves one run, then exits. Restart it for each evaluation.
- Both containers use host networking, UID 1000, and shared `/tmp/ipc` with `TEMP=/tmp/ipc` for Torch tensor sockets. Preserve these relationships when diagnosing transport errors.
- Headless means no GUI; RGB policies still need camera rendering. Do not apply renderer-disabled physics-only recipes to visual-policy evaluation.

## Isolate a failure

Distinguish image/build errors, server readiness, client connection, Kit startup, asset loading, first frame, stepping, and teardown. Gather the failing command, image identity, submodule SHAs, container logs, resource usage, and elapsed stage timings.

First shader compilation can take 5–15 minutes. A `carb.tasking is likely stuck` line alone does not establish deadlock. Compare log progress and CPU/GPU activity, retain useful caches, and use an external process deadline for genuinely blocked native calls. Python signal handlers may also be delayed by a native call; do not assume a Python thread or signal handler can always interrupt it.

Check mount permissions, shared IPC paths, stale servers, LFS pointer files, missing local assets, and VRAM before changing physics. The task submodule is intentionally patched by `build_sim.sh`; compare the working diff with `docker/patches/task-repo-6e3a66c-import-fixes.patch` before treating it as corruption.

Deliver the smallest justified fix and the exact successful or failing reproduction. Preserve unrelated containers, caches, submodule edits, and host configuration.

## Sources

Inspected NVIDIA skill snapshot: `7c206f75bdadd9e05fc457f19863ca4c3f0cb693`. Use installed task/runtime metadata to resolve compatibility with the active checkout.

- [Installation](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-sim-installation/SKILL.md)
- [Headless usage](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-sim-headless-deployment/SKILL.md)
- [Troubleshooting](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-sim-troubleshooting/SKILL.md)

Standalone source-build paths, remote Python execution, and renderer-disabled examples are not prerequisites for Astra's Docker workflows. No upstream helper scripts are bundled.
