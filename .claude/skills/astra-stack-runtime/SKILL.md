---
name: astra-stack-runtime
description: Run or diagnose Astra Docker services, GPU access, the gRPC evaluation service, and headless startup.
license: Apache-2.0
---

# Astra runtime

- Start with `docker/versions.env`, `docker/docker-compose.yml`, the relevant script in `scripts/`, and `git submodule status`; use their pinned versions.
- Run on Linux with Docker/Compose, Buildx, NVIDIA Container Toolkit, Git LFS, and an RTX GPU. Check `nvidia-smi`, Docker GPU access, available disk, and materialized LFS assets before diagnosing application code.
- Build with `scripts/build_sim.sh` and `scripts/build_policy.sh`. Source under `src/` and the task repo is bind-mounted; ordinary code edits do not require rebuilding images.
- Use `scripts/run_headless.sh src/smoke.yml` to check connectivity. The zero-action smoke policy returns failure by design; it does not test task completion or provide a hold controller.
- For faster headless runs, set `SIM_CPUSET` (Docker cpuset syntax, e.g. `0-3`) before `scripts/start_sim.sh` or `scripts/run_headless.sh`; `scripts/bench_sim.sh` compares CPU sets. See README, "CPU pinning".
- The evaluation service creates the scene at start, then listens for a gRPC policy client at `127.0.0.1:50000` and keeps running until stopped. It needs `ffmpeg` to record episodes; without it the service exits right after scene setup with no error shown. Its env config is `docker/eval_server.json`.
- Both services use host networking and UID 1000. `TEMP=/tmp/ipc` on a shared volume matters only for the older ipc protocol (`--remote_protocol=ipc`).
- To stop the service with its episode score written, send SIGINT to the Python child of the container's main process (`scripts/run_headless.sh` does this); `docker stop` kills it without a score.
- Headless disables the window; RGB observations still require rendering and an RTX GPU.
- First shader compilation can take 5-15 minutes. A `carb.tasking is likely stuck` line alone does not establish a deadlock; inspect log progress and CPU/GPU activity before deleting caches or rebuilding.
- `build_sim.sh` intentionally patches the task repo. Compare its diff with `docker/patches/task-repo-17159e5-import-fixes.patch` before discarding submodule changes.

Adapted under Apache-2.0 from NVIDIA Isaac Sim installation guidance by `ropardeshi` and headless/troubleshooting guidance by Renato Gasoto. See `LICENSE` at the repository root. Upstream references: [installation](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-sim-installation/SKILL.md), [headless](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-sim-headless-deployment/SKILL.md), [troubleshooting](https://github.com/isaac-sim/IsaacSim/blob/7c206f75bdadd9e05fc457f19863ca4c3f0cb693/skills/isaac-sim-troubleshooting/SKILL.md).
