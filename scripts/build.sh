#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_image="astra/isaac-lab-base:6.0.1"
sim_image="astra/sim"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command is missing: $1" >&2
        exit 1
    fi
}

require_command docker
require_command git
require_command nvidia-smi
require_command nvidia-container-cli

if ! docker info >/dev/null 2>&1; then
    echo "Cannot reach the Docker daemon. Start Docker and check socket access." >&2
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver is unavailable: nvidia-smi failed." >&2
    exit 1
fi
if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
    echo "Docker does not expose the NVIDIA runtime. Configure NVIDIA Container Toolkit." >&2
    exit 1
fi
if ! docker buildx version >/dev/null 2>&1; then
    echo "Docker Buildx is required." >&2
    exit 1
fi
if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required to fetch the ECM scene and robot assets." >&2
    exit 1
fi

git -C "${repo_root}" submodule update --init
git -C "${repo_root}/third_party/ecm2027_sim_track" lfs pull

if docker image inspect "${base_image}" >/dev/null 2>&1; then
    echo "${base_image} already exists; skipping base build."
else
    docker build \
        --file "${repo_root}/third_party/IsaacLab/docker/Dockerfile.base" \
        --build-arg ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim \
        --build-arg ISAACSIM_VERSION_ARG=6.0.1 \
        --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
        --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab \
        --build-arg DOCKER_USER_HOME_ARG=/root \
        --tag "${base_image}" \
        "${repo_root}/third_party/IsaacLab"
fi

if docker image inspect "${sim_image}" >/dev/null 2>&1; then
    echo "${sim_image} already exists; skipping sim build."
else
    docker build \
        --file "${repo_root}/docker/sim.Dockerfile" \
        --tag "${sim_image}" \
        "${repo_root}"
fi
