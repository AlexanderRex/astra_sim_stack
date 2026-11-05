#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
full_base_image="astra/isaac-lab-base:6.0.1"
base_image="astra/isaac-lab-slim:6.0.1"
sim_image="astra/sim"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command is missing: $1" >&2
        exit 1
    fi
}

image_config() {
    docker image inspect --format \
        '{{json .Config.Env}} {{json .Config.Entrypoint}} {{json .Config.Cmd}} {{json .Config.Shell}} {{.Config.User}} {{.Config.WorkingDir}}' \
        "$1"
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

# The slim image is expensive and its full base is gone, so it is reused until what it was built from changes.
slim_inputs="$({
    cat "${repo_root}/docker/slim-base.Dockerfile" \
        "${repo_root}/third_party/ecm2027_sim_track/requirements/runtime-constraints.txt"
    git -C "${repo_root}/third_party/IsaacLab" rev-parse HEAD
} | sha256sum | cut -d' ' -f1)"

if [[ "$(docker image inspect --format '{{index .Config.Labels "astra.slim.inputs"}}' "${base_image}" 2>/dev/null)" == "${slim_inputs}" ]]; then
    echo "${base_image} matches its inputs; skipping base build."
else
    docker build \
        --file "${repo_root}/third_party/IsaacLab/docker/Dockerfile.base" \
        --build-arg ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim \
        --build-arg ISAACSIM_VERSION_ARG=6.0.1 \
        --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
        --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab \
        --build-arg DOCKER_USER_HOME_ARG=/root \
        --tag "${full_base_image}" \
        "${repo_root}/third_party/IsaacLab"
    docker build \
        --file "${repo_root}/docker/slim-base.Dockerfile" \
        --label "astra.slim.inputs=${slim_inputs}" \
        --tag "${base_image}" \
        "${repo_root}"
    if [[ "$(image_config "${full_base_image}")" != "$(image_config "${base_image}")" ]]; then
        docker rmi "${base_image}"
        echo "${base_image} config differs from ${full_base_image}; update docker/slim-base.Dockerfile." >&2
        exit 1
    fi
    # The slim image shares no layer with the full one, so keeping both doubles the disk use.
    docker rmi "${full_base_image}"
    echo "The build cache still holds the full image; 'docker builder prune --all' frees it."
fi

docker build \
    --file "${repo_root}/docker/sim.Dockerfile" \
    --tag "${sim_image}" \
    "${repo_root}"
