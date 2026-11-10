#!/usr/bin/env bash
# Build astra/sim: Isaac Lab base -> slim base -> sim. Safe to rerun; finished stages are reused.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_root="${repo_root}/third_party/task_repo"
isaaclab_root="${repo_root}/third_party/IsaacLab"
task_fixes="${repo_root}/docker/patches/task-repo-6e3a66c-import-fixes.patch"

# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"

fail() {
    echo "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command is missing: $1"
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

docker info >/dev/null 2>&1 || fail "Cannot reach the Docker daemon. Start Docker and check socket access."
nvidia-smi >/dev/null 2>&1 || fail "NVIDIA driver is unavailable: nvidia-smi failed."
docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"' \
    || fail "Docker does not expose the NVIDIA runtime. Install and configure NVIDIA Container Toolkit."
docker buildx version >/dev/null 2>&1 || fail "Docker Buildx is required."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required (docker compose)."
git lfs version >/dev/null 2>&1 || fail "Git LFS is required to fetch the task repo's scene and robot assets."

git -C "${repo_root}" submodule update --init
git -C "${task_root}" lfs pull

# The task repo has import errors (see the patch). The image runs this tree mounted from the
# host, so the fixes go into the working tree, not into the image. Rerunning finds them applied.
if git -C "${task_root}" apply --reverse --check "${task_fixes}" 2>/dev/null; then
    echo "Task repo fixes already applied."
elif git -C "${task_root}" apply --check "${task_fixes}" 2>/dev/null; then
    git -C "${task_root}" apply "${task_fixes}"
    echo "Applied task repo fixes to ${task_root}."
else
    fail "Cannot apply ${task_fixes}: third_party/task_repo has conflicting changes."
fi

# The slim image is expensive and its full base is removed, so it is reused until what it was built from changes.
# Only the versions.env values it uses, so adding an unrelated image name does not rebuild it.
slim_inputs="$({
    cat "${repo_root}/docker/isaac-lab-slim.Dockerfile" \
        "${task_root}/requirements/runtime-constraints.txt"
    echo "${ISAACSIM_VERSION} ${ISAACLAB_BASE_IMAGE}"
    git -C "${isaaclab_root}" rev-parse HEAD
} | sha256sum | cut -d' ' -f1)"

if [[ "$(docker image inspect --format '{{index .Config.Labels "astra.slim.inputs"}}' "${ISAACLAB_SLIM_IMAGE}" 2>/dev/null)" == "${slim_inputs}" ]]; then
    echo "${ISAACLAB_SLIM_IMAGE} matches its inputs; skipping base build."
else
    # Isaac Lab's own Dockerfile, unchanged; only the arguments are ours.
    docker build \
        --file "${isaaclab_root}/docker/Dockerfile.base" \
        --build-arg ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim \
        --build-arg ISAACSIM_VERSION_ARG="${ISAACSIM_VERSION}" \
        --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
        --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab \
        --build-arg DOCKER_USER_HOME_ARG=/root \
        --tag "${ISAACLAB_BASE_IMAGE}" \
        "${isaaclab_root}"
    docker build \
        --file "${repo_root}/docker/isaac-lab-slim.Dockerfile" \
        --build-arg ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
        --build-arg ISAACLAB_BASE_IMAGE="${ISAACLAB_BASE_IMAGE}" \
        --label "astra.slim.inputs=${slim_inputs}" \
        --tag "${ISAACLAB_SLIM_IMAGE}" \
        "${repo_root}"
    if [[ "$(image_config "${ISAACLAB_BASE_IMAGE}")" != "$(image_config "${ISAACLAB_SLIM_IMAGE}")" ]]; then
        docker rmi "${ISAACLAB_SLIM_IMAGE}"
        fail "${ISAACLAB_SLIM_IMAGE} config differs from ${ISAACLAB_BASE_IMAGE}; update docker/isaac-lab-slim.Dockerfile."
    fi
    # The slim image shares no layer with the full one, so keeping both doubles the disk use.
    docker rmi "${ISAACLAB_BASE_IMAGE}"
    echo "The build cache still holds the full image; 'docker builder prune --all' frees it."
fi

# Cheap when nothing changed: every step comes from the build cache.
docker build \
    --file "${repo_root}/docker/sim.Dockerfile" \
    --build-arg ISAACLAB_SLIM_IMAGE="${ISAACLAB_SLIM_IMAGE}" \
    --tag "${SIM_IMAGE}" \
    "${repo_root}"
