#!/usr/bin/env bash
# Build astra/policy: Python, CUDA torch and the task repo's dependencies, no Isaac.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"

fail() {
    echo "$*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || fail "Required command is missing: docker"
command -v git >/dev/null 2>&1 || fail "Required command is missing: git"
docker info >/dev/null 2>&1 || fail "Cannot reach the Docker daemon. Start Docker and check socket access."
docker buildx version >/dev/null 2>&1 || fail "Docker Buildx is required."

# The image reads the task repo's dependency pins; its assets are not needed, so no LFS pull.
git -C "${repo_root}" submodule update --init third_party/task_repo

docker build \
    --file "${repo_root}/docker/policy.Dockerfile" \
    --tag "${POLICY_IMAGE}" \
    "${repo_root}"
