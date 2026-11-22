#!/usr/bin/env bash
# Opt-in development rollout. Does not install dependencies or pull/build images.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_root="${repo_root}/third_party/task_repo"
patch_file="${repo_root}/docker/patches/task-repo-base-drive-observations.patch"

case "${1:-}" in
    -h|--help)
        echo "Usage: scripts/run_base_drive.sh [src/CONFIG.yml]"
        echo "Default: src/base_drive.yml (hold-start check; configure waypoints before driving)."
        echo "Requires existing images/assets; does not install dependencies or pull/build images."
        exit 0
        ;;
esac
(($# <= 1)) || { echo "Expected at most one policy config." >&2; exit 2; }
command -v docker >/dev/null || { echo "Docker is unavailable; no changes made." >&2; exit 1; }
# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"
for image in "${SIM_IMAGE}" "${POLICY_IMAGE}"; do
    docker image inspect "${image}" >/dev/null 2>&1 || {
        echo "Missing existing image ${image}. This launcher does not pull or build it." >&2
        exit 1
    }
done

if git -C "${task_root}" apply --reverse --check "${patch_file}" 2>/dev/null; then
    echo "Development telemetry patch is already present."
else
    git -C "${task_root}" apply --check "${patch_file}"
    git -C "${task_root}" apply "${patch_file}"
    echo "Applied opt-in development telemetry patch to the task submodule."
fi

export ASTRA_BASE_DRIVE_TELEMETRY=1
exec "${repo_root}/scripts/run_headless.sh" "${1:-${repo_root}/src/base_drive.yml}"
