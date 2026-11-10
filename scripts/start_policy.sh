#!/usr/bin/env bash
# Run the policy client against the env server started by scripts/start_sim.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose=(docker compose --env-file "${repo_root}/docker/versions.env" -f "${repo_root}/docker/docker-compose.yml")

usage() {
    cat <<'USAGE'
Usage: scripts/start_policy.sh [CONFIG]

Runs the task repo's policy evaluation client in the policy container.
CONFIG is a policy YAML under src/ (default: src/smoke.yml, a zero-action check).
Start the env server first with scripts/start_sim.sh; each server serves one run.
Results are written to third_party/task_repo/eval_result/.
USAGE
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac
(($# <= 1)) || { usage >&2; exit 2; }

config="$(realpath "${1:-${repo_root}/src/smoke.yml}")"
if [[ "${config}" != "${repo_root}/src/"* || ! -f "${config}" ]]; then
    echo "Policy config must be a file under src/: ${config}" >&2
    exit 2
fi

# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"
if ! docker image inspect "${POLICY_IMAGE}" >/dev/null 2>&1; then
    echo "${POLICY_IMAGE} is not built. Run scripts/build_policy.sh first." >&2
    exit 1
fi
server="$(docker ps --quiet \
    --filter label=com.docker.compose.project=astra \
    --filter label=com.docker.compose.service=sim \
    --filter label=com.docker.compose.oneoff=False)"
if [[ -z "${server}" ]]; then
    echo "The env server is not running. Start it with scripts/start_sim.sh." >&2
    exit 1
fi

exec "${compose[@]}" run --rm policy "/workspace/src/${config#"${repo_root}/src/"}"
