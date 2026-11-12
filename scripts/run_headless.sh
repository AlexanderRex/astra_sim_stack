#!/usr/bin/env bash
# One headless run: start the env server in the background, run the policy client, stop the server.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose=(docker compose --env-file "${repo_root}/docker/versions.env" -f "${repo_root}/docker/docker-compose.yml")
# Seconds to wait for the env server to listen. Kit starts later, when the policy connects.
ready_timeout=120

usage() {
    cat <<'USAGE'
Usage: scripts/run_headless.sh [CONFIG]

Starts the env server without the Kit window, runs the policy client against it
(scripts/start_policy.sh CONFIG; default src/smoke.yml), then stops the server.
The policy's output stays in this terminal; the server's is shown if the run fails
and can be followed from another terminal with:
  docker logs -f astra-sim-1
Ctrl+C stops both containers.
USAGE
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac
(($# <= 1)) || { usage >&2; exit 2; }

# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"
for image in "${SIM_IMAGE}" "${POLICY_IMAGE}"; do
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        echo "${image} is not built. Run scripts/build_sim.sh and scripts/build_policy.sh first." >&2
        exit 1
    fi
done

server_id() {
    docker ps --quiet \
        --filter label=com.docker.compose.project=astra \
        --filter label=com.docker.compose.service=sim \
        --filter label=com.docker.compose.oneoff=False
}

# 'up' would recreate a server started with the window, ending that run.
if [[ -n "$(server_id)" ]]; then
    echo "An env server is already running. Stop it first: docker stop $(server_id)" >&2
    exit 1
fi

stop_server() {
    status=$?
    trap - EXIT
    if ((status != 0)); then
        echo "--- env server log (last 40 lines) ---" >&2
        "${compose[@]}" logs --no-log-prefix --tail 40 sim >&2 || true
    fi
    "${compose[@]}" rm --stop --force sim >/dev/null 2>&1 || true
    exit "${status}"
}
trap stop_server EXIT

SIM_VIZ=none "${compose[@]}" up --detach sim

echo "Waiting for the env server..."
deadline=$((SECONDS + ready_timeout))
until "${compose[@]}" logs --no-log-prefix sim 2>&1 | grep -q "Waiting for connection"; do
    if [[ -z "$(server_id)" ]]; then
        echo "The env server exited before it was ready." >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "The env server is not ready after ${ready_timeout} s." >&2
        exit 1
    fi
    sleep 1
done

"${repo_root}/scripts/start_policy.sh" "$@"
