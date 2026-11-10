#!/usr/bin/env bash
# Start the env server in the sim container, or open a shell in it with --shell.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose=(docker compose --env-file "${repo_root}/docker/versions.env" -f "${repo_root}/docker/docker-compose.yml")

usage() {
    cat <<'EOF'
Usage: scripts/start_sim.sh [--headless | --shell]

Without arguments, starts the task repo's env server (scripts/run_eval_env.sh),
which listens on 127.0.0.1:50000. Kit starts, with its window, when a policy
connects; each server serves one policy run, then exits.

  --headless  start the server without the Kit window (faster; for evaluation runs)
  --shell     open a bash shell instead: in the running server's container if
              there is one, otherwise in a new one
EOF
}

mode=window
case "${1:-}" in
    "") ;;
    --headless) mode=headless ;;
    --shell) mode=shell ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"
if ! docker image inspect "${SIM_IMAGE}" >/dev/null 2>&1; then
    echo "${SIM_IMAGE} is not built. Run scripts/build_sim.sh first." >&2
    exit 1
fi

if [[ "${mode}" == shell ]]; then
    # Only the server started by 'up', not shells from 'run' (oneoff=True).
    server="$(docker ps --quiet \
        --filter label=com.docker.compose.project=astra \
        --filter label=com.docker.compose.service=sim \
        --filter label=com.docker.compose.oneoff=False)"
    if [[ -n "${server}" ]]; then
        exec docker exec -it "${server}" bash
    fi
    exec "${compose[@]}" run --rm sim bash
fi

if [[ "${mode}" == headless ]]; then
    export SIM_VIZ=none
else
    # The container's user (uid 1000) must reach the host X server.
    if command -v xhost >/dev/null 2>&1; then
        xhost +local: >/dev/null || echo "xhost failed; the Kit window may not open." >&2
    else
        echo "xhost not found (package x11-xserver-utils); the Kit window may not open." >&2
    fi
fi
exec "${compose[@]}" up sim
