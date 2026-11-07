#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
    cat <<'EOF'
Usage: scripts/launch.sh [--build] [-- COMMAND [ARG...]]

Start an interactive shell in the Astra Isaac Sim container, or run a command
inside it. The image is built automatically when it is missing.

Examples:
  scripts/launch.sh
  scripts/launch.sh --build
  scripts/launch.sh -- bash scripts/run_replay_eval.sh /path/to/recording.hdf5 action
EOF
}

force_build=false
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == "--build" ]]; then
    force_build=true
    shift
fi
if [[ "${1:-}" == "--" ]]; then
    shift
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Required command is missing: docker" >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required (docker compose)." >&2
    exit 1
fi

if [[ "$force_build" == true ]] || ! docker image inspect astra/sim >/dev/null 2>&1; then
    "$repo_root/scripts/build.sh"
fi

if (($#)); then
    exec docker compose run --rm --remove-orphans sim "$@"
else
    exec docker compose run --rm --remove-orphans sim
fi
