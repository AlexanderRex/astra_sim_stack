#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
compose_file="$repo_root/docker/compose.yaml"

if ! command -v docker >/dev/null 2>&1; then
    echo "Required command is missing: docker" >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required (docker compose)." >&2
    exit 1
fi

export ASTRA_POLICY_UID="$(id -u)"
export ASTRA_POLICY_GID="$(id -g)"

exec docker compose --project-directory "$repo_root" -f "$compose_file" --profile policy run --build --rm policy
