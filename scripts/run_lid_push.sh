#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
config="$(realpath "${1:-src/stage1_lid_push.yml}")"
[[ "$config" == "$root/src/"* && -f "$config" ]] || { echo "Expected a YAML under src/" >&2; exit 2; }
if [[ -n "$(docker ps -q -f label=com.docker.compose.project=astra -f label=com.docker.compose.service=sim)" ]]; then
    echo "Astra sim is already running." >&2
    exit 1
fi
base_patch="$root/docker/patches/task-repo-base-drive-observations.patch"
lid_patch="$root/docker/patches/task-repo-lid-push-observations.patch"
# The lid patch extends the observation class already modified by the base patch.
if ! git -C third_party/task_repo apply --reverse --check "$lid_patch" 2>/dev/null; then
    if ! git -C third_party/task_repo apply --reverse --check "$base_patch" 2>/dev/null; then
        git -C third_party/task_repo apply "$base_patch"
    fi
    git -C third_party/task_repo apply "$lid_patch"
fi
if [[ "${ASTRA_LID_POSTURE:-0}" == 1 || "${ASTRA_LID_POSTURE:-0}" == replay ]]; then
    posture_patch="$root/docker/patches/task-repo-lid-posture-action.patch"
    if ! git -C third_party/task_repo apply --reverse --check "$posture_patch" 2>/dev/null; then
        git -C third_party/task_repo apply "$posture_patch"
    fi
fi
export ASTRA_LID_RESULTS
ASTRA_LID_RESULTS="$(mktemp -d "$root/lid-push-results.XXXXXX")"
chmod 0775 "$ASTRA_LID_RESULTS"
compose=(docker compose --env-file docker/versions.env -f docker/docker-compose.yml -f docker/lid-push.compose.yml)
# Bash calls cleanup through the EXIT trap below.
# shellcheck disable=SC2329
cleanup() {
    "${compose[@]}" logs --no-color sim > "$ASTRA_LID_RESULTS/sim.log" 2>&1 || true
    "${compose[@]}" rm --stop --force sim >/dev/null 2>&1 || true
    echo "Results: $ASTRA_LID_RESULTS"
}
trap cleanup EXIT
"${compose[@]}" up -d sim
for ((i=0; i<120; i++)); do
    if "${compose[@]}" logs --no-color sim 2>&1 | grep -F 'Waiting for connection' >/dev/null; then
        "${compose[@]}" run --rm policy "/workspace/src/${config#"$root/src/"}" "${@:2}"
        exit
    fi
    sleep 1
done
echo "Server did not become ready" >&2
exit 1
