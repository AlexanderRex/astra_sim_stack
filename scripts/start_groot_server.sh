#!/usr/bin/env bash
# Serve a GR00T checkpoint to the policy container (runtime.yaml groot.server) from Isaac-GR00T's own image.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"
# shellcheck source=scripts/gr00t_common.sh
source "${repo_root}/scripts/gr00t_common.sh"

port="${GROOT_PORT:-5555}"
container=astra-groot-server

usage() {
    cat <<USAGE
Usage: scripts/start_groot_server.sh [CHECKPOINT_DIR]

Starts Isaac-GR00T's inference server (gr00t/eval/run_gr00t_server.py) on 127.0.0.1:${port}
(GROOT_PORT) for CHECKPOINT_DIR (default: src/astra/checkpoints/checkpoint-30000), in the
foreground; Ctrl-C stops it. The policy reaches it with runtime.yaml groot.server: 127.0.0.1:${port}.

Isaac-GR00T cannot share the policy image's Python environment, so the model runs here, in
GR00T_IMAGE at ISAAC_GR00T_COMMIT (docker/versions.env), with the Hugging Face cache read-only and
HF_HUB_OFFLINE=1: it loads nothing from the network. The policy container uses host networking,
so the server listens on the host's loopback only.
USAGE
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac
(($# <= 1)) || { usage >&2; exit 2; }

checkpoint="$(realpath "${1:-${repo_root}/src/astra/checkpoints/checkpoint-30000}")"
gr00t_prepare "${checkpoint}"

if docker ps --quiet --filter "name=^${container}$" | grep -q .; then
    echo "${container} is already running; stop it with: docker stop ${container}" >&2
    exit 1
fi

exec docker run --rm --name "${container}" --network host "${gr00t_run_args[@]}" \
    "${gr00t_image}" python -u /workspace/gr00t/eval/run_gr00t_server.py \
    --model-path /ckpt --embodiment-tag new_embodiment --device cuda:0 \
    --host 127.0.0.1 --port "${port}"
