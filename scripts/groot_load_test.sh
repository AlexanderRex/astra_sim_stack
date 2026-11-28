#!/usr/bin/env bash
# Load a GR00T checkpoint offline in Isaac-GR00T's own image and run it through the Astra runtime.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"
# shellcheck source=scripts/gr00t_common.sh
source "${repo_root}/scripts/gr00t_common.sh"

usage() {
    cat <<'USAGE'
Usage: scripts/groot_load_test.sh [CHECKPOINT_DIR]

Loads CHECKPOINT_DIR (default: src/astra/checkpoints/checkpoint-30000) with networking off and
runs scripts/groot_load_test.py: weights, a prediction through the Astra pipeline, output ranges,
latency and GPU memory. Prints LOAD TEST PASSED or LOAD TEST FAILED and exits 0 or 1.

Uses Isaac-GR00T at ISAAC_GR00T_COMMIT (docker/versions.env), checked out in
third_party/isaac_gr00t/; builds GR00T_IMAGE:<commit> from it the first time (about 26 GB).

Needs Cosmos-Reason2-2B at COSMOS_REVISION in a Hugging Face hub cache, default
src/astra/checkpoints/hf_home/hub (override with HF_CACHE). If it is missing, the script prints
the download command (needs a Hugging Face login with access to the gated model).
USAGE
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac
(($# <= 1)) || { usage >&2; exit 2; }

checkpoint="$(realpath "${1:-${repo_root}/src/astra/checkpoints/checkpoint-30000}")"
gr00t_prepare "${checkpoint}"

exec docker run --rm --network none "${gr00t_run_args[@]}" \
    -v "${repo_root}:/astra:ro" \
    "${gr00t_image}" python /astra/scripts/groot_load_test.py /ckpt
