#!/usr/bin/env bash
# Load a GR00T checkpoint offline in Isaac-GR00T's own image and run it through the Astra runtime.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=docker/versions.env
source "${repo_root}/docker/versions.env"

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
hf_cache="$(realpath -m "${HF_CACHE:-${repo_root}/src/astra/checkpoints/hf_home/hub}")"
gr00t_src="${repo_root}/third_party/isaac_gr00t"
image="${GR00T_IMAGE}:${ISAAC_GR00T_COMMIT:0:7}"

[[ -f "${checkpoint}/model.safetensors.index.json" ]] || {
    echo "Not a GR00T checkpoint (no model.safetensors.index.json): ${checkpoint}" >&2
    exit 2
}

# Cosmos-Reason2-2B: GR00T loads it by name at COSMOS_REVISION. Offline loading resolves "main"
# through refs/main, which a download by commit hash does not write, so add it here.
cosmos="${hf_cache}/models--nvidia--Cosmos-Reason2-2B"
if [[ ! -f "${cosmos}/snapshots/${COSMOS_REVISION}/model.safetensors" ]]; then
    cat >&2 <<EOF
Cosmos-Reason2-2B ${COSMOS_REVISION} is not in ${hf_cache}. Download it (Hugging Face login with
access to nvidia/Cosmos-Reason2-2B needed; HF_HUB_CACHE, not HF_HOME, so the login is kept):

  HF_HUB_CACHE="${hf_cache}" hf download nvidia/Cosmos-Reason2-2B \\
      --revision ${COSMOS_REVISION} --exclude "*.png"
EOF
    exit 1
fi
if [[ "$(cat "${cosmos}/refs/main" 2>/dev/null)" != "${COSMOS_REVISION}" ]]; then
    mkdir -p "${cosmos}/refs"
    printf %s "${COSMOS_REVISION}" > "${cosmos}/refs/main"
    echo "Pointed ${cosmos}/refs/main at ${COSMOS_REVISION}"
fi

# Isaac-GR00T checkout at the pinned commit (code only; demo data and submodules are not needed).
checkout_pinned() {
    GIT_LFS_SKIP_SMUDGE=1 git -C "${gr00t_src}" -c advice.detachedHead=false checkout --quiet "${ISAAC_GR00T_COMMIT}"
}
if [[ ! -d "${gr00t_src}/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "${gr00t_src}"
    checkout_pinned  # always: HEAD may already name the commit while the tree is still empty
elif [[ "$(git -C "${gr00t_src}" rev-parse HEAD)" != "${ISAAC_GR00T_COMMIT}" ]]; then
    git -C "${gr00t_src}" fetch --quiet origin "${ISAAC_GR00T_COMMIT}"
    checkout_pinned
fi
if [[ -n "$(git -C "${gr00t_src}" status --porcelain --untracked-files=no)" ]]; then
    echo "${gr00t_src} has local changes; the test must run the pinned commit unchanged." >&2
    exit 1
fi

if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Building ${image} from Isaac-GR00T ${ISAAC_GR00T_COMMIT:0:7} (first time only, long)..."
    DOCKER_BUILDKIT=1 docker build --network host -f "${gr00t_src}/docker/Dockerfile" -t "${image}" "${gr00t_src}"
fi

# GROOT_PATCH_MISTRAL: transformers 4.57.3 asks the Hub whether a tokenizer is Mistral even when
# offline and fails; this switches on Isaac-GR00T's own fix. NO_ALBUMENTATIONS_UPDATE: no version
# check over the network at import.
exec docker run --rm --gpus all --network none --ipc=host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e HF_HUB_CACHE=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e GROOT_PATCH_MISTRAL=1 -e NO_ALBUMENTATIONS_UPDATE=1 -e PYTHONPATH=/workspace \
    -v "${gr00t_src}:/workspace:ro" \
    -v "${repo_root}:/astra:ro" \
    -v "${hf_cache}:/hf:ro" \
    -v "${checkpoint}:/ckpt:ro" \
    "${image}" python /astra/scripts/groot_load_test.py /ckpt
