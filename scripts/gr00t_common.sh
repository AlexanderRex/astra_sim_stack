# Shared setup for the scripts that run Isaac-GR00T in its own image (groot_load_test.sh,
# start_groot_server.sh). Source it; it defines gr00t_prepare and the gr00t_run_args array.
# Needs repo_root set and docker/versions.env sourced.

gr00t_src="${repo_root}/third_party/isaac_gr00t"
gr00t_image="${GR00T_IMAGE}:${ISAAC_GR00T_COMMIT:0:7}"
hf_cache="$(realpath -m "${HF_CACHE:-${repo_root}/src/astra/checkpoints/hf_home/hub}")"

# gr00t_prepare CHECKPOINT_DIR: check the checkpoint and the Cosmos cache, check out Isaac-GR00T at
# ISAAC_GR00T_COMMIT, build the image if it is missing, and fill gr00t_run_args.
gr00t_prepare() {
    local checkpoint="$1"
    [[ -f "${checkpoint}/model.safetensors.index.json" ]] || {
        echo "Not a GR00T checkpoint (no model.safetensors.index.json): ${checkpoint}" >&2
        return 2
    }

    # Cosmos-Reason2-2B: GR00T loads it by name at COSMOS_REVISION. Offline loading resolves "main"
    # through refs/main, which a download by commit hash does not write, so add it here.
    local cosmos="${hf_cache}/models--nvidia--Cosmos-Reason2-2B"
    if [[ ! -f "${cosmos}/snapshots/${COSMOS_REVISION}/model.safetensors" ]]; then
        cat >&2 <<EOF
Cosmos-Reason2-2B ${COSMOS_REVISION} is not in ${hf_cache}. Download it (Hugging Face login with
access to nvidia/Cosmos-Reason2-2B needed; HF_HUB_CACHE, not HF_HOME, so the login is kept):

  HF_HUB_CACHE="${hf_cache}" hf download nvidia/Cosmos-Reason2-2B \\
      --revision ${COSMOS_REVISION} --exclude "*.png"
EOF
        return 1
    fi
    if [[ "$(cat "${cosmos}/refs/main" 2>/dev/null)" != "${COSMOS_REVISION}" ]]; then
        mkdir -p "${cosmos}/refs"
        printf %s "${COSMOS_REVISION}" > "${cosmos}/refs/main"
        echo "Pointed ${cosmos}/refs/main at ${COSMOS_REVISION}"
    fi

    # Isaac-GR00T checkout at the pinned commit (code only; demo data and submodules are not needed).
    _gr00t_checkout() {
        GIT_LFS_SKIP_SMUDGE=1 git -C "${gr00t_src}" -c advice.detachedHead=false checkout --quiet "${ISAAC_GR00T_COMMIT}"
    }
    if [[ ! -d "${gr00t_src}/.git" ]]; then
        git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "${gr00t_src}"
        _gr00t_checkout  # always: HEAD may already name the commit while the tree is still empty
    elif [[ "$(git -C "${gr00t_src}" rev-parse HEAD)" != "${ISAAC_GR00T_COMMIT}" ]]; then
        git -C "${gr00t_src}" fetch --quiet origin "${ISAAC_GR00T_COMMIT}"
        _gr00t_checkout
    fi
    if [[ -n "$(git -C "${gr00t_src}" status --porcelain --untracked-files=no)" ]]; then
        echo "${gr00t_src} has local changes; GR00T must run the pinned commit unchanged." >&2
        return 1
    fi

    if ! docker image inspect "${gr00t_image}" >/dev/null 2>&1; then
        echo "Building ${gr00t_image} from Isaac-GR00T ${ISAAC_GR00T_COMMIT:0:7} (first time only, long)..."
        DOCKER_BUILDKIT=1 docker build --network host -f "${gr00t_src}/docker/Dockerfile" -t "${gr00t_image}" "${gr00t_src}"
    fi

    # GROOT_PATCH_MISTRAL: transformers 4.57.3 asks the Hub whether a tokenizer is Mistral even when
    # offline and fails; this switches on Isaac-GR00T's own fix. NO_ALBUMENTATIONS_UPDATE: no version
    # check over the network at import.
    gr00t_run_args=(
        --gpus all --ipc=host
        --user "$(id -u):$(id -g)" -e HOME=/tmp
        -e HF_HUB_CACHE=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
        -e GROOT_PATCH_MISTRAL=1 -e NO_ALBUMENTATIONS_UPDATE=1 -e PYTHONPATH=/workspace
        -v "${gr00t_src}:/workspace:ro"
        -v "${hf_cache}:/hf:ro"
        -v "${checkpoint}:/ckpt:ro"
    )
}
