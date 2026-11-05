FROM astra/isaac-lab-base:6.0.1 AS trimmed

USER root

# isaaclab.sh pins torch 2.10 on CUDA 12; the project pins 2.11 on CUDA 13 (its install_env.sh).
# Both CUDA sets write the same files, so remove all their wheels before installing the project's.
# Isaac Sim's own apps require omni.isaac.ml_archive: keep the extension, drop its second torch and CUDA.
RUN --mount=type=bind,source=third_party/ecm2027_sim_track/requirements/runtime-constraints.txt,target=/tmp/constraints.txt \
    set -o pipefail \
    && ./isaaclab.sh -p -m pip freeze \
        | grep -iE '^(nvidia[-_](cu|nccl|nvjitlink|nvshmem|nvtx)|cuda[-_]|torch(vision|audio)?==)' \
        | xargs ./isaaclab.sh -p -m pip uninstall -y \
    && ./isaaclab.sh -p -m pip install --no-cache-dir -c /tmp/constraints.txt \
        --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio \
    && grep -v '^isaacsim==' /tmp/constraints.txt \
        | xargs ./isaaclab.sh -p -m pip install --no-cache-dir \
    && rm -rf /isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/{torch,nvidia,cuda}* \
    && ./isaaclab.sh -p -c "import torch, torchvision, torchaudio"

# Kit dedups identical files across extensions with symlinks, and pip run through python.sh deletes
# those files as its own. RTX loads libnvrtc-builtins this way, so put the file in place of each broken
# link. Restoring it at the link's target would leave part of a package that shadows the real one.
RUN --mount=type=bind,from=nvcr.io/nvidia/isaac-sim:6.0.1,target=/original \
    find /isaac-sim -xtype l | while IFS= read -r link; do \
        cp -aL --remove-destination "/original$link" "$link" || exit 1; \
    done \
    && ! find /isaac-sim -xtype l | grep .

# A later layer only hides deleted files; copying into an empty image really drops them.
FROM scratch
COPY --from=trimmed / /

# scratch inherits no config. scripts/build.sh fails if this differs from the full base image.
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    MIN_DRIVER_VERSION=570.169 \
    HUB__CACHE__PATH=/var/cache/hub \
    HUB__ARGS__DETECT_ONLY=true \
    OMNICLIENT_HUB_EXE=/usr/local/bin/hub \
    ISAACSIM_VERSION="" \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab \
    DOCKER_USER_HOME=/root \
    HOME=/root \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/isaac-sim/runheadless.sh"]
USER isaaclab
WORKDIR /workspace/isaaclab
