# check=skip=InvalidDefaultArgInFrom
# Versions come from docker/versions.env through scripts/build_sim.sh.
ARG ISAACLAB_SLIM_IMAGE
FROM ${ISAACLAB_SLIM_IMAGE}

USER root
# /bin/sh is dash; fail when any part of a pipe fails.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# docker-compose.yml mounts the same tree over /workspace/task, so code edits need no rebuild.
# This copy only registers the editable install; scripts/build_sim.sh has already applied our fixes to it.
COPY --chown=isaaclab:isaaclab third_party/task_repo/ /workspace/task/
ENV PYTHONPATH=/workspace/task:/workspace/task/vendor/isaaclab_arena
# The project's scripts run $PYTHON_BIN, and the image has no python on PATH.
ENV PYTHON_BIN=/workspace/isaaclab/_isaac_sim/python.sh

# Align Isaac Lab's dependency metadata with Isaac Sim 6.0.1 (upstream's patch).
# pip can break Kit's links (see isaac-lab-slim.Dockerfile), so fail if it did.
# Kit logs to kit/logs, and the base image hands over only its Isaac-Sim subfolder.
# Isaac Sim's image gives the Hub asset cache to its own isaac-sim user (uid 1234), so Hub cannot start.
RUN git -C /workspace/isaaclab apply /workspace/task/patches/isaaclab-6.1.11-isaacsim-6.0.1.patch \
    && /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir --no-deps -e /workspace/isaaclab/source/isaaclab \
    && /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
        -c /workspace/task/requirements/runtime-constraints.txt -e /workspace/task \
    && ! find /isaac-sim -xtype l | grep . \
    && printf '#!/bin/sh\nexec %s "$@"\n' "$PYTHON_BIN" > /usr/local/bin/python \
    && chmod 755 /usr/local/bin/python \
    && chown -R isaaclab:isaaclab /isaac-sim/kit/logs "$HUB__CACHE__PATH"

USER isaaclab
WORKDIR /workspace/task
# The base starts Isaac Sim's own headless app; this image starts the task repo's env server.
ENTRYPOINT []
CMD ["bash", "scripts/run_eval_env.sh"]
