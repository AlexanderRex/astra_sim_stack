FROM astra/isaac-lab-slim:6.0.1

USER root

COPY --chown=isaaclab:isaaclab third_party/ecm2027_sim_track/ /workspace/ecm/
COPY --chown=isaaclab:isaaclab docker/patches/ /workspace/ecm/patches/
ENV PYTHONPATH=/workspace/ecm:/workspace/ecm/vendor/isaaclab_arena
# The project's scripts run $PYTHON_BIN, and the image has no python on PATH.
ENV PYTHON_BIN=/workspace/isaaclab/_isaac_sim/python.sh

# Align Isaac Lab's dependency metadata with Isaac Sim 6.0.1. pip can break Kit's links (see slim-base.Dockerfile).
RUN cd /workspace/isaaclab \
    && git apply /workspace/ecm/patches/isaaclab-6.1.11-isaacsim-6.0.1.patch \
    && ./isaaclab.sh -p -m pip install --no-deps -e /workspace/isaaclab/source/isaaclab \
    && ./isaaclab.sh -p -m pip install -c /workspace/ecm/requirements/runtime-constraints.txt -e /workspace/ecm \
    && ! find /isaac-sim -xtype l | grep .

# Fixes for the project's import errors. patch, because the copied submodule's .git points nowhere.
RUN cd /workspace/ecm \
    && patch -p1 < patches/ecm2027-sim-track-fixes.patch

# Kit logs here, and the base image hands over only the Isaac-Sim subfolder, not Isaac Lab's.
RUN chown -R isaaclab:isaaclab /isaac-sim/kit/logs

USER isaaclab
WORKDIR /workspace/ecm
