FROM astra/isaac-lab-base:6.0.1

USER root

COPY third_party/ecm2027_sim_track/ /workspace/ecm/
ENV PYTHONPATH=/workspace/ecm:/workspace/ecm/vendor/isaaclab_arena

# Align Isaac Lab's dependency metadata with Isaac Sim 6.0.1.
RUN cd /workspace/isaaclab \
    && git apply /workspace/ecm/patches/isaaclab-6.1.11-isaacsim-6.0.1.patch \
    && ./isaaclab.sh -p -m pip install --no-deps -e /workspace/isaaclab/source/isaaclab \
    && ./isaaclab.sh -p -m pip install -e /workspace/ecm

USER isaaclab
WORKDIR /workspace/ecm
