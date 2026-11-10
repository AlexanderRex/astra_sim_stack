# Light client image: Python, CUDA torch and the task repo's dependencies, no Isaac.
# The task repo and src/ are mounted at run time (docker-compose.yml), so code edits need no rebuild.
FROM python:3.12-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# torch first: the largest layer, and it changes only with the task repo's pin.
RUN --mount=type=bind,source=third_party/task_repo/requirements/runtime-constraints.txt,target=/tmp/constraints.txt \
    grep -E '^torch==' /tmp/constraints.txt > /tmp/torch.txt \
    && pip install --index-url https://download.pytorch.org/whl/cu130 -r /tmp/torch.txt \
    && rm /tmp/torch.txt

# libGL and glib for opencv; a compiler and kernel headers for pynput's evdev, built from source.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

# The task repo's own dependencies, read from its pyproject.toml, and the policy's from src/.
RUN --mount=type=bind,source=third_party/task_repo/pyproject.toml,target=/tmp/task/pyproject.toml \
    --mount=type=bind,source=third_party/task_repo/requirements/runtime-constraints.txt,target=/tmp/task/constraints.txt \
    --mount=type=bind,source=src/requirements.txt,target=/tmp/src-requirements.txt \
    python -c 'import tomllib; print("\n".join(tomllib.load(open("/tmp/task/pyproject.toml", "rb"))["project"]["dependencies"]))' \
        > /tmp/task-requirements.txt \
    && pip install -c /tmp/task/constraints.txt -r /tmp/task-requirements.txt -r /tmp/src-requirements.txt \
    && rm /tmp/task-requirements.txt

# Mount point for the volume shared with the sim container (see docker-compose.yml).
RUN install -d -m 1777 /tmp/ipc

ENV PYTHONPATH=/workspace/src:/workspace/task:/workspace/task/vendor/isaaclab_arena
WORKDIR /workspace/task
ENTRYPOINT ["bash", "scripts/run_policy_eval.sh"]
CMD ["/workspace/src/smoke.yml"]
