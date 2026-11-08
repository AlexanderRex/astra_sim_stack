FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ecm

COPY third_party/ecm2027_sim_track/ /workspace/ecm/
COPY docker/policy-requirements.txt /tmp/policy-requirements.txt
COPY docker/run-policy.sh /usr/local/bin/run-policy

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --constraint /workspace/ecm/requirements/runtime-constraints.txt \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 \
    && python -m pip install \
        --constraint /workspace/ecm/requirements/runtime-constraints.txt \
        --editable /workspace/ecm \
    && if [ -s /tmp/policy-requirements.txt ]; then \
        python -m pip install -r /tmp/policy-requirements.txt; \
    fi \
    && chmod +x /usr/local/bin/run-policy

ENTRYPOINT ["/usr/local/bin/run-policy"]
