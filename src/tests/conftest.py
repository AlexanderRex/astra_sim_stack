"""Test setup: put ``src/`` on the path and provide ``policy.base`` when the task repo is unusable.

``astra.policy`` imports ``policy.base.BasePolicy`` from the task repo. Inside the policy container
it is on PYTHONPATH; on a dev machine the submodule may be missing or its dependencies (mediapy) not
installed. Then a minimal stand-in with the same constructor and ``add_video_frame`` is registered,
so the control loop can still be tested.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
TASK_REPO = REPO / "third_party" / "task_repo"
for p in (str(SRC), str(TASK_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import policy.base  # noqa: F401
except Exception:  # noqa: BLE001 - missing submodule or missing dependency
    base = types.ModuleType("policy.base")

    class BasePolicy:  # minimal stand-in for the task repo's BasePolicy
        def __init__(self, usr_args):
            self.usr_args = usr_args
            self.model = None
            self.instruction = usr_args.get("instruction", "")
            self._video_writer = None
            self.get_model(usr_args)

        def add_video_frame(self, video_writer, obs, camera_key):
            if video_writer is not None:  # the real one writes the side-by-side image with add_image
                video_writer.add_image(len(camera_key))

    base.BasePolicy = BasePolicy
    pkg = types.ModuleType("policy")
    pkg.base = base
    sys.modules["policy"] = pkg
    sys.modules["policy.base"] = base


@pytest.fixture
def schema():
    from astra.schema import load_schema

    return load_schema(SRC / "astra" / "config" / "schema.yaml")


@pytest.fixture
def rng():
    return np.random.default_rng(0)
