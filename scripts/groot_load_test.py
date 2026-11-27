"""Offline load test of a GR00T checkpoint through the Astra runtime. Run by scripts/groot_load_test.sh.

Runs inside the Isaac-GR00T image (docker/versions.env: ISAAC_GR00T_COMMIT) with networking off and
these mounts: /workspace = Isaac-GR00T checkout, /astra = this repo (read-only), /hf = Hugging Face
hub cache holding Cosmos-Reason2-2B, /ckpt = the checkpoint.

Checks, in order (exit 1 on the first hard failure):
  1. Gr00tPolicyBackend.load(): loads offline, reads horizon and language key from the checkpoint.
  2. Weights: every checkpoint tensor reached the model unchanged (after the cast to its dtype).
  3. Pipeline: ObservationAdapter -> Gr00tView -> predict() gives a finite (horizon, 23) chunk.
  4. Direct get_action at the training-mean state: per action group, the share of values inside
     the training [min, max] and [q01, q99] (informational: the images are synthetic).
  5. Latency and peak GPU memory.

The images come from the test fakes, so this proves loading and interfaces, not task performance.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CK = Path(sys.argv[1] if len(sys.argv) > 1 else "/ckpt")
PROMPT = "Lift the chamber top"
sys.path[:0] = ["/workspace", "/astra/src", "/astra/src/tests", "/astra/third_party/task_repo"]


def fail(msg: str) -> None:
    print(f"LOAD TEST FAILED: {msg}")
    sys.exit(1)


if os.environ.get("HF_HUB_OFFLINE") != "1":
    fail("HF_HUB_OFFLINE is not 1; the test must prove nothing is downloaded")

import torch  # noqa: E402
import transformers  # noqa: E402
from safetensors import safe_open  # noqa: E402

try:
    import flash_attn

    fa = flash_attn.__version__
except ImportError:
    fa = "not installed"
cap = "".join(map(str, torch.cuda.get_device_capability(0)))
print(f"torch {torch.__version__} | transformers {transformers.__version__} | flash_attn {fa} | "
      f"{torch.cuda.get_device_name(0)} sm_{cap}")

from astra.adapter.observation_adapter import ObservationAdapter  # noqa: E402
from astra.policies.groot_policy import Gr00tPolicyBackend  # noqa: E402
from astra.schema import load_schema  # noqa: E402
from astra.types import MODEL_ACTION_DIM, OrchestratorEvent, OrchestratorOutput, Stage  # noqa: E402
from fakes import make_raw_obs  # noqa: E402

# 1. load through the runtime wrapper ----------------------------------------------------------
torch.cuda.reset_peak_memory_stats()
backend = Gr00tPolicyBackend(str(CK), device="cuda:0", strict=True)
t0 = time.time()
backend.load()
print(f"[1] load OK in {time.time() - t0:.1f} s | horizon {backend.action_horizon} | "
      f"language key {backend.language_key!r} | GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
policy = backend.policy
model = policy.model

# 2. every checkpoint weight reached the model -------------------------------------------------
sd = model.state_dict()
index = json.load(open(CK / "model.safetensors.index.json"))["weight_map"]
missing, differ, compared = [], [], 0
for fn in sorted(set(index.values())):
    with safe_open(CK / fn, framework="pt") as f:
        for key in f.keys():
            if key not in sd:
                missing.append(key)
                continue
            got = sd[key].detach().cpu()
            want = f.get_tensor(key).to(got.dtype)
            compared += 1
            if want.shape != got.shape or not torch.equal(want, got):
                differ.append(key)
extra = sorted(k for k in sd if k not in index)
print(f"[2] weights: {compared} compared, {len(differ)} differ, {len(missing)} not in model, "
      f"model-only {extra} | dtype {next(model.parameters()).dtype}")
if missing or differ:
    fail(f"weights not loaded as saved: missing {missing[:3]}, differ {differ[:3]}")
if any(not k.endswith("lm_head.weight") for k in extra):
    fail(f"model has weights the checkpoint does not set: {extra[:5]}")  # lm_head is unused by GR00T

# 3. predict through the Astra pipeline ------------------------------------------------------
schema = load_schema("/astra/src/astra/config/schema.yaml")
adapter = ObservationAdapter(schema, {"history_seconds": 3.0}, use_stage_conditioning=False)
adapter.load()
adapter.reset()
rng = np.random.default_rng(0)
orch = OrchestratorOutput(Stage.LIFT_LID, 0, PROMPT, OrchestratorEvent.NONE, 10.0)
latency = []
for _ in range(6):
    obs = adapter.process(make_raw_obs(rng))
    view = adapter.groot_view(obs, PROMPT)
    torch.cuda.synchronize()
    t = time.time()
    chunk = backend.predict(obs, view, orch)
    torch.cuda.synchronize()
    latency.append(time.time() - t)
a = chunk.actions
print(f"[3] pipeline predict: chunk {a.shape} source {chunk.source.name} fps {chunk.fps} | "
      f"finite {np.isfinite(a).all()} | latency first {latency[0] * 1e3:.0f} ms, "
      f"then median {np.median(latency[1:]) * 1e3:.0f} ms")
if a.shape != (backend.action_horizon, MODEL_ACTION_DIM) or not np.isfinite(a).all():
    fail(f"bad chunk: shape {a.shape}, finite {np.isfinite(a).all()}")

# 4. direct get_action at the training-mean state ----------------------------------------------
stats = json.load(open(CK / "statistics.json"))["new_embodiment"]
mc = policy.get_modality_config()
state = {k: np.asarray(stats["state"][k]["mean"], np.float32)[None, None] for k in mc["state"].modality_keys}
video = {k: view.video[k] for k in mc["video"].modality_keys}
action, _ = policy.get_action({"video": video, "state": state, "language": {backend.language_key: [[PROMPT]]}})
print("[4] direct get_action at the training-mean state (synthetic images):")
for k in mc["action"].modality_keys:
    v = np.asarray(action[k])[0]
    s = stats["action"][k]
    in_range = np.mean((v >= np.asarray(s["min"]) - 1e-6) & (v <= np.asarray(s["max"]) + 1e-6))
    in_q = np.mean((v >= np.asarray(s["q01"]) - 1e-6) & (v <= np.asarray(s["q99"]) + 1e-6))
    print(f"    {k:10s} {v.shape} finite {np.isfinite(v).all()} | in [min,max] {in_range:.0%} | "
          f"in [q01,q99] {in_q:.0%}")
    if not np.isfinite(v).all():
        fail(f"non-finite {k}")

# 5. resources ----------------------------------------------------------------------------------
print(f"[5] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("LOAD TEST PASSED")
