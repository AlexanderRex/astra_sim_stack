"""GR00T N1.7 policy wrapper (Isaac-GR00T ``Gr00tPolicy``).

Purpose
-------
Run a GR00T N1.7 checkpoint fine-tuned with ``EmbodimentTag.NEW_EMBODIMENT`` and return chunks in the
23-D model action space.

What the framework does (configured by galbot_config.py / modality.json, saved in the checkpoint)
-----------------------------------------------------------------------------------------------
- Processor (Gr00tN1d7Processor): images at native aspect ratio (shortest edge 256), state min-max
  normalised and padded to 132, prompt tokenised.
- Cosmos-Reason2-2B (Qwen3-VL) backbone (first 16 LLM layers kept), flow-matching DiT (32 blocks),
  40-step chunk, 4 denoising steps.
- Un-processor (``decode_action``): slices the used dims per action group and un-normalises. Every
  group is ABSOLUTE / NON_EEF / DEFAULT: the arm targets are poses in the robot root frame, and the
  state holds no pose in that frame for them to be relative to.

Inputs
------
- ``Gr00tView`` (video (1,1,H,W,3) uint8, state (1,1,D) float32, language [[prompt]]).

Outputs
-------
- ``ActionChunk`` (40, 23): ``action.left_eef`` (40, 9) | ``action.right_eef`` (40, 9) |
  ``action.gripper`` (40, 2, continuous ≈ [0, 1], no logits) | ``action.base_vel`` (40, 3);
  absolute, physical units, fps = data rate (50).

Written against the Isaac-GR00T ``Gr00tPolicy`` API (``gr00t.policy.gr00t_policy``, falling back to
``gr00t.model.policy``); ``get_action`` may return the action dict or ``(action, info)`` and the
group keys with or without the ``action.`` prefix. GR00T is imported only in ``load``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from astra.policies.base import PolicyBackend
from astra.types import (
    DATA_FPS,
    MODEL_ACTION_DIM,
    MODEL_BASE_VEL,
    MODEL_GRIPPER,
    MODEL_LEFT_EEF,
    MODEL_RIGHT_EEF,
    ActionChunk,
    CanonicalObs,
    Gr00tView,
    OrchestratorOutput,
    PolicySource,
)

log = logging.getLogger(__name__)
GROUP_SLICES = {"left_eef": MODEL_LEFT_EEF, "right_eef": MODEL_RIGHT_EEF, "gripper": MODEL_GRIPPER,
                "base_vel": MODEL_BASE_VEL}
# Keys Gr00tViewBuilder produces; the checkpoint's modality config must list the same ones.
VIDEO_KEYS = ("head", "left_wrist", "right_wrist")
STATE_KEYS = ("left_arm_q", "right_arm_q", "gripper", "left_eef_world", "right_eef_world")


def _import_gr00t() -> tuple[Any, Any]:
    try:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
    except ImportError:
        from gr00t.model.policy import Gr00tPolicy  # older layout
    from gr00t.data.embodiment_tags import EmbodimentTag

    return Gr00tPolicy, EmbodimentTag


class Gr00tPolicyBackend(PolicyBackend):
    """GR00T N1.7 chunk producer.

    Attributes:
        checkpoint: fine-tuned checkpoint directory (contains the processor and modality config).
        device: torch device string.
        action_groups: output group order concatenated into the 23-D vector.
        action_horizon: predicted chunk length (40 for N1.7).
        policy: ``gr00t.policy.Gr00tPolicy`` instance, or a ``GrootServerClient`` in server mode;
            None until ``load()``.
        server: ``host:port`` of a GR00T inference server (``scripts/start_groot_server.sh``), or None
            to load ``checkpoint`` in this process (needs ``gr00t`` installed here).
        server_timeout_s / server_wait_s: per-request timeout; how long ``load()`` waits for the
            server to answer (it loads the model before it listens).
        language_key: language modality key the checkpoint was trained with (read on ``load()``;
            Gr00tPolicy looks the prompt up under it, e.g. ``annotation.human.task_description``).
        strict: pass ``strict=True`` during development to validate shapes/dtypes.
    """

    source = PolicySource.GROOT
    fps = DATA_FPS

    def __init__(self, checkpoint: str | None, device: str = "cuda:0", strict: bool = True, fps: float = DATA_FPS,
                 server: str | None = None, server_timeout_s: float = 15.0, server_wait_s: float = 300.0) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.server = server
        self.server_timeout_s = float(server_timeout_s)
        self.server_wait_s = float(server_wait_s)
        self.device = device
        self.action_groups: tuple[str, ...] = ("left_eef", "right_eef", "gripper", "base_vel")
        self.action_horizon: int = 40
        self.policy: Any = None
        self.language_key: str | None = None
        self.strict = strict
        self.fps = float(fps)

    def load(self) -> None:
        """Connect to the GR00T server, or create ``Gr00tPolicy(model_path, NEW_EMBODIMENT, device, strict)``.

        Functionality:
            Server mode (``server`` set): wait until the server answers, then use it like a local
            policy; the checkpoint is whatever the server loaded. Otherwise lazy import of ``gr00t``
            and run fully offline (HF_HUB_OFFLINE=1, Cosmos-Reason2-2B cached in the image). Either
            way, check ``get_modality_config()`` lists the expected video/state/action keys and read
            the action horizon and language key.

        Raises:
            FileNotFoundError: checkpoint or backbone cache missing (in-process mode).
            TimeoutError: server did not answer within ``server_wait_s`` (server mode).
            ValueError: modality config does not match the schema.
        """
        if self.server:
            from astra.policies.groot_client import GrootServerClient

            self.policy = GrootServerClient(self.server, timeout_s=self.server_timeout_s)
            self.policy.wait_ready(self.server_wait_s)
            self._check_modality()
            self.loaded = True
            log.info("GR00T server %s: horizon %d, language key %r", self.server, self.action_horizon,
                     self.language_key)
            return
        if not self.checkpoint or not Path(self.checkpoint).is_dir():
            raise FileNotFoundError(f"GR00T checkpoint directory not found: {self.checkpoint}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        Gr00tPolicy, EmbodimentTag = _import_gr00t()
        kwargs: dict[str, Any] = {"embodiment_tag": EmbodimentTag.NEW_EMBODIMENT, "model_path": self.checkpoint,
                                  "device": self.device}
        try:
            self.policy = Gr00tPolicy(strict=self.strict, **kwargs)
        except TypeError:  # revisions without the strict flag
            self.policy = Gr00tPolicy(**kwargs)
        self._check_modality()
        self.loaded = True
        log.info("GR00T loaded from %s: horizon %d", self.checkpoint, self.action_horizon)

    def _check_modality(self) -> None:
        get = getattr(self.policy, "get_modality_config", None)
        if get is None:
            return
        cfg = get()
        if not isinstance(cfg, dict):
            return

        def keys(modality: str, prefix: str) -> list[str] | None:
            mc = cfg.get(modality)
            if mc is None:
                return None
            return [str(k).removeprefix(prefix) for k in getattr(mc, "modality_keys", [])]

        for modality, prefix, want in (("video", "video.", VIDEO_KEYS), ("state", "state.", STATE_KEYS),
                                       ("action", "action.", self.action_groups)):
            got = keys(modality, prefix)
            if got and got != list(want):
                raise ValueError(f"GR00T {modality} keys {got}, expected {list(want)}")
        language = cfg.get("language")
        lang_keys = list(getattr(language, "modality_keys", [])) if language is not None else []
        if len(lang_keys) > 1:
            raise ValueError(f"GR00T checkpoint has {len(lang_keys)} language keys {lang_keys}, expected one")
        if lang_keys:
            self.language_key = str(lang_keys[0])
        horizon = getattr(cfg.get("action"), "delta_indices", None)
        if horizon is not None:
            self.action_horizon = len(list(horizon))

    def reset(self) -> None:
        """No recurrent state in GR00T; clear any cached real-time-chunking context."""
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, obs: CanonicalObs, view: Gr00tView, orch: OrchestratorOutput) -> ActionChunk:
        """Call ``policy.get_action(observation)`` and concatenate the groups into (40, 23).

        Args:
            obs: canonical observation (tick → start_tick).
            view: GR00T view of this tick (prompt already inside).
            orch: orchestrator output (stage → chunk.stage).

        Returns:
            ActionChunk (40, 23), source GROOT, fps = data rate.

        Raises:
            RuntimeError: not loaded.
            KeyError: an action group is missing from the output.
            ValueError: group dimensions or lengths do not match.
        """
        if self.policy is None:
            raise RuntimeError("Gr00tPolicyBackend.predict before load()")
        language = view.language
        if self.language_key is not None and self.language_key not in language:
            language = {self.language_key: next(iter(language.values()))}  # the view uses a generic key
        observation = {"video": view.video, "state": view.state, "language": language}
        result = self.policy.get_action(observation)
        action = result[0] if isinstance(result, tuple) else result
        groups: dict[str, np.ndarray] = {}
        for name in self.action_groups:
            value = action.get(name, action.get(f"action.{name}"))
            if value is None:
                raise KeyError(f"GR00T output has no {name}; keys {sorted(action)}")
            arr = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            sl = GROUP_SLICES[name]
            if arr.ndim != 2 or arr.shape[1] != sl.stop - sl.start:
                raise ValueError(f"GR00T group {name} has shape {arr.shape}, expected (T, {sl.stop - sl.start})")
            groups[name] = arr
        lengths = {len(v) for v in groups.values()}
        if len(lengths) != 1:
            raise ValueError(f"GR00T groups have different lengths {lengths}")
        T = lengths.pop()
        actions = np.zeros((T, MODEL_ACTION_DIM), dtype=np.float32)
        for name, arr in groups.items():
            actions[:, GROUP_SLICES[name]] = arr
        actions[:, MODEL_GRIPPER] = np.clip(actions[:, MODEL_GRIPPER], 0.0, 1.0)
        return ActionChunk(actions=actions, source=self.source, fps=self.fps, start_tick=obs.tick, stage=orch.stage)
