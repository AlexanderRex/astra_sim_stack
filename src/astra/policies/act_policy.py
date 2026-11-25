"""Custom ACT policy wrapper (LeRobot).

Purpose
-------
Run a trained ACT checkpoint and return chunks in the 23-D model action space.

Model summary (trained offline)
-------------------------------
ResNet-18 per camera (3 × 224×224 → 49 tokens each) + 34-D state token + stage token + CVAE latent
(z = 0 at inference) → 4-layer transformer encoder → 1-layer DETR decoder with k queries → heads:
512 → 21 (left_eef 9, right_eef 9, base_vel 3) and 512 → 2 (gripper logits).

Inputs
------
- ``ActView``: images CHW float [0, 1], state (34,), stage one-hot (6,).
- Checkpoint directory with ``config.json``, ``model.safetensors``, ``policy_preprocessor.json`` and
  ``policy_postprocessor.json`` (normalisation stats saved with the checkpoint).

Outputs
-------
- ``ActionChunk`` (k, 23), fps = data rate (50), gripper dims converted from logits to probabilities in [0, 1].

Note
----
LeRobot's stock postprocessor un-normalises every action dim with mean/std, which would corrupt
gripper logits. The custom policy class must exclude the gripper dims from un-normalisation (or the
gripper is trained as a regressed ±1 value and mapped to [0, 1] here).

``gripper_mode`` selects how the two gripper dims are read:

- ``prob`` (default): regressed {0, 1} targets, as ``scripts/convert_dataset.py`` writes them; the
  un-normalised output is clipped to [0, 1]. Works with stock LeRobot ACT.
- ``logit``: the network emits logits; the sigmoid is applied to the raw (normalised-space) output,
  skipping un-normalisation for these dims.
- ``pm1``: regressed ±1 targets; mapped with ``(x + 1) / 2``.

Stage conditioning is fed as ``observation.stage`` (6,) when the checkpoint lists that input feature;
a stock ACT checkpoint without it ignores the stage.

Written against LeRobot ≥ 0.4 (``ACTPolicy.from_pretrained``, ``make_pre_post_processors``,
``predict_action_chunk``); LeRobot is imported only in ``load``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from astra.policies.base import PolicyBackend
from astra.types import (
    CAMERA_NAMES,
    DATA_FPS,
    MODEL_ACTION_DIM,
    MODEL_GRIPPER,
    STATE_DIM,
    ActionChunk,
    ActView,
    CanonicalObs,
    OrchestratorOutput,
    PolicySource,
)

log = logging.getLogger(__name__)
STAGE_FEATURE = "observation.stage"
STATE_FEATURE = "observation.state"


class ActPolicy(PolicyBackend):
    """ACT chunk producer.

    Attributes:
        checkpoint: path to ``pretrained_model/``.
        device: torch device string.
        chunk_size: k, read from the checkpoint config (50-100).
        model: the torch policy, None until ``load()``.
        preprocessor / postprocessor: LeRobot processor pipelines loaded from the checkpoint.
        use_stage_conditioning: feed the stage one-hot (requires a stage-conditioned checkpoint).
        gripper_mode: "prob", "logit" or "pm1" (see module docstring).
        image_keys: LeRobot image feature keys the checkpoint expects.
        has_stage_input: the checkpoint lists ``observation.stage``.
    """

    source = PolicySource.ACT
    fps = DATA_FPS

    def __init__(
        self,
        checkpoint: str | None,
        device: str = "cuda:0",
        use_stage_conditioning: bool = True,
        gripper_mode: str = "prob",
        fps: float = DATA_FPS,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.device = device
        self.chunk_size: int = 100
        self.model: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.use_stage_conditioning = use_stage_conditioning
        if gripper_mode not in ("prob", "logit", "pm1"):
            raise ValueError(f"ACT gripper_mode must be prob, logit or pm1, got {gripper_mode!r}")
        self.gripper_mode = gripper_mode
        self.fps = float(fps)
        self.image_keys: list[str] = []
        self.has_stage_input: bool = False

    def load(self) -> None:
        """Load the ACT checkpoint and its pre/postprocessors (lazy lerobot / torch import).

        Functionality:
            Read ``config.json``, check input features (3 cameras, state 34) and output (23) match
            the schema, load weights to ``device``, set eval mode, read ``chunk_size``.

        Raises:
            FileNotFoundError: checkpoint missing.
            ValueError: feature shapes do not match the schema.
        """
        if not self.checkpoint or not Path(self.checkpoint).is_dir():
            raise FileNotFoundError(f"ACT checkpoint directory not found: {self.checkpoint}")
        from lerobot.policies.act.modeling_act import ACTPolicy as LeRobotACT
        from lerobot.policies.factory import make_pre_post_processors

        policy = LeRobotACT.from_pretrained(self.checkpoint)
        cfg = policy.config
        inputs = {k: tuple(getattr(v, "shape", ())) for k, v in cfg.input_features.items()}
        outputs = {k: tuple(getattr(v, "shape", ())) for k, v in cfg.output_features.items()}
        if inputs.get(STATE_FEATURE) != (STATE_DIM,):
            raise ValueError(f"ACT checkpoint state feature {inputs.get(STATE_FEATURE)}, expected ({STATE_DIM},)")
        if outputs.get("action") != (MODEL_ACTION_DIM,):
            raise ValueError(f"ACT checkpoint action feature {outputs.get('action')}, expected ({MODEL_ACTION_DIM},)")
        self.image_keys = sorted(k for k in inputs if k.startswith("observation.images."))
        if len(self.image_keys) != len(CAMERA_NAMES):
            raise ValueError(f"ACT checkpoint has image inputs {self.image_keys}, expected {len(CAMERA_NAMES)} cameras")
        self.has_stage_input = STAGE_FEATURE in inputs
        if self.use_stage_conditioning and not self.has_stage_input:
            log.warning("ACT checkpoint has no %s input: stage conditioning is off", STAGE_FEATURE)
        self.chunk_size = int(cfg.chunk_size)
        policy.to(self.device)
        policy.eval()
        self.model = policy
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=self.checkpoint, preprocessor_overrides={"device_processor": {"device": self.device}}
        )
        self.loaded = True
        log.info("ACT loaded from %s: chunk %d, images %s", self.checkpoint, self.chunk_size, self.image_keys)

    def reset(self) -> None:
        """Clear LeRobot's internal action queue / temporal-ensembling state."""
        if self.model is not None and hasattr(self.model, "reset"):
            self.model.reset()

    def predict(self, obs: CanonicalObs, view: ActView, orch: OrchestratorOutput) -> ActionChunk:
        """Run ACT once and return the full chunk.

        Functionality:
            Build the batch dict from ``view`` (add batch dim), run the preprocessor, call the model
            for the whole chunk (not ``select_action``, so the executor controls re-planning), run
            the postprocessor on continuous dims, map the gripper dims per ``gripper_mode``.

        Args:
            obs: canonical observation (tick → ``start_tick``).
            view: ACT view of this tick.
            orch: orchestrator output (stage → chunk.stage).

        Returns:
            ActionChunk (k, 23), source ACT, fps = data rate.

        Raises:
            RuntimeError: not loaded.
            KeyError: the view lacks an image the checkpoint expects.
        """
        if self.model is None:
            raise RuntimeError("ActPolicy.predict before load()")
        import torch

        batch: dict[str, Any] = {}
        for key in self.image_keys:
            if key not in view.images:
                raise KeyError(f"ACT view has no {key}; it has {sorted(view.images)}")
            batch[key] = torch.from_numpy(view.images[key])[None]
        batch[STATE_FEATURE] = torch.from_numpy(np.asarray(view.state, dtype=np.float32))[None]
        if self.has_stage_input:
            batch[STAGE_FEATURE] = torch.from_numpy(np.asarray(view.stage_one_hot, dtype=np.float32))[None]
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            raw = self.model.predict_action_chunk(batch)  # (1, k, 23), normalised space
            post = self.postprocessor(raw)
        raw_np = raw.detach().float().cpu().numpy()[0]
        actions = post.detach().float().cpu().numpy()[0] if hasattr(post, "detach") else np.asarray(post)[0]
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, MODEL_ACTION_DIM)
        g = MODEL_GRIPPER
        if self.gripper_mode == "logit":
            actions[:, g] = 1.0 / (1.0 + np.exp(-raw_np[:, g]))
        elif self.gripper_mode == "pm1":
            actions[:, g] = np.clip((actions[:, g] + 1.0) / 2.0, 0.0, 1.0)
        else:
            actions[:, g] = np.clip(actions[:, g], 0.0, 1.0)
        return ActionChunk(actions=actions, source=self.source, fps=self.fps, start_tick=obs.tick, stage=orch.stage)
