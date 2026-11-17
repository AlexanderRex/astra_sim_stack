"""Custom ACT policy wrapper (LeRobot).

Purpose
-------
Run a trained ACT checkpoint and return chunks in the 24-D model action space.

Model summary (trained offline)
-------------------------------
ResNet-18 per camera (3 × 224×224 → 49 tokens each) + 34-D state token + stage token + CVAE latent
(z = 0 at inference) → 4-layer transformer encoder → 1-layer DETR decoder with k queries → heads:
512 → 22 (left_eef 9, right_eef 9, wheel_vel 4) and 512 → 2 (gripper logits).

Inputs
------
- ``ActView``: images CHW float [0, 1], state (34,), stage one-hot (6,).
- Checkpoint directory with ``config.json``, ``model.safetensors``, ``policy_preprocessor.json`` and
  ``policy_postprocessor.json`` (normalisation stats saved with the checkpoint).

Outputs
-------
- ``ActionChunk`` (k, 24), fps = 30, gripper dims converted from logits to probabilities in [0, 1].

Note
----
LeRobot's stock postprocessor un-normalises every action dim with mean/std, which would corrupt
gripper logits. The custom policy class must exclude the gripper dims from un-normalisation (or the
gripper is trained as a regressed ±1 value and mapped to [0, 1] here).
"""

from __future__ import annotations

from typing import Any

from astra.policies.base import PolicyBackend
from astra.types import DATA_FPS, ActionChunk, ActView, CanonicalObs, OrchestratorOutput, PolicySource


class ActPolicy(PolicyBackend):
    """ACT chunk producer.

    Attributes:
        checkpoint: path to ``pretrained_model/``.
        device: torch device string.
        chunk_size: k, read from the checkpoint config (50-100).
        model: the torch policy, None until ``load()``.
        preprocessor / postprocessor: LeRobot processor pipelines loaded from the checkpoint.
        use_stage_conditioning: feed the stage one-hot (requires a stage-conditioned checkpoint).
    """

    source = PolicySource.ACT
    fps = DATA_FPS

    def __init__(self, checkpoint: str | None, device: str = "cuda:0", use_stage_conditioning: bool = True) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.device = device
        self.chunk_size: int = 100
        self.model: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.use_stage_conditioning = use_stage_conditioning

    def load(self) -> None:
        """Load the ACT checkpoint and its pre/postprocessors (lazy lerobot / torch import).

        Functionality:
            Read ``config.json``, check input features (3 cameras, state 34) and output (24) match
            the schema, load weights to ``device``, set eval mode, read ``chunk_size``.

        Raises:
            FileNotFoundError: checkpoint missing.
            ValueError: feature shapes do not match the schema.
        """
        raise NotImplementedError("ActPolicy.load")

    def reset(self) -> None:
        """Clear LeRobot's internal action queue / temporal-ensembling state."""
        raise NotImplementedError("ActPolicy.reset")

    def predict(self, obs: CanonicalObs, view: ActView, orch: OrchestratorOutput) -> ActionChunk:
        """Run ACT once and return the full chunk.

        Functionality:
            Build the batch dict from ``view`` (add batch dim), run the preprocessor, call the model
            for the whole chunk (not ``select_action``, so the executor controls re-planning), run
            the postprocessor on continuous dims, sigmoid on gripper logits.

        Args:
            obs: canonical observation (tick → ``start_tick``).
            view: ACT view of this tick.
            orch: orchestrator output (stage → chunk.stage).

        Returns:
            ActionChunk (k, 24), source ACT, fps 30.
        """
        raise NotImplementedError("ActPolicy.predict")
