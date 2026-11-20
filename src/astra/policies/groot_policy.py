"""GR00T N1.7 policy wrapper (Isaac-GR00T ``Gr00tPolicy``).

Purpose
-------
Run a GR00T N1.7 checkpoint fine-tuned with ``EmbodimentTag.NEW_EMBODIMENT`` and return chunks in the
23-D model action space.

What the framework does (configured by galbot_config.py / modality.json, saved in the checkpoint)
-----------------------------------------------------------------------------------------------
- Processor (Gr00tN1d7Processor): images at native aspect ratio (shortest edge 256), state min-max
  normalised and padded to 132, prompt tokenised.
- Cosmos-Reason2-2B (Qwen3-VL) backbone, flow-matching DiT (16 layers), 40-step chunk, 4 denoising
  steps.
- Un-processor (``decode_action``): slices the used dims per action group, un-normalises, converts
  RELATIVE EEF actions (left_eef, right_eef: RELATIVE, EEF, XYZ_ROT6D, state_key=*_eef_9d) back to
  absolute using the current ``state.*_eef_9d``.

Inputs
------
- ``Gr00tView`` (video (1,1,H,W,3) uint8, state (1,1,D) float32, language [[prompt]]).

Outputs
-------
- ``ActionChunk`` (40, 23): ``action.left_eef`` (40, 9) | ``action.right_eef`` (40, 9) |
  ``action.gripper`` (40, 2, continuous ≈ [0, 1], no logits) | ``action.base_vel`` (40, 3);
  absolute, physical units, fps = data rate (50).
"""

from __future__ import annotations

from typing import Any

from astra.policies.base import PolicyBackend
from astra.types import DATA_FPS, ActionChunk, CanonicalObs, Gr00tView, OrchestratorOutput, PolicySource


class Gr00tPolicyBackend(PolicyBackend):
    """GR00T N1.7 chunk producer.

    Attributes:
        checkpoint: fine-tuned checkpoint directory (contains the processor and modality config).
        device: torch device string.
        action_groups: output group order concatenated into the 23-D vector.
        action_horizon: predicted chunk length (40 for N1.7).
        policy: ``gr00t.policy.Gr00tPolicy`` instance, None until ``load()``.
        strict: pass ``strict=True`` during development to validate shapes/dtypes.
    """

    source = PolicySource.GROOT
    fps = DATA_FPS

    def __init__(self, checkpoint: str | None, device: str = "cuda:0", strict: bool = True) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.device = device
        self.action_groups: tuple[str, ...] = ("left_eef", "right_eef", "gripper", "base_vel")
        self.action_horizon: int = 40
        self.policy: Any = None
        self.strict = strict

    def load(self) -> None:
        """Create ``Gr00tPolicy(model_path, embodiment_tag=NEW_EMBODIMENT, device, strict)``.

        Functionality:
            Lazy import of ``gr00t``; run fully offline (HF_HUB_OFFLINE=1, Cosmos-Reason2-2B cached
            in the image); check ``get_modality_config()`` lists the expected video/state/action keys
            and the action horizon.

        Raises:
            FileNotFoundError: checkpoint or backbone cache missing.
            ValueError: modality config does not match the schema.
        """
        raise NotImplementedError("Gr00tPolicyBackend.load")

    def reset(self) -> None:
        """No recurrent state in GR00T; clear any cached real-time-chunking context."""
        raise NotImplementedError("Gr00tPolicyBackend.reset")

    def predict(self, obs: CanonicalObs, view: Gr00tView, orch: OrchestratorOutput) -> ActionChunk:
        """Call ``policy.get_action(observation)`` and concatenate the groups into (40, 23).

        Args:
            obs: canonical observation (tick → start_tick).
            view: GR00T view of this tick (prompt already inside).
            orch: orchestrator output (stage → chunk.stage).

        Returns:
            ActionChunk (40, 23), source GROOT, fps = data rate.
        """
        raise NotImplementedError("Gr00tPolicyBackend.predict")
