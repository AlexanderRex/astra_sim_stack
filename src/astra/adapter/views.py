"""Builders for model-specific views of a canonical observation.

Purpose
-------
Each consumer wants the same information in a different format. These builders convert a
``CanonicalObs`` without changing its content; **no normalisation happens here** because each
checkpoint carries its own statistics (LeRobot pre/postprocessor for ACT, Gr00tN1d7Processor for
GR00T). Normalising here too would normalise twice or with the wrong stats.

Inputs:  ``CanonicalObs`` (+ stage / prompt from the orchestrator where needed).
Outputs: ``ActView``, ``Gr00tView``, classifier tensors.
"""

from __future__ import annotations

import numpy as np

from astra.schema import Schema
from astra.types import (
    STATE_GRIPPER,
    STATE_LEFT_ARM_Q,
    STATE_LEFT_EEF,
    STATE_RIGHT_ARM_Q,
    STATE_RIGHT_EEF,
    ActView,
    CanonicalObs,
    Gr00tView,
    Stage,
)


class ActViewBuilder:
    """CanonicalObs → ActView (LeRobot ACT input keys).

    Attributes:
        schema: resolved schema (camera specs).
        image_hw: (H, W) ACT input size, 224×224; the head camera (480×640) is resized to it with
            the same interpolation as the training pipeline (aspect ratio is squashed, as in training).
        feature_keys: Astra camera name → LeRobot feature key
            (``observation.images.first_person_camera_rgb`` etc.), must equal the checkpoint's
            ``input_features``.
        use_stage_conditioning: if False, ``stage_one_hot`` is all zeros.
    """

    def __init__(
        self, schema: Schema, image_hw: tuple[int, int] = (224, 224), use_stage_conditioning: bool = True
    ) -> None:
        self.schema = schema
        self.image_hw = image_hw
        self.feature_keys: dict[str, str] = {
            name: f"observation.images.{spec.obs_key}" for name, spec in schema.cameras.items()
        }
        self.use_stage_conditioning = use_stage_conditioning

    def build(self, obs: CanonicalObs, stage: Stage) -> ActView:
        """Build the ACT view.

        Functionality:
            Resize each image to ``image_hw``, convert HWC uint8 → CHW float32 in [0, 1], pass the
            34-D state through unchanged, and one-hot encode the stage (6 classes; DONE → zeros).

        Args:
            obs: canonical observation.
            stage: current orchestrator stage.

        Returns:
            ActView ready for the ACT wrapper (which adds the batch dim and runs the preprocessor).
        """
        raise NotImplementedError("ActViewBuilder.build")


class Gr00tViewBuilder:
    """CanonicalObs → Gr00tView in the ``Gr00tPolicy.get_action`` format.

    Attributes:
        schema: resolved schema (camera specs, state group order).
        video_keys: Astra camera name → GR00T video key (must match modality.json ``video``).
        state_keys: GR00T state group name → slice of the 34-D state, in galbot_config.py order.
        language_key: language modality key (``task`` in the Policy API dict).
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.video_keys: dict[str, str] = {name: name for name in schema.cameras}
        self.state_keys: dict[str, slice] = {
            "left_eef_9d": STATE_LEFT_EEF,
            "right_eef_9d": STATE_RIGHT_EEF,
            "left_arm_q": STATE_LEFT_ARM_Q,
            "right_arm_q": STATE_RIGHT_ARM_Q,
            "gripper": STATE_GRIPPER,
        }
        self.language_key = "task"

    def build(self, obs: CanonicalObs, prompt: str) -> Gr00tView:
        """Build the GR00T view.

        Functionality:
            Images stay HWC uint8 at native resolution (GR00T's processor resizes, keeping the
            aspect ratio), reshaped to (1, 1, H, W, 3). State groups are sliced from the 34-D state
            and reshaped to (1, 1, D) float32. Prompt goes to ``{"task": [[prompt]]}``.

        Args:
            obs: canonical observation.
            prompt: stage prompt from the orchestrator.

        Returns:
            Gr00tView.
        """
        raise NotImplementedError("Gr00tViewBuilder.build")


class ClassifierViewBuilder:
    """History window → stage-classifier input tensors.

    Attributes:
        seconds: history length fed to the classifier (3 s).
        rate_hz: sampling rate (5 Hz → 15 steps).
        image_hw: per-camera low-resolution size, e.g. head 224×224, wrists 112×112.
    """

    def __init__(self, seconds: float = 3.0, rate_hz: float = 5.0) -> None:
        self.seconds = seconds
        self.rate_hz = rate_hz
        self.image_hw: dict[str, tuple[int, int]] = {
            "head": (224, 224),
            "left_wrist": (112, 112),
            "right_wrist": (112, 112),
        }

    def build(self, window: list[CanonicalObs]) -> dict[str, np.ndarray]:
        """Stack a resampled history window into classifier inputs.

        Args:
            window: list of CanonicalObs from ``HistoryBuffer.window(seconds, rate_hz)``.

        Returns:
            ``{"images.<camera>": (T, 3, H, W) float32, "state": (T, 34) float32}``.
        """
        raise NotImplementedError("ClassifierViewBuilder.build")
