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
    CAMERA_NAMES,
    STATE_GRIPPER,
    STATE_LEFT_ARM_Q,
    STATE_LEFT_EEF_WORLD,
    STATE_RIGHT_ARM_Q,
    STATE_RIGHT_EEF_WORLD,
    ActView,
    CanonicalObs,
    Gr00tView,
    Stage,
)

N_STAGES = 6  # Stage.LIFT_LID .. Stage.CLOSE_LID; DONE has no one-hot slot


def resize_image(image: np.ndarray, hw: tuple[int, int], interpolation: str = "area") -> np.ndarray:
    """Resize an HWC uint8 image to (H, W) with OpenCV (``area`` or ``linear``), no-op if already that size.

    The offline dataset builder must resize with the same function so ACT sees identical pixels.
    """
    h, w = int(hw[0]), int(hw[1])
    if image.shape[0] == h and image.shape[1] == w:
        return image
    import cv2

    flag = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}[interpolation]
    return cv2.resize(image, (w, h), interpolation=flag)


def stage_one_hot(stage: Stage, enabled: bool = True) -> np.ndarray:
    """(6,) float32 one-hot of the stage; all zeros for DONE or when conditioning is off."""
    out = np.zeros(N_STAGES, dtype=np.float32)
    if enabled and 0 <= int(stage) < N_STAGES:
        out[int(stage)] = 1.0
    return out


def hwc_uint8_to_chw_float(image: np.ndarray) -> np.ndarray:
    """HWC uint8 → CHW float32 in [0, 1]."""
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)), dtype=np.float32) / 255.0


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
        self.interpolation: str = "area"

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
        images = {
            self.feature_keys[name]: hwc_uint8_to_chw_float(resize_image(obs.images[name], self.image_hw, self.interpolation))
            for name in CAMERA_NAMES
        }
        return ActView(
            images=images,
            state=np.asarray(obs.state, dtype=np.float32).copy(),
            stage_one_hot=stage_one_hot(stage, self.use_stage_conditioning),
        )


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
            "left_arm_q": STATE_LEFT_ARM_Q,
            "right_arm_q": STATE_RIGHT_ARM_Q,
            "gripper": STATE_GRIPPER,
            "left_eef_world": STATE_LEFT_EEF_WORLD,
            "right_eef_world": STATE_RIGHT_EEF_WORLD,
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
        video = {
            self.video_keys[name]: np.ascontiguousarray(obs.images[name], dtype=np.uint8)[None, None]
            for name in CAMERA_NAMES
        }
        state = {
            key: np.asarray(obs.state[sl], dtype=np.float32).reshape(1, 1, -1) for key, sl in self.state_keys.items()
        }
        return Gr00tView(video=video, state=state, language={self.language_key: [[prompt]]})


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

        Raises:
            ValueError: empty window.
        """
        if not window:
            raise ValueError("classifier window is empty")
        out: dict[str, np.ndarray] = {}
        for name, hw in self.image_hw.items():
            out[f"images.{name}"] = np.stack(
                [hwc_uint8_to_chw_float(resize_image(o.images[name], hw)) for o in window], axis=0
            )
        out["state"] = np.stack([np.asarray(o.state, dtype=np.float32) for o in window], axis=0)
        return out
