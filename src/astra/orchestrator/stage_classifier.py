"""Learned stage / phase / progress classifier: inference wrapper.

Purpose
-------
Provide visual evidence of task progress that the proprioceptive detectors cannot: how far the
hinged lid is open, whether the part is in the box, whether the replacement is seated, and which
stage the robot is in when the scene looks ambiguous (start vs stage 6 look alike in a single frame).

Model (trained offline, not part of this package yet)
-----------------------------------------------------
Frozen ResNet-18 features per camera (head 224×224, wrists 112×112) + 34-D state, sampled at 5 Hz
over the last 3 s → GRU → heads: stage (7 incl. DONE), phase (5), progress (0..1), scene_ok (0..1),
lid_open (0..1 regression of the normalized ``Lid001_joint`` angle; labels from the lid joint value
per frame in the HDF5 demos or a sim replay). The lid head is the only lid signal after the hand
releases the handle; stage 6 needs it to resolve "fully shut" (sim threshold 0.005 ≈ 0.45°).
The current stage estimate is **not** an input (avoids a feedback shortcut).

Inputs:  classifier tensors from ``ClassifierViewBuilder.build`` (history window).
Outputs: ``StagePrediction``. Runs at 5 Hz; between runs the last prediction is reused.
"""

from __future__ import annotations

from typing import Any

from astra.types import CONTROL_HZ, StagePrediction


class StageClassifier:
    """Wrapper around the trained classifier network.

    Attributes:
        checkpoint: path to the trained weights (``runtime.yaml: checkpoints.stage_classifier``).
        device: torch device string.
        rate_hz: inference rate (5 Hz).
        every_n_ticks: run inference every ``control_hz / rate_hz`` ticks.
        model: torch module, None until ``load()``.
        last_prediction: most recent prediction, reused between inference ticks.
        enabled: False when no checkpoint is configured; ``predict`` then returns an uninformative
            prediction and the orchestrator relies on detectors only.
    """

    def __init__(
        self, checkpoint: str | None, device: str = "cuda:0", rate_hz: float = 5.0, control_hz: float = CONTROL_HZ
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.rate_hz = rate_hz
        self.every_n_ticks = max(1, int(round(control_hz / rate_hz)))
        self.model: Any = None
        self.last_prediction: StagePrediction | None = None
        self.enabled = checkpoint is not None

    def load(self) -> None:
        """Load the network weights onto ``device`` and switch to eval mode (lazy torch import).

        Raises:
            FileNotFoundError: checkpoint missing.
        """
        raise NotImplementedError("StageClassifier.load")

    def reset(self) -> None:
        """Episode start: drop ``last_prediction`` and any recurrent state."""
        raise NotImplementedError("StageClassifier.reset")

    def predict(self, tick: int, inputs: dict[str, Any]) -> StagePrediction:
        """Return the stage prediction for this tick.

        Functionality:
            If ``tick % every_n_ticks == 0`` (or no prediction yet) run the network on ``inputs``
            and cache the result; otherwise return the cached prediction. When ``enabled`` is
            False, return uniform stage/phase probabilities, progress 0, scene_ok 0.5 and
            lid_open 0.0 (stages 1 and 6 then rely on the kinematic lid estimate only).

        Args:
            tick: current control tick.
            inputs: output of ``ClassifierViewBuilder.build``.

        Returns:
            StagePrediction.
        """
        raise NotImplementedError("StageClassifier.predict")
