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

Inputs:  classifier tensors from ``ClassifierViewBuilder.build`` (history window), or a callable
         returning them (built only on inference ticks).
Outputs: ``StagePrediction``. Runs at 5 Hz; between runs the last prediction is reused.

Checkpoint contract
-------------------
A TorchScript module (``torch.jit.save``) whose ``forward`` takes ``Dict[str, Tensor]`` with the
builder's keys plus a batch dimension (``images.<camera>`` (1, T, 3, H, W), ``state`` (1, T, 34))
and returns ``Dict[str, Tensor]``:

    stage_logits (1, 7) · phase_logits (1, 5) · progress (1,) · scene_ok (1,) · lid_open (1,)

``progress``, ``scene_ok`` and ``lid_open`` are in [0, 1] (apply the sigmoid inside the module).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from astra.types import CONTROL_HZ, StagePrediction

log = logging.getLogger(__name__)
N_STAGE_CLASSES = 7
N_PHASES = 5


def uninformative_prediction() -> StagePrediction:
    """Uniform stage/phase probabilities, progress 0, scene_ok 0.5, lid_open 0."""
    return StagePrediction(
        stage_probs=np.full(N_STAGE_CLASSES, 1.0 / N_STAGE_CLASSES, dtype=np.float32),
        phase_probs=np.full(N_PHASES, 1.0 / N_PHASES, dtype=np.float32),
        progress=0.0,
        scene_ok=0.5,
        lid_open=0.0,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


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
        if not self.enabled:
            log.info("stage classifier: no checkpoint, the orchestrator uses the detectors only")
            return
        path = Path(str(self.checkpoint))
        if not path.is_file():
            raise FileNotFoundError(f"stage classifier checkpoint not found: {path}")
        import torch

        self.model = torch.jit.load(str(path), map_location=self.device)
        self.model.eval()

    def reset(self) -> None:
        """Episode start: drop ``last_prediction`` and any recurrent state."""
        self.last_prediction = None

    def predict(self, tick: int, inputs: dict[str, Any] | Callable[[], dict[str, Any]]) -> StagePrediction:
        """Return the stage prediction for this tick.

        Functionality:
            If ``tick % every_n_ticks == 0`` (or no prediction yet) run the network on ``inputs``
            and cache the result; otherwise return the cached prediction. When ``enabled`` is
            False, return uniform stage/phase probabilities, progress 0, scene_ok 0.5 and
            lid_open 0.0 (stages 1 and 6 then rely on the kinematic lid estimate only).

        Args:
            tick: current control tick.
            inputs: output of ``ClassifierViewBuilder.build``, or a callable returning it.

        Returns:
            StagePrediction.
        """
        if not self.enabled or self.model is None:
            if self.last_prediction is None:
                self.last_prediction = uninformative_prediction()
            return self.last_prediction
        if self.last_prediction is not None and tick % self.every_n_ticks != 0:
            return self.last_prediction
        data = inputs() if callable(inputs) else inputs
        if not data:
            return self.last_prediction or uninformative_prediction()
        import torch

        batch = {k: torch.from_numpy(np.ascontiguousarray(v))[None].to(self.device) for k, v in data.items()}
        with torch.inference_mode():
            out = self.model(batch)
        def get(key: str) -> np.ndarray:
            return out[key].detach().float().cpu().numpy().reshape(-1)

        self.last_prediction = StagePrediction(
            stage_probs=_softmax(get("stage_logits")),
            phase_probs=_softmax(get("phase_logits")),
            progress=float(np.clip(get("progress")[0], 0.0, 1.0)),
            scene_ok=float(np.clip(get("scene_ok")[0], 0.0, 1.0)),
            lid_open=float(np.clip(get("lid_open")[0], 0.0, 1.0)),
        )
        return self.last_prediction
