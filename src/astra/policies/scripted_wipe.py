"""Scripted wiping primitive for the WIPE stage.

Purpose
-------
The sim credits wiping only for **separate** sponge-chamber contacts held ≥ 0.5 s, at least 3 of
them, counted after stage 3. Teleoperated wiping tends to drag continuously, which counts as one
contact. Once the learned policy has grasped the sponge and brought it into the chamber, this
primitive takes over and produces an exact press → dwell → lift → move pattern, then hands back.

Behaviour
---------
From the EEF pose at hand-over (sponge inside the chamber), visit ``n_contacts`` waypoints on the
floor / walls (offsets in the base frame, from a calibration file learned from demos). At each:
move to the waypoint, push the target ``press_depth_m`` beyond the surface (DiffIK tracking lags →
contact force), hold ``dwell_s``, lift ``lift_m`` to break contact, move on. Gripper stays closed
(holding the sponge), wheels zero. Reports done after the last waypoint.

Inputs:  CanonicalObs (current EEF pose), runtime.yaml ``scripted_wipe`` section, waypoint file.
Outputs: ActionChunk (T, 24) at fps = control rate, source SCRIPT; ``done`` flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.policies.base import PolicyBackend
from astra.types import CONTROL_HZ, ActionChunk, CanonicalObs, OrchestratorOutput, PolicySource


class ScriptedWipe(PolicyBackend):
    """Press-dwell-lift wiping primitive.

    Attributes:
        cfg: ``runtime.yaml: scripted_wipe`` section.
        press_depth_m / dwell_s / lift_m / n_contacts: pattern parameters.
        waypoints: (N, 3) base-frame offsets of contact targets relative to the hand-over pose,
            plus surface normals (N, 3); loaded from calibration, empty until then.
        wiping_arm: 0 = left, 1 = right (arm holding the sponge, from the event detectors).
        anchor_pose: (7,) EEF pose at hand-over; waypoints are relative to it.
        segment: index of the current waypoint.
        segment_tick: ticks spent in the current press/dwell/lift segment.
        done: True when all waypoints are finished.
    """

    source = PolicySource.SCRIPT
    fps = CONTROL_HZ

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.press_depth_m: float = float(cfg.get("press_depth_m", 0.018))
        self.dwell_s: float = float(cfg.get("dwell_s", 0.8))
        self.lift_m: float = float(cfg.get("lift_m", 0.03))
        self.n_contacts: int = int(cfg.get("n_contacts", 5))
        self.waypoints: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        self.wiping_arm: int = 1
        self.anchor_pose: np.ndarray | None = None
        self.segment: int = 0
        self.segment_tick: int = 0
        self.done: bool = False

    def load(self) -> None:
        """Load the waypoint calibration file (offsets + surface normals)."""
        raise NotImplementedError("ScriptedWipe.load")

    def reset(self) -> None:
        """Clear progress: segment 0, done False, anchor None."""
        raise NotImplementedError("ScriptedWipe.reset")

    def start(self, obs: CanonicalObs, wiping_arm: int) -> None:
        """Begin the primitive: store the anchor pose of ``wiping_arm`` and reset progress.

        Args:
            obs: observation at hand-over.
            wiping_arm: 0 left / 1 right.
        """
        raise NotImplementedError("ScriptedWipe.start")

    def predict(self, obs: CanonicalObs, view: Any, orch: OrchestratorOutput) -> ActionChunk:
        """Produce the next short chunk of the press/dwell/lift pattern.

        Functionality:
            Generate targets for the next ~10 ticks along the current segment; the non-wiping arm
            holds its current pose; gripper of the wiping arm = closed (1.0), other unchanged;
            wheels 0. Advance segment counters; set ``done`` after the last waypoint.

        Args:
            obs: canonical observation (current EEF pose for the hold arm).
            view: unused.
            orch: orchestrator output.

        Returns:
            ActionChunk (T, 24), source SCRIPT, fps = control rate.
        """
        raise NotImplementedError("ScriptedWipe.predict")
