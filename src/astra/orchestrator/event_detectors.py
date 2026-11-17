"""Rule-based event detectors using only the robot's own sensing and commands.

Purpose
-------
Cheap, robust evidence for the orchestrator that needs no learning:

- **holding** (per gripper): commanded closed (sim value < 0) but the measured gripper joint settles
  short of fully closed (1.703 rad - margin) → an object is between the fingers. Fully closed means
  an empty grasp.
- **grasp / release / drop** edges of ``holding``.
- **contact proxy**: with DiffIK pose control, pressing on a surface makes the measured EEF pose lag
  the commanded target; ‖target - actual‖ > threshold while holding the sponge ≈ contact.
- **wipe counter**: contact episodes with dwell ≥ wipe_dwell_s, each followed by a break in
  contact; mirrors the sim rule (≥ 3 contacts of ≥ 0.5 s) with margins.
- **region**: which workspace region the active EEF is in (chamber, empty_box, full_box, table),
  from base-frame positions; region centres learned from EEF positions at demo grasp/release events.

Inputs (per tick)
-----------------
- ``CanonicalObs`` (gripper_raw, base-frame EEF pose, tick/time).
- The sim action actually sent on the previous tick (gripper command, EEF target) from layer 5.

Outputs
-------
- ``EventState`` for the current tick.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.types import GRIPPER_CLOSED_RAD, CanonicalObs, EventState, SimAction


class EventDetectors:
    """Stateful detectors, reset every episode and when the wipe stage starts.

    Attributes:
        cfg: ``runtime.yaml: detectors`` section.
        holding_margin_rad: settled gripper position below ``GRIPPER_CLOSED_RAD - margin`` = holding.
        settle_ticks: ticks the gripper must be stationary before ``holding`` is evaluated.
        contact_error_m: EEF tracking error that counts as contact.
        wipe_dwell_ticks: contact duration that qualifies one wipe contact.
        regions: region name → (centre (3,), radius) in the base frame; loaded from a calibration
            file produced offline, empty until then.
        holding: current per-gripper holding state.
        settle_counter: per-gripper ticks since the gripper position stopped changing.
        contact_ticks: length of the current contact episode in ticks.
        contact_qualified: True once the current contact episode has been counted.
        wipe_count: qualified contacts since ``start_wipe_count``.
        counting_wipes: whether contacts are being counted (only during the WIPE stage).
        prev_gripper: previous gripper positions (2,).
    """

    def __init__(self, cfg: dict[str, Any], control_hz: float) -> None:
        self.cfg = cfg
        self.control_hz = control_hz
        self.holding_margin_rad: float = float(cfg.get("holding_margin_rad", 0.08))
        self.settle_ticks: int = int(round(float(cfg.get("gripper_settle_s", 0.2)) * control_hz))
        self.contact_error_m: float = float(cfg.get("contact_error_m", 0.015))
        self.wipe_dwell_ticks: int = int(round(float(cfg.get("wipe_dwell_s", 0.75)) * control_hz))
        self.closed_rad: float = GRIPPER_CLOSED_RAD
        self.regions: dict[str, tuple[np.ndarray, float]] = {}
        self.holding: list[bool] = [False, False]
        self.settle_counter: list[int] = [0, 0]
        self.contact_ticks: int = 0
        self.contact_qualified: bool = False
        self.wipe_count: int = 0
        self.counting_wipes: bool = False
        self.prev_gripper: np.ndarray = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        """Episode start: clear holding, counters, contact state and wipe count."""
        raise NotImplementedError("EventDetectors.reset")

    def load_regions(self, path: str) -> None:
        """Load workspace region centres/radii (base frame) from a calibration YAML.

        Args:
            path: file produced offline from demo grasp/release EEF positions.
        """
        raise NotImplementedError("EventDetectors.load_regions")

    def start_wipe_count(self) -> None:
        """Called by the orchestrator when the WIPE stage begins: wipe_count = 0, counting on."""
        raise NotImplementedError("EventDetectors.start_wipe_count")

    def stop_wipe_count(self) -> None:
        """Called when the WIPE stage ends: counting off (count is kept for logging)."""
        raise NotImplementedError("EventDetectors.stop_wipe_count")

    def update(self, obs: CanonicalObs, last_action: SimAction | None) -> EventState:
        """Compute all events for this tick.

        Functionality:
            1. Holding per gripper: command < 0, gripper settled for ``settle_ticks``, position
               < closed_rad - margin. Edges → grasp / release / drop.
            2. Contact proxy: tracking error between ``last_action`` EEF target and FK EEF pose,
               only while the wiping arm is holding.
            3. Wipe counter: contact_ticks ≥ wipe_dwell_ticks and not yet qualified → count += 1;
               reset contact_ticks / qualified when contact breaks.
            4. Region of the active EEF (nearest region centre within its radius).

        Args:
            obs: canonical observation of this tick.
            last_action: action sent on the previous tick (None on the first tick).

        Returns:
            EventState for this tick.
        """
        raise NotImplementedError("EventDetectors.update")
