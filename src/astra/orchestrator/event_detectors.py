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
- **region**: which workspace region the active EEF is in (chamber, empty_box, full_box, table,
  lid_handle), from base-frame positions; region centres learned from EEF positions at demo
  grasp/release events.
- **kinematic lid angle**: the lid is a hinge (``Lid001_joint``, 0..90°) whose angle is not observed.
  While a hand rigidly holds the lid handle, the hand rotates with the lid, so
  ``lid_angle ≈ angle(R_eef(at handle grasp), R_eef(now)) / 90°`` (plus the lid angle at the grasp:
  0 in stage 1, ~open in stage 6). A circle-fit check against the hinge axis (calibrated from demos
  in the base frame) flags slip: if the hand leaves the arc, the estimate is marked unreliable.

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
        hinge_axis_point / hinge_axis_dir: (3,) lid hinge axis in the base frame from calibration;
            None until loaded (slip check disabled then).
        lid_arm_enabled: True during LIFT_LID / CLOSE_LID; a handle grasp then starts lid tracking.
        lid_initial_angle: normalized lid angle assumed at the handle grasp (0.0 in stage 1; the
            last vision estimate in stage 6).
        lid_tracking_arm: arm holding the handle (0 left, 1 right), None when not tracking.
        lid_grasp_rotmat: (3, 3) EEF rotation at the handle grasp, the reference for the estimate.
        lid_angle_kin: current kinematic lid estimate (normalized), None when not tracking.
        lid_angle_kin_peak: max estimate in the current stage (kept after release).
        lid_arc_error_m: distance of the holding EEF from the hinge arc (slip indicator).
        lid_slip_threshold_m: arc error above which the kinematic estimate is flagged unreliable.
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
        self.hinge_axis_point: np.ndarray | None = None
        self.hinge_axis_dir: np.ndarray | None = None
        self.lid_arm_enabled: bool = False
        self.lid_initial_angle: float = 0.0
        self.lid_tracking_arm: int | None = None
        self.lid_grasp_rotmat: np.ndarray | None = None
        self.lid_angle_kin: float | None = None
        self.lid_angle_kin_peak: float = 0.0
        self.lid_arc_error_m: float = 0.0
        self.lid_slip_threshold_m: float = float(cfg.get("lid_slip_threshold_m", 0.03))

    def reset(self) -> None:
        """Episode start: clear holding, counters, contact state, wipe count and lid tracking."""
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

    # ------------------------------------------------------------------ lid estimation
    def load_hinge_axis(self, path: str) -> None:
        """Load the lid hinge axis (point + direction, base frame) from a calibration YAML.

        Functionality:
            The calibration is produced offline by fitting a circle to the holding-EEF positions
            during the LIFT_LID segments of the demos (base does not move during stage 1). Used only
            for the slip check; the angle estimate itself needs no hinge calibration.

        Args:
            path: calibration file.
        """
        raise NotImplementedError("EventDetectors.load_hinge_axis")

    def enable_lid_tracking(self, initial_angle: float) -> None:
        """Arm the lid estimator for LIFT_LID (initial 0.0) or CLOSE_LID (initial = vision estimate).

        Functionality:
            Set ``lid_arm_enabled`` and ``lid_initial_angle``; reset ``lid_angle_kin_peak``. The next
            grasp event in the ``lid_handle`` region starts tracking.

        Args:
            initial_angle: normalized lid angle at the moment the handle will be grasped.
        """
        raise NotImplementedError("EventDetectors.enable_lid_tracking")

    def disable_lid_tracking(self) -> None:
        """Stop lid estimation (stage left); clear the tracking arm and reference rotation."""
        raise NotImplementedError("EventDetectors.disable_lid_tracking")

    def _update_lid_estimate(self, obs: CanonicalObs, grasp: tuple[bool, bool], release: tuple[bool, bool]) -> None:
        """Advance the kinematic lid estimate by one tick.

        Functionality:
            - On a grasp edge in the ``lid_handle`` region while ``lid_arm_enabled``: store the arm
              and its EEF rotation as ``lid_grasp_rotmat``.
            - While tracking: ``delta = rotation_angle_between(lid_grasp_rotmat, R_eef_now) / LID_RANGE_RAD``;
              stage 1 → ``lid_angle_kin = lid_initial_angle + delta``; stage 6 →
              ``lid_initial_angle - delta``; clip to [0, 1]; update the peak; compute the arc error
              against the hinge axis (if calibrated) for the slip flag.
            - On release or drop of the tracking arm: stop tracking (keep the peak).

        Args:
            obs: canonical observation (FK EEF rotations).
            grasp: grasp edges of this tick.
            release: release edges of this tick.
        """
        raise NotImplementedError("EventDetectors._update_lid_estimate")

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
            5. Kinematic lid angle (``_update_lid_estimate``) when lid tracking is enabled; fills
               ``lid_tracking``, ``lid_angle_kin``, ``lid_angle_kin_peak`` in the EventState.

        Args:
            obs: canonical observation of this tick.
            last_action: action sent on the previous tick (None on the first tick).

        Returns:
            EventState for this tick.
        """
        raise NotImplementedError("EventDetectors.update")
