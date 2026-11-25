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
  contact; mirrors the sim rule (≥ 5 contacts of ≥ 0.5 s) with margins.
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

Calibration files (YAML, base frame, produced offline; all optional)
-------------------------------------------------------------------
- regions (``detectors.regions_calibration``)::

      regions:
        chamber:    {center: [x, y, z], radius: r}
        empty_box:  {center: [...], radius: ...}
        ...

  Without it every region reads "none" and the orchestrator skips its region tests.
- hinge (``detectors.hinge_calibration``)::

      hinge: {point: [x, y, z], direction: [dx, dy, dz]}

  Without it the slip check is off (the angle estimate itself needs no hinge).

The contact proxy needs a measured EEF pose, i.e. forward kinematics; without it
``contact_available`` is False and no wipe contacts are counted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from astra.adapter.rotations import rotation_angle_between
from astra.types import (
    GRIPPER_CLOSED_RAD,
    LID_RANGE_RAD,
    SIM_LEFT_EEF,
    SIM_LEFT_GRIPPER,
    SIM_RIGHT_EEF,
    CanonicalObs,
    EventState,
    SimAction,
)

log = logging.getLogger(__name__)

REGION_NAMES = ("chamber", "empty_box", "full_box", "table", "lid_handle")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


class EventDetectors:
    """Stateful detectors, reset every episode and when the wipe stage starts.

    Attributes:
        cfg: ``runtime.yaml: detectors`` section.
        holding_margin_rad: settled gripper position below ``GRIPPER_CLOSED_RAD - margin`` = holding.
        holding_min_rad: settled gripper position must also be above this (fingers actually closed
            on something, not stuck open).
        settle_ticks: ticks the gripper must be stationary before ``holding`` is evaluated.
        settle_eps_rad: per-tick change below which the gripper counts as stationary.
        contact_error_m: EEF tracking error that counts as contact.
        wipe_dwell_ticks: contact duration that qualifies one wipe contact.
        wipe_break_ticks: contact must be off this long before the next contact can count.
        regions: region name → (centre (3,), radius) in the base frame; loaded from a calibration
            file produced offline, empty until then.
        holding: current per-gripper holding state.
        settle_counter: per-gripper ticks since the gripper position stopped changing.
        contact_ticks: length of the current contact episode in ticks.
        break_ticks: ticks since contact was last seen.
        contact_qualified: True once the current contact episode has been counted.
        wipe_count: qualified contacts since ``start_wipe_count``.
        counting_wipes: whether contacts are being counted (only during the WIPE stage).
        prev_gripper: previous gripper positions (2,).
        hinge_axis_point / hinge_axis_dir: (3,) lid hinge axis in the base frame from calibration;
            None until loaded (slip check disabled then).
        lid_arm_enabled: True during LIFT_LID / CLOSE_LID; a handle grasp then starts lid tracking.
        lid_closing: True in CLOSE_LID (the estimate decreases with the hand rotation).
        lid_initial_angle: normalized lid angle assumed at the handle grasp (0.0 in stage 1; the
            last estimate in stage 6).
        lid_tracking_arm: arm holding the handle (0 left, 1 right), None when not tracking.
        lid_grasp_rotmat: (3, 3) EEF rotation at the handle grasp, the reference for the estimate.
        lid_grasp_pos: (3,) EEF position at the handle grasp (slip check).
        lid_angle_kin: current kinematic lid estimate (normalized), None when not tracking.
        lid_angle_kin_peak / lid_angle_kin_min: max / min estimate in the current stage.
        lid_arc_error_m: distance of the holding EEF from the hinge arc (slip indicator).
        lid_slip_threshold_m: arc error above which the kinematic estimate is flagged unreliable.
        active_arm: arm holding an object, else the one that held one last.
    """

    def __init__(self, cfg: dict[str, Any], control_hz: float) -> None:
        self.cfg = cfg
        self.control_hz = control_hz
        self.holding_margin_rad: float = float(cfg.get("holding_margin_rad", 0.08))
        self.holding_min_rad: float = float(cfg.get("holding_min_rad", 0.10))
        self.settle_ticks: int = int(round(float(cfg.get("gripper_settle_s", 0.2)) * control_hz))
        self.settle_eps_rad: float = float(cfg.get("gripper_settle_eps_rad", 0.005))
        self.contact_error_m: float = float(cfg.get("contact_error_m", 0.015))
        self.wipe_dwell_ticks: int = int(round(float(cfg.get("wipe_dwell_s", 0.75)) * control_hz))
        self.wipe_break_ticks: int = max(1, int(round(float(cfg.get("wipe_break_s", 0.1)) * control_hz)))
        self.closed_rad: float = GRIPPER_CLOSED_RAD
        self.regions: dict[str, tuple[np.ndarray, float]] = {}
        self.holding: list[bool] = [False, False]
        self.settle_counter: list[int] = [0, 0]
        self.empty_reported: list[bool] = [False, False]
        self.contact_ticks: int = 0
        self.break_ticks: int = 0
        self.contact_qualified: bool = False
        self.wipe_count: int = 0
        self.counting_wipes: bool = False
        self.prev_gripper: np.ndarray = np.zeros(2, dtype=np.float32)
        self.hinge_axis_point: np.ndarray | None = None
        self.hinge_axis_dir: np.ndarray | None = None
        self.lid_arm_enabled: bool = False
        self.lid_closing: bool = False
        self.lid_initial_angle: float = 0.0
        self.lid_tracking_arm: int | None = None
        self.lid_grasp_rotmat: np.ndarray | None = None
        self.lid_grasp_pos: np.ndarray | None = None
        self.lid_angle_kin: float | None = None
        self.lid_angle_kin_peak: float = 0.0
        self.lid_angle_kin_min: float = 1.0
        self.lid_arc_error_m: float = 0.0
        self.lid_slip_threshold_m: float = float(cfg.get("lid_slip_threshold_m", 0.03))
        self.active_arm: int | None = None
        self._first = True
        if cfg.get("regions_calibration"):
            self.load_regions(cfg["regions_calibration"])
        if cfg.get("hinge_calibration"):
            self.load_hinge_axis(cfg["hinge_calibration"])

    def reset(self) -> None:
        """Episode start: clear holding, counters, contact state, wipe count and lid tracking."""
        self.holding = [False, False]
        self.settle_counter = [0, 0]
        self.empty_reported = [False, False]
        self.contact_ticks = 0
        self.break_ticks = self.wipe_break_ticks
        self.contact_qualified = False
        self.wipe_count = 0
        self.counting_wipes = False
        self.prev_gripper = np.zeros(2, dtype=np.float32)
        self.active_arm = None
        self._first = True
        self.lid_arm_enabled = False
        self.lid_closing = False
        self.lid_initial_angle = 0.0
        self.lid_angle_kin_peak = 0.0
        self.lid_angle_kin_min = 1.0
        self._stop_lid_tracking()

    def load_regions(self, path: str) -> None:
        """Load workspace region centres/radii (base frame) from a calibration YAML.

        Args:
            path: file produced offline from demo grasp/release EEF positions.

        Raises:
            ValueError: malformed file or unknown region name.
        """
        data = _load_yaml(path)
        regions = data.get("regions", data)
        out: dict[str, tuple[np.ndarray, float]] = {}
        for name, spec in regions.items():
            if name not in REGION_NAMES:
                raise ValueError(f"{path}: unknown region {name!r}; known: {REGION_NAMES}")
            centre = np.asarray(spec["center"], dtype=np.float64).reshape(3)
            out[name] = (centre, float(spec["radius"]))
        self.regions = out

    def start_wipe_count(self) -> None:
        """Called by the orchestrator when the WIPE stage begins: wipe_count = 0, counting on."""
        self.wipe_count = 0
        self.contact_ticks = 0
        self.break_ticks = self.wipe_break_ticks
        self.contact_qualified = False
        self.counting_wipes = True

    def stop_wipe_count(self) -> None:
        """Called when the WIPE stage ends: counting off (count is kept for logging)."""
        self.counting_wipes = False

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
        data = _load_yaml(path)
        hinge = data.get("hinge", data)
        point = np.asarray(hinge["point"], dtype=np.float64).reshape(3)
        direction = np.asarray(hinge["direction"], dtype=np.float64).reshape(3)
        n = np.linalg.norm(direction)
        if n < 1e-9:
            raise ValueError(f"{path}: hinge direction is zero")
        self.hinge_axis_point = point
        self.hinge_axis_dir = direction / n

    def enable_lid_tracking(self, initial_angle: float, closing: bool = False) -> None:
        """Arm the lid estimator for LIFT_LID (initial 0.0) or CLOSE_LID (initial = vision estimate).

        Functionality:
            Set ``lid_arm_enabled`` and ``lid_initial_angle``; reset ``lid_angle_kin_peak``. The next
            grasp event in the ``lid_handle`` region (any grasp without region calibration) starts
            tracking.

        Args:
            initial_angle: normalized lid angle at the moment the handle will be grasped.
            closing: True in CLOSE_LID: the hand rotation closes the lid.
        """
        self.lid_arm_enabled = True
        self.lid_closing = bool(closing)
        self.lid_initial_angle = float(np.clip(initial_angle, 0.0, 1.0))
        self.lid_angle_kin_peak = self.lid_initial_angle if not closing else 0.0
        self.lid_angle_kin_min = self.lid_initial_angle if closing else 1.0
        self._stop_lid_tracking()

    def disable_lid_tracking(self) -> None:
        """Stop lid estimation (stage left); clear the tracking arm and reference rotation."""
        self.lid_arm_enabled = False
        self._stop_lid_tracking()

    def _stop_lid_tracking(self) -> None:
        self.lid_tracking_arm = None
        self.lid_grasp_rotmat = None
        self.lid_grasp_pos = None
        self.lid_angle_kin = None
        self.lid_arc_error_m = 0.0
        self._grasp_radius: float | None = None

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
            release: release edges of this tick (drops included).
        """
        if not self.lid_arm_enabled or not obs.eef_base_valid:
            return
        arm = self.lid_tracking_arm
        if arm is not None and release[arm]:
            self._stop_lid_tracking()
            return
        if arm is None:
            for i in (0, 1):
                if grasp[i] and (not self.regions or self._region_of(obs.eef_base_pos[i]) == "lid_handle"):
                    self.lid_tracking_arm = i
                    self.lid_grasp_rotmat = np.asarray(obs.eef_base_rotmat[i], dtype=np.float64).copy()
                    self.lid_grasp_pos = np.asarray(obs.eef_base_pos[i], dtype=np.float64).copy()
                    if self.hinge_axis_point is not None:
                        self._grasp_radius = self._axis_distance(self.lid_grasp_pos)
                    break
            arm = self.lid_tracking_arm
            if arm is None:
                return
        assert self.lid_grasp_rotmat is not None and self.lid_grasp_pos is not None
        delta = float(rotation_angle_between(self.lid_grasp_rotmat, obs.eef_base_rotmat[arm])) / LID_RANGE_RAD
        sign = -1.0 if self.lid_closing else 1.0
        self.lid_angle_kin = float(np.clip(self.lid_initial_angle + sign * delta, 0.0, 1.0))
        self.lid_angle_kin_peak = max(self.lid_angle_kin_peak, self.lid_angle_kin)
        self.lid_angle_kin_min = min(self.lid_angle_kin_min, self.lid_angle_kin)
        if self.hinge_axis_point is not None and self.hinge_axis_dir is not None and self._grasp_radius is not None:
            pos = np.asarray(obs.eef_base_pos[arm], dtype=np.float64)
            radial = self._axis_distance(pos) - self._grasp_radius
            axial = float(np.dot(pos - self.lid_grasp_pos, self.hinge_axis_dir))
            self.lid_arc_error_m = float(np.hypot(radial, axial))

    def _axis_distance(self, pos: np.ndarray) -> float:
        assert self.hinge_axis_point is not None and self.hinge_axis_dir is not None
        v = pos - self.hinge_axis_point
        return float(np.linalg.norm(v - np.dot(v, self.hinge_axis_dir) * self.hinge_axis_dir))

    # ------------------------------------------------------------------ helpers
    def _region_of(self, pos: np.ndarray) -> str:
        """Nearest calibrated region whose sphere contains ``pos``; "none" otherwise."""
        best, best_d = "none", np.inf
        for name, (centre, radius) in self.regions.items():
            d = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - centre))
            if d <= radius and d < best_d:
                best, best_d = name, d
        return best

    def update(self, obs: CanonicalObs, last_action: SimAction | None) -> EventState:
        """Compute all events for this tick.

        Functionality:
            1. Holding per gripper: command < 0, gripper settled for ``settle_ticks``, position
               < closed_rad - margin. Edges → grasp / release / drop; a settled fully closed
               gripper while commanded closed → empty_grasp (once per closing).
            2. Contact proxy: tracking error between ``last_action`` EEF target and FK EEF pose,
               only while the wiping arm is holding.
            3. Wipe counter: contact_ticks ≥ wipe_dwell_ticks and not yet qualified → count += 1;
               reset contact_ticks / qualified once contact has been off for ``wipe_break_ticks``.
            4. Region of the active EEF (nearest region centre within its radius).
            5. Kinematic lid angle (``_update_lid_estimate``) when lid tracking is enabled; fills
               ``lid_tracking``, ``lid_angle_kin``, ``lid_angle_kin_peak`` in the EventState.

        Args:
            obs: canonical observation of this tick.
            last_action: action sent on the previous tick (None on the first tick).

        Returns:
            EventState for this tick.
        """
        g = np.asarray(obs.gripper_raw, dtype=np.float64)
        if last_action is not None:
            cmd_vec = np.asarray(last_action.vector, dtype=np.float64)
        else:
            cmd_vec = np.asarray(obs.last_action, dtype=np.float64)
        cmd_closed = [bool(cmd_vec.shape[0] > SIM_LEFT_GRIPPER + i and cmd_vec[SIM_LEFT_GRIPPER + i] < 0) for i in (0, 1)]

        if self._first:
            self.prev_gripper = g.astype(np.float32)
            self._first = False
        grasp, release, drop, empty = [False, False], [False, False], [False, False], [False, False]
        for i in (0, 1):
            if abs(g[i] - float(self.prev_gripper[i])) < self.settle_eps_rad:
                self.settle_counter[i] += 1
            else:
                self.settle_counter[i] = 0
            settled = self.settle_counter[i] >= self.settle_ticks
            if not cmd_closed[i]:
                self.empty_reported[i] = False
            was = self.holding[i]
            if was:
                # keep holding while commanded closed and the fingers have not closed fully (no slip)
                now = cmd_closed[i] and g[i] < self.closed_rad - 0.5 * self.holding_margin_rad
            else:
                now = (
                    cmd_closed[i] and settled and self.holding_min_rad < g[i] < self.closed_rad - self.holding_margin_rad
                )
            if cmd_closed[i] and settled and not now and g[i] >= self.closed_rad - self.holding_margin_rad:
                if not self.empty_reported[i] and not was:
                    empty[i] = True
                self.empty_reported[i] = True
            if now and not was:
                grasp[i] = True
                self.active_arm = i
            elif was and not now:
                if cmd_closed[i]:
                    drop[i] = True
                else:
                    release[i] = True
            self.holding[i] = now
        self.prev_gripper = g.astype(np.float32)
        if self.holding[0] != self.holding[1]:
            self.active_arm = 0 if self.holding[0] else 1

        # contact proxy (needs a measured pose)
        contact_available = bool(obs.eef_base_from_fk and obs.eef_base_valid)
        in_contact = False
        arm = self.active_arm
        if contact_available and arm is not None and self.holding[arm] and cmd_vec.shape[0] >= 14:
            target = cmd_vec[(SIM_LEFT_EEF, SIM_RIGHT_EEF)[arm]][0:3]
            err = float(np.linalg.norm(target - np.asarray(obs.eef_base_pos[arm], dtype=np.float64)))
            in_contact = err > self.contact_error_m

        if self.counting_wipes:
            if in_contact:
                self.contact_ticks += 1
                self.break_ticks = 0
                if self.contact_ticks >= self.wipe_dwell_ticks and not self.contact_qualified:
                    self.wipe_count += 1
                    self.contact_qualified = True
                    log.info("tick %d: wipe contact %d", obs.tick, self.wipe_count)
            else:
                self.break_ticks += 1
                if self.break_ticks >= self.wipe_break_ticks:
                    self.contact_ticks = 0
                    self.contact_qualified = False

        if obs.eef_base_valid and self.regions:
            arm_regions = (self._region_of(obs.eef_base_pos[0]), self._region_of(obs.eef_base_pos[1]))
        else:
            arm_regions = ("none", "none")
        region = arm_regions[arm] if arm is not None else "none"

        lost = (release[0] or drop[0], release[1] or drop[1])
        self._update_lid_estimate(obs, (grasp[0], grasp[1]), lost)

        return EventState(
            holding=(self.holding[0], self.holding[1]),
            grasp=(grasp[0], grasp[1]),
            release=(release[0], release[1]),
            drop=(drop[0], drop[1]),
            empty_grasp=(empty[0], empty[1]),
            in_contact=in_contact,
            wipe_count=self.wipe_count,
            region=region,
            lid_tracking=self.lid_tracking_arm is not None,
            lid_angle_kin=self.lid_angle_kin,
            lid_angle_kin_peak=self.lid_angle_kin_peak,
            lid_angle_kin_min=self.lid_angle_kin_min,
            lid_slip=self.lid_tracking_arm is not None and self.lid_arc_error_m > self.lid_slip_threshold_m,
            contact_available=contact_available,
            arm_regions=arm_regions,
            active_arm=self.active_arm,
        )
