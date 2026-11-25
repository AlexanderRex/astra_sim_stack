"""Scripted wiping primitive for the WIPE stage.

Purpose
-------
The sim credits wiping only for **separate** sponge-chamber contacts held ≥ 0.5 s, at least 5 of
them, counted after stage 3. Teleoperated wiping tends to drag continuously, which counts as one
contact. Once the learned policy has grasped the sponge and brought it into the chamber, this
primitive takes over and produces an exact press → dwell → lift → move pattern, then hands back.

Behaviour
---------
From the EEF pose at hand-over (sponge inside the chamber), visit ``n_contacts`` waypoints on the
floor / walls (offsets in the base frame, from a calibration file learned from demos). At each:
move to the waypoint, push the target ``press_depth_m`` beyond the surface (DiffIK tracking lags →
contact force), hold ``dwell_s``, lift ``lift_m`` to break contact, move on. Gripper stays closed
(holding the sponge), base velocity zero. Reports done after the last waypoint.

Waypoint file (``scripted_wipe.waypoints``, YAML, base frame, relative to the hand-over position)::

    waypoints:
      - [dx, dy, dz, nx, ny, nz]    # offset of the contact point, outward surface normal
      - ...

Without it a default pattern is used: ``n_contacts`` points on a grid of ``grid_spacing_m`` around
the hand-over position on a horizontal surface (normal +z), i.e. it assumes the hand-over happened
with the sponge on the chamber floor (the contact proxy fired).

Per waypoint the plan is: move (above the point, ``lift_m`` off the surface) → press (to
``press_depth_m`` inside the surface) → dwell (``dwell_s``) → lift. Moves run at ``move_speed_mps``,
press and lift take ``press_s``. The orientation stays at the hand-over orientation.

Inputs:  CanonicalObs (current EEF pose), runtime.yaml ``scripted_wipe`` section, waypoint file.
Outputs: ActionChunk (T, 23) at fps = control rate, source SCRIPT; ``done`` flag.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from astra.adapter.rotations import matrix_to_quat, matrix_to_rot6d
from astra.policies.base import PolicyBackend
from astra.types import (
    CONTROL_HZ,
    GRIPPER_CLOSED_RAD,
    MODEL_ACTION_DIM,
    MODEL_GRIPPER,
    MODEL_LEFT_EEF,
    MODEL_RIGHT_EEF,
    ActionChunk,
    CanonicalObs,
    OrchestratorOutput,
    PolicySource,
)

log = logging.getLogger(__name__)
EEF_SLICES = (MODEL_LEFT_EEF, MODEL_RIGHT_EEF)


def default_waypoints(n: int, spacing: float) -> np.ndarray:
    """(n, 6) grid offsets around the origin on a horizontal surface, normal +z; nearest points first."""
    side = int(np.ceil(np.sqrt(max(n, 1))))
    coords = (np.arange(side) - (side - 1) / 2.0) * spacing
    pts = np.array([[x, y, 0.0] for x in coords for y in coords])
    order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
    pts = pts[order][:n]
    normals = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    return np.concatenate([pts, normals], axis=1).astype(np.float32)


class ScriptedWipe(PolicyBackend):
    """Press-dwell-lift wiping primitive.

    Attributes:
        cfg: ``runtime.yaml: scripted_wipe`` section.
        press_depth_m / dwell_s / lift_m / n_contacts: pattern parameters.
        press_s: duration of the press and of the lift motion.
        move_speed_mps: speed of the moves between waypoints.
        chunk_ticks: rows per predicted chunk.
        waypoints: (N, 6) base-frame offsets of contact targets relative to the hand-over position,
            plus outward surface normals; from the calibration file, or the default grid.
        wiping_arm: 0 = left, 1 = right (arm holding the sponge, from the event detectors).
        anchor_pose: (7,) EEF pose at hand-over, xyz + quaternion xyzw; waypoints are relative to it.
        plan: (T, 3) planned positions of the wiping arm, one per control tick.
        contact_end_ticks: plan index at which each dwell ends (completed contacts are counted there).
        segment: index of the current waypoint.
        segment_tick: ticks spent in the current press/dwell/lift segment.
        ptr: next row of ``plan`` to emit.
        done: True when all waypoints are finished.
    """

    source = PolicySource.SCRIPT
    fps = CONTROL_HZ

    def __init__(self, cfg: dict[str, Any], control_hz: float = CONTROL_HZ) -> None:
        super().__init__()
        self.cfg = cfg
        self.control_hz = control_hz
        self.fps = control_hz
        self.press_depth_m: float = float(cfg.get("press_depth_m", 0.018))
        self.dwell_s: float = float(cfg.get("dwell_s", 0.8))
        self.lift_m: float = float(cfg.get("lift_m", 0.03))
        self.n_contacts: int = int(cfg.get("n_contacts", 5))
        self.press_s: float = float(cfg.get("press_s", 0.3))
        self.move_speed_mps: float = float(cfg.get("move_speed_mps", 0.10))
        self.grid_spacing_m: float = float(cfg.get("grid_spacing_m", 0.04))
        self.chunk_ticks: int = int(cfg.get("chunk_ticks", 10))
        self.waypoints_path: str | None = cfg.get("waypoints")
        self.waypoints: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        self.wiping_arm: int = 1
        self.anchor_pose: np.ndarray | None = None
        self._anchor_rot6d: np.ndarray | None = None
        self._other_pose: np.ndarray | None = None
        self._other_gripper: float = 0.0
        self.plan: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.contact_end_ticks: list[int] = []
        self.segment: int = 0
        self.segment_tick: int = 0
        self.ptr: int = 0
        self.done: bool = False
        self.started: bool = False

    def load(self) -> None:
        """Load the waypoint calibration file (offsets + surface normals), else build the default grid.

        Raises:
            FileNotFoundError: configured file missing.
            ValueError: rows are not 6 values.
        """
        if self.waypoints_path:
            import yaml

            path = Path(self.waypoints_path)
            if not path.is_file():
                raise FileNotFoundError(f"scripted wipe waypoints not found: {path}")
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            wp = np.asarray(data.get("waypoints", data) if isinstance(data, dict) else data, dtype=np.float32)
            if wp.ndim != 2 or wp.shape[1] != 6 or len(wp) == 0:
                raise ValueError(f"{path}: waypoints must be rows of [dx, dy, dz, nx, ny, nz]")
            norms = np.linalg.norm(wp[:, 3:6], axis=1, keepdims=True)
            wp[:, 3:6] = wp[:, 3:6] / np.maximum(norms, 1e-9)
            self.waypoints = wp[: self.n_contacts] if len(wp) >= self.n_contacts else wp
            if len(wp) < self.n_contacts:
                log.warning("scripted wipe: %d waypoints for %d contacts", len(wp), self.n_contacts)
        else:
            self.waypoints = default_waypoints(self.n_contacts, self.grid_spacing_m)
        self.loaded = True

    def reset(self) -> None:
        """Clear progress: segment 0, done False, anchor None."""
        self.anchor_pose = None
        self._anchor_rot6d = None
        self._other_pose = None
        self.plan = np.zeros((0, 3), dtype=np.float64)
        self.contact_end_ticks = []
        self.segment = 0
        self.segment_tick = 0
        self.ptr = 0
        self.done = False
        self.started = False

    def start(self, obs: CanonicalObs, wiping_arm: int) -> None:
        """Begin the primitive: store the anchor pose of ``wiping_arm`` and reset progress.

        Args:
            obs: observation at hand-over.
            wiping_arm: 0 left / 1 right.
        """
        self.reset()
        if not self.loaded:
            self.load()
        self.wiping_arm = int(wiping_arm)
        self.started = True
        if not obs.eef_base_valid:
            log.warning("scripted wipe: no base-frame EEF pose at hand-over, skipping the primitive")
            self.done = True
            return
        a, o = self.wiping_arm, 1 - self.wiping_arm
        pos = np.asarray(obs.eef_base_pos[a], dtype=np.float64)
        rot = np.asarray(obs.eef_base_rotmat[a], dtype=np.float64)
        self.anchor_pose = np.concatenate([pos, matrix_to_quat(rot, "xyzw")]).astype(np.float32)
        self._anchor_rot6d = matrix_to_rot6d(rot)
        self._other_pose = np.concatenate(
            [np.asarray(obs.eef_base_pos[o], dtype=np.float64), matrix_to_rot6d(obs.eef_base_rotmat[o])]
        )
        self._other_gripper = 1.0 if float(obs.gripper_raw[o]) > 0.5 * GRIPPER_CLOSED_RAD else 0.0
        self._build_plan(pos)
        log.info("scripted wipe: %d contacts with the %s arm, %d ticks", len(self.waypoints),
                 "left" if a == 0 else "right", len(self.plan))

    def _build_plan(self, anchor: np.ndarray) -> None:
        hz = self.control_hz
        press_ticks = max(1, int(round(self.press_s * hz)))
        dwell_ticks = max(1, int(round(self.dwell_s * hz)))
        rows: list[np.ndarray] = []
        self.contact_end_ticks = []
        current = anchor.copy()
        for wp in self.waypoints:
            point = anchor + wp[0:3].astype(np.float64)
            normal = wp[3:6].astype(np.float64)
            above = point + normal * self.lift_m
            pressed = point - normal * self.press_depth_m
            dist = float(np.linalg.norm(above - current))
            move_ticks = max(1, int(np.ceil(dist / max(self.move_speed_mps, 1e-6) * hz)))
            for seg_from, seg_to, n in ((current, above, move_ticks), (above, pressed, press_ticks)):
                for k in range(1, n + 1):
                    rows.append(seg_from + (seg_to - seg_from) * (k / n))
            rows.extend([pressed.copy()] * dwell_ticks)
            self.contact_end_ticks.append(len(rows))
            for k in range(1, press_ticks + 1):
                rows.append(pressed + (above - pressed) * (k / press_ticks))
            current = above
        self.plan = np.asarray(rows, dtype=np.float64).reshape(-1, 3)

    @property
    def contacts_done(self) -> int:
        """Dwell phases completed so far."""
        return int(sum(1 for t in self.contact_end_ticks if self.ptr >= t))

    def _row(self, pos: np.ndarray) -> np.ndarray:
        assert self._anchor_rot6d is not None and self._other_pose is not None
        a = np.zeros(MODEL_ACTION_DIM, dtype=np.float64)
        w_sl, o_sl = EEF_SLICES[self.wiping_arm], EEF_SLICES[1 - self.wiping_arm]
        a[w_sl] = np.concatenate([pos, self._anchor_rot6d])
        a[o_sl] = self._other_pose
        a[MODEL_GRIPPER.start + self.wiping_arm] = 1.0
        a[MODEL_GRIPPER.start + 1 - self.wiping_arm] = self._other_gripper
        return a  # base velocity stays 0

    def predict(self, obs: CanonicalObs, view: Any, orch: OrchestratorOutput) -> ActionChunk:
        """Produce the next short chunk of the press/dwell/lift pattern.

        Functionality:
            Generate targets for the next ~10 ticks along the current segment; the non-wiping arm
            holds its pose from the hand-over; gripper of the wiping arm = closed (1.0), other
            unchanged; base velocity 0. Advance segment counters; set ``done`` after the last
            waypoint (later calls repeat the final pose).

        Args:
            obs: canonical observation (current EEF pose for the hold arm).
            view: unused.
            orch: orchestrator output.

        Returns:
            ActionChunk (T, 23), source SCRIPT, fps = control rate.

        Raises:
            RuntimeError: called before ``start`` or without a pose at hand-over.
        """
        if not self.started:
            self.start(obs, orch.active_arm if orch.active_arm is not None else self.wiping_arm)
        if self._anchor_rot6d is None or len(self.plan) == 0:
            raise RuntimeError("scripted wipe has no plan (no pose at hand-over)")
        end = min(self.ptr + self.chunk_ticks, len(self.plan))
        if self.ptr >= len(self.plan):
            rows = [self._row(self.plan[-1])]
        else:
            rows = [self._row(self.plan[i]) for i in range(self.ptr, end)]
            self.ptr = end
        self.segment = self.contacts_done
        if self.ptr >= len(self.plan):
            self.done = True
        return ActionChunk(
            actions=np.asarray(rows, dtype=np.float32),
            source=PolicySource.SCRIPT,
            fps=self.fps,
            start_tick=obs.tick,
            stage=orch.stage,
        )
