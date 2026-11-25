"""Safety filter: limits applied to every model-space action before it is decoded and sent.

Purpose
-------
Keep commands physically sensible regardless of what a policy outputs:

- EEF position clamped to a base-frame workspace box.
- Per-tick EEF step limits (translation and rotation) relative to the previous command, so a bad
  chunk or a chunk switch cannot make the arm jump.
- Base velocity clamp and deadband (tiny commands → 0, avoiding base creep).
- Gripper hysteresis: probability of "closed" must cross ``high`` to close and ``low`` to open,
  preventing chatter around the threshold.
- Hold action generation when the adapter reports too many invalid observations.

Inputs:  (23,) model action, previous model action, current CanonicalObs (EEF pose, gripper).
Outputs: (23,) clamped model action, (left, right) gripper decisions, ``clamped`` flag.

A non-finite action from a policy is replaced by the previous output (or the hold pose) and flagged.
The hold pose needs a base-frame EEF pose: FK, or without a URDF the last commanded one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.adapter.rotations import limit_rotation_step, matrix_to_rot6d, rot6d_to_matrix
from astra.types import (
    GRIPPER_CLOSED_RAD,
    MODEL_ACTION_DIM,
    MODEL_BASE_VEL,
    MODEL_GRIPPER,
    MODEL_LEFT_EEF,
    MODEL_RIGHT_EEF,
    CanonicalObs,
)

EEF_SLICES = (MODEL_LEFT_EEF, MODEL_RIGHT_EEF)


class SafetyFilter:
    """Stateful safety limits.

    Attributes:
        cfg: ``runtime.yaml: safety`` section.
        workspace_min / workspace_max: (3,) base-frame EEF bounds in metres.
        max_eef_step_m: max translation change of an EEF target per tick.
        max_eef_rot_step_rad: max rotation change per tick.
        max_base_lin_vel: limit on |vx| and |vy| in m/s.
        max_base_ang_vel: limit on |wz| in rad/s.
        base_vel_deadband: base velocity components below this magnitude → 0.
        gripper_low / gripper_high: hysteresis thresholds on p(closed).
        gripper_state: (left, right) current closed decisions.
        prev_action: (23,) previous output, None after reset (first tick limits against FK pose).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.workspace_min = np.asarray(cfg.get("workspace_min", [-0.2, -0.8, 0.3]), dtype=np.float32)
        self.workspace_max = np.asarray(cfg.get("workspace_max", [1.0, 0.8, 1.8]), dtype=np.float32)
        self.max_eef_step_m: float = float(cfg.get("max_eef_step_m", 0.02))
        self.max_eef_rot_step_rad: float = float(cfg.get("max_eef_rot_step_rad", 0.08))
        self.max_base_lin_vel: float = float(cfg.get("max_base_lin_vel", 1.0))
        self.max_base_ang_vel: float = float(cfg.get("max_base_ang_vel", 1.0))
        self.base_vel_deadband: float = float(cfg.get("base_vel_deadband", 0.02))
        low, high = cfg.get("gripper_hysteresis", [0.35, 0.65])
        self.gripper_low: float = float(low)
        self.gripper_high: float = float(high)
        self.gripper_state: list[bool] = [False, False]
        self.prev_action: np.ndarray | None = None
        self.clamp_count: int = 0

    def reset(self, obs: CanonicalObs) -> None:
        """Episode start: gripper_state from the measured gripper positions, prev_action from FK pose.

        Args:
            obs: first canonical observation of the episode.
        """
        closed_th = 0.5 * GRIPPER_CLOSED_RAD
        self.gripper_state = [bool(obs.gripper_raw[0] > closed_th), bool(obs.gripper_raw[1] > closed_th)]
        self.prev_action = self._pose_action(obs) if obs.eef_base_valid else None
        self.clamp_count = 0

    def apply(self, model_action: np.ndarray, obs: CanonicalObs) -> tuple[np.ndarray, tuple[bool, bool], bool]:
        """Clamp one model-space action.

        Functionality:
            1. EEF xyz → clip to workspace box.
            2. EEF xyz/rotation → limit change vs ``prev_action`` (translation norm, geodesic angle;
               rotation limited by slerp toward the target).
            3. Base velocity → deadband, clip to ±max_base_lin_vel / ±max_base_ang_vel.
            4. Grippers → hysteresis on p(closed) → closed decisions.
            5. Store as ``prev_action``.

        Args:
            model_action: (23,) model-space action from the chunk executor.
            obs: current canonical observation.

        Returns:
            (clamped (23,), (left_closed, right_closed), clamped_flag).
        """
        a = np.asarray(model_action, dtype=np.float64).reshape(MODEL_ACTION_DIM).copy()
        clamped = False
        if not np.all(np.isfinite(a)):
            a = self.hold_action(obs).astype(np.float64)
            clamped = True

        for sl in EEF_SLICES:
            # 1. workspace box
            pos = a[sl.start : sl.start + 3]
            boxed = np.clip(pos, self.workspace_min, self.workspace_max)
            if not np.allclose(boxed, pos):
                clamped = True
            rot = rot6d_to_matrix(a[sl.start + 3 : sl.stop])
            # 2. per-tick step limits against the previous command
            if self.prev_action is not None:
                prev_pos = self.prev_action[sl.start : sl.start + 3]
                d = boxed - prev_pos
                n = float(np.linalg.norm(d))
                if n > self.max_eef_step_m:
                    boxed = prev_pos + d * (self.max_eef_step_m / n)
                    clamped = True
                prev_rot = rot6d_to_matrix(self.prev_action[sl.start + 3 : sl.stop])
                rot, limited = limit_rotation_step(prev_rot, rot, self.max_eef_rot_step_rad)
                clamped = clamped or limited
            a[sl.start : sl.start + 3] = boxed
            a[sl.start + 3 : sl.stop] = matrix_to_rot6d(rot)  # also re-orthonormalises the network output

        # 3. base velocity: deadband, then limits
        base = a[MODEL_BASE_VEL].copy()
        base[np.abs(base) < self.base_vel_deadband] = 0.0
        limits = np.array([self.max_base_lin_vel, self.max_base_lin_vel, self.max_base_ang_vel])
        lim = np.clip(base, -limits, limits)
        if not np.allclose(lim, base):
            clamped = True
        a[MODEL_BASE_VEL] = lim

        # 4. gripper hysteresis on p(closed)
        for i in range(2):
            p = float(a[MODEL_GRIPPER.start + i])
            if self.gripper_state[i] and p < self.gripper_low:
                self.gripper_state[i] = False
            elif not self.gripper_state[i] and p > self.gripper_high:
                self.gripper_state[i] = True
            a[MODEL_GRIPPER.start + i] = 1.0 if self.gripper_state[i] else 0.0

        out = a.astype(np.float32)
        self.prev_action = out.astype(np.float64)
        if clamped:
            self.clamp_count += 1
        return out, (self.gripper_state[0], self.gripper_state[1]), clamped

    def hold_action(self, obs: CanonicalObs) -> np.ndarray:
        """Model-space hold: EEF targets = current FK poses, grippers per ``gripper_state``, base velocity 0.

        Args:
            obs: current canonical observation.

        Returns:
            (23,) model-space action.

        Raises:
            RuntimeError: no pose is known (no FK, nothing commanded yet, no previous output).
        """
        if obs.eef_base_valid and obs.eef_base_from_fk:
            a = self._pose_action(obs)
        elif self.prev_action is not None:
            a = self.prev_action.copy()
        elif obs.eef_base_valid:
            a = self._pose_action(obs)
        else:
            raise RuntimeError(
                "hold action: no base-frame EEF pose is known (set kinematics.urdf_path for forward kinematics)"
            )
        a[MODEL_GRIPPER] = [1.0 if c else 0.0 for c in self.gripper_state]
        a[MODEL_BASE_VEL] = 0.0
        return a.astype(np.float32)

    def _pose_action(self, obs: CanonicalObs) -> np.ndarray:
        """(23,) model action at the observed base-frame EEF poses, grippers per state, base 0."""
        a = np.zeros(MODEL_ACTION_DIM, dtype=np.float64)
        for i, sl in enumerate(EEF_SLICES):
            a[sl.start : sl.start + 3] = obs.eef_base_pos[i]
            a[sl.start + 3 : sl.stop] = matrix_to_rot6d(obs.eef_base_rotmat[i])
        a[MODEL_GRIPPER] = [1.0 if c else 0.0 for c in self.gripper_state]
        return a
