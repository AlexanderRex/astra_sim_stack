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
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.types import CanonicalObs


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

    def reset(self, obs: CanonicalObs) -> None:
        """Episode start: gripper_state from the measured gripper positions, prev_action from FK pose.

        Args:
            obs: first canonical observation of the episode.
        """
        raise NotImplementedError("SafetyFilter.reset")

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
        raise NotImplementedError("SafetyFilter.apply")

    def hold_action(self, obs: CanonicalObs) -> np.ndarray:
        """Model-space hold: EEF targets = current FK poses, grippers per ``gripper_state``, base velocity 0.

        Args:
            obs: current canonical observation.

        Returns:
            (23,) model-space action.
        """
        raise NotImplementedError("SafetyFilter.hold_action")
