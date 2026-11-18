"""Action codec: 24-D model action ↔ 20-D sim action.

Purpose
-------
The only place that knows how the model action space maps onto the sim's action terms. Used online
(decode, every tick) and by the offline dataset builder (encode, once per frame), so both directions
share one implementation.

Decode (model → sim), per tick
------------------------------
- left/right EEF: xyz copied; rot6d → rotation matrix (Gram-Schmidt) → unit quaternion in
  ``schema.action_quat_order``, sign made consistent with the previous command (no flips).
- grippers: probability of "closed" in [0, 1] → sim value ``-1`` (close) / ``+1`` (open).
  Isaac Lab BinaryJointPositionAction closes when the value is **< 0**; never send 0 for "close".
  The threshold itself (with hysteresis) is applied by the safety filter before decoding.
- wheels: copied (the env applies its own 0.01 scale).

Encode (sim → model), offline
-----------------------------
Inverse mapping for building datasets: quaternion → rot6d (rows), ±1 → {0, 1}.

Inputs / outputs: numpy arrays in the layouts of ``astra.types``.
"""

from __future__ import annotations

import numpy as np

from astra.schema import Schema


class ActionCodec:
    """Stateless mapping except for the previous quaternions used for sign consistency.

    Attributes:
        schema: resolved schema (quaternion order).
        quat_order: "xyzw" or "wxyz".
        prev_quat: (2, 4) last quaternions sent for left/right; None after reset.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.quat_order: str = schema.action_quat_order
        self.prev_quat: np.ndarray | None = None

    def reset(self) -> None:
        """Episode start: forget previous quaternions."""
        raise NotImplementedError("ActionCodec.reset")

    def decode(self, model_action: np.ndarray, gripper_closed: tuple[bool, bool]) -> np.ndarray:
        """24-D model action → 20-D sim action.

        Args:
            model_action: (24,) one model-space action (already clamped by the safety filter).
            gripper_closed: (left, right) decisions after hysteresis.

        Returns:
            (20,) float32 sim action.
        """
        raise NotImplementedError("ActionCodec.decode")

    def encode(self, sim_action: np.ndarray) -> np.ndarray:
        """20-D sim action → 24-D model action (offline dataset building).

        Args:
            sim_action: (20,) or (T, 20) recorded sim actions.

        Returns:
            (24,) or (T, 24) model-space actions; gripper {0, 1} with 1 = closed.
        """
        raise NotImplementedError("ActionCodec.encode")
