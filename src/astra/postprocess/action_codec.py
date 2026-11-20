"""Action codec: 23-D model action ↔ sim action (19-D, or the legacy 20-D form).

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
- base: ``velocity19`` copies ``[vx, vy, wz]``; the env then smooths it (alpha 0.1 per control
  step). ``wheels20`` maps it to four wheel values with ``WHEELS_FROM_BASE_VEL``; the env applies
  those as is. A model trained on demos recorded in the 20-D form learned the already smoothed
  command, so decode it as ``wheels20`` or the base is smoothed twice.

Encode (sim → model), offline
-----------------------------
Inverse mapping for building datasets: quaternion → rot6d (rows), ±1 → {0, 1}, recorded wheel
values → ``[vx, vy, wz]`` with ``BASE_VEL_FROM_WHEELS`` (19-D recordings: velocity copied).

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
        sim_format: "auto", "velocity19" or "wheels20" from the schema; ``select_format`` resolves
            "auto" once the env metadata is known.
        prev_quat: (2, 4) last quaternions sent for left/right; None after reset.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.quat_order: str = schema.action_quat_order
        self.sim_format: str = schema.sim_action_format
        self.prev_quat: np.ndarray | None = None

    def reset(self) -> None:
        """Episode start: forget previous quaternions."""
        raise NotImplementedError("ActionCodec.reset")

    def select_format(self, accepted_action_dims: list[int]) -> int:
        """Resolve ``sim_format`` against what the env accepts and return the sim action dimension.

        Functionality:
            "auto" → "velocity19" if 19 is accepted, else "wheels20". An explicit format the env
            does not accept is an error. Called at the start of ``AstraPolicy.eval``.

        Args:
            accepted_action_dims: ``usr_args["accepted_action_dims"]`` (or ``[actions_dim]``).

        Returns:
            19 or 20, the length of the vectors ``decode`` will produce.

        Raises:
            ValueError: the configured format is not accepted by the env.
        """
        raise NotImplementedError("ActionCodec.select_format")

    def decode(self, model_action: np.ndarray, gripper_closed: tuple[bool, bool]) -> np.ndarray:
        """23-D model action → sim action in the selected format.

        Args:
            model_action: (23,) one model-space action (already clamped by the safety filter).
            gripper_closed: (left, right) decisions after hysteresis.

        Returns:
            (19,) or (20,) float32 sim action, per ``sim_format``.
        """
        raise NotImplementedError("ActionCodec.decode")

    def encode(self, sim_action: np.ndarray) -> np.ndarray:
        """Recorded sim action (19-D or legacy 20-D) → 23-D model action (offline dataset building).

        Args:
            sim_action: (D,) or (T, D) recorded sim actions, D = 19 or 20.

        Returns:
            (23,) or (T, 23) model-space actions; gripper {0, 1} with 1 = closed.
        """
        raise NotImplementedError("ActionCodec.encode")
