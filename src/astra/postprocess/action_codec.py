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

from astra.adapter.rotations import (
    make_hemisphere_consistent,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
    rot6d_to_matrix,
)
from astra.schema import Schema
from astra.types import (
    BASE_VEL_FROM_WHEELS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    MODEL_ACTION_DIM,
    MODEL_BASE_VEL,
    MODEL_GRIPPER,
    MODEL_LEFT_EEF,
    MODEL_RIGHT_EEF,
    SIM_ACTION_DIM,
    SIM_ACTION_DIM_LEGACY,
    SIM_BASE_VEL,
    SIM_LEFT_EEF,
    SIM_LEFT_GRIPPER,
    SIM_RIGHT_EEF,
    SIM_RIGHT_GRIPPER,
    SIM_WHEELS_LEGACY,
    WHEELS_FROM_BASE_VEL,
)

FORMAT_DIMS = {"velocity19": SIM_ACTION_DIM, "wheels20": SIM_ACTION_DIM_LEGACY}


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
        self.prev_quat = None

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
        accepted = [int(d) for d in accepted_action_dims]
        fmt = self.schema.sim_action_format
        if fmt == "auto":
            if SIM_ACTION_DIM in accepted:
                fmt = "velocity19"
            elif SIM_ACTION_DIM_LEGACY in accepted:
                fmt = "wheels20"
            else:
                raise ValueError(f"env accepts action dims {accepted}; the codec produces 19 or 20")
        if fmt not in FORMAT_DIMS:
            raise ValueError(f"unknown sim_action_format {fmt!r}")
        if FORMAT_DIMS[fmt] not in accepted:
            raise ValueError(f"sim_action_format {fmt} needs {FORMAT_DIMS[fmt]} values, env accepts {accepted}")
        self.sim_format = fmt
        return FORMAT_DIMS[fmt]

    @property
    def sim_dim(self) -> int:
        """Length of decoded vectors (19 before ``select_format`` resolves "auto")."""
        return FORMAT_DIMS.get(self.sim_format, SIM_ACTION_DIM)

    def decode(self, model_action: np.ndarray, gripper_closed: tuple[bool, bool]) -> np.ndarray:
        """23-D model action → sim action in the selected format.

        Args:
            model_action: (23,) one model-space action (already clamped by the safety filter).
            gripper_closed: (left, right) decisions after hysteresis.

        Returns:
            (19,) or (20,) float32 sim action, per ``sim_format``.
        """
        a = np.asarray(model_action, dtype=np.float64).reshape(MODEL_ACTION_DIM)
        out = np.zeros(self.sim_dim, dtype=np.float64)
        quats = np.zeros((2, 4), dtype=np.float64)
        for i, (msl, ssl) in enumerate(((MODEL_LEFT_EEF, SIM_LEFT_EEF), (MODEL_RIGHT_EEF, SIM_RIGHT_EEF))):
            eef = a[msl]
            q = matrix_to_quat(rot6d_to_matrix(eef[3:9]), self.quat_order)
            ref = None if self.prev_quat is None else self.prev_quat[i]
            q = make_hemisphere_consistent(q, ref, self.quat_order)
            quats[i] = q
            out[ssl.start : ssl.start + 3] = eef[0:3]
            out[ssl.start + 3 : ssl.stop] = q
        self.prev_quat = quats
        out[SIM_LEFT_GRIPPER] = GRIPPER_CLOSE if gripper_closed[0] else GRIPPER_OPEN
        out[SIM_RIGHT_GRIPPER] = GRIPPER_CLOSE if gripper_closed[1] else GRIPPER_OPEN
        base = a[MODEL_BASE_VEL]
        if self.sim_format == "wheels20":
            out[SIM_WHEELS_LEGACY] = WHEELS_FROM_BASE_VEL.astype(np.float64) @ base
        else:
            out[SIM_BASE_VEL] = base
        return out.astype(np.float32)

    def encode(self, sim_action: np.ndarray) -> np.ndarray:
        """Recorded sim action (19-D or legacy 20-D) → 23-D model action (offline dataset building).

        Args:
            sim_action: (D,) or (T, D) recorded sim actions, D = 19 or 20.

        Returns:
            (23,) or (T, 23) model-space actions; gripper {0, 1} with 1 = closed.

        Raises:
            ValueError: D is neither 19 nor 20.
        """
        sa = np.asarray(sim_action, dtype=np.float64)
        single = sa.ndim == 1
        sa = sa.reshape(-1, sa.shape[-1])
        d = sa.shape[-1]
        if d not in (SIM_ACTION_DIM, SIM_ACTION_DIM_LEGACY):
            raise ValueError(f"recorded action has {d} values, expected 19 or 20")
        out = np.zeros((sa.shape[0], MODEL_ACTION_DIM), dtype=np.float64)
        for msl, ssl in ((MODEL_LEFT_EEF, SIM_LEFT_EEF), (MODEL_RIGHT_EEF, SIM_RIGHT_EEF)):
            pose = sa[:, ssl]
            out[:, msl.start : msl.start + 3] = pose[:, 0:3]
            out[:, msl.start + 3 : msl.stop] = matrix_to_rot6d(quat_to_matrix(pose[:, 3:7], self.quat_order))
        out[:, MODEL_GRIPPER.start] = (sa[:, SIM_LEFT_GRIPPER] < 0).astype(np.float64)
        out[:, MODEL_GRIPPER.start + 1] = (sa[:, SIM_RIGHT_GRIPPER] < 0).astype(np.float64)
        if d == SIM_ACTION_DIM_LEGACY:
            out[:, MODEL_BASE_VEL] = sa[:, SIM_WHEELS_LEGACY] @ BASE_VEL_FROM_WHEELS.astype(np.float64).T
        else:
            out[:, MODEL_BASE_VEL] = sa[:, SIM_BASE_VEL]
        out32 = out.astype(np.float32)
        return out32[0] if single else out32
