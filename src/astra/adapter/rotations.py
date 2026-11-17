"""Rotation conversions with one fixed convention for the whole stack.

Purpose
-------
The models predict rotations as rot6d, the sim's DiffIK command takes a quaternion, and FK produces
rotation matrices. Mixing conventions silently (row vs column rot6d, xyzw vs wxyz, q vs -q) is a
common source of large orientation errors, so all conversions live here.

Conventions
-----------
- rot6d = first two **rows** of the 3×3 rotation matrix, flattened ``[r00 r01 r02 r10 r11 r12]``.
  This matches GR00T N1.7 (``gr00t/data/state_action/pose.py`` ``_matrix_to_rot6d``); the dataset
  builder must use the same function.
- Quaternion order is a parameter ("xyzw" or "wxyz") taken from ``schema.action_quat_order``.
- Quaternions sent to the sim are unit length and hemisphere-consistent: sign chosen so the dot
  product with the previous command is ≥ 0 (no 360° flips between ticks).

Inputs / outputs: numpy arrays; leading batch dimensions are allowed on every function.
"""

from __future__ import annotations

import numpy as np


def matrix_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """Rotation matrix → rot6d.

    Args:
        rotmat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) first two rows flattened.
    """
    raise NotImplementedError("matrix_to_rot6d")


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d → rotation matrix via Gram-Schmidt.

    Functionality:
        Take the two 3-vectors, normalise the first, remove its component from the second and
        normalise, third row = cross product. Robust to the non-orthonormal rot6d a network outputs.

    Args:
        rot6d: (..., 6).

    Returns:
        (..., 3, 3) proper rotation matrices.
    """
    raise NotImplementedError("rot6d_to_matrix")


def matrix_to_quat(rotmat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Rotation matrix → unit quaternion in the requested component order.

    Args:
        rotmat: (..., 3, 3).
        order: "xyzw" or "wxyz".

    Returns:
        (..., 4) unit quaternions (sign not canonicalised; see ``make_hemisphere_consistent``).
    """
    raise NotImplementedError("matrix_to_quat")


def quat_to_matrix(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Unit quaternion → rotation matrix.

    Args:
        quat: (..., 4) in the given order; normalised internally.
        order: "xyzw" or "wxyz".

    Returns:
        (..., 3, 3).
    """
    raise NotImplementedError("quat_to_matrix")


def make_hemisphere_consistent(quat: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Flip quaternion sign so it lies in the same hemisphere as ``reference``.

    Functionality:
        If ``dot(quat, reference) < 0`` return ``-quat``. With ``reference=None`` use the
        convention "scalar part ≥ 0". Used by the action codec on every tick (reference = the
        previous command) and by the dataset builder (reference = previous frame).

    Args:
        quat: (..., 4).
        reference: (..., 4) or None.

    Returns:
        (..., 4) sign-adjusted quaternion.
    """
    raise NotImplementedError("make_hemisphere_consistent")


def rotation_angle_between(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    """Geodesic angle between two rotation matrices, in radians.

    Used by the safety filter (per-tick rotation step limit) and by tests.

    Args:
        rot_a: (..., 3, 3).
        rot_b: (..., 3, 3).

    Returns:
        (...,) angles in [0, π].
    """
    raise NotImplementedError("rotation_angle_between")
