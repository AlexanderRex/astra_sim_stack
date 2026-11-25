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

_EPS = 1e-12


def _check_order(order: str) -> None:
    if order not in ("xyzw", "wxyz"):
        raise ValueError(f"quaternion order must be 'xyzw' or 'wxyz', got {order!r}")


def matrix_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """Rotation matrix → rot6d.

    Args:
        rotmat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) first two rows flattened.
    """
    rotmat = np.asarray(rotmat)
    return rotmat[..., :2, :].reshape(*rotmat.shape[:-2], 6)


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
    r6 = np.asarray(rot6d, dtype=np.float64)
    a = r6[..., 0:3]
    b = r6[..., 3:6]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), _EPS)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), _EPS)
    c = np.cross(a, b)
    return np.stack([a, b, c], axis=-2)


def matrix_to_quat(rotmat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Rotation matrix → unit quaternion in the requested component order.

    Args:
        rotmat: (..., 3, 3).
        order: "xyzw" or "wxyz".

    Returns:
        (..., 4) unit quaternions (sign not canonicalised; see ``make_hemisphere_consistent``).
    """
    _check_order(order)
    m = np.asarray(rotmat, dtype=np.float64)
    batch = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    tr = m00 + m11 + m22
    wxyz = np.empty((m.shape[0], 4), dtype=np.float64)
    # Shepperd's method: pick the largest of w, x, y, z as the pivot for numerical stability.
    c0 = tr > 0
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    c2 = ~c0 & ~c1 & (m11 > m22)
    c3 = ~c0 & ~c1 & ~c2
    if c0.any():
        s = np.sqrt(tr[c0] + 1.0) * 2.0
        r = m[c0]
        wxyz[c0] = np.stack(
            [0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 1, 0] - r[:, 0, 1]) / s], -1
        )
    if c1.any():
        r = m[c1]
        s = np.sqrt(np.maximum(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], _EPS)) * 2.0
        wxyz[c1] = np.stack(
            [(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s], -1
        )
    if c2.any():
        r = m[c2]
        s = np.sqrt(np.maximum(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], _EPS)) * 2.0
        wxyz[c2] = np.stack(
            [(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s, (r[:, 1, 2] + r[:, 2, 1]) / s], -1
        )
    if c3.any():
        r = m[c3]
        s = np.sqrt(np.maximum(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], _EPS)) * 2.0
        wxyz[c3] = np.stack(
            [(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], -1
        )
    wxyz /= np.maximum(np.linalg.norm(wxyz, axis=-1, keepdims=True), _EPS)
    out = wxyz[:, [1, 2, 3, 0]] if order == "xyzw" else wxyz
    return out.reshape(*batch, 4)


def quat_to_matrix(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Unit quaternion → rotation matrix.

    Args:
        quat: (..., 4) in the given order; normalised internally.
        order: "xyzw" or "wxyz".

    Returns:
        (..., 3, 3).
    """
    _check_order(order)
    q = np.asarray(quat, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n < _EPS, 1.0, n)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def make_hemisphere_consistent(quat: np.ndarray, reference: np.ndarray | None, order: str = "xyzw") -> np.ndarray:
    """Flip quaternion sign so it lies in the same hemisphere as ``reference``.

    Functionality:
        If ``dot(quat, reference) < 0`` return ``-quat``. With ``reference=None`` use the
        convention "scalar part ≥ 0". Used by the action codec on every tick (reference = the
        previous command) and by the dataset builder (reference = previous frame).

    Args:
        quat: (..., 4).
        reference: (..., 4) or None.
        order: component order, needed only to find the scalar part when ``reference`` is None.

    Returns:
        (..., 4) sign-adjusted quaternion.
    """
    q = np.asarray(quat, dtype=np.float64)
    if reference is None:
        _check_order(order)
        scalar = q[..., 3] if order == "xyzw" else q[..., 0]
        sign = np.where(scalar < 0, -1.0, 1.0)
    else:
        dot = np.sum(q * np.asarray(reference, dtype=np.float64), axis=-1)
        sign = np.where(dot < 0, -1.0, 1.0)
    return q * sign[..., None]


def rotation_angle_between(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    """Geodesic angle between two rotation matrices, in radians.

    Used by the safety filter (per-tick rotation step limit) and by tests.

    Args:
        rot_a: (..., 3, 3).
        rot_b: (..., 3, 3).

    Returns:
        (...,) angles in [0, π].
    """
    rel = np.swapaxes(np.asarray(rot_a, dtype=np.float64), -1, -2) @ np.asarray(rot_b, dtype=np.float64)
    cos = (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cos, -1.0, 1.0))


def matrix_to_rotvec(rotmat: np.ndarray) -> np.ndarray:
    """Rotation matrix → rotation vector (axis × angle, angle in [0, π]); via the quaternion for stability."""
    q = matrix_to_quat(rotmat, "wxyz")
    q = make_hemisphere_consistent(q, None, "wxyz")  # w >= 0 → angle in [0, π]
    w = np.clip(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.where(sin_half < 1e-9, 2.0, angle / np.maximum(sin_half, 1e-12))
    return xyz * scale[..., None]


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector → rotation matrix (Rodrigues)."""
    v = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1)
    half = angle / 2.0
    scale = np.where(angle < 1e-9, 0.5, np.sin(half) / np.maximum(angle, 1e-12))
    wxyz = np.concatenate([np.cos(half)[..., None], v * scale[..., None]], axis=-1)
    return quat_to_matrix(wxyz, "wxyz")


def limit_rotation_step(rot_from: np.ndarray, rot_to: np.ndarray, max_angle: float) -> tuple[np.ndarray, bool]:
    """Rotate from ``rot_from`` toward ``rot_to`` by at most ``max_angle`` radians (geodesic slerp).

    Args:
        rot_from: (3, 3) previous rotation.
        rot_to: (3, 3) target rotation.
        max_angle: largest allowed step in radians.

    Returns:
        (rotation (3, 3), limited) where ``limited`` is True if the target was not reached.
    """
    rel = np.asarray(rot_from, dtype=np.float64).T @ np.asarray(rot_to, dtype=np.float64)
    rv = matrix_to_rotvec(rel)
    angle = float(np.linalg.norm(rv))
    if angle <= max_angle:
        return np.asarray(rot_to, dtype=np.float64), False
    return np.asarray(rot_from, dtype=np.float64) @ rotvec_to_matrix(rv * (max_angle / angle)), True


def pose_to_xyz_rot6d(pos: np.ndarray, quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Position (..., 3) + quaternion (..., 4) → (..., 9) xyz + rot6d (the state and action layout)."""
    return np.concatenate([np.asarray(pos, dtype=np.float64), matrix_to_rot6d(quat_to_matrix(quat, order))], axis=-1)
