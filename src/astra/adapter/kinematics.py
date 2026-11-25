"""Forward kinematics: arm joint angles → end-effector poses in the robot base frame.

Purpose
-------
The env's ``eef_pos`` / ``eef_quat`` observations are in the **world** frame (env origin), while the
DiffIK action command is in the robot **root (base)** frame, and the base drives around. The model
state does not need this module: it takes the arm joint angles, which carry the same information.
FK is for the parts that need a base-frame pose as numbers: the hold action, the contact proxy, the
lid estimate and the workspace limits. Because the torso (leg_joint*) and head are held at their default pose, the
base → arm_link7 chain depends only on the 7 arm joints, so FK gives an exact base-frame pose without
needing the base pose. The same FK runs offline on the HDF5 demos and online at runtime.

Inputs
------
- Robot description (URDF exported from ``galbot_one_golf.usda``), base link and tool link names,
  tool offset (currently zero, same as ``_GALBOT_TCP_OFFSET_*`` used by the DiffIK action).
- Per tick: ``joint_pos`` (77,), absolute joint angles as the adapter receives them.

Outputs
-------
- Left and right tool poses in the base frame: positions (2, 3) and rotation matrices (2, 3, 3).

Implementation
--------------
A small URDF reader (``xml.etree``) and a numpy chain evaluator: no extra dependency. Supported joint
types: fixed, revolute, continuous, prismatic. Movable chain joints that are not arm joints (the
legs) are looked up in ``joint_pos`` by name (``schema.joint_names``) or by an explicit index in
``kinematics.joint_ids``. With ``kinematics.urdf_path: null`` FK stays unavailable
(``available`` False); the adapter then falls back to the last commanded pose.

Validation (to do once in a dev run)
------------------------------------
Hold the arm still, compare FK output with the last commanded pose in ``obs["actions"][0:7]``.
Agreement within millimetres confirms the URDF, the tool offset and the quaternion order.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from astra.adapter.rotations import quat_to_matrix
from astra.schema import Schema

log = logging.getLogger(__name__)


@dataclass
class UrdfJoint:
    """One URDF joint: origin transform, motion axis and type."""

    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # (4, 4) parent link → joint frame at q = 0
    axis: np.ndarray  # (3,) unit axis in the joint frame


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) · Ry(pitch) · Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def parse_urdf(path: str | Path) -> dict[str, UrdfJoint]:
    """Parse the joints of a URDF file, keyed by child link name (each link has one parent joint)."""
    root = ET.parse(str(path)).getroot()
    joints: dict[str, UrdfJoint] = {}
    for el in root.findall("joint"):
        origin = np.eye(4)
        o = el.find("origin")
        if o is not None:
            origin[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            origin[:3, :3] = _rpy_to_matrix(np.array([float(v) for v in o.get("rpy", "0 0 0").split()]))
        a = el.find("axis")
        axis = np.array([float(v) for v in a.get("xyz", "1 0 0").split()]) if a is not None else np.array([1.0, 0, 0])
        n = np.linalg.norm(axis)
        axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
        parent, child = el.find("parent"), el.find("child")
        if parent is None or child is None:
            raise ValueError(f"{path}: joint {el.get('name')!r} lacks parent or child")
        j = UrdfJoint(
            name=str(el.get("name")),
            type=str(el.get("type", "fixed")),
            parent=str(parent.get("link")),
            child=str(child.get("link")),
            origin=origin,
            axis=axis,
        )
        joints[j.child] = j
    return joints


class KinematicChain:
    """Serial chain base link → tool link; ``joint_ids`` maps each movable joint to ``joint_pos``."""

    def __init__(self, joints: list[UrdfJoint], joint_ids: np.ndarray) -> None:
        self.joints = joints
        self.movable = [i for i, j in enumerate(joints) if j.type in ("revolute", "continuous", "prismatic")]
        self.joint_ids = np.asarray(joint_ids, dtype=np.int64)
        if len(self.joint_ids) != len(self.movable):
            raise ValueError("one joint_pos index per movable chain joint is required")

    def transform(self, joint_pos: np.ndarray) -> np.ndarray:
        """(4, 4) base → tool transform for the given absolute joint positions."""
        q = np.asarray(joint_pos, dtype=np.float64)[self.joint_ids]
        T = np.eye(4)
        k = 0
        for j in self.joints:
            T = T @ j.origin
            if j.type in ("revolute", "continuous"):
                M = np.eye(4)
                M[:3, :3] = _axis_angle_matrix(j.axis, float(q[k]))
                T = T @ M
                k += 1
            elif j.type == "prismatic":
                M = np.eye(4)
                M[:3, 3] = j.axis * float(q[k])
                T = T @ M
                k += 1
        return T




class ForwardKinematics:
    """Base-frame FK for both Galbot arms.

    Attributes:
        schema: resolved schema (joint indices, kinematics settings).
        urdf_path: path of the robot description, from ``schema.kinematics["urdf_path"]``.
        base_link: name of the base frame link (``base_link``).
        tool_links: (left, right) tool link names (``left_arm_link7``, ``right_arm_link7``).
        tool_offset_pos: (3,) offset from tool link to TCP.
        tool_offset_quat_xyzw: (4,) rotation offset from tool link to TCP.
        chains: per-arm kinematic chains, built in ``load()``; None until loaded (or when no URDF is
            configured).
        arm_joint_idx: (left, right) index arrays of the 7 arm joints in ``joint_pos``.
        extra_joint_ids: joint name → ``joint_pos`` index for chain joints that are not arm joints
            (``kinematics.joint_ids``), used when ``schema.joint_names`` is empty.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        kin = schema.kinematics
        self.urdf_path: str | None = kin.get("urdf_path")
        self.base_link: str = kin.get("base_link", "base_link")
        self.tool_links: tuple[str, str] = (
            kin.get("left_tool_link", "left_arm_link7"),
            kin.get("right_tool_link", "right_arm_link7"),
        )
        self.tool_offset_pos = np.asarray(kin.get("tool_offset_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.tool_offset_quat_xyzw = np.asarray(
            kin.get("tool_offset_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64
        )
        self.chains: tuple[Any, Any] | None = None
        self.arm_joint_idx: tuple[np.ndarray, np.ndarray] = (
            schema.state_joint_groups.get("left_arm_q", np.zeros(0, dtype=np.int64)),
            schema.state_joint_groups.get("right_arm_q", np.zeros(0, dtype=np.int64)),
        )
        self.extra_joint_ids: dict[str, int] = {str(k): int(v) for k, v in (kin.get("joint_ids") or {}).items()}
        self._tool_T = np.eye(4)
        self._tool_T[:3, :3] = quat_to_matrix(self.tool_offset_quat_xyzw, "xyzw")
        self._tool_T[:3, 3] = self.tool_offset_pos

    @property
    def available(self) -> bool:
        """True once both chains are built."""
        return self.chains is not None

    def load(self) -> None:
        """Build the two kinematic chains from the robot description.

        Functionality:
            Parse the URDF, build a serial chain from ``base_link`` to each tool link, and check
            that the chain's movable joints are exactly the 7 arm joints of that side (in schema
            order). Joints of the chain that are not arm joints (the legs) take their measured value
            from ``joint_pos`` on every call. Without a configured URDF this logs a warning and FK
            stays unavailable.

        Raises:
            FileNotFoundError: URDF missing.
            ValueError: chain joints do not match the schema arm joints.
        """
        if not self.urdf_path:
            log.warning("kinematics.urdf_path is not set: forward kinematics disabled, base-frame EEF poses "
                        "fall back to the last commanded pose")
            self.chains = None
            return
        path = Path(self.urdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")
        by_child = parse_urdf(path)
        chains = []
        for side, (tool, arm_idx) in enumerate(zip(self.tool_links, self.arm_joint_idx)):
            chain_joints: list[UrdfJoint] = []
            link = tool
            while link != self.base_link:
                if link not in by_child:
                    raise ValueError(f"{path}: no joint chain from {self.base_link!r} to {tool!r} (stuck at {link!r})")
                j = by_child[link]
                chain_joints.append(j)
                link = j.parent
            chain_joints.reverse()
            movable = [j for j in chain_joints if j.type in ("revolute", "continuous", "prismatic")]
            if len(arm_idx) != 7 or len(movable) < 7:
                raise ValueError(f"{path}: chain to {tool!r} has {len(movable)} movable joints, need the 7 arm joints")
            ids: list[int] = []
            arm_movable = movable[-7:]
            for j in movable[:-7]:
                if j.name in self.extra_joint_ids:
                    ids.append(self.extra_joint_ids[j.name])
                elif j.name in self.schema.joint_index:
                    ids.append(self.schema.joint_index[j.name])
                else:
                    raise ValueError(
                        f"{path}: chain joint {j.name!r} is not an arm joint and has no joint_pos index; fill "
                        "schema.yaml joint_names or kinematics.joint_ids"
                    )
            for k, j in enumerate(arm_movable):
                if j.name in self.schema.joint_index and self.schema.joint_index[j.name] != int(arm_idx[k]):
                    raise ValueError(
                        f"{path}: arm joint {j.name!r} is joint_pos[{self.schema.joint_index[j.name]}], schema "
                        f"state group says {int(arm_idx[k])}"
                    )
            ids.extend(int(i) for i in arm_idx)
            chains.append(KinematicChain(chain_joints, np.asarray(ids, dtype=np.int64)))
            log.info("FK chain %s: %s", "left" if side == 0 else "right", [j.name for j in movable])
        self.chains = (chains[0], chains[1])

    def compute(self, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame TCP poses of both arms.

        Functionality:
            Run each chain on its 7 arm joints (and the measured leg joints), apply the tool offset.

        Args:
            joint_pos: (77,) absolute joint positions.

        Returns:
            positions (2, 3) in metres and rotation matrices (2, 3, 3), order (left, right),
            both in the base frame.

        Raises:
            RuntimeError: FK not loaded.
        """
        if self.chains is None:
            raise RuntimeError("ForwardKinematics.compute called without a loaded URDF")
        pos = np.zeros((2, 3), dtype=np.float64)
        rot = np.zeros((2, 3, 3), dtype=np.float64)
        for i, chain in enumerate(self.chains):
            T = chain.transform(joint_pos) @ self._tool_T
            pos[i] = T[:3, 3]
            rot[i] = T[:3, :3]
        return pos, rot
