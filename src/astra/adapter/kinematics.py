"""Forward kinematics: arm joint angles → end-effector poses in the robot base frame.

Purpose
-------
The env's ``eef_pos`` / ``eef_quat`` observations are in the **world** frame (env origin), while the
DiffIK action command is in the robot **root (base)** frame, and the base drives around. Using the
world-frame pose as state would make it change whenever the base moves, and would break GR00T's
relative EEF actions. Because the torso (leg_joint*) and head are held at their default pose, the
base → arm_link7 chain depends only on the 7 arm joints, so FK gives an exact base-frame pose without
needing the base pose. The same FK runs offline on the HDF5 demos and online at runtime.

Inputs
------
- Robot description (URDF exported from ``galbot_one_golf.usda``), base link and tool link names,
  tool offset (currently zero, same as ``_GALBOT_TCP_OFFSET_*`` used by the DiffIK action).
- Per tick: absolute joint angles = ``joint_pos_rel + default_joint_pos`` (77,).

Outputs
-------
- Left and right tool poses in the base frame: positions (2, 3) and rotation matrices (2, 3, 3).

Validation (to do once in a dev run)
------------------------------------
Hold the arm still, compare FK output with the last commanded pose in ``obs["actions"][0:7]``.
Agreement within millimetres confirms the URDF, the tool offset and the quaternion order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.schema import Schema


class ForwardKinematics:
    """Base-frame FK for both Galbot arms.

    Attributes:
        schema: resolved schema (joint indices, default joint positions, kinematics settings).
        urdf_path: path of the robot description, from ``schema.kinematics["urdf_path"]``.
        base_link: name of the base frame link (``base_link``).
        tool_links: (left, right) tool link names (``left_arm_link7``, ``right_arm_link7``).
        tool_offset_pos: (3,) offset from tool link to TCP.
        tool_offset_quat_xyzw: (4,) rotation offset from tool link to TCP.
        chains: per-arm kinematic chain objects (e.g. pytorch_kinematics / pinocchio), built in
            ``load()``; None until loaded.
        arm_joint_idx: (left, right) index arrays of the 7 arm joints in ``joint_pos``.
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

    def load(self) -> None:
        """Build the two kinematic chains from the robot description.

        Functionality:
            Parse the URDF, build a serial chain from ``base_link`` to each tool link, and check
            that the chain's movable joints are exactly the 7 arm joints of that side (in schema
            order). Fixed joints (legs, head at default) are folded into constant transforms using
            ``schema.default_joint_pos``.

        Raises:
            FileNotFoundError: URDF missing.
            ValueError: chain joints do not match the schema arm joints.
        """
        raise NotImplementedError("ForwardKinematics.load: build base→tool chains from the URDF")

    def absolute_joint_pos(self, joint_pos_rel: np.ndarray) -> np.ndarray:
        """Convert the env's relative joint positions to absolute angles.

        Args:
            joint_pos_rel: (77,) ``joint_pos`` observation (position minus default).

        Returns:
            (77,) absolute joint angles = ``joint_pos_rel + schema.default_joint_pos``.
        """
        raise NotImplementedError("ForwardKinematics.absolute_joint_pos")

    def compute(self, joint_pos_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame TCP poses of both arms.

        Functionality:
            Convert to absolute angles, run each chain on its 7 arm joints, apply the tool offset.

        Args:
            joint_pos_rel: (77,) ``joint_pos`` observation.

        Returns:
            positions (2, 3) in metres and rotation matrices (2, 3, 3), order (left, right),
            both in the base frame.
        """
        raise NotImplementedError("ForwardKinematics.compute: FK for left and right TCP")
