"""Schema loader: reads ``config/schema.yaml`` and resolves names to indices.

Purpose
-------
``schema.yaml`` is the single source of truth for observation keys, joint names, state and action
layouts, camera keys and rates. This module loads it once, validates it and exposes resolved index
arrays, so no other module hard-codes a joint index or camera key.

Inputs
------
- Path to ``schema.yaml``.
- At runtime, ``usr_args["decimation"]`` from the harness, to check the control rate.

Outputs
-------
- A ``Schema`` object with:
  - joint name → index map over the 77 ``joint_pos`` entries,
  - index arrays for each state group taken directly from joints (arm q, gripper),
  - camera name → env observation key and native (H, W),
  - kinematics settings, quaternion order of the sim action, rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from astra.types import CONTROL_HZ, DATA_FPS


@dataclass
class CameraSpec:
    """One camera: Astra name, env observation key, native image size (H, W)."""

    name: str
    obs_key: str
    hw: tuple[int, int]


@dataclass
class Schema:
    """Resolved schema used by every runtime module.

    Attributes:
        version: schema version number.
        control_hz: control rate of the env (50).
        data_fps: time base of learned action chunks (50, the same as the control rate).
        action_quat_order: "xyzw" or "wxyz", order of quaternion components in the sim action.
        sim_action_format: "auto", "velocity19" or "wheels20", form of the action sent to the env
            (see ``ActionCodec``).
        joint_names: the 77 joint names in ``joint_pos`` order.
        joint_index: joint name → index into ``joint_pos``.
        state_joint_groups: state group name → index array into ``joint_pos`` (arm q, gripper).
        eef_world_groups: state group name → arm (0 left, 1 right) for the world-frame EEF groups.
        cameras: Astra camera name → CameraSpec.
        kinematics: raw kinematics settings (URDF path, links, tool offset).
        raw: the parsed YAML, for anything not resolved above.
    """

    version: int
    control_hz: float = CONTROL_HZ
    data_fps: float = DATA_FPS
    action_quat_order: str = "xyzw"
    sim_action_format: str = "auto"
    joint_names: list[str] = field(default_factory=list)
    joint_index: dict[str, int] = field(default_factory=dict)
    state_joint_groups: dict[str, np.ndarray] = field(default_factory=dict)
    eef_world_groups: dict[str, int] = field(default_factory=dict)
    cameras: dict[str, CameraSpec] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load_schema(path: str | Path) -> Schema:
    """Load and resolve ``schema.yaml``.

    Functionality:
        1. Parse YAML (PyYAML, imported lazily).
        2. Build ``joint_index`` from ``joint_names`` (may be empty until FK is set up); fail if
           names are duplicated.
        3. Turn every ``state`` group with ``joint_ids:`` into an index array; record groups with
           ``eef:`` in ``eef_world_groups``. Check the concatenated dims add up to ``types.STATE_DIM`` (34)
           in the documented order. The model action layout lives only in ``astra.types``.
        4. Build ``cameras`` from the ``cameras`` section.

    Args:
        path: filesystem path to schema.yaml.

    Returns:
        A fully resolved ``Schema``.

    Raises:
        ValueError: on any inconsistency (unknown joint name, wrong dims, empty joint list).
    """
    raise NotImplementedError("load_schema: parse and validate schema.yaml")


def check_runtime_rates(schema: Schema, usr_args: dict[str, Any]) -> None:
    """Fail fast if the env's control rate differs from the schema.

    Functionality:
        Read ``usr_args["decimation"]`` (set by the harness from ``get_eval_metadata()``), compute
        the control rate from the physics rate (200 Hz) and compare it with ``schema.control_hz``.
        A mismatch would make every learned chunk play back at the wrong speed.

    Args:
        schema: resolved schema.
        usr_args: harness arguments; must contain "decimation".

    Raises:
        ValueError: if the rates differ.
    """
    raise NotImplementedError("check_runtime_rates: compare decimation with schema.control_hz")
