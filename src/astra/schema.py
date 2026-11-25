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

from astra.types import (
    CAMERA_NAMES,
    CONTROL_HZ,
    DATA_FPS,
    STATE_DIM,
    STATE_GRIPPER,
    STATE_LEFT_ARM_Q,
    STATE_LEFT_EEF_WORLD,
    STATE_RIGHT_ARM_Q,
    STATE_RIGHT_EEF_WORLD,
)

PHYSICS_HZ: float = 200.0  # sim.dt = 1/200 in the ChamberServiceTask env config

# The documented state layout (``astra.types``); schema.yaml must list exactly these groups in this order.
EXPECTED_STATE_GROUPS: dict[str, slice] = {
    "left_arm_q": STATE_LEFT_ARM_Q,
    "right_arm_q": STATE_RIGHT_ARM_Q,
    "gripper": STATE_GRIPPER,
    "left_eef_world": STATE_LEFT_EEF_WORLD,
    "right_eef_world": STATE_RIGHT_EEF_WORLD,
}
_QUAT_ORDERS = ("xyzw", "wxyz")
_SIM_FORMATS = ("auto", "velocity19", "wheels20")


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
        obs_quat_order: "xyzw" or "wxyz", order of the components of ``eef_quat`` in the
            observation (FrameTransformer ``target_quat_w``; xyzw in the pinned Isaac Lab).
        sim_action_format: "auto", "velocity19" or "wheels20", form of the action sent to the env
            (see ``ActionCodec``).
        joint_names: the 77 joint names in ``joint_pos`` order.
        joint_index: joint name → index into ``joint_pos``.
        state_joint_groups: state group name → index array into ``joint_pos`` (arm q, gripper).
        state_slices: state group name → slice of the 34-D state, in concatenation order.
        eef_world_groups: state group name → arm (0 left, 1 right) for the world-frame EEF groups.
        cameras: Astra camera name → CameraSpec.
        kinematics: raw kinematics settings (URDF path, links, tool offset).
        raw: the parsed YAML, for anything not resolved above.
    """

    version: int
    control_hz: float = CONTROL_HZ
    data_fps: float = DATA_FPS
    action_quat_order: str = "xyzw"
    obs_quat_order: str = "xyzw"
    sim_action_format: str = "auto"
    joint_names: list[str] = field(default_factory=list)
    joint_index: dict[str, int] = field(default_factory=dict)
    state_joint_groups: dict[str, np.ndarray] = field(default_factory=dict)
    state_slices: dict[str, slice] = field(default_factory=dict)
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
    import yaml

    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return schema_from_dict(raw, source=str(path))


def schema_from_dict(raw: dict[str, Any], source: str = "schema") -> Schema:
    """Resolve an already parsed schema mapping (``load_schema`` without the file; used by tests)."""
    if "version" not in raw:
        raise ValueError(f"{source}: missing 'version'")

    control_hz = float(raw.get("control_hz", CONTROL_HZ))
    data_fps = float(raw.get("data_fps", DATA_FPS))
    if control_hz <= 0 or data_fps <= 0:
        raise ValueError(f"{source}: control_hz and data_fps must be positive")

    action_quat_order = str(raw.get("action_quat_order", "xyzw"))
    obs_quat_order = str(raw.get("obs_quat_order", "xyzw"))
    for key, value in (("action_quat_order", action_quat_order), ("obs_quat_order", obs_quat_order)):
        if value not in _QUAT_ORDERS:
            raise ValueError(f"{source}: {key} must be one of {_QUAT_ORDERS}, got {value!r}")
    sim_action_format = str(raw.get("sim_action_format", "auto"))
    if sim_action_format not in _SIM_FORMATS:
        raise ValueError(f"{source}: sim_action_format must be one of {_SIM_FORMATS}, got {sim_action_format!r}")

    # ---------------------------------------------------------------- joints
    joint_names = [str(n) for n in (raw.get("joint_names") or [])]
    joint_index: dict[str, int] = {}
    for i, name in enumerate(joint_names):
        if name in joint_index:
            raise ValueError(f"{source}: duplicate joint name {name!r} (indices {joint_index[name]} and {i})")
        joint_index[name] = i

    # ---------------------------------------------------------------- state groups
    state_cfg = raw.get("state") or {}
    if not isinstance(state_cfg, dict) or not state_cfg:
        raise ValueError(f"{source}: 'state' section is missing or empty")
    names = list(state_cfg)
    if names != list(EXPECTED_STATE_GROUPS):
        raise ValueError(f"{source}: state groups must be {list(EXPECTED_STATE_GROUPS)} in this order, got {names}")

    state_joint_groups: dict[str, np.ndarray] = {}
    eef_world_groups: dict[str, int] = {}
    state_slices: dict[str, slice] = {}
    offset = 0
    for name, group in state_cfg.items():
        if not isinstance(group, dict) or "dim" not in group:
            raise ValueError(f"{source}: state group {name!r} needs a 'dim'")
        dim = int(group["dim"])
        expected = EXPECTED_STATE_GROUPS[name]
        sl = slice(offset, offset + dim)
        if (sl.start, sl.stop) != (expected.start, expected.stop):
            raise ValueError(
                f"{source}: state group {name!r} occupies {sl.start}:{sl.stop}, astra.types expects "
                f"{expected.start}:{expected.stop}"
            )
        state_slices[name] = sl
        offset += dim
        if "joint_ids" in group or "joints" in group:
            if "joint_ids" in group:
                ids = [int(i) for i in group["joint_ids"]]
            else:
                missing = [j for j in group["joints"] if j not in joint_index]
                if missing:
                    raise ValueError(f"{source}: state group {name!r} names unknown joints {missing}")
                ids = [joint_index[j] for j in group["joints"]]
            if len(ids) != dim:
                raise ValueError(f"{source}: state group {name!r} has {len(ids)} joints, dim {dim}")
            if len(set(ids)) != len(ids) or min(ids) < 0:
                raise ValueError(f"{source}: state group {name!r} has duplicate or negative joint ids {ids}")
            if joint_names and max(ids) >= len(joint_names):
                raise ValueError(f"{source}: state group {name!r} joint id {max(ids)} >= {len(joint_names)} joints")
            state_joint_groups[name] = np.asarray(ids, dtype=np.int64)
        elif "eef" in group:
            arm = int(group["eef"])
            if arm not in (0, 1):
                raise ValueError(f"{source}: state group {name!r}: eef must be 0 (left) or 1 (right)")
            if dim != 9:
                raise ValueError(f"{source}: state group {name!r}: an eef group is xyz + rot6d, dim 9")
            eef_world_groups[name] = arm
        else:
            raise ValueError(f"{source}: state group {name!r} needs 'joint_ids', 'joints' or 'eef'")
    if offset != STATE_DIM:
        raise ValueError(f"{source}: state dims add up to {offset}, expected {STATE_DIM}")

    # ---------------------------------------------------------------- cameras
    cam_cfg = raw.get("cameras") or {}
    cameras: dict[str, CameraSpec] = {}
    for name, spec in cam_cfg.items():
        if not isinstance(spec, dict) or "key" not in spec or "hw" not in spec:
            raise ValueError(f"{source}: camera {name!r} needs 'key' and 'hw'")
        hw = tuple(int(v) for v in spec["hw"])
        if len(hw) != 2 or min(hw) <= 0:
            raise ValueError(f"{source}: camera {name!r}: hw must be [H, W], got {spec['hw']}")
        cameras[str(name)] = CameraSpec(name=str(name), obs_key=str(spec["key"]), hw=(hw[0], hw[1]))
    if set(cameras) != set(CAMERA_NAMES):
        raise ValueError(f"{source}: cameras must be {list(CAMERA_NAMES)}, got {list(cameras)}")

    kinematics = dict(raw.get("kinematics") or {})

    return Schema(
        version=int(raw["version"]),
        control_hz=control_hz,
        data_fps=data_fps,
        action_quat_order=action_quat_order,
        obs_quat_order=obs_quat_order,
        sim_action_format=sim_action_format,
        joint_names=joint_names,
        joint_index=joint_index,
        state_joint_groups=state_joint_groups,
        state_slices=state_slices,
        eef_world_groups=eef_world_groups,
        cameras=cameras,
        kinematics=kinematics,
        raw=raw,
    )


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
    if "decimation" not in usr_args:
        raise ValueError("usr_args has no 'decimation': the harness did not attach the env metadata")
    decimation = int(usr_args["decimation"])
    if decimation <= 0:
        raise ValueError(f"invalid decimation {decimation}")
    physics_hz = float(usr_args.get("physics_hz", PHYSICS_HZ))
    env_hz = physics_hz / decimation
    if abs(env_hz - schema.control_hz) > 1e-6:
        raise ValueError(
            f"env control rate {env_hz:g} Hz (physics {physics_hz:g} Hz / decimation {decimation}) "
            f"differs from schema control_hz {schema.control_hz:g} Hz"
        )
