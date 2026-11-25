"""Schema, rotations, forward kinematics."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from fakes import LEFT_ARM, RIGHT_ARM, ref_fk, write_test_urdf

from astra.adapter import rotations as R
from astra.adapter.kinematics import ForwardKinematics
from astra.schema import check_runtime_rates, schema_from_dict
from astra.types import STATE_DIM

from conftest import REPO


def test_schema_resolves_state_layout(schema):
    assert list(schema.state_joint_groups["left_arm_q"]) == LEFT_ARM
    assert list(schema.state_joint_groups["right_arm_q"]) == RIGHT_ARM
    assert schema.eef_world_groups == {"left_eef_world": 0, "right_eef_world": 1}
    assert sum(s.stop - s.start for s in schema.state_slices.values()) == STATE_DIM
    assert schema.cameras["head"].hw == (480, 640)


def test_schema_rejects_bad_layouts(schema):
    raw = dict(schema.raw)
    bad = dict(raw, state={**raw["state"], "gripper": {"joint_ids": [65], "dim": 2}})
    with pytest.raises(ValueError, match="joints, dim"):
        schema_from_dict(bad)
    swapped = dict(raw, state={k: raw["state"][k] for k in reversed(list(raw["state"]))})
    with pytest.raises(ValueError, match="in this order"):
        schema_from_dict(swapped)
    with pytest.raises(ValueError, match="duplicate"):
        schema_from_dict(dict(raw, joint_names=["a", "a"]))
    with pytest.raises(ValueError, match="action_quat_order"):
        schema_from_dict(dict(raw, action_quat_order="zyxw"))


def test_runtime_rate_check(schema):
    check_runtime_rates(schema, {"decimation": 4})
    with pytest.raises(ValueError, match="differs"):
        check_runtime_rates(schema, {"decimation": 2})
    with pytest.raises(ValueError, match="decimation"):
        check_runtime_rates(schema, {})


def _converter():
    spec = importlib.util.spec_from_file_location("convert_dataset", REPO / "scripts" / "convert_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rotations_match_the_dataset_converter(rng):
    cd = _converter()
    q = rng.normal(size=(300, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for order in ("xyzw", "wxyz"):
        M = R.quat_to_matrix(q, order)
        np.testing.assert_allclose(M, cd.quat_to_mat(q, order), atol=1e-12)
        q2 = R.matrix_to_quat(M, order)
        np.testing.assert_allclose(np.abs(np.sum(q2 * cd.mat_to_quat(M, order), axis=1)), 1.0, atol=1e-9)
        np.testing.assert_allclose(R.quat_to_matrix(q2, order), M, atol=1e-9)
    r6 = R.matrix_to_rot6d(M)
    np.testing.assert_allclose(r6, cd.mat_to_rot6d(M))
    noisy = r6 + rng.normal(scale=0.05, size=r6.shape)
    np.testing.assert_allclose(R.rot6d_to_matrix(noisy), cd.rot6d_to_mat(noisy), atol=1e-12)


def test_rot6d_gram_schmidt_gives_proper_rotations(rng):
    out = R.rot6d_to_matrix(rng.normal(size=(50, 6)))
    np.testing.assert_allclose(out @ np.swapaxes(out, 1, 2), np.tile(np.eye(3), (50, 1, 1)), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(out), 1.0, atol=1e-9)


def test_hemisphere_and_angles(rng):
    q = np.array([0.0, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(R.make_hemisphere_consistent(q, None, "xyzw"), [0, 0, 0, 1])
    np.testing.assert_allclose(R.make_hemisphere_consistent(q, np.array([0, 0, 0.1, 0.99])), [0, 0, 0, 1])
    a = R.rotvec_to_matrix(np.array([0.0, 0.0, 0.3]))
    b = R.rotvec_to_matrix(np.array([0.0, 0.0, -0.2]))
    assert R.rotation_angle_between(a, b) == pytest.approx(0.5)
    limited, hit = R.limit_rotation_step(a, b, 0.1)
    assert hit and R.rotation_angle_between(a, limited) == pytest.approx(0.1)
    same, hit = R.limit_rotation_step(a, a, 0.1)
    assert not hit
    rv = rng.normal(size=(20, 3))
    rv *= (np.pi * 0.99 / np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), np.pi)) * 3
    np.testing.assert_allclose(R.rotvec_to_matrix(R.matrix_to_rotvec(R.rotvec_to_matrix(rv))),
                               R.rotvec_to_matrix(rv), atol=1e-9)


def test_forward_kinematics_against_reference(schema, tmp_path, rng):
    urdf = write_test_urdf(tmp_path / "robot.urdf")
    schema.kinematics = dict(schema.kinematics, urdf_path=str(urdf), joint_ids={"leg_joint1": 0},
                             tool_offset_pos=[0.0, 0.0, 0.05])
    fk = ForwardKinematics(schema)
    fk.load()
    assert fk.available
    for _ in range(5):
        jp = rng.uniform(-1, 1, size=77)
        pos, rot = fk.compute(jp)
        for i, side in enumerate(("left", "right")):
            ref = ref_fk(jp, side)
            tool = np.eye(4)
            tool[2, 3] = 0.05
            ref = ref @ tool
            np.testing.assert_allclose(pos[i], ref[:3, 3], atol=1e-12)
            np.testing.assert_allclose(rot[i], ref[:3, :3], atol=1e-12)


def test_forward_kinematics_needs_leg_indices(schema, tmp_path):
    urdf = write_test_urdf(tmp_path / "robot.urdf")
    schema.kinematics = dict(schema.kinematics, urdf_path=str(urdf), joint_ids={})
    with pytest.raises(ValueError, match="leg_joint1"):
        ForwardKinematics(schema).load()


def test_forward_kinematics_off_without_urdf(schema):
    fk = ForwardKinematics(schema)
    fk.load()
    assert not fk.available
