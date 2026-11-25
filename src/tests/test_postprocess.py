"""Codec, safety filter, chunk executor, sim interface."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from fakes import FakeEnv

from astra.adapter.rotations import matrix_to_rot6d, rotation_angle_between, rot6d_to_matrix, rotvec_to_matrix
from astra.control.sim_interface import SimActionInterface
from astra.postprocess.action_codec import ActionCodec
from astra.postprocess.chunk_executor import ChunkExecutor
from astra.postprocess.safety import SafetyFilter
from astra.types import (
    BASE_VEL_FROM_WHEELS,
    WHEELS_FROM_BASE_VEL,
    ActionChunk,
    CanonicalObs,
    PolicySource,
    SimAction,
    Stage,
)


def model_action(pos_l=(0.5, 0.2, 1.0), pos_r=(0.5, -0.2, 1.0), rv=(0, 0, 0), grip=(0, 0), base=(0, 0, 0)):
    r6 = matrix_to_rot6d(rotvec_to_matrix(np.asarray(rv, float)))
    return np.concatenate([pos_l, r6, pos_r, r6, grip, base]).astype(np.float32)


def canon(pos=((0.5, 0.2, 1.0), (0.5, -0.2, 1.0)), grip=(0.0, 0.0), valid=True, from_fk=True):
    return CanonicalObs(
        tick=0, t=0.0, images={}, state=np.zeros(34, np.float32), joint_pos=np.zeros(77, np.float32),
        joint_vel=np.zeros(77, np.float32), gripper_raw=np.asarray(grip, np.float32),
        eef_base_pos=np.asarray(pos, float), eef_base_rotmat=np.tile(np.eye(3), (2, 1, 1)),
        last_action=np.zeros(19, np.float32), eef_base_valid=valid, eef_base_from_fk=from_fk,
    )


# ---------------------------------------------------------------- codec
def test_codec_roundtrip_both_formats(schema, rng):
    codec = ActionCodec(schema)
    for dims, dim in (([19, 20], 19), ([20], 20)):
        codec.sim_format = "auto"
        assert codec.select_format(dims) == dim
        codec.reset()
        a = model_action(rv=rng.normal(size=3), grip=(1, 0), base=(0.3, -0.1, 0.5))
        sim = codec.decode(a, (True, False))
        assert sim.shape == (dim,)
        assert sim[14] == -1.0 and sim[15] == 1.0
        back = codec.encode(sim)
        np.testing.assert_allclose(back[:18], a[:18], atol=1e-5)
        np.testing.assert_allclose(back[18:20], [1, 0])
        np.testing.assert_allclose(back[20:], a[20:], atol=1e-6)
    np.testing.assert_allclose(BASE_VEL_FROM_WHEELS @ WHEELS_FROM_BASE_VEL, np.eye(3), atol=1e-6)


def test_codec_quaternions_stay_in_one_hemisphere(schema):
    codec = ActionCodec(schema)
    codec.select_format([19])
    prev = None
    for ang in np.linspace(0, 2 * np.pi, 40):  # full turn: matrix_to_quat alone would flip sign
        q = codec.decode(model_action(rv=(0, 0, ang if ang < np.pi else ang - 2 * np.pi)), (False, False))[3:7]
        if prev is not None:
            assert float(np.dot(q, prev)) > 0
        prev = q


def test_codec_format_errors(schema):
    codec = ActionCodec(schema)
    with pytest.raises(ValueError):
        codec.select_format([16])
    codec.sim_format = "velocity19"
    schema.sim_action_format = "velocity19"
    with pytest.raises(ValueError, match="accepts"):
        ActionCodec(schema).select_format([20])
    with pytest.raises(ValueError):
        codec.encode(np.zeros(18))


# ---------------------------------------------------------------- safety
def test_safety_limits_and_hysteresis():
    sf = SafetyFilter({})
    obs = canon()
    sf.reset(obs)
    far = model_action(pos_l=(5.0, 0.2, 1.0), rv=(0, 0, 1.0), grip=(0.5, 0.9), base=(3.0, 0.01, -2.0))
    out, grip, clamped = sf.apply(far, obs)
    assert clamped
    assert np.linalg.norm(out[0:3] - [0.5, 0.2, 1.0]) == pytest.approx(0.02, abs=1e-6)
    assert rotation_angle_between(np.eye(3), rot6d_to_matrix(out[3:9])) == pytest.approx(0.08, abs=1e-6)
    np.testing.assert_allclose(out[20:], [1.0, 0.0, -1.0])  # clip, deadband
    assert grip == (False, True)
    # hysteresis: 0.5 keeps the right gripper closed, 0.3 opens it
    _, grip, _ = sf.apply(model_action(grip=(0.5, 0.5)), obs)
    assert grip == (False, True)
    _, grip, _ = sf.apply(model_action(grip=(0.5, 0.3)), obs)
    assert grip == (False, False)


def test_safety_workspace_and_nan():
    sf = SafetyFilter({"max_eef_step_m": 10.0})
    obs = canon()
    sf.reset(obs)
    out, _, clamped = sf.apply(model_action(pos_l=(0.5, 0.2, 0.0)), obs)
    assert clamped and out[2] == pytest.approx(0.3)
    bad = model_action()
    bad[4] = np.nan
    out, _, clamped = sf.apply(bad, obs)
    assert clamped and np.all(np.isfinite(out))


def test_safety_hold_without_any_pose_raises():
    sf = SafetyFilter({})
    obs = canon(valid=False, from_fk=False)
    sf.reset(obs)
    with pytest.raises(RuntimeError, match="no base-frame"):
        sf.hold_action(obs)
    held = SafetyFilter({})
    held.reset(canon(grip=(1.6, 0.0)))
    h = held.hold_action(canon(grip=(1.6, 0.0)))
    np.testing.assert_allclose(h[18:], [1, 0, 0, 0, 0])


# ---------------------------------------------------------------- executor
def chunk(T=40, start=0, source=PolicySource.ACT, fps=50.0, value=0.0):
    a = np.tile(model_action(), (T, 1))
    a[:, 0] = value + np.arange(T) * 0.001
    return ActionChunk(a, source, fps, start, Stage.LIFT_LID)


def test_executor_replan_and_index():
    ex = ChunkExecutor({"execute_steps": {"act": 25}, "blend_ticks": 0})
    ex.reset()
    assert ex.need_chunk(0)
    ex.push(chunk(), 0)
    for t in range(25):
        assert not ex.need_chunk(t)
        assert ex.step(t)[0] == pytest.approx(t * 0.001)
    assert ex.need_chunk(25)
    half = chunk(fps=25.0, start=0)
    assert ex.chunk_index(half, 9) == 4  # 25 Hz chunk on a 50 Hz loop
    assert ex.chunk_index(half, 10_000) == 39


def test_executor_blend_and_flush():
    ex = ChunkExecutor({"execute_steps": {"act": 25}, "blend_ticks": 4})
    ex.reset()
    ex.push(chunk(value=0.0), 0)
    ex.step(0)
    new = chunk(start=10, value=1.0)
    new.actions[:, 18] = 1.0
    ex.push(new, 10)
    a = ex.step(10)
    old = 10 * 0.001
    assert a[0] == pytest.approx(0.2 * 1.0 + 0.8 * old, abs=1e-6)
    assert a[18] == 1.0  # grippers are not blended
    assert ex.step(14)[0] == pytest.approx(1.0 + 4 * 0.001)
    ex.flush()
    assert ex.need_chunk(15)
    with pytest.raises(RuntimeError):
        ex.step(15)


def test_executor_plays_whole_script_chunk_and_ensembles():
    ex = ChunkExecutor({"execute_steps": {"act": 25}})
    ex.push(chunk(T=10, source=PolicySource.SCRIPT), 0)
    assert not ex.need_chunk(9) and ex.need_chunk(10)
    ens = ChunkExecutor({"execute_steps": {"act": 25}, "temporal_ensemble": True, "ensemble_coeff": 0.0,
                         "blend_ticks": 0})
    ens.push(chunk(value=0.0), 0)
    assert ens.need_chunk(1)
    ens.push(chunk(value=1.0, start=1), 1)
    a = ens.step(1)
    assert a[0] == pytest.approx((0.001 + 1.0) / 2, abs=1e-6)


# ---------------------------------------------------------------- sim interface
def test_sim_interface_step_and_contract(rng):
    for three in (True, False):
        env = FakeEnv(rng, max_steps=2, three_tuple=three)
        sim = SimActionInterface(env, {"actions_dim": 19, "accepted_action_dims": [19, 20]})
        sim.check_contract(19)
        with pytest.raises(ValueError):
            sim.check_contract(23)
        vec = np.zeros(19, np.float32)
        obs, done, extras = sim.step(SimAction(vec, 0, PolicySource.HOLD))
        assert not done and isinstance(extras, dict) and "policy" in obs
        _, done, _ = sim.step(SimAction(vec, 1, PolicySource.HOLD))
        assert done and sim.steps == 2
        with pytest.raises(ValueError):
            sim.step(SimAction(np.full(19, np.nan, np.float32), 2, PolicySource.HOLD))
    assert isinstance(torch.zeros(1), torch.Tensor)
