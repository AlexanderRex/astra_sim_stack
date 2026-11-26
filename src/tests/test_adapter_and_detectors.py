"""Observation adapter, views, history, event detectors."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from fakes import GRIPPERS, LEFT_ARM, RIGHT_ARM, make_raw_obs, write_test_urdf
from robofinals.distributed.chamber_schema import JOINT_DEFAULT_POS

from astra.adapter.observation_adapter import ObservationAdapter
from astra.orchestrator.event_detectors import EventDetectors
from astra.types import CanonicalObs, PolicySource, SimAction, Stage

from conftest import REPO


def adapter(schema, **cfg):
    a = ObservationAdapter(schema, {"max_invalid_ticks": 3, **cfg})
    a.load()
    a.reset()
    return a


def test_state_matches_the_dataset_converter(schema, rng):
    spec = importlib.util.spec_from_file_location("cd", REPO / "scripts" / "convert_dataset.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    raw = make_raw_obs(rng)
    obs = adapter(schema).process(raw)
    jp = raw["policy"]["joint_pos"].numpy()[0] + np.asarray(JOINT_DEFAULT_POS)
    want = cd.build_state(raw["policy"]["eef_pos"].numpy()[0], raw["policy"]["eef_quat"].numpy()[0], jp)[0]
    np.testing.assert_allclose(obs.state, want, atol=1e-6)
    np.testing.assert_allclose(obs.state[0:7], jp[LEFT_ARM], atol=1e-6)
    np.testing.assert_allclose(obs.state[7:14], jp[RIGHT_ARM], atol=1e-6)
    assert obs.valid and obs.tick == 0
    assert set(obs.images) == {"head", "left_wrist", "right_wrist"}
    assert obs.images["head"].shape == (480, 640, 3) and obs.images["head"].dtype == np.uint8


def test_float_images_and_absolute_joint_pos(schema, rng):
    raw = make_raw_obs(rng, image_dtype="float")
    obs = adapter(schema).process(raw)
    assert obs.valid and obs.images["left_wrist"].dtype == np.uint8
    # the relative joint_pos of the service becomes absolute: an all-zero one is the default pose
    zero = make_raw_obs(rng, joint_pos=np.asarray(JOINT_DEFAULT_POS))
    assert not zero["policy"]["joint_pos"].numpy().any()
    np.testing.assert_allclose(adapter(schema).process(zero).joint_pos, JOINT_DEFAULT_POS, atol=1e-6)


def test_invalid_observations_and_hold(schema, rng):
    a = adapter(schema)
    blank = a.process(make_raw_obs(rng, blank=True))
    assert not blank.valid and any("blank" in i for i in blank.issues)
    good = a.process(make_raw_obs(rng))
    assert good.valid and a.invalid_streak == 0
    jp = rng.normal(size=77)
    jp[3] = np.nan
    for k in range(3):
        bad = a.process(make_raw_obs(rng, joint_pos=jp))
        assert not bad.valid and bad.tick == 2 + k
        np.testing.assert_allclose(bad.state, good.state)  # proprio repeated from the last valid tick
    assert a.needs_hold


def test_pose_falls_back_to_the_command_without_fk(schema, rng):
    a = adapter(schema)
    first = a.process(make_raw_obs(rng))
    assert not first.eef_base_valid and not first.eef_base_from_fk
    act = np.zeros(19)
    act[0:7] = [0.4, 0.2, 1.1, 0, 0, 0, 1]
    act[7:14] = [0.4, -0.2, 1.1, 0, 0, 0, 1]
    second = a.process(make_raw_obs(rng, actions=act))
    assert second.eef_base_valid
    np.testing.assert_allclose(second.eef_base_pos[0], [0.4, 0.2, 1.1])


def test_views(schema, rng):
    a = adapter(schema)
    obs = None
    for _ in range(30):
        obs = a.process(make_raw_obs(rng))
    act = a.act_view(obs, Stage.WIPE)
    assert act.images["observation.images.first_person_camera_rgb"].shape == (3, 224, 224)
    assert 0.0 <= act.images["observation.images.left_hand_camera_rgb"].max() <= 1.0
    np.testing.assert_allclose(act.stage_one_hot, [0, 0, 0, 1, 0, 0])
    assert a.act_view(obs, Stage.DONE).stage_one_hot.sum() == 0
    g = a.groot_view(obs, "wipe")
    assert g.video["head"].shape == (1, 1, 480, 640, 3)
    assert g.state["gripper"].shape == (1, 1, 2) and g.state["left_eef_world"].shape == (1, 1, 9)
    assert g.language == {"task": [["wipe"]]}
    c = a.classifier_input()
    assert c["state"].shape == (15, 34) and c["images.left_wrist"].shape == (15, 3, 112, 112)
    w = a.history.window(3.0, 5.0)
    assert [o.tick for o in w][-3:] == [9, 19, 29]


def test_history_pads_after_reset(schema, rng):
    a = adapter(schema)
    a.process(make_raw_obs(rng))
    w = a.history.window(3.0, 5.0)
    assert len(w) == 15 and all(o.tick == 0 for o in w)


# ---------------------------------------------------------------- detectors
def det_obs(tick, grip, pos=((0.5, 0.2, 1.0), (0.5, -0.2, 1.0)), from_fk=True, rot=None):
    return CanonicalObs(
        tick=tick, t=tick / 50, images={}, state=np.zeros(34, np.float32), joint_pos=np.zeros(77, np.float32),
        joint_vel=np.zeros(77, np.float32), gripper_raw=np.asarray(grip, np.float32),
        eef_base_pos=np.asarray(pos, float),
        eef_base_rotmat=np.tile(np.eye(3), (2, 1, 1)) if rot is None else rot,
        last_action=np.zeros(19, np.float32), eef_base_from_fk=from_fk,
    )


def cmd(close_left=False, close_right=False, target_r=(0.5, -0.2, 1.0), target_l=(0.5, 0.2, 1.0)):
    v = np.zeros(19, np.float32)
    v[0:3], v[3:7] = target_l, [0, 0, 0, 1]
    v[7:10], v[10:14] = target_r, [0, 0, 0, 1]
    v[14] = -1 if close_left else 1
    v[15] = -1 if close_right else 1
    return SimAction(v, 0, PolicySource.ACT)


def run(det, seq):
    out = []
    for tick, (grip, action, kw) in enumerate(seq):
        out.append(det.update(det_obs(tick, grip, **kw), action))
    return out


def test_holding_grasp_release_drop_and_empty():
    det = EventDetectors({"gripper_settle_s": 0.1}, 50.0)  # 5 settle ticks
    det.reset()
    seq = [((0, 0), cmd(), {})] * 3
    seq += [((0, x), cmd(close_right=True), {}) for x in (0.4, 0.8, 1.2)]
    seq += [((0, 1.2), cmd(close_right=True), {})] * 8          # settles on an object → grasp
    seq += [((0, 0.6), cmd(), {})] * 2                            # opens → release
    seq += [((x, 0), cmd(close_left=True), {}) for x in (0.6, 1.2, 1.703)]
    seq += [((1.703, 0), cmd(close_left=True), {})] * 8         # fully closed → empty grasp
    ev = run(det, seq)
    grasps = [i for i, e in enumerate(ev) if e.grasp[1]]
    releases = [i for i, e in enumerate(ev) if e.release[1]]
    empties = [i for i, e in enumerate(ev) if e.empty_grasp[0]]
    assert len(grasps) == 1 and len(releases) == 1 and releases[0] > grasps[0]
    assert len(empties) == 1 and not any(e.holding[0] for e in ev)
    assert ev[grasps[0]].active_arm == 1

    det.reset()
    seq = [((0, 1.2), cmd(close_right=True), {})] * 8 + [((0, 1.703), cmd(close_right=True), {})]
    ev = run(det, seq)
    assert ev[-1].drop[1] and not ev[-1].release[1]


def test_wipe_contacts_are_counted_with_breaks():
    det = EventDetectors({"gripper_settle_s": 0.1, "wipe_dwell_s": 0.2, "wipe_break_s": 0.06,
                          "contact_error_m": 0.01}, 50.0)
    det.reset()
    det.start_wipe_count()
    hold = ((0, 1.2), cmd(close_right=True), {})
    press = ((0, 1.2), cmd(close_right=True, target_r=(0.5, -0.2, 0.97)), {})
    seq = [hold] * 8
    for _ in range(3):
        seq += [press] * 12 + [hold] * 5
    seq += [press] * 5  # too short
    ev = run(det, seq)
    assert ev[-1].wipe_count == 3
    assert any(e.in_contact for e in ev) and ev[-1].contact_available
    # without FK there is no contact proxy
    det.reset()
    det.start_wipe_count()
    ev = [det.update(det_obs(t, (0, 1.2), from_fk=False), press[1]) for t in range(40)]
    assert ev[-1].wipe_count == 0 and not ev[-1].contact_available


def test_kinematic_lid_estimate_opening_and_closing():
    from astra.adapter.rotations import rotvec_to_matrix

    det = EventDetectors({"gripper_settle_s": 0.1}, 50.0)
    det.reset()
    det.enable_lid_tracking(0.0)
    hold = cmd(close_right=True)
    ev = None
    for t in range(8):
        ev = det.update(det_obs(t, (0, 1.2)), hold)
    assert ev.lid_tracking and ev.lid_angle_kin == pytest.approx(0.0)
    for t, ang in enumerate(np.linspace(0, np.pi / 4, 10), start=8):  # 45° = 0.5
        rot = np.stack([np.eye(3), rotvec_to_matrix(np.array([0.0, ang, 0.0]))])
        ev = det.update(det_obs(t, (0, 1.2), rot=rot), hold)
    assert ev.lid_angle_kin == pytest.approx(0.5, abs=1e-6) and ev.lid_angle_kin_peak == pytest.approx(0.5)
    ev = det.update(det_obs(20, (0, 0.8)), cmd())
    assert not ev.lid_tracking and ev.lid_angle_kin_peak == pytest.approx(0.5)

    det.enable_lid_tracking(0.6, closing=True)
    for t in range(21, 30):
        ev = det.update(det_obs(t, (0, 1.2)), hold)
    rot = np.stack([np.eye(3), rotvec_to_matrix(np.array([0.0, -np.pi * 0.3, 0.0]))])
    ev = det.update(det_obs(31, (0, 1.2), rot=rot), hold)
    assert ev.lid_angle_kin == pytest.approx(0.0) and ev.lid_angle_kin_min == pytest.approx(0.0)


def test_regions_and_hinge_files(tmp_path):
    (tmp_path / "regions.yaml").write_text(
        "regions:\n  chamber: {center: [0.5, -0.2, 1.0], radius: 0.1}\n  table: {center: [0.0, 0.0, 0.0], radius: 0.1}\n"
    )
    (tmp_path / "hinge.yaml").write_text("hinge: {point: [0, 0, 0], direction: [0, 0, 2]}\n")
    det = EventDetectors({"regions_calibration": str(tmp_path / "regions.yaml"),
                          "hinge_calibration": str(tmp_path / "hinge.yaml")}, 50.0)
    det.reset()
    ev = det.update(det_obs(0, (0, 0)), cmd())
    assert ev.arm_regions == ("none", "chamber")
    np.testing.assert_allclose(det.hinge_axis_dir, [0, 0, 1])
    (tmp_path / "bad.yaml").write_text("regions:\n  kitchen: {center: [0, 0, 0], radius: 1}\n")
    with pytest.raises(ValueError, match="unknown region"):
        det.load_regions(str(tmp_path / "bad.yaml"))


def test_adapter_uses_fk_when_configured(schema, tmp_path, rng):
    urdf = write_test_urdf(tmp_path / "r.urdf")
    schema.kinematics = dict(schema.kinematics, urdf_path=str(urdf), joint_ids={"leg_joint1": 0})
    a = adapter(schema)
    obs = a.process(make_raw_obs(rng))
    assert obs.eef_base_valid and obs.eef_base_from_fk
    assert obs.gripper_raw.shape == (2,) and list(GRIPPERS) == [65, 69]
