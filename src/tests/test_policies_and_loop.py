"""Router, scripted wipe, and the full AstraPolicy loop against a fake env."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from fakes import FakeEnv, write_test_urdf
from test_postprocess import canon, model_action

from astra.policies.base import PolicyBackend
from astra.policies.router import PolicyRouter
from astra.policies.scripted_wipe import ScriptedWipe, default_waypoints
from astra.types import (
    ActionChunk,
    OrchestratorEvent,
    OrchestratorOutput,
    PolicySource,
    Stage,
)

from conftest import SRC


def orch_out(event=OrchestratorEvent.NONE, stage=Stage.LIFT_LID, arm=None):
    return OrchestratorOutput(stage, 0, "p", event, 10.0, 0, arm)


class Dummy(PolicyBackend):
    """Learned-policy stand-in: a chunk that holds the observed pose (or fails on demand)."""

    source = PolicySource.ACT
    fps = 50.0

    def __init__(self, fail=False, T=40):
        super().__init__()
        self.fail, self.T, self.calls, self.resets = fail, T, 0, 0

    def load(self):
        self.loaded = True

    def reset(self):
        self.resets += 1

    def predict(self, obs, view, orch):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        a = model_action(pos_l=tuple(obs.eef_base_pos[0]), pos_r=tuple(obs.eef_base_pos[1]), base=(0.2, 0, 0))
        return ActionChunk(np.tile(a, (self.T, 1)), self.source, self.fps, obs.tick, orch.stage)


class FakeAdapter:
    def act_view(self, obs, stage):
        return None

    def groot_view(self, obs, prompt):
        return None


def router(act=None, script=None, **cfg):
    r = PolicyRouter({"stage_to_source": ["act", "act", "act", "script", "act", "act"], **cfg}, act, None, script)
    r.load()
    r.reset()
    return r


def test_router_switching():
    script = ScriptedWipe({})
    r = router(Dummy(), script)
    assert r.active_source == PolicySource.ACT
    assert r.select_source(orch_out()) == (PolicySource.ACT, False)
    assert r.select_source(orch_out(OrchestratorEvent.ADVANCE_KEEP_QUEUE, Stage.PACK_COMPONENT)) == (PolicySource.ACT, False)
    assert r.select_source(orch_out(OrchestratorEvent.ADVANCE, Stage.WIPE_CHAMBER)) == (PolicySource.ACT, True)
    assert r.select_source(orch_out(OrchestratorEvent.ENTER_SCRIPT, Stage.WIPE_CHAMBER, 1)) == (PolicySource.SCRIPT, True)
    chunk = r.get_chunk(canon(grip=(0, 1.2)), orch_out(stage=Stage.WIPE_CHAMBER, arm=1), FakeAdapter())
    assert chunk.source == PolicySource.SCRIPT and script.started and script.wiping_arm == 1
    assert r.select_source(orch_out(OrchestratorEvent.RETRY, Stage.WIPE_CHAMBER)) == (PolicySource.ACT, True)
    assert r.backends[PolicySource.ACT].resets >= 2


def test_router_faults_fall_back_to_hold():
    r = router(Dummy(fail=True))
    c = r.get_chunk(canon(), orch_out(), FakeAdapter())
    assert c.source == PolicySource.HOLD and r.fault_count == 1
    np.testing.assert_allclose(c.actions[0, 0:3], [0.5, 0.2, 1.0])
    assert router().get_chunk(canon(), orch_out(), FakeAdapter()).source == PolicySource.HOLD  # no backends


def test_scripted_wipe_pattern():
    sw = ScriptedWipe({"n_contacts": 6, "dwell_s": 0.5, "press_s": 0.1, "lift_m": 0.03, "press_depth_m": 0.02})
    sw.load()
    assert sw.waypoints.shape == (6, 6)
    obs = canon(grip=(0.0, 1.2))
    sw.start(obs, 1)
    rows = []
    while not sw.done:
        c = sw.predict(obs, None, orch_out(stage=Stage.WIPE_CHAMBER, arm=1))
        assert c.fps == 50.0 and c.source == PolicySource.SCRIPT
        rows.append(c.actions)
    rows = np.concatenate(rows)
    assert sw.contacts_done == 6
    z = rows[:, 9 + 2]
    pressed = z < 1.0 - 0.019
    # six separate dwell phases at the pressed depth, each at least dwell_s long
    edges = np.flatnonzero(np.diff(pressed.astype(int)) == 1)
    assert len(edges) == 6
    assert np.all(rows[:, 19] == 1.0) and np.all(rows[:, 18] == 0.0)  # right closed, left as observed
    np.testing.assert_allclose(rows[:, 0:3], np.tile([0.5, 0.2, 1.0], (len(rows), 1)))  # other arm still
    np.testing.assert_allclose(rows[:, 20:], 0.0)
    assert len(default_waypoints(5, 0.04)) == 5


def test_scripted_wipe_waypoint_file(tmp_path):
    (tmp_path / "wp.yaml").write_text("waypoints:\n  - [0, 0, 0, 0, 0, 2]\n  - [0.05, 0, 0, 0, 0, 1]\n")
    sw = ScriptedWipe({"waypoints": str(tmp_path / "wp.yaml"), "n_contacts": 2})
    sw.load()
    np.testing.assert_allclose(sw.waypoints[0, 3:], [0, 0, 1])


# ---------------------------------------------------------------- full loop
def write_configs(tmp_path, urdf=None, **runtime_over):
    schema = yaml.safe_load((SRC / "astra/config/schema.yaml").read_text())
    if urdf:
        schema["kinematics"]["urdf_path"] = str(urdf)
        schema["kinematics"]["joint_ids"] = {"leg_joint1": 0}
    runtime = yaml.safe_load((SRC / "astra/config/runtime.yaml").read_text())
    runtime["device"] = "cpu"
    for k, v in runtime_over.items():
        runtime[k] = {**runtime.get(k, {}), **v} if isinstance(v, dict) else v
    (tmp_path / "schema.yaml").write_text(yaml.safe_dump(schema, sort_keys=False))
    (tmp_path / "runtime.yaml").write_text(yaml.safe_dump(runtime, sort_keys=False))
    return {
        "astra_schema": str(tmp_path / "schema.yaml"),
        "astra_runtime": str(tmp_path / "runtime.yaml"),
        "actions_dim": 19,
        "accepted_action_dims": [19, 20],
        "decimation": 4,
        "env_cfg": {"num_envs": 1},
        "save_path": str(tmp_path / "out"),
    }


def make_policy(usr_args, backend=None):
    from astra.policy import AstraPolicy

    pol = AstraPolicy(usr_args)
    if backend is not None:
        pol.router.backends[PolicySource.ACT] = backend
    pol.reset_model()
    return pol


def test_eval_holds_with_fk_and_no_models(tmp_path, rng):
    usr = write_configs(tmp_path, urdf=write_test_urdf(tmp_path / "r.urdf"))
    pol = make_policy(usr)
    env = FakeEnv(rng, max_steps=60)
    frames = []
    assert pol.eval(env, env.first_obs(), usr, frames) is False
    acts = np.stack(env.actions)
    assert acts.shape == (60, 19)
    assert np.all(acts[:, 14:16] == 1.0)  # grippers open
    np.testing.assert_allclose(acts[:, 16:19], 0.0)  # base still
    np.testing.assert_allclose(acts[1:, 0:7], np.repeat(acts[:1, 0:7], 59, 0), atol=1e-5)  # arm held
    assert len(frames) == pytest.approx(60 * 30 / 50, abs=2)
    summary = json.loads((tmp_path / "out" / "astra_episode.json").read_text())
    assert summary["stop_reason"] == "env terminated" and summary["ticks"] == 60


def test_eval_runs_a_policy_and_respects_the_safety_filter(tmp_path, rng):
    usr = write_configs(tmp_path, urdf=write_test_urdf(tmp_path / "r.urdf"))
    dummy = Dummy(T=40)
    pol = make_policy(usr, dummy)
    env = FakeEnv(rng, max_steps=100)
    pol.eval(env, env.first_obs(), usr, None)
    acts = np.stack(env.actions)
    assert dummy.calls == 4  # re-plan every 25 ticks
    np.testing.assert_allclose(acts[:, 16], 0.2, atol=1e-6)  # base velocity passed through
    steps = np.linalg.norm(np.diff(acts[:, 0:3], axis=0), axis=1)
    assert steps.max() <= 0.02 + 1e-6
    q = acts[:, 3:7]
    assert np.all(np.sum(q[1:] * q[:-1], axis=1) > 0)  # no quaternion sign flips


def test_eval_legacy_wheels_format(tmp_path, rng):
    usr = write_configs(tmp_path, urdf=write_test_urdf(tmp_path / "r.urdf"))
    usr["accepted_action_dims"] = [20]
    usr["actions_dim"] = 20
    pol = make_policy(usr, Dummy())
    env = FakeEnv(rng, max_steps=5)
    pol.eval(env, env.first_obs(), usr, None)
    acts = np.stack(env.actions)
    assert acts.shape == (5, 20)
    np.testing.assert_allclose(acts[:, 16:20], np.tile([0.2, 0.2, 0.2, 0.2], (5, 1)), atol=1e-6)


def test_eval_rejects_wrong_rate_and_unsafe_setup(tmp_path, rng):
    usr = write_configs(tmp_path)
    pol = make_policy(usr)
    env = FakeEnv(rng, max_steps=5)
    with pytest.raises(ValueError, match="differs"):
        pol.eval(env, env.first_obs(), dict(usr, decimation=8), None)
    pol.reset_model()
    with pytest.raises(RuntimeError, match="nothing can produce a safe arm command"):
        pol.eval(env, env.first_obs(), usr, None)


def test_eval_without_fk_uses_the_policy(tmp_path, rng):
    usr = write_configs(tmp_path)
    pol = make_policy(usr, Dummy())
    env = FakeEnv(rng, max_steps=30)
    pol.eval(env, env.first_obs(), usr, None)
    assert len(env.actions) == 30


def test_eval_stops_at_the_episode_limit(tmp_path, rng):
    usr = write_configs(tmp_path, urdf=write_test_urdf(tmp_path / "r.urdf"), episode_limit_s=1.0)
    pol = make_policy(usr)
    env = FakeEnv(rng, max_steps=10_000)
    pol.eval(env, env.first_obs(), usr, None)
    assert len(env.actions) == 50


def _groot_with_modality(monkeypatch, tmp_path, video=("head", "left_wrist", "right_wrist")):
    """Gr00tPolicyBackend around a stub Gr00tPolicy whose modality config matches the Galbot fine-tune."""
    from types import SimpleNamespace

    import astra.policies.groot_policy as gp

    seen = {}
    cfg = {
        "video": SimpleNamespace(modality_keys=list(video), delta_indices=[0]),
        "state": SimpleNamespace(modality_keys=list(gp.STATE_KEYS), delta_indices=[0]),
        "action": SimpleNamespace(modality_keys=list(gp.GROUP_SLICES), delta_indices=list(range(40))),
        "language": SimpleNamespace(modality_keys=["annotation.human.task_description"], delta_indices=[0]),
    }

    class StubPolicy:
        def __init__(self, **kwargs):
            pass

        def get_modality_config(self):
            return cfg

        def get_action(self, observation):
            seen["language"] = observation["language"]
            dims = {k: s.stop - s.start for k, s in gp.GROUP_SLICES.items()}
            return {k: np.zeros((1, 40, d), np.float32) for k, d in dims.items()}, {}

    monkeypatch.setattr(gp, "_import_gr00t", lambda: (StubPolicy, SimpleNamespace(NEW_EMBODIMENT="new_embodiment")))
    return gp.Gr00tPolicyBackend(str(tmp_path), device="cpu"), seen


def test_groot_uses_the_checkpoint_language_key(monkeypatch, tmp_path, schema, rng):
    from fakes import make_raw_obs

    from astra.adapter.observation_adapter import ObservationAdapter

    backend, seen = _groot_with_modality(monkeypatch, tmp_path)
    backend.load()
    assert backend.language_key == "annotation.human.task_description" and backend.action_horizon == 40
    a = ObservationAdapter(schema, {"history_seconds": 3.0}, use_stage_conditioning=False)
    a.load()
    a.reset()
    obs = a.process(make_raw_obs(rng))
    chunk = backend.predict(obs, a.groot_view(obs, "open the lid"), orch_out())
    assert chunk.actions.shape == (40, 23)
    assert seen["language"] == {"annotation.human.task_description": [["open the lid"]]}


def test_groot_rejects_a_checkpoint_with_other_cameras(monkeypatch, tmp_path):
    backend, _ = _groot_with_modality(monkeypatch, tmp_path, video=("ego_view",))
    with pytest.raises(ValueError, match="video keys"):
        backend.load()
