"""Task orchestrator driven by scripted detector events (no classifier, no region calibration)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from astra.orchestrator.event_detectors import EventDetectors
from astra.orchestrator.stage_classifier import StageClassifier
from astra.orchestrator.stage_specs import build_stage_specs, lid_open_estimate, stage_complete_lift_lid
from astra.orchestrator.task_orchestrator import TaskOrchestrator
from astra.types import CanonicalObs, EventState, OrchestratorEvent, Stage, StagePrediction

CFG = {
    "min_dwell_s": [0.1] * 6,
    "debounce_ticks": 3,
    "lid_stable_s": 0.2,
    "stage_budget_s": [100] * 6,
    "max_retries_per_stage": 2,
}


class ScriptedDetectors(EventDetectors):
    """Returns the queued EventState instead of computing it."""

    def __init__(self):
        super().__init__({}, 50.0)
        self.next: EventState = EventState()

    def update(self, obs, last_action):
        if self.counting_wipes:
            self.wipe_count = self.next.wipe_count
        return self.next


class World:
    """Right-hand pose and holding state, stepped one tick at a time through the orchestrator."""

    def __init__(self, orch: TaskOrchestrator):
        self.orch = orch
        self.det: ScriptedDetectors = orch.detectors  # type: ignore[assignment]
        self.tick = 0
        self.pos = np.array([0.5, -0.2, 1.0])
        self.holding = False
        self.base = EventState()
        self.outputs = []

    def step(self, n=1, move=(0, 0, 0), **ev):
        out = None
        for _ in range(n):
            self.pos = self.pos + np.asarray(move, float) / n
            fields = dict(dataclasses.asdict(self.base))
            fields.update(holding=(False, self.holding), active_arm=1 if self.holding or fields["active_arm"] == 1 else None)
            fields.update(ev)
            self.det.next = EventState(**fields)
            obs = CanonicalObs(
                tick=self.tick, t=self.tick / 50, images={}, state=np.zeros(34, np.float32),
                joint_pos=np.zeros(77, np.float32), joint_vel=np.zeros(77, np.float32),
                gripper_raw=np.zeros(2, np.float32), eef_base_pos=np.stack([[0.5, 0.2, 1.0], self.pos]),
                eef_base_rotmat=np.tile(np.eye(3), (2, 1, 1)), last_action=np.zeros(19, np.float32),
            )
            out = self.orch.update(obs, None, {})
            self.outputs.append(out)
            self.tick += 1
            ev = {k: v for k, v in ev.items() if k not in ("grasp", "release", "drop", "empty_grasp")}
        return out

    def grasp(self):
        self.holding = True
        self.base = dataclasses.replace(self.base, active_arm=1)
        return self.step(grasp=(False, True))

    def release(self):
        self.holding = False
        return self.step(release=(False, True))

    def events(self):
        return [(o.event, o.stage) for o in self.outputs if o.event != OrchestratorEvent.NONE]


def make(cfg=CFG, wipe_script=True):
    det = ScriptedDetectors()
    orch = TaskOrchestrator(cfg, build_stage_specs(cfg), det, StageClassifier(None), 50.0,
                            wipe_script=wipe_script, wipe_target_count=6)
    orch.reset()
    return orch, World(orch)


def test_full_episode_reaches_done():
    orch, w = make()
    # 1 LIFT_LID: grasp the handle, swing to 0.6, release, retract, lid stays open
    w.step(5)
    w.grasp()
    w.base = dataclasses.replace(w.base, lid_angle_kin_peak=0.6)
    w.step(20, lid_tracking=True, lid_angle_kin=0.6)
    w.release()
    out = w.step(30, move=(0.12, 0, 0))
    assert orch.stage == Stage.EXTRACT_COMPONENT, w.events()
    # 2 EXTRACT_COMPONENT: grasp and lift 12 cm → keep queue
    w.base = EventState()
    w.step(3)
    w.grasp()
    w.step(20, move=(0, 0, 0.12))
    assert (OrchestratorEvent.ADVANCE_KEEP_QUEUE, Stage.PACK_COMPONENT) in w.events()
    # 3 PACK_COMPONENT: carry arm inherited, release, retract
    w.step(5)
    w.release()
    w.step(30, move=(0, 0, 0.12))
    assert orch.stage == Stage.WIPE_CHAMBER, w.events()
    # 4 WIPE_CHAMBER: grasp sponge, contact → script, 6 wipes, script done, put back, retract
    w.step(3)
    w.grasp()
    out = w.step(1, in_contact=True)
    assert out.event == OrchestratorEvent.ENTER_SCRIPT and out.active_arm == 1
    w.step(20, wipe_count=6)
    orch.notify_script_done(6)
    assert w.step(1, wipe_count=6).event == OrchestratorEvent.EXIT_SCRIPT
    w.base = dataclasses.replace(w.base, wipe_count=6)
    w.step(5)
    w.release()
    w.step(30, move=(0, 0, 0.12))
    assert orch.stage == Stage.PLACE_REPLACEMENT, w.events()
    # 5 PLACE_REPLACEMENT
    w.base = EventState(active_arm=1)
    w.grasp()
    w.step(10)
    w.release()
    w.step(30, move=(0, 0, 0.12))
    assert orch.stage == Stage.CLOSE_LID, w.events()
    # 6 CLOSE_LID: grasp, swing to 0, release, retract, stays shut
    w.grasp()
    w.base = dataclasses.replace(w.base, lid_angle_kin_min=0.0)
    w.step(10, lid_tracking=True, lid_angle_kin=0.0)
    w.release()
    out = w.step(40, move=(0.12, 0, 0))
    assert orch.stage == Stage.DONE and out.stage == Stage.DONE, w.events()
    assert orch.retries == [0] * 6


def test_lift_lid_waits_for_release_and_clearance():
    orch, w = make()
    w.grasp()
    w.base = dataclasses.replace(w.base, lid_angle_kin_peak=0.7)
    w.step(60, lid_tracking=True, lid_angle_kin=0.7)
    assert orch.stage == Stage.LIFT_LID  # still holding the handle
    w.release()
    w.step(30)  # not moved away
    assert orch.stage == Stage.LIFT_LID


def test_failures_trigger_retries_then_stop_flushing():
    orch, w = make()
    out = w.step(1, empty_grasp=(False, True))
    assert out.event == OrchestratorEvent.RETRY and out.retries == 1
    out = w.step(1, empty_grasp=(True, False))
    assert out.event == OrchestratorEvent.RETRY and out.retries == 2
    out = w.step(1, empty_grasp=(True, False))
    assert out.event == OrchestratorEvent.NONE and orch.retries[0] == 2


def test_lid_released_too_early_is_a_failure():
    orch, w = make()
    w.grasp()
    w.base = dataclasses.replace(w.base, lid_angle_kin_peak=0.2)
    w.step(5, lid_tracking=True, lid_angle_kin=0.2)
    out = w.release()
    assert out.event == OrchestratorEvent.RETRY and "released before" in orch.last_reason


def test_budget_exceeded_retries():
    cfg = dict(CFG, stage_budget_s=[0.5] * 6)
    orch, w = make(cfg)
    w.step(30)
    assert (OrchestratorEvent.RETRY, Stage.LIFT_LID) in w.events()
    assert "budget" in orch.last_reason


def test_disabled_orchestrator_is_single_stage():
    orch, w = make(dict(CFG, enabled=False))
    out = w.step(5, empty_grasp=(True, True))
    assert out.stage == Stage.LIFT_LID and out.event == OrchestratorEvent.NONE


def test_lid_estimate_prefers_vision_when_slipping():
    ev = EventState(lid_tracking=True, lid_angle_kin=0.6, lid_slip=True)
    pred = StagePrediction(np.zeros(7), np.zeros(5), 0, 0, lid_open=0.3)
    assert lid_open_estimate(ev, pred, {"vision": True}) == pytest.approx(0.3)
    assert lid_open_estimate(ev, pred, {"vision": False}) == pytest.approx(0.6)
    assert lid_open_estimate(EventState(), pred, {"vision": False, "lid_last_estimate": 0.5}) == 0.5
    ctx = {"vision": True, "released_after_grasp": True, "dist_from_handle_m": 0.1, "lid_open_stable_ticks": 50,
           "control_hz": 50, "cfg": {}}
    good = StagePrediction(np.zeros(7), np.zeros(5), 0, 0, lid_open=0.6)
    assert stage_complete_lift_lid(EventState(), good, ctx)
    assert not stage_complete_lift_lid(EventState(), dataclasses.replace(good, lid_open=0.3), ctx)


def test_build_stage_specs_scales_to_total_budget():
    specs = build_stage_specs({"stage_budget_s": [100] * 6, "total_budget_s": 300})
    assert sum(s.budget_s for s in specs) == pytest.approx(300)
    with pytest.raises(ValueError):
        build_stage_specs({"min_dwell_s": [1, 2]})


def test_stage_prompts_match_the_training_labels():
    """GR00T knows the stage only from the prompt: it must be the converter's label for that stage."""
    import importlib.util

    from conftest import REPO

    from astra.orchestrator.stage_specs import STAGE_PROMPTS

    spec = importlib.util.spec_from_file_location("cd", REPO / "scripts" / "convert_dataset.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    assert list(STAGE_PROMPTS) == list(cd.STAGE_TASKS)
    specs = build_stage_specs(CFG)
    assert [s.stage for s in specs] == list(Stage)[:6]
    assert [s.prompt for s in specs] == list(cd.STAGE_TASKS)
