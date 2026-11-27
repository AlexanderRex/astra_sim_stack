"""Per-stage specifications for the six ChamberServiceTask stages.

Purpose
-------
Collect, in one table, everything the orchestrator needs to know about each stage: the prompt sent
to GR00T, the sim's success condition (for documentation and offline labelling), the orchestrator's
own completion predicate, time budget, minimum dwell and recovery behaviour.

Sim conditions (``robofinals_tasks/chamber_service_task.py``), each AND the previous stage flag:
    1 LIFT_LID           chamber.is_lid_open(th=0.4): normalized lid angle >= 0.4 (≈ 36°)
    2 EXTRACT_COMPONENT  old part (Chamber002) not in chamber (th 0.4)
    3 PACK_COMPONENT     old part in empty box Box291, no contact (th 0.15)
    4 WIPE_CHAMBER       ≥ 5 separate sponge-chamber contacts of ≥ 0.5 s, counted only after stage 3
    5 PLACE_REPLACEMENT  new part (Chamber002_01) in chamber, no contact (th 0.4)
    6 CLOSE_LID          chamber.is_lid_closed(th=0.005): normalized lid angle <= 0.005 (≈ 0.45°)

The lid is a **hinge** (``Lid001_joint``, -90°..0°, ``core/models/fixtures/chamber.py``), not a free
object: it is swung open and swung shut, always in the same place. Its angle is not observable by the
policy, so stages 1 and 6 use an estimate (``lid_open_estimate``):
- while a hand holds the handle: kinematic estimate from the hand's rotation since the grasp
  (``EventState.lid_angle_kin``; rigid grasp → hand rotation = lid rotation, lid starts closed);
- after the release: vision (``StagePrediction.lid_open`` regressed from the head camera).

Inputs:  runtime.yaml ``orchestrator`` section (budgets, dwell times, lid thresholds).
Outputs: ``STAGE_SPECS`` list and ``build_stage_specs(cfg)``.

Context keys used by the predicates (``ctx``, owned by the orchestrator, reset per stage):
    lid_open_stable_ticks    consecutive ticks with lid_open >= lid_stable_th after release
    lid_closed_stable_ticks  consecutive ticks with lid_open <= lid_closed_vision_th after release
    dist_from_handle_m       distance of the last holding EEF from its pose at the handle release
    released_after_grasp     True once a release followed a grasp of the carrying arm in this stage
    cfg                      the orchestrator config section (thresholds below)
    control_hz               tick rate
    vision                   True when the stage classifier runs (otherwise vision terms are skipped
                             and the stability windows count plain ticks since the release)
    regions                  True when region calibration is loaded (otherwise region tests pass)
    carry_arm                arm that grasped in this stage (or carried over from EXTRACT_COMPONENT)
    lift_m                   rise of the carrying EEF (base-frame z) since its grasp
    release_region           region of the carrying EEF at its release
    dist_from_release_m      distance of the carrying EEF from its pose at the release
    ticks_since_release      ticks since that release (0 before it)
    grasp_regions            regions of the grasps in this stage, in order
    lid_pushed_closed        CLOSE_LID: contact proxy fired with the lid estimate near closed
    lid_last_estimate        last lid estimate (carried across stages; the no-vision fallback)
    released_after_wipes     WIPE_CHAMBER: a release happened once wipe_count reached the target
    script_contacts          WIPE_CHAMBER: press-dwell-lift cycles the scripted wipe completed
    wipe_target_count        detectors.wipe_target_count
    enter_prob               classifier threshold

Failure predicates return a short reason string when the stage attempt has failed (the orchestrator
then retries), else None.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


from astra.types import LID_OPEN_SIM_TH, EventState, Stage, StagePrediction

# Signature of a completion predicate: (events, prediction, stage context) → done?
CompletionPredicate = Callable[[EventState, StagePrediction, dict[str, Any]], bool]
# Signature of a failure predicate: (events, prediction, stage context) → reason or None.
FailurePredicate = Callable[[EventState, StagePrediction, dict[str, Any]], "str | None"]

# Defaults for thresholds read from ctx["cfg"] (runtime.yaml: orchestrator).
DEFAULTS: dict[str, float] = {
    "lid_open_target": 0.55,
    "lid_stable_th": 0.45,
    "lid_closed_vision_th": 0.02,
    "lid_closed_kin_th": 0.03,
    "lid_stable_s": 1.0,
    "lid_retract_m": 0.08,
    "lid_disagree_th": 0.2,
    "lid_disagree_s": 1.0,
    "retract_m": 0.10,
    "extract_lift_m": 0.10,
    "enter_prob": 0.8,
}


def _c(ctx: dict[str, Any], key: str) -> float:
    return float((ctx.get("cfg") or {}).get(key, DEFAULTS[key]))


def _stable_ticks(ctx: dict[str, Any]) -> int:
    return int(round(_c(ctx, "lid_stable_s") * float(ctx.get("control_hz", 50.0))))


def _free(ev: EventState) -> bool:
    return not ev.holding[0] and not ev.holding[1]


def _region_is(ctx: dict[str, Any], region: str | None, name: str) -> bool:
    """Region test that passes when no region calibration is loaded."""
    return not ctx.get("regions") or region == name


def _vision_ok(ctx: dict[str, Any], pred: StagePrediction, next_stage: int | None = None) -> bool:
    """Classifier agrees the stage goal is met (scene_ok, or the next stage); passes without vision."""
    if not ctx.get("vision"):
        return True
    th = float(ctx.get("enter_prob", _c(ctx, "enter_prob")))
    if pred.scene_ok >= th:
        return True
    return next_stage is not None and next_stage < len(pred.stage_probs) and float(pred.stage_probs[next_stage]) >= th


def _empty_or_drop(ev: EventState, ctx: dict[str, Any], what: str) -> str | None:
    if ev.empty_grasp[0] or ev.empty_grasp[1]:
        return "empty grasp"
    arm = ctx.get("carry_arm")
    if arm is not None and ev.drop[arm]:
        return f"dropped {what}"
    return None


@dataclass
class StageSpec:
    """Static description of one stage.

    Attributes:
        stage: which stage.
        prompt: fixed language prompt (GR00T ``language.task``; also logged).
        sim_condition: human-readable sim success condition (documentation / labeller reference).
        budget_s: time budget before a retry is triggered.
        min_dwell_s: minimum time in the stage before an advance is accepted (guards flicker).
        ends_with_release: True if the stage ends with the grippers free; False for EXTRACT_COMPONENT,
            which hands over to PACK_COMPONENT while still carrying (queue is kept, no model switch).
        completion: orchestrator completion predicate (see ``stage_complete_*`` functions).
        failure: failure predicate (see ``stage_failed_*``); a reason triggers a retry.
        pre_grasp_pose: optional (2, 7) base-frame EEF poses to move to before a retry; learned
            from demos, None until available.
        notes: free-text design notes.
    """

    stage: Stage
    prompt: str
    sim_condition: str
    budget_s: float
    min_dwell_s: float
    ends_with_release: bool
    completion: CompletionPredicate | None = None
    pre_grasp_pose: Any = None
    notes: list[str] = field(default_factory=list)
    failure: FailurePredicate | None = None


# ----------------------------------------------------------------------------- lid estimate
def lid_open_estimate(ev: EventState, pred: StagePrediction, ctx: dict[str, Any] | None = None) -> float:
    """Best available normalized lid angle (0 closed … 1 open).

    Functionality:
        - If ``ev.lid_tracking`` (a hand holds the handle): return ``ev.lid_angle_kin``; if vision
          disagrees by more than ~0.2 for a sustained period (``ctx["lid_disagree_ticks"]``), or the
          slip check fires, trust vision instead.
        - Otherwise: return ``pred.lid_open`` (vision), or without vision the last estimate
          (``ctx["lid_last_estimate"]``; the lid stays where it was let go).

    Args:
        ev: events of this tick.
        pred: classifier prediction of this tick.
        ctx: stage context (``vision``, ``lid_disagree_ticks``, ``lid_last_estimate``); None = vision on.

    Returns:
        normalized lid angle estimate in [0, 1].
    """
    ctx = ctx if ctx is not None else {"vision": True}
    vision = bool(ctx.get("vision", True))
    if ev.lid_tracking and ev.lid_angle_kin is not None:
        if vision:
            disagree = int(ctx.get("lid_disagree_ticks", 0))
            limit = int(round(_c(ctx, "lid_disagree_s") * float(ctx.get("control_hz", 50.0))))
            if ev.lid_slip or disagree >= limit:
                return float(pred.lid_open)
        return float(ev.lid_angle_kin)
    if vision:
        return float(pred.lid_open)
    return float(ctx.get("lid_last_estimate", ev.lid_angle_kin_peak))


# ----------------------------------------------------------------------------- completion predicates
def stage_complete_lift_lid(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """LIFT_LID done: lid swung open with margin, handle released, hand clear, lid stays open.

    The sim flag latches as soon as the angle reaches 0.4, even while the hand still holds the
    handle; the orchestrator waits until the robot is free and the lid will not fall back shut
    (a hinged lid released before its balance point swings closed and blocks stage 2).

    All of the following must hold (then debounced by the orchestrator):
        1. Opened enough: ``max(ev.lid_angle_kin_peak, pred.lid_open) >= cfg.lid_open_target``
           (0.55 ≈ 50°, margin over the sim's 0.4).
        2. Released: ``ctx["released_after_grasp"]`` and both ``ev.holding`` False.
        3. Hand clear: ``ctx["dist_from_handle_m"] >= cfg.lid_retract_m`` (0.08 m).
        4. Stays open: ``ctx["lid_open_stable_ticks"]`` ≥ ``cfg.lid_stable_s`` × control rate with
           ``pred.lid_open >= cfg.lid_stable_th`` (0.45) — vision, since the hand has let go.

    Failure signatures (handled by the orchestrator as RETRY, never skip):
        - empty grasp on the handle (gripper fully closed) → re-approach;
        - drop while ``lid_angle_kin < 0.4`` → lid slipped back;
        - released with vision angle < target → re-grasp and push further;
        - vision angle falls below 0.45 inside the stability window → lid swung shut.

    Args:
        ev: events of this tick (holding, release, lid tracking).
        pred: classifier prediction (lid_open).
        ctx: stage context (see module docstring).

    Returns:
        True if the stage is complete on this tick.
    """
    vision = bool(ctx.get("vision"))
    opened = max(ev.lid_angle_kin_peak, pred.lid_open if vision else 0.0) >= _c(ctx, "lid_open_target")
    released = bool(ctx.get("released_after_grasp")) and _free(ev)
    clear = float(ctx.get("dist_from_handle_m", 0.0)) >= _c(ctx, "lid_retract_m")
    stays_open = int(ctx.get("lid_open_stable_ticks", 0)) >= _stable_ticks(ctx)
    return opened and released and clear and stays_open


def stage_failed_lift_lid(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> str | None:
    """Empty grasp on the handle, lid dropped below 0.4, or released and not open enough."""
    if ev.empty_grasp[0] or ev.empty_grasp[1]:
        return "empty grasp"
    arm = ctx.get("carry_arm")
    if arm is not None and ev.drop[arm] and ev.lid_angle_kin_peak < LID_OPEN_SIM_TH:
        return "lid slipped from the hand"
    if ctx.get("released_after_grasp") and _free(ev):
        waited = int(ctx.get("ticks_since_release", 0)) >= _stable_ticks(ctx)
        if ctx.get("vision"):
            if waited and pred.lid_open < _c(ctx, "lid_stable_th"):
                return "lid not open after release"
        elif ev.lid_angle_kin_peak < _c(ctx, "lid_open_target"):
            return "released before the lid was open"
    return None


def stage_complete_extract_part(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """EXTRACT_COMPONENT done: part held and lifted clear of the chamber rim (no release).

    Evidence: ``holding`` True on the carrying arm, EEF z above rim height + margin (region test),
    classifier agrees. Transition emits ADVANCE_KEEP_QUEUE (continuous carry into PACK_COMPONENT).
    Without region calibration the rim test is a rise of ``extract_lift_m`` since the grasp.
    """
    arm = ctx.get("carry_arm")
    if arm is None or not ev.holding[arm]:
        return False
    if ctx.get("regions"):
        clear = ev.arm_regions[arm] != "chamber" and float(ctx.get("lift_m", 0.0)) >= 0.5 * _c(ctx, "extract_lift_m")
    else:
        clear = float(ctx.get("lift_m", 0.0)) >= _c(ctx, "extract_lift_m")
    return clear and _vision_ok(ctx, pred, int(Stage.PACK_COMPONENT))


def stage_failed_extract_part(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> str | None:
    """Empty grasp or the part dropped."""
    return _empty_or_drop(ev, ctx, "part")


def _released_and_clear(ev: EventState, ctx: dict[str, Any], region: str) -> bool:
    return (
        bool(ctx.get("released_after_grasp"))
        and _free(ev)
        and _region_is(ctx, ctx.get("release_region"), region)
        and float(ctx.get("dist_from_release_m", 0.0)) >= _c(ctx, "retract_m")
    )


def stage_complete_pack_part(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """PACK_COMPONENT done: released above the empty box, retracted ≥ 10 cm, part seen in the box.

    Evidence: release event with region == "empty_box", retract distance, ``pred.scene_ok`` high.
    The sim requires no robot contact, so the retract is part of the condition.
    """
    return _released_and_clear(ev, ctx, "empty_box") and _vision_ok(ctx, pred, int(Stage.WIPE_CHAMBER))


def stage_failed_pack_part(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> str | None:
    """Part dropped on the way, or released outside the box."""
    reason = _empty_or_drop(ev, ctx, "part")
    if reason:
        return reason
    if ctx.get("regions") and ctx.get("released_after_grasp") and ctx.get("release_region") != "empty_box":
        return f"released over {ctx.get('release_region')}, not the empty box"
    return None


def stage_complete_wipe(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """WIPE_CHAMBER done: ≥ wipe_target_count qualified contacts and the sponge put back on the table.

    Evidence: ``ev.wipe_count >= wipe_target_count`` (margin over the sim's 5), then a release with
    region == "table". The task rules require the sponge back on the tabletop (any position); left
    in the chamber or dropped on the floor does not count, and in the chamber it would also block
    the replacement part. The sim flag does not test this, so the orchestrator has to.
    Without a contact proxy (no FK) the scripted wipe's completed presses count instead.
    """
    target = int(ctx.get("wipe_target_count", 6))
    count = ev.wipe_count if ev.contact_available else max(ev.wipe_count, int(ctx.get("script_contacts", 0)))
    if count < target or not ctx.get("released_after_wipes"):
        return False
    return _released_and_clear(ev, ctx, "table")


def stage_failed_wipe(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> str | None:
    """Sponge missed or dropped."""
    return _empty_or_drop(ev, ctx, "sponge")


def stage_complete_place_replacement(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """PLACE_REPLACEMENT done: new part released above the chamber, retracted, seen seated.

    Evidence: release with region == "chamber", retract, ``pred.scene_ok`` high. A grasp in the
    ``empty_box`` region during this stage means the wrong part → retry.
    """
    return _released_and_clear(ev, ctx, "chamber") and _vision_ok(ctx, pred, int(Stage.CLOSE_LID))


def stage_failed_place_replacement(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> str | None:
    """Empty grasp, drop, the old part picked from the empty box, or released outside the chamber."""
    reason = _empty_or_drop(ev, ctx, "part")
    if reason:
        return reason
    if ctx.get("regions"):
        if "empty_box" in (ctx.get("grasp_regions") or []):
            return "grasped in the empty box (old part)"
        if ctx.get("released_after_grasp") and ctx.get("release_region") != "chamber":
            return f"released over {ctx.get('release_region')}, not the chamber"
    return None


def stage_complete_close_lid(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """CLOSE_LID done: lid swung fully shut, handle released, and it stays shut.

    The sim requires a normalized angle <= 0.005 (≈ 0.45°), i.e. essentially flat. Strategy for the
    policy: swing the lid down by its handle (always in the same place, it is hinged), push it fully
    down or release it just above closed and let gravity seat it.

    All of the following must hold (then debounced by the orchestrator):
        1. Swung shut while held: ``ev.lid_angle_kin`` (tracked from the stage-6 handle grasp, relative
           to the open angle at grasp) reached ≈ 0, or the hand pushed down until the contact proxy
           fired with the lid near closed.
        2. Released and hand clear (``dist_from_handle_m >= lid_retract_m``).
        3. Vision confirms shut: ``ctx["lid_closed_stable_ticks"]`` ≥ ``cfg.lid_stable_s`` × control
           rate with ``pred.lid_open <= cfg.lid_closed_vision_th`` and/or ``pred.stage_probs[DONE]``
           high.
    The env may also terminate with success once all six flags are set; that ends the episode.

    Failure signatures → RETRY: lid resting slightly open (vision angle above threshold after the
    release) → re-grasp, lift a little, press down again.

    Args:
        ev: events of this tick.
        pred: classifier prediction (lid_open, stage_probs).
        ctx: stage context.

    Returns:
        True if the stage is complete on this tick.
    """
    swung_shut = ev.lid_angle_kin_min <= _c(ctx, "lid_closed_kin_th") or bool(ctx.get("lid_pushed_closed"))
    released = bool(ctx.get("released_after_grasp")) and _free(ev)
    clear = float(ctx.get("dist_from_handle_m", 0.0)) >= _c(ctx, "lid_retract_m")
    stays_shut = int(ctx.get("lid_closed_stable_ticks", 0)) >= _stable_ticks(ctx)
    if ctx.get("vision"):
        # vision alone may confirm when the hand pushed the lid shut without a clean kinematic track
        swung_shut = swung_shut or stays_shut
    return swung_shut and released and clear and stays_shut


def stage_failed_close_lid(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> str | None:
    """Empty grasp on the handle, or released and the lid is resting open."""
    if ev.empty_grasp[0] or ev.empty_grasp[1]:
        return "empty grasp"
    if ctx.get("released_after_grasp") and _free(ev):
        waited = int(ctx.get("ticks_since_release", 0)) >= 2 * _stable_ticks(ctx)
        if ctx.get("vision"):
            done_p = float(pred.stage_probs[int(Stage.DONE)]) if len(pred.stage_probs) > int(Stage.DONE) else 0.0
            if waited and pred.lid_open > _c(ctx, "lid_closed_vision_th") and done_p < _c(ctx, "enter_prob"):
                return "lid resting open after release"
        elif ev.lid_angle_kin_min > _c(ctx, "lid_closed_kin_th") and not ctx.get("lid_pushed_closed"):
            return "released before the lid was shut"
    return None


# ----------------------------------------------------------------------------- default table
# Language prompt per stage, exactly as the GR00T training data labels it (meta/tasks.jsonl of the
# dataset, written from STAGE_TASKS in scripts/convert_dataset.py). GR00T knows the stage only from
# this text, so it must match the training labels word for word; tests check it against the converter.
STAGE_PROMPTS: tuple[str, ...] = (
    "Lift the chamber top",
    "Extract the old part from the chamber",
    "Pack the old part into the box",
    "Wipe the chamber",
    "Unbox the new part and place it in the chamber",
    "Close and lock the chamber",
)


STAGE_SPECS: list[StageSpec] = [
    StageSpec(
        Stage.LIFT_LID,
        STAGE_PROMPTS[0],
        "chamber.is_lid_open(th=0.4): hinge angle >= 36°",
        35.0,
        4.0,
        True,
        stage_complete_lift_lid,
        notes=[
            "hinged lid: swing open by the handle, never set aside",
            "sim flag latches at 0.4 while still held; we wait for release + lid staying open",
        ],
        failure=stage_failed_lift_lid,
    ),
    StageSpec(
        Stage.EXTRACT_COMPONENT,
        STAGE_PROMPTS[1],
        "old part not in chamber (th 0.4)",
        30.0,
        3.0,
        False,
        stage_complete_extract_part,
        notes=["continuous carry into PACK_COMPONENT"],
        failure=stage_failed_extract_part,
    ),
    StageSpec(
        Stage.PACK_COMPONENT,
        STAGE_PROMPTS[2],
        "old part in Box291, no contact (th 0.15)",
        30.0,
        3.0,
        True,
        stage_complete_pack_part,
        failure=stage_failed_pack_part,
    ),
    StageSpec(
        Stage.WIPE_CHAMBER,
        STAGE_PROMPTS[3],
        ">= 5 separate sponge-chamber contacts >= 0.5 s after stage 3",
        70.0,
        8.0,
        True,
        stage_complete_wipe,
        notes=["sponge pose randomised ±5 cm, random yaw", "scripted wipe primitive inside"],
        failure=stage_failed_wipe,
    ),
    StageSpec(
        Stage.PLACE_REPLACEMENT,
        STAGE_PROMPTS[4],
        "new part in chamber, no contact (th 0.4)",
        50.0,
        5.0,
        True,
        stage_complete_place_replacement,
        failure=stage_failed_place_replacement,
    ),
    StageSpec(
        Stage.CLOSE_LID,
        STAGE_PROMPTS[5],
        "chamber.is_lid_closed(th=0.005): hinge angle <= 0.45°",
        45.0,
        4.0,
        True,
        stage_complete_close_lid,
        notes=[
            "hinged lid: handle is always in the same place",
            "fully shut is required; push down or release just above closed",
        ],
        failure=stage_failed_close_lid,
    ),
]


def build_stage_specs(cfg: dict[str, Any]) -> list[StageSpec]:
    """Copy ``STAGE_SPECS`` and override budgets / dwell times from ``runtime.yaml: orchestrator``.

    Functionality:
        Apply ``stage_budget_s`` and ``min_dwell_s`` lists (length 6), scale budgets so they sum to
        at most ``total_budget_s``.

    Args:
        cfg: orchestrator config section.

    Returns:
        list of 6 StageSpec in Stage order.

    Raises:
        ValueError: a list does not have 6 entries.
    """
    specs = [copy.copy(s) for s in STAGE_SPECS]
    budgets = cfg.get("stage_budget_s")
    dwells = cfg.get("min_dwell_s")
    for name, values in (("stage_budget_s", budgets), ("min_dwell_s", dwells)):
        if values is not None and len(values) != len(specs):
            raise ValueError(f"orchestrator.{name} needs {len(specs)} values, got {len(values)}")
    if budgets is not None:
        for spec, b in zip(specs, budgets):
            spec.budget_s = float(b)
    if dwells is not None:
        for spec, d in zip(specs, dwells):
            spec.min_dwell_s = float(d)
    total = cfg.get("total_budget_s")
    if total is not None:
        s_sum = sum(spec.budget_s for spec in specs)
        if s_sum > float(total) > 0:
            k = float(total) / s_sum
            for spec in specs:
                spec.budget_s *= k
    return specs
