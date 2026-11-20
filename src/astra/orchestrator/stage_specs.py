"""Per-stage specifications for the six ChamberServiceTask stages.

Purpose
-------
Collect, in one table, everything the orchestrator needs to know about each stage: the prompt sent
to GR00T, the sim's success condition (for documentation and offline labelling), the orchestrator's
own completion predicate, time budget, minimum dwell and recovery behaviour.

Sim conditions (``robofinals_tasks/chamber_service_task.py``), each AND the previous stage flag:
    1 LIFT_LID           chamber.is_lid_open(th=0.4): normalized lid angle >= 0.4 (≈ 36°)
    2 EXTRACT_PART       old part (Chamber002) not in chamber (th 0.4)
    3 PACK_PART          old part in empty box Box291, no contact (th 0.15)
    4 WIPE               ≥ 5 separate sponge-chamber contacts of ≥ 0.5 s, counted only after stage 3
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
    released_after_grasp     True once a release followed a handle grasp in this stage
    cfg                      the orchestrator config section (thresholds below)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from astra.types import EventState, Stage, StagePrediction

# Signature of a completion predicate: (events, prediction, stage context) → done?
CompletionPredicate = Callable[[EventState, StagePrediction, dict[str, Any]], bool]


@dataclass
class StageSpec:
    """Static description of one stage.

    Attributes:
        stage: which stage.
        prompt: fixed language prompt (GR00T ``language.task``; also logged).
        sim_condition: human-readable sim success condition (documentation / labeller reference).
        budget_s: time budget before a retry is triggered.
        min_dwell_s: minimum time in the stage before an advance is accepted (guards flicker).
        ends_with_release: True if the stage ends with the grippers free; False for EXTRACT_PART,
            which hands over to PACK_PART while still carrying (queue is kept, no model switch).
        completion: orchestrator completion predicate (see ``stage_complete_*`` functions).
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


# ----------------------------------------------------------------------------- lid estimate
def lid_open_estimate(ev: EventState, pred: StagePrediction) -> float:
    """Best available normalized lid angle (0 closed … 1 open).

    Functionality:
        - If ``ev.lid_tracking`` (a hand holds the handle): return ``ev.lid_angle_kin``; if vision
          disagrees by more than ~0.2 for a sustained period, treat the grasp as slipping and trust
          vision instead.
        - Otherwise: return ``pred.lid_open`` (vision).

    Args:
        ev: events of this tick.
        pred: classifier prediction of this tick.

    Returns:
        normalized lid angle estimate in [0, 1].
    """
    raise NotImplementedError("lid_open_estimate")


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
    raise NotImplementedError("stage_complete_lift_lid")


def stage_complete_extract_part(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """EXTRACT_PART done: part held and lifted clear of the chamber rim (no release).

    Evidence: ``holding`` True on the carrying arm, EEF z above rim height + margin (region test),
    classifier agrees. Transition emits ADVANCE_KEEP_QUEUE (continuous carry into PACK_PART).
    """
    raise NotImplementedError("stage_complete_extract_part")


def stage_complete_pack_part(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """PACK_PART done: released above the empty box, retracted ≥ 10 cm, part seen in the box.

    Evidence: release event with region == "empty_box", retract distance, ``pred.scene_ok`` high.
    The sim requires no robot contact, so the retract is part of the condition.
    """
    raise NotImplementedError("stage_complete_pack_part")


def stage_complete_wipe(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """WIPE done: ≥ wipe_target_count qualified contacts and the sponge put back on the table.

    Evidence: ``ev.wipe_count >= wipe_target_count`` (margin over the sim's 5), then a release with
    region == "table". The task rules require the sponge back on the tabletop (any position); left
    in the chamber or dropped on the floor does not count, and in the chamber it would also block
    the replacement part. The sim flag does not test this, so the orchestrator has to.
    """
    raise NotImplementedError("stage_complete_wipe")


def stage_complete_place_replacement(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """PLACE_REPLACEMENT done: new part released above the chamber, retracted, seen seated.

    Evidence: release with region == "chamber", retract, ``pred.scene_ok`` high. A grasp in the
    ``empty_box`` region during this stage means the wrong part → retry.
    """
    raise NotImplementedError("stage_complete_place_replacement")


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
    raise NotImplementedError("stage_complete_close_lid")


# ----------------------------------------------------------------------------- default table
STAGE_SPECS: list[StageSpec] = [
    StageSpec(
        Stage.LIFT_LID,
        "open the chamber lid",
        "chamber.is_lid_open(th=0.4): hinge angle >= 36°",
        35.0,
        4.0,
        True,
        stage_complete_lift_lid,
        notes=[
            "hinged lid: swing open by the handle, never set aside",
            "sim flag latches at 0.4 while still held; we wait for release + lid staying open",
        ],
    ),
    StageSpec(
        Stage.EXTRACT_PART,
        "pick the old part out of the chamber",
        "old part not in chamber (th 0.4)",
        30.0,
        3.0,
        False,
        stage_complete_extract_part,
        notes=["continuous carry into PACK_PART"],
    ),
    StageSpec(
        Stage.PACK_PART,
        "place the old part into the empty box",
        "old part in Box291, no contact (th 0.15)",
        30.0,
        3.0,
        True,
        stage_complete_pack_part,
    ),
    StageSpec(
        Stage.WIPE,
        "pick up the sponge, wipe the inside of the chamber and put the sponge back on the table",
        ">= 5 separate sponge-chamber contacts >= 0.5 s after stage 3",
        70.0,
        8.0,
        True,
        stage_complete_wipe,
        notes=["sponge pose randomised ±5 cm, random yaw", "scripted wipe primitive inside"],
    ),
    StageSpec(
        Stage.PLACE_REPLACEMENT,
        "take the new part from the box and place it into the chamber",
        "new part in chamber, no contact (th 0.4)",
        50.0,
        5.0,
        True,
        stage_complete_place_replacement,
    ),
    StageSpec(
        Stage.CLOSE_LID,
        "close the chamber lid",
        "chamber.is_lid_closed(th=0.005): hinge angle <= 0.45°",
        45.0,
        4.0,
        True,
        stage_complete_close_lid,
        notes=[
            "hinged lid: handle is always in the same place",
            "fully shut is required; push down or release just above closed",
        ],
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
    """
    raise NotImplementedError("build_stage_specs")
