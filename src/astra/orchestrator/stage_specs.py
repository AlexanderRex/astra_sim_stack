"""Per-stage specifications for the six ChamberServiceTask stages.

Purpose
-------
Collect, in one table, everything the orchestrator needs to know about each stage: the prompt sent
to GR00T, the sim's success condition (for documentation and offline labelling), the orchestrator's
own completion predicate, time budget, minimum dwell and recovery behaviour.

Sim conditions (``robofinals_tasks/chamber_service_task.py``), each AND the previous stage flag:
    1 LIFT_LID           chamber.is_lid_open(th=0.4)
    2 EXTRACT_PART       old part (Chamber002) not in chamber (th 0.4)
    3 PACK_PART          old part in empty box Box291, no contact (th 0.15)
    4 WIPE               ≥ 3 separate sponge-chamber contacts of ≥ 0.5 s, counted only after stage 3
    5 PLACE_REPLACEMENT  new part (Chamber002_01) in chamber, no contact (th 0.4)
    6 CLOSE_LID          chamber.is_lid_closed()

Inputs:  runtime.yaml ``orchestrator`` section (budgets, dwell times).
Outputs: ``STAGE_SPECS`` list and ``build_stage_specs(cfg)``.
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


# ----------------------------------------------------------------------------- completion predicates
def stage_complete_lift_lid(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """LIFT_LID done: lid released away from the chamber, both grippers empty, classifier agrees.

    Evidence: a release event this stage with the EEF outside the ``chamber`` region, both
    ``holding`` False, ``pred.stage_probs[EXTRACT_PART] > enter_prob`` (lid visibly off).
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
    """WIPE done: ≥ wipe_target_count qualified contacts and the sponge released outside the chamber.

    Evidence: ``ev.wipe_count >= wipe_target_count`` (margin over the sim's 3), then a release with
    region != "chamber" so the sponge cannot block the replacement part.
    """
    raise NotImplementedError("stage_complete_wipe")


def stage_complete_place_replacement(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """PLACE_REPLACEMENT done: new part released above the chamber, retracted, seen seated.

    Evidence: release with region == "chamber", retract, ``pred.scene_ok`` high. A grasp in the
    ``empty_box`` region during this stage means the wrong part → retry.
    """
    raise NotImplementedError("stage_complete_place_replacement")


def stage_complete_close_lid(ev: EventState, pred: StagePrediction, ctx: dict[str, Any]) -> bool:
    """CLOSE_LID done: lid released above the chamber and classifier says closed.

    Evidence: release with region == "chamber", ``pred.stage_probs[DONE]`` high. The env may also
    terminate with success once all six flags are set.
    """
    raise NotImplementedError("stage_complete_close_lid")


# ----------------------------------------------------------------------------- default table
STAGE_SPECS: list[StageSpec] = [
    StageSpec(
        Stage.LIFT_LID,
        "lift the chamber lid and set it aside",
        "chamber.is_lid_open(th=0.4)",
        35.0,
        4.0,
        True,
        stage_complete_lift_lid,
        notes=["sim flag fires at lift; we wait for the release"],
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
        "pick up the sponge and wipe the inside of the chamber",
        ">= 3 separate sponge-chamber contacts >= 0.5 s after stage 3",
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
        "put the lid back on the chamber",
        "chamber.is_lid_closed()",
        45.0,
        4.0,
        True,
        stage_complete_close_lid,
        notes=["lid location hint: EEF pose at the LIFT_LID release"],
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
