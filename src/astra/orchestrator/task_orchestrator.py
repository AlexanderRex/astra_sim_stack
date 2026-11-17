"""Task orchestrator (layer 3): forward-only stage state machine.

Purpose
-------
Fuse the learned classifier and the rule-based detectors into one decision per tick: the current
stage and phase, the prompt, and whether to advance or retry. Stages never go backwards and are
never skipped (the sim chains stage flags, so a skipped stage forfeits every later one).

Decision logic per tick
-----------------------
1. ``events = detectors.update(obs, last_action)``; ``pred = classifier.predict(...)``.
2. If ``spec.completion(events, pred, ctx)`` holds → ``hold_counter += 1`` else reset to 0.
3. If ``hold_counter >= debounce_ticks`` and time in stage ≥ ``min_dwell_s`` → advance:
   ADVANCE (grippers free) or ADVANCE_KEEP_QUEUE (EXTRACT_PART → PACK_PART while carrying).
4. Else if stalled (progress flat for ``stall_seconds`` or same phase repeated) or the stage budget
   is exceeded → RETRY (up to ``max_retries_per_stage``; after that keep trying without the
   pre-grasp reset, since skipping gains nothing).
5. WIPE stage: emit ENTER_SCRIPT once the sponge is grasped and inside the chamber, EXIT_SCRIPT when
   the scripted primitive reports done; start/stop the detectors' wipe counter.

Inputs:  CanonicalObs, last SimAction, classifier inputs (from the adapter), sim time.
Outputs: OrchestratorOutput (stage, phase, prompt, event, budget left, retries).
"""

from __future__ import annotations

from typing import Any

from astra.orchestrator.event_detectors import EventDetectors
from astra.orchestrator.stage_classifier import StageClassifier
from astra.orchestrator.stage_specs import StageSpec
from astra.types import (
    CONTROL_HZ,
    CanonicalObs,
    EventState,
    OrchestratorEvent,
    OrchestratorOutput,
    Phase,
    SimAction,
    Stage,
    StagePrediction,
)


class TaskOrchestrator:
    """Forward-only stage machine.

    Attributes:
        cfg: ``runtime.yaml: orchestrator`` section.
        specs: six StageSpec in Stage order.
        detectors: rule-based event detectors.
        classifier: learned stage classifier.
        enabled: False → always Stage.LIFT_LID with no events (single-policy mode).
        control_hz: tick rate.
        debounce_ticks: agreement length before an advance.
        enter_prob: classifier threshold used by completion predicates.
        stall_ticks: ticks without progress before a retry.
        max_retries: retries per stage before falling back to plain re-attempts.
        stage: current stage.
        phase: current phase estimate.
        stage_start_tick: tick at which the current stage began.
        hold_counter: consecutive ticks the completion predicate held.
        retries: retries used per stage (6,).
        best_progress: highest progress seen in this stage (stall detection).
        best_progress_tick: tick when ``best_progress`` last increased.
        in_script: True while the scripted wiping primitive is in control.
        context: per-episode memory passed to predicates (e.g. lid release pose, carrying arm).
        last_events / last_prediction: most recent inputs, kept for logging.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        specs: list[StageSpec],
        detectors: EventDetectors,
        classifier: StageClassifier,
        control_hz: float = CONTROL_HZ,
    ) -> None:
        self.cfg = cfg
        self.specs = specs
        self.detectors = detectors
        self.classifier = classifier
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.control_hz = control_hz
        self.debounce_ticks: int = int(cfg.get("debounce_ticks", 12))
        self.enter_prob: float = float(cfg.get("enter_prob", 0.8))
        self.stall_ticks: int = int(round(float(cfg.get("stall_seconds", 8.0)) * control_hz))
        self.max_retries: int = int(cfg.get("max_retries_per_stage", 3))
        self.stage: Stage = Stage.LIFT_LID
        self.phase: Phase = Phase.APPROACH
        self.stage_start_tick: int = 0
        self.hold_counter: int = 0
        self.retries: list[int] = [0] * 6
        self.best_progress: float = 0.0
        self.best_progress_tick: int = 0
        self.in_script: bool = False
        self.context: dict[str, Any] = {}
        self.last_events: EventState | None = None
        self.last_prediction: StagePrediction | None = None

    def reset(self) -> None:
        """Episode start: stage = LIFT_LID, counters cleared, detectors and classifier reset."""
        raise NotImplementedError("TaskOrchestrator.reset")

    def update(
        self, obs: CanonicalObs, last_action: SimAction | None, classifier_inputs: dict[str, Any]
    ) -> OrchestratorOutput:
        """Run the decision logic for one tick (see module docstring).

        Args:
            obs: canonical observation of this tick.
            last_action: sim action sent on the previous tick.
            classifier_inputs: output of ``ObservationAdapter.classifier_input()``.

        Returns:
            OrchestratorOutput for this tick.
        """
        raise NotImplementedError("TaskOrchestrator.update")

    def notify_script_done(self) -> None:
        """Called by the router when the scripted wiping primitive finishes; next update emits EXIT_SCRIPT."""
        raise NotImplementedError("TaskOrchestrator.notify_script_done")

    # ------------------------------------------------------------------ internals
    def _advance(self, keep_queue: bool) -> OrchestratorEvent:
        """Move to the next stage, reset per-stage counters, update the detectors' wipe counting.

        Args:
            keep_queue: True for the EXTRACT_PART → PACK_PART carry handover.

        Returns:
            ADVANCE or ADVANCE_KEEP_QUEUE.
        """
        raise NotImplementedError("TaskOrchestrator._advance")

    def _retry(self) -> OrchestratorEvent:
        """Register a retry of the current stage (counter++, restart stall timer).

        Returns:
            RETRY.
        """
        raise NotImplementedError("TaskOrchestrator._retry")

    def _is_stalled(self, tick: int, pred: StagePrediction) -> bool:
        """True if progress has not increased for ``stall_ticks`` or the phase keeps repeating."""
        raise NotImplementedError("TaskOrchestrator._is_stalled")

    def _budget_left_s(self, tick: int) -> float:
        """Remaining budget of the current stage in seconds (negative when exceeded)."""
        raise NotImplementedError("TaskOrchestrator._budget_left_s")
