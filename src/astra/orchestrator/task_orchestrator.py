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
   ADVANCE (grippers free) or ADVANCE_KEEP_QUEUE (EXTRACT_COMPONENT → PACK_COMPONENT while carrying).
4. Else if stalled (progress flat for ``stall_seconds`` or same phase repeated) or the stage budget
   is exceeded → RETRY (up to ``max_retries_per_stage``; after that keep trying without the
   pre-grasp reset, since skipping gains nothing).
5. WIPE_CHAMBER stage: emit ENTER_SCRIPT once the sponge is grasped and inside the chamber, EXIT_SCRIPT when
   the scripted primitive reports done; start/stop the detectors' wipe counter.
6. LIFT_LID / CLOSE_LID: on entering the stage call ``detectors.enable_lid_tracking(initial)``
   (0.0 for LIFT_LID, the vision ``lid_open`` for CLOSE_LID); on leaving call
   ``disable_lid_tracking()``. Every tick update the lid context keys used by the predicates
   (``lid_open_stable_ticks``, ``lid_closed_stable_ticks``, ``dist_from_handle_m``,
   ``released_after_grasp``; see ``stage_specs``). The lid is a hinge, so its angle is estimated
   kinematically while held and visually after release.

Failure predicates (``StageSpec.failure``: empty grasp, drop, lid resting in the wrong place) also
trigger a retry. Without a classifier checkpoint vision terms are skipped and stalls are not
detected (progress is unknown); only the budget and failure predicates trigger retries.

Inputs:  CanonicalObs, last SimAction, classifier inputs (from the adapter), sim time.
Outputs: OrchestratorOutput (stage, phase, prompt, event, budget left, retries).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from astra.orchestrator.event_detectors import EventDetectors
from astra.orchestrator.stage_classifier import StageClassifier
from astra.orchestrator.stage_specs import StageSpec, lid_open_estimate
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

log = logging.getLogger(__name__)

LID_STAGES = (Stage.LIFT_LID, Stage.CLOSE_LID)


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
        wipe_script: True when the router has a scripted wipe for the WIPE_CHAMBER stage.
        wipe_target_count: qualified contacts needed in WIPE_CHAMBER (``detectors.wipe_target_count``).
        script_enter_hold_ticks: without regions and contact proxy, hold the sponge this long in
            WIPE_CHAMBER before handing over to the script.
        stage: current stage.
        phase: current phase estimate.
        stage_start_tick: tick at which the current stage began.
        attempt_start_tick: tick at which the current attempt (stage start or last retry) began.
        hold_counter: consecutive ticks the completion predicate held.
        retries: retries used per stage (6,).
        best_progress: highest progress seen in this stage (stall detection).
        best_progress_tick: tick when ``best_progress`` last increased.
        in_script: True while the scripted wiping primitive is in control.
        script_done: the script finished in this attempt (EXIT_SCRIPT emitted or pending).
        context: per-stage memory passed to predicates (lid stability counters, distance from the
            handle since release, carrying arm, ...); keys documented in ``stage_specs``.
        last_events / last_prediction: most recent inputs, kept for logging.
        last_reason: why the last ADVANCE / RETRY happened (logging).
        transitions: (tick, event, stage, reason) of every non-NONE event this episode.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        specs: list[StageSpec],
        detectors: EventDetectors,
        classifier: StageClassifier,
        control_hz: float = CONTROL_HZ,
        wipe_script: bool = True,
        wipe_target_count: int = 6,
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
        self.wipe_script = wipe_script
        self.wipe_target_count = int(wipe_target_count)
        self.script_enter_hold_ticks: int = int(round(float(cfg.get("script_enter_hold_s", 2.0)) * control_hz))
        self.stage: Stage = Stage.LIFT_LID
        self.phase: Phase = Phase.APPROACH
        self.stage_start_tick: int = 0
        self.attempt_start_tick: int = 0
        self.hold_counter: int = 0
        self.retries: list[int] = [0] * 6
        self.best_progress: float = 0.0
        self.best_progress_tick: int = 0
        self.in_script: bool = False
        self.script_done: bool = False
        self._script_exit_pending: bool = False
        self._entered: bool = False
        self._lid_memory: float = 0.0
        self.context: dict[str, Any] = {}
        self.last_events: EventState | None = None
        self.last_prediction: StagePrediction | None = None
        self.last_reason: str = ""
        self.transitions: list[tuple[int, str, str, str]] = []
        self._tick: int = 0

    @property
    def vision(self) -> bool:
        """True when the classifier produces real predictions."""
        return bool(self.classifier.enabled and self.classifier.model is not None)

    def reset(self) -> None:
        """Episode start: stage = LIFT_LID, counters cleared, detectors and classifier reset."""
        self.detectors.reset()
        self.classifier.reset()
        self.stage = Stage.LIFT_LID
        self.phase = Phase.APPROACH
        self.stage_start_tick = 0
        self.attempt_start_tick = 0
        self.hold_counter = 0
        self.retries = [0] * 6
        self.best_progress = 0.0
        self.best_progress_tick = 0
        self.in_script = False
        self.script_done = False
        self._script_exit_pending = False
        self._entered = False
        self._lid_memory = 0.0
        self.context = {}
        self.last_events = None
        self.last_prediction = None
        self.last_reason = ""
        self.transitions = []
        self._tick = 0

    # ------------------------------------------------------------------ per tick
    def update(
        self,
        obs: CanonicalObs,
        last_action: SimAction | None,
        classifier_inputs: dict[str, Any] | Callable[[], dict[str, Any]],
    ) -> OrchestratorOutput:
        """Run the decision logic for one tick (see module docstring).

        Args:
            obs: canonical observation of this tick.
            last_action: sim action sent on the previous tick.
            classifier_inputs: output of ``ObservationAdapter.classifier_input()``, or the method
                itself (then the window is built only on classifier inference ticks).

        Returns:
            OrchestratorOutput for this tick.
        """
        tick = obs.tick
        self._tick = tick
        ev = self.detectors.update(obs, last_action)
        pred = self.classifier.predict(tick, classifier_inputs) if self.enabled else self._no_prediction()
        self.last_events, self.last_prediction = ev, pred

        if not self.enabled:
            return OrchestratorOutput(
                stage=Stage.LIFT_LID,
                phase=Phase.APPROACH,
                prompt=self.specs[0].prompt,
                event=OrchestratorEvent.NONE,
                budget_left_s=float("inf"),
                retries=0,
                active_arm=ev.active_arm,
            )

        if self.stage == Stage.DONE:
            return self._output(OrchestratorEvent.NONE, ev)

        if not self._entered:
            self._enter_stage(pred)

        self._update_stage_context(obs, ev, pred)
        self._update_phase(ev, pred)
        spec = self.specs[int(self.stage)]

        # script hand-over in WIPE_CHAMBER
        if self._script_exit_pending:
            self._script_exit_pending = False
            self.in_script = False
            self.script_done = True
            return self._emit(OrchestratorEvent.EXIT_SCRIPT, ev, "scripted wipe finished")

        # failures → retry (also leaves the script)
        reason = spec.failure(ev, pred, self.context) if spec.failure is not None else None
        if reason:
            return self._emit(self._retry(reason), ev, reason)

        if self.stage == Stage.WIPE_CHAMBER and self.wipe_script and not self.in_script and not self.script_done:
            if self._ready_for_script(ev):
                self.in_script = True
                return self._emit(OrchestratorEvent.ENTER_SCRIPT, ev, "sponge in the chamber")

        # completion → debounced advance
        done = bool(spec.completion(ev, pred, self.context)) if spec.completion is not None else False
        self.hold_counter = self.hold_counter + 1 if done else 0
        in_stage_s = (tick - self.stage_start_tick) / self.control_hz
        if self.hold_counter >= self.debounce_ticks and in_stage_s >= spec.min_dwell_s:
            keep = not spec.ends_with_release
            event = self._advance(keep_queue=keep)
            return self._emit(event, ev, "stage complete")

        # stall / budget → retry
        if self.retries[int(self.stage)] < self.max_retries:
            if self._budget_left_s(tick) < 0:
                return self._emit(self._retry("stage budget exceeded"), ev, "stage budget exceeded")
            if self._is_stalled(tick, pred):
                return self._emit(self._retry("no progress"), ev, "no progress")
        return self._output(OrchestratorEvent.NONE, ev)

    def notify_script_done(self, contacts: int | None = None) -> None:
        """Called by the router when the scripted wiping primitive finishes; next update emits EXIT_SCRIPT.

        Args:
            contacts: press-dwell-lift cycles the script completed (credited when no contact proxy).
        """
        if self.in_script:
            self._script_exit_pending = True
            if contacts is not None:
                self.context["script_contacts"] = int(contacts)

    # ------------------------------------------------------------------ internals
    def _no_prediction(self) -> StagePrediction:
        from astra.orchestrator.stage_classifier import uninformative_prediction

        return uninformative_prediction()

    def _output(self, event: OrchestratorEvent, ev: EventState) -> OrchestratorOutput:
        idx = min(int(self.stage), len(self.specs) - 1)
        return OrchestratorOutput(
            stage=self.stage,
            phase=self.phase,
            prompt=self.specs[idx].prompt,
            event=event,
            budget_left_s=self._budget_left_s(self._tick) if self.stage != Stage.DONE else 0.0,
            retries=self.retries[idx],
            active_arm=ev.active_arm,
        )

    def _emit(self, event: OrchestratorEvent, ev: EventState, reason: str) -> OrchestratorOutput:
        self.last_reason = reason
        self.transitions.append((self._tick, event.value, self.stage.name, reason))
        log.info("tick %d (%.1f s): %s → stage %s (%s)", self._tick, self._tick / self.control_hz, event.value,
                 self.stage.name, reason)
        return self._output(event, ev)

    def _new_context(self, carry_arm: int | None = None) -> dict[str, Any]:
        cfg = dict(self.cfg)
        return {
            "cfg": cfg,
            "control_hz": self.control_hz,
            "vision": self.vision,
            "regions": bool(self.detectors.regions),
            "enter_prob": self.enter_prob,
            "wipe_target_count": self.wipe_target_count,
            "carry_arm": carry_arm,
            "carry_start_z": None,
            "lift_m": 0.0,
            "released_after_grasp": False,
            "release_pos": None,
            "release_region": None,
            "dist_from_release_m": 0.0,
            "dist_from_handle_m": 0.0,
            "ticks_since_release": 0,
            "grasp_regions": [],
            "lid_open_stable_ticks": 0,
            "lid_closed_stable_ticks": 0,
            "lid_disagree_ticks": 0,
            "lid_pushed_closed": False,
            "lid_last_estimate": self._lid_memory,
            "released_after_wipes": False,
            "script_contacts": 0,
            "held_ticks": 0,
        }

    def _enter_stage(self, pred: StagePrediction, carry_arm: int | None = None) -> None:
        """Set up per-stage state for ``self.stage`` (also after a retry)."""
        self._entered = True
        self.context = self._new_context(carry_arm)
        self.hold_counter = 0
        self.best_progress = 0.0
        self.best_progress_tick = self._tick
        self.in_script = False
        self.script_done = False
        self._script_exit_pending = False
        if self.stage == Stage.LIFT_LID:
            self.detectors.enable_lid_tracking(0.0, closing=False)
        elif self.stage == Stage.CLOSE_LID:
            initial = pred.lid_open if self.vision else self._lid_memory
            self.detectors.enable_lid_tracking(initial, closing=True)
        else:
            self.detectors.disable_lid_tracking()
        if self.stage == Stage.WIPE_CHAMBER:
            self.detectors.start_wipe_count()
        else:
            self.detectors.stop_wipe_count()

    def _advance(self, keep_queue: bool) -> OrchestratorEvent:
        """Move to the next stage, reset per-stage counters, update the detectors' wipe counting.

        Args:
            keep_queue: True for the EXTRACT_COMPONENT → PACK_COMPONENT carry handover.

        Returns:
            ADVANCE or ADVANCE_KEEP_QUEUE.
        """
        carry = self.context.get("carry_arm") if keep_queue else None
        carry_z = self.context.get("carry_start_z") if keep_queue else None
        if self.stage == Stage.LIFT_LID:
            self._lid_memory = max(self._lid_memory, self.detectors.lid_angle_kin_peak)
            if self.last_prediction is not None and self.vision:
                self._lid_memory = max(self._lid_memory, self.last_prediction.lid_open)
        self.stage = Stage(int(self.stage) + 1)
        self.stage_start_tick = self._tick
        self.attempt_start_tick = self._tick
        if self.stage == Stage.DONE:
            self.detectors.disable_lid_tracking()
            self.detectors.stop_wipe_count()
            self._entered = True
            self.hold_counter = 0
            return OrchestratorEvent.ADVANCE
        self._enter_stage(self.last_prediction or self._no_prediction(), carry_arm=carry)
        if keep_queue:
            self.context["carry_start_z"] = carry_z
        return OrchestratorEvent.ADVANCE_KEEP_QUEUE if keep_queue else OrchestratorEvent.ADVANCE

    def _retry(self, reason: str = "") -> OrchestratorEvent:
        """Register a retry of the current stage (counter++, restart stall timer).

        After ``max_retries`` the attempt is still restarted (context cleared, so level-type failures
        do not fire again every tick), but NONE is returned: the policy keeps going without a flush.

        Returns:
            RETRY, or NONE once the retries of this stage are used up.
        """
        idx = int(self.stage)
        carry = self.context.get("carry_arm") if self.stage == Stage.PACK_COMPONENT and "dropped" not in reason else None
        self.attempt_start_tick = self._tick
        self._enter_stage(self.last_prediction or self._no_prediction(), carry_arm=carry)
        if self.retries[idx] >= self.max_retries:
            return OrchestratorEvent.NONE
        self.retries[idx] += 1
        return OrchestratorEvent.RETRY

    def _is_stalled(self, tick: int, pred: StagePrediction) -> bool:
        """True if the classifier's progress has not increased for ``stall_ticks``.

        Needs the classifier: without it progress is unknown and this is always False. While the
        scripted wipe runs, its own timing applies and this is False too.
        """
        if not self.vision or self.in_script:
            return False
        if pred.progress > self.best_progress + 0.01:
            self.best_progress = pred.progress
            self.best_progress_tick = tick
        return tick - max(self.best_progress_tick, self.attempt_start_tick) >= self.stall_ticks

    def _budget_left_s(self, tick: int) -> float:
        """Remaining budget of the current attempt in seconds (negative when exceeded)."""
        if self.stage == Stage.DONE:
            return 0.0
        return self.specs[int(self.stage)].budget_s - (tick - self.attempt_start_tick) / self.control_hz

    def _ready_for_script(self, ev: EventState) -> bool:
        """WIPE_CHAMBER hand-over: sponge held and in the chamber (region, else contact, else held long enough)."""
        arm = ev.active_arm
        if arm is None or not ev.holding[arm]:
            return False
        if self.detectors.regions:
            return ev.arm_regions[arm] == "chamber"
        if ev.contact_available:
            return ev.in_contact
        return int(self.context.get("held_ticks", 0)) >= self.script_enter_hold_ticks

    def _update_phase(self, ev: EventState, pred: StagePrediction) -> None:
        if self.vision:
            self.phase = Phase(int(np.argmax(pred.phase_probs)))
            return
        if ev.grasp[0] or ev.grasp[1]:
            self.phase = Phase.GRASP
        elif ev.holding[0] or ev.holding[1]:
            self.phase = Phase.TRANSPORT
        elif self.context.get("released_after_grasp"):
            self.phase = Phase.RETRACT
        else:
            self.phase = Phase.APPROACH

    def _update_stage_context(self, obs: CanonicalObs, ev: EventState, pred: StagePrediction) -> None:
        """Carry arm, lift, release pose / region / distance, wipe release; lid keys in lid stages."""
        ctx = self.context
        ctx["vision"] = self.vision
        pos = np.asarray(obs.eef_base_pos, dtype=np.float64)
        for i in (0, 1):
            if ev.grasp[i]:
                ctx["carry_arm"] = i
                ctx["carry_start_z"] = float(pos[i, 2]) if obs.eef_base_valid else None
                ctx["released_after_grasp"] = False
                ctx["release_pos"] = None
                ctx["ticks_since_release"] = 0
                ctx["lid_open_stable_ticks"] = 0
                ctx["lid_closed_stable_ticks"] = 0
                ctx["grasp_regions"].append(ev.arm_regions[i])
        arm = ctx.get("carry_arm")
        if arm is not None:
            if ev.holding[arm]:
                ctx["held_ticks"] = int(ctx.get("held_ticks", 0)) + 1
                if ctx.get("carry_start_z") is None and obs.eef_base_valid:
                    ctx["carry_start_z"] = float(pos[arm, 2])
                if ctx.get("carry_start_z") is not None and obs.eef_base_valid:
                    ctx["lift_m"] = float(pos[arm, 2]) - float(ctx["carry_start_z"])
            else:
                ctx["held_ticks"] = 0
            if ev.release[arm]:
                ctx["released_after_grasp"] = True
                ctx["release_pos"] = pos[arm].copy() if obs.eef_base_valid else None
                ctx["release_region"] = ev.arm_regions[arm]
                ctx["ticks_since_release"] = 0
                if self.stage == Stage.WIPE_CHAMBER:
                    count = ev.wipe_count if ev.contact_available else max(ev.wipe_count, ctx.get("script_contacts", 0))
                    if count >= self.wipe_target_count:
                        ctx["released_after_wipes"] = True
            elif ctx.get("released_after_grasp"):
                ctx["ticks_since_release"] = int(ctx["ticks_since_release"]) + 1
            if ctx.get("release_pos") is not None and obs.eef_base_valid:
                ctx["dist_from_release_m"] = float(np.linalg.norm(pos[arm] - ctx["release_pos"]))
            elif ctx.get("released_after_grasp") and not obs.eef_base_valid:
                # no pose: count the retract as done after the stability window
                ctx["dist_from_release_m"] = float("inf") if ctx["ticks_since_release"] > self.control_hz else 0.0
        if self.stage in LID_STAGES:
            self._update_lid_context(obs, ev, pred)

    def _update_lid_context(self, obs: CanonicalObs, ev: EventState, pred: StagePrediction) -> None:
        """Maintain the lid-related context keys during LIFT_LID and CLOSE_LID.

        Functionality:
            - ``released_after_grasp``: set on a release edge of the arm that was tracking the lid.
            - ``dist_from_handle_m``: distance of that arm's EEF from its pose at the release.
            - ``lid_open_stable_ticks``: after the release, +1 while ``pred.lid_open >= lid_stable_th``,
              reset to 0 otherwise (LIFT_LID: lid must stay open, not swing shut). Without vision
              it counts ticks since the release.
            - ``lid_closed_stable_ticks``: after the release, +1 while
              ``pred.lid_open <= lid_closed_vision_th``, reset otherwise (CLOSE_LID).
            - ``lid_disagree_ticks``: ticks the kinematic and vision estimates differ by more than
              ``lid_disagree_th`` while tracking (``lid_open_estimate`` then trusts vision).
            - ``lid_pushed_closed``: CLOSE_LID, contact proxy fired with the estimate near closed.
            - Entering / leaving the stage arms / disarms the detectors (``_enter_stage``).

        Args:
            obs: canonical observation (EEF poses).
            ev: events of this tick (release edges, lid tracking state).
            pred: classifier prediction (lid_open).
        """
        ctx = self.context
        c = self.cfg
        ctx["dist_from_handle_m"] = float(ctx.get("dist_from_release_m", 0.0))
        if ev.lid_tracking and ev.lid_angle_kin is not None and self.vision:
            if abs(ev.lid_angle_kin - pred.lid_open) > float(c.get("lid_disagree_th", 0.2)):
                ctx["lid_disagree_ticks"] = int(ctx["lid_disagree_ticks"]) + 1
            else:
                ctx["lid_disagree_ticks"] = 0
        estimate = lid_open_estimate(ev, pred, ctx)
        ctx["lid_last_estimate"] = estimate
        if ev.lid_tracking:
            self._lid_memory = estimate if self.stage == Stage.CLOSE_LID else max(self._lid_memory, estimate)
        if self.stage == Stage.CLOSE_LID and ev.in_contact and estimate <= 0.1:
            ctx["lid_pushed_closed"] = True
        released = bool(ctx.get("released_after_grasp")) and not (ev.holding[0] or ev.holding[1])
        if not released:
            ctx["lid_open_stable_ticks"] = 0
            ctx["lid_closed_stable_ticks"] = 0
            return
        if self.vision:
            open_ok = pred.lid_open >= float(c.get("lid_stable_th", 0.45))
            done_p = float(pred.stage_probs[int(Stage.DONE)]) if len(pred.stage_probs) > int(Stage.DONE) else 0.0
            closed_ok = pred.lid_open <= float(c.get("lid_closed_vision_th", 0.02)) or done_p >= self.enter_prob
        else:
            open_ok = closed_ok = True
        ctx["lid_open_stable_ticks"] = int(ctx["lid_open_stable_ticks"]) + 1 if open_ok else 0
        ctx["lid_closed_stable_ticks"] = int(ctx["lid_closed_stable_ticks"]) + 1 if closed_ok else 0
