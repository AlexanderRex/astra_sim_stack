"""Policy router (layer 4): picks the chunk producer for the current stage and manages switching.

Purpose
-------
Decide which producer (ACT, GR00T, scripted wipe, or a safety hold) supplies the next action chunk,
and make switches clean: switch only at stage boundaries, flush the executor's queue on a switch,
and fall back to a hold on faults (exceptions, NaN chunks). In the synchronous sim, inference latency
costs no sim time, so there is no latency watchdog; a wall-clock budget is only logged.

Inputs (per tick)
-----------------
- ``OrchestratorOutput`` (stage, prompt, event).
- ``CanonicalObs`` and the views from the adapter (built lazily for the selected producer only).
- ``need_chunk`` flag from the chunk executor (re-plan point reached or queue empty).

Outputs
-------
- ``ActionChunk`` or None (no new chunk needed this tick).
- ``flush`` flag telling the executor to drop its queue (model switch, ADVANCE, RETRY).

``stage_to_source`` names ``script`` for WIPE_CHAMBER; the learned default runs that stage until the
orchestrator emits ENTER_SCRIPT and again after EXIT_SCRIPT. When no learned backend is available
the router stays on HOLD. ``script_finished`` tells the control loop to notify the orchestrator.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from astra.adapter.observation_adapter import ObservationAdapter
from astra.adapter.rotations import matrix_to_rot6d
from astra.policies.act_policy import ActPolicy
from astra.policies.base import PolicyBackend
from astra.policies.groot_policy import Gr00tPolicyBackend
from astra.policies.scripted_wipe import ScriptedWipe
from astra.types import (
    CONTROL_HZ,
    GRIPPER_CLOSED_RAD,
    MODEL_ACTION_DIM,
    MODEL_GRIPPER,
    MODEL_LEFT_EEF,
    MODEL_RIGHT_EEF,
    ActionChunk,
    CanonicalObs,
    OrchestratorEvent,
    OrchestratorOutput,
    PolicySource,
    Stage,
)

log = logging.getLogger(__name__)
LEARNED = (PolicySource.ACT, PolicySource.GROOT)


class PolicyRouter:
    """Chooses the active producer and requests chunks from it.

    Attributes:
        cfg: ``runtime.yaml: router`` section.
        backends: source → producer (only the configured ones are present).
        stage_to_source: producer per Stage 0..5 (``script`` for WIPE_CHAMBER uses the scripted primitive
            once the orchestrator emits ENTER_SCRIPT; before that the learned default is used).
        default_source: producer used when a stage maps to an unavailable backend.
        active_source: producer currently in control.
        switch_only_at_stage_boundary: never change producer mid-stage (except script enter/exit).
        fault_count: consecutive producer failures; > 0 makes the router emit HOLD chunks.
        fault_total: producer failures this episode (logging).
        inference_wall_s: last inference wall-clock time (logging only).
        hold_ticks: length of HOLD chunks (the router asks the producer again after it).
        hold_fn: callable obs → (23,) hold action (``SafetyFilter.hold_action``); when unset the
            hold is built from the observation.
        control_hz: tick rate (time base of HOLD chunks).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        act: ActPolicy | None,
        groot: Gr00tPolicyBackend | None,
        script: ScriptedWipe | None,
        control_hz: float = CONTROL_HZ,
    ) -> None:
        self.cfg = cfg
        self.backends: dict[PolicySource, PolicyBackend] = {}
        if act is not None:
            self.backends[PolicySource.ACT] = act
        if groot is not None:
            self.backends[PolicySource.GROOT] = groot
        if script is not None:
            self.backends[PolicySource.SCRIPT] = script
        self.stage_to_source: list[PolicySource] = [PolicySource(s) for s in cfg.get("stage_to_source", ["act"] * 6)]
        if len(self.stage_to_source) != 6:
            raise ValueError(f"router.stage_to_source needs 6 entries, got {len(self.stage_to_source)}")
        self.default_source = PolicySource(cfg.get("default_source", "act"))
        self.active_source: PolicySource = self.default_source
        self.switch_only_at_stage_boundary: bool = bool(cfg.get("switch_only_at_stage_boundary", True))
        self.fault_count: int = 0
        self.fault_total: int = 0
        self.inference_wall_s: float = 0.0
        self.hold_ticks: int = max(1, int(cfg.get("hold_chunk_ticks", 10)))
        self.hold_fn: Callable[[CanonicalObs], np.ndarray] | None = None
        self.control_hz = control_hz
        self._script_pending_start: bool = False
        self._script_arm: int | None = None

    def load(self) -> None:
        """Load every configured backend (``backend.load()``)."""
        for source, backend in self.backends.items():
            backend.load()
            log.info("router: %s loaded", source.value)

    def reset(self) -> None:
        """Episode start: active_source = source for LIFT_LID, reset all backends, fault_count = 0."""
        for backend in self.backends.values():
            backend.reset()
        self.active_source = self.learned_source(Stage.LIFT_LID)
        self.fault_count = 0
        self.fault_total = 0
        self._script_pending_start = False
        self._script_arm = None

    @property
    def script(self) -> ScriptedWipe | None:
        b = self.backends.get(PolicySource.SCRIPT)
        return b if isinstance(b, ScriptedWipe) else None

    @property
    def script_finished(self) -> bool:
        """True while the scripted wipe is in control and has played its last waypoint."""
        s = self.script
        return self.active_source == PolicySource.SCRIPT and s is not None and s.started and s.done

    def learned_source(self, stage: Stage) -> PolicySource:
        """Learned producer for ``stage``: ``stage_to_source`` (script → default), else any loaded one, else HOLD."""
        idx = min(int(stage), 5)
        want = self.stage_to_source[idx]
        for cand in (want, self.default_source, *LEARNED):
            if cand in LEARNED and cand in self.backends:
                return cand
        return PolicySource.HOLD

    def select_source(self, orch: OrchestratorOutput) -> tuple[PolicySource, bool]:
        """Decide the producer for this tick.

        Functionality:
            - On ADVANCE / ADVANCE_KEEP_QUEUE: look up ``stage_to_source[stage]`` (fallback to
              default if unavailable); switching flushes unless ADVANCE_KEEP_QUEUE and same source.
            - On ENTER_SCRIPT / EXIT_SCRIPT: switch to / from SCRIPT, flush.
            - On RETRY: learned source of the stage (leaves the script), flush, reset the backend.
            - Otherwise keep ``active_source`` (or follow ``stage_to_source`` mid-stage when
              ``switch_only_at_stage_boundary`` is False).

        Args:
            orch: orchestrator output of this tick.

        Returns:
            (source, flush) where flush tells the executor to drop its queue.
        """
        ev = orch.event
        flush = False
        if ev in (OrchestratorEvent.ADVANCE, OrchestratorEvent.ADVANCE_KEEP_QUEUE):
            if orch.stage == Stage.DONE:
                return self.active_source, False
            new = self.learned_source(orch.stage)
            flush = not (ev == OrchestratorEvent.ADVANCE_KEEP_QUEUE and new == self.active_source)
            self.active_source = new
        elif ev == OrchestratorEvent.ENTER_SCRIPT:
            if self.script is not None:
                self.active_source = PolicySource.SCRIPT
                self._script_pending_start = True
                self._script_arm = orch.active_arm
                flush = True
        elif ev == OrchestratorEvent.EXIT_SCRIPT:
            self.active_source = self.learned_source(orch.stage)
            flush = True
        elif ev == OrchestratorEvent.RETRY:
            self.active_source = self.learned_source(orch.stage)
            backend = self.backends.get(self.active_source)
            if backend is not None:
                backend.reset()
            flush = True
        elif not self.switch_only_at_stage_boundary and self.active_source != PolicySource.SCRIPT:
            new = self.learned_source(orch.stage)
            if new != self.active_source:
                self.active_source, flush = new, True
        return self.active_source, flush

    def get_chunk(self, obs: CanonicalObs, orch: OrchestratorOutput, adapter: ObservationAdapter) -> ActionChunk:
        """Ask the active producer for a chunk.

        Functionality:
            Build only the view that producer needs (``adapter.act_view`` / ``adapter.groot_view``),
            call ``predict``, check the chunk (shape (T, 23), finite values, stage matches); on an
            exception or invalid chunk increment ``fault_count`` and return ``hold_chunk``.

        Args:
            obs: canonical observation.
            orch: orchestrator output.
            adapter: observation adapter (view builders).

        Returns:
            ActionChunk from the producer, or a HOLD chunk on failure.
        """
        source = self.active_source
        backend = self.backends.get(source)
        if source == PolicySource.HOLD or backend is None:
            return self.hold_chunk(obs, orch.stage)
        start = time.perf_counter()
        try:
            if source == PolicySource.SCRIPT:
                assert isinstance(backend, ScriptedWipe)
                if self._script_pending_start:
                    arm = self._script_arm if self._script_arm is not None else backend.wiping_arm
                    backend.start(obs, arm)
                    self._script_pending_start = False
                view = None
            elif source == PolicySource.ACT:
                view = adapter.act_view(obs, orch.stage)
            else:
                view = adapter.groot_view(obs, orch.prompt)
            chunk = backend.predict(obs, view, orch)
            self._check_chunk(chunk)
        except Exception as exc:  # noqa: BLE001 - any producer fault falls back to a hold
            self.fault_count += 1
            self.fault_total += 1
            if self.fault_count <= 3 or self.fault_count % 50 == 0:
                log.exception("tick %d: %s producer failed (%d in a row): %s", obs.tick, source.value,
                              self.fault_count, exc)
            return self.hold_chunk(obs, orch.stage)
        finally:
            self.inference_wall_s = time.perf_counter() - start
        self.fault_count = 0
        chunk.start_tick = obs.tick
        chunk.stage = orch.stage
        return chunk

    @staticmethod
    def _check_chunk(chunk: ActionChunk) -> None:
        acts = np.asarray(chunk.actions)
        if acts.ndim != 2 or acts.shape[0] < 1 or acts.shape[1] != MODEL_ACTION_DIM:
            raise ValueError(f"chunk has shape {acts.shape}, expected (T, {MODEL_ACTION_DIM})")
        if not np.all(np.isfinite(acts)):
            raise ValueError("chunk contains NaN/Inf")
        if chunk.fps <= 0:
            raise ValueError(f"chunk fps {chunk.fps}")

    def hold_chunk(self, obs: CanonicalObs, stage: Stage) -> ActionChunk:
        """Safety chunk: both EEF targets = current FK pose, grippers unchanged, base velocity 0.

        Args:
            obs: canonical observation (current EEF poses, gripper state).
            stage: current stage.

        Returns:
            ActionChunk (hold_ticks, 23), source HOLD, fps = control rate.

        Raises:
            RuntimeError: no pose is known to hold.
        """
        if self.hold_fn is not None:
            row = np.asarray(self.hold_fn(obs), dtype=np.float32)
        else:
            if not obs.eef_base_valid:
                raise RuntimeError("hold chunk: no base-frame EEF pose (set kinematics.urdf_path)")
            row = np.zeros(MODEL_ACTION_DIM, dtype=np.float32)
            for i, sl in enumerate((MODEL_LEFT_EEF, MODEL_RIGHT_EEF)):
                row[sl.start : sl.start + 3] = obs.eef_base_pos[i]
                row[sl.start + 3 : sl.stop] = matrix_to_rot6d(obs.eef_base_rotmat[i])
            row[MODEL_GRIPPER] = (np.asarray(obs.gripper_raw) > 0.5 * GRIPPER_CLOSED_RAD).astype(np.float32)
        return ActionChunk(
            actions=np.tile(row, (self.hold_ticks, 1)),
            source=PolicySource.HOLD,
            fps=self.control_hz,
            start_tick=obs.tick,
            stage=stage,
        )
