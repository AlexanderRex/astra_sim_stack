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
"""

from __future__ import annotations

from typing import Any

from astra.adapter.observation_adapter import ObservationAdapter
from astra.policies.act_policy import ActPolicy
from astra.policies.base import PolicyBackend
from astra.policies.groot_policy import Gr00tPolicyBackend
from astra.policies.scripted_wipe import ScriptedWipe
from astra.types import ActionChunk, CanonicalObs, OrchestratorOutput, PolicySource, Stage


class PolicyRouter:
    """Chooses the active producer and requests chunks from it.

    Attributes:
        cfg: ``runtime.yaml: router`` section.
        backends: source → producer (only the configured ones are present).
        stage_to_source: producer per Stage 0..5 (``script`` for WIPE uses the scripted primitive
            once the orchestrator emits ENTER_SCRIPT; before that the learned default is used).
        default_source: producer used when a stage maps to an unavailable backend.
        active_source: producer currently in control.
        switch_only_at_stage_boundary: never change producer mid-stage (except script enter/exit).
        fault_count: consecutive producer failures; > 0 makes the router emit HOLD chunks.
        inference_wall_s: last inference wall-clock time (logging only).
    """

    def __init__(
        self, cfg: dict[str, Any], act: ActPolicy | None, groot: Gr00tPolicyBackend | None, script: ScriptedWipe | None
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
        self.default_source = PolicySource(cfg.get("default_source", "act"))
        self.active_source: PolicySource = self.default_source
        self.switch_only_at_stage_boundary: bool = bool(cfg.get("switch_only_at_stage_boundary", True))
        self.fault_count: int = 0
        self.inference_wall_s: float = 0.0

    def load(self) -> None:
        """Load every configured backend (``backend.load()``)."""
        raise NotImplementedError("PolicyRouter.load")

    def reset(self) -> None:
        """Episode start: active_source = source for LIFT_LID, reset all backends, fault_count = 0."""
        raise NotImplementedError("PolicyRouter.reset")

    def select_source(self, orch: OrchestratorOutput) -> tuple[PolicySource, bool]:
        """Decide the producer for this tick.

        Functionality:
            - On ADVANCE / ADVANCE_KEEP_QUEUE: look up ``stage_to_source[stage]`` (fallback to
              default if unavailable); switching flushes unless ADVANCE_KEEP_QUEUE and same source.
            - On ENTER_SCRIPT / EXIT_SCRIPT: switch to / from SCRIPT, flush.
            - On RETRY: same source, flush, reset the backend.
            - Otherwise keep ``active_source``.

        Args:
            orch: orchestrator output of this tick.

        Returns:
            (source, flush) where flush tells the executor to drop its queue.
        """
        raise NotImplementedError("PolicyRouter.select_source")

    def get_chunk(self, obs: CanonicalObs, orch: OrchestratorOutput, adapter: ObservationAdapter) -> ActionChunk:
        """Ask the active producer for a chunk.

        Functionality:
            Build only the view that producer needs (``adapter.act_view`` / ``adapter.groot_view``),
            call ``predict``, check the chunk (shape (T, 24), finite values, stage matches); on an
            exception or invalid chunk increment ``fault_count`` and return ``hold_chunk``.

        Args:
            obs: canonical observation.
            orch: orchestrator output.
            adapter: observation adapter (view builders).

        Returns:
            ActionChunk from the producer, or a HOLD chunk on failure.
        """
        raise NotImplementedError("PolicyRouter.get_chunk")

    def hold_chunk(self, obs: CanonicalObs, stage: Stage) -> ActionChunk:
        """Safety chunk: both EEF targets = current FK pose, grippers unchanged, wheels 0.

        Args:
            obs: canonical observation (current EEF poses, gripper state).
            stage: current stage.

        Returns:
            ActionChunk (1, 24), source HOLD, fps = control rate.
        """
        raise NotImplementedError("PolicyRouter.hold_chunk")
