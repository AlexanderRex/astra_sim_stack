"""AstraPolicy: harness entry point that owns layers 2-6 and runs the control loop.

Purpose
-------
The task repo's evaluation harness (``robofinals/applications/eval/entrypoints/policy.py``) imports
``policy_name: astra.policy:AstraPolicy``, constructs it with ``usr_args`` (which calls
``get_model``), then for every episode calls ``reset_model()`` followed by
``eval(task_env, observation, usr_args, video_writer)``. ``eval`` owns the step loop: the env waits
for each action (synchronous stepping), so inference time does not consume sim time.

Per-tick loop inside ``eval`` (to implement)
--------------------------------------------
::

    obs   = adapter.process(raw_obs)                                 # layer 2
    orch  = orchestrator.update(obs, last_action,                    # layer 3
                                adapter.classifier_input())
    source, flush = router.select_source(orch)                       # layer 4
    if flush: executor.flush()
    if adapter.needs_hold:  model_a = safety.hold_action(obs)
    else:
        if executor.need_chunk(obs.tick):
            executor.push(router.get_chunk(obs, orch, adapter), obs.tick)
        model_a = executor.step(obs.tick)                            # layer 5
    model_a, grip, clamped = safety.apply(model_a, obs)
    sim_a  = SimAction(codec.decode(model_a, grip), obs.tick, source, clamped)
    raw_obs, done, extras = sim.step(sim_a)                          # layer 6 (env side)
    last_action = sim_a
    stop when done, orch.stage == DONE, or sim time ≥ EPISODE_LIMIT_S

Inputs
------
- ``usr_args``: harness config (``astra/config/astra_eval.yml``) plus ``actions_dim``,
  ``accepted_action_dims`` and ``decimation`` injected by the harness; ``astra_schema`` / ``astra_runtime`` point at the YAMLs.
- ``task_env``: env client; ``observation``: first observation after reset.

Outputs
-------
- ``eval`` returns True if the orchestrator reached ``Stage.DONE``, as required by
  ``BasePolicy.format_eval_results``. This is our own estimate: the env reports no success flag or
  score to the policy, the simulator-side metrics are the authority. Per-stage progress is logged.
"""

from __future__ import annotations

from typing import Any

from policy.base import BasePolicy  # task repo, on PYTHONPATH in the policy container

from astra.adapter.observation_adapter import ObservationAdapter
from astra.control.sim_interface import SimActionInterface
from astra.orchestrator.task_orchestrator import TaskOrchestrator
from astra.policies.router import PolicyRouter
from astra.postprocess.action_codec import ActionCodec
from astra.postprocess.chunk_executor import ChunkExecutor
from astra.postprocess.safety import SafetyFilter
from astra.schema import Schema
from astra.types import EPISODE_LIMIT_S, SimAction


class AstraPolicy(BasePolicy):
    """Owns every runtime subsystem; one instance serves all episodes of an evaluation run.

    Attributes:
        schema: resolved schema.yaml.
        runtime_cfg: parsed runtime.yaml.
        adapter: layer 2 observation adapter.
        orchestrator: layer 3 task orchestrator (with detectors and stage classifier).
        router: layer 4 policy router (ACT, GR00T, scripted wipe).
        executor: layer 5 chunk executor.
        safety: layer 5 safety filter.
        codec: layer 5 action codec (model 23 → sim 19, or the legacy 20).
        sim: layer 6 sim action interface, created per episode in ``eval`` (needs task_env).
        last_action: action sent on the previous tick (input to detectors / safety).
        episode_limit_s: stop the loop after this much sim time.
        episode_log: per-episode records (stage transitions, retries, clamps) for analysis.
    """

    def __init__(self, usr_args: dict[str, Any]) -> None:
        self.schema: Schema | None = None
        self.runtime_cfg: dict[str, Any] = {}
        self.adapter: ObservationAdapter | None = None
        self.orchestrator: TaskOrchestrator | None = None
        self.router: PolicyRouter | None = None
        self.executor: ChunkExecutor | None = None
        self.safety: SafetyFilter | None = None
        self.codec: ActionCodec | None = None
        self.sim: SimActionInterface | None = None
        self.last_action: SimAction | None = None
        self.episode_limit_s: float = EPISODE_LIMIT_S
        self.episode_log: list[dict[str, Any]] = []
        super().__init__(usr_args)  # calls initialize() → get_model(usr_args)

    def get_model(self, usr_args: dict[str, Any]) -> Any:
        """Build and load every subsystem once (called by ``BasePolicy.initialize``).

        Functionality:
            1. ``load_schema(usr_args["astra_schema"])``; read runtime.yaml.
            2. Create ObservationAdapter, EventDetectors, StageClassifier, stage specs,
               TaskOrchestrator, ActPolicy / Gr00tPolicyBackend / ScriptedWipe (as configured),
               PolicyRouter, ChunkExecutor, SafetyFilter, ActionCodec.
            3. Call ``load()`` on adapter, classifier, router (loads model weights).
            Note: ``usr_args["decimation"]`` / ``["actions_dim"]`` are injected by the harness only
            after construction, so rate and action-dim checks run at the start of ``eval``.

        Args:
            usr_args: harness configuration.

        Returns:
            The router (the object holding the models), stored by BasePolicy as ``self.model``.
        """
        raise NotImplementedError("AstraPolicy.get_model: build and load layers 2-6")

    def reset_model(self) -> None:
        """Called by the harness before every episode: reset adapter, orchestrator, router,
        executor, codec; clear ``last_action`` and ``episode_log``. The safety filter is reset on
        the first observation inside ``eval`` because it needs the initial EEF pose."""
        raise NotImplementedError("AstraPolicy.reset_model")

    def get_action(self) -> Any:
        """Required by BasePolicy but unused: actions are produced inside ``eval``'s loop."""
        raise NotImplementedError("AstraPolicy.get_action: not used, see eval()")

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any) -> bool:
        """Run one episode (see the per-tick loop in the module docstring).

        Functionality:
            1. ``check_runtime_rates(schema, usr_args)``; create ``SimActionInterface``;
               ``check_contract(codec.select_format(sim.accepted_action_dims))``.
            2. Process the initial observation; ``safety.reset(obs)``.
            3. Loop until done / DONE stage / episode limit; write video frames via
               ``add_video_frame`` if ``video_writer`` is set.
            4. Log stage transitions, retries, clamps and timing.

        Args:
            task_env: env client.
            observation: first observation after reset.
            usr_args: harness configuration.
            video_writer: optional video writer from the harness.

        Returns:
            True if the orchestrator reached Stage.DONE, else False.
        """
        raise NotImplementedError("AstraPolicy.eval: run the layer 2-6 control loop")
