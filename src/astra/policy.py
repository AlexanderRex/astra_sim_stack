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
    if router.script_finished: orchestrator.notify_script_done(contacts)
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

Config paths
------------
``usr_args["astra_schema"]`` / ``["astra_runtime"]``; when absent, the YAMLs next to this module
(``astra/config/``). Relative checkpoint and calibration paths in runtime.yaml are resolved against
the directory of runtime.yaml.

Outputs
-------
- ``eval`` returns True if the orchestrator reached ``Stage.DONE``, as required by
  ``BasePolicy.format_eval_results``. This is our own estimate: the env reports no success flag or
  score to the policy, the simulator-side metrics are the authority. Per-stage progress is logged,
and written to ``<usr_args["save_path"]>/astra_episode.json`` when the harness sets a save path.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from policy.base import BasePolicy  # task repo, on PYTHONPATH in the policy container

from astra.adapter.observation_adapter import ObservationAdapter
from astra.control.sim_interface import SimActionInterface
from astra.orchestrator.event_detectors import EventDetectors
from astra.orchestrator.stage_classifier import StageClassifier
from astra.orchestrator.stage_specs import build_stage_specs
from astra.orchestrator.task_orchestrator import TaskOrchestrator
from astra.policies.act_policy import ActPolicy
from astra.policies.groot_policy import Gr00tPolicyBackend
from astra.policies.router import PolicyRouter
from astra.policies.scripted_wipe import ScriptedWipe
from astra.postprocess.action_codec import ActionCodec
from astra.postprocess.chunk_executor import ChunkExecutor
from astra.postprocess.safety import SafetyFilter
from astra.schema import Schema, check_runtime_rates, load_schema
from astra.types import EPISODE_LIMIT_S, OrchestratorEvent, PolicySource, SimAction, Stage

log = logging.getLogger("astra")
CONFIG_DIR = Path(__file__).resolve().parent / "config"
VIDEO_FPS = 30.0  # BasePolicy.add_video_frame writes at 30 fps


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    """Parse runtime.yaml and resolve relative checkpoint / calibration paths against its directory."""
    import yaml

    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = path.resolve().parent

    def resolve(value: Any) -> Any:
        if value in (None, ""):
            return None
        p = Path(str(value)).expanduser()
        return str(p if p.is_absolute() else base / p)

    for key in ("act", "groot", "stage_classifier"):
        cfg.setdefault("checkpoints", {})
        cfg["checkpoints"][key] = resolve(cfg["checkpoints"].get(key))
    det = cfg.setdefault("detectors", {})
    for key in ("hinge_calibration", "regions_calibration"):
        det[key] = resolve(det.get(key))
    sw = cfg.setdefault("scripted_wipe", {})
    sw["waypoints"] = resolve(sw.get("waypoints"))
    return cfg


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
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
        schema = load_schema(usr_args.get("astra_schema") or CONFIG_DIR / "schema.yaml")
        rt = load_runtime_config(usr_args.get("astra_runtime") or CONFIG_DIR / "runtime.yaml")
        self.schema, self.runtime_cfg = schema, rt
        hz = schema.control_hz
        device = str(rt.get("device", "cuda:0"))
        ckpt = rt.get("checkpoints") or {}
        orch_cfg = dict(rt.get("orchestrator") or {})
        router_cfg = dict(rt.get("router") or {})
        det_cfg = dict(rt.get("detectors") or {})
        self.episode_limit_s = float(rt.get("episode_limit_s", EPISODE_LIMIT_S))

        stage_cond = bool(orch_cfg.get("enabled", True)) and bool((rt.get("act") or {}).get("use_stage_conditioning", True))
        self.adapter = ObservationAdapter(schema, dict(rt.get("adapter") or {}), use_stage_conditioning=stage_cond)
        detectors = EventDetectors(det_cfg, hz)
        classifier = StageClassifier(ckpt.get("stage_classifier"), device=device, control_hz=hz)

        act_cfg = dict(rt.get("act") or {})
        act = (
            ActPolicy(ckpt["act"], device=device, use_stage_conditioning=stage_cond,
                      gripper_mode=str(act_cfg.get("gripper_mode", "prob")), fps=schema.data_fps)
            if ckpt.get("act") else None
        )
        groot_cfg = dict(rt.get("groot") or {})
        groot = (
            Gr00tPolicyBackend(ckpt["groot"], device=device, strict=bool(groot_cfg.get("strict", False)),
                               fps=schema.data_fps)
            if ckpt.get("groot") else None
        )
        wants_script = "script" in [str(s) for s in router_cfg.get("stage_to_source", [])]
        script = ScriptedWipe(dict(rt.get("scripted_wipe") or {}), control_hz=hz) if wants_script else None
        self.router = PolicyRouter(router_cfg, act, groot, script, control_hz=hz)
        self.orchestrator = TaskOrchestrator(
            orch_cfg, build_stage_specs(orch_cfg), detectors, classifier, control_hz=hz,
            wipe_script=script is not None, wipe_target_count=int(det_cfg.get("wipe_target_count", 6)),
        )
        self.executor = ChunkExecutor(dict(rt.get("executor") or {}), control_hz=hz)
        self.safety = SafetyFilter(dict(rt.get("safety") or {}))
        self.codec = ActionCodec(schema)
        self.router.hold_fn = self.safety.hold_action

        self.adapter.load()
        classifier.load()
        self.router.load()
        if not any(s in self.router.backends for s in (PolicySource.ACT, PolicySource.GROOT)):
            log.warning("no ACT or GR00T checkpoint configured: the robot will only hold its pose")
        self.model = self.router
        return self.router

    def reset_model(self) -> None:
        """Called by the harness before every episode: reset adapter, orchestrator, router,
        executor, codec; clear ``last_action`` and ``episode_log``. The safety filter is reset on
        the first observation inside ``eval`` because it needs the initial EEF pose."""
        assert self.adapter and self.orchestrator and self.router and self.executor and self.codec
        self.adapter.reset()
        self.orchestrator.reset()
        self.router.reset()
        self.executor.reset()
        self.codec.reset()
        self.last_action = None
        self.episode_log = []

    def get_action(self) -> Any:
        """Required by BasePolicy but unused: actions are produced inside ``eval``'s loop. Returns None."""
        return None

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
        assert self.schema and self.adapter and self.orchestrator and self.router
        assert self.executor and self.safety and self.codec
        schema, adapter, orch_, router = self.schema, self.adapter, self.orchestrator, self.router
        executor, safety, codec = self.executor, self.safety, self.codec

        check_runtime_rates(schema, usr_args)
        self.sim = sim = SimActionInterface(task_env, usr_args)
        sim.reset()
        sim.check_contract(codec.select_format(sim.accepted_action_dims))
        if self.last_action is not None or adapter.tick != 0:
            self.reset_model()  # eval called without reset_model (e.g. a custom loop)

        # add_video_frame puts the images side by side, so they must share a height: head camera only
        # by default (runtime.yaml video_cameras lists Astra camera names).
        cam_names = list(self.runtime_cfg.get("video_cameras") or ["head"])
        cam_keys = [schema.cameras[name].obs_key for name in cam_names if name in schema.cameras]
        raw_obs = observation
        obs = adapter.process(raw_obs)
        safety.reset(obs)
        if not obs.eef_base_valid and not any(s in router.backends for s in (PolicySource.ACT, PolicySource.GROOT)):
            raise RuntimeError(
                "nothing can produce a safe arm command: no forward kinematics (kinematics.urdf_path) to hold "
                "the pose and no ACT / GR00T checkpoint"
            )

        wall0 = time.perf_counter()
        frames_written = 0
        clamps = 0
        holds = 0
        stage_times: dict[str, float] = {}
        done = False
        stop_reason = "episode limit"
        while True:
            if video_writer is not None and obs.t * VIDEO_FPS >= frames_written:
                self.add_video_frame(video_writer, raw_obs, cam_keys)
                frames_written += 1

            orch = orch_.update(obs, self.last_action, adapter.classifier_input)
            if orch.event != OrchestratorEvent.NONE:
                self._log_event(obs.t, orch, stage_times)
            if orch.stage == Stage.DONE:
                stop_reason = "all stages done"
                break

            source, flush = router.select_source(orch)
            if flush:
                executor.flush()
            if router.script_finished and router.script is not None:
                orch_.notify_script_done(router.script.contacts_done)

            if adapter.needs_hold:
                executor.flush()
                model_a = safety.hold_action(obs)
                source = PolicySource.HOLD
                holds += 1
            else:
                if executor.need_chunk(obs.tick):
                    executor.push(router.get_chunk(obs, orch, adapter), obs.tick)
                model_a = executor.step(obs.tick)
                source = executor.current.source if executor.current is not None else source
            model_a, grip, clamped = safety.apply(model_a, obs)
            clamps += int(clamped)
            sim_a = SimAction(codec.decode(model_a, grip), obs.tick, source, clamped)

            raw_obs, done, _extras = sim.step(sim_a)
            self.last_action = sim_a
            obs = adapter.process(raw_obs)
            if done:
                stop_reason = "env terminated"
                break
            if obs.t >= self.episode_limit_s:
                break

        success = orch_.stage == Stage.DONE
        summary = {
            "success_estimate": success,
            "stop_reason": stop_reason,
            "final_stage": orch_.stage.name,
            "ticks": obs.tick,
            "sim_time_s": round(obs.t, 3),
            "wall_time_s": round(time.perf_counter() - wall0, 1),
            "stage_reached_s": stage_times,
            "retries": dict(zip([s.name for s in Stage][:6], orch_.retries)),
            "wipe_count": orch_.detectors.wipe_count,
            "clamped_ticks": clamps,
            "hold_ticks": holds,
            "invalid_obs_ticks": adapter.invalid_total,
            "producer_faults": router.fault_total,
            "events": self.episode_log,
        }
        log.info("episode: %s at %s after %.1f s sim (%s); retries %s, clamps %d, holds %d, faults %d",
                 "DONE" if success else "stopped", orch_.stage.name, obs.t, stop_reason, summary["retries"],
                 clamps, holds, router.fault_total)
        self._write_summary(usr_args, summary)
        return success

    # ------------------------------------------------------------------ logging
    def _log_event(self, t: float, orch: Any, stage_times: dict[str, float]) -> None:
        assert self.orchestrator is not None
        rec = {
            "t": round(t, 3),
            "event": orch.event.value,
            "stage": orch.stage.name,
            "retries": orch.retries,
            "reason": self.orchestrator.last_reason,
        }
        self.episode_log.append(rec)
        if orch.event in (OrchestratorEvent.ADVANCE, OrchestratorEvent.ADVANCE_KEEP_QUEUE):
            stage_times.setdefault(orch.stage.name, round(t, 3))

    @staticmethod
    def _write_summary(usr_args: dict[str, Any], summary: dict[str, Any]) -> None:
        save = usr_args.get("save_path")
        if not save:
            return
        try:
            out = Path(str(save))
            out.mkdir(parents=True, exist_ok=True)
            with (out / "astra_episode.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=_json_default)
        except OSError as exc:
            log.warning("could not write the episode summary to %s: %s", save, exc)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
