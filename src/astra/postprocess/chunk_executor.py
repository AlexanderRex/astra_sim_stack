"""Chunk executor: plays action chunks back one control tick at a time.

Purpose
-------
Chunks carry their own time base (``fps``); the env runs at **50 Hz**. The demonstrations hold one row
per control step, so for policies trained on them the mapping below is the identity; it stays general
for a dataset that was subsampled. The executor maps control ticks to chunk indices, decides when to request a new chunk (re-plan point), blends the old and new chunk
at a switch, optionally applies ACT temporal ensembling (continuous dims only), and hands one
model-space action per tick to the safety filter.

Time mapping
------------
For a chunk predicted at tick ``n0`` with time base ``fps``: at tick ``n`` use index
``floor((n - n0) * fps / control_hz)``. Repeating an absolute EEF target for a tick is harmless
(DiffIK keeps tracking it), so no interpolation is needed. Scripted chunks use fps = control rate.

Re-planning
-----------
``need_chunk`` becomes True when the executed index reaches ``execute_steps[source]`` (ACT ≈ 25,
GR00T 8-16 of 40), when the queue is empty, or after a flush.

Sources without an ``execute_steps`` entry (scripted wipe, hold) play their whole chunk before the
next request. With temporal ensembling a chunk is requested every tick, and the overlapping chunks
are averaged with weights ``exp(-ensemble_coeff * age_rank)`` (oldest weighted most, as in ACT) on
the continuous dims; the gripper dims follow the newest chunk.

Inputs:  ActionChunk from the router, flush flag, current tick.
Outputs: one (23,) model-space action per tick; ``need_chunk`` flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.types import CONTROL_HZ, MODEL_GRIPPER, ActionChunk

_CONTINUOUS = np.ones(23, dtype=bool)
_CONTINUOUS[MODEL_GRIPPER] = False


class ChunkExecutor:
    """Queue and time mapping for action chunks.

    Attributes:
        cfg: ``runtime.yaml: executor`` section.
        control_hz: tick rate.
        execute_steps: source name → number of chunk steps (in chunk time) before re-planning.
        temporal_ensemble: enable ACT-style exponential ensembling over overlapping chunks.
        ensemble_coeff: ensembling weight decay.
        blend_ticks: crossfade length (ticks) between the old and a new chunk (continuous dims).
        current: chunk being executed, None when empty.
        previous: chunk being blended out, None when no blend is active.
        blend_start_tick: tick at which the current blend started.
        history: overlapping chunks kept for temporal ensembling.
    """

    def __init__(self, cfg: dict[str, Any], control_hz: float = CONTROL_HZ) -> None:
        self.cfg = cfg
        self.control_hz = control_hz
        self.execute_steps: dict[str, int] = dict(cfg.get("execute_steps", {"act": 25, "groot": 20}))
        self.temporal_ensemble: bool = bool(cfg.get("temporal_ensemble", False))
        self.ensemble_coeff: float = float(cfg.get("ensemble_coeff", 0.01))
        self.blend_ticks: int = int(cfg.get("blend_ticks", 5))
        self.current: ActionChunk | None = None
        self.previous: ActionChunk | None = None
        self.blend_start_tick: int = 0
        self.history: list[ActionChunk] = []
        self.flushed: bool = True
        self.max_history: int = int(cfg.get("ensemble_max_chunks", 100))

    def reset(self) -> None:
        """Episode start: empty queue, no blend, no ensembling history."""
        self.flush()

    def flush(self) -> None:
        """Drop the queue (stage switch, model switch, retry). Next ``need_chunk`` is True."""
        self.current = None
        self.previous = None
        self.blend_start_tick = 0
        self.history = []
        self.flushed = True

    def need_chunk(self, tick: int) -> bool:
        """True when a new chunk should be requested this tick (empty, flushed or re-plan point).

        Args:
            tick: current control tick.
        """
        if self.current is None or self.flushed:
            return True
        if self.temporal_ensemble and self.current.source.value in self.execute_steps:
            return True
        elapsed = self._elapsed_steps(self.current, tick)
        if elapsed >= len(self.current.actions):
            return True
        limit = self.execute_steps.get(self.current.source.value)
        return limit is not None and elapsed >= int(limit)

    def push(self, chunk: ActionChunk, tick: int) -> None:
        """Make ``chunk`` current; the old one (if any) is blended out over ``blend_ticks``.

        Args:
            chunk: new chunk from the router.
            tick: current control tick.

        Raises:
            ValueError: chunk is not (T, 23) with T ≥ 1.
        """
        acts = np.asarray(chunk.actions)
        if acts.ndim != 2 or acts.shape[0] < 1 or acts.shape[1] != 23:
            raise ValueError(f"chunk actions must be (T, 23), got {acts.shape}")
        if self.current is not None and not self.flushed and self.blend_ticks > 0:
            self.previous = self.current
            self.blend_start_tick = tick
        else:
            self.previous = None
        self.current = chunk
        self.flushed = False
        if self.temporal_ensemble:
            self.history.append(chunk)
            self.history = [c for c in self.history if self._elapsed_steps(c, tick) < len(c.actions)]
            self.history = self.history[-self.max_history :]

    def step(self, tick: int) -> np.ndarray:
        """Model-space action for this tick.

        Functionality:
            Index the current chunk with the time mapping (clamp to the last step if exhausted),
            crossfade continuous dims with the previous chunk while blending (grippers are not
            blended: take the new chunk), apply temporal ensembling if enabled.

        Args:
            tick: current control tick.

        Returns:
            (23,) model-space action.

        Raises:
            RuntimeError: queue empty (caller must push a chunk or use a hold action first).
        """
        if self.current is None:
            raise RuntimeError("chunk executor: no chunk queued")
        cur = self.current
        action = np.asarray(cur.actions[self.chunk_index(cur, tick)], dtype=np.float64).copy()

        if self.temporal_ensemble and len(self.history) > 1:
            live = [c for c in self.history if self._elapsed_steps(c, tick) < len(c.actions)]
            if len(live) > 1:
                rows = np.stack([np.asarray(c.actions[self.chunk_index(c, tick)], dtype=np.float64) for c in live])
                w = np.exp(-self.ensemble_coeff * np.arange(len(live)))  # live is oldest first
                w /= w.sum()
                action[_CONTINUOUS] = (w[:, None] * rows).sum(axis=0)[_CONTINUOUS]

        if self.previous is not None:
            k = tick - self.blend_start_tick
            if 0 <= k < self.blend_ticks:
                alpha = (k + 1) / (self.blend_ticks + 1)  # weight of the new chunk
                old = np.asarray(self.previous.actions[self.chunk_index(self.previous, tick)], dtype=np.float64)
                action[_CONTINUOUS] = alpha * action[_CONTINUOUS] + (1 - alpha) * old[_CONTINUOUS]
            else:
                self.previous = None
        return action.astype(np.float32)

    def chunk_index(self, chunk: ActionChunk, tick: int) -> int:
        """``floor((tick - chunk.start_tick) * chunk.fps / control_hz)``, clamped to the chunk length."""
        return int(min(max(self._elapsed_steps(chunk, tick), 0), len(chunk.actions) - 1))

    def _elapsed_steps(self, chunk: ActionChunk, tick: int) -> int:
        """Chunk steps elapsed since the chunk's observation (unclamped)."""
        return int(np.floor((tick - chunk.start_tick) * chunk.fps / self.control_hz + 1e-9))
