"""Chunk executor: plays action chunks back one control tick at a time.

Purpose
-------
Learned chunks are in **30 fps data time**; the env runs at **50 Hz**. The executor maps control ticks
to chunk indices, decides when to request a new chunk (re-plan point), blends the old and new chunk
at a switch, optionally applies ACT temporal ensembling (continuous dims only), and hands one
model-space action per tick to the safety filter.

Time mapping
------------
For a chunk predicted at tick ``n0`` with time base ``fps``: at tick ``n`` use index
``floor((n - n0) * fps / control_hz)``. Repeating an absolute EEF target for a tick is harmless
(DiffIK keeps tracking it), so no interpolation is needed. Scripted chunks use fps = control rate.

Re-planning
-----------
``need_chunk`` becomes True when the executed index reaches ``execute_steps[source]`` (ACT ≈ 15,
GR00T 8-16 of 40), when the queue is empty, or after a flush.

Inputs:  ActionChunk from the router, flush flag, current tick.
Outputs: one (24,) model-space action per tick; ``need_chunk`` flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.types import CONTROL_HZ, ActionChunk


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
        self.execute_steps: dict[str, int] = dict(cfg.get("execute_steps", {"act": 15, "groot": 12}))
        self.temporal_ensemble: bool = bool(cfg.get("temporal_ensemble", False))
        self.ensemble_coeff: float = float(cfg.get("ensemble_coeff", 0.01))
        self.blend_ticks: int = int(cfg.get("blend_ticks", 5))
        self.current: ActionChunk | None = None
        self.previous: ActionChunk | None = None
        self.blend_start_tick: int = 0
        self.history: list[ActionChunk] = []

    def reset(self) -> None:
        """Episode start: empty queue, no blend, no ensembling history."""
        raise NotImplementedError("ChunkExecutor.reset")

    def flush(self) -> None:
        """Drop the queue (stage switch, model switch, retry). Next ``need_chunk`` is True."""
        raise NotImplementedError("ChunkExecutor.flush")

    def need_chunk(self, tick: int) -> bool:
        """True when a new chunk should be requested this tick (empty, flushed or re-plan point).

        Args:
            tick: current control tick.
        """
        raise NotImplementedError("ChunkExecutor.need_chunk")

    def push(self, chunk: ActionChunk, tick: int) -> None:
        """Make ``chunk`` current; the old one (if any) is blended out over ``blend_ticks``.

        Args:
            chunk: new chunk from the router.
            tick: current control tick.
        """
        raise NotImplementedError("ChunkExecutor.push")

    def step(self, tick: int) -> np.ndarray:
        """Model-space action for this tick.

        Functionality:
            Index the current chunk with the time mapping (clamp to the last step if exhausted),
            crossfade continuous dims with the previous chunk while blending (grippers are not
            blended: take the new chunk), apply temporal ensembling if enabled.

        Args:
            tick: current control tick.

        Returns:
            (24,) model-space action.

        Raises:
            RuntimeError: queue empty (caller must push a chunk or use a hold action first).
        """
        raise NotImplementedError("ChunkExecutor.step")

    def chunk_index(self, chunk: ActionChunk, tick: int) -> int:
        """``floor((tick - chunk.start_tick) * chunk.fps / control_hz)``, clamped to the chunk length."""
        raise NotImplementedError("ChunkExecutor.chunk_index")
