"""Common interface for every action-chunk producer (ACT, GR00T, scripted primitive).

Purpose
-------
The router treats all producers the same way: load once, reset per episode (and on stage switches),
and predict a chunk from the current inputs. Each producer is responsible for its own
normalisation / un-normalisation, so chunks always leave layer 4 in physical units.

Inputs:  producer-specific view (ActView, Gr00tView or CanonicalObs) + orchestrator output.
Outputs: ``ActionChunk`` (T, 24) with source tag, time base (fps) and start tick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from astra.types import ActionChunk, CanonicalObs, OrchestratorOutput, PolicySource


class PolicyBackend(ABC):
    """Abstract chunk producer.

    Attributes:
        source: tag carried on every chunk (act / groot / script).
        fps: time base of produced chunks (30 for learned policies, 50 for scripts).
        loaded: True after ``load()`` succeeded.
    """

    source: PolicySource
    fps: float

    def __init__(self) -> None:
        self.loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load weights / build the producer. Called once from ``AstraPolicy.get_model``."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (temporal ensembling buffers, recurrent state, script progress)."""

    @abstractmethod
    def predict(self, obs: CanonicalObs, view: Any, orch: OrchestratorOutput) -> ActionChunk:
        """Predict an action chunk.

        Args:
            obs: canonical observation (for tick, current state; scripts use it directly).
            view: producer-specific input (ActView for ACT, Gr00tView for GR00T, None for scripts).
            orch: orchestrator output (stage, prompt).

        Returns:
            ActionChunk in the 24-D model space, physical units, absolute targets.
        """
