"""Time-indexed ring buffer of canonical observations.

Purpose
-------
Several consumers need the recent past, not just the current tick:
- the stage classifier (GRU over the last ~3 s, sampled at 5 Hz),
- the event detectors (gripper settle time, contact dwell time),
- the chunk executor's chunk time ↔ control tick mapping (needs the tick a chunk was requested at).
Entries are keyed by **sim time** (tick / control_hz), never wall-clock time, because the env steps
synchronously with the policy.

Inputs:  one ``CanonicalObs`` per tick via ``push``.
Outputs: the latest entry, or entries resampled at a given rate over a window.
"""

from __future__ import annotations

from collections import deque

from astra.types import CONTROL_HZ, CanonicalObs


class HistoryBuffer:
    """Fixed-length FIFO of CanonicalObs.

    Attributes:
        capacity: maximum number of ticks kept (``history_seconds × control_hz``).
        control_hz: tick rate used to convert between ticks and seconds.
        buffer: deque of CanonicalObs, oldest first.
    """

    def __init__(self, history_seconds: float = 3.0, control_hz: float = CONTROL_HZ) -> None:
        self.control_hz = control_hz
        self.capacity = int(round(history_seconds * control_hz))
        self.buffer: deque[CanonicalObs] = deque(maxlen=self.capacity)

    def reset(self) -> None:
        """Clear all entries. Called at every episode reset."""
        self.buffer.clear()

    def push(self, obs: CanonicalObs) -> None:
        """Append the observation of the current tick.

        Args:
            obs: canonical observation; ``obs.tick`` must be strictly increasing.

        Raises:
            ValueError: ``obs.tick`` does not increase.
        """
        if self.buffer and obs.tick <= self.buffer[-1].tick:
            raise ValueError(f"history: tick {obs.tick} after {self.buffer[-1].tick}")
        self.buffer.append(obs)

    def latest(self) -> CanonicalObs:
        """Most recent observation.

        Raises:
            IndexError: buffer empty.
        """
        if not self.buffer:
            raise IndexError("history buffer is empty")
        return self.buffer[-1]

    def __len__(self) -> int:
        return len(self.buffer)

    def window(self, seconds: float, rate_hz: float) -> list[CanonicalObs]:
        """Observations over the last ``seconds``, resampled at ``rate_hz`` (nearest tick).

        Functionality:
            Pick ticks ``now - k * control_hz / rate_hz`` for k = n-1 … 0. If the buffer is shorter
            than the window (just after reset), pad by repeating the oldest entry so the
            classifier always sees a fixed-length sequence.

        Args:
            seconds: window length.
            rate_hz: sampling rate (e.g. 5 Hz for the classifier).

        Returns:
            list of length ``round(seconds * rate_hz)``, oldest first.

        Raises:
            IndexError: buffer empty.
        """
        if not self.buffer:
            raise IndexError("history buffer is empty")
        n = max(1, int(round(seconds * rate_hz)))
        step = self.control_hz / rate_hz
        now = self.buffer[-1].tick
        oldest = self.buffer[0].tick
        out: list[CanonicalObs] = []
        for k in range(n - 1, -1, -1):
            want = now - k * step
            if want <= oldest:
                out.append(self.buffer[0])
                continue
            # Ticks are consecutive except after invalid observations (repeats keep their tick), so
            # index by offset from the newest entry and correct if a tick is missing.
            idx = len(self.buffer) - 1 - int(round(now - want))
            idx = min(max(idx, 0), len(self.buffer) - 1)
            out.append(self.buffer[idx])
        return out
