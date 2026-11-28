"""Client for an Isaac-GR00T inference server (``gr00t/eval/run_gr00t_server.py``).

Purpose
-------
Isaac-GR00T cannot share the policy image's Python environment (it pins numpy 1.26 and torch 2.9;
the task repo needs numpy 2). The model therefore runs in Isaac-GR00T's own environment as a
server, and ``Gr00tPolicyBackend`` talks to it through this client, which needs only pyzmq,
msgpack and msgpack-numpy. ``scripts/start_groot_server.sh`` starts the server.

Protocol (``gr00t/policy/server_client.py`` at ISAAC_GR00T_COMMIT)
-----------------------------------------------------------------
ZeroMQ REQ/REP. A request is ``{"endpoint": name, "data": {...}}`` packed with msgpack, numpy arrays
as msgpack-numpy payloads. Endpoints: ``ping``, ``get_modality_config`` (no data), ``get_action``
(data ``{"observation", "options"}``, reply ``[action, info]``), ``reset`` (data ``{}``). Errors come
back as ``{"error": message}``. Modality configs arrive as ``{"__ModalityConfig__": True,
"as_json": {...}}`` and become namespaces with the same attributes (``modality_keys``,
``delta_indices``, ...). Object-dtype arrays are refused in both directions, as the server does.

The interface mirrors ``Gr00tPolicy`` where ``Gr00tPolicyBackend`` uses it: ``get_modality_config``,
``get_action`` (returns ``(action, info)``) and ``reset``.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _encode(obj: Any) -> Any:
    import msgpack_numpy as mnp

    if isinstance(obj, np.ndarray) and obj.dtype.hasobject:
        raise TypeError(f"refusing to send an object-dtype array (shape {obj.shape})")
    return mnp.encode(obj)


def _decode(obj: Any) -> Any:
    import msgpack_numpy as mnp

    if not isinstance(obj, dict):
        return obj
    if obj.get("__ModalityConfig__") or obj.get("__ModalityConfig_class__"):
        payload = obj.get("as_json")
        if isinstance(payload, bytes):
            payload = payload.decode()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("malformed ModalityConfig payload from the GR00T server")
        return SimpleNamespace(**payload)
    if b"nd" in obj and obj.get(b"kind") in (b"O", "O"):
        raise ValueError("refusing to decode an object-dtype array from the GR00T server")
    return mnp.decode(obj)


def pack(data: Any) -> bytes:
    import msgpack

    return msgpack.packb(data, default=_encode)


def unpack(message: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(message, object_hook=_decode, raw=False)


class GrootServerClient:
    """REQ client for one GR00T inference server.

    Attributes:
        address: ``host:port`` of the server.
        timeout_s: per-request send/receive timeout.
    """

    def __init__(self, address: str, timeout_s: float = 15.0) -> None:
        import zmq

        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"GR00T server address must be host:port, got {address!r}")
        self.address = address
        self.timeout_s = float(timeout_s)
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = None
        self._connect()

    def _connect(self) -> None:
        zmq = self._zmq
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        ms = int(self.timeout_s * 1000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, ms)
        self._socket.setsockopt(zmq.SNDTIMEO, ms)
        self._socket.connect(f"tcp://{self.address}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """Send one request; raise RuntimeError on a server error, TimeoutError on no reply."""
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(pack(request))
            reply = unpack(self._socket.recv())
        except self._zmq.error.Again as exc:
            self._connect()  # a REQ socket that missed its reply cannot send again
            raise TimeoutError(f"GR00T server {self.address}: no reply to {endpoint} in {self.timeout_s:.0f} s") from exc
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"GR00T server {self.address}: {reply['error']}")
        return reply

    def wait_ready(self, wait_s: float) -> None:
        """Ping until the server answers (it loads the model before it listens)."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                self.call("ping")
                return
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"GR00T server {self.address} did not answer within {wait_s:.0f} s "
                        "(start it with scripts/start_groot_server.sh)"
                    ) from None
                log.info("waiting for the GR00T server at %s", self.address)

    def get_modality_config(self) -> dict[str, Any]:
        return self.call("get_modality_config")

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> tuple[Any, Any]:
        action, info = self.call("get_action", {"observation": observation, "options": options})
        return action, info

    def reset(self) -> None:
        self.call("reset", {})

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._context.term()
