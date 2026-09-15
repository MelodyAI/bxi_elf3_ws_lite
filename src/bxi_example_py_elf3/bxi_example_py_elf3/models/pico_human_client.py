"""Standalone PICO human-pose receiver for neural retargeting.

This module deliberately has no dependency on the GRIT repository.  A small
PICO/XR bridge publishes multipart ZeroMQ messages:

    JSON header, float32 positions [14,3], float32 rotations_wxyz [14,4]

The receiver keeps the latest 20 frames required by neural_retarget.onnx.
"""

from __future__ import annotations

from collections import deque
import json
import time

import numpy as np
from .pico_contract import decode_packet



class PicoHumanPoseClient:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:28704", *, history_size: int = 20):
        import zmq

        self.history_size = int(history_size)
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.connect(str(endpoint))
        self._positions = deque(maxlen=self.history_size)
        self._rotations = deque(maxlen=self.history_size)
        self.last_timestamp_ns = 0
        self.last_receive_ns = 0
        self._received_count = 0
        self.last_error = None

    def poll(self) -> int:
        """Drain the socket and return the number of newly received frames."""
        import zmq

        received = 0
        for _ in range(64):  # Bound work per control tick, even under a flood.
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) != 3:
                continue
            try:
                header, positions, rotations = decode_packet(parts)
            except (ValueError, KeyError, TypeError) as error:
                if str(error) != self.last_error:
                    print(f"PICO rejected: {error}", flush=True)
                    self.last_error = str(error)
                continue
            self.last_error = None
            if self.stale():
                self._positions.clear()
                self._rotations.clear()
            self._positions.append(positions)
            self._rotations.append(rotations)
            self.last_timestamp_ns = int(header.get("recv_ns", 0))
            self.last_receive_ns = time.monotonic_ns()
            received += 1
            self._received_count += 1
            if self._received_count == 1 or self._received_count % 50 == 0:
                print(f"PICO human frames received: {self._received_count}", flush=True)
        return received

    def history(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self._positions) < self.history_size:
            return None
        positions = list(self._positions)
        rotations = list(self._rotations)
        while len(positions) < self.history_size:
            positions.insert(0, positions[0])
            rotations.insert(0, rotations[0])
        return np.stack(positions), np.stack(rotations)

    def stale(self, timeout_s: float = 0.25) -> bool:
        return self.last_receive_ns <= 0 or (time.monotonic_ns() - self.last_receive_ns) > timeout_s * 1e9

    def close(self) -> None:
        self._socket.close(0)


__all__ = ["PicoHumanPoseClient"]
