"""ZeroLab packed-pose receiver for the live neural RGMT path.

ZeroLab publishes the same SMPL pose family used by the SONIC bridge, but its
wire format is a packed topic message rather than the PICO multipart packet.
This adapter keeps the latest twenty frames and reconstructs the fourteen
body transforms expected by ``neural_retarget_fast.onnx``.
"""

from __future__ import annotations

from collections import deque
import json
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


HEADER_SIZE = 1280
_DTYPE_MAP = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("<?"),
}

# SMPL 24-joint parent graph and the exact 14 nodes used by the transformer.
_SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
_NEURAL_BODY_INDICES = np.asarray(
    [0, 9, 1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21],
    dtype=np.int64,
)

# ``zerolab.converter`` publishes ``body_quat_w`` from SONIC's SMPL helper.
# That value is ``root_z_up * SMPL_BASE_INVERSE``.  The neural retarget model
# uses the PICO/SMPL world basis, so recover ``root_z_up`` before composing
# the local body rotations.  This is the inverse of the source helper's
# ``[0.5, -0.5, -0.5, -0.5]`` WXYZ basis quaternion.
_SONIC_ROOT_TO_SMPL = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def _decode_packed_message(message: bytes, topic: str) -> dict[str, np.ndarray] | None:
    prefix = str(topic).encode("utf-8")
    if not message.startswith(prefix):
        return None
    payload = message[len(prefix):]
    if len(payload) < HEADER_SIZE:
        raise ValueError("ZeroLab packed pose message is shorter than its header")
    raw_header = payload[:HEADER_SIZE].split(b"\0", 1)[0]
    if not raw_header:
        raise ValueError("ZeroLab packed pose message has an empty header")
    header: dict[str, Any] = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header.get("fields"), list):
        raise ValueError("ZeroLab packed pose header has no fields list")

    data = memoryview(payload[HEADER_SIZE:])
    result: dict[str, np.ndarray] = {}
    offset = 0
    for field in header["fields"]:
        name = str(field["name"])
        if name in result:
            raise ValueError(f"duplicate ZeroLab field {name!r}")
        dtype = _DTYPE_MAP.get(field["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported ZeroLab field dtype {field['dtype']!r}")
        shape = tuple(int(value) for value in field.get("shape", ()))
        if any(value < 0 for value in shape):
            raise ValueError(f"invalid shape for ZeroLab field {name!r}")
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = dtype.itemsize * count
        if offset + nbytes > len(data):
            raise ValueError(f"ZeroLab field {name!r} exceeds payload bounds")
        result[name] = np.frombuffer(
            data[offset:offset + nbytes], dtype=dtype, count=count
        ).reshape(shape).copy()
        offset += nbytes
    return result


def _smpl_frames_to_transforms(
    smpl_joints: object,
    smpl_body_pose: object,
    body_quat_w: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ZeroLab local SMPL fields to ``[N,14,3]``/``[N,14,4]``."""
    joints = np.asarray(smpl_joints, dtype=np.float32)
    body_pose = np.asarray(smpl_body_pose, dtype=np.float32)
    root_quat = np.asarray(body_quat_w, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"ZeroLab smpl_joints must have shape (N,24,3), got {joints.shape}")
    frames = joints.shape[0]
    if body_pose.shape != (frames, 21, 3):
        raise ValueError(
            f"ZeroLab smpl_body_pose must have shape ({frames},21,3), got {body_pose.shape}"
        )
    if root_quat.shape != (frames, 4):
        raise ValueError(f"ZeroLab body_quat_w must have shape ({frames},4), got {root_quat.shape}")
    if not np.isfinite(joints).all() or not np.isfinite(body_pose).all() or not np.isfinite(root_quat).all():
        raise ValueError("ZeroLab SMPL fields must be finite")

    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(norms < 1.0e-6):
        raise ValueError("ZeroLab body_quat_w contains a zero quaternion")
    root_quat = root_quat / norms
    root_xyzw = root_quat[:, [1, 2, 3, 0]]
    root_matrix = Rotation.from_quat(root_xyzw).as_matrix().astype(np.float32)
    body_local_matrix = Rotation.from_rotvec(body_pose.reshape(-1, 3)).as_matrix()
    body_local_matrix = body_local_matrix.reshape(frames, 21, 3, 3).astype(np.float32)
    local_matrix = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frames, 24, 3, 3)
    ).copy()
    # ``local_matrix[:, joint - 1]`` is the local transform for SMPL joint
    # ``joint``; body-pose element 0 therefore belongs at slot 0 (joint 1).
    local_matrix[:, :21] = body_local_matrix

    world_matrix = np.empty((frames, 24, 3, 3), dtype=np.float32)
    world_matrix[:, 0] = root_matrix @ _SONIC_ROOT_TO_SMPL
    for joint in range(1, 24):
        parent = int(_SMPL_PARENTS[joint])
        world_matrix[:, joint] = world_matrix[:, parent] @ local_matrix[:, joint - 1]

    positions = np.einsum("nij,nkj->nki", root_matrix, joints, dtype=np.float32)
    selected_positions = np.ascontiguousarray(positions[:, _NEURAL_BODY_INDICES], dtype=np.float32)
    selected_matrix = world_matrix[:, _NEURAL_BODY_INDICES]
    selected_xyzw = Rotation.from_matrix(selected_matrix.reshape(-1, 3, 3)).as_quat()
    selected_rotations = np.concatenate(
        (selected_xyzw[:, 3:4], selected_xyzw[:, :3]), axis=1
    ).reshape(frames, 14, 4).astype(np.float32)
    return selected_positions, np.ascontiguousarray(selected_rotations)


class ZeroLabHumanPoseClient:
    """Receive ZeroLab's rolling packed pose chunks over ZMQ."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5558",
        *,
        topic: str = "pose",
        history_size: int = 20,
        stale_timeout_s: float = 0.25,
        _socket=None,
    ) -> None:
        import zmq

        if history_size < 1:
            raise ValueError("history_size must be positive")
        if not np.isfinite(stale_timeout_s) or stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be positive")
        self.history_size = int(history_size)
        self.topic = str(topic)
        self.stale_timeout_s = float(stale_timeout_s)
        self._context = None
        self._owns_socket = _socket is None
        if _socket is None:
            self._context = zmq.Context()
            _socket = self._context.socket(zmq.SUB)
            _socket.setsockopt(zmq.LINGER, 0)
            _socket.setsockopt(zmq.RCVHWM, 64)
            _socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
            _socket.connect(str(endpoint))
        self._socket = _socket
        self._positions: deque[np.ndarray] = deque(maxlen=self.history_size)
        self._rotations: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.last_frame_index: int | None = None
        self.last_timestamp_ns = 0
        self.last_receive_ns = 0
        self.last_error: str | None = None
        self._source_generation: int | None = None
        self._stale = True

    def _clear(self) -> None:
        self._positions.clear()
        self._rotations.clear()
        self.last_frame_index = None

    def _accept_fields(self, fields: dict[str, np.ndarray]) -> int:
        required = ("frame_index", "smpl_joints", "smpl_body_pose", "body_quat_w")
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"ZeroLab pose message is missing {missing[0]!r}")
        indices = np.asarray(fields["frame_index"], dtype=np.int64).reshape(-1)
        joints = np.asarray(fields["smpl_joints"], dtype=np.float32)
        poses = np.asarray(fields["smpl_body_pose"], dtype=np.float32)
        roots = np.asarray(fields["body_quat_w"], dtype=np.float32)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("ZeroLab frame_index must be a non-empty vector")
        if joints.shape != (indices.size, 24, 3) or poses.shape != (indices.size, 21, 3) or roots.shape != (indices.size, 4):
            raise ValueError("ZeroLab pose fields have inconsistent frame counts")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("ZeroLab frame_index must be strictly increasing")
        positions, rotations = _smpl_frames_to_transforms(joints, poses, roots)

        generation = fields.get("source_generation")
        if generation is not None:
            generation_value = int(np.asarray(generation).reshape(-1)[0])
            if self._source_generation is not None and generation_value != self._source_generation:
                self._clear()
            self._source_generation = generation_value
        elif self.last_frame_index is not None and int(indices[-1]) <= self.last_frame_index:
            self._clear()

        appended = 0
        for frame_index, position, rotation in zip(indices, positions, rotations, strict=True):
            frame_index = int(frame_index)
            if self.last_frame_index is not None and frame_index <= self.last_frame_index:
                continue
            self._positions.append(np.ascontiguousarray(position, dtype=np.float32))
            self._rotations.append(np.ascontiguousarray(rotation, dtype=np.float32))
            self.last_frame_index = frame_index
            appended += 1
        if appended:
            self.last_timestamp_ns = time.monotonic_ns()
        return appended

    def poll(self) -> int:
        """Drain available chunks and return the number of new source frames."""
        import zmq

        received = 0
        appended = 0
        for _ in range(64):
            try:
                message = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            received += 1
            self.last_receive_ns = time.monotonic_ns()
            try:
                fields = _decode_packed_message(message, self.topic)
                if fields is None:
                    continue
                stale = int(np.asarray(fields.get("source_stale", [0])).reshape(-1)[0]) != 0
                if stale:
                    self._clear()
                    self._stale = True
                    continue
                ready = fields.get("real_stream_ready")
                if ready is not None and int(np.asarray(ready).reshape(-1)[0]) == 0:
                    self._clear()
                    self._stale = True
                    continue
                appended += self._accept_fields(fields)
            except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as error:
                message_text = str(error)
                if message_text != self.last_error:
                    print(f"ZeroLab rejected: {message_text}", flush=True)
                    self.last_error = message_text
                continue
            self.last_error = None
        if appended:
            self._stale = False
        elif received == 0 and self.stale():
            self._clear()
            self._stale = True
        return appended

    def history(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self._positions) < self.history_size:
            return None
        return np.stack(tuple(self._positions)), np.stack(tuple(self._rotations))

    def stale(self, timeout_s: float | None = None) -> bool:
        timeout = self.stale_timeout_s if timeout_s is None else float(timeout_s)
        if self._stale or self.last_receive_ns <= 0:
            return True
        return time.monotonic_ns() - self.last_receive_ns > timeout * 1.0e9

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._owns_socket and self._context is not None:
            self._context.term()


__all__ = [
    "HEADER_SIZE",
    "ZeroLabHumanPoseClient",
    "_decode_packed_message",
    "_smpl_frames_to_transforms",
]
