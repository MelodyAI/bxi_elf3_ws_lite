"""Cache and project ROS depth messages without a camera driver dependency."""
from threading import Lock
import time

import numpy as np

from .depth_projection import ProjectionSpec, project_depth


class DepthCameraInput:
    def __init__(self, *, mode="origin_camera", timeout_s=.5, uint16_scale=.001):
        if mode not in {"origin_camera", "depth_walk"}:
            raise ValueError("Depth mode must be origin_camera or depth_walk")
        if not np.isfinite(timeout_s) or timeout_s <= 0.:
            raise ValueError("Depth timeout must be positive")
        if not np.isfinite(uint16_scale) or uint16_scale <= 0.:
            raise ValueError("Depth unit scale must be positive")
        self.timeout_s = float(timeout_s)
        self.uint16_scale = float(uint16_scale)
        self.projection = (ProjectionSpec(36, 48, 45.2, 58.0616969, .2, 3.) if mode == "origin_camera"
                           else ProjectionSpec(64, 36, 89.24, 58.06, .2, 2.5))
        self.update_period_s = .05 if mode == "origin_camera" else .02
        self._lock = Lock()
        self._camera_info = None
        self._latest = None
        self._frame_id = 0
        self._received_at = None
        self._last_stamp = None
        self._selected = None
        self._next_sample_at = None
        self.last_error = "Waiting for CameraInfo and depth image"
        self.last_crop = None

    def camera_info_callback(self, msg):
        if (msg.width <= 0 or msg.height <= 0 or len(msg.k) != 9
                or not np.isfinite(msg.k).all() or msg.k[0] <= 0. or msg.k[4] <= 0.):
            self.last_error = "Invalid depth CameraInfo"
            return
        with self._lock:
            self._camera_info = msg

    def _decode(self, msg):
        encoding = msg.encoding.lower()
        if encoding in {"16uc1", "mono16"}:
            dtype, scale = np.dtype(np.uint16), self.uint16_scale
        elif encoding == "32fc1":
            dtype, scale = np.dtype(np.float32), 1.
        else:
            raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
        dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
        if (msg.width <= 0 or msg.height <= 0 or msg.step % dtype.itemsize
                or msg.step < msg.width * dtype.itemsize or len(msg.data) < msg.step * msg.height):
            raise ValueError("Invalid or incomplete depth image layout")
        rows = np.frombuffer(msg.data, dtype=dtype, count=msg.step * msg.height // dtype.itemsize)
        depth = rows.reshape(msg.height, msg.step // dtype.itemsize)[:, :msg.width].astype(np.float32)
        depth *= scale
        depth[~np.isfinite(depth) | (depth <= 0.)] = 0.
        if not np.any(depth > 0.):
            raise ValueError("Depth image has no valid pixels")
        return depth

    def image_callback(self, msg):
        with self._lock:
            info = self._camera_info
        if info is None:
            self.last_error = "Waiting for depth CameraInfo"
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        try:
            depth = self._decode(msg)
            projected, crop = project_depth(depth, info, self.projection)
            if not np.any(projected > 0.):
                raise ValueError("Depth ROI has no valid pixels")
            rotated = np.ascontiguousarray(np.rot90(projected, k=-1), dtype=np.float32)
        except ValueError as error:
            self.last_error = str(error)
            return
        now = time.monotonic()
        with self._lock:
            # A duplicated timestamp does not manufacture a fresh camera frame.
            # Accept backwards simulation time after a model reset.
            if stamp != (0, 0) and stamp == self._last_stamp:
                return
            self._last_stamp = stamp
            self._frame_id += 1
            self._latest = (rotated, self._frame_id)
            self._received_at = now
            self.last_crop = crop
            self.last_error = None

    def sample(self, *, advance=True):
        now = time.monotonic()
        with self._lock:
            if self._received_at is None or now - self._received_at > self.timeout_s:
                return None
            if self._selected is None or self._next_sample_at is None or now + 1e-9 >= self._next_sample_at:
                if advance:
                    self._selected = self._latest
                    if self._next_sample_at is None:
                        self._next_sample_at = now + self.update_period_s
                    else:
                        # Advance the scheduled deadline, not the control tick.
                        # At 50Hz this alternates 2/3 ticks for a true 20Hz average.
                        periods = int(max(0., now-self._next_sample_at+1e-9) / self.update_period_s) + 1
                        self._next_sample_at += periods * self.update_period_s
                else:
                    return self._latest
            return self._selected

    def reset_policy_sampling(self):
        with self._lock:
            self._selected = self._next_sample_at = None

    def age_ms(self):
        with self._lock:
            received = self._received_at
        return None if received is None else (time.monotonic() - received) * 1000.
