"""PICO SMPL22 -> isolated live UMR -> causal RGMT feedback control."""
from collections import deque
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from threading import Condition, Thread
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .rgmt import RgmtExternalReferencePolicy
from .umr_rgmt import _load_runtime


def causal_window(history, *, paused=False, fps=50., prediction_time_s=0., joint_limits=None):
    """Measured past/current plus held or bounded, decaying future commands.

    Prediction uses only the latest velocity, with a maximum 0.2 rad joint
    offset and 0.1 rad root rotation. It introduces no future-frame wait.
    """
    latest = history[-1][1]
    if paused:
        return (np.repeat(latest[0][None], 21, axis=0), np.zeros((21, 29), np.float32),
                np.repeat(latest[2][None], 21, axis=0), np.zeros((21, 3), np.float32))
    times = np.asarray([row[0] for row in history], dtype=np.float64)
    positions, velocities, quaternions, angular = [np.stack([row[1][i] for row in history]) for i in range(4)]
    sample_times = np.clip(times[-1] + np.arange(-10, 1) / fps, times[0], times[-1])
    def interpolate(values):
        return np.column_stack([np.interp(sample_times, times, column) for column in values.T])
    past_q = (np.repeat(quaternions[-1][None], 11, axis=0) if len(times) == 1 else
              Slerp(times, Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]))(sample_times).as_quat()[:, [3, 0, 1, 2]])
    future_pos = np.repeat(latest[0][None], 10, axis=0)
    future_vel = np.zeros((10, 29))
    future_quat = np.repeat(latest[2][None], 10, axis=0)
    future_angular = np.zeros((10, 3))
    if not np.isfinite(prediction_time_s) or not 0. <= prediction_time_s <= .12:
        raise ValueError("Prediction time must be in [0, 0.12] seconds")
    if prediction_time_s > 0. and len(times) > 1:
        limits = np.asarray(joint_limits, dtype=np.float64)
        if limits.shape != (29, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("Prediction requires 29 finite joint limits")
        future_time = np.arange(1, 11) / fps
        decay = np.exp(-future_time / prediction_time_s)
        effective_time = prediction_time_s * (1. - decay)
        max_speed = min(4., .2 / prediction_time_s)
        velocity = np.clip(latest[1], -max_speed, max_speed)
        future_pos = np.clip(latest[0] + effective_time[:, None] * velocity,
                             limits[:, 0], limits[:, 1])
        future_vel = decay[:, None] * velocity
        saturated = (future_pos <= limits[:, 0] + 1e-6) | (future_pos >= limits[:, 1] - 1e-6)
        future_vel[saturated] = 0.
        predicted_angular = np.asarray(latest[3], dtype=np.float64).copy()
        predicted_angular *= min(1., min(1.5, .1 / prediction_time_s) / max(np.linalg.norm(predicted_angular), 1e-9))
        current = Rotation.from_quat(latest[2][[1, 2, 3, 0]])
        future_quat = (Rotation.from_rotvec(effective_time[:, None] * predicted_angular) * current).as_quat()[:, [3, 0, 1, 2]]
        future_angular = decay[:, None] * predicted_angular
    return tuple(np.asarray(value, dtype=np.float32) for value in (
        np.concatenate((interpolate(positions), future_pos)),
        np.concatenate((interpolate(velocities), future_vel)),
        np.concatenate((past_q, future_quat)),
        np.concatenate((interpolate(angular), future_angular)),
    ))


class PicoUmrRGMT:
    """Freshest-input asynchronous retargeting, with robot feedback every tick."""

    def __init__(self, model_onnx_path, *, robot_xml, runtime_root,
                 reference_timeout_s=.15, reference_yaw_mode="initial", prediction_time_s=.08, **kwargs):
        self.rgmt = RgmtExternalReferencePolicy.for_live_reference(
            str(model_onnx_path), reference_yaw_mode=reference_yaw_mode,
            intra_op_num_threads=kwargs.pop("intra_op_num_threads", 1), **kwargs)
        self._timeout = float(reference_timeout_s)
        if not 0.02 <= self._timeout <= .25:
            raise ValueError("Live UMR reference timeout must be in [0.02, 0.25] seconds")
        self._prediction_time_s = float(prediction_time_s)
        if not np.isfinite(self._prediction_time_s) or not 0. <= self._prediction_time_s <= .12:
            raise ValueError("Prediction time must be in [0, 0.12] seconds")
        runtime = _load_runtime(runtime_root)
        model = runtime.load_robot(robot_xml)
        _qaddrs, _daddrs, ranges, names = runtime.solver.scalar_qpos_joint_addrs(model)
        self._joint_limits = ranges[[names.index(name) for name in self.rgmt.joint_name]].copy()
        for index, name in enumerate(self.rgmt.joint_name):
            if name in ("l_knee_y_joint", "r_knee_y_joint"):
                self._joint_limits[index, 0] = max(.1, self._joint_limits[index, 0])
        self._condition = Condition()
        self._stop = False
        self._epoch = 0
        self._sequence = 0
        self._input = None
        self._result = None
        self._last_submitted_ns = 0
        self._last_consumed_sequence = -1
        self._history = deque(maxlen=32)
        self._paused = False
        self._was_stale = True
        self._processed = 0
        self._compute_seconds = 0.
        self.last_error = None
        self.last_raw_target = None
        config = dict(robot_xml=str(Path(robot_xml).resolve()), metadata=self.rgmt.metadata)
        worker = Path(runtime_root).resolve() / "live_worker.py"
        if not worker.is_file():
            raise FileNotFoundError(f"Live UMR worker missing: {worker}; rebuild the ROS package")
        environment = os.environ.copy()
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "1"
        self._process = subprocess.Popen([sys.executable, "-u", str(worker), json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0, pipesize=4096, env=environment)
        self._writer = Thread(target=self._write_inputs, name="pico-umr-input", daemon=True)
        self._reader = Thread(target=self._read_outputs, name="pico-umr-output", daemon=True)
        self._writer.start()
        self._reader.start()

    def reset(self):
        with self._condition:
            self._epoch += 1
            self._input = self._result = None
            self._last_submitted_ns = 0
            self._last_consumed_sequence = -1
            self._history.clear()
            self._was_stale = True
            self.last_raw_target = None
        self.rgmt.reset()

    def _write_inputs(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or self._input is not None)
                    if self._stop:
                        return
                    data, self._input = self._input, None
                view = memoryview(data)
                while view:
                    count = self._process.stdin.write(view)
                    if not count:
                        raise OSError("Live UMR input pipe closed")
                    view = view[count:]
        except (OSError, ValueError) as error:
            if not self._stop:
                self.last_error = str(error)

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size:
            part = self._process.stdout.read(size - len(data))
            if not part:
                raise OSError("Live UMR worker closed its output")
            data.extend(part)
        return bytes(data)

    def _read_outputs(self):
        try:
            while not self._stop:
                length = struct.unpack("<I", self._read_exact(4))[0]
                if not 1 <= length <= 1024 * 1024:
                    raise ValueError("Invalid live UMR packet length")
                packet = self._read_exact(length)
                if packet[:1] == b"E":
                    raise RuntimeError(packet[1:].decode("utf-8", errors="replace"))
                if packet[:1] == b"W":
                    epoch = struct.unpack("<q", packet[1:9])[0]
                    with self._condition:
                        if epoch == self._epoch:
                            self._result = None
                            self.last_error = packet[9:].decode("utf-8", errors="replace")
                    continue
                if packet[:1] != b"F" or length != 301:
                    raise ValueError("Invalid live UMR reference packet")
                epoch, sequence, source_ns, produced_ns, elapsed = struct.unpack("<qqqqd", packet[1:41])
                values = np.frombuffer(packet[41:], dtype="<f4").copy()
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite live UMR reference")
                reference = (values[:29], values[29:58], values[58:62], values[62:])
                with self._condition:
                    if epoch == self._epoch:
                        self._result = (sequence, source_ns, produced_ns, reference)
                        self._processed += 1
                        self._compute_seconds += elapsed
                        self.last_error = None
        except (OSError, ValueError, RuntimeError) as error:
            if not self._stop:
                with self._condition:
                    self._result = None
                    self.last_error = str(error)

    def inference_step(self, q, dq, quat, omega, body_positions, body_rotations,
                       source_time_s=None, advance=True):
        if advance and self._paused:
            self.reset()
        self._paused = not advance
        positions = np.asarray(body_positions, dtype=np.float32)
        rotations = np.asarray(body_rotations, dtype=np.float32)
        if positions.ndim == 3:
            positions, rotations = positions[-1], rotations[-1]
        if positions.shape != (22, 3) or rotations.shape != (22, 4):
            raise ValueError("PICO -> UMR requires 22 world body transforms")
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError("PICO -> UMR received non-finite transforms")
        source_ns = time.monotonic_ns() if source_time_s is None else int(float(source_time_s) * 1e9)
        now_ns = time.monotonic_ns()
        with self._condition:
            if advance and source_ns > self._last_submitted_ns and now_ns - source_ns < int(self._timeout * 1e9):
                self._sequence += 1
                self._input = struct.pack("<qqq", self._epoch, self._sequence, source_ns)
                self._input += np.concatenate((positions.reshape(-1), rotations.reshape(-1))).astype("<f4", copy=False).tobytes()
                self._last_submitted_ns = source_ns
                self._condition.notify_all()
            result = self._result
        if not advance and self._history:
            window = causal_window(self._history, paused=True, fps=self.rgmt.policy_fps)
        else:
            if result is None or now_ns - result[1] > self._timeout * 1e9:
                if not self._was_stale:
                    self.reset()  # Invalidate contacts and in-flight outputs once per outage.
                self._was_stale = True
                return None
            sequence, sample_ns, _produced_ns, reference = result
            if self._was_stale:
                self._history.clear()
                self.rgmt.reset()
                self._was_stale = False
            if sequence != self._last_consumed_sequence:
                self._history.append((sample_ns * 1e-9, reference))
                self._last_consumed_sequence = sequence
            window = causal_window(self._history, fps=self.rgmt.policy_fps,
                                   prediction_time_s=self._prediction_time_s, joint_limits=self._joint_limits)
        target = self.rgmt.inference_step_with_reference_window(
            q, dq, quat, omega, reference_joint_pos_window=window[0],
            reference_joint_vel_window=window[1], reference_root_quat_window_w=window[2],
            reference_root_ang_vel_window_w=window[3])
        self.last_raw_target = target.copy()
        return target

    def performance_stats(self):
        with self._condition:
            result = self._result
            return dict(backend="process", worker_pid=self._process.pid,
                        generation_hz=self._processed / max(self._compute_seconds, 1e-12),
                        reference_age_ms=None if result is None else (time.monotonic_ns() - result[1]) * 1e-6,
                        solve_to_source_ms=None if result is None else (result[2] - result[1]) * 1e-6,
                        processed=self._processed, prediction_time_ms=self._prediction_time_s * 1000.,
                        error=self.last_error)

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._writer.join(timeout=2.)
        self._reader.join(timeout=2.)
        self._process.stdin.close()
        self._process.stdout.close()


__all__ = ["PicoUmrRGMT", "causal_window"]
