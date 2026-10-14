"""Asynchronous streaming and offline UMR adapters for raw ELF3 motions."""
from pathlib import Path
from collections import deque
import importlib.util
import json
import os
import struct
import subprocess
import sys
from threading import Condition, Thread

import numpy as np

from .rgmt import RgmtExternalReferencePolicy


def _load_runtime(runtime_root):
    root = Path(runtime_root).expanduser().resolve()
    name = "_bxi_elf3_umr"
    if name in sys.modules:
        loaded = sys.modules[name]
        if Path(loaded.__file__).parent != root:
            raise RuntimeError("An ELF3 UMR runtime from a different package is already loaded")
    else:
        if not (root / "__init__.py").is_file():
            raise FileNotFoundError(f"Bundled UMR runtime not found: {root}; rebuild the ROS package")
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return importlib.import_module(f"{name}.runtime")


class UmrRGMT(RgmtExternalReferencePolicy):
    """Retarget once at startup, then use the existing RGMT reference player.

    Raw BVH or SMPL-X input is converted to the exact ONNX joint/body contract.
    Repeated loads reuse a content-addressed cache outside the installed package.
    No UMR code runs in ``inference_step`` or in the motor control loop.
    """

    def __init__(self, source_path, model_onnx_path, *, robot_xml, runtime_root,
                 body_model_path=None, cache_dir=None, max_frames=0, **kwargs):
        super().__init__(None, model_onnx_path, **kwargs)
        runtime = _load_runtime(runtime_root)
        reference = runtime.prepare_reference(
            source_path, robot_xml, self.metadata, body_model_path=body_model_path,
            cache_dir=cache_dir, max_frames=max_frames)
        self.umr_source_path = str(source_path)
        self.load_motion_npz(str(reference))


class UmrStreamingRGMT(RgmtExternalReferencePolicy):
    """Generate a bounded RGMT window in an isolated process while playing.

    The control thread never solves UMR, waits for a worker, or reads a cache.
    The IPC reader thread only copies packets; numeric computation has its own GIL.
    Missing initial data returns None; a later underrun holds the last window
    while continuing RGMT feedback, and does not advance motion time.
    """

    def __init__(self, source_path, model_onnx_path, *, robot_xml, runtime_root,
                 body_model_path=None, buffer_frames=100, reserve_frames=25, **kwargs):
        kwargs.setdefault("intra_op_num_threads", 1)
        if kwargs.get("reference_joint_order", "isaac") != "isaac":
            raise ValueError("Streaming UMR references use the ONNX/Isaac joint order")
        super().__init__(None, model_onnx_path, **kwargs)
        runtime = _load_runtime(runtime_root)
        self._streaming = importlib.import_module(f"{runtime.__package__}.streaming")
        self._source, self._assets, self._settings = self._streaming.load_source(source_path, body_model_path)
        if isinstance(buffer_frames, bool) or int(buffer_frames) < 21:
            raise ValueError("UMR buffer_frames must be at least 21")
        self._capacity = int(buffer_frames)
        if isinstance(reserve_frames, bool) or int(reserve_frames) < 0:
            raise ValueError("UMR reserve_frames must be non-negative")
        self._reserve = min(int(reserve_frames), self._capacity - 21)
        self.umr_source_path = str(source_path)
        self._robot_xml = str(robot_xml)
        self._body_model_path = None if body_model_path is None else str(body_model_path)
        self._runtime_root = str(Path(runtime_root).resolve())
        self._process = None
        self._recovering = True
        self.end_frame = self._streaming.reference_frame_count(self._source, self.policy_fps) - 1
        self._condition = Condition()
        self._frames = deque()
        self._base_frame = 0
        self._worker_epoch = 0
        self._worker_stop = False
        self._worker_error = None
        self._worker_done = False
        self._last_window = None
        self._last_window_frame = None
        self._generated_count = 0
        self._generation_seconds = 0.
        self._played_count = 0
        self._underrun_count = 0
        self._initial_wait_count = 0
        self.buffering = True
        self._worker = Thread(target=self._worker_loop, name="umr-retarget", daemon=True)
        self._worker.start()

    @property
    def buffered_frames(self):
        with self._condition:
            return len(self._frames)

    def performance_stats(self):
        """Generation rate excludes intentional waiting for a full buffer."""
        with self._condition:
            return dict(generation_hz=self._generated_count / max(self._generation_seconds, 1e-12),
                        buffered_frames=len(self._frames), played_frames=self._played_count,
                        underruns=self._underrun_count, initial_waits=self._initial_wait_count,
                        backend="process", worker_pid=None if self._process is None else self._process.pid)

    def _iter_worker_references(self, cancelled):
        environment = os.environ.copy()
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[variable] = "1"
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(self._runtime_root) / "worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment, bufsize=0, pipesize=4096)
        with self._condition:
            if cancelled():
                process.terminate()
            else:
                self._process = process

        def read_exact(count):
            chunks = bytearray()
            while len(chunks) < count:
                data = process.stdout.read(count - len(chunks))
                if not data:
                    if cancelled():
                        return None
                    raise RuntimeError(f"UMR process closed its output (exit={process.poll()})")
                chunks.extend(data)
            return bytes(chunks)

        try:
            if cancelled():
                return
            configuration = dict(source=self.umr_source_path, body_model=self._body_model_path,
                                 robot_xml=self._robot_xml, metadata=self.metadata)
            process.stdin.write((json.dumps(configuration) + "\n").encode("utf-8"))
            process.stdin.close()
            while not cancelled():
                header = read_exact(4)
                if header is None:
                    return
                length = struct.unpack("<I", header)[0]
                if not 1 <= length <= 1024 * 1024:
                    raise RuntimeError("Invalid UMR worker packet length")
                packet = read_exact(length)
                if packet is None:
                    return
                if packet[:1] == b"D":
                    return
                if packet[:1] == b"E":
                    raise RuntimeError(packet[1:].decode("utf-8", errors="replace"))
                if packet[:1] != b"F" or len(packet) != 273:
                    raise RuntimeError("Invalid UMR worker reference packet")
                index, elapsed = struct.unpack("<id", packet[1:13])
                values = np.frombuffer(packet[13:], dtype="<f4").copy()
                yield index, (values[:29], values[29:58], values[58:62], values[62:]), elapsed
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
            with self._condition:
                if self._process is process:
                    self._process = None

    def _worker_loop(self):
        previous_epoch = -1
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._worker_stop or self._worker_epoch != previous_epoch)
                if self._worker_stop:
                    return
                epoch = self._worker_epoch
                previous_epoch = epoch
            def cancelled():
                return self._worker_stop or self._worker_epoch != epoch
            try:
                references = iter(self._iter_worker_references(cancelled))
                while not cancelled():
                    try:
                        index, reference, compute_seconds = next(references)
                    except StopIteration:
                        break
                    with self._condition:
                        self._condition.wait_for(lambda: cancelled() or len(self._frames) < self._capacity)
                        if cancelled():
                            break
                        if index != self._base_frame + len(self._frames):
                            raise RuntimeError("UMR worker produced a nonconsecutive reference frame")
                        self._frames.append(reference)
                        self._generated_count += 1
                        self._generation_seconds += compute_seconds
                        self._condition.notify_all()
                references.close()
                with self._condition:
                    if not cancelled():
                        self._worker_done = True
                        self._condition.notify_all()
            except Exception as error:
                with self._condition:
                    if not cancelled():
                        self._worker_error = error
                        self._condition.notify_all()
            finally:
                if 'references' in locals():
                    references.close()

    def reset(self, *, start_frame=None):
        start_frame = self.start_frame if start_frame is None else start_frame
        if start_frame != 0:
            raise ValueError("Streaming UMR replays from source frame zero")
        with self._condition:
            restart = self._timestep > 0 or self._base_frame > 0 or self._worker_error is not None
            super().reset(start_frame=0)
            self._last_window = None
            self._last_window_frame = None
            self.buffering = True
            self._recovering = True
            if restart:
                self._worker_epoch += 1
                if self._process is not None and self._process.poll() is None:
                    self._process.terminate()
                self._frames.clear()
                self._base_frame = 0
                self._worker_error = None
                self._worker_done = False
                self._condition.notify_all()

    @RgmtExternalReferencePolicy.timestep.setter
    def timestep(self, value):
        if isinstance(value, (int, np.integer)) and value == 0 and self._timestep > 0:
            self.reset(start_frame=0)
        else:
            RgmtExternalReferencePolicy.timestep.fset(self, value)

    def inference_step(self, q, dq, quat, omega, base_lin_vel_b=None, robot_pos_w=None, advance=False):
        del base_lin_vel_b, robot_pos_w
        with self._condition:
            if self._worker_error is not None:
                raise RuntimeError(f"UMR background retargeting failed: {self._worker_error}") from self._worker_error
            indices = np.clip(np.arange(self._timestep - 10, self._timestep + 11), 0, self.end_frame)
            ready = (len(self._frames) > 0 and indices[0] >= self._base_frame and
                     indices[-1] < self._base_frame + len(self._frames))
            if not ready:
                self._recovering = True
            elif self._recovering:
                resume_end = min(self.end_frame, int(indices[-1]) + self._reserve)
                ready = resume_end < self._base_frame + len(self._frames)
                if ready:
                    self._recovering = False
            self.buffering = not ready
            if ready and self._last_window_frame != self._timestep:
                rows = [self._frames[int(i) - self._base_frame] for i in indices]
                self._last_window = tuple(np.stack([row[field] for row in rows]) for field in range(4))
                self._last_window_frame = self._timestep
            window = self._last_window
            if window is None:
                self._initial_wait_count += 1
                return None
            if advance and not ready:
                self._underrun_count += 1
        if not advance or not ready:
            # Holding a pose also requires zero desired velocity and a constant
            # horizon; retaining a moving future window would still cue motion.
            window = (np.repeat(window[0][10:11], 21, axis=0), np.zeros_like(window[1]),
                      np.repeat(window[2][10:11], 21, axis=0), np.zeros_like(window[3]))
        target = self.inference_step_with_reference_window(
            q, dq, quat, omega, reference_joint_pos_window=window[0],
            reference_joint_vel_window=window[1], reference_root_quat_window_w=window[2],
            reference_root_ang_vel_window_w=window[3])
        if advance and ready:
            with self._condition:
                if self._timestep < self.end_frame:
                    self._played_count += 1
                self._timestep = min(self._timestep + 1, self.end_frame)
                # Retain the ten past command frames, and only bounded future data.
                minimum = max(0, self._timestep - 10)
                while self._base_frame < minimum:
                    self._frames.popleft()
                    self._base_frame += 1
                self._condition.notify_all()
        return target

    def close(self):
        with self._condition:
            self._worker_stop = True
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
            self._condition.notify_all()
        self._worker.join(timeout=2.)


__all__ = ["UmrRGMT", "UmrStreamingRGMT"]
