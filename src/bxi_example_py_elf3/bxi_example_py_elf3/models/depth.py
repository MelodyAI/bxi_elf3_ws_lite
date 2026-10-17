"""ELF3 depth locomotion ONNX adapter, using the existing inference_step API.

Observation, preprocessing and temporal selection follow com.bxi.normal_depth.
Robot measurements and returned targets use this project's MuJoCo joint order.
"""
from collections import deque

import numpy as np
import onnxruntime as ort
from scipy.ndimage import convolve1d

from .rgmt import MUJOCO_TO_ISAAC_INDEX, ISAAC_TO_MUJOCO_INDEX
from ..utils.tfs import get_gravity_orientation


class HumanoidGaitDepthPolicy:
    def __init__(self, model_onnx_path, *, mode="origin_camera", crop_rows=None, depth_range=None):
        if mode not in {"origin_camera", "depth_walk"}:
            raise ValueError("Depth mode must be origin_camera or depth_walk")
        self.mode = mode
        self.model_onnx_path = str(model_onnx_path)
        self.default_dof_pos = np.array([
            0., 0., 0., -.3, 0., 0., .6, -.3, 0., -.3, 0., 0., .6, -.3, 0.,
            .2, .2, 0., .6, 0., 0., 0., .2, -.2, 0., .6, 0., 0., 0.,
        ], dtype=np.float32)
        self.kps = np.array([
            108.448, 162.672, 176.421,
            176.421, 176.421, 54.224, 176.421, 33.493, 21.771,
            176.421, 176.421, 54.224, 176.421, 33.493, 21.771,
            54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
            54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
        ], dtype=np.float32)
        self.kds = np.array([
            6.904, 10.356, 11.231,
            11.231, 11.231, 3.452, 11.231, 2.132, 1.386,
            11.231, 11.231, 3.452, 11.231, 2.132, 1.386,
            3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
            3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
        ], dtype=np.float32)
        # Action scales are in Isaac order, as in the reference policy.
        self.action_scale = np.array([
            .231, .231, .231, .231, .231, .154, .373, .373, .213,
            .231, .231, .213, .213, .373, .373, .213, .213, .373,
            .373, .231, .231, .373, .373, .213, .213, .373, .373, .230, .230,
        ], dtype=np.float32)
        origin = mode == "origin_camera"
        self.depth_update_period = .05 if origin else .02
        self.depth_shape = (36, 36) if origin else (21, 32)
        self.received_depth_shape = (36, 48) if origin else (64, 36)
        self.depth_range = tuple(float(value) for value in (depth_range if depth_range is not None
                                                           else (.2, 3. if origin else 2.5)))
        if (len(self.depth_range) != 2 or not np.isfinite(self.depth_range).all()
                or not 0. <= self.depth_range[0] < self.depth_range[1]):
            raise ValueError('Depth range must contain two finite increasing nonnegative values')
        self.crop_rows = (2, 38) if origin and crop_rows is None else (tuple(crop_rows) if crop_rows is not None else None)
        if self.crop_rows is not None:
            if (not origin or len(self.crop_rows) != 2
                    or any(isinstance(row, bool) or not isinstance(row, (int, np.integer)) for row in self.crop_rows)
                    or not 0 <= self.crop_rows[0] < self.crop_rows[1] <= 48
                    or self.crop_rows[1]-self.crop_rows[0] != 36):
                raise ValueError('origin_camera crop_rows must select 36 rows within [0,48)')
        self.depth_indices = np.array([-1] if origin else [-36, -31, -26, -21, -16, -11, -6, -1])
        self.depth_history_length = 1 if origin else 37
        self.command_lower = np.array([0., -.5 if origin else 0., -1.57])
        self.command_upper = np.array([.8 if origin else 1., .5 if origin else 0., 1.57])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(self.model_onnx_path, sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self._configure_model_io()
        kernel = np.exp(-.5 * np.arange(-1, 2, dtype=np.float64) ** 2)
        self._gaussian_kernel = kernel / kernel.sum()
        self.reset()
        # Warm the backend without inventing measured history or advancing h_in.
        feeds = {self.obs_input: np.zeros((1, 960), np.float32),
                 self.depth_input: np.zeros(self._depth_input_shape, np.float32)}
        if self.recurrent_input is not None:
            feeds[self.recurrent_input] = self._hidden
        for _ in range(3):
            self.session.run(None, feeds)

    def _configure_model_io(self):
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        by_name = {item.name: item for item in inputs}
        if not {"obs_history", "depth_data"}.issubset(by_name):
            raise ValueError("Depth ONNX requires obs_history and depth_data")
        if len(inputs) not in (2, 3) or by_name["obs_history"].shape[1:] != [960]:
            raise ValueError("Depth ONNX must use 10 x 96 proprio history")
        self.obs_input, self.depth_input = "obs_history", "depth_data"
        expected = list(self.depth_shape) if self.mode == "origin_camera" else [8, *self.depth_shape]
        if by_name[self.depth_input].shape[1:] != expected:
            raise ValueError(f"Depth ONNX shape does not match mode {self.mode}")
        if any(item.type != "tensor(float)" for item in inputs):
            raise ValueError("Depth ONNX inputs must be float32")
        output_map = {item.name: item for item in outputs}
        if "actions" not in output_map or output_map["actions"].shape[-1] != 29:
            raise ValueError("Depth ONNX must output 29 actions")
        self._depth_input_shape = (1, *expected)
        self.recurrent_input = "h_in" if len(inputs) == 3 else None
        self.recurrent_output = "h_out" if self.recurrent_input else None
        self._hidden_shape = None
        if self.recurrent_input:
            if "h_in" not in by_name or "h_out" not in output_map:
                raise ValueError("Recurrent depth ONNX requires h_in and h_out")
            def resolve(shape):
                return tuple(1 if isinstance(x, str) and "batch" in x.lower() else x for x in shape)
            self._hidden_shape = resolve(by_name["h_in"].shape)
            if (self._hidden_shape != resolve(output_map["h_out"].shape)
                    or not all(isinstance(x, int) and x > 0 for x in self._hidden_shape)):
                raise ValueError("Invalid recurrent depth state shape")

    def reset(self):
        self.obs_history = deque(maxlen=10)
        self.depth_history = deque(maxlen=self.depth_history_length)
        self.last_action = np.zeros(29, np.float32)
        self.last_depth_frame_id = None
        self.action = np.zeros(29, np.float32)
        self.target_dof_pos = self.default_dof_pos.copy()
        self.obs = np.zeros((1, 960), np.float32)
        self.depth_input_buffer = np.zeros(self._depth_input_shape, np.float32)
        self._hidden = None if self._hidden_shape is None else np.zeros(self._hidden_shape, np.float32)
        self.last_error = None

    def preprocess_depth(self, depth_image):
        image = np.asarray(depth_image, dtype=np.float32)
        if image.shape != self.received_depth_shape or not np.isfinite(image).all() or np.any(image < 0.):
            raise ValueError(f"Expected finite nonnegative depth {self.received_depth_shape} in metres")
        # Receiver rotates clockwise; policy reverses it, matching the Mod.
        image = np.flip(image.T, axis=0)
        if self.mode == "depth_walk":
            image = image[6:27, 16:48]
        # Exact 3x3, sigma=1 Gaussian with OpenCV BORDER_REFLECT_101 semantics.
        image = convolve1d(image, self._gaussian_kernel, axis=0, mode="mirror")
        image = convolve1d(image, self._gaussian_kernel, axis=1, mode="mirror")
        minimum, maximum = self.depth_range
        image = (np.clip(image, minimum, maximum) - minimum) / (maximum - minimum)
        if self.mode == "origin_camera":
            image = (image - .5)[self.crop_rows[0]:self.crop_rows[1], :]
        if image.shape != self.depth_shape:
            raise ValueError(f"Processed depth shape mismatch: {image.shape}")
        return np.ascontiguousarray(image, dtype=np.float32)

    def inference_step(self, q, dq, quat, omega, cmd_vel, depth_image,
                       depth_frame_id=None, *, advance=True):
        def values(value, size):
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"Expected {size} finite values")
            return array
        q, dq = values(q, 29), values(dq, 29)
        quat, omega, command = values(quat, 4), values(omega, 3), values(cmd_vel, 3)
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            raise ValueError("Depth policy received zero quaternion")
        command = np.clip(command, self.command_lower, self.command_upper)
        observation = np.concatenate((omega, get_gravity_orientation(quat / norm), command,
            (q - self.default_dof_pos)[MUJOCO_TO_ISAAC_INDEX], dq[MUJOCO_TO_ISAAC_INDEX], self.last_action))
        observation = np.clip(observation, -100., 100.).astype(np.float32)
        history = list(self.obs_history)
        history = history[1:] + [observation] if history else [observation] * 10
        obs = np.stack(history).reshape(1, 960)
        changed = not self.depth_history or depth_frame_id is None or depth_frame_id != self.last_depth_frame_id
        depth_history = list(self.depth_history)
        if changed:
            depth = self.preprocess_depth(depth_image)
            depth_history = depth_history[1:] + [depth] if depth_history else [depth] * self.depth_history_length
        selected = np.stack([depth_history[index] for index in self.depth_indices])
        depth_input = selected.reshape(self._depth_input_shape)
        feeds = {self.obs_input: obs, self.depth_input: depth_input}
        if self.recurrent_input:
            feeds[self.recurrent_input] = self._hidden
        output_names = ["actions"] + ([self.recurrent_output] if self.recurrent_output else [])
        results = self.session.run(output_names, feeds)
        action = np.asarray(results[0], dtype=np.float32).reshape(-1)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Depth ONNX returned invalid actions")
        if self.recurrent_output and not np.isfinite(results[1]).all():
            raise ValueError("Depth ONNX returned invalid hidden state")
        action = np.clip(action, -100., 100.)
        target = self.default_dof_pos + (action * self.action_scale)[ISAAC_TO_MUJOCO_INDEX]
        if advance:
            self.obs_history.clear()
            self.obs_history.extend(history)
            if changed:
                self.depth_history.clear()
                self.depth_history.extend(depth_history)
                self.last_depth_frame_id = depth_frame_id
            self.last_action = action.copy()
            if self.recurrent_output:
                np.copyto(self._hidden, results[1])
            self.action, self.target_dof_pos = action.copy(), target.copy()
            self.obs, self.depth_input_buffer = obs.copy(), depth_input.copy()
        return target.astype(np.float32, copy=False)
