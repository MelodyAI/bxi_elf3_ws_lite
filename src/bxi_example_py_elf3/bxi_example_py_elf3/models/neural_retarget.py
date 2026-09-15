"""GMR pelvis-local ONNX inference with no post-model motion filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

HISTORY_FRAMES = 20
HORIZON_FRAMES = 11
TARGET_FPS = 50.0
DOF_COUNT = 29
HUMAN_BODY_COUNT = 14
# The deployed graph owns the GMR feature construction.  Its public ABI is
# canonical body transforms rather than the flattened node feature tensor used
# by the earlier network-only export.
HUMAN_NODE_DIM = 13
HUMAN_FRAME_DIM = HUMAN_BODY_COUNT * HUMAN_NODE_DIM
# deployment-abi/rgmt output: root angular velocity (3), root rotation 6D
# (6), 29 DoF positions and 29 DoF velocities.
ROBOT_REFERENCE_DIM = 67
MODEL_INPUT_NAMES = ("body_positions", "body_rotations_wxyz")
MODEL_OUTPUT_NAME = "robot_prediction"
MODEL_OUTPUT_DIM = ROBOT_REFERENCE_DIM
NEURAL_RETARGET_DOF_NAMES = (
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint",
    "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint",
    "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint",
    "l_wrist_z_joint", "r_shoulder_y_joint", "r_shoulder_x_joint",
    "r_shoulder_z_joint", "r_elbow_y_joint", "r_wrist_x_joint",
    "r_wrist_y_joint", "r_wrist_z_joint",
)


@dataclass(frozen=True, slots=True)
class TransformerPrediction:
    """The five tensors emitted by the Transformer, unchanged."""

    root_velocity_local: np.ndarray
    root_angular_velocity_local: np.ndarray
    root_rotation_6d: np.ndarray
    dof_position: np.ndarray
    dof_velocity: np.ndarray


def reconstruct_current_joint_velocity(
    prediction: TransformerPrediction,
    *,
    previous_position: np.ndarray | None = None,
    previous_velocity: np.ndarray | None = None,
    dt_s: float | None = None,
) -> TransformerPrediction:
    """Repair the ABI's first-frame velocity using the actual source timeline.

    ``deployment-abi`` derives velocity with a prepended copy, so the first
    query is always zero.  That query is used as the RGMT center frame.  Keep
    the model's predicted future velocities untouched and replace only the
    center token with a causal finite difference.  On the first sample use a
    forward difference inside the prediction horizon.
    """

    position = np.asarray(prediction.dof_position, dtype=np.float32).copy()
    velocity = np.asarray(prediction.dof_velocity, dtype=np.float32).copy()
    if position.shape != velocity.shape or position.ndim != 2 or position.shape[0] < 2:
        raise ValueError("prediction joint position/velocity windows are invalid")
    if previous_position is not None:
        previous = np.asarray(previous_position, dtype=np.float32)
        if previous.shape != position.shape[1:] or not np.isfinite(previous).all():
            raise ValueError("previous_position must match one joint position frame")
    if previous_velocity is not None:
        previous_v = np.asarray(previous_velocity, dtype=np.float32)
        if previous_v.shape != position.shape[1:] or not np.isfinite(previous_v).all():
            raise ValueError("previous_velocity must match one joint velocity frame")
    else:
        previous_v = None
    if dt_s is not None and (not np.isfinite(dt_s) or dt_s <= 1.0e-6):
        dt_s = None
    if previous_position is not None and dt_s is not None:
        velocity[0] = (position[0] - previous) / np.float32(dt_s)
    elif previous_v is not None:
        # A repeated source packet has no elapsed source time.  Do not use the
        # predicted future step as a fake current velocity; hold the last
        # causal estimate until a new timestamp arrives.
        velocity[0] = previous_v
    else:
        velocity[0] = (position[1] - position[0]) * np.float32(TARGET_FPS)
    return TransformerPrediction(
        root_velocity_local=np.asarray(prediction.root_velocity_local, dtype=np.float32).copy(),
        root_angular_velocity_local=np.asarray(prediction.root_angular_velocity_local, dtype=np.float32).copy(),
        root_rotation_6d=np.asarray(prediction.root_rotation_6d, dtype=np.float32).copy(),
        dof_position=position,
        dof_velocity=velocity,
    )


def blend_prediction_overlap(
    previous: TransformerPrediction,
    current: TransformerPrediction,
    alpha: float = 0.0,
) -> TransformerPrediction:
    """Match export-motion's adjacent-window overlap blend in NumPy."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("overlap blend must be inside [0,1]")
    blend = float(alpha)

    def mixed(name: str) -> np.ndarray:
        old = np.asarray(getattr(previous, name), dtype=np.float32)
        new = np.asarray(getattr(current, name), dtype=np.float32).copy()
        if old.shape != new.shape or old.ndim != 2 or old.shape[0] < 2:
            raise ValueError("prediction windows do not share a contract")
        new[:-1] = blend * old[1:] + (1.0 - blend) * new[:-1]
        return new

    rotation_old = np.asarray(previous.root_rotation_6d, dtype=np.float32)
    rotation_new = np.asarray(current.root_rotation_6d, dtype=np.float32).copy()
    if rotation_old.shape != rotation_new.shape or rotation_old.shape[0] < 2:
        raise ValueError("prediction windows do not share a contract")
    rotation_new[:-1] = matrix_to_rotation_6d(
        rotation_6d_to_matrix(
            blend * rotation_old[1:] + (1.0 - blend) * rotation_new[:-1]
        )
    )
    return TransformerPrediction(
        root_velocity_local=mixed("root_velocity_local"),
        root_angular_velocity_local=mixed("root_angular_velocity_local"),
        root_rotation_6d=rotation_new,
        dof_position=mixed("dof_position"),
        dof_velocity=mixed("dof_velocity"),
    )


def rotation_6d_to_matrix(rotation: object) -> np.ndarray:
    value = np.array(rotation, dtype=np.float32, copy=True)
    if value.shape[-1:] != (6,):
        raise ValueError("rotation_6d must end in six values")
    columns = value.reshape(value.shape[:-1] + (3, 2))
    first = columns[..., 0]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1.0e-8)
    second = columns[..., 1] - np.sum(first * columns[..., 1], axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1.0e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1).astype(np.float32, copy=False)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def matrix_to_quaternion_wxyz(matrix: object) -> np.ndarray:
    matrices = np.asarray(matrix, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    flat = matrices.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, value in enumerate(flat):
        trace = float(np.trace(value))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            result[index] = (0.25 * scale, (value[2, 1] - value[1, 2]) / scale,
                             (value[0, 2] - value[2, 0]) / scale, (value[1, 0] - value[0, 1]) / scale)
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                scale = np.sqrt(max(1.0 + value[0, 0] - value[1, 1] - value[2, 2], 1.0e-12)) * 2.0
                result[index] = ((value[2, 1] - value[1, 2]) / scale, 0.25 * scale,
                                 (value[0, 1] + value[1, 0]) / scale, (value[0, 2] + value[2, 0]) / scale)
            elif axis == 1:
                scale = np.sqrt(max(1.0 + value[1, 1] - value[0, 0] - value[2, 2], 1.0e-12)) * 2.0
                result[index] = ((value[0, 2] - value[2, 0]) / scale,
                                 (value[0, 1] + value[1, 0]) / scale, 0.25 * scale,
                                 (value[1, 2] + value[2, 1]) / scale)
            else:
                scale = np.sqrt(max(1.0 + value[2, 2] - value[0, 0] - value[1, 1], 1.0e-12)) * 2.0
                result[index] = ((value[1, 0] - value[0, 1]) / scale,
                                 (value[0, 2] + value[2, 0]) / scale,
                                 (value[1, 2] + value[2, 1]) / scale, 0.25 * scale)
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result.reshape(matrices.shape[:-2] + (4,)).astype(np.float32)


def _matrix_log_vector(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    angle = np.arccos(cosine)
    vee = np.stack(
        (matrix[..., 2, 1] - matrix[..., 1, 2],
         matrix[..., 0, 2] - matrix[..., 2, 0],
         matrix[..., 1, 0] - matrix[..., 0, 1]),
        axis=-1,
    )
    sine = np.sin(angle)
    scale = np.where(angle < 1.0e-4, 0.5, angle / np.maximum(2.0 * sine, 1.0e-6))
    return (vee * scale[..., None]).astype(np.float32)


def reference_world_velocities(
    root_position: np.ndarray,
    root_rotation: np.ndarray,
    *,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    linear = np.empty_like(root_position)
    linear[0] = (root_position[1] - root_position[0]) * fps
    linear[1:-1] = (root_position[2:] - root_position[:-2]) * (0.5 * fps)
    linear[-1] = (root_position[-1] - root_position[-2]) * fps
    # HoloMotion defines angular velocity in the reference local frame.  The
    # existing RGMT wrapper accepts world values and rotates each token back
    # into its local frame, so convert that exact local definition to the
    # per-token world representation expected by the wrapper.
    angular_local = np.empty_like(root_position)
    first_world = _matrix_log_vector(root_rotation[1] @ root_rotation[0].T) * fps
    angular_local[0] = root_rotation[1].T @ first_world
    center_world = _matrix_log_vector(
        root_rotation[2:] @ np.swapaxes(root_rotation[:-2], -1, -2)
    ) * (0.5 * fps)
    angular_local[1:-1] = (
        np.swapaxes(root_rotation[1:-1], -1, -2) @ center_world[..., None]
    )[..., 0]
    last_world = _matrix_log_vector(root_rotation[-1] @ root_rotation[-2].T) * fps
    angular_local[-1] = root_rotation[-1].T @ last_world
    angular = (root_rotation @ angular_local[..., None])[..., 0]
    return linear.astype(np.float32), angular.astype(np.float32)


class NeuralRetargeter:
    def __init__(
        self,
        model_path: str | Path,
        *,
        intra_op_threads: int = 4,
        backend: str = "auto",
        runtime=None,
    ) -> None:
        self.model_path = Path(model_path)
        # Decoder graph nodes follow the BXI/MuJoCo kinematic order.  The
        # NeuralRetargetRGMT adapter explicitly converts this to Isaac order.
        self.dof_names = NEURAL_RETARGET_DOF_NAMES
        if not isinstance(intra_op_threads, int) or not 1 <= intra_op_threads <= 8:
            raise ValueError("intra_op_threads must be an integer inside [1,8]")
        if backend not in {"auto", "onnxruntime"}:
            raise ValueError("backend must be 'auto' or 'onnxruntime'")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"neural retarget ONNX does not exist: {self.model_path}")

        # This export has no sidecar metadata. Validate its public ONNX ABI
        # directly and use the same ONNX Runtime dependency as rgmt.py.
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(intra_op_threads)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.backend_name = "onnxruntime"
        inputs_info = self._session.get_inputs()
        outputs_info = self._session.get_outputs()
        inputs = tuple(item.name for item in inputs_info)
        outputs = tuple(item.name for item in outputs_info)
        if inputs != MODEL_INPUT_NAMES:
            raise ValueError(
                "neural retarget ONNX inputs must be "
                f"{MODEL_INPUT_NAMES}, got {inputs}"
            )
        for item, width in zip(inputs_info, (3, 4), strict=True):
            name = item.name
            shape = tuple(item.shape)
            if len(shape) != 4 or shape[-2:] != (HUMAN_BODY_COUNT, width):
                raise ValueError(f"unexpected {name} shape {shape}")
        if outputs != (MODEL_OUTPUT_NAME,):
            raise ValueError(
                f"neural retarget ONNX output must be {MODEL_OUTPUT_NAME!r}, "
                f"got {outputs}"
            )
        output_shape = tuple(outputs_info[0].shape)
        if len(output_shape) != 3 or output_shape[-2:] != (
            HORIZON_FRAMES,
            MODEL_OUTPUT_DIM,
        ):
            raise ValueError(f"unexpected {MODEL_OUTPUT_NAME} shape {output_shape}")
        self._diagnostic_root_position = np.zeros(3, dtype=np.float32)
        self.last_inference_ms = 0.0

    def reset(self) -> None:
        """Reset visualization-only root integration state."""

        self._diagnostic_root_position.fill(0.0)

    def joint_reorder_indices(self, target_names: object) -> np.ndarray:
        names = tuple(str(name) for name in target_names)
        if len(names) != DOF_COUNT or set(names) != set(self.dof_names):
            raise ValueError("target joint names do not match the neural model's 29 DoFs")
        return np.asarray([self.dof_names.index(name) for name in names], dtype=np.int64)

    @staticmethod
    def _legacy_features_to_transforms(
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover canonical transforms from the former local-feature ABI.

        This keeps offline tools and old recorded callers usable while all
        live paths now send the model's canonical two-input ABI directly.
        """

        if features.shape != (HISTORY_FRAMES, HUMAN_BODY_COUNT, HUMAN_NODE_DIM):
            raise ValueError(f"unsupported legacy feature shape {features.shape}")
        rotations = rotation_6d_to_matrix(features[..., :6])
        pelvis_rotation = rotations[:, 0]
        anchor_rotation = pelvis_rotation[-1]
        pelvis_position = np.einsum(
            "ij,tj->ti", anchor_rotation, features[:, 0, 6:9]
        )
        positions = pelvis_position[:, None, :] + np.einsum(
            "tij,tkj->tki", pelvis_rotation, features[..., 6:9]
        )
        global_rotations = np.einsum(
            "tij,tkjl->tkil", pelvis_rotation, rotations
        )
        return positions.astype(np.float32), matrix_to_quaternion_wxyz(
            global_rotations
        )

    def infer(
        self,
        body_positions: object,
        body_rotations_wxyz: object | None = None,
    ) -> TransformerPrediction:
        positions = np.asarray(body_positions, dtype=np.float32)
        if body_rotations_wxyz is None:
            positions, rotations = self._legacy_features_to_transforms(positions)
        else:
            rotations = np.asarray(body_rotations_wxyz, dtype=np.float32)
        expected_prefix = (HISTORY_FRAMES, HUMAN_BODY_COUNT)
        if (
            positions.shape != (*expected_prefix, 3)
            or rotations.shape != (*expected_prefix, 4)
            or not np.isfinite(positions).all()
            or not np.isfinite(rotations).all()
        ):
            raise ValueError(
                "body transform history must be finite with shapes "
                f"[{HISTORY_FRAMES},{HUMAN_BODY_COUNT},3] and "
                f"[{HISTORY_FRAMES},{HUMAN_BODY_COUNT},4]"
            )
        started = time.perf_counter_ns()
        packed = self._session.run(
            [MODEL_OUTPUT_NAME],
            {
                "body_positions": np.ascontiguousarray(positions[None]),
                "body_rotations_wxyz": np.ascontiguousarray(rotations[None]),
            },
        )[0]
        self.last_inference_ms = (time.perf_counter_ns() - started) * 1.0e-6
        packed = np.asarray(packed[0], dtype=np.float32)
        if packed.shape != (HORIZON_FRAMES, MODEL_OUTPUT_DIM) or not np.isfinite(packed).all():
            raise ValueError(f"neural retarget output {MODEL_OUTPUT_NAME} is invalid: {packed.shape}")
        # Pose-only deployment ABI has no root linear velocity.  Keep a zero
        # field for the in-process prediction structure; the live RGMT packet
        # consumes angular velocity and joint fields below.
        root_velocity_local = np.zeros((HORIZON_FRAMES, 3), dtype=np.float32)
        root_angular_velocity_local = packed[:, :3]
        root_rotation_6d = packed[:, 3:9]
        dof_position = packed[:, 9 : 9 + DOF_COUNT]
        dof_velocity = packed[:, 9 + DOF_COUNT : 9 + 2 * DOF_COUNT]
        for name, value, shape in (
            ("root_velocity_local", root_velocity_local, (HORIZON_FRAMES, 3)),
            ("root_angular_velocity_local", root_angular_velocity_local, (HORIZON_FRAMES, 3)),
            ("root_rotation_6d", root_rotation_6d, (HORIZON_FRAMES, 6)),
            ("dof_position", dof_position, (HORIZON_FRAMES, DOF_COUNT)),
            ("dof_velocity", dof_velocity, (HORIZON_FRAMES, DOF_COUNT)),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"neural retarget output {name} is invalid: {value.shape}")
        return TransformerPrediction(
            root_velocity_local=root_velocity_local,
            root_angular_velocity_local=root_angular_velocity_local,
            root_rotation_6d=root_rotation_6d,
            dof_position=dof_position,
            dof_velocity=dof_velocity,
        )

    def close(self) -> None:
        """Release the selected inference backend."""

        self._session = None

    def infer_world_reference(
        self,
        body_positions: object,
        body_rotations_wxyz: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw output for offline visualization only.

        The live RGMT path never calls this method.
        """

        prediction = self.infer(body_positions, body_rotations_wxyz)
        matrices = rotation_6d_to_matrix(prediction.root_rotation_6d)
        quaternion = matrix_to_quaternion_wxyz(matrices)
        linear = (matrices @ prediction.root_velocity_local[..., None])[..., 0]
        angular = (
            matrices @ prediction.root_angular_velocity_local[..., None]
        )[..., 0]
        root_position = np.empty((HORIZON_FRAMES, 3), dtype=np.float32)
        root_position[0] = self._diagnostic_root_position
        root_position[1:] = self._diagnostic_root_position + np.cumsum(
            linear[:-1] / np.float32(TARGET_FPS), axis=0, dtype=np.float32
        )
        self._diagnostic_root_position += linear[0] / np.float32(TARGET_FPS)
        return (
            prediction.dof_position,
            prediction.dof_velocity,
            root_position,
            quaternion,
            linear,
            angular,
        )


__all__ = [
    "DOF_COUNT",
    "HISTORY_FRAMES",
    "HUMAN_BODY_COUNT",
    "HUMAN_FRAME_DIM",
    "HUMAN_NODE_DIM",
    "HORIZON_FRAMES",
    "NeuralRetargeter",
    "ROBOT_REFERENCE_DIM",
    "TARGET_FPS",
    "TransformerPrediction",
    "matrix_to_quaternion_wxyz",
    "reference_world_velocities",
    "rotation_6d_to_matrix",
]
