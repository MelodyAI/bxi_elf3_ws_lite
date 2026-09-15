"""Small, dependency-light SMPL-X/ACCAD motion adapter.

ACCAD stage-II files contain SMPL-X axis-angle poses rather than the
``body_pos_w/body_quat_w`` arrays consumed by RGMT.  This module performs the
body-only forward kinematics needed by the neural retargeter.  Hand, face and
finger joints are intentionally ignored because the deployed transformer
contract uses fourteen body nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


TARGET_FPS = 50.0
SMPLX_BODY_COUNT = 22

# SMPL-X body-joint order for pelvis + pose_body (21 joints).
BODY_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

# Approximate neutral SMPL-X offsets in metres.  The ACCAD files do not carry
# the SMPL-X model vertices, and these offsets preserve the kinematic layout
# while avoiding a hard dependency on a separately downloaded SMPL-X model.
BODY_OFFSETS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.070, -0.095, 0.0], [-0.070, -0.095, 0.0], [0.0, 0.105, 0.0],
        [0.0, -0.420, 0.0], [0.0, -0.420, 0.0], [0.0, 0.105, 0.0],
        [0.0, -0.430, 0.0], [0.0, -0.430, 0.0], [0.0, 0.105, 0.0],
        [0.0, -0.060, 0.130], [0.0, -0.060, 0.130], [0.0, 0.120, 0.0],
        [0.085, 0.060, 0.0], [-0.085, 0.060, 0.0], [0.0, 0.150, 0.0],
        [0.170, 0.0, 0.0], [-0.170, 0.0, 0.0],
        [0.260, 0.0, 0.0], [-0.260, 0.0, 0.0],
        [0.250, 0.0, 0.0], [-0.250, 0.0, 0.0],
    ],
    dtype=np.float32,
)

# Exact 14-node graph embedded in neural_retarget.onnx:
# pelvis, spine3, left leg, right leg, left arm and right arm.
NEURAL_BODY_INDICES = np.asarray(
    [0, 9, 1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21],
    dtype=np.int64,
)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    values = np.asarray(axis_angle, dtype=np.float32)
    flat = values.reshape(-1, 3)
    matrices = Rotation.from_rotvec(flat.astype(np.float64)).as_matrix()
    return matrices.reshape(values.shape[:-1] + (3, 3,)).astype(np.float32)


def matrix_to_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    flat = values.reshape(-1, 3, 3)
    xyzw = Rotation.from_matrix(flat).as_quat()
    wxyz = np.concatenate((xyzw[:, 3:4], xyzw[:, :3]), axis=1)
    return wxyz.reshape(values.shape[:-2] + (4,)).astype(np.float32)


def resample_body_transforms(
    positions: np.ndarray,
    rotations_wxyz: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample positions linearly and rotations with quaternion Slerp."""

    if np.isclose(source_fps, target_fps, rtol=0.0, atol=1.0e-8):
        return positions.astype(np.float32, copy=True), rotations_wxyz.astype(
            np.float32, copy=True
        )
    source_time = np.arange(len(positions), dtype=np.float64) / source_fps
    target_count = max(1, int(np.floor(source_time[-1] * target_fps)) + 1)
    target_time = np.arange(target_count, dtype=np.float64) / target_fps
    flat_position = positions.reshape(len(positions), -1)
    resampled_position = np.empty((target_count, flat_position.shape[1]), dtype=np.float32)
    for channel in range(flat_position.shape[1]):
        resampled_position[:, channel] = np.interp(
            target_time, source_time, flat_position[:, channel]
        )

    resampled_rotation = np.empty(
        (target_count, rotations_wxyz.shape[1], 4), dtype=np.float32
    )
    for body in range(rotations_wxyz.shape[1]):
        wxyz = rotations_wxyz[:, body]
        xyzw = np.concatenate((wxyz[:, 1:], wxyz[:, :1]), axis=1)
        interpolated = Slerp(source_time, Rotation.from_quat(xyzw))(target_time).as_quat()
        resampled_rotation[:, body] = np.concatenate(
            (interpolated[:, 3:4], interpolated[:, :3]), axis=1
        )
    return (
        resampled_position.reshape((target_count,) + positions.shape[1:]),
        resampled_rotation,
    )


def smplx_body_forward_kinematics(
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert SMPL-X body poses to world body positions and WXYZ rotations."""

    trans = np.asarray(trans, dtype=np.float32)
    root_orient = np.asarray(root_orient, dtype=np.float32)
    pose_body = np.asarray(pose_body, dtype=np.float32)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must have shape [frames,3], got {trans.shape}")
    frames = trans.shape[0]
    if root_orient.shape != (frames, 3) or pose_body.shape != (frames, 63):
        raise ValueError("ACCAD root_orient/pose_body shapes are incompatible")

    local_rot = np.empty((frames, SMPLX_BODY_COUNT, 3, 3), dtype=np.float32)
    local_rot[:, 0] = axis_angle_to_matrix(root_orient)
    local_rot[:, 1:] = axis_angle_to_matrix(pose_body.reshape(frames, 21, 3))
    world_rot = np.empty_like(local_rot)
    world_pos = np.empty((frames, SMPLX_BODY_COUNT, 3), dtype=np.float32)
    world_rot[:, 0] = local_rot[:, 0]
    world_pos[:, 0] = trans
    for joint in range(1, SMPLX_BODY_COUNT):
        parent = int(BODY_PARENTS[joint])
        world_rot[:, joint] = world_rot[:, parent] @ local_rot[:, joint]
        world_pos[:, joint] = world_pos[:, parent] + np.einsum(
            "fij,j->fi", world_rot[:, parent], BODY_OFFSETS[joint]
        )
    return world_pos[:, NEURAL_BODY_INDICES], matrix_to_quaternion_wxyz(
        world_rot[:, NEURAL_BODY_INDICES]
    )


@dataclass
class AccadSmplxMotion:
    """A resampled ACCAD sequence exposing transformer history windows."""

    body_positions_w: np.ndarray
    body_rotations_wxyz: np.ndarray
    fps: float = TARGET_FPS

    @classmethod
    def from_npz(cls, path: str | Path, *, target_fps: float = TARGET_FPS) -> "AccadSmplxMotion":
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=True) as data:
            required = ("mocap_frame_rate", "trans", "root_orient", "pose_body")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"ACCAD file is missing fields: {missing}")
            source_fps = float(np.asarray(data["mocap_frame_rate"]).reshape(-1)[0])
            positions, rotations = smplx_body_forward_kinematics(
                data["trans"], data["root_orient"], data["pose_body"]
            )
        if not np.isfinite(source_fps) or source_fps <= 0 or target_fps <= 0:
            raise ValueError("ACCAD and target FPS must be positive")
        positions, rotations = resample_body_transforms(
            positions, rotations, source_fps, target_fps
        )
        return cls(positions, rotations, float(target_fps))

    def history(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        if self.body_positions_w.ndim != 3 or self.body_positions_w.shape[1:] != (14, 3):
            raise ValueError("invalid ACCAD body position shape")
        indices = np.clip(np.arange(frame - 19, frame + 1), 0, len(self.body_positions_w) - 1)
        return self.body_positions_w[indices], self.body_rotations_wxyz[indices]


__all__ = [
    "AccadSmplxMotion",
    "resample_body_transforms",
    "smplx_body_forward_kinematics",
]
