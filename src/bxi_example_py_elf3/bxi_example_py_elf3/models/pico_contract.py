"""PICO SDK world transforms -> SMPL body bases used by neural_retarget.

World conversion and bone-basis conversion are different operations:
  p_world = A p_xr; R_world_bone = A R_xr_bone B.
B is a proper 180-degree local-Y rotation, NOT a world reflection.
The common basis follows the XR/SMPL rest-frame difference (also present
in GMR's xrobot/smplx ELF3 orientation offsets). No robot IK offsets or
human-to-robot scale factors are applied to these human inputs.
"""

import numpy as np
from scipy.spatial.transform import Rotation

CONTRACT = "elf3_smpl_bodies_v1"
SMPL22_CONTRACT = "elf3_smpl22_bodies_v1"
SMPL22_BODY_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_toe", "right_toe",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
BODY_NAMES = (
    "pelvis", "spine3", "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle", "left_shoulder",
    "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
)
XR_INDICES = np.array([0, 9, 1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21])
WORLD_FROM_XR = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
XR_BONE_FROM_SMPL_BONE = np.diag([-1., 1., -1.])


def convert_xr_pose(poses, *, body_layout="neural14"):
    """Return [14,3] positions and [14,4] WXYZ; select nodes exactly once."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (24, 7) or not np.isfinite(poses).all():
        raise ValueError("XR pose must be finite [24,7] xyz+xyzw")
    if body_layout not in {"neural14", "smpl22"}:
        raise ValueError(f"Unknown PICO body layout {body_layout!r}")
    selected = poses[XR_INDICES if body_layout == "neural14" else np.arange(22)]
    if np.any(np.linalg.norm(selected[:, 3:], axis=-1) < 1e-6):
        raise ValueError("XR pose has zero quaternions")
    positions = selected[:, :3] @ WORLD_FROM_XR.T
    rotations = WORLD_FROM_XR @ Rotation.from_quat(selected[:, 3:]).as_matrix() @ XR_BONE_FROM_SMPL_BONE
    xyzw = Rotation.from_matrix(rotations).as_quat()
    return positions.astype(np.float32), xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def decode_packet(parts, *, body_layout="neural14"):
    """Reject old unversioned senders instead of silently permuting bones."""
    import json
    if len(parts) != 3:
        raise ValueError("PICO packet must have three parts")
    header = json.loads(parts[0])
    if not isinstance(header, dict):
        raise ValueError("PICO packet header must be a JSON object")
    if body_layout not in {"neural14", "smpl22"}:
        raise ValueError(f"Unknown PICO body layout {body_layout!r}")
    contract, names = (CONTRACT, BODY_NAMES) if body_layout == "neural14" else (SMPL22_CONTRACT, SMPL22_BODY_NAMES)
    if (header.get("contract") != contract or tuple(header.get("body_names", [])) != names or
            header.get("body_count", len(names)) != len(names)):
        raise ValueError("PICO contract mismatch: restart the updated sender")
    if body_layout == "smpl22":
        stamp = header.get("recv_ns")
        if type(stamp) is not int or not 0 < stamp < 2 ** 63:
            raise ValueError("Live PICO packet requires a positive sender timestamp")
    p = np.frombuffer(parts[1], dtype="<f4").reshape(len(names), 3).copy()
    q = np.frombuffer(parts[2], dtype="<f4").reshape(len(names), 4).copy()
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(norms < 1e-6):
        raise ValueError("PICO packet contains invalid transforms")
    return header, p, q / norms
