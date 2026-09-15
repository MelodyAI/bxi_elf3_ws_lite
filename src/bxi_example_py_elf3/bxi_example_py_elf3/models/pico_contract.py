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
BODY_NAMES = (
    "pelvis", "spine3", "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle", "left_shoulder",
    "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
)
XR_INDICES = np.array([0, 9, 1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21])
WORLD_FROM_XR = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
XR_BONE_FROM_SMPL_BONE = np.diag([-1., 1., -1.])


def convert_xr_pose(poses):
    """Return [14,3] positions and [14,4] WXYZ; select nodes exactly once."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (24, 7) or not np.isfinite(poses).all():
        raise ValueError("XR pose must be finite [24,7] xyz+xyzw")
    selected = poses[XR_INDICES]
    if np.any(np.linalg.norm(selected[:, 3:], axis=-1) < 1e-6):
        raise ValueError("XR pose has zero quaternions")
    positions = selected[:, :3] @ WORLD_FROM_XR.T
    rotations = WORLD_FROM_XR @ Rotation.from_quat(selected[:, 3:]).as_matrix() @ XR_BONE_FROM_SMPL_BONE
    xyzw = Rotation.from_matrix(rotations).as_quat()
    return positions.astype(np.float32), xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def decode_packet(parts):
    """Reject old unversioned senders instead of silently permuting bones."""
    import json
    if len(parts) != 3:
        raise ValueError("PICO packet must have three parts")
    header = json.loads(parts[0])
    if header.get("contract") != CONTRACT or tuple(header.get("body_names", [])) != BODY_NAMES:
        raise ValueError("PICO contract mismatch: restart the updated sender")
    p = np.frombuffer(parts[1], dtype="<f4").reshape(14, 3).copy()
    q = np.frombuffer(parts[2], dtype="<f4").reshape(14, 4).copy()
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(norms < 1e-6):
        raise ValueError("PICO packet contains invalid transforms")
    return header, p, q / norms
