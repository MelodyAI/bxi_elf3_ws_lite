"""Author LAFAN's fixed bone frames from the original paired UMR reference.

The paired data is needed only at asset export, never on the deployment robot.
Verify constancy across the entire sequence to reject a clip-dependent fit.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def export_lafan_frames(bvh_path, smplx_path, parents, read_bvh, body_names):
    names, _parents, _offsets, _positions, rotations, fps = read_bvh(bvh_path)
    with np.load(smplx_path, allow_pickle=False) as data:
        poses = data["poses"].reshape(len(data["poses"]), -1)[:, :66]
        betas = np.asarray(data["betas"], dtype=np.float32)[:10]
        reference_fps = float(data["mocap_frame_rate"])
    if len(poses) != len(rotations) or not np.isclose(fps, reference_fps, rtol=2e-5):
        raise ValueError("LAFAN calibration requires paired frames at the same frame rate")
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    world = local.copy()
    for joint in range(1, 22):
        world[:, joint] = world[:, int(parents[joint])] @ local[:, joint]
    y_from_z = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    bvh_world = rotations[:, [names.index(name) for name in body_names]]
    frames = np.swapaxes(bvh_world, -1, -2) @ y_from_z @ world
    calibration = np.stack([Rotation.from_matrix(frames[:, joint]).mean().as_matrix()
                            for joint in range(22)])
    error = Rotation.from_matrix((np.swapaxes(calibration, -1, -2) @ frames).reshape(-1, 3, 3)).magnitude()
    if np.max(error) > 1e-5:
        raise ValueError(f"Bone calibration varies with motion: max angle error {np.max(error):g} rad")
    return {"bvh_bone_frames": calibration.astype(np.float64), "bvh_reference_betas": betas,
            "bvh_calibration_max_angle_error": np.asarray(np.max(error))}
