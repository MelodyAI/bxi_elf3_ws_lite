"""Raw SMPL/SMPL-X NPZ and LAFAN1 BVH readers, with NumPy surface skinning."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def triangle_basis(triangles):
    first = unit(triangles[:, 1] - triangles[:, 0])
    normal = unit(np.cross(first, triangles[:, 2] - triangles[:, 0]))
    return np.stack((first, unit(np.cross(normal, first)), normal), axis=-1)


@dataclass
class SurfaceSource:
    fps: float
    frame_count: int
    surface_model: dict
    poses: np.ndarray | None = None
    translations: np.ndarray | None = None
    world_positions: np.ndarray | None = None
    world_rotations: np.ndarray | None = None
    output_up: str = "z"
    human_scale: float = 1.0
    human_scale_mode: str = "off"
    joint_positions_w: np.ndarray | None = None

    def frame(self, index):
        model = self.surface_model
        rest = model["rest_joints"]
        if self.poses is not None:
            local = Rotation.from_rotvec(self.poses[index].reshape(22, 3)).as_matrix()
            rotations = np.empty_like(local)
            positions = np.empty((22, 3))
            rotations[0] = local[0]
            positions[0] = rest[0] + self.translations[index]
            for joint in range(1, 22):
                parent = int(model["parents"][joint])
                rotations[joint] = rotations[parent] @ local[joint]
                positions[joint] = positions[parent] + rotations[parent] @ (rest[joint] - rest[parent])
            corrective = (local[1:] - np.eye(3)).reshape(-1)
            vertices = model["rest_vertices"] + np.einsum("p,pvc->vc", corrective, model["posedirs"])
        else:
            rotations = self.world_rotations[index]
            positions = self.world_positions[index]
            vertices = model["rest_vertices"]
        if self.joint_positions_w is not None:
            tracked = np.asarray(self.joint_positions_w[index], dtype=np.float64)
            if tracked.shape != (22, 3) or not np.isfinite(tracked).all():
                raise ValueError("Tracked SMPL22 joints must be finite [22,3]")
            positions = tracked.copy()
        translation = positions - np.einsum("jcd,jd->jc", rotations, rest)
        # Equivalent to body-only SMPL-X LBS; non-body weights are folded to ancestors.
        blended_rotation = np.einsum("vj,jcd->vcd", model["weights"], rotations)
        vertices = np.einsum("vcd,vd->vc", blended_rotation, vertices) + model["weights"] @ translation
        if self.human_scale_mode == "local":
            origin = self.translations[index]
            vertices = (vertices - origin) * self.human_scale + origin
            positions = (positions - origin) * self.human_scale + origin
        elif self.human_scale_mode == "world":
            vertices *= self.human_scale
            positions *= self.human_scale
        # UMR consumes source motion in Z-up world coordinates.
        if self.output_up == "y":
            transform = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            vertices = vertices @ transform.T
            positions = positions @ transform.T
        triangles = vertices[model["triangles"]]
        slots = np.einsum("si,sic->sc", model["bary"], triangles)
        basis = triangle_basis(triangles)
        normal_targets = np.einsum("sij,sj->si", basis, model["normal_in_triangle"])
        return slots, basis[:, :, 2], unit(normal_targets), positions


def shaped_surface(assets, betas, body_model_path=None, gender="neutral"):
    model = dict(assets)
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    values = np.zeros(10)
    values[:min(10, len(betas))] = betas[:10]
    vertices, dirs = assets["v_template"], assets["shapedirs"]
    joints, joint_dirs = assets["joints"], assets["joint_shapedirs"]
    if body_model_path is not None:
        path = Path(body_model_path)
        # Existing deployment models already contain all three genders.
        sibling = path.with_name(f"SMPLX_{gender.upper()}_compact.npz")
        if sibling.is_file():
            path = sibling
        with np.load(path, allow_pickle=False) as data:
            full_vertices = np.asarray(data["v_template"], dtype=np.float64)
            full_dirs = np.asarray(data["shapedirs"], dtype=np.float64)[..., :10]
            regressor = np.asarray(data["J_regressor"], dtype=np.float64)[:22]
            if len(full_vertices) != 10475 or regressor.shape != (22, 10475):
                raise ValueError(f"Incompatible SMPL-X body model: {path}")
            vertices = full_vertices[assets["vertex_ids"]]
            dirs = full_dirs[assets["vertex_ids"]]
            joints = regressor @ full_vertices
            joint_dirs = np.einsum("jv,vci->jci", regressor, full_dirs)
    elif gender != "neutral":
        raise ValueError(f"{gender} motion requires the existing gender-specific compact SMPL-X model")
    model["rest_vertices"] = vertices + np.einsum("vci,i->vc", dirs, values)
    model["rest_joints"] = joints + np.einsum("jci,i->jc", joint_dirs, values)
    reference_basis = triangle_basis(assets["reference_triangles"])
    model["normal_in_triangle"] = np.einsum("sji,sj->si", reference_basis, assets["normal_reference"])
    return model


def load_smplx(path, assets, body_model_path=None):
    with np.load(path, allow_pickle=False) as data:
        if "poses" in data or "pose_aa" in data:
            key = "poses" if "poses" in data else "pose_aa"
            raw = np.asarray(data[key], dtype=np.float64)
            if raw.ndim < 2:
                raise ValueError("SMPL poses require a frame dimension")
            poses = raw.reshape(len(raw), -1)
        elif "root_orient" in data and "pose_body" in data:
            root = np.asarray(data["root_orient"], dtype=np.float64).reshape(-1, 3)
            body = np.asarray(data["pose_body"], dtype=np.float64).reshape(len(root), -1)
            poses = np.concatenate((root, body), axis=1)
        else:
            raise ValueError("UMR expects raw SMPL-X/SMPL poses, or root_orient + pose_body")
        if poses.shape[1] < 66 or len(poses) == 0:
            raise ValueError(f"Invalid SMPL body pose shape: {poses.shape}")
        poses = poses[:, :66].copy()
        trans = np.asarray(data.get("trans", data.get("trans_orig", np.zeros((len(poses), 3)))), dtype=np.float64)
        if trans.shape != (len(poses), 3):
            raise ValueError(f"Invalid SMPL translation shape: {trans.shape}")
        def scalar(key, default):
            value = np.asarray(data.get(key, default)).reshape(-1)
            if len(value) != 1:
                raise ValueError(f"{key} must be scalar")
            return value[0].item()
        fps = float(scalar("mocap_frame_rate", scalar("mocap_framerate", scalar("fps", 30.))))
        gender = scalar("gender", "neutral")
        gender = gender.decode() if isinstance(gender, bytes) else str(gender)
        gender = gender.lower()
        if gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Unsupported SMPL gender: {gender}")
        model = shaped_surface(assets, data.get("betas", data.get("beta", np.zeros(10))),
                               body_model_path, gender)
        output_up = str(scalar("samp_output_up", scalar("output_up", "z"))).lower()
        human_scale = float(scalar("human_scale", 1.0))
        scale_mode = str(scalar("human_scale_mode", "off")).lower()
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("SMPL motion contains non-finite values")
    if not np.isfinite(fps) or fps <= 0 or output_up not in {"y", "z"}:
        raise ValueError("Invalid SMPL frame rate or up axis")
    if not np.isfinite(human_scale) or human_scale <= 0 or scale_mode not in {"off", "local", "world"}:
        raise ValueError("Invalid human_scale/human_scale_mode")
    return SurfaceSource(fps, len(poses), model, poses=poses, translations=trans,
                         output_up=output_up, human_scale=human_scale, human_scale_mode=scale_mode)


# Body joint order follows SMPL-X (pelvis + 21 body joints).
BVH_BODY_NAMES = ["Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
                  "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToe", "RightToe",
                  "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
                  "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]


def read_bvh(path):
    text = Path(path).read_text(encoding="utf-8")
    parts = re.split(r"\bMOTION\b", text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("BVH missing MOTION section")
    tokens = re.findall(r"[^\s{}]+|[{}]", parts[0])
    cursor = 0
    names, parents, offsets, channels = [], [], [], []
    def token():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Truncated BVH hierarchy")
        value = tokens[cursor]
        cursor += 1
        return value
    def joint(parent, end=False):
        name = token()
        index = len(names)
        names.append(name if not end else f"__end_{index}")
        parents.append(parent)
        offsets.append([0., 0., 0.])
        channels.append([])
        if token() != "{":
            raise ValueError("BVH joint missing opening brace")
        while True:
            key = token()
            if key == "}":
                return
            if key == "OFFSET":
                offsets[index] = [float(token()) for _ in range(3)]
            elif key == "CHANNELS":
                channels[index] = [token() for _ in range(int(token()))]
            elif key in {"JOINT", "End"}:
                joint(index, key == "End")
            else:
                raise ValueError(f"Unsupported BVH hierarchy token: {key}")
    if token() != "HIERARCHY" or token() != "ROOT":
        raise ValueError("BVH must start with HIERARCHY ROOT")
    joint(-1)
    header = re.match(r"\s*Frames:\s*(\d+)\s+Frame\s+Time:\s*([\d.eE+-]+)\s*", parts[1])
    if header is None:
        raise ValueError("BVH missing frame count or frame time")
    count, timestep = int(header[1]), float(header[2])
    width = sum(map(len, channels))
    values = np.fromstring(parts[1][header.end():], sep=" ")
    if count < 1 or not np.isfinite(timestep) or timestep <= 0 or values.size != count * width:
        raise ValueError("Invalid BVH frame table or frame time")
    values = values.reshape(count, width)
    if not np.isfinite(values).all() or not np.isfinite(offsets).all():
        raise ValueError("BVH contains non-finite values")
    positions = np.empty((count, len(names), 3))
    rotations = np.empty((count, len(names), 3, 3))
    column = 0
    for index, channel_names in enumerate(channels):
        translation = np.zeros((count, 3))
        angles, axes = [], []
        for name in channel_names:
            series = values[:, column]
            column += 1
            if name in {"Xposition", "Yposition", "Zposition"}:
                translation[:, "XYZ".index(name[0])] = series
            elif name in {"Xrotation", "Yrotation", "Zrotation"}:
                axes.append(name[0])
                angles.append(series)
            else:
                raise ValueError(f"Unsupported BVH channel: {name}")
        local = (Rotation.from_euler("".join(axes), np.stack(angles, axis=1), degrees=True).as_matrix()
                 if axes else np.broadcast_to(np.eye(3), (count, 3, 3)))
        if parents[index] < 0:
            positions[:, index] = translation if any(n.endswith("position") for n in channel_names) else offsets[index]
            rotations[:, index] = local
        else:
            parent = parents[index]
            rotations[:, index] = rotations[:, parent] @ local
            positions[:, index] = positions[:, parent] + np.einsum(
                "fij,fj->fi", rotations[:, parent], translation + offsets[index])
    return names, np.asarray(parents), np.asarray(offsets), positions * 0.01, rotations, 1.0 / timestep


def load_lafan(path, assets):
    names, _parents, _offsets, positions, rotations, fps = read_bvh(path)
    missing = sorted(set(BVH_BODY_NAMES) - set(names))
    if missing:
        raise ValueError(f"LAFAN BVH missing body joints: {missing}")
    ids = [names.index(name) for name in BVH_BODY_NAMES]
    if "bvh_bone_frames" not in assets:
        raise ValueError("UMR assets lack validated LAFAN bone frames; rebuild the package")
    model = shaped_surface(assets, assets["bvh_reference_betas"])
    # A bone direction alone does not determine twist. The baked frames reproduce
    # the actual BVH -> SMPL-X conversion used by the original UMR reference.
    calibration = assets["bvh_bone_frames"]
    if calibration.shape != (22, 3, 3):
        raise ValueError("Invalid LAFAN bone-frame calibration")
    z_from_y = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    world = z_from_y @ rotations[:, ids] @ calibration
    local = world.copy()
    for joint in range(1, 22):
        parent = int(model["parents"][joint])
        local[:, joint] = np.swapaxes(world[:, parent], -1, -2) @ world[:, joint]
    poses = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(len(positions), 66)
    # Use a coherent SMPL-X FK + pose-corrective surface, rather than independently
    # placing skinning joints on a differently proportioned BVH skeleton.
    translation = positions[:, ids[0]] @ z_from_y.T - model["rest_joints"][0]
    model["rest_height"] = float(assets["bvh_reference_height"])
    return SurfaceSource(fps, len(positions), model, poses=poses, translations=translation)
