"""Generate and cache an RGMT reference before starting the control loop."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import solver
from .self_contact import compute_source_self_contact_map_groups
from .sources import load_lafan, load_smplx


ROOT = Path(__file__).resolve().parent


def robot_signature(xml_path):
    """Portable content signature for compiled body and mesh-local geom frames."""
    xml_path = Path(xml_path).resolve()
    digest = hashlib.sha256(xml_path.read_bytes())
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    for mesh in root.findall("asset/mesh"):
        if mesh.get("file"):
            digest.update(mesh.get("file").encode())
            digest.update((xml_path.parent / meshdir / mesh.get("file")).resolve().read_bytes())
    return digest.hexdigest()


def load_robot(xml_path, use_fast=True):
    """Use the installed ELF3 meshes and add UMR's floating wrapper in memory."""
    xml_path = Path(xml_path).resolve()
    fast_xml = ROOT / "elf3_kinematics.xml"
    signature = ROOT / "elf3_kinematics.sha256"
    if use_fast and fast_xml.is_file() and signature.is_file() and robot_signature(xml_path) == signature.read_text().strip():
        # Exact compiled body/joint/geom transforms, without loading STL meshes,
        # textures, actuators or sensors. Only UMR IK uses this private model.
        return mujoco.MjModel.from_xml_path(str(fast_xml))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            if compiler.get(key):
                compiler.set(key, str((xml_path.parent / compiler.get(key)).resolve()))
    for element in root.iter():
        if element.tag == "include":
            raise ValueError("The bundled ELF3 UMR adapter requires a standalone robot MJCF")
        if element.get("file") and element.tag != "mesh":
            element.set("file", str((xml_path.parent / element.get("file")).resolve()))
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"Missing worldbody in {xml_path}")
    if not any(j.tag == "freejoint" or j.tag == "joint" and j.get("type") == "free"
               for j in root.iter()):
        wrapper = ET.Element("body", name="elf3_lite_float_root")
        ET.SubElement(wrapper, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
        ET.SubElement(wrapper, "freejoint", name="elf3_lite_float_root_freejoint")
        for body in list(world):
            if body.tag == "body":
                world.remove(body)
                wrapper.append(body)
        world.append(wrapper)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"Expected floating ELF3 with 29 joints, got nq={model.nq}, nv={model.nv}")
    return model


def robot_template(model, assets):
    geom_ids = []
    for name, ordinal in zip(assets["robot_body_names"], assets["robot_geom_ordinals"]):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        candidates = np.flatnonzero(model.geom_bodyid == body)
        if body < 0 or ordinal >= len(candidates):
            raise ValueError(f"ELF3 surface binding does not match robot body {name}")
        geom_ids.append(candidates[int(ordinal)])
    return dict(geom_ids=np.asarray(geom_ids, dtype=np.int32),
                local_pos=assets["robot_local_pos"], local_normals=assets["robot_local_normals"])


def enforce_floor(model, data, template, qpos, margin):
    solver.set_qpos(model, data, qpos)
    height = solver.template_points_world_z(data, template, np.arange(len(template["geom_ids"])))
    qpos[2] += max(0., float(margin) - float(np.min(height)))
    return qpos


class FrameRetargeter:
    """Stateful UMR frame solver shared by file playback and live tracking."""

    def __init__(self, source, model, assets, settings):
        self.model = model
        self.assets = assets
        self.settings = settings
        self.index = 0
        self.args = SimpleNamespace(**settings)
        self.template = robot_template(model, assets)
        self.data = mujoco.MjData(model)
        self.qaddrs, self.daddrs, _ranges, _names = solver.scalar_qpos_joint_addrs(model)
        self.limits, _matched = solver.build_scalar_joint_limits(
            model, {"l_knee_y_joint": (0.1, np.inf), "r_knee_y_joint": (0.1, np.inf)})
        self.selected = assets["selected"]
        self.slot_count = len(self.template["geom_ids"])
        self.part_ids = np.zeros(self.slot_count, dtype=np.int32)
        self.part_ids[self.selected] = assets["part_ids"]
        self.point_costs, self.normal_costs = np.zeros(self.slot_count), np.zeros(self.slot_count)
        self.point_costs[self.selected], self.normal_costs[self.selected] = assets["point_costs"], assets["normal_costs"]
        # Height includes head and feet from the surface, rather than joint height.
        rest_height = source.surface_model.get("rest_height", float(np.ptp(source.surface_model["rest_vertices"][:, 1])))
        self.scale = float(assets["robot_height"]) / max(rest_height, 1e-8)
        self.previous = self.previous2 = None
        self.anchor_state = {}
        self.contact_topology = {}
        self.foot_anchor_ids = {}

    def step(self, transforms, *, foot_contacts=None):
        model = self.model
        positions, normals, normal_targets, joints = transforms
        positions, joints = positions * self.scale, joints * self.scale
        # Keep horizontal self.origin local to the clip, while preserving vertical placement.
        if self.index == 0:
            self.origin = joints[0].copy()
            self.origin[2] = 0.
            # BVH surface synthesis can place its heel just below the mocap floor.
            self.floor_offset = -min(0., float(positions[:, 2].min()))
        positions -= self.origin
        joints -= self.origin
        positions[:, 2] += self.floor_offset
        joints[:, 2] += self.floor_offset
        if foot_contacts is not None:
            for name, support in foot_contacts.items():
                ids = np.flatnonzero(self.part_ids[self.selected] == self.settings["part_name_to_id"][name])
                if len(ids) and support:
                    positions[ids, 2] -= np.min(positions[ids, 2])
        slots = np.zeros((self.slot_count, 3))
        source_normals = np.zeros_like(slots)
        targets = np.zeros_like(slots)
        slots[self.selected], source_normals[self.selected], targets[self.selected] = positions, normals, normal_targets
        ground = np.full(self.slot_count, np.inf)
        rank = np.full(self.slot_count, np.inf)
        raw_height = positions[:, 2]
        ground[self.selected] = np.maximum(raw_height, 0.)
        ground[ground < self.args.ground_contact_map_snap_threshold] = 0.
        rank[self.selected] = raw_height - raw_height.min()
        if foot_contacts is not None:
            # Online support is derived from tracked foot motion, not whether a
            # reconstructed skin point happens to fall inside a 5 mm band.
            for name, support in foot_contacts.items():
                ids = self.selected[self.part_ids[self.selected] == self.settings["part_name_to_id"][name]]
                if not len(ids):
                    continue
                if not support:
                    ground[ids] = np.inf
                    rank[ids] = np.inf
                    self.foot_anchor_ids.pop(name, None)
                else:
                    ground[ids] = np.maximum(ground[ids], .006)
                    if name not in self.foot_anchor_ids:
                        z = slots[ids, 2]
                        self.foot_anchor_ids[name] = ids[z <= np.min(z) + .012]
                    anchor_ids = self.foot_anchor_ids[name]
                    ground[anchor_ids] = 0.
                    rank[anchor_ids] = 0.
        groups = compute_source_self_contact_map_groups(
            slots[None], self.selected, self.part_ids, part_name_to_id=self.settings["part_name_to_id"],
            non_adjacent_pair_mask=lambda *_: self.assets["valid_pairs"],
            threshold=self.args.self_contact_map_threshold, max_pairs=self.args.self_contact_map_max_pairs,
            body_topk={"default": 1}, log_prefix="UMR", quiet=True, prepared=self.contact_topology)
        contacts = solver.merge_self_contact_maps(list(groups.values()))[0]
        if self.previous is None:
            initial = solver.clamp_joint_ranges(model, model.qpos0.copy(), self.limits)
            initial[:3] = joints[0]
            left = joints[1] - joints[2]
            left[2] = 0.
            left /= max(np.linalg.norm(left), 1e-8)
            up = np.array([0., 0., 1.])
            quat = Rotation.from_matrix(np.stack((np.cross(left, up), left, up), axis=1)).as_quat()
            initial[3:7] = quat[[3, 0, 1, 2]]
        else:
            initial = self.previous.copy()
        try:
            qpos, _cost = solver.solve_frame_body_segment_qp(
                model, self.data, initial, self.previous, self.previous2, slots, source_normals, targets,
                self.selected, self.part_ids, self.point_costs, self.normal_costs, contacts, ground, rank,
                None, self.template, self.qaddrs, self.daddrs, self.args,
                iters=self.args.pose_init_iters if self.previous is None else self.args.iters,
                joint_limits_by_qpos=self.limits, ground_contact_anchor_state=self.anchor_state)
        except RuntimeError as error:
            raise RuntimeError(f"UMR failed at source frame {self.index}: {error}") from error
        self.previous2, self.previous = self.previous, qpos
        self.index += 1
        return qpos.copy(), self.template, self.limits


def iter_retarget(source, model, assets, settings, max_frames=0, stop_requested=None):
    retargeter = FrameRetargeter(source, model, assets, settings)
    count = min(source.frame_count, max_frames) if max_frames > 0 else source.frame_count
    progress_time = time.monotonic()
    for frame in range(count):
        if stop_requested is not None and stop_requested():
            return
        yield retargeter.step(source.frame(frame))
        if time.monotonic() - progress_time >= 5. or frame == count - 1:
            print(f"[UMR] retarget {frame + 1}/{count} frames", flush=True)
            progress_time = time.monotonic()


def retarget(source, model, assets, settings, max_frames=0):
    args = SimpleNamespace(**settings)
    frames = []
    for qpos, template, limits in iter_retarget(source, model, assets, settings, max_frames):
        frames.append(qpos)
    output = np.asarray(frames)
    qaddrs, _daddrs, _ranges, _names = solver.scalar_qpos_joint_addrs(model)
    output = solver.lqr_smooth_qpos_sequence(
        model, output, qaddrs, limits, data_cost=args.trajectory_filter_data_cost,
        velocity_cost=args.trajectory_filter_velocity_cost,
        acceleration_cost=args.trajectory_filter_acceleration_cost,
        jerk_cost=args.trajectory_filter_jerk_cost, include_root_translation=True,
        anchor_start_frames=1, anchor_end_frames=1)
    return output, template, limits


def resample_qpos(qpos, source_fps, target_fps):
    if len(qpos) == 1:
        return qpos.copy()
    times = np.arange(len(qpos)) / source_fps
    output_times = np.arange(int(np.floor(times[-1] * target_fps)) + 1) / target_fps
    result = np.column_stack([np.interp(output_times, times, values) for values in qpos.T])
    rotation = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    result[:, 3:7] = Slerp(times, rotation)(output_times).as_quat()[:, [3, 0, 1, 2]]
    return result


def pack_reference(qpos, model, template, limits, metadata, fps, margin):
    qaddrs, _daddrs, _ranges, names = solver.scalar_qpos_joint_addrs(model)
    joint_names = metadata["joint_names"].split(",")
    body_names = metadata["body_names"].split(",")
    if len(joint_names) != 29 or set(joint_names) != set(names):
        raise ValueError("RGMT joint names do not match the ELF3 MJCF")
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if min(body_ids) < 0:
        raise ValueError("RGMT body names do not match the ELF3 MJCF")
    pos = np.empty((len(qpos), len(body_ids), 3))
    quat = np.empty((len(qpos), len(body_ids), 4))
    data = mujoco.MjData(model)
    for frame in range(len(qpos)):
        qpos[frame] = solver.clamp_joint_ranges(model, qpos[frame], limits)
        qpos[frame] = enforce_floor(model, data, template, qpos[frame], margin)
        solver.set_qpos(model, data, qpos[frame])
        pos[frame], quat[frame] = data.xpos[body_ids], data.xquat[body_ids]
    joints = qpos[:, [qaddrs[names.index(name)] for name in joint_names]]
    def velocity(values):
        return np.gradient(values, 1. / fps, axis=0) if len(values) > 1 else np.zeros_like(values)
    ang_vel = np.zeros_like(pos)
    if len(qpos) > 1:
        before = np.maximum(np.arange(len(qpos)) - 1, 0)
        after = np.minimum(np.arange(len(qpos)) + 1, len(qpos) - 1)
        for body in range(len(body_ids)):
            rotation = Rotation.from_quat(quat[:, body][:, [1, 2, 3, 0]])
            ang_vel[:, body] = (rotation[after] * rotation[before].inv()).as_rotvec() * (fps / (after - before))[:, None]
    return dict(fps=np.asarray(fps, dtype=np.float32), joint_pos=joints.astype(np.float32),
                joint_vel=velocity(joints).astype(np.float32), body_pos_w=pos.astype(np.float32),
                body_quat_w=quat.astype(np.float32), body_ang_vel_w=ang_vel.astype(np.float32),
                body_lin_vel_w=velocity(pos).astype(np.float32), body_names=np.asarray(body_names),
                joint_names=np.asarray(joint_names), qpos=qpos.astype(np.float32))


def cache_key(source, robot_xml, body_model_path, metadata, max_frames):
    digest = hashlib.sha256()
    paths = [source, robot_xml, ROOT / "elf3_surface.npz", ROOT / "solver_config.json"]
    paths += [path for path in (ROOT / "elf3_kinematics.xml", ROOT / "elf3_kinematics.sha256") if path.is_file()]
    paths += sorted(ROOT.glob("*.py"))
    if body_model_path is not None:
        paths += sorted(Path(body_model_path).parent.glob("SMPLX_*_compact.npz"))
        paths.append(Path(body_model_path))
    # Include the robot's mesh files so changing its geometry invalidates cached references.
    xml = ET.parse(robot_xml).getroot()
    compiler = xml.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    paths += [(Path(robot_xml).parent / meshdir / mesh.get("file")).resolve()
              for mesh in xml.findall("asset/mesh") if mesh.get("file")]
    for path in paths:
        digest.update(str(Path(path).resolve()).encode())
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps([metadata, max_frames], sort_keys=True).encode())
    return digest.hexdigest()


def prepare_reference(source_path, robot_xml, metadata, *, body_model_path=None,
                      cache_dir=None, max_frames=0):
    source_path = Path(source_path).expanduser().resolve()
    if source_path.suffix.lower() not in {".npz", ".bvh"}:
        raise ValueError("UMR reference must be a raw SMPL-X NPZ or LAFAN1 BVH")
    if not source_path.is_file():
        raise FileNotFoundError(f"UMR source motion not found: {source_path}")
    fps = float(metadata["policy_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("RGMT policy FPS must be positive and finite")
    relevant_metadata = {k: metadata[k] for k in ("policy_fps", "joint_names", "body_names")}
    key = cache_key(source_path, robot_xml, body_model_path, relevant_metadata, max_frames)
    directory = Path(cache_dir) if cache_dir else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "bxi_example_py_elf3/umr"
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{source_path.stem}-{key[:24]}.npz"
    if output_path.is_file():
        try:
            with np.load(output_path, allow_pickle=False) as data:
                if str(data["umr_cache_key"].item()) == key and data["qpos"].shape[1:] == (36,) and np.isfinite(data["qpos"]).all():
                    print(f"[UMR] cached reference: {output_path}", flush=True)
                    return output_path
        except (OSError, ValueError, KeyError):
            pass
    print(f"[UMR] preparing {source_path.name}; the first load retargets and caches the motion", flush=True)
    with np.load(ROOT / "elf3_surface.npz", allow_pickle=False) as archive:
        assets = {k: archive[k] for k in archive.files}
    settings = json.loads((ROOT / "solver_config.json").read_text())
    source = (load_lafan(source_path, assets) if source_path.suffix.lower() == ".bvh"
              else load_smplx(source_path, assets, body_model_path))
    model = load_robot(robot_xml)
    qpos, template, limits = retarget(source, model, assets, settings, max_frames)
    qpos = resample_qpos(qpos, source.fps, fps)
    reference = pack_reference(qpos, model, template, limits, relevant_metadata, fps,
                               settings["ground_penetration_margin"])
    reference["umr_cache_key"] = np.asarray(key)
    reference["umr_source"] = np.asarray(str(source_path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **reference)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"[UMR] ready: {len(qpos)} frames at {fps:g} Hz; {output_path}", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare an ELF3 UMR reference without ROS or robot control")
    parser.add_argument("source", type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--rgmt-onnx", required=True, type=Path)
    parser.add_argument("--body-model", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.rgmt_onnx), providers=["CPUExecutionProvider"])
    prepare_reference(args.source, args.robot_xml, session.get_modelmeta().custom_metadata_map,
                      body_model_path=args.body_model, cache_dir=args.cache_dir, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
