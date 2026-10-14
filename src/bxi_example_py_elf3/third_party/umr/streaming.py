"""Bounded-lookahead UMR references; all generation runs on a worker thread."""
from collections import deque
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import runtime, solver
from .sources import load_lafan, load_smplx
from .smoothing import smoothing_row


def load_source(source_path, body_model_path=None):
    source_path = Path(source_path).expanduser().resolve()
    with np.load(runtime.ROOT / "elf3_surface.npz", allow_pickle=False) as archive:
        assets = {k: archive[k] for k in archive.files}
    settings = json.loads((runtime.ROOT / "solver_config.json").read_text())
    if source_path.suffix.lower() == ".bvh":
        source = load_lafan(source_path, assets)
    elif source_path.suffix.lower() == ".npz":
        source = load_smplx(source_path, assets, body_model_path)
    else:
        raise ValueError("UMR streaming requires a raw SMPL-X NPZ or LAFAN BVH")
    return source, assets, settings


def filtered_source_frames(source, model, assets, settings, stop_requested):
    """Use the original LQR costs on a rolling window: 12 past, 6 future frames."""
    retained = deque()
    next_output = 0
    qaddrs, _vaddrs, _ranges, _names = solver.scalar_qpos_joint_addrs(model)
    columns = np.concatenate((np.arange(3), qaddrs))
    template = limits = None

    def emit(index, final=False):
        start = max(0, index - 12)
        end = min(retained[-1][0], index + 6)
        selected = [(i, q) for i, q in retained if start <= i <= end]
        qpos = np.stack([q for _i, q in selected])
        row = smoothing_row(len(qpos), index - start, settings["trajectory_filter_data_cost"],
                            settings["trajectory_filter_velocity_cost"],
                            settings["trajectory_filter_acceleration_cost"],
                            settings["trajectory_filter_jerk_cost"], start == 0, final)
        result = qpos[index - start].copy()
        result[columns] = row @ qpos[:, columns]
        return index, solver.clamp_joint_ranges(model, result, limits), template, limits

    for index, (qpos, template, limits) in enumerate(runtime.iter_retarget(
            source, model, assets, settings, stop_requested=stop_requested)):
        retained.append((index, qpos))
        while next_output <= index - 6:
            yield emit(next_output)
            next_output += 1
            while retained and retained[0][0] < next_output - 12:
                retained.popleft()
    if stop_requested():
        return
    while retained and next_output <= retained[-1][0]:
        yield emit(next_output, final=True)
        next_output += 1


def resampled_frames(frames, source_fps, target_fps):
    previous = None
    target_index = 0
    for index, qpos, template, limits in frames:
        if previous is None:
            previous = (index, qpos)
            continue
        left_index, left = previous
        right_time = index / source_fps
        left_time = left_index / source_fps
        left_rotation = Rotation.from_quat(left[[4, 5, 6, 3]])
        right_rotation = Rotation.from_quat(qpos[[4, 5, 6, 3]])
        delta = (right_rotation * left_rotation.inv()).as_rotvec()
        while target_index / target_fps <= right_time + 1e-10:
            fraction = np.clip((target_index / target_fps - left_time) / (right_time - left_time), 0., 1.)
            result = left + fraction * (qpos - left)
            quat = (Rotation.from_rotvec(fraction * delta) * left_rotation).as_quat()
            result[3:7] = quat[[3, 0, 1, 2]]
            yield target_index, result, template, limits
            target_index += 1
        previous = (index, qpos)
    if target_index == 0 and previous is not None:
        yield 0, previous[1], template, limits


def iter_references(source, assets, settings, robot_xml, metadata, stop_requested, model=None):
    if model is None:
        model = runtime.load_robot(robot_xml)
    fps = float(metadata["policy_fps"])
    qaddrs, _vaddrs, _ranges, names = solver.scalar_qpos_joint_addrs(model)
    joint_names = metadata["joint_names"].split(",")
    if len(joint_names) != 29 or set(joint_names) != set(names):
        raise ValueError("RGMT joint names do not match the ELF3 MJCF")
    columns = [qaddrs[names.index(name)] for name in joint_names]
    anchor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, metadata["anchor_body_name"])
    if anchor < 0:
        raise ValueError("RGMT anchor body does not match the ELF3 MJCF")
    data = mujoco.MjData(model)
    pending = deque(maxlen=3)
    emitted = -1

    def reference(center, before, after):
        elapsed = (after[0] - before[0]) / fps
        if elapsed > 0:
            velocity = (after[1] - before[1]) / elapsed
            angular = (Rotation.from_quat(after[2][[1, 2, 3, 0]]) *
                       Rotation.from_quat(before[2][[1, 2, 3, 0]]).inv()).as_rotvec() / elapsed
        else:
            velocity, angular = np.zeros(29), np.zeros(3)
        return tuple(np.asarray(item, dtype=np.float32).copy()
                     for item in (center[1], velocity, center[2], angular))

    frames = filtered_source_frames(source, model, assets, settings, stop_requested)
    for index, qpos, template, limits in resampled_frames(frames, source.fps, fps):
        if stop_requested():
            return
        qpos = solver.clamp_joint_ranges(model, qpos, limits)
        runtime.enforce_floor(model, data, template, qpos, settings["ground_penetration_margin"])
        solver.set_qpos(model, data, qpos)
        pending.append((index, qpos[columns].copy(), data.xquat[anchor].copy()))
        if index == 1:
            yield 0, reference(pending[0], pending[0], pending[1])
            emitted = 0
        elif index >= 2:
            yield index - 1, reference(pending[1], pending[0], pending[2])
            emitted = index - 1
    if pending and not stop_requested():
        if pending[-1][0] > emitted:
            yield pending[-1][0], reference(pending[-1], pending[0] if len(pending) == 1 else pending[-2], pending[-1])


def reference_frame_count(source, target_fps):
    return max(1, int(math.floor((source.frame_count - 1) * target_fps / source.fps)) + 1)
