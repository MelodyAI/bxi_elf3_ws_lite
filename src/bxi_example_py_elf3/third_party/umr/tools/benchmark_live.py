"""Headless SDK-format -> ZeroMQ -> live UMR -> RGMT -> MuJoCo PD feedback.

Synthetic human data, virtual root support, no ROS/motor publication. This checks
the software feedback loop and latency; it is not a free-standing balance test.
"""
from pathlib import Path
import argparse
import json
import sys
import time

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "third_party"))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import zmq

from bxi_example_py_elf3.models.pico_contract import (
    SMPL22_CONTRACT, SMPL22_BODY_NAMES, WORLD_FROM_XR, XR_BONE_FROM_SMPL_BONE, convert_xr_pose,
)
from bxi_example_py_elf3.models.pico_human_client import PicoHumanPoseClient
from bxi_example_py_elf3.models.pico_umr import PicoUmrRGMT
from umr import runtime, sources


def synthetic_human(model, phase):
    local = np.repeat(np.eye(3)[None], 22, axis=0)
    local[0] = Rotation.from_euler("X", 90, degrees=True).as_matrix()
    local[16] = Rotation.from_euler("Z", -.9 + .2 * np.sin(phase)).as_matrix()
    local[17] = Rotation.from_euler("Z", .9 - .2 * np.sin(phase)).as_matrix()
    local[18] = Rotation.from_euler("Y", -.7).as_matrix()
    local[19] = Rotation.from_euler("Y", .7).as_matrix()
    source = sources.SurfaceSource(50., 1, model,
        poses=Rotation.from_matrix(local).as_rotvec().reshape(1, 66), translations=np.array([[0., 0., 1.]]))
    positions = source.frame(0)[3]
    world = local.copy()
    for joint in range(1, 22):
        world[joint] = world[int(model["parents"][joint])] @ local[joint]
    raw = np.zeros((24, 7))
    raw[:, 6] = 1.
    raw[:22, :3] = positions @ WORLD_FROM_XR
    raw[:22, 3:] = Rotation.from_matrix(WORLD_FROM_XR.T @ world @ XR_BONE_FROM_SMPL_BONE.T).as_quat()
    return convert_xr_pose(raw, body_layout="smpl22")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=500)
    args = parser.parse_args()
    context = zmq.Context.instance()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.LINGER, 0)
    sender.setsockopt(zmq.SNDHWM, 2)
    port = sender.bind_to_random_port("tcp://127.0.0.1")
    client = PicoHumanPoseClient(f"tcp://127.0.0.1:{port}", history_size=1,
                                body_layout="smpl22", hold_missing_frames=False, async_receive=True)
    policy = PicoUmrRGMT(PACKAGE / "policy/rgmtr_125200.onnx",
        robot_xml=PACKAGE / "robot/elf3_lite/xml/elf3.xml", runtime_root=PACKAGE / "third_party/umr")
    model = runtime.load_robot(PACKAGE / "robot/elf3_lite/xml/elf3.xml", use_fast=False)
    model.opt.timestep = .002
    data = mujoco.MjData(model)
    addrs, vaddrs, ranges, names = runtime.solver.scalar_qpos_joint_addrs(model)
    if len(names) != 29:
        raise RuntimeError("Expected ELF3 joint order")
    data.qpos[addrs] = policy.rgmt.default_dof_pos
    root_qpos = data.qpos[:7].copy()
    with np.load(PACKAGE / "third_party/umr/elf3_surface.npz") as archive:
        assets = {key: archive[key] for key in archive.files}
    human_model = sources.shaped_surface(assets, np.zeros(10))
    ages, compute, available, feedback, commands = [], [], 0, [], []
    next_tick = time.monotonic()
    initial_waits = 0
    warm_gaps = 0
    started = False
    try:
        for tick in range(args.ticks):
            positions, rotations = synthetic_human(human_model, tick * .02)
            header = dict(contract=SMPL22_CONTRACT, body_names=SMPL22_BODY_NAMES,
                          body_count=22, seq=tick, recv_ns=time.monotonic_ns())
            sender.send_multipart([json.dumps(header).encode(), positions.astype("<f4").tobytes(),
                                   rotations.astype("<f4").tobytes()], flags=zmq.NOBLOCK)
            client.poll()
            sample = client.history_sample()
            history = None if sample is None else sample[0]
            measured = data.qpos[addrs].copy()
            velocity = data.qvel[vaddrs].copy()
            begin = time.perf_counter()
            target = None if history is None else policy.inference_step(
                measured, velocity, data.qpos[3:7].copy(), data.qvel[3:6].copy(), *history,
                source_time_s=sample[1] * 1e-9)
            compute.append(time.perf_counter() - begin)
            if target is not None:
                assert np.isfinite(target).all()
                started = True
                available += 1
                ages.append(policy.performance_stats()["reference_age_ms"])
                # Virtual motor PD and measured joints close the feedback path.
                command = np.clip(target, ranges[:, 0], ranges[:, 1])
            else:
                initial_waits += int(not started)
                warm_gaps += int(started)
                command = policy.rgmt.default_dof_pos
            feedback.append(measured)
            commands.append(command.copy())
            for _ in range(10):
                data.qpos[:7] = root_qpos
                data.qvel[:6] = 0.
                data.qfrc_applied[vaddrs] = policy.rgmt.kps * (command - data.qpos[addrs]) - policy.rgmt.kds * data.qvel[vaddrs]
                mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            next_tick += .02
            time.sleep(max(0., next_tick - time.monotonic()))
        stats = policy.performance_stats()
        print(json.dumps(dict(ticks=args.ticks, tracking_ticks=available, initial_waits=initial_waits,
            warm_gaps=warm_gaps, control_call_ms_p99=float(np.percentile(compute, 99) * 1000),
            reference_age_ms_p50=float(np.median(ages)), reference_age_ms_p99=float(np.percentile(ages, 99)),
            joint_feedback_change=float(np.max(np.ptp(feedback, axis=0))),
            command_change=float(np.max(np.ptp(commands, axis=0))), stats=stats), indent=2), flush=True)
        if available < args.ticks // 2 or np.max(np.ptp(feedback, axis=0)) < .01:
            raise RuntimeError("Loopback did not reach active feedback tracking")
    finally:
        policy.close()
        client.close()
        sender.close(0)


if __name__ == "__main__":
    main()
