"""Contract and numeric checks; no upstream UMR environment is required."""
from pathlib import Path
import json
import time
import os
from threading import Event, current_thread

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bxi_example_py_elf3.models.umr_rgmt import _load_runtime, UmrStreamingRGMT


PACKAGE = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = PACKAGE / "third_party/umr"
runtime = _load_runtime(RUNTIME_ROOT)
sources = __import__(runtime.__package__ + ".sources", fromlist=["sources"])
qp = __import__(runtime.__package__ + ".qp", fromlist=["qp"])
contact = __import__(runtime.__package__ + ".self_contact", fromlist=["contact"])
streaming = __import__(runtime.__package__ + ".streaming", fromlist=["streaming"])


@pytest.fixture(scope="module")
def assets():
    with np.load(RUNTIME_ROOT / "elf3_surface.npz") as data:
        return {k: data[k] for k in data.files}


def test_body_only_lbs_preserves_rest_surface_and_translation(tmp_path, assets):
    path = tmp_path / "motion.npz"
    translation = np.array([[.2, -.3, .8], [.3, -.4, .8]])
    np.savez(path, root_orient=np.zeros((2, 3)), pose_body=np.zeros((2, 63)),
             trans=translation, mocap_frame_rate=30., betas=np.zeros(10))
    source = sources.load_smplx(path, assets)
    for frame in range(2):
        slots, normals, targets, joints = source.frame(frame)
        expected = np.einsum("si,sic->sc", assets["bary"], assets["v_template"][assets["triangles"]])
        np.testing.assert_allclose(slots, expected + translation[frame], atol=2e-7)
        np.testing.assert_allclose(joints, assets["joints"] + translation[frame], atol=2e-7)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(targets, axis=1), 1., atol=1e-7)


def test_bvh_channel_order_and_root_offset(tmp_path):
    path = tmp_path / "motion.bvh"
    path.write_text("""HIERARCHY
ROOT Hips {
 OFFSET 100 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Spine { OFFSET 100 0 0 CHANNELS 3 Zrotation Yrotation Xrotation
 End Site { OFFSET 0 0 0 } }
}
MOTION
Frames: 1
Frame Time: 0.033333333333
10 20 30 90 0 0 0 0 0
""")
    names, _parents, _offsets, positions, rotations, fps = sources.read_bvh(path)
    np.testing.assert_allclose(positions[0, names.index("Hips")], [.1, .2, .3])
    np.testing.assert_allclose(positions[0, names.index("Spine")], [.1, 1.2, .3])
    np.testing.assert_allclose(rotations[0, 0], Rotation.from_euler("Z", 90, degrees=True).as_matrix())
    assert fps == pytest.approx(30.)
    path.write_text(path.read_text().replace("Frames: 1", "Frames: 2"))
    with pytest.raises(ValueError, match="frame table"):
        sources.read_bvh(path)


def test_lafan_bone_twist_matches_original_conversion(assets):
    # Golden local body poses from the original UMR paired SMPL-X conversion.
    # End-point/bone-direction tests alone cannot detect a 180-degree arm twist.
    source = sources.load_lafan(PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh", assets)
    with np.load(Path(__file__).parent / "lafan_reference_frames.npz") as fixture:
        expected = Rotation.from_rotvec(fixture["poses"].reshape(-1, 3))
        actual = Rotation.from_rotvec(source.poses[fixture["frames"]].reshape(-1, 3))
    assert np.max((expected.inv() * actual).magnitude()) < 1e-6
    assert float(assets["bvh_calibration_max_angle_error"]) < 1e-6
    for frame in [0, 300, 1500]:
        slots, _normals, _targets, joints = source.frame(frame)
        assert np.isfinite(slots).all()
        rest = source.surface_model["rest_joints"]
        for joint in range(1, 22):
            parent = int(assets["parents"][joint])
            assert np.linalg.norm(joints[joint] - joints[parent]) == pytest.approx(
                np.linalg.norm(rest[joint] - rest[parent]), abs=1e-7)


def test_lafan_elf3_joint_pose_matches_original_umr(assets):
    source = sources.load_lafan(PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh", assets)
    model = runtime.load_robot(PACKAGE / "robot/elf3_lite/xml/elf3.xml")
    settings = json.loads((RUNTIME_ROOT / "solver_config.json").read_text())
    with np.load(Path(__file__).parent / "lafan_reference_frames.npz") as fixture:
        expected = fixture["elf3_qpos"]
    actual, _template, _limits = runtime.retarget(source, model, assets, settings, len(expected))
    # Compare articulated pose with the original Clarabel/PyTorch UMR output;
    # horizontal clip origin and BVH root placement are intentionally independent.
    error = np.abs(actual[:, 7:] - expected[:, 7:])
    assert np.percentile(error, 95) < .01
    assert np.max(error) < .03


def test_qp_respects_ground_and_l2_limits():
    # Unconstrained optimum is [2, -2]; the floor requires y >= 0.2.
    # On the unit ball the optimum is [sqrt(0.96), 0.2].
    result = qp.solve_qp_step(np.eye(2), np.array([-2., 2.]), 0.,
                             np.full(2, -np.inf), np.full(2, np.inf),
                             ineq_A=np.array([[0., -1.]]), ineq_b=np.array([-.2]),
                             ineq_soft_costs=np.zeros(1),
                             l2_step_limits=[(np.arange(2), 1., "unit ball")])
    np.testing.assert_allclose(result, [np.sqrt(.96), .2], atol=2e-5)


def test_fast_kinematics_and_batch_jacobians_match_full_robot(assets):
    xml = PACKAGE / "robot/elf3_lite/xml/elf3.xml"
    full_model = runtime.load_robot(xml, use_fast=False)
    fast_model = runtime.load_robot(xml)
    assert fast_model.nq == full_model.nq == 36
    assert fast_model.nmesh == 0 and full_model.nmesh > 0
    full_template = runtime.robot_template(full_model, assets)
    fast_template = runtime.robot_template(fast_model, assets)
    full_data, fast_data = mujoco.MjData(full_model), mujoco.MjData(fast_model)
    random = np.random.default_rng(24)
    ids = random.choice(4096, 48, replace=False)
    for _ in range(4):
        qpos = full_model.qpos0.copy()
        qpos[:3] = random.normal(size=3)
        qpos[3:7] = Rotation.from_rotvec(random.normal(size=3)).as_quat()[[3, 0, 1, 2]]
        qpos[7:] = random.uniform(-.5, .5, 29)
        full_data.qpos[:] = qpos
        mujoco.mj_forward(full_model, full_data)
        runtime.solver.set_qpos(fast_model, fast_data, qpos)
        points = runtime.solver.template_points_to_world(full_data, full_template, ids)
        cache = runtime.solver.TemplateSlotKinematicsCache(fast_model, fast_data, fast_template)
        np.testing.assert_allclose(cache.points(ids), points, atol=1e-12)
        np.testing.assert_allclose(fast_data.xpos, full_data.xpos, atol=1e-12)
        point_jacobians, normal_jacobians = cache.point_jacobians(ids), cache.normal_jacobians(ids)
        for row, slot in enumerate(ids):
            geom = int(full_template["geom_ids"][slot])
            body = int(full_model.geom_bodyid[geom])
            jacp, jacr = np.zeros((3, full_model.nv)), np.zeros((3, full_model.nv))
            mujoco.mj_jac(full_model, full_data, jacp, jacr, points[row], body)
            np.testing.assert_allclose(point_jacobians[row], jacp, atol=1e-12)
            normal = cache.normal(slot)
            expected = np.cross(jacr.T, normal).T
            np.testing.assert_allclose(normal_jacobians[row], expected, atol=1e-12)


def test_cached_smoothing_row_matches_original_lqr(assets):
    model = runtime.load_robot(PACKAGE / "robot/elf3_lite/xml/elf3.xml")
    addrs, _, _, _ = runtime.solver.scalar_qpos_joint_addrs(model)
    limits, _ = runtime.solver.build_scalar_joint_limits(model)
    random = np.random.default_rng(37)
    for frames in [7, 13, 19]:
        data = np.repeat(model.qpos0[None], frames, axis=0)
        data[:, 7:] = random.uniform(-.3, .3, (frames, 29))
        for start, end in [(False, False), (True, False), (False, True), (True, True)]:
            expected = runtime.solver.lqr_smooth_qpos_sequence(
                model, data, addrs, limits, acceleration_cost=.1, jerk_cost=.01,
                include_root_translation=True, anchor_start_frames=int(start), anchor_end_frames=int(end))
            columns = np.r_[np.arange(3), addrs]
            for index in [0, frames // 2, frames - 1]:
                row = streaming.smoothing_row(frames, index, 1., 0., .1, .01, start, end)
                actual = data[index].copy()
                actual[columns] = row @ data[:, columns]
                actual = runtime.solver.clamp_joint_ranges(model, actual, limits)
                np.testing.assert_allclose(actual, expected[index], atol=1e-12)


def test_reference_uses_metadata_names_and_world_angular_velocity(assets):
    model = runtime.load_robot(PACKAGE / "robot/elf3_lite/xml/elf3.xml")
    template = runtime.robot_template(model, assets)
    qaddrs, _vaddrs, _ranges, names = runtime.solver.scalar_qpos_joint_addrs(model)
    limits, _matches = runtime.solver.build_scalar_joint_limits(model)
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    qpos[:, qaddrs] = np.linspace(.01, .1, 29)
    angles = np.arange(3) * .01
    qpos[:, 3:7] = Rotation.from_euler("Z", angles).as_quat()[:, [3, 0, 1, 2]]
    ordered_names = sorted(names)
    metadata = {"joint_names": ",".join(ordered_names),
                "body_names": "torso_link,l_ankle_x_link,r_ankle_x_link"}
    data = runtime.pack_reference(qpos, model, template, limits, metadata, 50., .01)
    expected = qpos[:, [qaddrs[names.index(name)] for name in ordered_names]]
    np.testing.assert_allclose(data["joint_pos"], expected, atol=1e-7)
    np.testing.assert_allclose(data["body_ang_vel_w"][:, 0], np.tile([0., 0., .5], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1., atol=1e-7)
    mjdata = mujoco.MjData(model)
    for frame in data["qpos"]:
        runtime.solver.set_qpos(model, mjdata, frame.astype(np.float64))
        height = runtime.solver.template_points_world_z(mjdata, template, np.arange(4096))
        assert np.min(height) >= .01 - 1e-6


def test_contact_topology_reuse_preserves_maps(assets):
    selected = assets["selected"]
    slots = np.zeros((1, 4096, 3))
    slots[0, selected] = np.einsum("si,sic->sc", assets["bary"], assets["reference_triangles"])
    parts = np.zeros(4096, dtype=np.int32)
    parts[selected] = assets["part_ids"]
    settings = json.loads((RUNTIME_ROOT / "solver_config.json").read_text())
    kwargs = dict(part_name_to_id=settings["part_name_to_id"],
                  non_adjacent_pair_mask=lambda *_: assets["valid_pairs"], quiet=True)
    topology = {}
    expected = contact.compute_source_self_contact_map_groups(slots, selected, parts, **kwargs)
    contact.compute_source_self_contact_map_groups(slots, selected, parts, prepared=topology, **kwargs)
    actual = contact.compute_source_self_contact_map_groups(slots, selected, parts, prepared=topology, **kwargs)
    for mode in expected:
        for key in expected[mode][0]:
            np.testing.assert_array_equal(actual[mode][0][key], expected[mode][0][key])


def test_offline_reference_cache_and_invalidation(tmp_path, monkeypatch):
    import onnxruntime as ort
    session = ort.InferenceSession(str(PACKAGE / "policy/rgmtr_125200.onnx"), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    source = tmp_path / "motion.npz"
    np.savez(source, poses=np.zeros((1, 66)), trans=np.array([[0., 0., .95]]),
             mocap_framerate=30., output_up="z")
    kwargs = dict(cache_dir=tmp_path / "cache", max_frames=1)
    robot = PACKAGE / "robot/elf3_lite/xml/elf3.xml"
    reference = runtime.prepare_reference(source, robot, metadata, **kwargs)
    with np.load(reference) as data:
        assert data["joint_pos"].shape == (1, 29)
        assert data["body_quat_w"].shape == (1, 22, 4)
        assert data["fps"].item() == 50.
        assert np.isfinite(data["joint_vel"]).all()
    original = runtime.retarget
    def unexpected(*_args, **_kwargs):
        raise AssertionError("cache hit must not invoke retargeting")
    monkeypatch.setattr(runtime, "retarget", unexpected)
    assert runtime.prepare_reference(source, robot, metadata, **kwargs) == reference
    changed = dict(metadata, policy_fps="25")
    with pytest.raises(AssertionError, match="cache hit"):
        runtime.prepare_reference(source, robot, changed, **kwargs)
    monkeypatch.setattr(runtime, "retarget", original)


def test_streaming_reference_contract_and_derivatives(assets):
    import onnxruntime as ort
    metadata = ort.InferenceSession(str(PACKAGE / "policy/rgmtr_125200.onnx"),
                                    providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    source = sources.load_lafan(PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh", assets)
    source.frame_count = 40
    settings = json.loads((RUNTIME_ROOT / "solver_config.json").read_text())
    items = list(streaming.iter_references(source, assets, settings,
                 PACKAGE / "robot/elf3_lite/xml/elf3.xml", metadata, lambda: False))
    assert [i for i, _ in items] == list(range(streaming.reference_frame_count(source, 50.)))
    position, velocity, quaternion, angular = [np.stack([r[field] for _, r in items]) for field in range(4)]
    assert position.shape[1:] == (29,)
    assert quaternion.shape[1:] == (4,)
    assert all(np.isfinite(x).all() for x in (position, velocity, quaternion, angular))
    np.testing.assert_allclose(velocity, np.gradient(position, .02, axis=0), atol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(quaternion, axis=-1), 1., atol=1e-6)
    before = np.maximum(np.arange(len(quaternion)) - 1, 0)
    after = np.minimum(np.arange(len(quaternion)) + 1, len(quaternion) - 1)
    rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    expected = (rotation[after] * rotation[before].inv()).as_rotvec() * (50. / (after - before))[:, None]
    np.testing.assert_allclose(angular, expected, atol=1e-5)
    names = metadata["joint_names"].split(",")
    assert abs(position[0, names.index("l_shoulder_y_joint")] - .16609) < .01
    assert abs(position[0, names.index("r_shoulder_y_joint")] + .37615) < .01


def test_async_control_holds_when_worker_is_delayed_and_replays(monkeypatch):
    release = Event()
    generation_threads = []
    def fake_references(_self, cancelled):
        generation_threads.append(current_thread().name)
        for i in range(60):
            if i == 11:
                while not release.wait(.01):
                    if cancelled():
                        return
            if cancelled():
                return
            yield i, (np.full(29, .001 * i, dtype=np.float32), np.zeros(29, np.float32),
                      np.array([1., 0., 0., 0.], np.float32), np.zeros(3, np.float32)), .001
    monkeypatch.setattr(UmrStreamingRGMT, "_iter_worker_references", fake_references)
    policy = UmrStreamingRGMT(
        PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh", PACKAGE / "policy/rgmtr_125200.onnx",
        robot_xml=PACKAGE / "robot/elf3_lite/xml/elf3.xml", runtime_root=RUNTIME_ROOT,
        buffer_frames=24, reserve_frames=0, providers=["CPUExecutionProvider"])
    def wait_for(predicate):
        deadline = time.monotonic() + 5.
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(.005)
    args = (policy.default_dof_pos, np.zeros(29), [1., 0., 0., 0.], np.zeros(3))
    try:
        wait_for(lambda: policy.buffered_frames == 11)
        first = policy.inference_step(*args, advance=True)
        assert np.isfinite(first).all() and policy.timestep == 1
        # The worker is deliberately blocked. The control call still returns,
        # holds the last reference, and leaves motion time unchanged.
        held = policy.inference_step(*args, advance=True)
        assert np.isfinite(held).all() and policy.timestep == 1 and policy.buffering
        assert policy.performance_stats()["underruns"] == 1
        command = policy.obs[:21 * 61].reshape(21, 61)
        np.testing.assert_allclose(command[:, :3], 0.)
        np.testing.assert_allclose(command[:, 32:], 0.)
        np.testing.assert_array_equal(command[:, 3:32], np.repeat(command[10:11, 3:32], 21, axis=0))
        release.set()
        wait_for(lambda: policy.buffered_frames == 24)
        policy.inference_step(*args, advance=False)
        assert policy.timestep == 1 and not policy.buffering
        policy.inference_step(*args, advance=True)
        assert policy.timestep == 2
        policy.reset()
        assert policy.timestep == 0
        wait_for(lambda: len(generation_threads) == 2 and policy.buffered_frames >= 11)
        replay = policy.inference_step(*args, advance=True)
        np.testing.assert_allclose(replay, first, atol=1e-7)
        assert policy.buffered_frames <= 24
        assert generation_threads == ["umr-retarget", "umr-retarget"]
    finally:
        release.set()
        policy.close()
    assert not policy._worker.is_alive()


def test_buffer_recovery_requires_reserve_instead_of_one_frame(monkeypatch):
    first_release, second_release = Event(), Event()
    def fake_references(_self, cancelled):
        for i in range(80):
            release = first_release if i == 36 else second_release if i == 40 else None
            if release is not None:
                while not release.wait(.005):
                    if cancelled():
                        return
            if cancelled():
                return
            yield i, (np.zeros(29, np.float32), np.zeros(29, np.float32),
                      np.array([1., 0., 0., 0.], np.float32), np.zeros(3, np.float32)), .001
    monkeypatch.setattr(UmrStreamingRGMT, "_iter_worker_references", fake_references)
    policy = UmrStreamingRGMT(
        PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh", PACKAGE / "policy/rgmtr_125200.onnx",
        robot_xml=PACKAGE / "robot/elf3_lite/xml/elf3.xml", runtime_root=RUNTIME_ROOT,
        buffer_frames=60, reserve_frames=5, providers=["CPUExecutionProvider"])
    def wait_for(predicate):
        deadline = time.monotonic() + 5.
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(.005)
    args = (policy.default_dof_pos, np.zeros(29), [1., 0., 0., 0.], np.zeros(3))
    try:
        wait_for(lambda: policy.buffered_frames == 36)
        for _ in range(26):
            assert policy.inference_step(*args, advance=True) is not None
        assert policy.timestep == 26
        policy.inference_step(*args, advance=True)
        assert policy.buffering and policy.timestep == 26
        first_release.set()
        wait_for(lambda: policy._base_frame + policy.buffered_frames == 40)
        # The normal 21-frame horizon is available again, but not the reserve.
        # Holding here avoids the user's alternating hold/resume every tick.
        policy.inference_step(*args, advance=True)
        assert policy.buffering and policy.timestep == 26
        second_release.set()
        wait_for(lambda: policy._base_frame + policy.buffered_frames > 41)
        policy.inference_step(*args, advance=True)
        assert not policy.buffering and policy.timestep == 27
    finally:
        first_release.set()
        second_release.set()
        policy.close()


def test_process_reference_matches_in_process_and_restarts_cleanly(monkeypatch):
    # Run the real child interpreter even if the caller has a bad BLAS setting.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    policy = UmrStreamingRGMT(
        PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh", PACKAGE / "policy/rgmtr_125200.onnx",
        robot_xml=PACKAGE / "robot/elf3_lite/xml/elf3.xml", runtime_root=RUNTIME_ROOT,
        buffer_frames=45, reserve_frames=5, providers=["CPUExecutionProvider"])
    def wait_ready():
        deadline = time.monotonic() + 10.
        while policy.buffered_frames < 36:
            if policy._worker_error:
                raise policy._worker_error
            assert time.monotonic() < deadline
            time.sleep(.005)
    pid = new_pid = None
    try:
        wait_ready()
        pid = policy.performance_stats()["worker_pid"]
        assert pid is not None and pid != os.getpid()
        source, assets, settings = streaming.load_source(
            PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh")
        source.frame_count = 40
        expected = list(streaming.iter_references(source, assets, settings,
                        PACKAGE / "robot/elf3_lite/xml/elf3.xml", policy.metadata, lambda: False))
        with policy._condition:
            rows = list(policy._frames)[:25]
        for i, row in enumerate(rows):
            for actual, wanted in zip(row, expected[i][1]):
                np.testing.assert_allclose(actual, wanted, atol=1e-6)
        args = (policy.default_dof_pos, np.zeros(29), [1., 0., 0., 0.], np.zeros(3))
        policy.inference_step(*args, advance=True)
        policy.reset()
        wait_ready()
        new_pid = policy.performance_stats()["worker_pid"]
        assert new_pid != pid
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert np.isfinite(policy.inference_step(*args, advance=True)).all()
    finally:
        policy.close()
    assert not policy._worker.is_alive() and policy._process is None
    if new_pid is not None:
        with pytest.raises(ProcessLookupError):
            os.kill(new_pid, 0)


def test_worker_process_failure_is_reported_and_reaped(tmp_path):
    policy = UmrStreamingRGMT(
        PACKAGE / "policy/lafan1_orin/dance1_subject2.bvh", PACKAGE / "policy/rgmtr_125200.onnx",
        robot_xml=tmp_path / "missing_robot.xml", runtime_root=RUNTIME_ROOT,
        providers=["CPUExecutionProvider"])
    try:
        deadline = time.monotonic() + 10.
        while policy._worker_error is None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        with pytest.raises(RuntimeError, match="missing_robot.xml"):
            policy.inference_step(policy.default_dof_pos, np.zeros(29), [1, 0, 0, 0], np.zeros(3))
    finally:
        policy.close()
    assert policy._process is None and not policy._worker.is_alive()
