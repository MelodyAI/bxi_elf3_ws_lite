"""PICO packet -> causal UMR -> RGMT tests; no tracker or motor is required."""
import json
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bxi_example_py_elf3.models.pico_contract import (
    BODY_NAMES, CONTRACT, XR_INDICES, SMPL22_CONTRACT, SMPL22_BODY_NAMES,
    WORLD_FROM_XR, XR_BONE_FROM_SMPL_BONE, convert_xr_pose, decode_packet,
)
from bxi_example_py_elf3.models.pico_human_client import PicoHumanPoseClient
from bxi_example_py_elf3.models.pico_umr import PicoUmrRGMT, causal_window
from bxi_example_py_elf3.models.umr_rgmt import _load_runtime


PACKAGE = Path(__file__).resolve().parents[3]
runtime = _load_runtime(PACKAGE / "third_party/umr")
live = __import__(runtime.__package__ + ".live", fromlist=["live"])
sources = __import__(runtime.__package__ + ".sources", fromlist=["sources"])


@pytest.fixture(scope="module")
def assets():
    with np.load(PACKAGE / "third_party/umr/elf3_surface.npz") as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def metadata():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(PACKAGE / "policy/rgmtr_125200.onnx"), sess_options=options,
        providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map


def human_frame(assets, arm_angle=0.):
    model = sources.shaped_surface(assets, np.zeros(10))
    local = np.repeat(np.eye(3)[None], 22, axis=0)
    local[0] = Rotation.from_euler("X", 90, degrees=True).as_matrix()
    local[16] = Rotation.from_euler("Z", arm_angle).as_matrix()
    local[17] = Rotation.from_euler("Z", -arm_angle).as_matrix()
    poses = Rotation.from_matrix(local).as_rotvec().reshape(1, 66)
    source = sources.SurfaceSource(50., 1, model, poses=poses, translations=np.array([[0., 0., 1.]]))
    slots, normals, targets, joints = source.frame(0)
    world = local.copy()
    for joint in range(1, 22):
        world[joint] = world[int(model["parents"][joint])] @ local[joint]
    quaternions = Rotation.from_matrix(world).as_quat()[:, [3, 0, 1, 2]]
    return joints, quaternions, (slots, normals, targets, joints)


def xr_frame(positions, rotations):
    xr = np.zeros((24, 7))
    xr[:, 6] = 1.
    xr[:22, :3] = positions @ WORLD_FROM_XR
    world = Rotation.from_quat(rotations[:, [1, 2, 3, 0]]).as_matrix()
    xr[:22, 3:] = Rotation.from_matrix(WORLD_FROM_XR.T @ world @ XR_BONE_FROM_SMPL_BONE.T).as_quat()
    return xr


def packet(positions, rotations, stamp):
    return [json.dumps(dict(contract=SMPL22_CONTRACT, body_names=SMPL22_BODY_NAMES,
                           body_count=22, recv_ns=stamp)).encode(),
            positions.astype("<f4").tobytes(), rotations.astype("<f4").tobytes()]


def test_smpl22_sdk_contract_preserves_all_body_frames(assets):
    p, q, _ = human_frame(assets, .4)
    xr = xr_frame(p, q)
    p22, q22 = convert_xr_pose(xr, body_layout="smpl22")
    np.testing.assert_allclose(p22, p, atol=1e-7)
    actual = Rotation.from_quat(q22[:, [1, 2, 3, 0]])
    expected = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    assert np.max((actual.inv() * expected).magnitude()) < 1e-7
    p14, q14 = convert_xr_pose(xr)
    np.testing.assert_array_equal(p14, p22[XR_INDICES])
    np.testing.assert_array_equal(q14, q22[XR_INDICES])
    message = packet(p22, q22, 123)
    _, decoded_p, decoded_q = decode_packet(message, body_layout="smpl22")
    np.testing.assert_array_equal(decoded_p, p22)
    assert decoded_q.shape == (22, 4)
    with pytest.raises(ValueError, match="contract mismatch"):
        decode_packet(message)
    missing_stamp = message.copy()
    missing_stamp[0] = json.dumps(dict(contract=SMPL22_CONTRACT, body_names=SMPL22_BODY_NAMES)).encode()
    with pytest.raises(ValueError, match="timestamp"):
        decode_packet(missing_stamp, body_layout="smpl22")


def test_live_surface_matches_smpl_pose_and_does_not_mutate_inputs(assets):
    source = live.PicoSurfaceSource(assets)
    for angle in [0., .2, -.3, .6]:
        positions, rotations, expected = human_frame(assets, angle)
        snapshot = rotations.copy()
        actual = source.transforms(positions, rotations)
        for got, wanted in zip(actual, expected):
            np.testing.assert_allclose(got, wanted, atol=1e-7)
        np.testing.assert_array_equal(rotations, snapshot)
    assert source.ratio == pytest.approx(1.)
    with pytest.raises(ValueError, match="SMPL22"):
        source.transforms(positions[XR_INDICES], rotations[XR_INDICES])


def test_tracked_foot_centres_override_orientation_only_fk(assets):
    p, q, _ = human_frame(assets)
    tracked = live.PicoSurfaceSource(assets)
    original = live.PicoSurfaceSource(assets, use_tracked_positions=False)
    before = tracked.transforms(p, q)
    original.transforms(p, q)  # Calibrate both before perturbing the ankle marker.
    shifted = p.copy()
    shifted[[7, 10], 0] += .04
    snapshot = shifted.copy()
    actual = tracked.transforms(shifted, q)
    ignored = original.transforms(shifted, q)
    np.testing.assert_allclose(actual[3], shifted, atol=1e-7)
    np.testing.assert_allclose(ignored[3], p, atol=1e-7)
    selected_weights = tracked.surface_model['weights'][tracked.surface_model['triangles']]
    influence = np.einsum('si,sij->sj', assets['bary'], selected_weights)[:, [7, 10]].sum(axis=1)
    expected = np.zeros_like(actual[0])
    expected[:, 0] = .04 * influence
    np.testing.assert_allclose(actual[0] - before[0], expected, atol=1e-7)
    assert np.max(expected[:, 0]) > .03
    np.testing.assert_array_equal(shifted, snapshot)


def test_foot_contacts_ignore_floor_origin_hold_jitter_and_release_lift(assets):
    p, _q, _ = human_frame(assets)
    p[:, 2] -= 2.  # SDK origin is not the physical floor.
    detector = live.PicoFootContacts()
    detector.update(p, .02)
    assert detector.update(p, .02) == {'leftFoot': True, 'rightFoot': True}
    for frame in range(20):
        jittered = p.copy()
        jittered[[7, 8, 10, 11], 2] += .0015 * (frame + 1)
        assert detector.update(jittered, .02)['leftFoot']
    lifted = p.copy()
    lifted[[7, 10], 2] += .12
    assert detector.update(lifted, .02)['leftFoot']  # Debounce one bad sample.
    contact = detector.update(lifted, .02)
    assert not contact['leftFoot'] and contact['rightFoot']
    for _ in range(15):
        contact = detector.update(p, .02)
    assert contact == {'leftFoot': True, 'rightFoot': True}
    # A slow pivot remains support; rapid horizontal movement releases it.
    for frame in range(4):
        moved = p.copy()
        moved[[7, 10], 0] += (frame + 1) * .02
        contact = detector.update(moved, .02)
    assert not contact['leftFoot'] and contact['rightFoot']


@pytest.mark.parametrize('lift_height', [.02, .06])
def test_slow_small_lift_releases_support_and_waits_for_landing(assets, lift_height):
    p, _q, _ = human_frame(assets)
    detector = live.PicoFootContacts()
    for _ in range(10):
        detector.update(p, .02)
    released_at = None
    for frame in range(100):
        raised = p.copy()
        raised[[7, 10], 2] += min((frame + 1) * .002, lift_height)
        contact = detector.update(raised, .02)
        if not contact['leftFoot'] and released_at is None:
            released_at = frame * .02
        assert contact['rightFoot']
    assert released_at is not None and released_at < .24
    assert not contact['leftFoot']  # Held swing is not a new quiet stance.
    for _ in range(20):
        contact = detector.update(p, .02)
    assert contact == {'leftFoot': True, 'rightFoot': True}


def test_contact_floor_ignores_pivot_and_unilateral_low_outlier(assets):
    p, _q, _ = human_frame(assets)
    detector = live.PicoFootContacts()
    for _ in range(10):
        detector.update(p, .02)
    floor = detector.floor.copy()
    for frame in range(30):
        pivot = p.copy()
        pivot[7, 2] += min((frame + 1) * .004, .10)
        assert detector.update(pivot, .02)['leftFoot']  # Toe is still planted.
    np.testing.assert_array_equal(detector.floor, floor)
    for _ in range(20):
        detector.update(p, .02)
    low = p.copy()
    low[[7, 10], 2] -= .04
    detector.update(low, .02)
    for _ in range(30):
        detector.update(p, .02)
    np.testing.assert_array_equal(detector.floor, floor)
    assert detector.contacts.all()


def test_live_slow_lift_releases_ik_anchors_on_only_swing_foot(assets, metadata):
    retarget = live.LiveRetargeter(PACKAGE / 'robot/elf3_lite/xml/elf3.xml', metadata)
    p, q, _ = human_frame(assets)
    for frame in range(10):
        retarget.step(p, q, 1_000_000_000 + frame * 20_000_000)
    left_ids = retarget.solver.foot_anchor_ids['leftFoot'].copy()
    right_ids = retarget.solver.foot_anchor_ids['rightFoot'].copy()
    for frame in range(30):
        raised = p.copy()
        raised[[7, 10], 2] += (frame + 1) * .002
        retarget.step(raised, q, 1_200_000_000 + frame * 20_000_000)
    assert not retarget.foot_contacts.contacts[0]
    assert retarget.foot_contacts.contacts[1]
    assert 'leftFoot' not in retarget.solver.foot_anchor_ids
    assert not set(left_ids).intersection(retarget.solver.anchor_state)
    assert set(right_ids).issubset(retarget.solver.anchor_state)


def test_support_anchor_ids_persist_and_swing_releases_them(assets, metadata):
    retarget = live.LiveRetargeter(PACKAGE / 'robot/elf3_lite/xml/elf3.xml', metadata)
    p, q, _ = human_frame(assets)
    transform = retarget.source.transforms(p, q)
    support = {'leftFoot': True, 'rightFoot': True}
    retarget.solver.step(transform, foot_contacts=support)
    solver = retarget.solver
    left = solver.foot_anchor_ids['leftFoot'].copy()
    right = solver.foot_anchor_ids['rightFoot'].copy()
    assert len(left) and len(right)
    anchors = {i: solver.anchor_state[int(i)].copy() for i in left}
    moved = [item.copy() for item in transform]
    left_mask = assets['part_ids'] == solver.settings['part_name_to_id']['leftFoot']
    moved[0][left_mask, 0] += .02
    moved[0][left_mask, 2] += np.linspace(0., .02, left_mask.sum())
    solver.step(moved, foot_contacts=support)
    np.testing.assert_array_equal(solver.foot_anchor_ids['leftFoot'], left)
    for i in left:
        np.testing.assert_array_equal(solver.anchor_state[int(i)], anchors[i])
    solver.step(moved, foot_contacts={'leftFoot': False, 'rightFoot': True})
    assert 'leftFoot' not in solver.foot_anchor_ids
    assert not set(left).intersection(solver.anchor_state)
    assert set(right).issubset(solver.anchor_state)


def test_live_reference_limits_time_and_reset(assets, metadata):
    retarget = live.LiveRetargeter(PACKAGE / "robot/elf3_lite/xml/elf3.xml", metadata)
    previous = None
    for frame in range(15):
        p, q, _ = human_frame(assets, frame * .01)
        result = retarget.step(p, q, 1_000_000_000 + frame * 20_000_000)
        assert [x.shape for x in result] == [(29,), (29,), (4,), (3,)]
        assert all(np.isfinite(x).all() for x in result)
        if previous is not None:
            np.testing.assert_allclose(result[1], (result[0] - previous[0]) / .02, atol=1e-5)
        for info in retarget.solver.limits.values():
            value = retarget._filtered[retarget.model.jnt_qposadr[info["joint_id"]]]
            assert info["lower"] - 1e-6 <= value <= info["upper"] + 1e-6
        previous = result
    with pytest.raises(ValueError, match="increase"):
        retarget.step(p, q, 1_000_000_000)
    restarted = retarget.step(p, q, 2_000_000_000)
    np.testing.assert_array_equal(restarted[1], np.zeros(29))
    np.testing.assert_array_equal(restarted[3], np.zeros(3))
    assert retarget.solver.index == 1
    assert not retarget.solver.anchor_state


def test_causal_window_has_no_future_wait_and_pause_zeroes_velocity():
    history = []
    for frame in range(15):
        history.append((frame * .02, (np.full(29, frame, np.float32), np.full(29, 50., np.float32),
            np.array([1., 0., 0., 0.], np.float32), np.ones(3, np.float32))))
    p, v, q, w = causal_window(history)
    np.testing.assert_allclose(p[:11, 0], np.arange(4, 15))
    np.testing.assert_allclose(p[10:], 14.)
    np.testing.assert_array_equal(v[11:], np.zeros((10, 29)))
    np.testing.assert_array_equal(w[11:], np.zeros((10, 3)))
    held = causal_window(history, paused=True)
    np.testing.assert_allclose(held[0], 14.)
    np.testing.assert_array_equal(held[1], np.zeros((21, 29)))
    assert causal_window(history[-1:])[0].shape == (21, 29)


def test_short_prediction_is_causal_bounded_and_respects_joint_limits():
    limits = np.tile([-.5, .5], (29, 1))
    pos = np.zeros(29)
    pos[0] = .49
    quat = Rotation.from_euler('XYZ', [.2, -.3, .7]).as_quat()[[3, 0, 1, 2]]
    latest = (pos, np.full(29, 10.), quat, np.array([10., 0., 0.]))
    history = [(0., latest), (.02, latest)]
    window = causal_window(history, prediction_time_s=.08, joint_limits=limits)
    np.testing.assert_allclose(window[0][10], pos)
    assert np.all(window[0][11:] <= limits[:, 1] + 1e-7)
    assert np.max(abs(window[0][11:] - pos)) <= .2
    np.testing.assert_allclose(window[0][11:, 0], .5)
    np.testing.assert_array_equal(window[1][11:, 0], 0.)
    assert np.all(np.diff(window[1][11:, 1]) < 0.)
    assert 0. < window[1][-1, 1] < window[1][11, 1] <= 2.5
    rotations = Rotation.from_quat(window[2][11:, [1, 2, 3, 0]])
    current = Rotation.from_quat(quat[[1, 2, 3, 0]])
    increments = (rotations * current.inv()).as_rotvec()
    assert np.max(np.linalg.norm(increments, axis=1)) < .1
    np.testing.assert_allclose(increments[:, 1:], 0., atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(window[2], axis=1), 1., atol=1e-7)
    held = causal_window(history, paused=True, prediction_time_s=.08, joint_limits=limits)
    np.testing.assert_allclose(held[0], np.tile(pos, (21, 1)))
    np.testing.assert_array_equal(held[1], 0.)
    np.testing.assert_array_equal(held[3], 0.)
    first = causal_window(history[-1:], prediction_time_s=.08, joint_limits=limits)
    np.testing.assert_array_equal(first[1][11:], 0.)
    stopped = (pos, np.zeros(29), quat, np.zeros(3))
    stop_window = causal_window(history + [(.04, stopped)], prediction_time_s=.08, joint_limits=limits)
    np.testing.assert_array_equal(stop_window[1][11:], 0.)
    with pytest.raises(ValueError, match='joint limits'):
        causal_window(history, prediction_time_s=.08)
    with pytest.raises(ValueError, match='Prediction time'):
        causal_window(history, prediction_time_s=.5, joint_limits=limits)


def test_live_receiver_does_not_manufacture_freshness(assets):
    import zmq
    p, q, _ = human_frame(assets)
    client = PicoHumanPoseClient("inproc://live-receiver-contract", history_size=1,
                                body_layout="smpl22", hold_missing_frames=False)
    original = client._socket
    try:
        client._socket = Mock()
        client._socket.recv_multipart.side_effect = [packet(p, q, 2), zmq.Again()]
        assert client.poll() == 1
        stamp, receipt = client.last_timestamp_ns, client.last_receive_ns
        client._socket.recv_multipart.side_effect = [packet(p, q, 2), packet(p, q, 1), zmq.Again()]
        assert client.poll() == 0
        assert client.last_timestamp_ns == stamp and client.last_receive_ns == receipt
        client._socket.recv_multipart.side_effect = [zmq.Again()]
        assert client.poll() == 0 and client.last_timestamp_ns == stamp
    finally:
        original.close(0)


def test_real_pico_transport_live_process_and_rgmt_feedback(assets):
    import zmq
    context = zmq.Context.instance()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.LINGER, 0)
    port = sender.bind_to_random_port("tcp://127.0.0.1")
    client = PicoHumanPoseClient(f"tcp://127.0.0.1:{port}", history_size=1,
                                body_layout="smpl22", hold_missing_frames=False, async_receive=True)
    policy = PicoUmrRGMT(PACKAGE / "policy/rgmtr_125200.onnx",
        robot_xml=PACKAGE / "robot/elf3_lite/xml/elf3.xml", runtime_root=PACKAGE / "third_party/umr")
    targets, ages = [], []
    first_epoch = policy._epoch
    try:
        next_tick = time.monotonic()
        for tick in range(100):
            p, q, _ = human_frame(assets, .2 * np.sin(tick * .04))
            p, q = convert_xr_pose(xr_frame(p, q), body_layout="smpl22")
            sender.send_multipart(packet(p, q, time.monotonic_ns()))
            client.poll()
            sample = client.history_sample()
            history = None if sample is None else sample[0]
            if history is not None:
                target = policy.inference_step(policy.rgmt.default_dof_pos, np.zeros(29), [1, 0, 0, 0],
                    np.zeros(3), *history, source_time_s=sample[1] * 1e-9)
                if target is not None:
                    assert target.shape == (29,) and np.isfinite(target).all()
                    targets.append(target.copy())
                    ages.append(policy.performance_stats()["reference_age_ms"])
            next_tick += .02
            time.sleep(max(0., next_tick - time.monotonic()))
        assert len(targets) > 30, policy.performance_stats()
        assert max(ages) <= 150.
        assert np.max(abs(targets[-1] - targets[0])) > .01
        stats = policy.performance_stats()
        assert stats["backend"] == "process" and stats["processed"] > 30 and stats["error"] is None
        # Robot measurement changes the actor observation on each tick, even when
        # the human/reference is held. This exercises the feedback side of RGMT.
        history = client.history()
        policy.inference_step(policy.rgmt.default_dof_pos, np.zeros(29), [1, 0, 0, 0], np.zeros(3),
                              *history, source_time_s=client.last_timestamp_ns * 1e-9, advance=False)
        observation = policy.rgmt.obs.copy()
        policy.inference_step(policy.rgmt.default_dof_pos + .01, np.full(29, .02), [1, 0, 0, 0], np.zeros(3),
                              *history, source_time_s=client.last_timestamp_ns * 1e-9, advance=False)
        assert np.max(abs(policy.rgmt.obs - observation)) > .005
        np.testing.assert_array_equal(policy.rgmt.obs[:21 * 61].reshape(21, 61)[:, 32:], np.zeros((21, 29)))
        # Late references never continue commanding old human motion.
        time.sleep(.17)
        assert policy.inference_step(policy.rgmt.default_dof_pos, np.zeros(29), [1, 0, 0, 0], np.zeros(3),
                    *history, source_time_s=client.last_timestamp_ns * 1e-9) is None
        assert policy._epoch > first_epoch  # Resume invalidated the paused epoch.
        policy.reset()
        assert not policy._history and policy._result is None
        recovered = None
        epoch = policy._epoch
        next_tick = time.monotonic()
        for tick in range(40):
            p, q, _ = human_frame(assets, -.1)
            sender.send_multipart(packet(p, q, time.monotonic_ns()))
            sample = client.history_sample()
            if sample is not None:
                recovered = policy.inference_step(policy.rgmt.default_dof_pos, np.zeros(29), [1, 0, 0, 0],
                    np.zeros(3), *sample[0], source_time_s=sample[1] * 1e-9)
                if recovered is not None:
                    break
            next_tick += .02
            time.sleep(max(0., next_tick - time.monotonic()))
        assert recovered is not None and np.isfinite(recovered).all()
        assert policy._epoch == epoch and policy._result[1] > sample[1] - 150_000_000
    finally:
        pid = policy._process.pid
        policy.close()
        client.close()
        sender.close(0)
    assert not policy._writer.is_alive() and not policy._reader.is_alive()
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
