"""Depth input contract and upstream numeric fixtures; no ROS or camera needed."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bxi_example_py_elf3.models.depth import HumanoidGaitDepthPolicy
from bxi_example_py_elf3.models.depth_camera import DepthCameraInput
from bxi_example_py_elf3.models.depth_projection import project_depth
from bxi_example_py_elf3.models.rgmt import MUJOCO_TO_ISAAC_INDEX, ISAAC_TO_MUJOCO_INDEX


PACKAGE = Path(__file__).resolve().parents[1]


def info(width=848, height=480):
    focal = height / (2 * np.tan(np.radians(58.605976) / 2))
    return SimpleNamespace(width=width, height=height,
        k=[focal, 0., width / 2, 0., focal, height / 2, 0., 0., 1.])


def image(array, encoding="16UC1", stamp=1, pad=0):
    height, width = array.shape
    padded = np.pad(array, ((0, 0), (0, pad)))
    return SimpleNamespace(height=height, width=width, encoding=encoding, step=padded.strides[0],
        is_bigendian=int(array.dtype.byteorder == '>'), data=padded.tobytes(),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nanosec=0)))


@pytest.mark.parametrize('encoding,dtype,scale', [('16UC1', '<u2', .001), ('16UC1', '>u2', .001),
                                               ('32FC1', '<f4', 1.)])
def test_ros_depth_units_endianness_stride_fov_and_orientation(encoding, dtype, scale):
    receiver = DepthCameraInput()
    receiver.camera_info_callback(info(8, 6))
    raw = np.arange(1, 49).reshape(6, 8).astype(dtype)
    raw[0, 0] = 0
    receiver.image_callback(image(raw, encoding, pad=3))
    sample = receiver.sample()
    assert sample is not None
    expected, _ = project_depth(raw.astype(np.float32) * scale, info(8, 6), receiver.projection)
    np.testing.assert_array_equal(sample[0], np.rot90(expected, k=-1))
    assert sample[0].shape == (36, 48)


def test_bad_camera_and_images_do_not_refresh_depth_and_duplicates_expire(monkeypatch):
    import bxi_example_py_elf3.models.depth_camera as module
    now = [10.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    receiver = DepthCameraInput()
    receiver.image_callback(image(np.full((6, 8), 1000, np.uint16)))
    assert receiver.sample() is None
    receiver.camera_info_callback(info(8, 6))
    good = image(np.full((6, 8), 1000, np.uint16))
    receiver.image_callback(good)
    assert receiver.sample()[1] == 1
    now[0] += .3
    receiver.image_callback(good)
    assert receiver.sample()[1] == 1
    bad = image(np.full((6, 8), np.nan, np.float32), '32FC1', stamp=2)
    receiver.image_callback(bad)
    assert receiver.sample()[1] == 1
    now[0] += .21
    assert receiver.sample() is None
    receiver.image_callback(image(np.full((6, 8), 1000, np.uint16), stamp=3))
    assert receiver.sample()[1] == 2
    truncated = image(np.full((6, 8), 1000, np.uint16), stamp=4)
    truncated.data = truncated.data[:4]
    receiver.image_callback(truncated)
    assert receiver.sample()[1] == 2
    assert 'incomplete' in receiver.last_error


@pytest.mark.parametrize('filename,mode', [('dagger2.onnx', 'origin_camera'),
    ('dagger_new_position.onnx', 'origin_camera'), ('normal_depth.onnx', 'depth_walk')])
def test_matches_original_mod_targets_and_inputs(filename, mode):
    # Fixture exported from unmodified com.bxi.normal_depth using real ONNX.
    policy = HumanoidGaitDepthPolicy(PACKAGE / 'policy/depth_walk' / filename, mode=mode)
    rng = np.random.default_rng(73)
    snapshot = 0
    with np.load(Path(__file__).with_name('depth_reference.npz')) as reference:
        for tick in range(45):
            q = policy.default_dof_pos + rng.normal(0., .02, 29).astype(np.float32)
            dq = rng.normal(0., .05, 29).astype(np.float32)
            omega = rng.normal(0., .01, 3).astype(np.float32)
            quat = np.array([np.cos(.03), np.sin(.03), 0., 0.], np.float32)
            command = np.array([.3, .1, .2], np.float32)
            h, w = policy.received_depth_shape
            depth = np.linspace(.1, 3.4, h * w, dtype=np.float32).reshape(h, w) + .05 * np.sin(tick * .2)
            depth[::8, ::7] = 0.
            target = policy.inference_step(q, dq, quat, omega, command, depth, tick // 3)
            if tick in (0, 5, 12, 38):
                np.testing.assert_allclose(target, reference[filename + '_target'][snapshot], atol=1e-5)
                np.testing.assert_allclose(policy.obs.ravel(), reference[filename + '_obs'][snapshot], atol=2e-5)
                np.testing.assert_allclose(policy.depth_input_buffer.ravel(), reference[filename + '_depth'][snapshot].ravel(), atol=2e-6)
                snapshot += 1


def test_recurrent_preview_reset_command_clamping_and_joint_order():
    policy = HumanoidGaitDepthPolicy(PACKAGE / 'policy/depth_walk/dagger_new_position.onnx')
    args = (policy.default_dof_pos, np.arange(29) * .01, [1., 0., 0., 0.], [0., 0., .1],
            [-.4, .9, -2.], np.full(policy.received_depth_shape, 1.5, np.float32), 1)
    policy.inference_step(*args)
    hidden = policy._hidden.copy()
    history, last_action = np.stack(policy.obs_history).copy(), policy.last_action.copy()
    depth_history = np.stack(policy.depth_history).copy()
    policy.inference_step(*args, advance=False)
    np.testing.assert_array_equal(policy._hidden, hidden)
    np.testing.assert_array_equal(np.stack(policy.obs_history), history)
    np.testing.assert_array_equal(policy.last_action, last_action)
    np.testing.assert_array_equal(np.stack(policy.depth_history), depth_history)
    np.testing.assert_allclose(policy.obs[0, 6:9], [0., .5, -1.57])
    np.testing.assert_allclose(policy.obs[0, 38:67], np.asarray(args[1])[MUJOCO_TO_ISAAC_INDEX])
    np.testing.assert_allclose(policy.target_dof_pos,
        policy.default_dof_pos + (policy.action * policy.action_scale)[ISAAC_TO_MUJOCO_INDEX])
    policy.inference_step(*args)
    assert np.any(policy._hidden != hidden)
    policy.reset()
    assert not policy.obs_history and not policy.depth_history
    np.testing.assert_array_equal(policy._hidden, 0.)
    np.testing.assert_array_equal(policy.last_action, 0.)


def scene_config(profile='dagger_new_position'):
    from bxi_example_py_elf3 import model_config as config
    settings = config.DEPTH_WALK_PROFILES[profile]
    return dict(name=config.DEPTH_CAMERA_NAME, parent=config.DEPTH_CAMERA_PARENT,
                position=settings['position'], look_down_deg=settings['look_down_deg'],
                roll_deg=config.DEPTH_CAMERA_POSE['roll_deg'], stream=dict(config.DEPTH_CAMERA_STREAM))


@pytest.mark.parametrize('profile,angle', [('dagger2', 70.), ('dagger_new_position', 65.)])
def test_generated_xml_preserves_physics_and_camera_mount(tmp_path, profile, angle):
    from bxi_example_py_elf3.utils.simulation_xml import generate_simulation_xml
    from scipy.spatial.transform import Rotation
    base_path = PACKAGE / 'robot/elf3_lite/xml/elf3.xml'
    settings = scene_config(profile)
    path = generate_simulation_xml(base_path, camera=settings, output_dir=tmp_path)
    assert ET.parse(path).getroot().get('model') == 'elf3'
    base = mujoco.MjModel.from_xml_path(str(base_path))
    depth = mujoco.MjModel.from_xml_path(path)
    assert depth.nq == base.nq == 36 and depth.nu == base.nu == 29
    for name in ['body_mass', 'body_inertia', 'body_pos', 'jnt_range', 'dof_damping', 'geom_pos',
                 'geom_size', 'geom_friction', 'actuator_ctrlrange', 'qpos0']:
        np.testing.assert_array_equal(getattr(base, name), getattr(depth, name))
    data = mujoco.MjData(depth)
    mujoco.mj_forward(depth, data)
    camera = mujoco.mj_name2id(depth, mujoco.mjtObj.mjOBJ_CAMERA, settings['name'])
    np.testing.assert_allclose(depth.cam_pos[camera], settings['position'], atol=1e-7)
    expected = Rotation.from_euler('y', angle - 90, degrees=True) * Rotation.from_euler('z', -90, degrees=True)
    np.testing.assert_allclose(data.cam_xmat[camera].reshape(3, 3), expected.as_matrix(), atol=1e-7)
    sensor = mujoco.mj_name2id(depth, mujoco.mjtObj.mjOBJ_SENSOR, settings['name'])
    np.testing.assert_allclose(depth.sensor_user[sensor, :11],
        [640, 360, 43.173360, 848, 480, 58.605976, 30, 1, .1, 10., 1])


def test_config_controls_generated_scene_and_disabled_depth(tmp_path, monkeypatch):
    from bxi_example_py_elf3 import model_config as config
    original = (PACKAGE / config.SIMULATION_MODEL_XML).read_bytes()
    first = config.get_simulation_model_file(package_share_dir=PACKAGE, output_dir=tmp_path)
    assert first == config.get_simulation_model_file(package_share_dir=PACKAGE, output_dir=tmp_path)
    stream = dict(config.DEPTH_CAMERA_STREAM, depth_width=424, depth_height=240, fps=20., preview_enabled=False)
    monkeypatch.setattr(config, 'DEPTH_CAMERA_STREAM', stream)
    monkeypatch.setattr(config, 'SIMULATION_TERRAIN_ENABLED', True)
    steps = [dict(config.SIMULATION_TERRAIN_STEPS[0], position=(2., 0., .08), size=(.4, .8, .08))]
    monkeypatch.setattr(config, 'SIMULATION_TERRAIN_STEPS', steps)
    second = config.get_simulation_model_file(package_share_dir=PACKAGE, output_dir=tmp_path)
    assert first != second and Path(first).is_file()  # An active launch keeps its file.
    model = mujoco.MjModel.from_xml_path(second)
    sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, config.DEPTH_CAMERA_NAME)
    np.testing.assert_allclose(model.sensor_user[sensor, 3:7], [424, 240, 58.605976, 20])
    assert model.sensor_user[sensor, 10] == 0.
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'depth_step_1')
    np.testing.assert_allclose(model.geom_pos[geom], [2., 0., .08])
    np.testing.assert_allclose(model.geom_size[geom], [.4, .8, .08])
    monkeypatch.setattr(config, 'DEPTH_WALK_ENABLED', False)
    disabled = mujoco.MjModel.from_xml_path(config.get_simulation_model_file(package_share_dir=PACKAGE, output_dir=tmp_path))
    assert mujoco.mj_name2id(disabled, mujoco.mjtObj.mjOBJ_SENSOR, config.DEPTH_CAMERA_NAME) == -1
    assert (PACKAGE / config.SIMULATION_MODEL_XML).read_bytes() == original


def test_invalid_camera_parent_and_stream_fail_before_launch(tmp_path):
    from bxi_example_py_elf3.utils.simulation_xml import generate_simulation_xml
    camera = scene_config()
    camera['parent'] = 'missing_body'
    with pytest.raises(ValueError, match='parent body'):
        generate_simulation_xml(PACKAGE / 'robot/elf3_lite/xml/elf3.xml', camera=camera, output_dir=tmp_path)
    camera = scene_config()
    camera['stream']['depth_width'] = -1
    with pytest.raises(ValueError, match='resolutions'):
        generate_simulation_xml(PACKAGE / 'robot/elf3_lite/xml/elf3.xml', camera=camera, output_dir=tmp_path)
