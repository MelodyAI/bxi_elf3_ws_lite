"""Check D435i -> ROS -> depth policy with existing numerical dependencies."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--domain', type=int, default=88)
parser.add_argument('--seconds', type=float, default=10.)
parser.add_argument('--output', type=Path, default=Path('/tmp/bxi_d435i_check'))
options = parser.parse_args()
if options.seconds < 3.:
    parser.error('--seconds must be at least 3')
os.environ['ROS_DOMAIN_ID'] = str(options.domain)
os.environ['ROS_LOCALHOST_ONLY'] = '1'
from bxi_example_py_elf3.realsense_runtime import use_bundled_realsense
use_bundled_realsense()
import numpy as np
import onnxruntime
import pyrealsense2
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from bxi_example_py_elf3.models.depth_camera import DepthCameraInput
from bxi_example_py_elf3.models.depth import HumanoidGaitDepthPolicy
from bxi_example_py_elf3.model_config import (
    DEPTH_WALK_ENABLED, DEPTH_CAMERA_NAME, DEPTH_CAMERA_STREAM, DEPTH_WALK_MODE, DEPTH_WALK_PREPROCESS, DEPTH_TIMEOUT_S, get_model_file_dicts,
)

if not DEPTH_WALK_ENABLED:
    raise RuntimeError('Set DEPTH_WALK_ENABLED=True before checking the camera launch')
out = options.output
out.mkdir(parents=True, exist_ok=True)
log = (out/'camera.log').open('w')
camera = subprocess.Popen(['ros2', 'launch', 'bxi_example_py_elf3', 'depth_camera.launch.py'],
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
node = None
try:
    _, models = get_model_file_dicts()
    policy = HumanoidGaitDepthPolicy(models['depth_walk'], mode=DEPTH_WALK_MODE, **DEPTH_WALK_PREPROCESS)
    receiver = DepthCameraInput(mode=DEPTH_WALK_MODE, timeout_s=DEPTH_TIMEOUT_S,
                                depth_range=DEPTH_WALK_PREPROCESS.get('depth_range'))
    rclpy.init()
    node = rclpy.create_node('check_d435i_depth_policy')
    samples, timings, targets, control_times, depth_updates = [], [], [], [], []
    latest = [None]
    base = f'/hardware/{DEPTH_CAMERA_NAME}/depth'
    node.create_subscription(CameraInfo, base+'/camera_info', receiver.camera_info_callback,
                             qos_profile_sensor_data)
    def image_callback(msg):
        receiver.image_callback(msg)
        samples.append((time.monotonic(), msg.header.stamp.sec*10**9+msg.header.stamp.nanosec))
        latest[0] = msg
    node.create_subscription(Image, base+'/image_rect_raw', image_callback, qos_profile_sensor_data)
    def spin(seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.005)
    deadline = time.monotonic()+20.
    while time.monotonic() < deadline and receiver.sample(advance=False) is None and camera.poll() is None:
        spin(.05)
    assert receiver.sample(advance=False) is not None, (out/'camera.log').read_text()[-3000:]
    spin(1.)
    samples.clear()
    deadline = time.monotonic()+options.seconds
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.002)
        now = time.monotonic()
        if now >= next_tick:
            sample = receiver.sample()
            assert sample is not None, (receiver.last_error or 'depth timeout',
                                       {'received_frames':len(samples), 'age_ms':receiver.age_ms()})
            control_times.append(now)
            if not depth_updates or sample[1] != depth_updates[-1][1]:
                depth_updates.append((now, sample[1]))
            begin = time.perf_counter()
            target = policy.inference_step(policy.default_dof_pos, np.zeros(29), [1.,0.,0.,0.],
                                          np.zeros(3), [.2,0.,0.], *sample)
            timings.append((time.perf_counter()-begin)*1000.)
            targets.append(target)
            next_tick += .02
    msg = latest[0]
    raw = np.frombuffer(msg.data, '<u2').reshape(msg.height, msg.width)
    np.save(out/'depth_mm.npy', raw)
    np.save(out/'policy_depth.npy', policy.depth_input_buffer)
    np.save(out/'targets.npy', np.asarray(targets))
    assert np.isfinite(targets).all()
    assert policy.depth_input_buffer.shape[1:] == ((36,36) if DEPTH_WALK_MODE == 'origin_camera' else (8,21,32))
    hz = (len(samples)-1)/(samples[-1][0]-samples[0][0])
    target_hz = int(DEPTH_CAMERA_STREAM['fps'])
    assert hz >= target_hz*.9, f'Camera below 90% of configured {target_hz} Hz: {hz:g}'
    assert len(set(s[1] for s in samples)) == len(samples), 'Duplicate camera timestamps'
    depth_hz = (len(depth_updates)-1)/(depth_updates[-1][0]-depth_updates[0][0])
    expected_depth_hz = min(float(target_hz), 1./receiver.update_period_s)
    assert expected_depth_hz*.9 <= depth_hz <= expected_depth_hz*1.1, f'Depth selection rate: {depth_hz:g}'
    report = {'passed': True, 'camera_hz': hz, 'target_publish_hz': target_hz, 'camera_frames': len(samples),
              'model_depth_update_hz':depth_hz, 'model_depth_updates':len(depth_updates),
              'model_depth_target_hz':expected_depth_hz,
              'control_hz':(len(control_times)-1)/(control_times[-1]-control_times[0]),
              'encoding': msg.encoding, 'resolution': [msg.width,msg.height],
              'filtered_depth_valid_ratio': float(np.mean(raw > 0)),
              'projection_roi': receiver.last_crop, 'model': policy.model_onnx_path,
              'depth_preprocess': {'crop_rows':policy.crop_rows, 'depth_range':policy.depth_range},
              'policy_input_shape': list(policy.depth_input_buffer.shape),
              'policy_input_range': [float(policy.depth_input_buffer.min()),float(policy.depth_input_buffer.max())],
              'inference_frames': len(targets), 'inference_ms_p99': float(np.percentile(timings,99)),
              'hidden_state_norm': None if policy._hidden is None else float(np.linalg.norm(policy._hidden)),
              'runtime_paths': {'numpy':np.__file__,'onnxruntime':onnxruntime.__file__,'sdk':pyrealsense2.__file__},
              'robot_hardware_started': False}
    os.killpg(camera.pid, signal.SIGINT)
    camera.wait(timeout=10)
    log.flush()
    shutdown_log = (out/'camera.log').read_text()
    assert 'process has finished cleanly' in shutdown_log and 'process has died' not in shutdown_log, shutdown_log[-3000:]
    report['clean_shutdown'] = True
    count = len(samples)
    spin(DEPTH_TIMEOUT_S+.3)  # Drain any already-delivered images before measuring expiration.
    spin(DEPTH_TIMEOUT_S+.1)
    assert receiver.sample(advance=False) is None, 'Stale depth remained ready after stopping the camera'
    report['stopped_stream_expires'] = True
    report['shutdown_tail_frames'] = len(samples)-count
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:
    if camera.poll() is None:
        os.killpg(camera.pid,signal.SIGINT)
        try:
            camera.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(camera.pid,signal.SIGTERM)
            camera.wait(timeout=5)
    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    log.close()
