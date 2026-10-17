"""D435i acquisition and Instinct NS hole filling, separate from control."""
from .realsense_runtime import use_bundled_realsense
use_bundled_realsense()

import signal
import time
from threading import Event, Thread

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image

from .model_config import DEPTH_CAMERA_HARDWARE, DEPTH_CAMERA_NAME, DEPTH_CAMERA_STREAM, DEPTH_WALK_MODE
from .d435i_filters import instinct_depth
from .models.depth_camera import DepthCameraInput


def depth_millimetres(raw, scale):
    """Publish 16UC1 in millimetres even when device depth units are customized."""
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError("Invalid RealSense depth scale")
    if abs(scale - .001) < 1e-9:
        return np.ascontiguousarray(raw, dtype='<u2')
    return np.clip(np.rint(np.asarray(raw, dtype=np.float32) * (scale * 1000.)),
                   0., 65535.).astype('<u2')


def camera_info(intrinsics, header):
    info = CameraInfo()
    info.header = header
    info.width, info.height = intrinsics.width, intrinsics.height
    info.distortion_model = 'plumb_bob'
    info.d = list(intrinsics.coeffs)
    info.k = [intrinsics.fx, 0., intrinsics.ppx, 0., intrinsics.fy, intrinsics.ppy, 0., 0., 1.]
    info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    info.p = [intrinsics.fx, 0., intrinsics.ppx, 0.,
              0., intrinsics.fy, intrinsics.ppy, 0., 0., 0., 1., 0.]
    return info


def select_device(context, serial_no):
    devices = []
    for device in context.query_devices():
        if device.is_update_device():
            continue
        if device.get_info(rs.camera_info.product_id).upper() != '0B3A':
            continue
        serial = device.get_info(rs.camera_info.serial_number)
        if not serial_no or serial == serial_no:
            devices.append(device)
    if not devices:
        raise RuntimeError("No matching D435i; check USB connection, permissions and recovery mode")
    if len(devices) != 1:
        raise RuntimeError("Multiple D435i cameras connected; set DEPTH_CAMERA_HARDWARE['serial_no']")
    return devices[0]


class D435iCamera(Node):
    def __init__(self):
        super().__init__('d435i_camera')
        self.declare_parameter('/topic_prefix', 'hardware/')
        self.declare_parameter('/depth_camera_name', DEPTH_CAMERA_NAME)
        self.declare_parameter('/depth_walk_mode', DEPTH_WALK_MODE)
        self.depth_projection = DepthCameraInput(mode=str(self.get_parameter('/depth_walk_mode').value)).projection
        prefix = str(self.get_parameter('/topic_prefix').value).strip('/')
        self.camera_name = str(self.get_parameter('/depth_camera_name').value).strip('/')
        base = '/' + '/'.join(part for part in (prefix, self.camera_name) if part)
        defaults = dict(DEPTH_CAMERA_HARDWARE)
        for key in ('depth_width', 'depth_height', 'color_width', 'color_height'):
            defaults[key] = int(DEPTH_CAMERA_STREAM[key])
        defaults['fps'] = int(DEPTH_CAMERA_STREAM['fps'])
        self.settings = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            self.settings[name] = self.get_parameter(name).value
        if (not np.isfinite(self.settings['invalid_below_m']) or self.settings['invalid_below_m'] < 0.
                or not np.isfinite(self.settings['inpaint_radius_px']) or self.settings['inpaint_radius_px'] <= 0.):
            raise ValueError('Invalid Instinct depth threshold or inpaint radius')
        if not self.camera_name or any(self.settings[key] <= 0 for key in
                ('depth_width', 'depth_height', 'color_width', 'color_height', 'fps', 'reconnect_s', 'frame_timeout_s')):
            raise ValueError('Camera name, dimensions, FPS and timeouts must be positive')
        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability,
                         reliability=qos_profile_sensor_data.reliability)
        self.depth_pub = self.create_publisher(Image, base + '/depth/image_rect_raw', qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, base + '/depth/camera_info', qos)
        self.color_pub = self.color_info_pub = None
        if self.settings['color_enabled']:
            self.color_pub = self.create_publisher(Image, base + '/color/image_raw', qos)
            self.color_info_pub = self.create_publisher(CameraInfo, base + '/color/camera_info', qos)
        self.get_logger().info(f'D435i SDK: {rs.__file__}; depth topic: {base}/depth/image_rect_raw')
        self._stop = Event()
        self._worker = Thread(target=self._run, name='d435i_capture', daemon=True)
        self._worker.start()

    def _publish(self, frame, stamp, *, depth, scale=.001):
        raw = np.asanyarray(frame.get_data())
        array = depth_millimetres(raw, scale) if depth else np.ascontiguousarray(raw)
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = self.camera_name + ('_depth_optical_frame' if depth else '_color_optical_frame')
        intrinsics = frame.profile.as_video_stream_profile().intrinsics
        info = camera_info(intrinsics, image.header)
        if depth:
            array, info = instinct_depth(array, info, self.depth_projection,
                invalid_below_m=self.settings['invalid_below_m'], radius_px=self.settings['inpaint_radius_px'])
        image.height, image.width = array.shape[:2]
        image.encoding = '16UC1' if depth else 'rgb8'
        image.is_bigendian = False
        image.step = image.width * (2 if depth else 3)
        # Bulk-copy avoids per-pixel Python checks in the generated ROS setter.
        image.data.frombytes(array.tobytes())
        image_pub = self.depth_pub if depth else self.color_pub
        info_pub = self.depth_info_pub if depth else self.color_info_pub
        info_pub.publish(info)
        image_pub.publish(image)

    def _run(self):
        settings = self.settings
        context = rs.context()
        while not self._stop.is_set() and rclpy.ok():
            pipeline = None
            started = False
            try:
                device = select_device(context, settings['serial_no'])
                serial = device.get_info(rs.camera_info.serial_number)
                pipeline = rs.pipeline(context)
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.depth, settings['depth_width'], settings['depth_height'],
                                     rs.format.z16, settings['fps'])
                if settings['color_enabled']:
                    config.enable_stream(rs.stream.color, settings['color_width'], settings['color_height'],
                                         rs.format.rgb8, settings['fps'])
                profile = pipeline.start(config)
                started = True
                sensor = profile.get_device().first_depth_sensor()
                scale = sensor.get_depth_scale()
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1.)
                self.get_logger().info(f'D435i {serial}: {settings["depth_width"]}x{settings["depth_height"]}'
                                       f' capture={settings["fps"]} FPS, publish={settings["fps"]} Hz, depth scale={scale:g} m; '
                                       'filter=Instinct NS; streaming')
                last_number = None
                last_frame = time.monotonic()
                while not self._stop.is_set() and rclpy.ok():
                    frames = pipeline.wait_for_frames(int(settings['frame_timeout_s'] * 1000))
                    frame = frames.get_depth_frame()
                    if not frame or frame.get_frame_number() == last_number:
                        if time.monotonic() - last_frame > settings['frame_timeout_s']:
                            raise RuntimeError('No new D435i depth frames')
                        continue
                    stamp = self.get_clock().now().to_msg()
                    last_number = frame.get_frame_number()
                    last_frame = time.monotonic()
                    if self._stop.is_set():
                        break
                    self._publish(frame, stamp, depth=True, scale=scale)
                    if settings['color_enabled']:
                        color = frames.get_color_frame()
                        if color:
                            self._publish(color, stamp, depth=False)
            except Exception as error:
                if not self._stop.is_set() and rclpy.ok():
                    self.get_logger().warning(f'D435i stopped: {error}; retrying in {settings["reconnect_s"]:g}s. '
                                              'Use the project root hardware deployment or configure USB permissions.')
            finally:
                if pipeline is not None and started:
                    try:
                        pipeline.stop()
                    except RuntimeError:
                        pass
            self._stop.wait(settings['reconnect_s'])

    def destroy_node(self):
        self._stop.set()
        self._worker.join(timeout=self.settings['frame_timeout_s'] + 5.)
        return super().destroy_node()


def main(args=None):
    # A terminal SIGINT and launch's forwarded SIGINT can both arrive. Keep
    # repeated shutdown signals from interrupting SDK thread/pipeline cleanup.
    stopped = Event()
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stopped.set())
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = D435iCamera()
        while rclpy.ok() and not stopped.is_set():
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
