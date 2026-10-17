"""D435i launch action shared by hardware deployment and camera-only checks."""
from launch_ros.actions import Node

from .model_config import (
    DEPTH_WALK_ENABLED, DEPTH_CAMERA_HARDWARE, DEPTH_CAMERA_NAME, DEPTH_CAMERA_STREAM, DEPTH_WALK_MODE,
)


def make_depth_camera_launch_actions():
    if not DEPTH_WALK_ENABLED:
        return []
    parameters = dict(DEPTH_CAMERA_HARDWARE)
    parameters.update({'/topic_prefix': 'hardware/', '/depth_camera_name': DEPTH_CAMERA_NAME,
                       '/depth_walk_mode': DEPTH_WALK_MODE,
                       'fps': int(DEPTH_CAMERA_STREAM['fps'])})
    for key in ('depth_width', 'depth_height', 'color_width', 'color_height'):
        parameters[key] = int(DEPTH_CAMERA_STREAM[key])
    return [Node(package='bxi_example_py_elf3', executable='d435i_camera',
                 name='d435i_camera', output='screen', parameters=[parameters], emulate_tty=True)]
