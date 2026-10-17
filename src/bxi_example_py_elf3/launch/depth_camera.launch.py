"""Camera-only preview/check; does not start the robot hardware or controller."""
from launch import LaunchDescription
from bxi_example_py_elf3.depth_launch import make_depth_camera_launch_actions


def generate_launch_description():
    return LaunchDescription(make_depth_camera_launch_actions())
