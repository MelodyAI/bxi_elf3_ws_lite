import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

from bxi_example_py_elf3.model_config import (
    RGMT_REFERENCE_MODE, PICO_POSE_ENDPOINT, PICO_UMR_PREDICTION_TIME_S, get_model_file_dicts,
    DEPTH_WALK_ENABLED, DEPTH_WALK_MODE, DEPTH_CAMERA_NAME, DEPTH_TIMEOUT_S, get_simulation_model_file,
)
from bxi_example_py_elf3.pico_launch import make_pico_launch_actions

def generate_launch_description():

    xml_file = get_simulation_model_file()
    
    npz_file_dict, onnx_file_dict = get_model_file_dicts()

    actions = make_pico_launch_actions()
    actions.extend([
            Node(
                package="mujoco",
                executable="simulation",
                name="simulation_mujoco",
                output="screen",
                parameters=[
                    {"simulation/model_file": xml_file},
                ],
                emulate_tty=True,
            ),

            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_dance",
                name="bxi_example_py_elf3_dance",
                output="screen",
                parameters=[
                    {"/topic_prefix": "simulation/"},
                    {"/use_hardware": False},
                    {"/rgmt_reference_mode": RGMT_REFERENCE_MODE},
                    {"/pico_pose_endpoint": PICO_POSE_ENDPOINT},
                    {"/pico_umr_prediction_time_s": PICO_UMR_PREDICTION_TIME_S},
                    {"/depth_walk_enabled": DEPTH_WALK_ENABLED},
                    {"/depth_walk_mode": DEPTH_WALK_MODE},
                    {"/depth_camera_name": DEPTH_CAMERA_NAME},
                    {"/depth_timeout_s": DEPTH_TIMEOUT_S},
                    {"/npz_file_dict": json.dumps(npz_file_dict)},
                    {"/onnx_file_dict": json.dumps(onnx_file_dict)},
                ],
                emulate_tty=True,
            ),
        ])
    return LaunchDescription(actions)
