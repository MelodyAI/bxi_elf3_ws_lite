import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

from bxi_example_py_elf3.model_config import RGMT_REFERENCE_MODE, PICO_POSE_ENDPOINT, ZEROLAB_POSE_ENDPOINT, get_model_file_dicts
from bxi_example_py_elf3.pico_launch import make_pico_launch_actions

def generate_launch_description():

    xml_file_name = "robot/elf3_lite/xml/elf3.xml"
    xml_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), xml_file_name)
    
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
                    {"/zerolab_pose_endpoint": ZEROLAB_POSE_ENDPOINT},
                    {"/npz_file_dict": json.dumps(npz_file_dict)},
                    {"/onnx_file_dict": json.dumps(onnx_file_dict)},
                ],
                emulate_tty=True,
            ),
        ])
    return LaunchDescription(actions)
