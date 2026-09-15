import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

from bxi_example_py_elf3.model_config import RGMT_REFERENCE_MODE, PICO_POSE_ENDPOINT, get_model_file_dicts
from bxi_example_py_elf3.model_config import PICO_ARM_VELOCITY, PICO_ARM_ACCELERATION

def generate_launch_description():

    npz_file_dict, onnx_file_dict = get_model_file_dicts()

    return LaunchDescription(
        [
            Node(
                package="hardware_elf3",
                executable="hardware_elf3",
                name="hardware_elf3",
                output="screen",
                parameters=[
                    {"hardware_config/imu": True},      #start imu
                    {"hardware_config/motor_pwr": True}, #motor poweron
                    {"hardware_config/motor_disable": 0x60000000}, #motor disable head
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_dance",
                name="bxi_example_py_elf3_dance",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/use_hardware": True},
                    {"/rgmt_reference_mode": RGMT_REFERENCE_MODE},
                    {"/pico_pose_endpoint": PICO_POSE_ENDPOINT},
                    {"/pico_arm_velocity": PICO_ARM_VELOCITY},
                    {"/pico_arm_acceleration": PICO_ARM_ACCELERATION},
                    {"/npz_file_dict": json.dumps(npz_file_dict)},
                    {"/onnx_file_dict": json.dumps(onnx_file_dict)},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
