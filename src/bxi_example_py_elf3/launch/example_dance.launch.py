import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

def generate_launch_description():

    xml_file_name = "robot/elf3_lite/xml/elf3.xml"
    xml_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), xml_file_name)
    
    npz_file_dict = {
        
        # isaaclab
        "lie_down": "policy/dance_isaaclab/lie_down.npz",
        "getup_face": "policy/dance_isaaclab/getup_face.npz",
        "getup_back": "policy/dance_isaaclab/getup_back.npz",
        
        
        # "lafan1": "policy/lafan1_npz/ground1_subject1.npz",
        # "lafan1": "policy/lafan1_npz/fight1_subject2.npz",
        # "lafan1": "policy/lafan1_npz/aiming1_subject1.npz",
        # "lafan1": "policy/lafan1_npz/multipleActions1_subject1.npz",
        # "lafan1": "policy/lafan1_npz/fallAndGetUp1_subject4.npz",
        # "lafan1": "policy/lafan1_npz/aiming1_subject1.npz",
        # "lafan1": "policy/lafan1_npz/run1_subject2.npz",
        # "lafan1": "policy/lafan1_npz/jumps1_subject2.npz",
        # "lafan1": "policy/dance_isaaclab/pico_fix_fix_fix_final.npz",
        # "lafan1": "policy/lafan1_npz/run1_subject5.npz",
        # "lafan1": "policy/lafan1_npz/walk1_subject2.npz",
        # "lafan1": "policy/lafan1_npz/dance1_subject2.npz",
        "lafan1": "policy/lafan1_npz/walk1_subject6.npz",
        
        # "cmu": "policy/cmu_1h_new/75/75_08_stageii.npz",    #360旋转
        # "cmu": "policy/cmu_1h_new/88/88_11_stageii.npz",    #热身
        # "cmu": "policy/cmu_1h_new/90/90_01_stageii.npz",    #后滚翻
        # "cmu": "policy/cmu_1h_new/90/90_02_stageii.npz",    #侧手翻
        # "cmu": "policy/cmu_1h_new/90/90_03_stageii.npz",    #侧手翻
        # "cmu": "policy/cmu_1h_new/90/90_04_stageii.npz",    #侧手翻
        # "cmu": "policy/cmu_1h_new/90/90_05_stageii.npz",    #回旋踢
        # "cmu": "policy/cmu_1h_new/90/90_06_stageii.npz",    #回旋踢
        # "cmu": "policy/cmu_1h_new/90/90_30_stageii.npz",
        # "cmu": "policy/cmu_1h_new/15/15_01_stageii.npz",
        # "cmu": "policy/cmu_1h_new/15/15_01_nur.npz",
        
    }  
    onnx_file_dict = {
        "amp_walk": "policy/amp_dwaq3.onnx",##symmetry
        # "amp_run": "policy/myrun6.onnx",##sim 5.5=6
        "amp_run": "policy/myrun10.onnx",##hw 5=5.18
        # "amp_run": "policy/myrun14.onnx",#run_dwaq
        
        "host": "policy/elf3_ground.onnx",
        "rgmt": "policy/rgmtr_119600.onnx",
        
        
        # isaaclab3
        "getup_face": "policy/dance_isaaclab/getup_face.onnx",
        "getup_back": "policy/dance_isaaclab/getup_back.onnx",

    }
    
    for key, value in npz_file_dict.items():
        npz_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_elf3"), value)
    for key, value in onnx_file_dict.items():
        onnx_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_elf3"), value)

    return LaunchDescription(
        [
            Node(
                package="mujoco",
                executable="simulation",
                name="simulation_mujoco",
                output="screen",
                parameters=[
                    {"simulation/model_file": xml_file},
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
                    {"/topic_prefix": "simulation/"},
                    {"/use_hardware": False},
                    {"/npz_file_dict": json.dumps(npz_file_dict)},
                    {"/onnx_file_dict": json.dumps(onnx_file_dict)},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
