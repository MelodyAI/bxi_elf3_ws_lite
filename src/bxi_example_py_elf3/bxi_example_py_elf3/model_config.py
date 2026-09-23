"""Shared model file configuration for simulation and hardware launches."""

from pathlib import Path
import os
import sys

from ament_index_python.packages import get_package_share_path


PACKAGE_NAME = "bxi_example_py_elf3"

# Change this one value to select the reference source used by the X key.
# ``npz`` uses the existing retargeted RGMT motion; ``neural_retarget`` uses
# the SMPL-X/ACCAD input and the Transformer adapter before RGMT.
# RGMT_REFERENCE_MODE = "npz"
# RGMT_REFERENCE_MODE = "neural_retarget"  # "npz", "neural_retarget", "pico", or "zerolab"
RGMT_REFERENCE_MODE = "zerolab"
PICO_POSE_ENDPOINT = "tcp://127.0.0.1:28704"
# ZeroLab's source node publishes packed SMPL windows on this endpoint.
ZEROLAB_POSE_ENDPOINT = "tcp://127.0.0.1:5558"

# PICO runtime is optional.  Set RGMT_REFERENCE_MODE to "pico" to enable
# launch-managed service and sender processes.  A matching prebuilt runtime is
# bundled under third_party/pico; external paths can still be configured.
PICO_AUTO_START_SERVICE = True
PICO_AUTO_START_SENDER = True
PICO_USE_BUNDLED_RUNTIME = True
PICO_SERVICE_ROOT = os.environ.get("PICO_SERVICE_ROOT", "")
PICO_SERVICE_START_DELAY = float(os.environ.get("PICO_SERVICE_START_DELAY", "2.0"))
PICO_SENDER_ENDPOINT = os.environ.get("PICO_SENDER_ENDPOINT", "tcp://*:28704")
PICO_SENDER_FPS = float(os.environ.get("PICO_SENDER_FPS", "50.0"))
PICO_SENDER_PYTHON = os.environ.get("PICO_SENDER_PYTHON", sys.executable)
PICO_SENDER_PYTHONPATH = os.environ.get("PICO_SENDER_PYTHONPATH", "")

# Keep these paths relative to the installed package share directory.  Both
# simulation and hardware launch files use the same dictionaries.
NPZ_FILE_PATHS = {

    # 官方 SMPL-X 模型：用于 neural_retarget 的形状相关骨架偏移。
    "smplx_model": "policy/SMPLX_NEUTRAL.npz",

    #smplx
    "smplx": "policy/cmu_1h_new/88/88_06_poses.npz",    #旋转踢
    # "smplx": "policy/ACCAD/Male2MartialArtsExtended_c3d/Form_1_stageii.npz",
    # "smplx": "policy/ACCAD/amp/walk/pico_stageii.npz",
    # "smplx": "policy/ACCAD/amp/walk/0007_Walking001_stageii.npz",
    # "smplx": "policy/ACCAD/amp/run/0005_Jogging001_stageii.npz",
    # "smplx": "policy/ACCAD/Male2MartialArtsKicks_c3d/G18-__push_kick_right_stageii.npz",
    # "smplx": "policy/ACCAD/Female1General_c3d/A15_-_skip_to_stand_stageii.npz",

    #lafan1
    # "rgmt": "policy/lafan1_npz/ground1_subject1.npz",
    # "rgmt": "policy/lafan1_npz/fight1_subject2.npz",
    # "rgmt": "policy/lafan1_npz/aiming1_subject1.npz",
    # "rgmt": "policy/lafan1_npz/multipleActions1_subject1.npz",
    # "rgmt": "policy/lafan1_npz/fallAndGetUp1_subject4.npz",
    # "rgmt": "policy/lafan1_npz/aiming1_subject1.npz",
    # "rgmt": "policy/lafan1_npz/run1_subject2.npz",
    # "rgmt": "policy/lafan1_npz/jumps1_subject2.npz",
    # "rgmt": "policy/dance_isaaclab/pico_fix_fix_fix_final.npz",
    # "rgmt": "policy/lafan1_npz/run1_subject5.npz",
    # "rgmt": "policy/lafan1_npz/walk1_subject2.npz",
    # "rgmt": "policy/lafan1_npz/dance1_subject2.npz",
    # "rgmt": "policy/lafan1_npz/walk1_subject5.npz",

    #cmu
    # "rgmt": "policy/cmu_1h_new/75/75_15_stageii.npz",    #跳
    # "rgmt": "policy/cmu_1h_new/80/80_15_stageii.npz",    #操作
    
    # "rgmt": "policy/cmu_1h_new/85/85_01_stageii.npz",    #升龙踢腿
    # "rgmt": "policy/cmu_1h_new/85/85_03_stageii.npz",    #街舞
    # "rgmt": "policy/cmu_1h_new/85/85_04_stageii.npz",    #街舞
    # "rgmt": "policy/cmu_1h_new/85/85_05_stageii.npz",    #街舞
    # "rgmt": "policy/cmu_1h_new/85/85_06_stageii.npz",    #街舞
    # "rgmt": "policy/cmu_1h_new/85/85_07_stageii.npz",    #街舞-后脚翻
    # "rgmt": "policy/cmu_1h_new/85/85_08_stageii.npz",    #街舞-托马斯
    # "rgmt": "policy/cmu_1h_new/85/85_09_stageii.npz",    #马步
    # "rgmt": "policy/cmu_1h_new/85/85_10_stageii.npz",    #bad?
    # "rgmt": "policy/cmu_1h_new/85/85_11_stageii.npz",    #街舞
    # "rgmt": "policy/cmu_1h_new/85/85_12_stageii.npz",    #街舞-hard
    # "rgmt": "policy/cmu_1h_new/85/85_13_stageii.npz",    #街舞-倒立x2
    # "rgmt": "policy/cmu_1h_new/85/85_14_stageii.npz",    #街舞-后手翻
    
    # "rgmt": "policy/cmu_1h_new/87/87_01_stageii.npz",    #慢速？
    
    # "rgmt": "policy/cmu_1h_new/88/88_03_stageii.npz",    #马步
    # "rgmt": "policy/cmu_1h_new/88/88_05_stageii.npz",    #bad
    # "rgmt": "policy/cmu_1h_new/88/88_06_stageii.npz",    #旋转踢
    # "rgmt": "policy/cmu_1h_new/88/88_07_stageii.npz",    #侧倒立
    # "rgmt": "policy/cmu_1h_new/88/88_08_stageii.npz",    #后手翻
    # "rgmt": "policy/cmu_1h_new/88/88_09_stageii.npz",    #后手翻
    # "rgmt": "policy/cmu_1h_new/88/88_10_stageii.npz",    #热身
    # "rgmt": "policy/cmu_1h_new/88/88_11_stageii.npz",    #热身-hard
    
    # "rgmt": "policy/cmu_1h_new/90/90_01_stageii.npz",    #后滚翻
    "rgmt": "policy/cmu_1h_new/90/90_02_stageii.npz",    #侧手翻
    # "rgmt": "policy/cmu_1h_new/90/90_03_stageii.npz",    #侧手翻
    # "rgmt": "policy/cmu_1h_new/90/90_04_stageii.npz",    #侧手翻
    # "rgmt": "policy/cmu_1h_new/90/90_05_stageii.npz",    #回旋踢
    # "rgmt": "policy/cmu_1h_new/90/90_06_stageii.npz",    #回旋踢
    # "rgmt": "policy/cmu_1h_new/90/90_07_stageii.npz",    #回旋踢
    
    # "rgmt": "policy/cmu_1h_new/90/90_08_stageii.npz",    #加速？
    
    # "rgmt": "policy/cmu_1h_new/90/90_11_stageii.npz",    #bad?
    # "rgmt": "policy/cmu_1h_new/90/90_12_stageii.npz",    #后空翻
    # "rgmt": "policy/cmu_1h_new/90/90_13_stageii.npz",    #后空翻
    # "rgmt": "policy/cmu_1h_new/90/90_14_stageii.npz",    #前手翻
    # "rgmt": "policy/cmu_1h_new/90/90_15_stageii.npz",    #前手翻
    # "rgmt": "policy/cmu_1h_new/90/90_16_stageii.npz",    #旋转爬下
    # "rgmt": "policy/cmu_1h_new/90/90_18_stageii.npz",    #后躺爬下
    # "rgmt": "policy/cmu_1h_new/90/90_19_stageii.npz",    #猴子
    # "rgmt": "policy/cmu_1h_new/90/90_22_stageii.npz",    #walk
    # "rgmt": "policy/cmu_1h_new/90/90_23_stageii.npz",    #walk
    # "rgmt": "policy/cmu_1h_new/90/90_28_stageii.npz",    #乌龙角柱
    # "rgmt": "policy/cmu_1h_new/90/90_29_stageii.npz",    #连续翻
    # "rgmt": "policy/cmu_1h_new/90/90_30_stageii.npz",   #俄罗斯舞
    # "rgmt": "policy/cmu_1h_new/90/90_31_stageii.npz",    #俄罗斯舞
    # "rgmt": "policy/cmu_1h_new/90/90_32_stageii.npz",    #walk_bad
    # "rgmt": "policy/cmu_1h_new/90/90_33_stageii.npz",    #前滚翻
    # "rgmt": "policy/cmu_1h_new/90/90_34_stageii.npz",    #前滚翻
    # "rgmt": "policy/cmu_1h_new/90/90_35_stageii.npz",    #前滚翻
    # "rgmt": "policy/cmu_1h_new/90/90_36_stageii.npz",    #前滚翻
    
    
    # "rgmt": "policy/cmu_1h_new/75/75_11_stageii.npz",    #连续前跳
    # "rgmt": "policy/cmu_1h_new/75/75_12_stageii.npz",    #转圈跳
    # "rgmt": "policy/cmu_1h_new/75/75_15_stageii.npz",    #跳 后退
    # "rgmt": "policy/cmu_1h_new/75/75_08_stageii.npz",    #360旋转
    
    # "rgmt": "policy/cmu_1h_new/15/15_01_stageii.npz",   #walk
    # "rgmt": "policy/cmu_1h_new/15/15_01_nur.npz",   #walk slide

    # isaaclab
    "lie_down": "policy/dance_isaaclab/lie_down.npz",
    "getup_face": "policy/dance_isaaclab/getup_face.npz",
    "getup_back": "policy/dance_isaaclab/getup_back.npz",
}

ONNX_FILE_PATHS = {
    "amp_walk": "policy/amp_dwaq3.onnx",##symmetry
    # "amp_run": "policy/myrun6.onnx",##sim 5.5=6
    "amp_run": "policy/myrun10.onnx",##hw 5=5.18
    # "amp_run": "policy/myrun14.onnx",#run_dwaq

    # "rgmt": "policy/rgmtr_195000.onnx",
    # "rgmt": "policy/rgmtr_9200.onnx",
    # "rgmt": "policy/rgmtr_13000.onnx",
    # "rgmt": "policy/rgmtr_24600.onnx",
    "rgmt": "policy/rgmtr_40200.onnx",
    
    "neural_retarget": "policy/neural_retarget_fast.onnx",
    
    # isaaclab3
    "lie_down": "policy/dance_isaaclab/lie_down.onnx",
    "getup_face": "policy/dance_isaaclab/getup_face.onnx",
    "getup_back": "policy/dance_isaaclab/getup_back.onnx",
}


def get_model_file_dicts():
    """Return absolute model paths for the installed package resources."""
    share_dir = Path(get_package_share_path(PACKAGE_NAME))
    npz_file_dict = {
        key: str(share_dir / relative_path)
        for key, relative_path in NPZ_FILE_PATHS.items()
    }
    onnx_file_dict = {
        key: str(share_dir / relative_path)
        for key, relative_path in ONNX_FILE_PATHS.items()
    }
    return npz_file_dict, onnx_file_dict
