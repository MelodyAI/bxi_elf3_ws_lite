"""Shared model file configuration for simulation and hardware launches."""

from pathlib import Path

from ament_index_python.packages import get_package_share_path


PACKAGE_NAME = "bxi_example_py_elf3"

# Change this one value to select the reference source used by the X key.
# ``npz`` uses the existing retargeted RGMT motion; ``neural_retarget`` uses
# the SMPL-X/ACCAD input and the Transformer adapter before RGMT.
# RGMT_REFERENCE_MODE = "npz"
# RGMT_REFERENCE_MODE = "neural_retarget"  # "npz", "neural_retarget", or "pico"
RGMT_REFERENCE_MODE = "pico"
PICO_POSE_ENDPOINT = "tcp://127.0.0.1:28704"

# Keep these paths relative to the installed package share directory.  Both
# simulation and hardware launch files use the same dictionaries.
NPZ_FILE_PATHS = {

    #smplx
    # "smplx": "policy/ACCAD/Male2MartialArtsExtended_c3d/Form_1_stageii.npz",
    # "smplx": "policy/ACCAD/amp/walk/pico_stageii.npz",
    "smplx": "policy/ACCAD/amp/walk/0007_Walking001_stageii.npz",
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
    # "rgmt": "policy/cmu_1h_new/75/75_08_stageii.npz",    #360旋转
    "rgmt": "policy/cmu_1h_new/88/88_11_stageii.npz",    #热身
    # "rgmt": "policy/cmu_1h_new/90/90_01_stageii.npz",    #后滚翻
    # "rgmt": "policy/cmu_1h_new/90/90_02_stageii.npz",    #侧手翻
    # "rgmt": "policy/cmu_1h_new/90/90_03_stageii.npz",    #侧手翻
    # "rgmt": "policy/cmu_1h_new/90/90_04_stageii.npz",    #侧手翻
    # "rgmt": "policy/cmu_1h_new/90/90_05_stageii.npz",    #回旋踢
    # "rgmt": "policy/cmu_1h_new/90/90_06_stageii.npz",    #回旋踢
    # "rgmt": "policy/cmu_1h_new/90/90_30_stageii.npz",
    # "rgmt": "policy/cmu_1h_new/15/15_01_stageii.npz",
    # "rgmt": "policy/cmu_1h_new/15/15_01_nur.npz",

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
    "rgmt": "policy/rgmtr_160600.onnx",
    
    # "neural_retarget": "policy/neural_retarget.onnx",
    "neural_retarget": "policy/neural_retarget_fast.onnx",
    
    
    # isaaclab3
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
