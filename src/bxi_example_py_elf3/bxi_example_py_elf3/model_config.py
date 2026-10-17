"""Shared model file configuration for simulation and hardware launches."""

from pathlib import Path
import os
import sys

PACKAGE_NAME = "bxi_example_py_elf3"

# Change this one value to select the reference source used by the X key.
# ``npz`` uses the existing retargeted RGMT motion; ``neural_retarget`` uses
# the SMPL-X/ACCAD input and the Transformer adapter before RGMT.
# ``umr`` asynchronously retargets raw SMPL/SMPL-X NPZ or LAFAN1 BVH while
# playing, using a bounded lookahead buffer and the existing deployment libraries.
# ``pico_umr`` uses live PICO SMPL22 poses -> UMR -> RGMT with robot feedback.
RGMT_REFERENCE_MODE = "npz"
# RGMT_REFERENCE_MODE = "neural_retarget"  # "npz", "neural_retarget", or "pico"
# RGMT_REFERENCE_MODE = "pico"
# RGMT_REFERENCE_MODE = "pico_umr"  # "npz", "neural_retarget", "pico", "umr", "pico_umr"
PICO_POSE_ENDPOINT = "tcp://127.0.0.1:28704"
# 80 ms 衰减时间常数的限幅预测，无需等待未来帧；设为 0 可对照保持模式。
PICO_UMR_PREDICTION_TIME_S = float(os.environ.get("PICO_UMR_PREDICTION_TIME_S", "0.08"))

# 仿真／真机 B 键：深度行走；A 键回普通 AMP，B 再按一次也可退出。
DEPTH_WALK_ENABLED = True
# DEPTH_WALK_ENABLED = False
DEPTH_WALK_MODEL = "g1_25"  # origin_camera；相机位姿和输入约定一起选择
# 模型、图像契约与位姿一起选择。相机沿局部-Z看，生成器将视线俯角转换为四元数。
DEPTH_WALK_PROFILES = {
    "dagger2": {
        "mode": "origin_camera", "position": (0.1152, 0.0175, -0.1358), "look_down_deg": 70.0,
    },
    "normal_depth": {
        "mode": "depth_walk", "position": (0.1152, 0.0175, -0.1358), "look_down_deg": 70.0,
    },
    "dagger_new_position": {
        "mode": "origin_camera", "position": (0.08599, 0.0175, -0.20651), "look_down_deg": 65.0,
    },
    "g1_25": {
        "mode": "origin_camera", "position": (0.08599, 0.0175, -0.20651), "look_down_deg": 65.0,
        "preprocess": {"crop_rows": (12, 48), "depth_range": (0.2, 2.0)},
    },
}
DEPTH_WALK_MODE = DEPTH_WALK_PROFILES[DEPTH_WALK_MODEL]["mode"]
DEPTH_WALK_PREPROCESS = DEPTH_WALK_PROFILES[DEPTH_WALK_MODEL].get("preprocess", {})
DEPTH_CAMERA_NAME = "body_depth_camera"
DEPTH_CAMERA_PARENT = "torso_link"
DEPTH_CAMERA_POSE = {
    "position": DEPTH_WALK_PROFILES[DEPTH_WALK_MODEL]["position"],
    "look_down_deg": DEPTH_WALK_PROFILES[DEPTH_WALK_MODEL]["look_down_deg"],
    "roll_deg": -90.0,  # 相机局部Z旋转，保持原策略的图像方向
}
DEPTH_CAMERA_STREAM = {
    "color_width": 640, "color_height": 360, "color_fov_y_deg": 43.173360,
    "depth_width": 848, "depth_height": 480, "depth_fov_y_deg": 58.605976,
    "fps": 30.0, "depth_enabled": True,
    "min_distance_m": 0.1, "max_distance_m": 10.0, "preview_enabled": True,
}
DEPTH_TIMEOUT_S = 0.5

# 真机 launch 自动启动独立 D435i 节点；仅相机 SDK 放入 third_party，推理库使用已有环境。
# 深度沿用上面的采集和发布 FPS；保留真实内参，策略按 20Hz 选择深度输入。
# 多台 D435i 时必须填写 serial_no；单台留空自动选择，不绑定开发机序列号。
DEPTH_CAMERA_HARDWARE = {
    "serial_no": "", "color_enabled": False,
    # FOV小图 + Instinct NS补洞，只修复无效深度点。
    "invalid_below_m": 0.2, "inpaint_radius_px": 3.0,
    "reconnect_s": 2.0, "frame_timeout_s": 1.0,
}

# 仿真XML以原Lite模型为基础，启动时注入上面的相机和下面的场景参数。
SIMULATION_MODEL_XML = "robot/elf3_lite/xml/elf3.xml"
SIMULATION_TERRAIN_ENABLED = False  # True：添加下列10/20 cm测试台阶
SIMULATION_TERRAIN_STEPS = [
    {"name": "depth_step_1", "position": (2.0, 0.0, 0.05), "size": (0.4, 0.8, 0.05)},
    {"name": "depth_step_2", "position": (2.8, 0.0, 0.10), "size": (0.4, 0.8, 0.10)},
    {"name": "depth_step_3", "position": (3.6, 0.0, 0.05), "size": (0.4, 0.8, 0.05)},
]
SIMULATION_TERRAIN_RGBA = (0.6, 0.55, 0.45, 1.0)
SIMULATION_TERRAIN_FRICTION = (0.8, 0.01, 0.001)

# PICO runtime is optional.  Set RGMT_REFERENCE_MODE to "pico" or "pico_umr" to enable
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
ONNX_FILE_PATHS = {
    "depth_walk": f"policy/depth_walk/{DEPTH_WALK_MODEL}.onnx",
    "amp_walk": "policy/amp_dwaq3.onnx",##symmetry
    # "amp_run": "policy/myrun6.onnx",##sim 5.5=6
    "amp_run": "policy/myrun10.onnx",##hw 5=5.18
    # "amp_run": "policy/myrun14.onnx",#run_dwaq

    # "rgmt": "policy/rgmtr_125200.onnx",
    # "rgmt": "policy/rgmtr_53600.onnx",
    "rgmt": "policy/rgmtr_80000.onnx",

    
    "neural_retarget": "policy/neural_retarget_fast.onnx",
    
    # isaaclabv2
    "lie_down": "policy/dance_isaaclab/lie_down.onnx",
    "getup_face": "policy/dance_isaaclab/getup_face.onnx",
    "getup_back": "policy/dance_isaaclab/getup_back.onnx",
    
    # "cxk": "policy/dance_isaaclab/cxk_2k.onnx",
    "cxk": "policy/dance_isaaclab/cxk2_2w.onnx",
    # "cxk": "policy/dance_isaaclab/cxk2_3k.onnx",
}

NPZ_FILE_PATHS = {

    # UMR 原始动作输入：选一个 NPZ/BVH 文件，无需预先转换。
    "umr": "policy/lafan1_orin/dance1_subject2.bvh",
    # "umr": "policy/cmu_orin/88/88_06_poses.npz",  # CMU 旋转踢
    "umr_robot": "robot/elf3_lite/xml/elf3.xml",
    "umr_runtime": "third_party/umr",

    # 官方 SMPL-X 紧凑模型：用于 neural_retarget 骨架及 UMR NPZ 的人体形状。
    "smplx_model": "policy/body_models/smplx/SMPLX_NEUTRAL_compact.npz",

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
    "rgmt": "policy/lafan1_npz/fightAndSports1_subject1.npz",
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
    # "rgmt": "policy/cmu_1h_new/75/75_03_stageii.npz",    #跳
    # "rgmt": "policy/cmu_1h_new/75/75_15_stageii.npz",    #跳
    # "rgmt": "policy/cmu_1h_new/80/80_15_stageii.npz",    #操作
    
    
    # "rgmt": "policy/cmu_1h_new/85/85_01_stageii.npz",    #升龙踢腿
    # "rgmt": "policy/cmu_1h_new/85/85_03_stageii.npz",    #街舞#
    # "rgmt": "policy/cmu_1h_new/85/85_04_stageii.npz",    #街舞
    # "rgmt": "policy/cmu_1h_new/85/85_05_stageii.npz",    #街舞-倒立
    # "rgmt": "policy/cmu_1h_new/85/85_06_stageii.npz",    #街舞-韦伯斯特
    # "rgmt": "policy/cmu_1h_new/85/85_07_stageii.npz",    #街舞-后脚翻
    # "rgmt": "policy/cmu_1h_new/85/85_08_stageii.npz",    #街舞-托马斯
    # "rgmt": "policy/cmu_1h_new/85/85_09_stageii.npz",    #马步
    # "rgmt": "policy/cmu_1h_new/85/85_10_stageii.npz",    #bad?
    # "rgmt": "policy/cmu_1h_new/85/85_11_stageii.npz",    #街舞#
    # "rgmt": "policy/cmu_1h_new/85/85_12_stageii.npz",    #街舞-hard
    # "rgmt": "policy/cmu_1h_new/85/85_13_stageii.npz",    #街舞-倒立x2
    # "rgmt": "policy/cmu_1h_new/85/85_14_stageii.npz",    #街舞-后手翻
    
    # "rgmt": "policy/cmu_1h_new/87/87_01_stageii.npz",    #慢速？
    
    # "rgmt": "policy/cmu_1h_new/88/88_03_stageii.npz",    #马步
    # "rgmt": "policy/cmu_1h_new/88/88_05_stageii.npz",    #bad
    # "rgmt": "policy/cmu_1h_new/88/88_06_stageii.npz",    #旋转踢#
    # "rgmt": "policy/cmu_1h_new/88/88_07_stageii.npz",    #侧倒立
    # "rgmt": "policy/cmu_1h_new/88/88_08_stageii.npz",    #后手翻
    # "rgmt": "policy/cmu_1h_new/88/88_09_stageii.npz",    #后手翻
    # "rgmt": "policy/cmu_1h_new/88/88_10_stageii.npz",    #热身
    # "rgmt": "policy/cmu_1h_new/88/88_11_stageii.npz",    #热身-hard
    
    # "rgmt": "policy/cmu_1h_new/90/90_01_stageii.npz",    #后滚翻
    # "rgmt": "policy/cmu_1h_new/90/90_02_stageii.npz",    #侧手翻#
    # "rgmt": "policy/cmu_1h_new/90/90_03_stageii.npz",    #侧手翻#
    # "rgmt": "policy/cmu_1h_new/90/90_04_stageii.npz",    #侧手翻#
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
    # "rgmt": "policy/cmu_1h_new/90/90_30_stageii.npz",   #俄罗斯舞#
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
    
    # "cxk": "policy/dance_isaaclab/cxk.npz",
    "cxk": "policy/dance_isaaclab/cxk2.npz",
}


def get_simulation_model_file(*, package_share_dir=None, output_dir=None):
    """Apply the camera and scene settings above to the original robot MJCF."""
    from .utils.simulation_xml import generate_simulation_xml
    if package_share_dir is None:
        from ament_index_python.packages import get_package_share_path
        package_share_dir = get_package_share_path(PACKAGE_NAME)
    share_dir = Path(package_share_dir)
    camera = None
    if DEPTH_WALK_ENABLED:
        camera = dict(name=DEPTH_CAMERA_NAME, parent=DEPTH_CAMERA_PARENT,
                      stream=DEPTH_CAMERA_STREAM, **DEPTH_CAMERA_POSE)
    return generate_simulation_xml(share_dir / SIMULATION_MODEL_XML, camera=camera,
        terrain_steps=SIMULATION_TERRAIN_STEPS if SIMULATION_TERRAIN_ENABLED else (),
        terrain_rgba=SIMULATION_TERRAIN_RGBA, terrain_friction=SIMULATION_TERRAIN_FRICTION,
        output_dir=output_dir)


def get_model_file_dicts():
    """Return absolute model paths for the installed package resources."""
    from ament_index_python.packages import get_package_share_path
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
