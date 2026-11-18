# 深度行走模型

`3624.onnx` 和 `g1_25.onnx` 由用户新增，当前默认选择 `3624`；其余模型来自本机
`/home/szz/bxi/com.bxi.normal_depth/assets`。
`3624` 的 ONNX 深度输入为 `[batch,24,36]`，按配置裁剪行 `[24:48]`，
深度范围为 **0.2–2.0 m**；使用相同的高斯滤波及 [-0.5,0.5] 归一化，不做缩放。
适配器根据裁剪行数确定输出高度，并与 ONNX 输入尺寸核对。
`g1_25` 的深度范围和裁剪由用户明确为 **0.2–2.0 m、行 `[12:48]`**，
仍使用 3×3 Gaussian（sigma=1）及归一化到 [-0.5,0.5]，最终张量 `[1,36,36]`。
相机端按 `realsense_depth_instinct_ros2_ws.tar.gz` 的 FOV／最近邻／NS 补洞流程处理。

`g1_25` 接收 `obs_history=[batch,960]`、`depth_data=[batch,36,36]`，
输出 `actions=[batch,29]`；没有 `h_in/h_out`，不需要维护外部循环状态。
ONNX 没有导出关节、归一化或相机外参元数据。相机位姿沿用用户在配置中选择的
腰部 65°；机器人默认姿态、关节映射和 action scale 沿用现有 ELF3 适配器。
旧模型位姿取自 `bxi_controller_ros2/dev` 的
`data/mujoco_simulation/sensors/body_depth_camera_camera.xml`：新位置分支。

| 模型 | 模式 | 相机相对 torso_link 的位置（m） | 视线向下角度 | 隐状态 |
| --- | --- | --- | ---: | --- |
| 3624.onnx | origin_camera，行24:48 / 0.2–2.0m，输入24×36 | 0.08599, 0.0175, -0.20651 | 65°（配置值） | 无 |
| g1_25.onnx | origin_camera，行12:48 / 0.2–2.0m | 0.08599, 0.0175, -0.20651 | 65°（配置值） | 无 |
| dagger_new_position.onnx | origin_camera | 0.08599, 0.0175, -0.20651 | 65° | h_in/h_out [1,1,256] |
| dagger2.onnx | origin_camera | 0.1152, 0.0175, -0.1358 | 70° | 无 |
| normal_depth.onnx | depth_walk | 0.1152, 0.0175, -0.1358 | 70° | 无 |

修改 `model_config.py` 的 `DEPTH_WALK_MODEL` 时，同时切换模式和生成场景所用的相机位置／俯角。相机输出和台阶参数同样在该文件设置。
不要将头部相机或 viewer tracking 相机的话题直接作为这些模型的输入。

原文件 SHA256：

```text
99906ed4fc1a9179eb5cf93766ee9f688ed19ecd2141843228404239c95e4a14  dagger2.onnx
21031ed16330e59bb3cc6c7ecdafe8235987c94df4c18c06b91091592f00d3ab  dagger_new_position.onnx
b57fccab09485d7d5dfb330da921f00b421fcf1283fa00e179a77323cac75ca5  g1_25.onnx
```

预处理和关节契约说明见工作区 `docs/depth_walk.md`。
