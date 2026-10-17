# 深度行走模型

模型直接复制自本机 `/home/szz/bxi/com.bxi.normal_depth/assets`。
默认使用用户确认的最新腰部相机 65° 配置和 `dagger_new_position.onnx`。
模型没有导出相机外参元数据，位姿取自 `bxi_controller_ros2/dev` 的
`data/mujoco_simulation/sensors/body_depth_camera_camera.xml`：新位置分支。

| 模型 | 模式 | 相机相对 torso_link 的位置（m） | 视线向下角度 | 隐状态 |
| --- | --- | --- | ---: | --- |
| dagger_new_position.onnx | origin_camera | 0.08599, 0.0175, -0.20651 | 65° | h_in/h_out [1,1,256] |
| dagger2.onnx | origin_camera | 0.1152, 0.0175, -0.1358 | 70° | 无 |
| normal_depth.onnx | depth_walk | 0.1152, 0.0175, -0.1358 | 70° | 无 |

修改 `model_config.py` 的 `DEPTH_WALK_MODEL` 时，同时切换模式和生成场景所用的相机位置／俯角。相机输出和台阶参数同样在该文件设置。
不要将头部相机或 viewer tracking 相机的话题直接作为这些模型的输入。

原文件 SHA256：

```text
99906ed4fc1a9179eb5cf93766ee9f688ed19ecd2141843228404239c95e4a14  dagger2.onnx
21031ed16330e59bb3cc6c7ecdafe8235987c94df4c18c06b91091592f00d3ab  dagger_new_position.onnx
```

预处理和关节契约说明见工作区 `docs/depth_walk.md`。
