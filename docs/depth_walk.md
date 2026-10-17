# ELF3 Lite 深度行走仿真

沿用当前 `models`、`model_config.py`、`motionType`、`inference_step` 和
`send_to_motor` 的结构。预处理和输入契约来自本机 `com.bxi.normal_depth`，
相机用户传感器接口来自更新后的 `bxi_ros2_pkg`，不依赖原工程的 Mod 框架、
相机 SDK、RKNN、OpenVINO 或 cv_bridge。

## 启动和按键

必须使用支持深度话题的新 MuJoCo 包，当前机器的 `/opt/bxi/bxi_ros2_pkg` 是旧版本。
在工作区根目录执行：

```bash
source /home/szz/bxi/bxi_ros2_pkg/setup.bash
colcon build --symlink-install --packages-select bxi_example_py_elf3
source install/local_setup.bash
ros2 pkg prefix mujoco  # 应为 /home/szz/bxi/bxi_ros2_pkg
ros2 launch bxi_example_py_elf3 example_dance.launch.py
```

等待原来的关节软启动完成，按 A 释放仿真悬挂并进入 AMP 普通行走。
双脚站稳、深度图就绪后按 B，进入深度行走。摇杆仍给速度命令，不是自动导航。
B 再按一次或按 A 返回普通行走；X 的参考跟踪和 Y 的躺下/起身保持原流程。
从其他动作进入深度行走前先按 A。切换仍使用当前框架的 0.4 秒动作混合；
混合中对退出的深度策略保持最后已提交输出，不额外推进历史或循环隐状态。

深度缺失时拒绝 B 进入；运行中超过 0.5 秒没有新有效图像，则回 AMP 零速度平衡，
清除深度策略状态，需要新图就绪后再按 B。深度模式沿用现有跌倒锁存及 Y 起身流程。
当前仅仿真 launch 启用；硬件 launch 的 B 仍保留原有 AMP 跑步行为。

## 模型与相机绑定

`model_config.py` 默认：

```python
DEPTH_WALK_ENABLED = True
DEPTH_WALK_MODEL = "dagger_new_position"
```

`DEPTH_WALK_PROFILES` 一起选择 ONNX、预处理模式、相机位置和视线俯角。
用户已确认最新模型采用腰部安装、视线向下 **65°**。
外参取自 [dev 分支的躯干相机配置](https://github.com/bxirobotics/bxi_controller_ros2/blob/dev/src/bxi_example_py_elf3/data/mujoco_simulation/sensors/body_depth_camera_camera.xml)，
使用其新位置分支；不是 `head_depth_camera` 的物理头部安装位姿。

| 模型 | 模式 | torso_link 下的位置（m） | MuJoCo wxyz 四元数 | 最终视线向下角度 |
| --- | --- | --- | --- | ---: |
| dagger_new_position | origin_camera | 0.08599, 0.0175, -0.20651 | 0.6903455271, 0.1530459187, -0.1530459187, -0.6903455271 | 65° |
| dagger2 | origin_camera | 0.1152, 0.0175, -0.1358 | 0.6963642386, 0.1227878136, -0.1227878136, -0.6963642386 | 70° |
| normal_depth | depth_walk | 同旧位置 | 同旧位置 | 70° |

MuJoCo 相机沿局部 `-Z` 看；新姿态由 Y 轴 -25° 与相机局部 Z 轴 -90° 合成，
最终单位视线为 `(sin25°, 0, -cos25°)`。不能把 XML 的欧拉角直接设为 -65°。
局部 Z 的旋转用于匹配图像轴，不能仅保留俯角而丢掉该旋转。
回归检查验证这两套完整旋转矩阵与位置，深度 XML 的质量、惯量、关节、
碰撞几何及驱动器参数与原 Lite 模型一致。模型名称仍为 `elf3`，匹配原重置服务。

XML 参数统一放在 `model_config.py`：

- `DEPTH_CAMERA_PARENT/POSE`：安装父节点、位置、视线向下角度和相机局部轴旋转。
- `DEPTH_CAMERA_STREAM`：颜色／深度分辨率、两路竖直 FOV、FPS、距离范围、深度和预览开关。
- `SIMULATION_MODEL_XML`：基础机器人 XML，默认原 `robot/elf3_lite/xml/elf3.xml`。
- `SIMULATION_TERRAIN_ENABLED`：默认 False；True 添加台阶。
- `SIMULATION_TERRAIN_STEPS`：每个台阶的位置和半尺寸。
- `SIMULATION_TERRAIN_RGBA/FRICTION`：台阶颜色和摩擦。

启动时 `get_simulation_model_file()` 读取基础 XML，注入配置并将资源路径转为绝对路径，
生成文件放在 `/tmp/bxi_elf3_sim_<uid>/`。配置改变后生成新文件名，不覆盖运行中的
场景，不写入安装目录。原 `elf3.xml` 和 UMR 的机器人路径保留，不再维护多个深度 XML。
默认台阶仍为 10/20 cm，入口前 1.6 m 为平地，自动使用当前模型的相机位姿。
这些参数提供地形输入场景，尚未验证台阶通过率。

## 深度数据和预处理

仿真输出：

```text
/simulation/body_depth_camera/depth/image_rect_raw
/simulation/body_depth_camera/depth/camera_info
```

MuJoCo 用户传感器配置请求 30 Hz、`848×480` 深度图，编码 `16UC1`，单位毫米；
颜色预览为 `640×360`，与深度分别渲染。图像和内参使用 sensor-data QoS、队列深度 1。
控制仍为 50 Hz。相机线程实际帧率取决于渲染负载，不等于请求值。

接收端支持 `16UC1`/`mono16` 和米单位 `32FC1`，处理字节序和行填充。
根据 CameraInfo 内参计算中心 FOV 裁剪，最近邻缩放，保持无效值为零；拒绝全无效
图和错误布局，重复非零时间戳不刷新图像年龄。按模型更新周期使用最新图像。

| 模式 | FOV（水平 × 竖直） | 投影大小（宽 × 高） | 策略图 | 深度时间选择 |
| --- | --- | --- | --- | --- |
| origin_camera | 45.2° × 58.0616969° | 36×48 | 36×36，范围 [-0.5, 0.5] | 最新一帧，更新周期 0.05 s |
| depth_walk | 89.24° × 58.06° | 64×36 | 21×32，范围 [0, 1] | 37 帧缓存中取 -36,-31,…,-1 共 8 帧 |

投影后沿用原 Mod 的顺时针旋转，策略侧逆旋转；保留两个操作以对齐原接口。
origin_camera 的 3×3 Gaussian（sigma=1）在归一化前执行，随后裁剪行 `[2:38]`。
距离范围为 `[0.2, 3.0] m`；旧 8 帧模式为 `[0.2, 2.5] m`。
SciPy 的分离卷积与 OpenCV `BORDER_REFLECT_101` 对齐，不新增 OpenCV 依赖。

每个控制帧的 96 维观察按原顺序包含机体角速度、重力方向、速度命令、29 维关节
相对位置、29 维速度和29 维上一动作。十帧按时间从旧到新堆叠为 `[1,960]`。
按原 Isaac 关节顺序和 action scale 推理，再转换回现有电机顺序。
最新循环模型另有 `h_in/h_out=[1,1,256]`，进入／重新进入时清零，只在真实控制帧推进。
origin_camera 命令范围为 vx `[0,0.8]`、vy `[-0.5,0.5]`、yaw `[-1.57,1.57]`；
当前模型不接收负向前进速度命令。

## 验证

```bash
PYTHONPATH=src/bxi_example_py_elf3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m pytest -q src/bxi_example_py_elf3/tests src/bxi_example_py_elf3/test \
  src/bxi_example_py_elf3/third_party/umr/tests
```

50 项检查通过。三个模型各 45 帧的输入／输出与未修改的原 Mod 对照，
目标关节最大差约 `4.3e-6 rad`；参考快照保存于 `tests/depth_reference.npz`。
包含深度单位、字节序、ROI、旋转、无效图、重复图、超时、关节排列、循环预览／重置、
原机器人动力学参数、两套相机位姿、配置修改生成新场景、关闭深度和源文件不变的检查。

实际 ROS 仿真反馈检查可运行：

```bash
source /home/szz/bxi/bxi_ros2_pkg/setup.bash
source install/local_setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 src/bxi_example_py_elf3/tools/check_depth_sim.py --domain 86
```

该工具在独立 ROS domain 内启动 MuJoCo 和当前控制类，使用真实深度、关节、IMU
及电机话题，模拟 A/B 按键，确认初始化／释放悬挂成功，验证 B 进入／退出、A 返回、
断流回退和过期图拒绝进入。随后回收两个节点；结果默认在 `/tmp/bxi_depth_check`。

本机最新 65° 循环模型的平地反馈检查中，vx=0.3 m/s 的短段产生约 1.09 m 前向位移，
未触发跌倒锁存；循环隐状态正常推进。策略推理 P99 约 7.85 ms。实际相机约
20–23 Hz，满足当前单帧深度的 20 Hz 更新周期附近需求，但没有证明任意负载下稳定
30 Hz。这里只验证短段平地与部署闭环，不代表台阶性能或实机效果。
