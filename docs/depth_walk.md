# ELF3 Lite 深度行走：仿真与 D435i 真机

沿用当前 `models`、`model_config.py`、`motionType`、`inference_step` 和
`send_to_motor` 的结构。预处理和输入契约来自本机 `com.bxi.normal_depth`，
仿真相机用户传感器接口来自更新后的 `bxi_ros2_pkg`。真机使用独立的 D435i
Python 节点和包内 SDK，不引入原工程的 Mod 框架、RKNN、OpenVINO 或 cv_bridge。

## D435i 真机部署

`model_config.py` 的 `DEPTH_WALK_ENABLED=True` 同时作用于仿真和真机。
真机 launch 自动启动 `d435i_camera`，控制节点自动订阅
`/hardware/body_depth_camera/depth/image_rect_raw` 和对应的 `camera_info`。
设为 False 时不启动相机，控制节点保留原有非深度 B 键行为。

仅相机 SDK 放在 `src/bxi_example_py_elf3/third_party/realsense/python`，保留官方
pyrealsense2 的 USB 相机核心、元数据和许可证，约 **24 MB**。未使用的 DDS 和
独立工具扩展已移除。原来的 `third_party/depth_walk` 通用依赖目录已删除。

NumPy、SciPy、ONNX、ONNX Runtime 及其传递依赖直接使用现有 ROS Python 环境，
不再复制到项目、不通过控制节点修改这些库的导入路径。`bxi_ros2_pkg` 和 ROS 包
沿用 `/opt/bxi/bxi_ros2_pkg/setup.bash` 激活的环境；不复制 ROS 或通信包。

当前包适用于 **Ubuntu 22.04 / x86_64 / Python 3.10**，沿用现有 ROS 2 与
`bxi_ros2_pkg` 环境。不再执行 pip、apt，不依赖系统或用户安装的相机 SDK、
`realsense2_camera` 或 PyTorch，不安装额外 UVC 驱动、DKMS 内核补丁。
默认相机补洞复用当前环境已有 OpenCV，不额外打包。
Linux 基础库及设备权限仍属于操作系统。沿用原项目的 root 真机启动方式，
不需要安装相机 udev 规则；普通用户只使用已被系统允许的视频接口。
相机 SDK 的固定版本、来源、SHA256 和许可证见 `third_party/realsense` 的说明与 manifest。

先构建一次，把节点、launch 和 third_party 资源部署到 ROS 安装空间：

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
colcon build --packages-select bxi_example_py_elf3
source install/local_setup.bash
```

先仅启动相机，不启动硬件控制和电机：

```bash
ros2 launch bxi_example_py_elf3 depth_camera.launch.py
```

相机被 SDK 独占，测试前关闭 `realsense-viewer` 和其他相机进程。
当前 `DEPTH_CAMERA_STREAM['fps']=30`，仿真和真机均按 30 FPS 采集／发布。
D435i 在 848×480 下原生支持 6/15/30/60/90 FPS，不能直接请求 20 FPS。
当前 origin_camera 模型的深度输入按 0.05 秒固定时间表选择最新帧，目标平均 20 Hz。
控制仍为 50 Hz，因此相邻深度更新在 2／3 个控制周期之间交替，即约 40／60 ms。
取帧时间表不从每次控制回调重新计时，避免频率下降到 16.7 Hz；发生延迟后跳过
错过的时间点，不积攒或重放旧帧。
Instinct 输出 `36x48` 小图及相应内参。
如需彩色话题，在 `DEPTH_CAMERA_HARDWARE`
中设 `color_enabled=True`，沿用 `DEPTH_CAMERA_STREAM` 的彩色尺寸。
单台 D435i 留空 `serial_no`，多台时指定序列号。USB 重插后自动重连。
相机自身的 IMU 不用于控制；姿态和角速度继续使用机器人的原有 IMU 话题。

按项目原有流程进入 root shell、激活 ROS 和工作区，再启动真机：

```bash
sudo -s
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/local_setup.bash
ros2 launch bxi_example_py_elf3 example_dance_hw.launch.py
```

完成原来的软启动后，A 进入普通 AMP；有效深度就绪后 B 进入深度行走，
B 再按或 A 退出。相机缺失、图像无效或超时超过 0.5 秒时，拒绝进入或回到
AMP 零速度平衡。断流后持续忽略摇杆速度，直到显式按 A 恢复普通行走，
或新图就绪后按 B 重新进入深度行走。原有跌倒锁存、Y 起身与动作混合保持原流程。

当前默认策略对应腰部位置 `(0.08599, 0.0175, -0.20651)` m、视线向下 **65°**。
真机相机必须按选定模型的位姿和图像方向安装；内参裁剪无法补偿错误的实际安装位姿。
此策略仍由摇杆给速度命令，不是自动导航。

## 启动和按键

必须使用支持深度话题的 MuJoCo 包。当前机器的 `/opt/bxi/bxi_ros2_pkg` 已包含
`share/mujoco/CAMERA_SENSOR.md`，可直接使用；无需 source 开发目录下的另一份包。
在工作区根目录执行：

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
colcon build --packages-select bxi_example_py_elf3
source install/local_setup.bash
ros2 pkg prefix mujoco  # 应为 /opt/bxi/bxi_ros2_pkg
ros2 launch bxi_example_py_elf3 example_dance.launch.py
```

等待原来的关节软启动完成，按 A 释放仿真悬挂并进入 AMP 普通行走。
双脚站稳、深度图就绪后按 B，进入深度行走。摇杆仍给速度命令，不是自动导航。
B 再按一次或按 A 返回普通行走；X 的参考跟踪和 Y 的躺下/起身保持原流程。
从其他动作进入深度行走前先按 A。切换仍使用当前框架的 0.4 秒动作混合；
混合中对退出的深度策略保持最后已提交输出，不额外推进历史或循环隐状态。

深度缺失时拒绝 B 进入；运行中超过 0.5 秒没有新有效图像，则回 AMP 零速度平衡，
清除深度策略状态，需要新图就绪后再按 B。深度模式沿用现有跌倒锁存及 Y 起身流程。
仿真与硬件 launch 均传入同一组深度开关、模式、逻辑相机名和超时参数。

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

仿真输出（真机只把前缀改为 `/hardware`）：

```text
/simulation/body_depth_camera/depth/image_rect_raw
/simulation/body_depth_camera/depth/camera_info
```

MuJoCo 用户传感器按 `DEPTH_CAMERA_STREAM['fps']` 请求帧率（当前 30 Hz），
输出 `848×480` 深度图，编码 `16UC1`，单位毫米；
颜色预览为 `640×360`，与深度分别渲染。图像和内参使用 sensor-data QoS、队列深度 1。
控制仍为 50 Hz。相机线程实际帧率取决于渲染负载，不等于请求值。

真机相机端按真实内参
进行 FOV 中心裁剪和最近邻缩放，再用 OpenCV Navier–Stokes 补洞：小于 0.2 m
为无效点，半径 3 px。该顺序和补洞函数对照用户提供的
`realsense_depth_instinct_ros2_ws.tar.gz`，仅使用这条 Instinct NS 补洞流程。
相机保持原采集分辨率、帧率及激光设置；只改变软件处理，当前输出宽×高为 36×48。

Instinct 小图同步调整 CameraInfo 的焦距、主点和投影矩阵；接收端按该内参选择
完整小图，不做第二次裁剪或缩放。模型文件、接收／投影／归一化代码均不修改。
相机端不额外旋转，不对齐到彩色，也不执行模型的高斯或距离归一化。
相机 depth scale 自动转换为毫米 `16UC1`，保留无效值 0，避免自定义设备单位
导致距离比例错误。图像和 CameraInfo 每帧共用时间戳及光学坐标系名。
发布采用 ROS 原生字节数组批量拷贝，不逐像素经过 Python 类型检查。

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
策略仍使用 SciPy 的分离卷积，与 OpenCV `BORDER_REFLECT_101` 对齐；相机 NS
补洞复用已有 OpenCV。

每个控制帧的 96 维观察按原顺序包含机体角速度、重力方向、速度命令、29 维关节
相对位置、29 维速度和29 维上一动作。十帧按时间从旧到新堆叠为 `[1,960]`。
按原 Isaac 关节顺序和 action scale 推理，再转换回现有电机顺序。
最新循环模型另有 `h_in/h_out=[1,1,256]`，进入／重新进入时清零，只在真实控制帧推进。
origin_camera 命令范围为 vx `[0,0.8]`、vy `[-0.5,0.5]`、yaw `[-1.57,1.57]`；
当前模型不接收负向前进速度命令。

## 验证

先激活 ROS，保留其 PYTHONPATH。相机、策略和原控制适配测试：

```bash
PYTHONPATH=src/bxi_example_py_elf3:$PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m pytest -q src/bxi_example_py_elf3/tests src/bxi_example_py_elf3/test
```

相机到模型的独立实机检查（只启动相机，不启动机器人硬件节点、初始化服务或电机）：

```bash
source install/local_setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 src/bxi_example_py_elf3/tools/check_depth_camera.py --domain 88
```

采集同一个 D435i 源帧的原始深度、NS 补洞后深度和模型实际输入：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 src/bxi_example_py_elf3/tools/capture_depth_stages.py
```

先用 60 帧预热设备，再保存一组对照；NS 补洞不依赖时域历史。
原始图取自相机软件滤波之前，滤波后图保存为 36×48 小图。
默认目录为 `/tmp/bxi_d435i_capture/<时间>`，可通过 `--output` 指定。
`overview.png` 为三图对比，`*_vis.png` 为伪彩预览，`*_mm.png` 为无损 16 位毫米图，
`model_input.npy` 为真正送入模型的 float32 张量；`depth_stages.npz` 包含完整深度、
投影结果、内参和模型输入。预览颜色只显示 0.2–3 m，不截断保存的原始数值。
采集工具复用当前驱动滤波和策略预处理函数，不启动机器人节点；绘图使用电脑
现有的 Matplotlib/Pillow，仅用于诊断，不向 third_party 增加依赖。

该工具通过相机专用 launch 启动内置 SDK，在独立 ROS domain 接收图像与内参，
以 50 Hz 调用当前深度 ONNX，检查输入形状、有限关节目标、循环状态、实际帧率
和断流后图像过期；最后回收相机进程。原始毫米图、策略深度输入、推理目标和
报告默认写入 `/tmp/bxi_d435i_check`。关节和机身观察在该工具中是合成静止姿态，
因此它验证采图及推理链路，不代表机器人实际行走或台阶通过率。

检查报告的 `runtime_paths` 同时记录 NumPy、ONNX Runtime 和相机 SDK 的实际位置。
精简后，前两者应来自当前 Python 环境，只有 SDK 来自包内 `third_party/realsense`。
工具保留当前环境正常的 site-packages 搜索路径，包括已有的用户安装包；不使用
`python -s` 或 `PYTHONNOUSERSITE` 隔离通用推理依赖。停止 launch 后检查相机正常
退出及接收缓存超时失效。实测指标以当前报告为准，不保证任意 CPU 竞争下固定帧率。

上述 37 项检查通过。三个模型各 45 帧的输入／输出与未修改的原 Mod 对照，
目标关节最大差约 `4.3e-6 rad`；参考快照保存于 `tests/depth_reference.npz`。
包含深度单位、字节序、ROI、旋转、无效图、重复图、超时、关节排列、循环预览／重置、
原机器人动力学参数、两套相机位姿、配置修改生成新场景、关闭深度和源文件不变的检查。
新增检查覆盖真机启动开关、深度单位变更、真实内参、相机选择、断流后的速度锁存，
以及三个深度模型复用已有数值／推理库、仅相机 SDK 使用包内版本时完成推理。

NS 补洞已用 60 帧对照压缩包中的 `depth_instinct_postprocess.py`，量化输出全部
逐像素一致。该批画面的 ROI／缩放／补洞／校准耗时中位数约 0.58 ms，原始有效点
数值保持不变。这不代表测距准确率；实际指标随场景和机器负载变化。

额外执行 `third_party/umr/tests` 时，当前基础 XML 的 `scene_bx2.xml` include
会触发现有 UMR 加载器的 standalone MJCF 限制，11 项失败；此 include 和 UMR
加载器均已存在于本次修改前的 HEAD，本次未修改它们。深度策略和仿真不使用该加载器。

实际 ROS 仿真反馈检查可运行：

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/local_setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 src/bxi_example_py_elf3/tools/check_depth_sim.py --domain 86
```

该工具在独立 ROS domain 内启动 MuJoCo 和当前控制类，使用真实深度、关节、IMU
及电机话题，模拟 A/B 按键，确认初始化／释放悬挂成功，验证 B 进入／退出、A 返回、
断流回退和过期图拒绝进入。随后回收两个节点；结果默认在 `/tmp/bxi_depth_check`。

也可以用电脑上的真实 D435i 深度图和 MuJoCo 机器人反馈测试同一个策略：

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/local_setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 src/bxi_example_py_elf3/tools/check_depth_sim.py \
  --depth-source d435i --domain 89 --output /tmp/bxi_d435i_mujoco_check
```

该模式不添加仿真深度传感器，独立 D435i 节点是唯一深度来源，发布到独立 domain
内的 `/simulation/body_depth_camera/depth/...`。机器人关节、IMU、控制指令和
初始化服务全部留在 MuJoCo，不启动硬件节点。工具还在不推进模型状态的条件下，
对照真实图、0.3 m 近平面和 3 m 远平面，确认关节目标对深度输入有响应。
输入快照、策略目标和根节点轨迹保存为 NPZ，指标保存为 `ros_report.json`。

本机短段测试中，真实深度 ROS 接收约 20.0 Hz，推理 P99 约 7.82 ms；机器人
前向位移约 0.65 m、侧向位移约 0.28 m，未触发跌倒锁存，B/A 切换与断流回退通过。
相同机身观察和隐状态下，近／远平面关节目标差的 L2 范数约 0.313 rad。
真实房间的图像与虚拟平地未做空间对应，也未随机器人移动；这验证感知输入到
仿真控制的链路及策略响应，不能据此评价地形通过率或实际安装位姿下的行走质量。

本机最新 65° 循环模型的平地反馈检查中，vx=0.3 m/s 的短段产生约 1.09 m 前向位移，
未触发跌倒锁存；循环隐状态正常推进。策略推理 P99 约 7.85 ms。实际相机约
20–23 Hz，满足当前单帧深度的 20 Hz 更新周期附近需求，但没有证明任意负载下稳定
30 Hz。这里只验证短段平地与部署闭环，不代表台阶性能或实机效果。
