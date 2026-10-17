# D435i SDK

`python/pyrealsense2/` 是 Intel RealSense 官方 CPython 3.10 / x86_64
SDK **2.58.1.10581** 的 USB 相机核心，约 **24 MB**，包含 librealsense 原生扩展。
仅保留 `__init__.py`、`pyrealsense2` 原生模块及元数据、许可证；不打包未使用的
`pyrealdds` 网络 DDS 扩展和独立 `pyrsutils` 工具模块。不依赖系统安装的
librealsense、`realsense2_camera` 或 `pyrealsense2`。版本及官方来源校验值见
`manifest.json`，Apache-2.0 许可证保留在 wheel 的 dist-info/licenses 中。

**只内置相机 SDK。** NumPy、SciPy、ONNX、ONNX Runtime 及其依赖均复用当前
ROS Python 环境，不在 third_party 重复保存，不通过节点修改它们的导入路径。
默认 Instinct 补洞使用当前环境已有的 OpenCV，同样不在 third_party 打包。
先 source `/opt/bxi/bxi_ros2_pkg/setup.bash`，再 source 工作区 `install/local_setup.bash`。

此 SDK 使用 Linux 自带的 UVC 支持；在当前 Ubuntu 22.04 / 6.8 内核和 D435i
上采图验证，不安装厂商内核驱动或 DKMS 补丁。USB 与视频设备权限仍由操作系统
控制：本项目原有的 **root 真机启动流程**可以直接访问设备，无需安装 udev 规则。
普通用户需要已有的视频设备访问权限；没有权限时节点打印原因并重试。
本节点只采集行走需要的深度，彩色可配置，不打开相机 IMU；机器人姿态继续来自
原有 `/hardware/imu_data`，因此不会受到普通用户 IIO 权限的影响。

采集节点 `bxi_example_py_elf3/d435i_camera.py` 是本项目的独立 Python 节点，
不引入参考工程的 Mod、状态机或相机管理框架。仅使用 Instinct NS 补洞：
参照用户提供的 `realsense_depth_instinct_ros2_ws.tar.gz`，按模型 FOV
中心裁剪并最近邻缩到 36×48，再对小于 0.2 m 的像素执行半径 3 px 的
OpenCV Navier–Stokes 补洞。全无效图保留全零，不制造深度。

滤波后小图和相应调整后的内参发布到 `/hardware/<DEPTH_CAMERA_NAME>/depth/...`。
同时缩放焦距、修正主点和投影矩阵，现有接收端据此使用完整小图，不再重复裁剪。
相机节点不做策略距离限幅、高斯、旋转或归一化；模型及其原有预处理文件保持不变。
采集和发布均为 848×480 / 30 FPS；当前 origin_camera 模型由接收端按固定时间表
以平均 20 Hz 选择最新深度，控制频率为 50 Hz。preset、发射器和激光功率沿用已有设置。

配置集中在 `model_config.py` 的 `DEPTH_CAMERA_STREAM` 和
`DEPTH_CAMERA_HARDWARE`。没有硬编码开发机的相机序列号；连接多台 D435i 时
必须指定 `serial_no`。断线停止发布并重连，重复源帧不制造新时间戳，
补洞不使用历史深度。无需联网或修改相机固件。

维护者重新生成 SDK 时，在工作区根目录执行（不是部署或运行步骤）：

```bash
python3 -m pip download --no-deps --only-binary=:all: --dest /tmp/bxi_camera_wheels \
  -r src/bxi_example_py_elf3/third_party/realsense/requirements.txt
python3 src/bxi_example_py_elf3/third_party/realsense/tools/bundle_sdk.py \
  --wheel /tmp/bxi_camera_wheels/pyrealsense2-2.58.1.10581-cp310-cp310-manylinux1_x86_64.whl
```

维护工具核对官方 wheel SHA256，仅提取 manifest 指定的运行文件、元数据和许可证，
并记录实际部署文件的 SHA256。SDK 二进制保持官方原样。
