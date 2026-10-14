# ELF3 UMR 轻量部署

从本机 UMR 的 ELF3 Lite 适配迁移，沿用已训练的表面对应关系和足部跟随配置
`humanoid_retarget_elf3_lite_contact.json`。部署资源约 2.7 MB：456 个分段采样点、
1168 个相关人体顶点，以及 4096 个机器人表面点。保留表面位置/法线误差、
语义自接触、地面接触锚点、地面约束、膝关节下限和 LQR 平滑。

运行只使用当前部署已有的 NumPy、SciPy、MuJoCo，以及 RGMT 已有的 ONNX Runtime。
不需要激活 UMR 环境、安装 PyTorch/SMPL-X/Clarabel、训练对应关系或访问原 UMR 项目。
人体蒙皮采用 NumPy；约束求解保留 UMR 的目标与约束，使用 SciPy SLSQP 后端。
`pico_umr` 实时模式的完整输入、控制闭环及验证说明见 [LIVE_PICO.md](LIVE_PICO.md)。
本页的 `umr` 配置和文件播放说明仍适用于原始 NPZ/BVH。

速度优化保留全部 456 个采样点和接触约束：每个刚体只计算一次雅可比，再批量转换
为表面点/法线雅可比；运动学计算跳过碰撞、动力学和传感器；小窗口 LQR 的系数按
窗口形状缓存。QP 通过 Cholesky 坐标变换消除 Hessian 的条件数问题，保留原坐标
求解作为失败回退，并复核所有约束。

`elf3_kinematics.xml`（约 32 KB）保存完整机器人编译后的身体、关节与几何坐标系，
不加载 STL、纹理或驱动器，仅供 UMR 求解使用。FK 和雅可比已与完整机器人随机姿态
对照。机器人 XML/网格内容签名不匹配时自动使用完整模型；仿真和实机模型保持原样。
快速模型用于后台进程启动和重播，避免重复加载完整网格。

## 配置和运行

在 `bxi_example_py_elf3/model_config.py` 中设置：

```python
RGMT_REFERENCE_MODE = "umr"

# NPZ_FILE_PATHS 中选一个原始动作文件：
"umr": "policy/lafan1_orin/dance1_subject2.bvh",
# "umr": "policy/cmu_orin/88/88_06_poses.npz",
```

`smplx_model` 沿用 `policy/body_models/smplx/SMPLX_NEUTRAL_compact.npz`，
NPZ 输入按 `gender` 自动选取同目录的 male/female/neutral 紧凑人体模型。
`umr_robot` 和 `umr_runtime` 分别指向包内机器人 MJCF 和本目录。
三者在 `get_model_file_dicts()` 中解析为安装后的绝对路径。

重新构建 `bxi_example_py_elf3` 并重启现有仿真或实机 launch。`umr` 模式使用
`UmrStreamingRGMT`：后台逐帧重定向，边生成边播放。原始数据会解析成身体姿态，
参考轨迹只保留有限的缓冲，不预先求解完整动作，也不读取旧参考缓存。
重定向计算运行在独立 Python 进程，通过有界管道传递参考帧；控制节点中的读取线程
只收帧，不执行人体蒙皮和 QP。子进程在导入数值库前将 BLAS/OpenMP 线程数设为 1，
隔离控制节点与重定向的 GIL 竞争及小矩阵计算线程竞争，无需新增依赖。
后台按 RGMT ONNX 元数据重采样（当前为 50 Hz），按名称转换到 Isaac 关节顺序，
从机器人 FK 提取 torso 姿态和世界角速度。控制线程只构建观察并运行 RGMT。

后台使用 12 个过去帧和 6 个未来帧的小窗口执行原 LQR 平滑，默认最多缓冲 100 个
策略帧。RGMT 需要的 21 帧窗口包含过去 10 帧和未来 10 帧。启动需要准备一小段
参考及额外 25 帧（0.5 秒）的储备；此时保持 AMP 零速度平衡。发生后续缓冲不足时，
等待储备重新补满后才恢复播放，避免每补一帧就立刻播放造成频繁停顿。
缺帧时继续 RGMT 反馈并保持最后
有效姿态并将参考速度归零，暂停动作时间推进。初始参考准备好后，从 AMP 输出
用 0.6 秒过渡进入跟踪。X 键暂停保留当前姿态，从走路模式再次启动时重置
求解状态并重播。节点退出或重播时终止并回收旧求解进程和读取线程。

这是原始文件输入的异步播放路径。在线 PICO 使用独立的 `pico_umr` 模式，
身体节点实测位置、支撑脚接触判断及因果未来窗口见 [LIVE_PICO.md](LIVE_PICO.md)。
求解器速度不作为硬实时保证；控制线程不会等待求解器。
播放期间每 5 秒输出 `UMR rates`：`control` 是实际 ROS 定时回调频率，`motion` 是
动作推进频率，`generation` 是扣除满缓冲等待后的参考生成能力，`buffer` 是缓冲帧数，
`underruns` 是本周期因参考不足而停帧的次数。正常播放应看到 `control/motion` 接近
50 Hz、`underruns=0`；暂停时 `motion=0`。这可用于检查 NUC 上的实际运行状态。
`backend=process` 和 `worker_pid` 用于确认当前部署已启用进程隔离。
如果设备上的参考生成速度长期低于 50 Hz，缓冲储备仍会耗尽；储备只吸收短时抖动。
独立检查工具仍保留离线生成缓存功能，默认缓存目录为
`${XDG_CACHE_HOME:-~/.cache}/bxi_example_py_elf3/umr/`，与异步播放相互独立。

## 原始数据格式

- SMPL/SMPL-X NPZ：支持 `poses` / `pose_aa`，或者 `root_orient` + `pose_body`；
  位移为 `trans` / `trans_orig`，帧率为 `mocap_framerate` / `mocap_frame_rate` / `fps`。
  读取前 22 个身体关节及前 10 个 betas；沿用 UMR 零手指/面部姿态，忽略 DMPL。
  接受 CMU 的 156 维 SMPL-H 姿态，身体部分按同构骨架解释。
- LAFAN1 BVH：直接解析层级、厘米单位和 Y-up 坐标，按声明的 Euler 顺序计算 FK；
  使用从原 UMR 配对数据验证的固定骨骼坐标系转换为 SMPL-X 身体姿态，再执行一致的
  SMPL-X FK、蒙皮和身体姿态修正。该转换保留骨骼扭转，而不仅是骨骼方向。
  所有校准系数已打包；运行时不需要原项目、`lafan_to_smplx` 或额外转换环境。

BVH 使用 ELF3 对应关系训练时的 neutral 身体形状，根节点位移来自 BVH。
不同性别的 NPZ 使用已有紧凑模型的形状和关节位置，复用 neutral 的蒙皮权重、
身体姿态修正和表面拓扑。异步模式使用有限窗口平滑，离线工具使用全序列平滑。
本部署不包含重新训练对应关系和物体交互。

## 不启动 ROS 的检查

从工作区根目录运行，例如检查前 120 帧：

```bash
PYTHONPATH=src/bxi_example_py_elf3/third_party python3 -m umr.runtime \
  src/bxi_example_py_elf3/policy/lafan1_orin/dance1_subject2.bvh \
  --robot-xml src/bxi_example_py_elf3/robot/elf3_lite/xml/elf3.xml \
  --rgmt-onnx src/bxi_example_py_elf3/policy/rgmtr_125200.onnx \
  --body-model src/bxi_example_py_elf3/policy/body_models/smplx/SMPLX_NEUTRAL_compact.npz \
  --max-frames 120
```

删除 `--max-frames 120` 生成全序列。NPZ 输入只需替换第一个路径。
`--cache-dir` 可以指定缓存位置；截取检查与全序列使用不同缓存键。

独立速度测试（不启动 ROS、电机或读取参考缓存）：

```bash
python3 src/bxi_example_py_elf3/third_party/umr/tools/benchmark.py --frames 480
# --frames 0 测整个动作；--source 可以指定原始 NPZ/BVH。
```

## 资源来源

`elf3_surface.npz` 从 UMR 已训练的 ELF3 对应关系导出。`solver.py` 和
`self_contact.py` 提取原项目的推理函数，保留 MIT 许可，见 `LICENSE`。
SMPL-X 系数由本机已有的人体模型导出，适用人体模型原有授权。
仅重新导出资源时，才需在原 UMR 环境执行：

```bash
python tools/export_assets.py /path/to/UMR
```

这不是部署步骤；导出结果已随包提供。
更新机器人模型后，可以在当前部署环境重新导出快速运动学资源：

```bash
python3 tools/export_kinematics.py
```

## 验证记录

当前部署 Python 3.10 环境中，59 个原始 NPZ/BVH 均通过解析和首尾帧数值检查。
完整 `dance1_subject2.bvh` 生成 6574 帧，`88_06_poses.npz` 生成 191 帧，均为
50 Hz；全部帧通过有限值、四元数归一化、关节限位和机器人表面点地面高度检查。
两类输出均完成现有 RGMT ONNX 的推理、末帧保持和重置检查。
Neutral NPZ 蒙皮对照原 UMR 的 PyTorch/SMPL-X 实现，所测帧表面误差小于
`5e-7 m`。
修正后的 BVH 身体旋转与原 UMR 配对输入在全部 3945 帧中逐帧对齐，最大角度误差
约 `1.5e-7 rad`。前 120 帧离线重定向对照原 ELF3 足部跟随结果，关节角平均误差
约 `6.4e-5 rad`。校准资源由 `tools/lafan_calibration.py` 导出，并验证整段动作中
校准坐标系恒定；回归测试保存原转换的姿态和 ELF3 关节样本，防止上肢扭转错误复发。
完整异步舞蹈参考与原 UMR 的关节角平均差约 `0.0041 rad`，95 分位约 `0.0164 rad`，
最大约 `0.145 rad`；有限窗口平滑及 BVH 根节点位移会造成剩余差异。两类输入的
完整流式输出分别为 6574 帧和 191 帧，均通过有限值及参考契约检查。

速度优化前后同机对照（i5-12600KF，480 个原始 LAFAN 帧，输出 799 个参考帧）：

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 持续参考生成能力 | 64.7 Hz | 206.1 Hz |
| UMR 流水线首参考时间 | 1002 ms | 227 ms |
| 参考生成间隔 P95 | 33.1 ms | 10.0 ms |

完整 LAFAN 6574 帧参考生成约 40.9 秒，持续约 161 Hz。完整 CMU 191 帧约 1.95 秒，
持续约 107 Hz。QP 对照原 UMR Clarabel 的采样问题，步长最大差约 `6.8e-7`；
纯批量雅可比/平滑和快速模型优化的参考关节输出与原实现一致。

限制到一个 CPU 核的 50 Hz 异步播放检查运行了 1000 个控制周期，包含暂停；
首次参考约 0.66 秒可用，随后无缓冲不足，控制调用 P99 约 3.2 ms、最大约 3.9 ms，
定时调度迟到 P99 约 1.2 ms，缓冲不超过 100 帧。热重播参考约 0.24 秒可用。
这些指标不包含电机通信，也不代表 NUC 或实机跟踪结果。
新增与既有测试共 21 项通过，包括后台延迟、保持参考、暂停、重播、流式导数及
批量雅可比/快速模型和原模型的对照检查。
以上单进程速度测试不包含 ROS 竞争负载。用户现场日志显示控制仍为 49.9 Hz，
动作推进仅 34.2 Hz、生成 43.3 Hz，5 秒内缺帧 79 次，说明原读取/求解线程共用
解释器的部署方式缺乏负载余量。当前默认已改成独立求解进程，并增加缓冲恢复储备。

进程版在 ROS 多线程执行器中以 50 Hz 调用策略持续 60 秒，额外开启竞争 GIL 的
Python 负载线程，并将父进程 BLAS/OpenMP 环境设置为 16 线程：实测控制约 50.0 Hz，
启动后缺帧 0 次，参考生成能力约 163 Hz；策略调用 P99 约 7.8 ms，最大约 9.4 ms。
首次参考约 1.12 秒可用，包含独立解释器启动、数据读取及储备准备。该检查不包含
真实仿真渲染或电机通信，不能替代 NUC 和实机验证。管道传输与同进程参考数值已对照，
重播会切换并回收旧 PID，退出不残留子进程。新增与既有测试共 24 项通过。
未启动实机控制。

```bash
PYTHONPATH=src/bxi_example_py_elf3 python3 -m pytest -q \
  src/bxi_example_py_elf3/third_party/umr/tests/test_runtime.py \
  src/bxi_example_py_elf3/test
```
