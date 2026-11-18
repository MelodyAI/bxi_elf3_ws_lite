# B 键行走切舞蹈

`DEPTH_WALK_ENABLED=False` 时，先按 A 启动行走，再按 B 进入舞蹈。

切换先由 AMP 将速度指令平滑降到零，等待机身和腿部稳定，再混合接管舞蹈首帧。
接管期间忽略摇杆速度，首帧保持不推进；接管完成后开始正常播放。
舞蹈和行走模型的上一帧动作都使用实际发送的混合目标，避免反馈未执行的动作。
每次进入舞蹈重新初始化观测历史，不再用相同观测连续预推理 20 次。

准备期间再按 B 可取消，A 返回行走，X/Y 仍按原来的功能执行。
进入舞蹈后 B 暂停或继续播放。连续 3 秒未站稳会取消本次进入并输出提示。

## 参数

仿真和真机 launch 都支持以下参数，时间单位为秒：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `dance_stop_time` | `0.3` | 速度指令降为零的时长 |
| `dance_settle_time` | `0.12` | 需要连续保持稳定的时长 |
| `dance_settle_timeout` | `3.0` | 从按 B 起等待稳定的最长时间 |
| `dance_transition_time` | `0.6` | 首帧接管的 smoothstep 混合时长，0 关闭混合 |

稳定条件为机身倾角小于约 8.6°、机身角速度模长小于 0.35 rad/s、
腿部各关节速度绝对值小于 1 rad/s。等待稳定不能只靠指令速度为零判断。

```bash
source bxi_ros_pkg/setup.bash
colcon build --packages-select bxi_example_py_elf3
source install/local_setup.bash
ros2 launch bxi_example_py_elf3 example_dance.launch.py
```

调整示例：

```bash
ros2 launch bxi_example_py_elf3 example_dance.launch.py dance_transition_time:=0.8
```

`/transition_time` 继续控制其他动作切换。B 键的首帧接管单独使用
`dance_transition_time`，避免改变躺下、起身等动作的切换行为。
