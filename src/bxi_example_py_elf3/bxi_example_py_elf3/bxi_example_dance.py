import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg 
import sensor_msgs.msg
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

import os
import sys
import math
import json
import time
import datetime

# import torch
import numpy as np
from threading import Lock
from collections import deque

from bxi_example_py_elf3.models.rgmt import RgmtExternalReferencePolicy
from bxi_example_py_elf3.models.neural_rgmt import NeuralRetargetRGMT
from bxi_example_py_elf3.models.accad_smplx import AccadSmplxMotion
from bxi_example_py_elf3.models.pico_human_client import PicoHumanPoseClient
from bxi_example_py_elf3.models.zerolab_human_client import ZeroLabHumanPoseClient
from bxi_example_py_elf3.models.beyondmimic import DanceMotionPolicyGravityIsaaclabV3
from bxi_example_py_elf3.models.amp import  HumanoidGaitPolicyLite
from bxi_example_py_elf3.utils.tfs import get_gravity_orientation

robot_name = "elf3"

dof_num = 29

dof_use = 29#26

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    
    "l_hip_y_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_z_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴

    "r_hip_y_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_z_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴

    "l_shoulder_y_joint",   # 左臂_肩关节_y轴
    "l_shoulder_x_joint",   # 左臂_肩关节_x轴
    "l_shoulder_z_joint",   # 左臂_肩关节_z轴
    "l_elbow_y_joint",   # 左臂_肘关节_y轴
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    
    "r_shoulder_y_joint",   # 右臂_肩关节_y轴   
    "r_shoulder_x_joint",   # 右臂_肩关节_x轴
    "r_shoulder_z_joint",   # 右臂_肩关节_z轴
    "r_elbow_y_joint",    # 右臂_肘关节_y轴
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
    )   

class robotState:
    stand = 1
    stand_to_motion = 2
    
    motion = 3
    motion_to_stand = 4
    
    tumble = 5
    tumble_to_stand = 6
    
class motionType:
    amp_walk = 1
    amp_run = 2
    rgmt = 3
    dance_lie_down = 4
    dance_getup_face = 5
    dance_getup_back = 6
    rgmt_neural = 7
     

class BxiExample(Node):
    
    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.load_files()
        
        self.init_pub_sub()
        
        self.init_controller()

        # 机器人状态变量
        self.qpos = np.zeros(dof_num,dtype=np.double)
        self.qvel = np.zeros(dof_num,dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat = np.zeros(4,dtype=np.double)   
        
        # 状态机相关变量
        self.stand_to_motion_counter = None
        self.motion_to_stand_counter = None
        self.tumble_to_stand_counter = None
        self.state = robotState.stand
        
        self.init_models()
        
        self.motion_type = motionType.amp_walk
        
        # 软启动参数
        self.start_frame_pos = self.amp_walk.default_dof_pos
        
        self.soft_start_kps = self.amp_walk.kps 
        self.soft_start_kds = self.amp_walk.kds
        
        # 定时器回调
        self.step = 0
        self.loop_count = 0
        self.dt = 0.02  # loop 模型时间1/dt=50Hz
        self.dance_flag = 1
        
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)
        
    def init_models(self):
        
        # AMP模型
        self.amp_run = HumanoidGaitPolicyLite(self.onnx_file_dict["amp_run"])
        self.amp_walk = HumanoidGaitPolicyLite(self.onnx_file_dict["amp_walk"])
        
        self.rgmt = None
        self.neural_rgmt = None
        self.pico_pose_client = None
        self._pico_waiting_logged = False
        self._pico_active_logged = False
        self._pico_tracking_active = False
        self._pico_blend_step = 0
        self._pico_wait_target = None
        self.smplx_motion = None
        self.smplx_frame = 19
        if self.rgmt_reference_mode == 'npz':
            self.rgmt = RgmtExternalReferencePolicy(
                self.npz_file_dict["rgmt"],
                self.onnx_file_dict["rgmt"],
                reference_yaw_mode="initial",  # 实机推荐
                # reference_yaw_mode="continuous",
            )
        elif self.rgmt_reference_mode == 'neural_retarget':
            self.neural_rgmt = NeuralRetargetRGMT(
                self.onnx_file_dict["neural_retarget"],
                self.onnx_file_dict["rgmt"],
                reference_yaw_mode="initial",
            )
            self.smplx_motion = AccadSmplxMotion.from_npz(
                self.npz_file_dict["smplx"],
                body_model_path=self.npz_file_dict.get("smplx_model"),
            )
            print(
                "RGMT reference mode: neural_retarget, "
                f"SMPL-X frames={len(self.smplx_motion.body_positions_w)}"
            )
        else:
            self.neural_rgmt = NeuralRetargetRGMT(
                self.onnx_file_dict["neural_retarget"],
                self.onnx_file_dict["rgmt"],
                reference_yaw_mode="initial",
            )
            if self.rgmt_reference_mode == 'pico':
                self.pico_pose_client = PicoHumanPoseClient(self.pico_pose_endpoint)
                print(f"RGMT reference mode: pico, endpoint={self.pico_pose_endpoint}")
            else:
                self.pico_pose_client = ZeroLabHumanPoseClient(self.zerolab_pose_endpoint)
                print(f"RGMT reference mode: zerolab, endpoint={self.zerolab_pose_endpoint}")
        
        # beyondmimic模型
        self.dance_lie_down = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["lie_down"], self.onnx_file_dict["lie_down"], start_frame=160,fixed_pos=True)#fixed policy
        self.dance_getup_face = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["getup_face"], self.onnx_file_dict["getup_face"], start_frame=1, fixed_pos=False)#fixed policy
        self.dance_getup_back = DanceMotionPolicyGravityIsaaclabV3(self.npz_file_dict["getup_back"], self.onnx_file_dict["getup_back"], start_frame=1, fixed_pos=False)#fixed policy

        self.dance_lie_down.end_frame = self.dance_lie_down.end_frame - 190

    def _run_motion_dispatch(self, q, dq, quat, omega, cmd_vel):
        """运行当前 self.motion_type 对应的推理分支（输出会经 send_to_motor 发布/捕获）。"""
        if self.motion_type == motionType.rgmt:
            self.target_dof_pos = self.rgmt.inference_step(
                q,
                dq,
                quat,
                omega,
                advance=self.dance_flag == 1,
            )
            # inference_step 会将 timestep 饱和在 end_frame。到达末帧后继续
            # 运行策略以保留 proprio/action 历史，不再 end+1 -> end 回退。
            self.send_to_motor(self.target_dof_pos, self.rgmt.kps, self.rgmt.kds)

        if self.motion_type == motionType.rgmt_neural:
            if self.neural_rgmt is None or (self.smplx_motion is None and self.pico_pose_client is None):
                raise RuntimeError("neural RGMT mode is not initialized")
            if self.pico_pose_client is not None:
                self.pico_pose_client.poll()
                history = self.pico_pose_client.history()
                if history is None or self.pico_pose_client.stale(timeout_s=0.25):
                    if not self._pico_waiting_logged:
                        print("Tracking pose unavailable/stale; holding default standing pose")
                        self._pico_waiting_logged = True
                    if self._pico_tracking_active:
                        self.neural_rgmt.reset()
                    self._pico_tracking_active = False
                    self._pico_active_logged = False
                    self._pico_blend_step = 0
                    self._pico_wait_balance(q, dq, quat, omega)
                    return
                self._pico_waiting_logged = False
                body_pos, body_rot = history
            else:
                body_pos, body_rot = self.smplx_motion.history(self.smplx_frame)
            self.target_dof_pos = self.neural_rgmt.inference_step(
                q,
                dq,
                quat,
                omega,
                body_pos,
                body_rot,
                source_time_s=(self.pico_pose_client.last_timestamp_ns * 1e-9
                               if self.pico_pose_client is not None else self.smplx_frame * self.dt),
            )
            if self.target_dof_pos is None:
                self._pico_wait_balance(q, dq, quat, omega)
                return
            kp, kd = self.neural_rgmt.rgmt.kps, self.neural_rgmt.rgmt.kds
            if self.pico_pose_client is not None:
                self._pico_tracking_active = True
                if not self._pico_active_logged:
                    print("Canonical tracking reference ready; blending into RGMT", flush=True)
                    self._pico_active_logged = True
                self._pico_blend_step += 1
                alpha = min(1.0, self._pico_blend_step * self.dt / 0.6)
                if self._pico_wait_target is not None and alpha < 1.0:
                    self.target_dof_pos = (1-alpha)*self._pico_wait_target + alpha*self.target_dof_pos
                    kp = (1-alpha)*self.amp_walk.kps + alpha*kp
                    kd = (1-alpha)*self.amp_walk.kds + alpha*kd
            self.send_to_motor(
                self.target_dof_pos,
                kp,
                kd,
            )
            if self.smplx_motion is not None and self.dance_flag == 1:
                self.smplx_frame = min(
                    self.smplx_frame + 1,
                    len(self.smplx_motion.body_positions_w) - 1,
                )
        
        # 倒地后按 Y 的第一阶段：从当前实测关节位置缓慢移动到统一的
        # 默认关节角，避免直接跳变到起身姿态。
        if (
            self._getup_ramp_active
            and self.motion_type in (
                motionType.dance_getup_face,
                motionType.dance_getup_back,
            )
        ):
            model = (
                self.dance_getup_face
                if self.motion_type == motionType.dance_getup_face
                else self.dance_getup_back
            )
            self._getup_ramp_step += 1
            alpha = min(
                1.0,
                self._getup_ramp_step / max(1, self._getup_ramp_total_steps),
            )
            self.target_dof_pos = (
                (1.0 - alpha) * self._getup_ramp_start_pos
                + alpha * self._getup_ramp_target_pos
            )
            self.send_to_motor(self.target_dof_pos, model.kps, model.kds)
            if self._getup_ramp_step >= self._getup_ramp_total_steps:
                self._getup_ramp_active = False
                model.timestep = model.start_frame
                model.timeinit = 0.0
                print("Get-up ramp finished, starting motion replay.")
            return

        if self.motion_type == motionType.dance_getup_face:
            if self.dance_getup_face.timestep <= self.dance_getup_face.end_frame:
                self.target_dof_pos = self.dance_getup_face.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_getup_face.kps, self.dance_getup_face.kds)

            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_jojo.timestep)
                self.dance_getup_face.timestep += 1
                
            # 动作结束检测
            if self.dance_getup_face.timestep > self.dance_getup_face.end_frame:
                print("Get-up motion finished, switching to walking.")
                # dance_walk 不是有效的 motionType；预热后直接切回 AMP 行走，
                # 不再对已经结束的起身动作做二次过渡。
                self.switch_to_motion(
                    self.amp_walk,
                    motionType.amp_walk,
                    num=20,
                    with_cmd_vel=True,
                    transition_time=0.0,
                )
                
        if self.motion_type == motionType.dance_getup_back:
            if self.dance_getup_back.timestep <= self.dance_getup_back.end_frame:
                self.target_dof_pos = self.dance_getup_back.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_getup_back.kps, self.dance_getup_back.kds)

            # 动作管理
            if self.dance_flag==1:
                # print("timestep:", self.dance_jojo.timestep)
                self.dance_getup_back.timestep += 1
                
            # 动作结束检测
            if self.dance_getup_back.timestep > self.dance_getup_back.end_frame:
                print("Get-up motion finished, switching to walking.")
                # dance_walk 不是有效的 motionType；预热后直接切回 AMP 行走，
                # 不再对已经结束的起身动作做二次过渡。
                self.switch_to_motion(
                    self.amp_walk,
                    motionType.amp_walk,
                    num=20,
                    with_cmd_vel=True,
                    transition_time=0.0,
                )
                      
        if self.motion_type == motionType.dance_lie_down:
            if self.dance_lie_down.timestep <= self.dance_lie_down.end_frame:
                self.target_dof_pos = self.dance_lie_down.inference_step(q, dq, quat, omega)
                # 发布关节控制指令
                self.send_to_motor(self.target_dof_pos, self.dance_lie_down.kps, self.dance_lie_down.kds)
                
            # 动作管理
            if self.dance_flag==1:
                self.dance_lie_down.timestep += 1
                
            # 动作结束检测
            if self.dance_lie_down.timestep > self.dance_lie_down.end_frame:
                # 躺下结束后保持末帧，等待再次按 Y 键触发起身。
                self.dance_lie_down.timestep = self.dance_lie_down.end_frame
        
        if self.motion_type == motionType.amp_walk:
            # print("AMP walking...")
            self.target_dof_pos = self.amp_walk.inference_step(q, dq, quat, omega, cmd_vel)
            self.send_to_motor(self.target_dof_pos, self.amp_walk.kps, self.amp_walk.kds)
            
        if self.motion_type == motionType.amp_run:
            # print("AMP running...")
            self.target_dof_pos = self.amp_run.inference_step(q, dq, quat, omega, cmd_vel)
            self.send_to_motor(self.target_dof_pos, self.amp_run.kps, self.amp_run.kds)    
     
    def _pico_wait_balance(self, q, dq, quat, omega):
        # A fixed PD joint pose is not a balance controller. Continue running
        # the locomotion policy with zero velocity until the reference is ready.
        self.target_dof_pos = self.amp_walk.inference_step(
            q, dq, quat, omega, np.zeros(3, dtype=np.float32)
        )
        self._pico_wait_target = self.target_dof_pos.copy()
        self.send_to_motor(self.target_dof_pos, self.amp_walk.kps, self.amp_walk.kds)

    def joy_callback(self, msg):
        with self.lock_in:
            if self.motion_type == motionType.amp_walk:
                self.vx = np.clip(msg.vel_des.x, -0.6, 1.0)
            else:
                self.vx = np.clip(msg.vel_des.x, -1.0, 5.0)
   
            self.vy = msg.vel_des.y
            self.dyaw = msg.yawdot_des

            motion_a = msg.btn_5 # A
            motion_x = msg.btn_6 # X
            motion_y = msg.btn_7 # Y
            motion_b = msg.btn_10 # B

            #防止误触
            if self.step < 2:
                self.motion_y_prev = motion_y
                self.motion_b_prev = motion_b
            if self.step < 1:
                self.motion_a_prev = motion_a
                self.motion_x_prev = motion_x
                self.motion_y_prev = motion_y
                self.motion_b_prev = motion_b
                
            #按键状态变化检测
            _now = self.get_clock().now().nanoseconds * 1e-9
            def _debounced(changed, until_attr):
                if changed:
                    if _now >= getattr(self, until_attr):
                        setattr(self, until_attr, _now + self._motion_x_debounce)
                        return True
                    return False
                return False

            # 只响应按下沿；如果把松开沿也当作触发，会在起身结束后
            # 因 Y 键释放再次进入 lie_down。
            self.motion_a_changed = _debounced(
                motion_a and not self.motion_a_prev,
                '_motion_a_debounce_until',
            )
            self.motion_x_changed = _debounced(
                motion_x and not self.motion_x_prev,
                '_motion_x_debounce_until',
            )
            self.motion_y_changed = _debounced(
                motion_y and not self.motion_y_prev,
                '_motion_y_debounce_until',
            )
            self.motion_b_changed = _debounced(
                motion_b and not self.motion_b_prev,
                '_motion_b_debounce_until',
            )
            # print(f"Received motion command: A={motion_a} (changed: {self.motion_a_changed}), X={motion_x} (changed: {self.motion_x_changed}), Y={motion_y} (changed: {self.motion_y_changed}), B={motion_b} (changed: {self.motion_b_changed})")
            
            #按键状态保存
            self.motion_a_prev = motion_a
            self.motion_x_prev = motion_x
            self.motion_y_prev = motion_y
            self.motion_b_prev = motion_b
            
            use_button1 = True
            
            if use_button1: 
                # 常用按键组合：A键切换amp_walk，X/Y/B键切换三个舞蹈动作
                if self.motion_a_changed == 1:
                    if self.step < 2:
                        self.robot_reset(2, True) # first reset
                        self.step = 2
                    if self.motion_type != motionType.amp_walk:
                        self.switch_to_motion(self.amp_walk, motionType.amp_walk, num=20, with_cmd_vel=True)
                    else:
                        self.motion_type = motionType.amp_walk

                elif self.motion_x_changed == 1:
                    self.dance_flag += 1
                    if self.dance_flag > 1:
                        self.dance_flag = 0
                    if self.motion_type == motionType.amp_walk:
                        self.dance_flag = 1
                        if self.rgmt_reference_mode in {'neural_retarget', 'pico', 'zerolab'}:
                            self.neural_rgmt.reset()
                            self._pico_active_logged = False
                            self._pico_tracking_active = False
                            self._pico_blend_step = 0
                            self._pico_wait_target = self.amp_walk.default_dof_pos.copy()
                            if self.smplx_motion is not None:
                                # Start after one 20-frame history window.
                                self.smplx_frame = min(
                                    19,
                                    len(self.smplx_motion.body_positions_w) - 1,
                                )
                            previous_motion = self.motion_type
                            self.motion_type = motionType.rgmt_neural
                            if self.rgmt_reference_mode in {'pico', 'zerolab'}:
                                # No valid tracking frame may be available yet;
                                # switch immediately to the safe standing
                                # fallback instead of blending a stale motion.
                                self.transition_active = False
                                self.prev_motion_type = None
                                self._blend_pending = False
                            else:
                                self.start_motion_transition(previous_motion)
                            print("X: start neural_retarget -> RGMT tracking")
                        else:
                            self.rgmt.timestep = self.rgmt.start_frame
                            self.rgmt.timeinit = 0.0
                            self.switch_to_motion(self.rgmt, motionType.rgmt, num=20)
                       
                elif self.motion_y_changed == 1:
                    # Y 键在站立时执行躺下，躺倒时根据身体朝向选择起身策略。
                    self.select_fall_getup_motion()
                    

                elif self.motion_b_changed == 1:

                    self.switch_to_motion(self.amp_run, motionType.amp_run, num=20, with_cmd_vel=True)
                                           
    def send_to_motor(self, dof_pos_target, kps, kds):
        dof_pos_target = np.asarray(dof_pos_target, dtype=np.float32)
        kps = np.asarray(kps, dtype=np.float32)
        kds = np.asarray(kds, dtype=np.float32)

        # 跌倒保护锁存期间仍发送通信心跳，避免电机节点超时；但不再
        # 发送走路模型的目标和增益。使用当前实测关节位置、零 kp/kd，
        # 使保护状态不主动驱动倒地后的机器人。按 Y 起身后解除锁存。
        if self.fall_protection_latched:
            hold_pos = np.asarray(self.qpos, dtype=np.float32).reshape(-1)
            if hold_pos.shape != dof_pos_target.shape or not np.all(np.isfinite(hold_pos)):
                hold_pos = np.zeros_like(dof_pos_target)
            dof_pos_target = hold_pos
            kps = np.zeros_like(kps)
            kds = np.zeros_like(kds)

        # 切换过渡阶段：捕获旧模型输出，跳过实际发布
        if self._capture_motor:
            self._captured = (dof_pos_target.copy(), kps.copy(), kds.copy())
            return

        # 切换过渡阶段：把新模型输出与之前捕获的旧模型输出按权重混合
        if self._blend_pending and self._old_action is not None and self.transition_active:
            # t = min(1.0, max(0.0, (self.transition_step_count + 1) / max(1, self.transition_total_steps)))
            # alpha = t * t * (3.0 - 2.0 * t)  # smoothstep: slow start, slow end
            alpha = min(1.0, max(0.0, (self.transition_step_count + 1) / max(1, self.transition_total_steps)))
            old_pos, old_kps, old_kds = self._old_action
            if old_pos.shape == dof_pos_target.shape:
                dof_pos_target = (1.0 - alpha) * old_pos + alpha * dof_pos_target
            if old_kps.shape == kps.shape:
                kps = (1.0 - alpha) * old_kps + alpha * kps
            if old_kds.shape == kds.shape:
                kds = (1.0 - alpha) * old_kds + alpha * kds
            self._blend_pending = False  # 每个 tick 仅混合一次

        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = dof_pos_target.tolist()
        msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.kp = kps.tolist()
        msg.kd = kds.tolist()
        self.act_pub.publish(msg)   
    
    def robot_reset(self, reset_step, release):
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name
    
        while not self.rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.rest_srv.call_async(req)
        
    def sim_robot_reset(self):        
        req = bxiSrv.SimulationReset.Request()
        req.header.frame_id = robot_name

        base_pose = Pose()
        # base_pose.position.x = 0.0
        # base_pose.position.y = 0.0
        # base_pose.position.z = 1.0
        # base_pose.orientation.x = 0.0
        # base_pose.orientation.y = 0.0
        # base_pose.orientation.z = 0.0
        # base_pose.orientation.w = 1.0        
        
        #[0.707, 0.0, -0.707, 0.0]
        base_pose.position.x = 0.0
        base_pose.position.y = 0.0
        base_pose.position.z = 0.5
        base_pose.orientation.x = 0.0
        base_pose.orientation.y = -0.707
        base_pose.orientation.z = 0.0
        base_pose.orientation.w = 0.707  

        joint_state = JointState()
        joint_state.name = joint_name
        joint_state.position = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.velocity = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.effort = np.zeros(dof_num, dtype=np.float32).tolist()
        
        req.base_pose = base_pose
        req.joint_state = joint_state
    
        while not self.sim_rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.sim_rest_srv.call_async(req)
    
    def joint_callback(self, msg):
        joint_pos = msg.position
        joint_vel = msg.velocity
        with self.lock_in:
            self.qpos = np.array(joint_pos)
            self.qvel = np.array(joint_vel)

    def timer_callback(self):
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_reset(1, False) # first reset
            # self.sim_robot_reset()
            print('robot reset 1!')
            self.step = 1
            return

        if self.step == 1: #软启动
            soft_start = self.loop_count/(3./self.dt) # 3秒关节缓启动
            if soft_start > 1:
                soft_start = 1
            #软启动到舞蹈动作的第一帧    
            soft_joint_kp = self.soft_start_kps * soft_start
            soft_joint_kd = self.soft_start_kds
               
            self.send_to_motor(self.start_frame_pos, soft_joint_kp, soft_joint_kd)

            # 仿真启动时保持虚拟悬挂。只有按 A 键发送
            # robot_reset(2, True) 后才释放悬挂并进入正常运动状态。
                    
        elif self.step == 2:
            # 参数读取
            with self.lock_in:
                q = self.qpos
                dq = self.qvel
                quat = self.quat
                omega = self.omega
                cmd_vel = np.array([self.vx, self.vy, self.dyaw])

            # 只对 AMP 走路/跑步模型启用跌倒锁存。躺下和起身动作由
            # Y 键流程管理，不能在这些动作期间被保护逻辑打断。
            walk_motion = self.motion_type in (
                motionType.amp_walk,
                motionType.amp_run,
            )
            quat_valid = bool(np.all(np.isfinite(quat)) and np.linalg.norm(quat) > 0.5)
            gravity_body = get_gravity_orientation(quat) if quat_valid else None
            fallen = (
                quat_valid
                and abs(float(gravity_body[2])) < math.cos(math.pi / 2.5)
            )
            if walk_motion and fallen and not self.fall_protection_latched:
                self.fall_protection_latched = True
                self.transition_active = False
                self.prev_motion_type = None
                self._old_action = None
                self._blend_pending = False
                print(
                    "Fall protection latched: motion output stopped, "
                    "heartbeat active; press Y to get up."
                )
                            
            # 状态机
            if self.state==robotState.stand:
                self.state = robotState.stand_to_motion
                print("state: stand_to_motion [dance]")
                
            elif self.state==robotState.stand_to_motion:
                #动作过渡
                self.state=robotState.motion


            elif self.state==robotState.motion:
                # --- 模型切换平滑过渡（双模型加权） ---
                if self.transition_active and self.prev_motion_type is not None \
                        and self.prev_motion_type != self.motion_type:
                    self._capture_motor = True
                    self._captured = None
                    _saved_mt = self.motion_type
                    self.motion_type = self.prev_motion_type
                    try:
                        self._run_motion_dispatch(q, dq, quat, omega, cmd_vel)
                    finally:
                        self.motion_type = _saved_mt
                        self._capture_motor = False
                    self._old_action = self._captured
                    self._blend_pending = self._old_action is not None
                else:
                    self._blend_pending = False
                    self._old_action = None

                self._run_motion_dispatch(q, dq, quat, omega, cmd_vel)

                # 推进过渡进度
                if self.transition_active:
                    self.transition_step_count += 1
                    if self.transition_step_count >= self.transition_total_steps:
                        self.transition_active = False
                        self.prev_motion_type = None
                        self._old_action = None
                        print(f"motion transition finished -> {self.motion_type}")

            elif self.state==robotState.motion_to_stand:
                #站立过渡
                self.state=robotState.stand
                print("state: stand")
    
            else:
                #其他状态机情况
                raise Exception   

        self.loop_count += 1
      
    def start_motion_transition(self, prev_motion_type, transition_time=None):
        """启动从 prev_motion_type 到当前 self.motion_type 的混合过渡。"""
        if prev_motion_type is None or prev_motion_type == self.motion_type:
            return
        duration = self.transition_duration if transition_time is None else float(transition_time)
        if duration <= 0:
            # 显式关闭过渡时必须清理旧过渡状态；否则上一次起身/躺下
            # 的 transition_active 可能继续把动作切换逻辑卡住。
            self.transition_active = False
            self.prev_motion_type = None
            self._old_action = None
            self._blend_pending = False
            return
        self.transition_total_steps = max(1, int(round(duration / self.dt)))
        self.transition_step_count = 0
        self.prev_motion_type = prev_motion_type
        self.transition_active = True
        self._old_action = None
        self._blend_pending = False
        print(f"motion transition start: {prev_motion_type} -> {self.motion_type}, "
              f"{self.transition_total_steps} steps ({duration:.2f}s)")

    def switch_to_motion(self, new_model, new_motion_type, num=20, with_cmd_vel=False, transition_time=None):
        """预热新模型并启动平滑过渡（旧模型推理 + 新模型推理按权重混合）。"""
        prev = self.motion_type
        self.preheat_model(new_model, num=num, with_cmd_vel=with_cmd_vel)
        self.motion_type = new_motion_type
        self.start_motion_transition(prev, transition_time=transition_time)

    # --- 模型切换过渡逻辑 ---
    def preheat_model(self, model, num=2, with_cmd_vel=False):
        # 用当前观测预推理 num 帧，不输出到电机
        #####
        if getattr(model, "skip_legacy_preheat", False):
            # RGMT history must contain real consecutive control frames.  Its
            # first real inference call repeats the current proprio token ten
            # times, matching the training reset semantics.
            model.reset(start_frame=model.timestep)
            return
        #####
        q = self.qpos.copy()
        dq = self.qvel.copy()
        quat = self.quat.copy()
        omega = self.omega.copy()
        cmd_vel = np.array([self.vx, self.vy, self.dyaw], dtype=np.float32)
        for _ in range(num):
            if with_cmd_vel:
                model.inference_step(q, dq, quat, omega, cmd_vel)
            else:
                model.inference_step(q, dq, quat, omega)

    def _start_getup_ramp(self, model, motion):
        """启动倒地后的关节缓启动，先到达模型默认关节角。"""
        q = np.asarray(self.qpos, dtype=np.float32).reshape(-1).copy()

        model.timestep = model.start_frame
        model.timeinit = 0.0
        if hasattr(model, "action_buffer"):
            model.action_buffer.fill(0.0)
        if hasattr(model, "history_buffers"):
            model.history_buffers.clear()
        default_target = getattr(model, "default_dof_pos", None)
        if default_target is None:
            raise AttributeError("get-up model must provide default_dof_pos")
        default_target = np.asarray(default_target, dtype=np.float32).reshape(-1)
        if q.shape != default_target.shape or not np.all(np.isfinite(q)):
            q = default_target.copy()

        self.motion_type = motion
        self.dance_flag = 1
        self.fall_protection_latched = False
        self.transition_active = False
        self.prev_motion_type = None
        self._old_action = None
        self._blend_pending = False
        self._getup_ramp_start_pos = q
        self._getup_ramp_target_pos = default_target.copy()
        self._getup_ramp_step = 0
        self._getup_ramp_total_steps = max(1, int(round(1.0 / self.dt)))
        self._getup_ramp_active = True
        print(
            f"Get-up ramp started: {self._getup_ramp_total_steps} steps "
            f"({self._getup_ramp_total_steps * self.dt:.2f}s)."
        )

    def select_fall_getup_motion(self):
        """Y 键在站立/躺倒之间切换动作。

        ``get_gravity_orientation`` 返回机体坐标系中的重力方向；站立时
        z 分量接近 -1，躺倒时 z 分量接近 0。x 分量的符号沿用原工程
        的约定：x < 0 表示正面朝上，反之表示背面朝上。
        """
        gravity_body = get_gravity_orientation(self.quat.copy())
        fall_latched = self.fall_protection_latched

        # 动作状态优先于瞬时 IMU 姿态：躺下动作完成后机器人可能仍有
        # 一定倾角，单靠 gravity_body[2] 会偶尔把它误判为站立，再次
        # 按 Y 就会重新播放 lie_down。
        if fall_latched:
            # 跌倒保护已经确认机器人倒地，Y 直接选择起身，不再依赖
            # 此刻可能不稳定的 IMU z 分量。
            fallen = True
        elif self.motion_type == motionType.dance_lie_down:
            fallen = True
        elif self.motion_type in (
            motionType.dance_getup_face,
            motionType.dance_getup_back,
        ):
            print("Y ignored: get-up motion is still running")
            return False
        else:
            # 与状态机中的跌倒阈值（约 72 度）保持一致，避免站立时误触发。
            fallen = abs(float(gravity_body[2])) < math.cos(math.pi / 2.5)

        self.dance_flag = 1
        if not fallen:
            model = self.dance_lie_down
            motion = motionType.dance_lie_down
            getup_name = "lie down"
        else:
            if gravity_body[0] < 0.0:
                model = self.dance_getup_face
                motion = motionType.dance_getup_face
                getup_name = "face up"
            else:
                model = self.dance_getup_back
                motion = motionType.dance_getup_back
                getup_name = "back up"

        model.timestep = model.start_frame
        model.timeinit = 0.0
        if motion in (motionType.dance_getup_face, motionType.dance_getup_back):
            if fallen:
                self._start_getup_ramp(model, motion)
            else:
                self.fall_protection_latched = False
                self.switch_to_motion(model, motion, num=20)
        else:
            self.switch_to_motion(model, motion, num=20)
        print(
            f"Y motion: {getup_name}, gravity_body="
            f"{np.round(gravity_body, 3).tolist()}"
        )
        return True
             
    def load_files(self):
        self.declare_parameter('/use_hardware') # 声明 use_hardware 参数，默认 False
        self.use_hardware = self.get_parameter('/use_hardware').value
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        # print('topic_prefix:', self.topic_prefix)
        
        self.declare_parameter('/npz_file_dict', json.dumps({}))
        npz_file_json = self.get_parameter('/npz_file_dict').value
        self.npz_file_dict = json.loads(npz_file_json)
        # print('npz_file:')
        # for key,value in self.npz_file_dict.items():
            # print("Load motion from ",key,": ",value)
            
        self.declare_parameter('/onnx_file_dict', json.dumps({}))
        onnx_file_json = self.get_parameter('/onnx_file_dict').value
        self.onnx_file_dict = json.loads(onnx_file_json)

        self.declare_parameter('/rgmt_reference_mode', 'npz')
        self.rgmt_reference_mode = str(
            self.get_parameter('/rgmt_reference_mode').value
        ).lower()
        self.declare_parameter('/pico_pose_endpoint', 'tcp://127.0.0.1:28704')
        self.pico_pose_endpoint = str(self.get_parameter('/pico_pose_endpoint').value)
        self.declare_parameter('/zerolab_pose_endpoint', 'tcp://127.0.0.1:5558')
        self.zerolab_pose_endpoint = str(self.get_parameter('/zerolab_pose_endpoint').value)
        if self.rgmt_reference_mode not in {'npz', 'neural_retarget', 'pico', 'zerolab'}:
            raise ValueError(
                "'/rgmt_reference_mode' must be 'npz', 'neural_retarget', 'pico', or 'zerolab', "
                f"got {self.rgmt_reference_mode!r}"
            )

        # 模型切换过渡时长（秒），可在 launch 时配置；<=0 表示关闭混合
        # self.declare_parameter('/transition_time', 0.3)
        self.declare_parameter('/transition_time', 0.4)
        # self.declare_parameter('/transition_time', 0.5)
        # self.declare_parameter('/transition_time', 0.6)
        self._param_transition_time = float(self.get_parameter('/transition_time').value)
        # print('onnx_file:')
        # for key,value in self.onnx_file_dict.items():
            # print("Load model from ",key,": ",value)

    def init_pub_sub(self):
        # 订阅和发布主题
        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, reliability=qos_profile_sensor_data.reliability)
        
        self.act_pub = self.create_publisher(bxiMsg.ActuatorCmds, self.topic_prefix+'actuators_cmds', qos)  # CHANGE
        
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, self.topic_prefix+'odom', self.odom_callback, qos)
        self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', self.imu_callback, qos)
        self.touch_sub = self.create_subscription(bxiMsg.TouchSensor, self.topic_prefix+'touch_sensor', self.touch_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', self.joy_callback, qos)

        self.rest_srv = self.create_client(bxiSrv.RobotReset, self.topic_prefix+'robot_reset')
        self.sim_rest_srv = self.create_client(bxiSrv.SimulationReset, self.topic_prefix+'sim_reset')
        
        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group_2 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()
    
    def init_controller(self):
        self.vae_vel = np.zeros(3, dtype=np.float32)
        
        # 运动命令变量
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0
        self.stand_height = 1.0
        
        # 速度偏移变量
        self.vx_offset = 0.0
        self.vy_offset = 0.0
        self.dyaw_offset = 0.0
        
        # 遥控器相关变量
        self.motion_a_prev = False
        self.motion_x_prev = False
        self.motion_y_prev = False
        self.motion_b_prev = False
        self.motion_a_changed = False
        self.motion_x_changed = False
        self.motion_y_changed = False
        self.motion_b_changed = False

        # X 按键防抖：变化触发后，缓冲期内再次变化不更新 motion_x_changed
        # 缓冲时长 0.3s，可按需调整
        self._motion_x_debounce = 0.5  # 秒
        self._motion_x_debounce_until = -999.0

        # ABXY 按键统一防抖（与 X 共用同一缓冲时长）
        self._motion_a_debounce_until = -999.0
        self._motion_y_debounce_until = -999.0
        self._motion_b_debounce_until = -999.0

        # --- 模型切换平滑过渡状态 ---
        # 切换两个模型时，旧模型继续推理，与新模型按权重 alpha(0->1) 加权后发给电机。
        # transition_duration 单位为秒，可通过 ROS 参数 /transition_time 调整；
        # 若 <=0 则关闭过渡（保留旧版本即时切换行为）。
        self.transition_duration = float(getattr(self, '_param_transition_time', 0.4))
        self.transition_active = False
        self.transition_total_steps = 0
        self.transition_step_count = 0
        self.prev_motion_type = None
        self._capture_motor = False
        self._captured = None
        self._old_action = None
        self._blend_pending = False

        # 跌倒保护：走路/跑步时一旦检测到倒地就锁存，停止发布电机指令，
        # 只有 Y 键触发起身后才解除。
        self.fall_protection_latched = False
        self._getup_ramp_active = False
        self._getup_ramp_step = 0
        self._getup_ramp_total_steps = 0
        self._getup_ramp_start_pos = None
        self._getup_ramp_target_pos = None

    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        # quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)
        quat_tmp1 = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.double)

        with self.lock_in:
            self.quat = quat_tmp1
            self.omega = np.array([avel.x, avel.y, avel.z])

    def touch_callback(self, msg):
        foot_force = msg.value
        
    def odom_callback(self, msg): # 全局里程计（上帝视角，仅限仿真使用）
        base_pose = msg.pose
        base_twist = msg.twist

def main(args=None):
   
    time.sleep(5)
    
    rclpy.init(args=args)
    node = BxiExample()
    
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        if node.neural_rgmt is not None:
            node.neural_rgmt.close()
        if node.pico_pose_client is not None:
            node.pico_pose_client.close()
        node.destroy_node()
        
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()
