"""Actual simulation and the existing controller, isolated ROS domain."""
from pathlib import Path
import os,time,json,signal,subprocess,argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--domain", type=int, default=86)
parser.add_argument("--output", type=Path, default=Path("/tmp/bxi_depth_check"))
options=parser.parse_args()
os.environ["ROS_DOMAIN_ID"]=str(options.domain)
from ament_index_python.packages import get_package_share_path, get_package_prefix
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry
from bxi_example_py_elf3.bxi_example_dance import BxiExample,motionType
from bxi_example_py_elf3.model_config import get_model_file_dicts, get_simulation_model_file, DEPTH_CAMERA_NAME, DEPTH_WALK_MODE
import communication.msg as messages

out=options.output;out.mkdir(parents=True,exist_ok=True)
p=Path(get_package_share_path('bxi_example_py_elf3'))
simulator=Path(get_package_prefix('mujoco'))
if not (simulator/'share/mujoco/CAMERA_SENSOR.md').is_file():
 raise RuntimeError('Source the updated bxi_ros2_pkg before running this check')
log=(out/'simulation.log').open('w')
sim=subprocess.Popen([str(simulator/'lib/mujoco/simulation'),'--ros-args','-p',f'simulation/model_file:={get_simulation_model_file()}'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
npz,onnx=get_model_file_dicts()
args=['--ros-args','-p','/use_hardware:=false','-p','/topic_prefix:=simulation/','-p','/depth_walk_enabled:=true','-p','/depth_walk_mode:='+DEPTH_WALK_MODE,'-p','/depth_camera_name:='+DEPTH_CAMERA_NAME,'-p','/rgmt_reference_mode:=npz','-p','/npz_file_dict:='+repr(json.dumps(npz)),'-p','/onnx_file_dict:='+repr(json.dumps(onnx))]
node=None;executor=None
try:
 rclpy.init(args=args)
 node=BxiExample();executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
 received=[];infos=[];outputs=[];timings=[];resets=[];poses=[]
 original_reset=node.rest_srv.call_async
 def reset(request):
  future=original_reset(request);resets.append((request.reset_step,request.release,future));return future
 node.rest_srv.call_async=reset
 node.create_subscription(Odometry,"/simulation/odom",lambda m:poses.append([m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z]),qos_profile_sensor_data)
 node.create_subscription(Image,f'/simulation/{DEPTH_CAMERA_NAME}/depth/image_rect_raw',lambda m:received.append((time.monotonic(),m.width,m.height,m.encoding,len(m.data))),qos_profile_sensor_data)
 node.create_subscription(CameraInfo,f'/simulation/{DEPTH_CAMERA_NAME}/depth/camera_info',lambda m:infos.append((m.width,m.height,list(m.k))),qos_profile_sensor_data)
 original=node.send_to_motor
 def monitor(target,kp,kd):
  if not node._capture_motor:outputs.append((time.monotonic(),node.motion_type,np.asarray(target).copy(),np.asarray(kp).copy()))
  return original(target,kp,kd)
 node.send_to_motor=monitor
 publisher=node.create_publisher(messages.MotionCommands,'motion_commands',qos_profile_sensor_data)
 original_infer=node.depth_walk.inference_step
 def infer(*a,**kw):
  start=time.perf_counter();result=original_infer(*a,**kw);timings.append((time.perf_counter()-start)*1000);return result
 node.depth_walk.inference_step=infer
 def spin(seconds):
  end=time.monotonic()+seconds
  while time.monotonic()<end:executor.spin_once(timeout_sec=.01)
 def command(button=None,vx=0.):
  message=messages.MotionCommands();message.vel_des.x=float(vx);message.height_des=1.
  if button: setattr(message,button,1)
  publisher.publish(message)
  spin(.05);message.btn_5=message.btn_10=0;publisher.publish(message)
 spin(5.)
 assert received and infos,'No real depth topics: '+(out/'simulation.log').read_text()[-3000:]
 assert node.depth_camera.sample(advance=False) is not None,node.depth_camera.last_error
 assert node.step==1
 command('btn_5');spin(2.)
 assert node.step==2
 assert len(resets)==2 and all(f.done() and f.result().is_success for _,_,f in resets), [(step,release,f.result())for step,release,f in resets]
 assert resets[-1][:2]==(2,True)
 command('btn_10');spin(.6)
 assert node.motion_type==motionType.depth_walk
 walk_start=np.asarray(poses[-1])
 for _ in range(80):command(vx=.3)
 assert node.motion_type==motionType.depth_walk and not node.fall_protection_latched
 walk_delta=np.asarray(poses[-1])-walk_start
 assert np.linalg.norm(walk_delta[:2])>.15, f'No walking displacement: {walk_delta}'
 hidden_norm=None if node.depth_walk._hidden is None else float(np.linalg.norm(node.depth_walk._hidden))
 assert hidden_norm is None or hidden_norm>0.
 depth_outputs=[row for row in outputs if row[1]==motionType.depth_walk]
 assert len(depth_outputs)>80
 assert all(row[2].shape==(29,) and np.isfinite(row[2]).all() for row in depth_outputs)
 command('btn_10');spin(.8)
 assert node.motion_type==motionType.amp_walk
 command('btn_10');spin(.8)
 assert node.motion_type==motionType.depth_walk
 command('btn_5');spin(.8)
 assert node.motion_type==motionType.amp_walk
 command('btn_10');spin(.8)
 assert node.motion_type==motionType.depth_walk
 node.destroy_subscription(node.depth_image_sub);node.depth_image_sub=None
 spin(.8)
 assert node.motion_type==motionType.amp_walk
 assert node.vx==node.vy==node.dyaw==0.
 command('btn_10');spin(.3)
 assert node.motion_type==motionType.amp_walk
 assert not node.fall_protection_latched
 report={'reset_steps_acknowledged':[step for step,_,_ in resets],'root_z_range_m':np.ptp(np.asarray(poses)[:,2]).item(),'joint_feedback_error_norm':np.linalg.norm(node.qpos-node.depth_walk.default_dof_pos).item(),'camera_frames':len(received),'camera_width':received[-1][1],'camera_height':received[-1][2],'encoding':received[-1][3],'camera_hz':(len(received)-1)/(received[-1][0]-received[0][0]),'depth_motor_frames':len(depth_outputs),'inference_ms_p99':float(np.percentile(timings,99)),'projection_roi':node.depth_camera.last_crop,'processed_depth_shape':node.depth_walk.depth_input_buffer.shape,'B_entry_B_exit_A_exit':'passed','timeout_fallback_and_stale_entry_rejection':'passed','fall_latch':node.fall_protection_latched,'walk_displacement_m':walk_delta.tolist(),'active_hidden_state_norm':hidden_norm,'model':node.depth_walk.model_onnx_path}
 print(json.dumps(report,indent=2));(out/'ros_report.json').write_text(json.dumps(report,indent=2))
finally:
 if executor:executor.shutdown()
 if node:
  if node.neural_rgmt is not None:node.neural_rgmt.close()
  if node.rgmt is not None and hasattr(node.rgmt,'close'):node.rgmt.close()
  if node.pico_pose_client is not None:node.pico_pose_client.close()
  node.destroy_node()
 if rclpy.ok():rclpy.shutdown()
 if sim.poll() is None:os.killpg(sim.pid,signal.SIGTERM)
 try:sim.wait(timeout=5.)
 except subprocess.TimeoutExpired:os.killpg(sim.pid,signal.SIGKILL);sim.wait()
 log.close()
