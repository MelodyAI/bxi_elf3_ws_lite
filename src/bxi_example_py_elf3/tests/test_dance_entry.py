"""B-key entry waits for balance and resets the real dance policy on re-entry."""
from pathlib import Path
from threading import Lock
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from communication.msg import MotionCommands

from bxi_example_py_elf3.bxi_example_dance import BxiExample, motionType
from bxi_example_py_elf3.models.beyondmimic import DanceMotionPolicyGravityIsaaclabV3


@pytest.fixture
def controller():
    node = NS()
    BxiExample.init_controller(node)
    node.dt = .02
    node.step = 2
    node.motion_type = motionType.amp_walk
    node.depth_walk_enabled = False
    node.dance_flag = 1
    node.lock_in = Lock()
    node.dance_mimic = Mock(start_frame=1, timestep=1, end_frame=100)
    node.amp_walk = Mock()
    node.switch_to_motion = Mock()
    node.act_pub = Mock()
    node.get_logger = Mock(return_value=Mock())
    clock = NS(nanoseconds=10**12, to_msg=lambda: Time())
    node.get_clock = lambda: NS(now=lambda: clock)
    for name in ('_cancel_dance_entry', '_update_dance_entry', 'start_motion_transition',
                 'joy_callback', 'send_to_motor'):
        setattr(node, name, MethodType(getattr(BxiExample, name), node))
    return node


def request_dance(node, speed=.3):
    command = MotionCommands()
    command.btn_10 = 1
    command.vel_des.x = speed
    node.joy_callback(command)


def update(node, *, moving=False, quat=None):
    return node._update_dance_entry(
        np.full(29, 2. if moving else 0.),
        np.array([1., 0., 0., 0.] if quat is None else quat),
        np.zeros(3), np.array([.3, .2, .1]),
    )


def test_B_queues_without_blocking_inference_and_stops_before_blending(controller):
    node = controller
    request_dance(node)
    node.dance_mimic.reset.assert_not_called()
    node.switch_to_motion.assert_not_called()
    commands = [update(node, moving=True)[0] for _ in range(30)]
    assert commands[0] < .3 and commands[0] > .29
    assert np.all(np.diff(commands) <= 0.) and commands[-1] == 0.
    assert node.motion_type == motionType.amp_walk
    for _ in range(5):
        update(node)
    assert node.motion_type == motionType.amp_walk
    update(node, moving=True)  # Stability must be continuous.
    for _ in range(6):
        update(node)
    assert node.motion_type == motionType.dance_mimic
    node.dance_mimic.reset.assert_called_once_with()
    assert node._dance_hold_frame and node._transition_smooth
    assert node.transition_total_steps == 30
    np.testing.assert_array_equal(update(node), np.zeros(3))


@pytest.mark.parametrize('quat', [[0., 0., 0., 0.], [np.nan, 0., 0., 0.],
                                  [np.cos(.2), np.sin(.2), 0., 0.]])
def test_invalid_or_tilted_robot_never_enters_dance_and_times_out(controller, quat):
    node = controller
    request_dance(node)
    for _ in range(151):
        update(node, quat=quat)
    assert node._dance_prepare is None
    assert node.motion_type == motionType.amp_walk
    node.dance_mimic.reset.assert_not_called()
    node.get_logger().warning.assert_called_once()


def test_A_cancels_pending_B_without_a_delayed_dance_start(controller):
    node = controller
    request_dance(node)
    msg = MotionCommands()
    msg.btn_5 = 1
    node.joy_callback(msg)
    for _ in range(100):
        update(node)
    assert node.motion_type == motionType.amp_walk
    assert node._dance_prepare is None
    node.dance_mimic.reset.assert_not_called()


def test_smooth_blend_has_small_first_step_and_reaches_new_target(controller):
    node = controller
    node.motion_type = motionType.dance_mimic
    node.start_motion_transition(motionType.amp_walk, transition_time=.6, smooth=True)
    node._old_action = (np.zeros(29), np.ones(29), np.ones(29))
    weights = []
    for i in range(node.transition_total_steps):
        node.transition_step_count = i
        node._blend_pending = True
        node.send_to_motor(np.ones(29), np.ones(29), np.ones(29))
        weights.append(node.act_pub.publish.call_args[0][0].pos[0])
    assert 0. < weights[0] < .004
    assert np.all(np.diff(weights) > 0.)
    assert weights[-1] == 1.
    node.start_motion_transition(motionType.amp_walk)  # Other switches stay linear.
    node._old_action = (np.zeros(29), np.ones(29), np.ones(29))
    node._blend_pending = True
    node.send_to_motor(np.ones(29), np.ones(29), np.ones(29))
    assert node.act_pub.publish.call_args[0][0].pos[0] == pytest.approx(.05)


def test_dance_replay_holds_first_frame_until_takeover(controller):
    node = controller
    node.motion_type = motionType.dance_mimic
    node._dance_hold_frame = True
    node.send_to_motor = Mock()
    BxiExample._run_motion_dispatch(node, np.zeros(29), np.zeros(29),
                                    np.array([1., 0., 0., 0.]), np.zeros(3), np.zeros(3))
    assert node.dance_mimic.timestep == 1
    node._dance_hold_frame = False
    BxiExample._run_motion_dispatch(node, np.zeros(29), np.zeros(29),
                                    np.array([1., 0., 0., 0.]), np.zeros(3), np.zeros(3))
    assert node.dance_mimic.timestep == 2


def test_blend_feeds_executed_target_back_to_both_policies(controller):
    node = controller
    node.motion_type = motionType.dance_mimic
    node.start_motion_transition(motionType.amp_walk, transition_time=.6, smooth=True)
    node._dance_hold_frame = True
    node.amp_walk.default_dof_pos = np.linspace(-.3, .3, 29)
    node.amp_walk.mujoco_to_isaac_idx = np.arange(28, -1, -1)
    node.amp_walk.action_scale = np.linspace(.1, .4, 29)
    node._old_action = (np.zeros(29), np.ones(29), np.ones(29))
    node._blend_pending = True
    node.send_to_motor(np.ones(29), np.ones(29), np.ones(29))
    executed = np.array(node.act_pub.publish.call_args[0][0].pos)
    node.dance_mimic.set_previous_target.assert_called_once()
    np.testing.assert_allclose(node.dance_mimic.set_previous_target.call_args[0][0], executed)
    reconstructed = (node.amp_walk.last_action * node.amp_walk.action_scale)[::-1]
    reconstructed += node.amp_walk.default_dof_pos
    np.testing.assert_allclose(reconstructed, executed, atol=1e-7)


def test_reentering_real_dance_policy_matches_fresh_initialization():
    root = Path(__file__).resolve().parents[1] / 'policy/dance_isaaclab'
    policy = DanceMotionPolicyGravityIsaaclabV3(
        root / 'old_face2.npz', root / 'old_face2_v1_6k.onnx', start_frame=1, fixed_pos=True,
    )
    args = (policy.default_dof_pos, np.zeros(29), np.array([1., 0., 0., 0.]), np.zeros(3))
    initial = policy.inference_step(*args)
    for _ in range(20):
        policy.inference_step(*args, advance=True)
    policy.reset()
    np.testing.assert_allclose(policy.inference_step(*args), initial, atol=1e-6)
    executed = policy.motioninputpos[1][policy.isaac_to_mujoco_idx]
    policy.set_previous_target(executed)
    reconstructed = (policy.default_joint_pos_isaac + policy.action_buffer * policy.action_scale)
    np.testing.assert_allclose(reconstructed[policy.isaac_to_mujoco_idx], executed, atol=1e-7)
