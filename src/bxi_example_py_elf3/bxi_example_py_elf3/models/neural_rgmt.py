"""Compose the 20-frame Transformer retargeter with the live RGMT actor."""

from __future__ import annotations

from collections import deque
from threading import Condition, Thread
import time

import numpy as np

from .neural_retarget import (
    NeuralRetargeter,
    TransformerPrediction,
    blend_prediction_overlap,
    matrix_to_quaternion_wxyz,
    reconstruct_current_joint_velocity,
    rotation_6d_to_matrix,
)
from .rgmt import MUJOCO_TO_ISAAC_INDEX, RgmtExternalReferencePolicy


class NeuralRetargetRGMT:
    """Predict an 11-frame robot horizon, then feed a 21-frame RGMT window.

    The Transformer emits the current frame plus ten future frames.  RGMT
    needs ten past, current, and ten future frames, so this adapter keeps the
    ten already-consumed Transformer center frames in a causal history queue.
    The first samples are padded with the current prediction.
    """

    def __init__(self, retarget_onnx_path: str, rgmt_onnx_path: str, **kwargs) -> None:
        # These names were accepted by older launches, but arm target and
        # reference filtering is intentionally removed from the live path.
        # Consume them here so stale launch parameters are not forwarded to
        # the RGMT actor.
        for name in (
            "arm_target_slew_limit",
            "arm_target_velocity",
            "arm_target_acceleration",
            "arm_filter_enabled",
            "arm_reference_filter_enabled",
            "output_joint_order",
            "robot_joint_order",
        ):
            kwargs.pop(name, None)
        self.retargeter = NeuralRetargeter(
            retarget_onnx_path,
            # Two threads have a tighter tail latency on the deployment CPU.
            # Four threads occasionally cross the 20 ms control period and
            # make the async result arrive one tick late.
            intra_op_threads=kwargs.pop("retarget_intra_op_threads", 2),
        )
        self.overlap_blend = float(kwargs.pop("overlap_blend", 0.5))
        if not 0.0 <= self.overlap_blend <= 1.0:
            raise ValueError("overlap_blend must be inside [0,1]")
        reference_joint_order = kwargs.pop("reference_joint_order", "isaac")
        if reference_joint_order != "isaac":
            raise ValueError(
                "NeuralRetargetRGMT always feeds RGMT in IsaacLab joint order"
            )
        self._async_inference = bool(kwargs.pop("async_inference", True))
        self.rgmt = RgmtExternalReferencePolicy.for_live_reference(
            rgmt_onnx_path,
            reference_yaw_mode=kwargs.pop("reference_yaw_mode", "initial"),
            reference_joint_order="isaac",
            # The RGMT actor is small. One thread avoids contending with the
            # two-thread Transformer inside the same 20 ms control period.
            intra_op_num_threads=kwargs.pop("intra_op_num_threads", 1),
            **kwargs,
        )
        self._center_history: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=10
        )
        self._last_position: np.ndarray | None = None
        self._last_velocity: np.ndarray | None = None
        self._last_prediction: TransformerPrediction | None = None
        self._last_target: np.ndarray | None = None
        self.last_raw_target = None
        self._last_source_time = None
        self._submitted_source_time = None
        self._reference_window = None
        self._consumed_generation = -1
        self._sync_generation = 0
        self._worker_condition = Condition()
        self._worker_stop = False
        self._worker_epoch = 0
        self._worker_input_generation = 0
        self._worker_input = None
        self._worker_result = None
        self._worker_error = None
        self._worker_thread = None
        if self._async_inference:
            self._worker_thread = Thread(
                target=self._worker_loop,
                name="neural-retarget-inference",
                daemon=True,
            )
            self._worker_thread.start()

    def reset(self) -> None:
        self._center_history.clear()
        self._last_position = None
        self._last_velocity = None
        self._last_prediction = None
        self._last_target = None
        self._last_source_time = None
        self._submitted_source_time = None
        self._reference_window = None
        self._consumed_generation = -1
        self._sync_generation = 0
        if self._async_inference:
            with self._worker_condition:
                self._worker_epoch += 1
                self._worker_input = None
                self._worker_result = None
                self._worker_error = None
                self._worker_input_generation += 1
                self._worker_condition.notify_all()
        else:
            self.retargeter.reset()
        self.rgmt.reset()

    def _prepare_prediction(
        self,
        body_positions_history,
        body_rotations_history_wxyz,
        previous_prediction,
        previous_position,
        previous_velocity,
        source_dt_s=0.02,
    ):
        prediction = self.retargeter.infer(
            body_positions_history, body_rotations_history_wxyz
        )
        # Existing overlap helper shifts by exactly one 50-Hz source frame.
        # Applying it after skipped frames mixes different moments in time.
        if previous_prediction is not None and np.isclose(source_dt_s, 0.02, atol=0.002, rtol=0):
            prediction = blend_prediction_overlap(
                previous_prediction,
                prediction,
                alpha=self.overlap_blend,
            )
        prediction = reconstruct_current_joint_velocity(
            prediction,
            previous_position=previous_position,
            previous_velocity=previous_velocity,
            dt_s=source_dt_s,
        )
        return prediction

    def _worker_loop(self) -> None:
        previous_prediction = None
        previous_position = None
        previous_velocity = None
        previous_source_time = None
        local_epoch = 0
        processed_generation = 0
        while True:
            with self._worker_condition:
                while (
                    not self._worker_stop
                    and self._worker_input_generation <= processed_generation
                ):
                    self._worker_condition.wait()
                if self._worker_stop:
                    return
                if local_epoch != self._worker_epoch:
                    previous_prediction = None
                    previous_position = None
                    previous_velocity = None
                    previous_source_time = None
                    local_epoch = self._worker_epoch
                generation = self._worker_input_generation
                if self._worker_input is None:
                    processed_generation = generation
                    continue
                body_positions, body_rotations, source_time = self._worker_input

            try:
                prediction = self._prepare_prediction(
                    body_positions,
                    body_rotations,
                    previous_prediction,
                    previous_position,
                    previous_velocity,
                    0.02 if previous_source_time is None else source_time - previous_source_time,
                )
                error = None
            except Exception as exc:  # surface the error on the control thread
                prediction = None
                error = exc

            processed_generation = generation
            if prediction is not None:
                previous_prediction = prediction
                previous_position = prediction.dof_position[0].copy()
                previous_velocity = prediction.dof_velocity[0].copy()
                previous_source_time = source_time
            with self._worker_condition:
                if local_epoch == self._worker_epoch:
                    self._worker_result = (generation, prediction, time.monotonic())
                    self._worker_error = error
                    self._worker_condition.notify_all()

    def _submit_async_input(self, body_positions_history, body_rotations_history_wxyz, source_time):
        with self._worker_condition:
            if self._submitted_source_time is not None and source_time <= self._submitted_source_time:
                return
            self._submitted_source_time = source_time
            self._worker_input = (
                np.asarray(body_positions_history, dtype=np.float32).copy(),
                np.asarray(body_rotations_history_wxyz, dtype=np.float32).copy(),
                source_time,
            )
            self._worker_input_generation += 1
            self._worker_condition.notify_all()

    def _get_async_prediction(self):
        with self._worker_condition:
            if self._worker_error is not None:
                error = self._worker_error
                self._worker_error = None
                raise RuntimeError("neural retarget worker failed") from error
            if self._worker_result is None or time.monotonic() - self._worker_result[2] > 0.25:
                return None  # The controller keeps balancing during warmup.
            return self._worker_result

    def _prediction_window(self, prediction):
        # neural_retarget.onnx decodes the 29 joints in BXI/MuJoCo order,
        # while the live RGMT reference ABI is IsaacLab articulation order.
        joint_position_isaac = prediction.dof_position[..., MUJOCO_TO_ISAAC_INDEX]
        model_velocity_isaac = prediction.dof_velocity[..., MUJOCO_TO_ISAAC_INDEX]
        # Keep position and velocity fields consistent for the future horizon.
        joint_velocity_isaac = np.empty_like(joint_position_isaac)
        joint_velocity_isaac[0] = model_velocity_isaac[0]
        joint_velocity_isaac[1:-1] = (
            joint_position_isaac[2:] - joint_position_isaac[:-2]
        ) * np.float32(25.0)
        joint_velocity_isaac[-1] = (
            joint_position_isaac[-1] - joint_position_isaac[-2]
        ) * np.float32(50.0)
        matrices = rotation_6d_to_matrix(prediction.root_rotation_6d)
        root_quat = matrix_to_quaternion_wxyz(matrices)
        root_ang_vel_w = (
            matrices @ prediction.root_angular_velocity_local[..., None]
        )[..., 0]
        current = (
            joint_position_isaac[0].copy(),
            joint_velocity_isaac[0].copy(),
            root_quat[0].copy(),
            root_ang_vel_w[0].copy(),
        )
        past = list(self._center_history)
        pad = past[0] if past else current
        past = ([pad] * (10 - len(past))) + past
        future = [
            (
                joint_position_isaac[index],
                joint_velocity_isaac[index],
                root_quat[index],
                root_ang_vel_w[index],
            )
            for index in range(11)
        ]
        self._center_history.append(current)
        return past + future

    def inference_standing_step(self, q, dq, quat, omega) -> np.ndarray:
        """Run RGMT on a static standing reference while PICO is unavailable."""
        # RGMT live references are supplied in IsaacLab order, while its
        # metadata default pose is stored in MuJoCo/BXI order.
        stand_pos = self.rgmt.default_dof_pos[MUJOCO_TO_ISAAC_INDEX].copy()
        stand_vel = np.zeros_like(stand_pos)
        stand_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
        stand_ang_vel = np.zeros(3, dtype=np.float32)
        target = self.rgmt.inference_step_with_reference_window(
            q,
            dq,
            quat,
            omega,
            reference_joint_pos_window=np.repeat(stand_pos[None], 21, axis=0),
            reference_joint_vel_window=np.repeat(stand_vel[None], 21, axis=0),
            reference_root_quat_window_w=np.repeat(stand_quat[None], 21, axis=0),
            reference_root_ang_vel_window_w=np.repeat(stand_ang_vel[None], 21, axis=0),
        )
        return target

    def inference_step(
        self,
        q,
        dq,
        quat,
        omega,
        body_positions_history,
        body_rotations_history_wxyz,
        *,
        source_time_s=None,
    ) -> np.ndarray:
        self._sync_generation += 1
        source_time = self._sync_generation * 0.02 if source_time_s is None else float(source_time_s)
        if not np.isfinite(source_time):
            raise ValueError("source_time_s must be finite")
        if self._async_inference:
            self._submit_async_input(body_positions_history, body_rotations_history_wxyz, source_time)
            result = self._get_async_prediction()
            if result is None:
                return None
            generation, prediction, _ = result
            if generation != self._consumed_generation:
                self._reference_window = self._prediction_window(prediction)
                self._consumed_generation = generation
            window = self._reference_window
        else:
            prediction = self._prepare_prediction(
                body_positions_history,
                body_rotations_history_wxyz,
                self._last_prediction,
                self._last_position,
                self._last_velocity,
                0.02 if self._last_source_time is None else source_time - self._last_source_time,
            )
            self._last_prediction = prediction
            self._last_position = prediction.dof_position[0].copy()
            self._last_velocity = prediction.dof_velocity[0].copy()
            self._last_source_time = source_time
            window = self._prediction_window(prediction)
        target = self.rgmt.inference_step_with_reference_window(
            q,
            dq,
            quat,
            omega,
            reference_joint_pos_window=np.stack([item[0] for item in window]),
            reference_joint_vel_window=np.stack([item[1] for item in window]),
            reference_root_quat_window_w=np.stack([item[2] for item in window]),
            reference_root_ang_vel_window_w=np.stack([item[3] for item in window]),
        )
        self.last_raw_target = np.asarray(target, dtype=np.float32).copy()
        self._last_target = target.copy()
        return target

    def close(self) -> None:
        if self._async_inference:
            with self._worker_condition:
                self._worker_stop = True
                self._worker_condition.notify_all()
            if self._worker_thread is not None:
                self._worker_thread.join(timeout=1.0)
        self.retargeter.close()
        session = getattr(self.rgmt, "session", None)
        if session is not None:
            self.rgmt.session = None


__all__ = ["NeuralRetargetRGMT"]
