"""Causal SMPL22 pose -> UMR surface -> ELF3 reference, with persistent IK state."""
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import runtime, solver
from .sources import SurfaceSource, shaped_surface


class PicoSurfaceSource:
    """PICO contract is already Z-up world / SMPL local bone bases.

    All 22 tracked joint centres and orientations are preserved. Human segment
    lengths calibrate the neutral surface scale once.
    No BVH basis offsets or second world-axis conversion are applied.
    """

    def __init__(self, assets, *, use_tracked_positions=True):
        self.surface_model = shaped_surface(assets, np.zeros(10))
        self.ratio = None
        self._surface = SurfaceSource(50., 1, self.surface_model)
        self.use_tracked_positions = bool(use_tracked_positions)

    def transforms(self, positions, quaternions):
        positions = np.asarray(positions, dtype=np.float64)
        quaternion = np.asarray(quaternions, dtype=np.float64).copy()
        if positions.shape != (22, 3) or quaternion.shape != (22, 4):
            raise ValueError("Live UMR requires the SMPL22 PICO contract")
        if not np.isfinite(positions).all() or not np.isfinite(quaternion).all():
            raise ValueError("PICO pose contains non-finite transforms")
        norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.min(norms) < 1e-6:
            raise ValueError("PICO pose contains a zero quaternion")
        quaternion /= norms
        rest = self.surface_model["rest_joints"]
        if self.ratio is None:
            segments = [(4, 1), (7, 4), (5, 2), (8, 5)]
            lengths = np.asarray([np.linalg.norm(positions[c] - positions[p]) for c, p in segments])
            if np.min(lengths) < .08 or np.max(lengths) > .8:
                raise ValueError("PICO leg dimensions invalid; check tracker calibration and metre units")
            self.ratio = float(np.median(lengths / [np.linalg.norm(rest[c] - rest[p]) for c, p in segments]))
        world = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
        local = world.copy()
        for joint in range(1, 22):
            parent = int(self.surface_model["parents"][joint])
            local[joint] = world[parent].T @ world[joint]
        self._surface.poses = Rotation.from_matrix(local).as_rotvec().reshape(1, 66)
        self._surface.translations = (positions[0] / self.ratio - rest[0])[None]
        self._surface.joint_positions_w = (positions / self.ratio)[None] if self.use_tracked_positions else None
        return self._surface.frame(0)


class PicoFootContacts:
    """Tracked foot height/speed with hysteresis, independent of skin thickness.

    PICO's floor origin may be negative and its foot markers lie above the sole.
    Calibrate each foot's ankle/toe marker levels, then recognize support from
    low marker clearance and low velocity. Detect rising swing feet before a
    large height threshold, and require landing evidence before anchoring again.
    """

    def __init__(self):
        self.floor = None
        self.previous = None
        self.velocity = np.zeros((2, 2, 3))
        self.contacts = np.zeros(2, dtype=bool)
        self.enter_count = np.zeros(2, dtype=np.int32)
        self.leave_count = np.zeros(2, dtype=np.int32)
        self.height = np.zeros(2)
        self.speed = np.zeros(2)
        self.was_supported = np.zeros(2, dtype=bool)
        self.swing_peak = np.zeros(2)
        self.rising_speed = np.zeros(2)

    def update(self, positions, dt, ratio=1.):
        probes = np.asarray(positions, dtype=np.float64)[[[7, 10], [8, 11]]] / ratio
        if self.floor is None:
            self.floor = probes[:, :, 2].copy()
        if self.previous is not None:
            measured = (probes - self.previous) / max(dt, .005)
            alpha = -math.expm1(-dt / .05)
            self.velocity += alpha * (measured - self.velocity)
        self.previous = probes.copy()
        horizontal = np.linalg.norm(self.velocity[:, :, :2], axis=-1)
        self.speed = np.min(horizontal, axis=1)
        vertical = np.min(np.abs(self.velocity[:, :, 2]), axis=1)
        self.rising_speed = np.min(self.velocity[:, :, 2], axis=1)
        offsets = probes[:, :, 2] - self.floor
        shared_direction = np.all(offsets > .001) or np.all(offsets < -.001)
        # Only adapt a shared offset while both feet support and all four
        # markers move coherently and slowly. A unilateral minimum can come
        # from a heel/toe pivot or a tracking outlier, not a lower floor.
        common_drift = (self.contacts.all() and shared_direction and np.max(horizontal) < .1
                        and np.max(np.abs(self.velocity[:, :, 2])) < .12
                        and np.ptp(offsets) < .015)
        if common_drift:
            self.floor += -math.expm1(-dt / .25) * np.mean(offsets)
        self.height = np.min(probes[:, :, 2] - self.floor, axis=1)
        for side in range(2):
            if self.contacts[side]:
                # Both ankle and toe must rise: one stationary marker can
                # still represent heel/toe support during a foot pivot.
                lift = not common_drift and self.height[side] > .012 and self.rising_speed[side] > .06
                moving_lift = self.height[side] > .006 and self.speed[side] > .18
                release = (self.height[side] > .05 or lift or moving_lift
                           or vertical[side] > .3 or self.speed[side] > .35)
                self.leave_count[side] = self.leave_count[side] + 1 if release else 0
                if self.leave_count[side] >= 2:
                    self.contacts[side] = False
                    self.enter_count[side] = 0
                    self.swing_peak[side] = self.height[side]
            else:
                self.swing_peak[side] = max(self.swing_peak[side], self.height[side])
                landed = (not self.was_supported[side] or self.height[side] < .008
                          or self.swing_peak[side] - self.height[side] > .008)
                support = (landed and self.height[side] < .025 and self.speed[side] < .18
                           and vertical[side] < .15 and self.rising_speed[side] < .04)
                self.enter_count[side] = self.enter_count[side] + 1 if support else 0
                if self.enter_count[side] >= 2:
                    self.contacts[side] = True
                    self.was_supported[side] = True
                    self.leave_count[side] = 0
        return {"leftFoot": bool(self.contacts[0]), "rightFoot": bool(self.contacts[1])}


class LiveRetargeter:
    """UMR costs/constraints with a causal filter, no lookahead or file playback."""

    def __init__(self, robot_xml, metadata, filter_time_s=.035, *, use_tracked_positions=True,
                 use_foot_contacts=True):
        with np.load(runtime.ROOT / "elf3_surface.npz", allow_pickle=False) as data:
            self.assets = {key: data[key] for key in data.files}
        self.settings = json.loads((runtime.ROOT / "solver_config.json").read_text())
        self.model = runtime.load_robot(robot_xml)
        self.data = mujoco.MjData(self.model)
        qaddrs, _daddrs, _ranges, names = solver.scalar_qpos_joint_addrs(self.model)
        joint_names = metadata["joint_names"].split(",")
        if len(joint_names) != 29 or set(joint_names) != set(names):
            raise ValueError("RGMT joint contract does not match ELF3")
        self.columns = np.asarray([qaddrs[names.index(name)] for name in joint_names])
        self.anchor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, metadata["anchor_body_name"])
        if self.anchor < 0:
            raise ValueError("RGMT anchor does not match ELF3")
        self.filter_time_s = float(filter_time_s)
        self.use_tracked_positions = bool(use_tracked_positions)
        self.use_foot_contacts = bool(use_foot_contacts)
        if self.use_foot_contacts:
            # The original anchor row scale 5 is weaker than 30 foot point
            # targets at cost 50. Live tracked feet drift by centimetres; give
            # established support precedence over following marker jitter.
            self.settings["ground_contact_anchor_cost"] = 15.
        self.reset()

    def reset(self):
        self.source = PicoSurfaceSource(self.assets, use_tracked_positions=self.use_tracked_positions)
        self.solver = runtime.FrameRetargeter(self.source, self.model, self.assets, self.settings)
        self.foot_contacts = PicoFootContacts()
        self._previous = None
        self._filtered = None
        self._timestamp_ns = None

    def step(self, positions, rotations, timestamp_ns):
        timestamp_ns = int(timestamp_ns)
        if self._timestamp_ns is not None and timestamp_ns <= self._timestamp_ns:
            raise ValueError("PICO sample time must increase")
        if self._timestamp_ns is not None and timestamp_ns - self._timestamp_ns > 250_000_000:
            self.reset()  # Never bridge contact anchors across a tracking gap.
        transforms = self.source.transforms(positions, rotations)
        dt = .02 if self._timestamp_ns is None else (timestamp_ns - self._timestamp_ns) * 1e-9
        contacts = self.foot_contacts.update(positions, dt, self.source.ratio) if self.use_foot_contacts else None
        qpos, template, limits = self.solver.step(transforms, foot_contacts=contacts)
        if self._filtered is not None:
            alpha = 1. if self.filter_time_s <= 0 else -math.expm1(-dt / self.filter_time_s)
            result = self._filtered + alpha * (qpos - self._filtered)
            before = Rotation.from_quat(self._filtered[[4, 5, 6, 3]])
            after = Rotation.from_quat(qpos[[4, 5, 6, 3]])
            result[3:7] = (Rotation.from_rotvec(alpha * (after * before.inv()).as_rotvec()) * before).as_quat()[[3, 0, 1, 2]]
            qpos = result
        qpos = solver.clamp_joint_ranges(self.model, qpos, limits)
        runtime.enforce_floor(self.model, self.data, template, qpos, self.settings["ground_penetration_margin"])
        solver.set_qpos(self.model, self.data, qpos)
        position = qpos[self.columns].copy()
        quaternion = self.data.xquat[self.anchor].copy()
        if self._previous is None:
            velocity, angular = np.zeros(29), np.zeros(3)
        else:
            velocity = (position - self._previous[0]) / dt
            current = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
            previous = Rotation.from_quat(self._previous[1][[1, 2, 3, 0]])
            angular = (current * previous.inv()).as_rotvec() / dt
        self._previous = (position.copy(), quaternion.copy())
        self._filtered = qpos.copy()
        self._timestamp_ns = timestamp_ns
        return tuple(np.asarray(item, dtype=np.float32) for item in (position, velocity, quaternion, angular))
