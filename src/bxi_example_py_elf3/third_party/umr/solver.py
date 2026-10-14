"""ELF3 surface-vector solver extracted from UMR (MIT, see LICENSE).

Training, mesh binding, viewers and other robot/source types are excluded.
The original least-squares costs and ground/step constraints are preserved;
SciPy SLSQP replaces the Clarabel backend to use the existing environment.
"""
from __future__ import annotations
import sys
import mujoco
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from .qp import solve_qp_step as solve_clarabel_qp_step
common = sys.modules[__name__]

def template_points_to_world(data, template, point_ids):
    point_ids = np.asarray(point_ids, dtype=np.int32)
    rotation = data.geom_xmat[template["geom_ids"][point_ids]].reshape(-1, 3, 3)
    return np.einsum("nij,nj->ni", rotation, template["local_pos"][point_ids]) + data.geom_xpos[template["geom_ids"][point_ids]]


def template_points_world_z(data, template, point_ids):
    point_ids = np.asarray(point_ids, dtype=np.int32)
    geom_ids = template["geom_ids"][point_ids]
    rotation_z = data.geom_xmat[geom_ids].reshape(-1, 3, 3)[:, 2]
    return np.einsum("ni,ni->n", rotation_z, template["local_pos"][point_ids]) + data.geom_xpos[geom_ids, 2]


class TemplateSlotKinematicsCache:
    """Batch point Jacobians from one spatial Jacobian per rigid body.

    J(point) = J(body origin) + omega x (point - body origin).
    The 456 surface samples share only about 30 bodies. Computing the body
    Jacobians once preserves the exact MuJoCo derivatives without per-point
    Python/C calls or dictionary lookups in the solver's inner loop.
    """

    def __init__(self, model, data, template):
        self.model, self.data, self.template = model, data, template
        self._body_ids = model.geom_bodyid[template["geom_ids"]]
        self._points = template_points_to_world(data, template, np.arange(len(self._body_ids)))
        rotation = data.geom_xmat[template["geom_ids"]].reshape(-1, 3, 3)
        self._normals = np.einsum("nij,nj->ni", rotation, template["local_normals"])
        self._normals /= np.maximum(np.linalg.norm(self._normals, axis=1, keepdims=True), 1e-12)
        self._body_jp = np.empty((model.nbody, 3, model.nv))
        self._body_jr = np.empty_like(self._body_jp)
        self._body_ready = np.zeros(model.nbody, dtype=bool)

    def points(self, point_ids):
        return self._points[point_ids]

    def point(self, slot_id):
        return self._points[slot_id]

    def points_z(self, point_ids):
        return self._points[point_ids, 2]

    def point_z(self, slot_id):
        return float(self._points[slot_id, 2])

    def normals(self, point_ids):
        return self._normals[point_ids]

    def normal(self, slot_id):
        return self._normals[slot_id]

    def rotation_jacobians(self, point_ids):
        bodies = self._body_ids[point_ids]
        for body in np.unique(bodies):
            if not self._body_ready[body]:
                mujoco.mj_jacBody(self.model, self.data, self._body_jp[body], self._body_jr[body], int(body))
                self._body_ready[body] = True
        return self._body_jr[bodies]

    def point_jacobians(self, point_ids):
        point_ids = np.asarray(point_ids, dtype=np.int32)
        jacr = self.rotation_jacobians(point_ids)
        bodies = self._body_ids[point_ids]
        offset = self._points[point_ids] - self.data.xpos[bodies]
        return self._body_jp[bodies] + np.swapaxes(np.cross(np.swapaxes(jacr, 1, 2), offset[:, None, :]), 1, 2)

    def normal_jacobians(self, point_ids):
        point_ids = np.asarray(point_ids, dtype=np.int32)
        jacr = self.rotation_jacobians(point_ids)
        return np.swapaxes(np.cross(np.swapaxes(jacr, 1, 2), self._normals[point_ids, None, :]), 1, 2)

    def point_rotation_jacobians(self, slot_id):
        ids = np.asarray([slot_id], dtype=np.int32)
        return self.point_jacobians(ids)[0], self.rotation_jacobians(ids)[0]

    def point_jacobian(self, slot_id):
        return self.point_jacobians(np.asarray([slot_id], dtype=np.int32))[0]

    def normal_jacobian(self, slot_id):
        return self.normal_jacobians(np.asarray([slot_id], dtype=np.int32))[0]

def set_qpos(model, data, qpos):
    data.qpos[:] = qpos
    # IK needs geometry and cdof, not collision, dynamics or sensor evaluation.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

def scalar_qpos_joint_addrs(model):
    addrs = []
    vaddrs = []
    ranges = []
    names = []
    for joint_id in range(model.njnt):
        jtype = int(model.jnt_type[joint_id])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        addrs.append(int(model.jnt_qposadr[joint_id]))
        vaddrs.append(int(model.jnt_dofadr[joint_id]))
        ranges.append(model.jnt_range[joint_id].copy() if int(model.jnt_limited[joint_id]) else np.asarray([-np.inf, np.inf]))
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}")
    return np.asarray(addrs, dtype=np.int32), np.asarray(vaddrs, dtype=np.int32), np.asarray(ranges), names


class ScalarJointLimits(dict):
    """The usual named limit records plus arrays for batch clipping."""

    def __init__(self, records):
        super().__init__(records)
        self.qpos_addrs = np.asarray(list(records), dtype=np.int32)
        self.lower = np.asarray([record["lower"] for record in records.values()])
        self.upper = np.asarray([record["upper"] for record in records.values()])

def build_scalar_joint_limits(model, configured_limits=None):
    configured_limits = {} if configured_limits is None else dict(configured_limits)
    by_qpos = {}
    matched = []
    skipped_no_overlap = []
    for joint_id in range(model.njnt):
        jtype = int(model.jnt_type[joint_id])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        if int(model.jnt_limited[joint_id]):
            lower, upper = (float(v) for v in model.jnt_range[joint_id])
        else:
            lower, upper = -np.inf, np.inf
        source = "model"
        if name in configured_limits:
            cfg_lower, cfg_upper = configured_limits[name]
            new_lower = max(lower, float(cfg_lower))
            new_upper = min(upper, float(cfg_upper))
            if new_lower <= new_upper:
                lower, upper = new_lower, new_upper
                source = "model+config"
                matched.append(name)
            else:
                skipped_no_overlap.append((name, lower, upper, cfg_lower, cfg_upper))
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        by_qpos[qadr] = {
            "joint_id": int(joint_id),
            "dof_id": int(dadr),
            "name": str(name),
            "lower": float(lower),
            "upper": float(upper),
            "source": source,
        }
    if skipped_no_overlap:
        preview = ", ".join(name for name, *_ in skipped_no_overlap[:8])
        print(
            f"[SurfaceRetarget][JointLimit][WARN] skipped {len(skipped_no_overlap)} configured limits "
            f"with no overlap against model XML: {preview}"
        )
    return ScalarJointLimits(by_qpos), matched

def clamp_joint_ranges(model, qpos, joint_limits_by_qpos=None):
    qpos = qpos.copy()
    if isinstance(joint_limits_by_qpos, ScalarJointLimits):
        limits = joint_limits_by_qpos
        qpos[limits.qpos_addrs] = np.clip(qpos[limits.qpos_addrs], limits.lower, limits.upper)
        qpos[3:7] /= max(float(np.linalg.norm(qpos[3:7])), 1e-12)
        return qpos
    for joint_id in range(model.njnt):
        jtype = int(model.jnt_type[joint_id])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        adr = int(model.jnt_qposadr[joint_id])
        info = None if joint_limits_by_qpos is None else joint_limits_by_qpos.get(adr)
        if info is not None:
            lo, hi = float(info["lower"]), float(info["upper"])
            if np.isfinite(lo) or np.isfinite(hi):
                qpos[adr] = np.clip(qpos[adr], lo, hi)
        elif int(model.jnt_limited[joint_id]):
            lo, hi = model.jnt_range[joint_id]
            qpos[adr] = np.clip(qpos[adr], lo, hi)
    qpos[3:7] /= max(float(np.linalg.norm(qpos[3:7])), 1e-12)
    return qpos

def _finite_difference_matrix(length: int, order: int):
    length = int(length)
    order = int(order)
    if order <= 0:
        return sparse.eye(length, dtype=np.float64, format="csc")
    rows = length - order
    if rows <= 0:
        return sparse.csc_matrix((0, length), dtype=np.float64)
    if order == 1:
        coeffs = (-1.0, 1.0)
    elif order == 2:
        coeffs = (1.0, -2.0, 1.0)
    elif order == 3:
        coeffs = (-1.0, 3.0, -3.0, 1.0)
    else:
        raise ValueError(f"Unsupported finite-difference order {order}")
    diagonals = [np.full(rows, coeff, dtype=np.float64) for coeff in coeffs]
    return sparse.diags(diagonals, offsets=np.arange(order + 1), shape=(rows, length), format="csc")

def lqr_smooth_qpos_sequence(
    model,
    qpos_seq,
    qpos_columns,
    joint_limits_by_qpos=None,
    data_cost=1.0,
    velocity_cost=0.0,
    acceleration_cost=0.0,
    jerk_cost=0.0,
    include_root_translation=False,
    anchor_start_frames=0,
    anchor_end_frames=0,
):
    """LQR-style fixed-interval smoother for retargeted qpos trajectories."""
    qpos = np.asarray(qpos_seq, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"qpos_seq must be 2D, got {qpos.shape}")
    filtered = qpos.copy()
    frames = int(filtered.shape[0])
    if frames <= 1:
        return filtered

    data_cost = max(float(data_cost), 1e-12)
    velocity_cost = max(float(velocity_cost), 0.0)
    acceleration_cost = max(float(acceleration_cost), 0.0)
    jerk_cost = max(float(jerk_cost), 0.0)
    if velocity_cost <= 0.0 and acceleration_cost <= 0.0 and jerk_cost <= 0.0:
        return filtered

    columns = []
    if include_root_translation:
        columns.extend([0, 1, 2])
    columns.extend(int(col) for col in qpos_columns)
    columns = sorted({col for col in columns if 0 <= col < filtered.shape[1]})
    if not columns:
        return filtered

    system = data_cost * sparse.eye(frames, dtype=np.float64, format="csc")
    if velocity_cost > 0.0 and frames > 1:
        diff = _finite_difference_matrix(frames, 1)
        system = system + velocity_cost * (diff.T @ diff)
    if acceleration_cost > 0.0 and frames > 2:
        diff = _finite_difference_matrix(frames, 2)
        system = system + acceleration_cost * (diff.T @ diff)
    if jerk_cost > 0.0 and frames > 3:
        diff = _finite_difference_matrix(frames, 3)
        system = system + jerk_cost * (diff.T @ diff)

    rhs = data_cost * filtered[:, columns]
    anchor_start_frames = max(int(anchor_start_frames), 0)
    anchor_end_frames = max(int(anchor_end_frames), 0)
    fixed = []
    if anchor_start_frames > 0:
        fixed.extend(range(min(anchor_start_frames, frames)))
    if anchor_end_frames > 0:
        start = max(frames - anchor_end_frames, 0)
        fixed.extend(range(start, frames))
    fixed = np.asarray(sorted(set(fixed)), dtype=np.int32)
    if fixed.size:
        free_mask = np.ones(frames, dtype=bool)
        free_mask[fixed] = False
        free = np.flatnonzero(free_mask).astype(np.int32)
        filtered[fixed[:, None], columns] = qpos[fixed[:, None], columns]
        if free.size:
            system_csc = system.tocsc()
            rhs_free = rhs[free] - system_csc[free][:, fixed] @ qpos[fixed[:, None], columns]
            solved = sparse_linalg.spsolve(system_csc[free][:, free], rhs_free)
            solved = np.asarray(solved, dtype=np.float64)
            if solved.ndim == 1:
                solved = solved.reshape(free.size, 1)
            filtered[free[:, None], columns] = solved
    else:
        solved = sparse_linalg.spsolve(system.tocsc(), rhs)
        solved = np.asarray(solved, dtype=np.float64)
        if solved.ndim == 1:
            solved = solved.reshape(frames, 1)
        filtered[:, columns] = solved
    for frame_idx in range(frames):
        filtered[frame_idx] = clamp_joint_ranges(model, filtered[frame_idx], joint_limits_by_qpos=joint_limits_by_qpos)
    return filtered

def root_qvel_dof_groups(model):
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            dof_start = int(model.jnt_dofadr[joint_id])
            return (
                np.arange(dof_start, dof_start + 3, dtype=np.int32),
                np.arange(dof_start + 3, dof_start + 6, dtype=np.int32),
            )
    empty = np.zeros(0, dtype=np.int32)
    return empty, empty

def local_pose_qvel_dof_ids(model):
    root_translation, root_rotation = root_qvel_dof_groups(model)
    root_dofs = np.concatenate([root_translation, root_rotation])
    if root_dofs.size == 0:
        return np.arange(model.nv, dtype=np.int32)
    mask = np.ones(model.nv, dtype=bool)
    mask[root_dofs] = False
    return np.flatnonzero(mask).astype(np.int32)

def step_limit_flags(mode, allow_off=False, label="step_limit_mode"):
    normalized = str(mode).lower()
    if normalized in {"off", "none", "unlimited"}:
        if allow_off:
            return False, False
        raise ValueError(f"Unsupported {label}={mode!r}")
    if normalized not in {"box", "l2", "box_l2"}:
        raise ValueError(f"Unsupported {label}={mode!r}")
    return normalized in {"box", "box_l2"}, normalized in {"l2", "box_l2"}

def apply_qvel_box_limit(lower, upper, dof_ids, max_dq):
    dof_ids = np.asarray(dof_ids, dtype=np.int32).reshape(-1)
    if dof_ids.size == 0:
        return
    max_dq = abs(float(max_dq))
    if not np.isfinite(max_dq):
        return
    lower[dof_ids] = np.maximum(lower[dof_ids], -max_dq)
    upper[dof_ids] = np.minimum(upper[dof_ids], max_dq)

def apply_dof_max_dq_box(lower, upper, dof_max_dq_box):
    if not dof_max_dq_box:
        return
    for info in dof_max_dq_box.values():
        dof_id = int(info["dof_id"])
        max_dq = abs(float(info["max_dq"]))
        lower[dof_id] = -max_dq
        upper[dof_id] = max_dq

def qvel_step_bounds(
    model,
    qpos,
    max_dq,
    joint_limits_by_qpos=None,
    use_step_box=True,
    limited_dof_ids=None,
    dof_max_dq_box=None,
):
    max_dq = abs(float(max_dq))
    lower = np.full(model.nv, -np.inf, dtype=np.float64)
    upper = np.full(model.nv, np.inf, dtype=np.float64)
    if use_step_box:
        if limited_dof_ids is None:
            limited_dof_ids = np.arange(model.nv, dtype=np.int32)
        apply_qvel_box_limit(lower, upper, limited_dof_ids, max_dq)
    apply_dof_max_dq_box(lower, upper, dof_max_dq_box)
    if joint_limits_by_qpos is None:
        return lower, upper
    qpos = np.asarray(qpos, dtype=np.float64)
    for info in joint_limits_by_qpos.values():
        dadr = int(info["dof_id"])
        qadr = int(model.jnt_qposadr[int(info["joint_id"])])
        lo = float(info["lower"])
        hi = float(info["upper"])
        if not (np.isfinite(lo) or np.isfinite(hi)):
            continue
        joint_lower = lo - float(qpos[qadr]) if np.isfinite(lo) else -np.inf
        joint_upper = hi - float(qpos[qadr]) if np.isfinite(hi) else np.inf
        lower[dadr] = max(lower[dadr], joint_lower)
        upper[dadr] = min(upper[dadr], joint_upper)
        if lower[dadr] > upper[dadr]:
            # If the current qpos is already outside the limit by more than max_dq,
            # prioritize moving toward the feasible interval over the nominal step cap.
            if not use_step_box:
                center = 0.5 * (joint_lower + joint_upper)
                lower[dadr] = center
                upper[dadr] = center
            elif float(qpos[qadr]) < lo:
                lower[dadr] = upper[dadr] = min(joint_lower, max_dq)
            elif float(qpos[qadr]) > hi:
                lower[dadr] = upper[dadr] = max(joint_upper, -max_dq)
    return lower, upper

def ground_contact_active_slots(target_distances, threshold, max_points, rank_distances=None, candidate_slot_ids=None):
    target_distances = np.asarray(target_distances, dtype=np.float64).reshape(-1)
    rank_distances = target_distances if rank_distances is None else np.asarray(rank_distances, dtype=np.float64).reshape(-1)
    if len(rank_distances) != len(target_distances):
        raise ValueError(f'Ground contact rank count mismatch: rank={len(rank_distances)}, target={len(target_distances)}')
    if candidate_slot_ids is None:
        candidate = np.arange(len(target_distances), dtype=np.int32)
    else:
        candidate = np.asarray(candidate_slot_ids, dtype=np.int32).reshape(-1)
        candidate = candidate[(candidate >= 0) & (candidate < len(target_distances))]
    active = candidate[target_distances[candidate] <= float(threshold)].astype(np.int32)
    if active.size > 0 and int(max_points) > 0 and (active.size > int(max_points)):
        active = active[np.argsort(rank_distances[active])[:int(max_points)]]
    return active

def ground_contact_anchor_slots(target_distances, max_points, rank_distances=None, candidate_slot_ids=None):
    target_distances = np.asarray(target_distances, dtype=np.float64).reshape(-1)
    rank_distances = target_distances if rank_distances is None else np.asarray(rank_distances, dtype=np.float64).reshape(-1)
    if len(rank_distances) != len(target_distances):
        raise ValueError(f'Ground contact anchor rank count mismatch: rank={len(rank_distances)}, target={len(target_distances)}')
    if candidate_slot_ids is None:
        candidate = np.arange(len(target_distances), dtype=np.int32)
    else:
        candidate = np.asarray(candidate_slot_ids, dtype=np.int32).reshape(-1)
        candidate = candidate[(candidate >= 0) & (candidate < len(target_distances))]
    active = candidate[np.isclose(target_distances[candidate], 0.0, atol=1e-08)].astype(np.int32)
    if active.size > 0 and int(max_points) > 0 and (active.size > int(max_points)):
        active = active[np.argsort(rank_distances[active])[:int(max_points)]]
    return active

def ground_penetration_constraint_slots(points, slot_ids, max_points, threshold, margin=0.0):
    slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
    if slot_ids.size == 0:
        return (slot_ids, np.zeros(0, dtype=np.float64))
    values = np.asarray(points, dtype=np.float64)
    z = values.reshape(-1) if values.ndim == 1 else values[:, 2]
    if z.size != slot_ids.size:
        raise ValueError(f'Ground penetration z count {z.size} does not match slot count {slot_ids.size}')
    threshold = float(threshold)
    if threshold >= 0.0:
        active = z < float(margin) + threshold
        slot_ids = slot_ids[active]
        z = z[active]
    if slot_ids.size == 0:
        return (slot_ids.astype(np.int32), z.astype(np.float64, copy=False))
    max_points = int(max_points)
    if max_points > 0 and slot_ids.size > max_points:
        chosen = np.argpartition(z, max_points - 1)[:max_points]
        chosen = chosen[np.lexsort((slot_ids[chosen], z[chosen]))]
        slot_ids = slot_ids[chosen]
        z = z[chosen]
    return (slot_ids.astype(np.int32), z.astype(np.float64, copy=False))

def merge_self_contact_maps(map_groups):
    if not map_groups:
        return None
    frame_count = len(map_groups[0])
    merged = []
    for frame_idx in range(frame_count):
        pair_data = {}
        for maps in map_groups:
            item = maps[frame_idx]
            pair_slot_ids = np.asarray(item['slot_pairs'], dtype=np.int32).reshape(-1, 2)
            distances = np.asarray(item['distances'], dtype=np.float32).reshape(-1)
            weights = np.asarray(item.get('weights', np.ones(len(distances), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if len(pair_slot_ids) != len(distances) or len(weights) != len(distances):
                raise ValueError(f'Self-contact merge count mismatch at frame {frame_idx}: pairs={len(pair_slot_ids)} distances={len(distances)} weights={len(weights)}')
            for (pair, distance, weight) in zip(pair_slot_ids, distances, weights):
                key = tuple(sorted((int(pair[0]), int(pair[1]))))
                existing = pair_data.get(key)
                if existing is None or float(weight) > float(existing[1]):
                    pair_data[key] = (float(distance), float(weight))
        if pair_data:
            pair_ids = np.asarray(list(pair_data.keys()), dtype=np.int32)
            distances = np.asarray([item[0] for item in pair_data.values()], dtype=np.float32)
            weights = np.asarray([item[1] for item in pair_data.values()], dtype=np.float32)
        else:
            pair_ids = np.zeros((0, 2), dtype=np.int32)
            distances = np.zeros(0, dtype=np.float32)
            weights = np.zeros(0, dtype=np.float32)
        merged.append({'slot_pairs': pair_ids, 'distances': distances, 'weights': weights})
    return merged

def update_ground_contact_anchors(model, data, qpos, robot_template, active_slot_ids, anchor_state):
    active_slot_ids = np.asarray(active_slot_ids, dtype=np.int32).reshape(-1)
    active_set = {int(slot_id) for slot_id in active_slot_ids}
    for slot_id in list(anchor_state):
        if int(slot_id) not in active_set:
            del anchor_state[int(slot_id)]
    new_slot_ids = np.asarray([int(slot_id) for slot_id in active_slot_ids if int(slot_id) not in anchor_state], dtype=np.int32)
    if new_slot_ids.size == 0:
        return
    common.set_qpos(model, data, qpos)
    new_points = common.template_points_to_world(data, robot_template, new_slot_ids)
    for (slot_id, point) in zip(new_slot_ids, new_points):
        target = np.asarray([point[0], point[1], 0.0], dtype=np.float64)
        anchor_state[int(slot_id)] = target

def solve_frame_body_segment_qp(model, data, qpos_init, qpos_prev, qpos_prev2, source_slots, source_slot_normals, source_surface_normal_targets, selected_slot_ids, source_slot_part_ids, surface_point_slot_costs, surface_normal_slot_costs, source_self_contact_map, source_ground_contact_distances, source_ground_contact_weight_distances, object_contact_frame, robot_template, joint_qpos_addrs, joint_dof_addrs, args, iters=None, joint_limits_by_qpos=None, robot_self_penetration_cache=None, ground_penetration_collision_cache=None, robot_object_penetration_cache=None, ground_contact_anchor_state=None):
    qpos = qpos_init.copy()
    costs = []
    sqrt_smooth = np.sqrt(float(args.smooth_cost))
    sqrt_temporal_smooth = np.sqrt(float(args.temporal_smooth_cost))
    ground_contact_threshold = float(args.ground_contact_map_threshold)
    ground_contact_max_points = int(args.ground_contact_map_max_points)
    self_contact_cost = float(args.self_contact_map_cost)
    object_contact_threshold = float(args.object_contact_map_threshold)
    object_contact_max_points = int(args.object_contact_map_max_points)
    if ground_contact_anchor_state is None:
        ground_contact_anchor_state = {}
    target_distances_for_contact = None
    weight_distances_for_contact = None
    if source_ground_contact_distances is not None:
        target_distances_for_contact = np.asarray(source_ground_contact_distances, dtype=np.float64).reshape(-1)
        weight_distances_for_contact = target_distances_for_contact if source_ground_contact_weight_distances is None else np.asarray(source_ground_contact_weight_distances, dtype=np.float64).reshape(-1)
        if len(target_distances_for_contact) != len(robot_template['geom_ids']):
            raise ValueError(f"Ground contact slot count mismatch: target={len(target_distances_for_contact)}, robot={len(robot_template['geom_ids'])}")
        if len(weight_distances_for_contact) != len(target_distances_for_contact):
            raise ValueError(f'Ground contact weight count mismatch: weights={len(weight_distances_for_contact)}, target={len(target_distances_for_contact)}')
    anchor_active = np.zeros(0, dtype=np.int32)
    if float(args.ground_contact_anchor_cost) > 0.0 and target_distances_for_contact is not None:
        anchor_active = ground_contact_anchor_slots(target_distances_for_contact, ground_contact_max_points, rank_distances=weight_distances_for_contact, candidate_slot_ids=selected_slot_ids)
    update_ground_contact_anchors(model, data, qpos, robot_template, anchor_active, ground_contact_anchor_state)
    selected_slot_ids = np.asarray(selected_slot_ids, dtype=np.int32).reshape(-1)
    source_slots = np.asarray(source_slots, dtype=np.float64)
    pair_slot_ids = np.zeros((0, 2), dtype=np.int32)
    source_pair_distances = np.zeros(0, dtype=np.float64)
    source_pair_weights = np.zeros(0, dtype=np.float64)
    unique_pair_slots = np.zeros(0, dtype=np.int32)
    pair_i_unique = np.zeros(0, dtype=np.int32)
    pair_j_unique = np.zeros(0, dtype=np.int32)
    if self_contact_cost > 0.0 and source_self_contact_map is not None:
        pair_slot_ids = np.asarray(source_self_contact_map['slot_pairs'], dtype=np.int32).reshape(-1, 2)
        source_pair_distances = np.asarray(source_self_contact_map['distances'], dtype=np.float64).reshape(-1)
        source_pair_weights = np.asarray(source_self_contact_map.get('weights', np.ones(len(source_pair_distances), dtype=np.float32)), dtype=np.float64).reshape(-1)
        if len(pair_slot_ids) != len(source_pair_distances):
            raise ValueError(f'Self-contact pair count mismatch: pairs={len(pair_slot_ids)}, distances={len(source_pair_distances)}')
        if len(source_pair_weights) != len(source_pair_distances):
            raise ValueError(f'Self-contact weight count mismatch: weights={len(source_pair_weights)}, distances={len(source_pair_distances)}')
        if len(pair_slot_ids) > 0:
            unique_pair_slots = np.unique(pair_slot_ids.reshape(-1)).astype(np.int32)
            pair_i_unique = np.searchsorted(unique_pair_slots, pair_slot_ids[:, 0])
            pair_j_unique = np.searchsorted(unique_pair_slots, pair_slot_ids[:, 1])
    ground_active = np.zeros(0, dtype=np.int32)
    ground_target_z = np.zeros(0, dtype=np.float64)
    ground_row_costs = np.zeros(0, dtype=np.float64)
    if float(args.ground_contact_map_cost) > 0.0 and source_ground_contact_distances is not None and (ground_contact_threshold >= 0.0):
        ground_active = ground_contact_active_slots(target_distances_for_contact, ground_contact_threshold, ground_contact_max_points, rank_distances=weight_distances_for_contact, candidate_slot_ids=selected_slot_ids)
        if ground_active.size > 0:
            ground_target_z = target_distances_for_contact[ground_active]
            ground_strength = np.clip((ground_contact_threshold - weight_distances_for_contact[ground_active]) / max(ground_contact_threshold, 1e-08), 0.05, 1.0)
            ground_row_costs = float(args.ground_contact_map_cost) * ground_strength
    anchor_row_costs = np.zeros(0, dtype=np.float64)
    if float(args.ground_contact_anchor_cost) > 0.0 and target_distances_for_contact is not None and (anchor_active.size > 0):
        anchor_strength = np.clip((ground_contact_threshold - weight_distances_for_contact[anchor_active]) / max(ground_contact_threshold, 1e-08), 0.05, 1.0)
        anchor_row_costs = float(args.ground_contact_anchor_cost) * anchor_strength
    smooth_jac = None
    if len(joint_dof_addrs) > 0:
        smooth_jac = np.zeros((len(joint_dof_addrs), model.nv), dtype=np.float64)
        smooth_jac[np.arange(len(joint_dof_addrs)), joint_dof_addrs] = 1.0
    frame_iters = int(args.iters) if iters is None else int(iters)
    for _iter in range(max(1, frame_iters)):
        common.set_qpos(model, data, qpos)
        rows = []
        residuals = []
        slot_cache = common.TemplateSlotKinematicsCache(model, data, robot_template)
        point_costs = np.asarray(surface_point_slot_costs, dtype=np.float64)[selected_slot_ids]
        point_active = point_costs > 0.
        if np.any(point_active):
            ids = selected_slot_ids[point_active]
            weight = np.sqrt(point_costs[point_active])
            rows.append((weight[:, None, None] * slot_cache.point_jacobians(ids)).reshape(-1, model.nv))
            residuals.append((weight[:, None] * (slot_cache.points(ids) - source_slots[ids])).reshape(-1))
        if source_slot_normals is not None and surface_normal_slot_costs is not None:
            normal_targets = np.asarray(source_slot_normals if source_surface_normal_targets is None
                                        else source_surface_normal_targets, dtype=np.float64)
            if len(normal_targets) != len(source_slot_normals):
                raise ValueError("Surface normal target count mismatch")
            normal_costs = np.asarray(surface_normal_slot_costs, dtype=np.float64)[selected_slot_ids]
            normal_active = normal_costs > 0.
            if np.any(normal_active):
                ids = selected_slot_ids[normal_active]
                target = normal_targets[ids]
                target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
                weight = np.sqrt(normal_costs[normal_active])
                rows.append((weight[:, None, None] * slot_cache.normal_jacobians(ids)).reshape(-1, model.nv))
                residuals.append((weight[:, None] * (slot_cache.normals(ids) - target)).reshape(-1))
        if self_contact_cost > 0.0 and len(pair_slot_ids) > 0:
            unique_points = slot_cache.points(unique_pair_slots)
            point_i = unique_points[pair_i_unique]
            point_j = unique_points[pair_j_unique]
            delta = point_i - point_j
            robot_distances = np.linalg.norm(delta, axis=1)
            pair_scales = np.sqrt(self_contact_cost) * np.sqrt(np.maximum(source_pair_weights, 0.0))
            active_pairs = (pair_scales > 0.0) & (robot_distances < source_pair_distances)
            if np.any(active_pairs):
                directions = np.empty_like(delta)
                nonzero = robot_distances > 1e-12
                directions[nonzero] = delta[nonzero] / robot_distances[nonzero, None]
                if np.any(~nonzero):
                    source_delta = source_slots[pair_slot_ids[~nonzero, 0]] - source_slots[pair_slot_ids[~nonzero, 1]]
                    source_norm = np.maximum(np.linalg.norm(source_delta, axis=1), 1e-12)
                    directions[~nonzero] = source_delta / source_norm[:, None]
                active_pair_slot_ids = pair_slot_ids[active_pairs]
                active_dirs = directions[active_pairs]
                active_scales = pair_scales[active_pairs]
                active_residuals = active_scales * (robot_distances[active_pairs] - source_pair_distances[active_pairs])
                jac_i = slot_cache.point_jacobians(active_pair_slot_ids[:, 0])
                jac_j = slot_cache.point_jacobians(active_pair_slot_ids[:, 1])
                jac_rows = np.einsum('pi,pij->pj', active_dirs, jac_i - jac_j)
                rows.append(active_scales[:, None] * jac_rows)
                residuals.append(active_residuals)
        if float(args.ground_contact_map_cost) > 0.0 and source_ground_contact_distances is not None and (ground_contact_threshold >= 0.0):
            if ground_active.size > 0:
                ground_points = slot_cache.points(ground_active)
                errors = ground_points[:, 2] - ground_target_z
                rows.append(ground_row_costs[:, None] * slot_cache.point_jacobians(ground_active)[:, 2])
                residuals.append(ground_row_costs * errors)
        if float(args.ground_contact_anchor_cost) > 0.0 and target_distances_for_contact is not None and (anchor_active.size > 0):
            anchor_points = slot_cache.points(anchor_active)
            targets = np.stack([ground_contact_anchor_state[int(slot_id)] for slot_id in anchor_active])
            rows.append((anchor_row_costs[:, None, None] * slot_cache.point_jacobians(anchor_active)).reshape(-1, model.nv))
            residuals.append((anchor_row_costs[:, None] * (anchor_points - targets)).reshape(-1))
        if qpos_prev is not None and float(args.smooth_cost) > 0.0 and (smooth_jac is not None):
            rows.append(sqrt_smooth * smooth_jac)
            residuals.append(sqrt_smooth * (qpos[joint_qpos_addrs] - qpos_prev[joint_qpos_addrs]))
        if qpos_prev is not None and qpos_prev2 is not None and (float(args.temporal_smooth_cost) > 0.0) and (smooth_jac is not None):
            rows.append(sqrt_temporal_smooth * smooth_jac)
            current_dq = qpos[joint_qpos_addrs] - qpos_prev[joint_qpos_addrs]
            previous_dq = qpos_prev[joint_qpos_addrs] - qpos_prev2[joint_qpos_addrs]
            residuals.append(sqrt_temporal_smooth * (current_dq - previous_dq))
        robot_self_cost = float(args.robot_self_penetration_cost)
        robot_self_hard = bool(args.robot_self_penetration_hard_constraint)
        jacobians_self = []
        distances_self = []
        robot_object_soft_cost = float(args.robot_object_penetration_soft_cost)
        robot_object_hard = bool(args.robot_object_hard_constraint)
        jacobians_object = []
        distances_object = []
        J = np.concatenate(rows, axis=0)
        r = np.concatenate(residuals, axis=0)
        (use_step_box, use_step_l2) = common.step_limit_flags(args.step_limit_mode, label='step_limit_mode')
        (use_root_step_box, use_root_step_l2) = common.step_limit_flags(args.root_step_limit_mode, allow_off=True, label='root_step_limit_mode')
        local_dof_ids = common.local_pose_qvel_dof_ids(model)
        (root_translation_dof_ids, root_rotation_dof_ids) = common.root_qvel_dof_groups(model)
        (step_lower, step_upper) = common.qvel_step_bounds(model, qpos, args.max_dq, joint_limits_by_qpos=joint_limits_by_qpos, use_step_box=use_step_box, limited_dof_ids=local_dof_ids, dof_max_dq_box=getattr(args, 'dof_max_dq_box_by_dof', None))
        if use_root_step_box:
            common.apply_qvel_box_limit(step_lower, step_upper, root_translation_dof_ids, args.root_max_translation_dq)
            common.apply_qvel_box_limit(step_lower, step_upper, root_rotation_dof_ids, args.root_max_rotation_dq)
        l2_step_limits = []
        if use_step_l2:
            l2_step_limits.append((local_dof_ids, args.global_step_size, 'global_step_size(local_pose)'))
        if use_root_step_l2:
            l2_step_limits.append((root_translation_dof_ids, args.root_global_translation_step_size, 'root_global_translation_step_size'))
            l2_step_limits.append((root_rotation_dof_ids, args.root_global_rotation_step_size, 'root_global_rotation_step_size'))
        ineq_rows = []
        ineq_bounds = []
        ineq_soft_costs = []
        ground_mode = str(args.ground_penetration_hard_constraint_mode)
        if bool(args.ground_penetration_hard_constraint) and ground_mode in {'surface_slots', 'surface_slots_all'}:
            if ground_mode == 'surface_slots_all':
                candidate_slot_ids = np.arange(len(robot_template['geom_ids']), dtype=np.int32)
            else:
                candidate_slot_ids = np.asarray(selected_slot_ids, dtype=np.int32)
            candidate_z = common.template_points_world_z(data, robot_template, candidate_slot_ids)
            (constraint_slot_ids, constraint_z) = ground_penetration_constraint_slots(candidate_z, candidate_slot_ids, args.ground_penetration_max_points, args.ground_penetration_threshold, margin=args.ground_penetration_margin)
            if constraint_slot_ids.size > 0:
                floor_z = float(args.ground_penetration_margin)
                ground_slack_cost = float(args.ground_penetration_hard_slack_cost) if bool(args.ground_penetration_hard_slack) else 0.0
                ineq_rows.extend(-slot_cache.point_jacobians(constraint_slot_ids)[:, 2])
                ineq_bounds.extend(constraint_z - floor_z)
                ineq_soft_costs.extend(np.full(len(constraint_slot_ids), ground_slack_cost))
        ineq_A = np.asarray(ineq_rows, dtype=np.float64) if ineq_rows else None
        ineq_b = np.asarray(ineq_bounds, dtype=np.float64) if ineq_bounds else None
        ineq_soft_costs_array = np.asarray(ineq_soft_costs, dtype=np.float64) if ineq_rows else None
        dq = common.solve_clarabel_qp_step(J, r, args.damping, step_lower, step_upper, ineq_A=ineq_A, ineq_b=ineq_b, ineq_soft_costs=ineq_soft_costs_array, l2_step_limits=l2_step_limits)
        mujoco.mj_integratePos(model, qpos, dq, 1.0)
        qpos = common.clamp_joint_ranges(model, qpos, joint_limits_by_qpos=joint_limits_by_qpos)
        costs.append(float(np.mean(r * r)))
    common.set_qpos(model, data, qpos)
    return (qpos, costs[-1] if costs else 0.0)
