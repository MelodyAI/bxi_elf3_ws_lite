"""One-time asset export in the original UMR environment; not a runtime dependency.

Usage: python export_assets.py /path/to/UMR
Exports the learned correspondence, mesh bindings and body-only LBS coefficients.
"""
from pathlib import Path
import inspect
import ast
import json
import sys

import numpy as np


def main():
    upstream = Path(sys.argv[1]).resolve()
    destination = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(upstream / "scripts"))
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(destination.parent))
    import retarget_smpl_to_humanoid_surface_vector as pipeline
    import retarget_body_segment_surface as segments
    import mujoco
    from umr.sources import read_bvh, BVH_BODY_NAMES
    from lafan_calibration import export_lafan_frames

    captured = {}
    original_source = pipeline.bind_source_slots_with_normals
    original_robot = pipeline.bind_robot_slots

    def capture_source(*args, **kwargs):
        result = original_source(*args, **kwargs)
        captured.update(vertices=args[1], faces=args[2], binding=result[4])
        return result

    def capture_robot(*args, **kwargs):
        result = original_robot(*args, **kwargs)
        captured["robot"] = result
        return result

    class ExportFinished(Exception):
        pass

    def export(*args, **kwargs):
        call = inspect.signature(original_solve).bind(*args, **kwargs).arguments
        selected = call["selected_slot_ids"]
        faces = captured["faces"][captured["binding"]["face_ids"][selected]]
        vertex_ids, triangle_ids = np.unique(faces, return_inverse=True)
        model = pipeline.common.build_smplx_model(upstream / "smpl", "neutral", 1)
        def array(value):
            return value.detach().cpu().numpy()
        parents = array(model.parents)
        weights = array(model.lbs_weights)[vertex_ids]
        # Zero finger/face poses make their transforms equal to their body ancestor.
        body_weights = np.zeros((len(vertex_ids), 22), np.float32)
        for joint in range(weights.shape[1]):
            ancestor = joint
            while ancestor >= 22:
                ancestor = int(parents[ancestor])
            body_weights[:, ancestor] += weights[:, joint]
        v = array(model.v_template)
        shapedirs = array(model.shapedirs)[..., :10]
        regressor = array(model.J_regressor)[:22]
        robot = captured["robot"]
        ids = robot["geom_ids"]
        mjmodel = call["model"]
        # Geoms in this MJCF are unnamed; use owning body + ordinal within body.
        body_names = [mujoco.mj_id2name(mjmodel, mujoco.mjtObj.mjOBJ_BODY,
                                      int(mjmodel.geom_bodyid[i])) for i in ids]
        ordinals = [list(np.flatnonzero(mjmodel.geom_bodyid == mjmodel.geom_bodyid[i])).index(i)
                    for i in ids]
        payload = dict(
            format_version=np.asarray(1), vertex_ids=vertex_ids,
            triangles=triangle_ids.reshape(-1, 3),
            bary=captured["binding"]["bary"][selected],
            reference_triangles=captured["vertices"][faces],
            v_template=v[vertex_ids], shapedirs=shapedirs[vertex_ids],
            posedirs=array(model.posedirs)[:189].reshape(189, -1, 3)[:, vertex_ids],
            weights=body_weights, parents=parents[:22],
            joints=regressor @ v,
            joint_shapedirs=np.einsum("jv,vci->jci", regressor, shapedirs),
            selected=selected, part_ids=call["source_slot_part_ids"][selected],
            point_costs=call["surface_point_slot_costs"][selected],
            normal_costs=call["surface_normal_slot_costs"][selected],
            normal_reference=captured["normal_reference"][selected],
            robot_body_names=np.asarray(body_names), robot_geom_ordinals=np.asarray(ordinals),
            robot_local_pos=robot["local_pos"], robot_local_normals=robot["local_normals"],
            robot_height=np.asarray(call["args"].robot_height),
            valid_pairs=segments.non_adjacent_segment_pair_mask(
                call["source_slot_part_ids"], selected),
        )
        calibration = export_lafan_frames(
            destination.parents[1] / "policy/lafan1_orin/dance1_subject2.bvh",
            upstream / "sample_data/lafan1_smplx/dance1_subject2.npz",
            parents[:22], read_bvh, BVH_BODY_NAMES)
        calibration_shape = v + np.einsum("vci,i->vc", shapedirs, calibration["bvh_reference_betas"])
        payload.update(calibration)
        payload["bvh_reference_height"] = np.asarray(np.ptp(calibration_shape[:, 1]))
        np.savez_compressed(destination / "elf3_surface.npz", **payload)
        settings = {k: v for k, v in vars(call["args"]).items()
                    if isinstance(v, (str, int, float, bool)) or v is None}
        settings["part_name_to_id"] = segments.SMPLX_PART_IDS
        solver_tree = ast.parse((destination / "solver.py").read_text())
        used_settings = {node.attr for node in ast.walk(solver_tree)
                         if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                         and node.value.id == "args"}
        used_settings.update({"part_name_to_id", "trajectory_filter_data_cost",
                              "trajectory_filter_velocity_cost", "trajectory_filter_acceleration_cost",
                              "trajectory_filter_jerk_cost", "pose_init_iters", "self_contact_map_threshold",
                              "self_contact_map_max_pairs", "ground_contact_map_snap_threshold"})
        settings = {k: v for k, v in settings.items() if k in used_settings}
        (destination / "solver_config.json").write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Exported {len(selected)} surface slots, {len(vertex_ids)} vertices")
        raise ExportFinished

    original_solve = pipeline.solve_frame_body_segment_qp
    pipeline.solve_frame_body_segment_qp = export
    # load_body_segment_module in parse_args replaces the source wrapper.
    original_load = pipeline.load_body_segment_module
    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        pipeline.bind_source_slots_with_normals = capture_source
        return result
    pipeline.load_body_segment_module = load
    pipeline.bind_robot_slots = capture_robot
    original_normals = pipeline.compute_tpose_surface_normal_offsets
    def normals(*args, **kwargs):
        result = original_normals(*args, **kwargs)
        captured["normal_reference"] = result[1]
        return result
    # This function is also replaced by load_body_segment_module.
    def load_with_normals(*args, **kwargs):
        result = load(*args, **kwargs)
        pipeline.compute_tpose_surface_normal_offsets = normals
        return result
    pipeline.load_body_segment_module = load_with_normals
    sys.argv = ["export", "--config", str(upstream / "robot_configs/humanoid_retarget_elf3_lite_contact.json"),
                "--slots", str(upstream / "output/elf3_lite_smplx/correspondence/correspondence_slots_final.npz"),
                "--max-frames", "1", "--smplx-device", "cpu"]
    try:
        pipeline.main()
    except ExportFinished:
        pass


if __name__ == "__main__":
    main()
