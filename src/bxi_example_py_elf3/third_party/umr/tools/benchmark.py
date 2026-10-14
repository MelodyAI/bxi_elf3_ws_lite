"""Measure streamed reference throughput without ROS, cache or motor control.

python tools/benchmark.py --frames 240 --save /tmp/umr-benchmark.npz
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import onnxruntime as ort
from umr import streaming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[3]
    source_path = args.source or package / "policy/lafan1_orin/dance1_subject2.bvh"
    source, assets, settings = streaming.load_source(
        source_path, package / "policy/body_models/smplx/SMPLX_NEUTRAL_compact.npz")
    if args.frames > 0:
        source.frame_count = min(source.frame_count, args.frames)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(package / "policy/rgmtr_125200.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    start = time.perf_counter()
    timestamps, references = [], []
    for index, reference in streaming.iter_references(
            source, assets, settings, package / "robot/elf3_lite/xml/elf3.xml", metadata, lambda: False):
        assert index == len(references)
        references.append(reference)
        timestamps.append(time.perf_counter())
    elapsed = timestamps[-1] - start
    fields = {key: np.stack([ref[i] for ref in references]) for i, key in enumerate(
        ("joint_pos", "joint_vel", "root_quat", "root_ang_vel"))}
    if args.save:
        np.savez_compressed(args.save, **fields)
    result = dict(source=source_path.name, source_frames=source.frame_count,
                  reference_frames=len(references), total_seconds=elapsed,
                  first_reference_ms=(timestamps[0] - start) * 1000,
                  reference_hz=len(references) / elapsed,
                  steady_reference_hz=(len(references) - 1) / (timestamps[-1] - timestamps[0]),
                  gap_ms_p95=float(np.percentile(np.diff(timestamps), 95) * 1000),
                  gap_ms_max=float(np.max(np.diff(timestamps)) * 1000))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
