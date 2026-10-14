"""Private UMR process: single-threaded numeric libraries, framed stdout IPC."""
import contextlib
import json
from pathlib import Path
import struct
import sys
import time


def main():
    # The parent sets BLAS/OpenMP limits before starting this interpreter.
    # No ROS imports and no GIL shared with the robot control callback.
    output = sys.stdout.buffer
    config = json.loads(sys.stdin.readline())
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    def packet(kind, payload=b""):
        output.write(struct.pack("<I", 1 + len(payload)) + kind + payload)
        output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            from umr import streaming
            source, assets, settings = streaming.load_source(config["source"], config["body_model"])
            references = iter(streaming.iter_references(
                source, assets, settings, config["robot_xml"], config["metadata"], lambda: False))
            while True:
                start = time.perf_counter()
                try:
                    index, reference = next(references)
                except StopIteration:
                    break
                elapsed = time.perf_counter() - start
                values = np.concatenate(reference).astype("<f4", copy=False)
                if values.shape != (65,) or not np.isfinite(values).all():
                    raise ValueError("UMR generated an invalid reference packet")
                packet(b"F", struct.pack("<id", index, elapsed) + values.tobytes())
        packet(b"D")
    except Exception as error:
        packet(b"E", str(error).encode("utf-8"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
