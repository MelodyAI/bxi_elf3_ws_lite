"""Latest-input UMR process; no unbounded queue and no future-pose wait."""
import contextlib
import json
import os
from pathlib import Path
import struct
import sys
from threading import Condition, Thread
import time


def main():
    config = json.loads(sys.argv[1])
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    output = sys.stdout.buffer
    condition = Condition()
    pending = [None]
    closed = [False]
    input_size = struct.calcsize("<qqq154f")
    def read_inputs():
        try:
            while True:
                data = bytearray()
                while len(data) < input_size:
                    chunk = os.read(0, input_size - len(data))
                    if not chunk:
                        return
                    data.extend(chunk)
                with condition:
                    pending[0] = bytes(data)  # Overwrite stale unsolved inputs.
                    condition.notify_all()
        finally:
            with condition:
                closed[0] = True
                condition.notify_all()
    Thread(target=read_inputs, name="umr-live-input", daemon=True).start()
    def packet(kind, payload):
        output.write(struct.pack("<I", 1 + len(payload)) + kind + payload)
        output.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            from umr.live import LiveRetargeter
            retargeter = LiveRetargeter(config["robot_xml"], config["metadata"])
            epoch = None
            while True:
                with condition:
                    condition.wait_for(lambda: closed[0] or pending[0] is not None)
                    if pending[0] is None:
                        return 0
                    sample, pending[0] = pending[0], None
                new_epoch, sequence, source_ns = struct.unpack("<qqq", sample[:24])
                # Sender and worker share a host monotonic clock. Discard stale
                # work after process startup; the next fresh packet replaces it.
                if time.monotonic_ns() - source_ns > 150_000_000:
                    continue
                if epoch != new_epoch:
                    retargeter.reset()
                    epoch = new_epoch
                values = np.frombuffer(sample[24:], dtype="<f4")
                positions, rotations = values[:66].reshape(22, 3), values[66:].reshape(22, 4)
                start = time.perf_counter()
                try:
                    result = retargeter.step(positions, rotations, source_ns)
                except (ValueError, RuntimeError) as error:
                    retargeter.reset()
                    packet(b"W", struct.pack("<q", epoch) + str(error).encode("utf-8"))
                    continue
                elapsed = time.perf_counter() - start
                produced_ns = time.monotonic_ns()
                payload = struct.pack("<qqqqd", epoch, sequence, source_ns, produced_ns, elapsed)
                payload += np.concatenate(result).astype("<f4", copy=False).tobytes()
                packet(b"F", payload)
    except Exception as error:
        packet(b"E", str(error).encode("utf-8"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
