"""Read XRoboToolkit PICO tracking and publish the ELF3 14-body protocol."""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np
from .models.pico_contract import BODY_NAMES, CONTRACT, convert_xr_pose


XR_BODY_COUNT = 24


class PicoPoseSender:
    def __init__(self, endpoint: str, publish_fps: float) -> None:
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise ImportError(
                "xrobotoolkit_sdk is required; install the standalone XRoboToolkit Python binding"
            ) from exc
        import zmq

        self.xrt = xrt
        self.period_s = 1.0 / float(publish_fps)
        self._lock = threading.Lock()
        self._latest = None
        self._latest_timestamp = None
        self._seq = 0
        self._last_observed_pose = None
        self._last_observed_stamp = None
        self._last_change_time = 0.0
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.bind(endpoint)

    def _poll_sdk(self) -> None:
        # Never keep broadcasting the last pose after availability is lost.
        with self._lock:
            self._latest = None
        if not bool(self.xrt.is_body_data_available()):
            return
        try:
            poses = np.asarray(self.xrt.get_body_joints_pose(), dtype=np.float32)
            poses = poses[:XR_BODY_COUNT, :7]
            if poses.shape != (XR_BODY_COUNT, 7) or not np.isfinite(poses).all():
                return
            timestamp = int(self.xrt.get_body_timestamp_ns())
        except Exception:
            return
        # Some SDK builds retain stale arrays after disconnect and report a
        # permanent zero timestamp. Require either a changing source timestamp
        # or changing pose bytes; never manufacture freshness from send time.
        changed = (self._last_observed_pose is None
                   or not np.array_equal(poses, self._last_observed_pose)
                   or (timestamp > 0 and timestamp != self._last_observed_stamp))
        if changed:
            self._last_change_time = time.monotonic()
        self._last_observed_pose = poses.copy()
        self._last_observed_stamp = timestamp
        if time.monotonic() - self._last_change_time > 0.25:
            return
        with self._lock:
            self._latest = poses.copy()
            self._latest_timestamp = timestamp

    def run(self) -> None:
        import zmq

        sent_count = 0
        last_sent_pose = None
        static_count = 0
        last_poll_pose = None
        print("Standalone PICO pose sender started")
        try:
            self.xrt.init()
            while True:
                self._poll_sdk()
                with self._lock:
                    poses = None if self._latest is None else self._latest.copy()
                    timestamp = self._latest_timestamp
                poll_delta = None if poses is None or last_poll_pose is None else float(np.max(np.abs(poses - last_poll_pose)))
                if poses is not None:
                    last_poll_pose = poses.copy()
                if poses is not None:
                    try:
                        positions, rotations = convert_xr_pose(poses)
                    except ValueError:
                        time.sleep(self.period_s)
                        continue
                    header = json.dumps(
                        {"contract": CONTRACT, "body_names": BODY_NAMES,
                         "body_count": 14, "recv_ns": time.monotonic_ns(), "seq": self._seq}
                    ).encode("utf-8")
                    try:
                        self._socket.send_multipart(
                            [header, positions.astype('<f4').tobytes(), rotations.astype('<f4').tobytes()],
                            flags=zmq.NOBLOCK,
                        )
                        self._seq += 1
                        sent_count += 1
                        if last_sent_pose is not None and float(np.max(np.abs(poses - last_sent_pose))) < 1e-6:
                            static_count += 1
                        else:
                            static_count = 0
                        last_sent_pose = poses.copy()
                        if sent_count % 50 == 0:
                            print(
                                f"PICO pose frames sent: {sent_count}, "
                                f"latest_delta={poll_delta}",
                                flush=True,
                            )
                        if static_count == 100:
                            print(
                                "WARNING: PICO body pose is unchanged for 2 seconds; "
                                "check Motion Tracker connection/calibration",
                                flush=True,
                            )
                    except zmq.Again:
                        pass
                time.sleep(self.period_s)
        except KeyboardInterrupt:
            pass
        finally:
            self.xrt.close()
            self._socket.close(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://*:28704")
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()
    PicoPoseSender(args.endpoint, args.fps).run()


if __name__ == "__main__":
    main()
