# Standalone PICO pose input

The ELF3 node does not import or launch the GRIT repository.  It optionally
connects to a ZeroMQ `PULL` endpoint (default `tcp://127.0.0.1:28704`) and
expects one multipart message per pose frame:

1. UTF-8 JSON header with `contract="elf3_smpl_bodies_v1"`, `body_names`
   (the exact lower-case names below), `body_count=14`, `recv_ns`, and `seq`.
2. `float32` little-endian body positions, shape `[14, 3]`, in Z-up world metres
3. `float32` little-endian body rotations in `wxyz`, shape `[14, 4]`

The sender-side body order is:

```text
pelvis, spine3, left_hip, left_knee, left_ankle,
right_hip, right_knee, right_ankle,
left_shoulder, left_elbow, left_wrist,
right_shoulder, right_elbow, right_wrist
```

The sender selects nodes exactly once. The receiver does NOT reorder them.
Old unversioned messages are rejected rather than silently corrupting limbs.
The receiver waits for 20 frames before enabling tracking.

World coordinates and bone bases are distinct. The conversion is:
`p = A p_XR`, `R = A R_XR B`, where
`A = [[1,0,0],[0,0,-1],[0,1,0]]` and `B = diag(-1,1,-1)`.
B changes the local bone basis to SMPL; it is NOT a reflection of world X.
Both transforms have determinant +1 and preserve yaw direction. The model
constructs pelvis-relative features internally. Global Z-up and local SMPL
Y-up bone axes must not be confused. No GRIT or GR00T code is imported.

During missing input/initial prediction warmup, AMP runs with zero velocity
commands to balance. Tracking starts with a 0.6 s transition and includes the
full lower body; sitting input is not masked. A fixed PD joint posture alone
does not provide active balance.

Do not attach a second PULL diagnostic client while controlling the robot:
PUSH/PULL distributes frames between clients instead of broadcasting them.

## Direct PICO usage

After installing XRoboToolkit PC Service and its standalone Python binding,
start the service and this package's sender:

```bash
cd /opt/apps/roboticsservice
bash runService.sh

# in another terminal
source install/setup.bash
PYTHONPATH=src/bxi_example_py_elf3 /home/szz/anaconda3/envs/gmr/bin/python \
  -m bxi_example_py_elf3.pico_pose_sender --endpoint 'tcp://*:28704'
```

Set `RGMT_REFERENCE_MODE = "pico"` in `model_config.py`, rebuild, and launch the
ELF3 controller. The sender imports only `xrobotoolkit_sdk`; it does not import
or execute GRIT. Restart both sender and controller after protocol updates.

Regression checks (no hardware communication):

```bash
PYTHONPATH=src/bxi_example_py_elf3 python3 -m pytest -q src/bxi_example_py_elf3/test/test_pico_contract.py
PYTHONPATH=src/bxi_example_py_elf3 python3 src/bxi_example_py_elf3/test/replay_pico_headless.py /path/to/raw_capture.npz
```

The headless test uses ELF3 XML, AMP and RGMT at 50 Hz, with PD torques at
500 Hz. Capture format is `positions[T,14,3]`, `rotations[T,14,4]` WXYZ, in
raw XR coordinates and the canonical node order above. A bounded simulation
test does not guarantee stability for every live pose.

## Arm prediction

The live PICO path does not apply arm velocity or acceleration filtering. The
Transformer output is passed directly to the RGMT reference window and the
RGMT target is passed directly to the controller. This avoids adding response
lag; any arm discontinuity must therefore be handled at the PICO/Transformer
input or model-output source rather than by a runtime slew limiter.

The standalone sender defaults to 50 Hz. Async source gaps use the local 50 Hz
control-sample timeline; the sender timestamp is used only for packet freshness.

```bash
PYTHONPATH=src/bxi_example_py_elf3 python3 -m pytest -q src/bxi_example_py_elf3/test/test_arm_continuity.py
```
