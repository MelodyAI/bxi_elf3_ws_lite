"""Capture one D435i source frame at raw, NS-filtered and model-input stages."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from bxi_example_py_elf3.realsense_runtime import use_bundled_realsense
use_bundled_realsense()

import numpy as np
import pyrealsense2 as rs
from PIL import Image as PngImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from bxi_example_py_elf3.d435i_camera import camera_info, depth_millimetres, select_device
from bxi_example_py_elf3.models.depth_camera import DepthCameraInput
from bxi_example_py_elf3.models.depth import HumanoidGaitDepthPolicy
from bxi_example_py_elf3.d435i_filters import instinct_depth
from bxi_example_py_elf3.model_config import (
    DEPTH_CAMERA_HARDWARE, DEPTH_CAMERA_NAME, DEPTH_CAMERA_STREAM, DEPTH_WALK_MODE, DEPTH_WALK_PREPROCESS, get_model_file_dicts,
)


def save_visual(path, array, *, vmin, vmax, invalid_mask=None):
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('black')
    pixels = np.ma.masked_where(invalid_mask, array) if invalid_mask is not None else array
    rgba = cmap(plt.Normalize(vmin, vmax, clip=True)(pixels), bytes=True)
    PngImage.fromarray(rgba[:, :, :3]).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/bxi_d435i_capture') /
                        datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--warmup-frames', type=int, default=60)
    args = parser.parse_args()
    if args.warmup_frames < 1:
        parser.error('--warmup-frames must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    _, models = get_model_file_dicts()
    policy = HumanoidGaitDepthPolicy(models['depth_walk'], mode=DEPTH_WALK_MODE, **DEPTH_WALK_PREPROCESS)
    context = rs.context()
    device = select_device(context, DEPTH_CAMERA_HARDWARE['serial_no'])
    serial = device.get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    capture_fps = int(DEPTH_CAMERA_STREAM['fps'])
    config.enable_stream(rs.stream.depth, int(DEPTH_CAMERA_STREAM['depth_width']),
                         int(DEPTH_CAMERA_STREAM['depth_height']), rs.format.z16,
                         capture_fps)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        sensor = profile.get_device().first_depth_sensor()
        scale = sensor.get_depth_scale()
        for _ in range(args.warmup_frames + 1):
            raw_frame = pipeline.wait_for_frames(5000).get_depth_frame()
            # Copy before filtering: these values belong to this exact source frame.
            raw_z16 = np.asanyarray(raw_frame.get_data()).copy()
        intrinsics = raw_frame.profile.as_video_stream_profile().intrinsics
        sdk_stamp = raw_frame.get_timestamp()
        frame_number = raw_frame.get_frame_number()
    finally:
        if started:
            pipeline.stop()

    raw_m = raw_z16.astype(np.float32) * scale
    raw_mm = depth_millimetres(raw_z16, scale)
    header = Header()
    now_ns = time.time_ns()
    header.stamp.sec, header.stamp.nanosec = divmod(now_ns, 10**9)
    header.frame_id = DEPTH_CAMERA_NAME + '_depth_optical_frame'
    info = camera_info(intrinsics, header)
    raw_resolution = [int(info.width), int(info.height)]
    receiver = DepthCameraInput(mode=DEPTH_WALK_MODE, depth_range=DEPTH_WALK_PREPROCESS.get('depth_range'))
    filtered_mm, info = instinct_depth(raw_mm, info, receiver.projection,
        invalid_below_m=DEPTH_CAMERA_HARDWARE['invalid_below_m'],
        radius_px=DEPTH_CAMERA_HARDWARE['inpaint_radius_px'])
    filtered_m = filtered_mm.astype(np.float32)*.001
    image = Image()
    image.header = header
    image.height, image.width = filtered_mm.shape
    image.encoding, image.is_bigendian = '16UC1', False
    image.step = image.width * 2
    image.data.frombytes(filtered_mm.tobytes())
    receiver.camera_info_callback(info)
    receiver.image_callback(image)
    sample = receiver.sample(advance=False)
    assert sample is not None, receiver.last_error
    target = policy.inference_step(policy.default_dof_pos, np.zeros(29), [1., 0., 0., 0.],
                                   np.zeros(3), np.zeros(3), *sample)
    model_input = policy.depth_input_buffer.copy()
    assert np.isfinite(target).all() and np.isfinite(model_input).all()
    PngImage.fromarray(raw_mm).save(out/'raw_depth_mm.png')
    PngImage.fromarray(filtered_mm).save(out/'filtered_depth_mm.png')
    np.save(out/'model_input.npy', model_input)
    np.savez_compressed(out/'depth_stages.npz', raw_z16=raw_z16, filtered_depth_mm=filtered_mm,
                        raw_depth_m=raw_m, filtered_depth_m=filtered_m,
                        projected_depth_m=sample[0], model_input=model_input,
                        camera_k=np.asarray(info.k), target=target)
    distance_min, distance_max = policy.depth_range
    save_visual(out/'raw_depth_vis.png', raw_m, vmin=distance_min, vmax=distance_max, invalid_mask=raw_z16 == 0)
    save_visual(out/'filtered_depth_vis.png', filtered_m, vmin=distance_min, vmax=distance_max, invalid_mask=filtered_mm == 0)
    model_image = model_input[0] if DEPTH_WALK_MODE == 'origin_camera' else model_input[0, -1]
    limits = (-.5, .5) if DEPTH_WALK_MODE == 'origin_camera' else (0., 1.)
    save_visual(out/'model_input_vis.png', model_image, vmin=limits[0], vmax=limits[1])
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('black')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    fig.suptitle(f'D435i frame {frame_number}: raw -> Instinct NS filter -> {model_input.shape} model input')
    filtered_title = 'FOV + nearest resize + NS inpaint'
    for ax, data, title in zip(axes[:2], (raw_m, filtered_m),
                               ('Raw depth (before software filters)', filtered_title)):
        artist = ax.imshow(np.ma.masked_where(data <= 0., data), cmap=cmap, vmin=distance_min, vmax=distance_max)
        ax.set_title(title)
        ax.set_xlabel('pixel x')
        ax.set_ylabel('pixel y')
        fig.colorbar(artist, ax=ax, label=f'metres (display clipped to {distance_min:g}-{distance_max:g} m)', shrink=.8)
    artist = axes[2].imshow(model_image, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation='nearest')
    axes[2].set_title(f'Model input: {model_image.shape[1]}x{model_image.shape[0]}')
    axes[2].set_xlabel('column')
    axes[2].set_ylabel('row')
    fig.colorbar(artist, ax=axes[2], label='normalized depth', shrink=.8)
    fig.savefig(out/'overview.png', dpi=160)
    plt.close(fig)
    report = {'captured_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'serial_number': serial, 'source_frame_number': frame_number,
              'source_timestamp_ms': sdk_stamp, 'same_source_frame': True,
              'warmup_frames': args.warmup_frames, 'depth_scale_m': scale,
              'capture_fps': capture_fps, 'configured_publish_hz': int(DEPTH_CAMERA_STREAM['fps']),
              'raw_resolution': raw_resolution, 'resolution': [int(info.width), int(info.height)],
              'raw_valid_ratio': float(np.mean(raw_z16 > 0)),
              'filtered_valid_ratio': float(np.mean(filtered_mm > 0)),
              'model': policy.model_onnx_path, 'mode': DEPTH_WALK_MODE,
              'depth_preprocess': {'crop_rows':policy.crop_rows,'depth_range':policy.depth_range},
              'model_input_shape': list(model_input.shape),
              'model_input_range': [float(model_input.min()), float(model_input.max())],
              'model_inference_finite': bool(np.isfinite(target).all()),
              'projection_roi': receiver.last_crop, 'camera_k': list(info.k),
              'filters': DEPTH_CAMERA_HARDWARE, 'sdk_path': rs.__file__,
              'preview_depth_range_m': list(policy.depth_range), 'invalid_pixels_color': 'black',
              'data_note': '16-bit PNG values are millimetres; NPY/NPZ retain float32 input and full-range depth. Preview clipping changes colors only.',
              'output': str(out.resolve())}
    (out/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
