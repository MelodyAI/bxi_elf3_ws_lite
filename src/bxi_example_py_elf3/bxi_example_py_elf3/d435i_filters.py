"""Camera-side FOV sampling and Instinct Navier-Stokes hole filling."""
from copy import deepcopy

import cv2
import numpy as np

from .models.depth_projection import choose_fov_crop, resize_nearest


def inpaint_invalid_depth(depth_m, invalid_below_m=.2, radius_px=3.):
    depth = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    invalid = depth < invalid_below_m
    count = int(np.count_nonzero(invalid))
    if count == 0:
        return depth, 0
    if count == depth.size:
        return np.zeros_like(depth), count
    filled = cv2.inpaint(depth, invalid.astype(np.uint8), float(radius_px), cv2.INPAINT_NS)
    return np.asarray(filled, dtype=np.float32), count


def instinct_depth(depth_mm, info, projection, *, invalid_below_m=.2, radius_px=3.):
    """Filter at reference small-image scale; keep output calibration accurate.

    No policy distance clipping, rotation, Gaussian or normalization here.
    The existing model receiver performs those operations after publication.
    """
    if not np.isfinite(invalid_below_m) or invalid_below_m < 0.:
        raise ValueError('Instinct invalid depth threshold must be finite and nonnegative')
    if not np.isfinite(radius_px) or radius_px <= 0.:
        raise ValueError('Instinct inpaint radius must be finite and positive')
    height, width = depth_mm.shape
    if (width, height) != (info.width, info.height):
        raise ValueError('Depth dimensions and CameraInfo must match before camera filtering')
    x, y, crop_width, crop_height = choose_fov_crop(width, height, info.k[0], info.k[4], projection)
    small = resize_nearest(depth_mm[y:y+crop_height, x:x+crop_width].astype(np.float32) * .001,
                           projection.output_width, projection.output_height)
    filled, _ = inpaint_invalid_depth(small, invalid_below_m, radius_px)
    output = np.rint(np.clip(np.nan_to_num(filled, nan=0., posinf=0., neginf=0.)*1000.,
                             0., 65535.)).astype('<u2')
    calibrated = deepcopy(info)
    sx, sy = projection.output_width/crop_width, projection.output_height/crop_height
    calibrated.width, calibrated.height = projection.output_width, projection.output_height
    calibrated.k = [info.k[0]*sx, 0., (info.k[2]-x)*sx,
                    0., info.k[4]*sy, (info.k[5]-y)*sy, 0., 0., 1.]
    calibrated.p = [info.p[0]*sx, 0., (info.p[2]-x)*sx, info.p[3]*sx,
                    0., info.p[5]*sy, (info.p[6]-y)*sy, info.p[7]*sy,
                    0., 0., 1., 0.]
    return np.ascontiguousarray(output), calibrated
