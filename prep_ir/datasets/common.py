"""Shared helpers of the dataset converters (standalone: numpy, OpenCV, PIL; no renderer imports)."""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

# Neural surface reconstruction runs in DiffReg-PBIR's frame for Synthetic4Relight / TensoIR (see mii_to_world);
# vertices map back to the NeRF world frame by v @ NSR_TO_WORLD_MII.
NSR_TO_WORLD_MII = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def link(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_symlink() and not dst.exists():
        os.symlink(src.resolve(), dst)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1) + '\n')


def read_exr_rgb(path):
    import cv2
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(x[..., 2::-1]).astype(np.float32)


def srgb(x):
    """Linear -> sRGB on clipped values (as the datasets' own converters)."""
    x = np.clip(x, 0., 1.)
    return np.where(x <= .0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - .055)


def mii_to_world(transform_matrix):
    """NeRF camera-to-world -> the `to_world` of DiffReg-PBIR's cameras.json for Synthetic4Relight / TensoIR
    (scripts/preprocess/mii.py of DiffReg-PBIR)."""
    c2w = np.array(transform_matrix).astype(np.float32)
    c2w[:, [1, 2]] *= -1
    c2w[[1, 2]] = np.stack([c2w[2], -c2w[1]])
    c2w[:, :2] *= -1
    return c2w.tolist()


def diffreg_cameras(frames, width, height, camera_angle_x):
    """cameras.json of DiffReg-PBIR's surface reconstruction (train frames only)."""
    focal = .5 * width / np.tan(.5 * float(camera_angle_x))
    return {'split': {'train': list(range(len(frames))), 'test': [0]}, 'frames': frames, 'fx': focal, 'fy': focal,
            'cx': .5 * width, 'cy': .5 * height, 'aabb': [[-1.5] * 3, [1.5] * 3]}
