"""Colour transfer functions and image I/O."""
import numpy as np
import torch


def srgb_to_linear(x):
    return torch.where(x <= .04045, x / 12.92, ((x + .055) / 1.055).clamp_min(0) ** 2.4)


def linear_to_srgb(x):
    # Clamp only inside the inactive power branch, so the linear branch keeps its gradient.
    return torch.where(x <= .0031308, x * 12.92, 1.055 * x.clamp_min(.0031308) ** (1 / 2.4) - .055)


def _mitsuba():
    import mitsuba as mi
    if mi.variant() is None:   # bitmap I/O works in any variant; the renderer selects cuda_ad_rgb itself
        mi.set_variant('scalar_rgb')
    return mi


def read_exr(path):
    """Linear RGB float32 array (H, W, 3) read with Mitsuba's bitmap reader (also reads .hdr)."""
    mi = _mitsuba()
    bitmap = mi.Bitmap(str(path)).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
    return np.array(bitmap, copy=True).astype(np.float32)


def write_exr(path, rgb):
    mi = _mitsuba()
    mi.Bitmap(np.ascontiguousarray(rgb, dtype=np.float32)).write(str(path))


def read_png(path, rgba=True):
    """PNG as float32 in [0, 1] keeping its bit depth (8 or 16 bit); RGBA (alpha 1 if absent) or RGB."""
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    scale = 65535. if a.dtype == np.uint16 else 255.
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    a = a[..., [2, 1, 0, 3] if a.shape[-1] == 4 else [2, 1, 0]]
    a = np.ascontiguousarray(a).astype(np.float32) / scale
    if rgba and a.shape[-1] == 3:
        a = np.concatenate([a, np.ones_like(a[..., :1])], -1)
    return a if rgba else a[..., :3]


def write_png(path, image):
    """Save an (H, W, 3) image in [0, 1] (tensor or array) as an 8-bit PNG."""
    from PIL import Image
    x = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    Image.fromarray((np.clip(x, 0, 1) * 255 + .5).astype(np.uint8)).save(path)
