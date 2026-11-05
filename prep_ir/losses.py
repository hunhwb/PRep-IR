"""Photometric objective and per-vertex gradient clipping."""
import torch
import torch.nn.functional as F

from .image import linear_to_srgb


def ssim(x, y):
    """Mean SSIM of two (H, W, 3) images: 11x11 Gaussian window (sigma 1.5), valid convolution."""
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    z = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-z * z / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None].expand(3, 1, 11, 11)
    filt = lambda a: F.conv2d(a, kernel, groups=3)
    mx, my = filt(x), filt(y)
    vx, vy, cov = filt(x * x) - mx * mx, filt(y * y) - my * my, filt(x * y) - mx * my
    return (((2 * mx * my + .01 ** 2) * (2 * cov + .03 ** 2)) / ((mx * mx + my * my + .01 ** 2) * (vx + vy + .03 ** 2))).mean()


def photometric_loss(rendered, target, ssim_weight=.2):
    """(1 - w) L1 + w (1 - SSIM) between the sRGB-encoded linear rendering and the sRGB target (whole frame)."""
    x = linear_to_srgb(rendered)
    return (1 - ssim_weight) * (x - target).abs().mean() + ssim_weight * (1 - ssim(x, target))


def clip_vertex_norms(gradient, quantile, nonzero_only):
    """Scale per-vertex gradient rows whose norm exceeds the `quantile` of the row norms (over all rows, or over the
    nonzero rows only) down to that quantile, to limit Monte Carlo outliers."""
    norms = torch.linalg.vector_norm(gradient, dim=-1)
    if quantile >= 1:
        return gradient
    if nonzero_only:
        nonzero = norms > 0
        if not nonzero.any():
            return gradient
        cap = torch.quantile(norms[nonzero].double(), quantile).to(gradient.dtype)
    else:
        cap = torch.quantile(norms.double(), quantile).to(gradient.dtype)
    return gradient * (cap / norms.clamp_min(1e-30)).clamp(max=1.)[:, None]
