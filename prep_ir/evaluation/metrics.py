"""Image metrics of the synthetic evaluation protocol (supplement, "Evaluation Protocol")."""
import numpy as np
import torch
import torch.nn.functional as F


def ssim(a, b):
    """SSIM of two (H, W, 3) images with the public TensoIR / mip-NeRF filter (11x11 Gaussian, sigma 1.5, valid)."""
    z = torch.stack([a, b, a * a, b * b, a * b]).permute(0, 3, 1, 2).double()
    f = torch.exp(-.5 * (torch.arange(-5, 6, device=z.device, dtype=z.dtype) / 1.5) ** 2)
    f /= f.sum()
    z = F.conv2d(z, f[None, None, :, None].expand(3, 1, 11, 1), groups=3)
    z = F.conv2d(z, f[None, None, None, :].expand(3, 1, 1, 11), groups=3)
    m0, m1, aa, bb, ab = z
    v0, v1 = (aa - m0 * m0).clamp_min(0), (bb - m1 * m1).clamp_min(0)
    c = ab - m0 * m1
    c = c.sign() * torch.minimum((v0 * v1).sqrt(), c.abs())
    return float((((2 * m0 * m1 + .01 ** 2) * (2 * c + .03 ** 2)) / ((m0 * m0 + m1 * m1 + .01 ** 2) * (v0 + v1 + .03 ** 2))).mean())


def score(a, b):
    """MSE over H x W x 3, its PSNR and SSIM."""
    mse = float((a - b).square().mean())
    return dict(mse=mse, psnr=-10 * np.log10(max(mse, 1e-12)), ssim=ssim(a, b))
