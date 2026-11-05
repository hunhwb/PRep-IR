"""RGB-guided selection of predicted albedo, with spatial and prior-colour affinity.

Only predicted *linear albedo values* are averaged. Input RGB and prior colours
are compared in sRGB; distances are measured in native prediction pixels.
Overlapping tiles bound the candidate set and blend their estimates with a Hann
window. The silhouette uses bilinear prior interpolation instead of selection.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .image import linear_to_srgb


@dataclass(frozen=True)
class RefinementConfig:
    spatial_sigma: float = 12.
    prior_sigma: float = .2
    rgb_sigma: float = .03
    tile_size: int = 64
    tile_stride: int = 32
    context: int = 32
    candidates: int = 1024
    max_upsample: int = 4

    def __post_init__(self):
        if min(self.spatial_sigma, self.prior_sigma, self.rgb_sigma) <= 0:
            raise ValueError('affinity bandwidths must be positive')
        if min(self.tile_size, self.tile_stride, self.candidates, self.max_upsample) < 1 or self.context < 0:
            raise ValueError('invalid candidate/tile settings')
        if self.tile_stride > self.tile_size:
            raise ValueError('tile stride must not leave gaps')

    def as_dict(self):
        return asdict(self)


def _tiles(length, size, stride):
    last = max(length - size, 0)
    starts = list(range(0, last + 1, stride))
    return starts if starts[-1] == last else starts + [last]


@torch.no_grad()
def refine_albedo(image, prior, valid, config=None):
    """Linear RGB tensors (1,3,H,W), native valid mask (1,1,H,W) -> refined linear RGB.

    The output is at most 4x the prediction size, following the input resolution.
    Candidate sampling is deterministic and shared across affinity settings.
    """
    c = config or RefinementConfig()
    if image.ndim != 4 or prior.ndim != 4 or image.shape[:2] != (1, 3) or prior.shape[:2] != (1, 3):
        raise ValueError('expected one RGB image and one RGB prediction')
    if valid.shape != (1, 1, *prior.shape[-2:]):
        raise ValueError('valid mask must have the native prediction size')
    ph, pw = prior.shape[-2:]
    u = max(1, min(c.max_upsample, round(image.shape[-1] / pw)))
    shape = (ph * u, pw * u)
    image = F.interpolate(image, shape, mode='area' if image.shape[-1] >= shape[1] else 'bilinear').clamp(0, 1)
    encoded = linear_to_srgb(image)
    low = F.avg_pool2d(encoded, u, u) if u > 1 else encoded
    fallback = F.interpolate(prior, shape, mode='bilinear', align_corners=False)
    pcode, query_prior = linear_to_srgb(prior), linear_to_srgb(fallback)
    eroded = -F.max_pool2d(-valid, 3, 1, 1)
    window = torch.hann_window(c.tile_size * u, periodic=False, device=prior.device).clamp_min(1e-3)
    window = window[:, None] * window[None, :]
    acc = torch.zeros(1, 3, *shape, device=prior.device, dtype=prior.dtype)
    den = torch.zeros(1, 1, *shape, device=prior.device, dtype=prior.dtype)
    for ty in _tiles(ph, c.tile_size, c.tile_stride):
        for tx in _tiles(pw, c.tile_size, c.tile_stride):
            bottom, right = min(ty + c.tile_size, ph), min(tx + c.tile_size, pw)
            if valid[0, 0, ty:bottom, tx:right].sum() < 8:
                continue
            top0, left0 = max(ty - c.context, 0), max(tx - c.context, 0)
            bottom0, right0 = min(bottom + c.context, ph), min(right + c.context, pw)
            ys, xs = torch.where(eroded[0, 0, top0:bottom0, left0:right0] > .5)
            if not len(ys):
                continue
            if len(ys) > c.candidates:
                take = torch.linspace(0, len(ys) - 1, c.candidates, device=prior.device).round().long()
                ys, xs = ys[take], xs[take]
            ys, xs = ys + top0, xs + left0
            ci, cp, pc = low[0, :, ys, xs].T, prior[0, :, ys, xs].T, pcode[0, :, ys, xs].T
            h, w, yy0, xx0 = (bottom - ty) * u, (right - tx) * u, ty * u, tx * u
            query = encoded[0, :, yy0:yy0+h, xx0:xx0+w].reshape(3, -1).T
            pq = query_prior[0, :, yy0:yy0+h, xx0:xx0+w].reshape(3, -1).T
            yy, xx = torch.meshgrid(torch.arange(h, device=prior.device), torch.arange(w, device=prior.device), indexing='ij')
            gy, gx = (yy.flatten().float() + .5) / u + ty, (xx.flatten().float() + .5) / u + tx
            d2 = (gy[:, None] - (ys[None].float() + .5))**2 + (gx[:, None] - (xs[None].float() + .5))**2
            cost = torch.cdist(query, ci)**2 / (2 * c.rgb_sigma**2) + d2 / (2 * c.spatial_sigma**2)
            cost += torch.cdist(pq, pc)**2 / (2 * c.prior_sigma**2)
            colour = (torch.softmax(-cost, 1) @ cp).T.reshape(3, h, w)
            win = window[:h, :w]
            acc[0, :, yy0:yy0+h, xx0:xx0+w] += colour * win
            den[0, :, yy0:yy0+h, xx0:xx0+w] += win
    selected = torch.where(den > 0, acc / den.clamp_min(1e-6), fallback)
    mask = F.interpolate(valid, shape, mode='nearest')
    interior = -F.max_pool2d(-mask, 4*u+1, 1, 2*u) > .5
    return torch.where(interior, selected, fallback).clamp(0, 1)
