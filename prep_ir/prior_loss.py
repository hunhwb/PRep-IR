"""Control "prior as a loss": DiffReg-PBIR's joint-bilateral material regularizer, adapted to mesh vertices.

Adapted from DifferentKeyJointBilateralLoss of DiffReg-PBIR (Ling et al., ECCV 2026;
https://github.com/gerwang/DiffReg-PBIR, diffusion_reg/models/losses.py, CC BY-NC 4.0). DiffReg-PBIR applies it per
training view to rendered material buffers; here the same guide, filter and weights act on the prior-supported vertices:

    x     = log max(albedo, 1e-3), log max(roughness, 1e-3)
    guide = fused prior: albedo sRGB -> linear scaled so its luminance q95 is .9, metallic, roughness; / 0.02
    blur  = permutohedral Gaussian filter of the guide (splat, [1 2 1]/4 along the d+1 lattice axes, slice),
            area-weighted (a vertex stands for the pixels covering it) and normalized
    loss  = frame fraction * sum_v a_v (|x_b - blur_b| b / 3 + .1 |x_r - blur_r|)    (a_v: area share)

Metallic is fixed in the paper's experiments, so its term is absent. The L1 has a 1e-5 dead zone so that float
rounding of a constant field does not produce full-size gradients of random sign.
"""
import torch

from .image import srgb_to_linear
from .lattice import gaussian_blur_operators
from .material import _SparseMatmul, to_device_csr
from .mesh import vertex_areas

GUIDE_STD = .02
ROUGHNESS_WEIGHT = .1
LOG_FLOOR = 1e-3
DEADZONE = 1e-5


class PriorBilateralLoss:
    def __init__(self, prior, vertices, faces, device='cuda'):
        observations = torch.as_tensor(prior['observations']).reshape(-1)
        idx = torch.where(observations > 0)[0]
        self.supported = idx.to(device)
        material = torch.as_tensor(prior['features']).double()[idx]
        albedo = srgb_to_linear(material[:, :3].clamp(0, 1))
        q95 = float((albedo @ torch.tensor([.2126, .7152, .0722], dtype=torch.float64)).quantile(.95))
        if not q95 > 0:
            raise ValueError('prior albedo luminance q95 is zero')
        guide = torch.cat((albedo * (.9 / q95), material[:, 4:5], material[:, 3:4]), 1) / GUIDE_STD
        S, blurs = gaussian_blur_operators(guide.numpy())
        self.S = to_device_csr(S, device, torch.float32)
        self.ST = to_device_csr(S.T, device, torch.float32)
        self.blurs = [to_device_csr(B, device, torch.float32) for B in blurs]   # each symmetric
        area = vertex_areas(torch.as_tensor(vertices), torch.as_tensor(faces))[idx]
        self.area = (area / area.mean()).float().to(device)[:, None]
        self.norm = self._filter(self.area).clamp_min(1e-12)
        self.weight = (area / area.sum()).float().to(device)[:, None]

    def _filter(self, x):
        y = _SparseMatmul.apply(x, self.S, self.ST)
        for B in self.blurs:
            y = _SparseMatmul.apply(y, B, B)
        return _SparseMatmul.apply(y, self.ST, self.S)

    def _blur(self, x):
        return self._filter(self.area * x) / self.norm

    def __call__(self, albedo, roughness, frame_fraction):
        """albedo (N, 3) linear and roughness (N, 1) perceptual of every vertex; frame_fraction: mean object share
        of the step's frames (DiffReg-PBIR averages over whole frames)."""
        b = albedo.index_select(0, self.supported)
        r = roughness.index_select(0, self.supported)
        xb, xr = b.clamp_min(LOG_FLOOR).log(), r.clamp_min(LOG_FLOOR).log()
        l1 = lambda d: (d.abs() - DEADZONE).clamp_min(0)
        term_b = (l1(xb - self._blur(xb)) * b.detach() * self.weight).sum() / 3
        term_r = (l1(xr - self._blur(xr)) * self.weight).sum()
        return frame_fraction * (term_b + ROUGHNESS_WEIGHT * term_r)
