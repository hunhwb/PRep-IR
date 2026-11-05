"""Environment map: Neural-PBIR's two-phase parameterization and direct background observations.

Phase 1, `SGEnvmap`: K spherical Gaussians on fixed Fibonacci axes, L(d) = sum_k mu_k exp(lambda_k (d . xi_k - 1)),
with log lambda and log mu learned. Phase 2, `TextureEnvmap`: from the phase-1 result, a 64x128 log-radiance texture
under a Large Steps change of variables on the sphere grid. Both re-implement the parameterization of Neural-PBIR
(Sun et al., ICCV 2023; pbir/models/envmap_sg.py and utils_sg.py in https://github.com/facebookresearch/DigitalTwinCatalog,
CC BY-NC 4.0); the background averaging follows its neural_distillation/distill_util.py:compute_avg_bg.

Every forward() returns the H x W x 3 log radiance in the renderer's lat-long layout: texel (i, j) is the emitter
direction theta = pi i / (H - 1), phi = 2 pi (j + .5) / W, v = (sin theta sin phi, cos theta, -sin theta cos phi).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .image import linear_to_srgb
from .largesteps import SPDSolver, inverse, laplacian, sphere_grid_edges, system


def texel_directions(height, width, device='cuda'):
    """Emitter-frame unit directions of the lat-long texels (H, W, 3)."""
    theta = torch.arange(height, device=device, dtype=torch.float32) * math.pi / (height - 1)
    phi = (torch.arange(width, device=device, dtype=torch.float32) + .5) * 2 * math.pi / width
    t, p = torch.meshgrid(theta, phi, indexing='ij')
    return torch.stack((t.sin() * p.sin(), t.cos(), -t.sin() * p.cos()), -1)


def fibonacci_sphere(n):
    """n roughly uniform unit vectors (y up), as Neural-PBIR's utils_sg.fibonacci_sphere."""
    i = np.arange(n, dtype=np.float64)
    y = 1 - (i / (n - 1)) * 2
    r = np.sqrt(1 - y * y)
    t = math.pi * (3. - math.sqrt(5.)) * i
    return np.stack((np.cos(t) * r, y, np.sin(t) * r), 1).astype(np.float32)


class SGEnvmap(nn.Module):
    def __init__(self, shape, target, lobes=128, fit_steps=1000):
        """shape: (H, W) of the output texture; target: linear radiance (H', W', 3) the lobes are fitted to first
        (Adam, lr .1, MSE over all texels, as Neural-PBIR's Envmap2SG)."""
        super().__init__()
        device = target.device
        self.register_buffer('axes', torch.as_tensor(fibonacci_sphere(lobes), device=device))
        self.log_lambda = nn.Parameter(torch.full((lobes, 1), math.log(20.), device=device))
        self.log_mu = nn.Parameter(torch.full((lobes, 3), math.log(.5), device=device))
        dirs = texel_directions(*target.shape[:2], device=device).reshape(-1, 3)
        target = target.float().reshape(-1, 3)
        optimizer = torch.optim.Adam(self.parameters(), lr=.1)
        for _ in range(fit_steps):
            optimizer.zero_grad()
            (self.radiance(dirs) - target).square().mean().backward()
            optimizer.step()
        self.register_buffer('dirs', texel_directions(*shape, device=device))

    def radiance(self, dirs):
        weights = torch.exp(self.log_lambda.exp().T * (dirs @ self.axes.T - 1))
        return weights @ self.log_mu.exp()

    def forward(self):
        return self.radiance(self.dirs).clamp_min(1e-6).log()


class TextureEnvmap(nn.Module):
    """log E = log E0 + (I + lambda K / mean degree)^-1 w, K the sphere-grid Laplacian; w starts at zero."""

    def __init__(self, reference, strength):
        super().__init__()
        H, W, _ = reference.shape
        edges, weights = sphere_grid_edges(H, W)
        self.solver = SPDSolver(system(laplacian(H * W, edges, weights), strength, 'mean_positive_degree'), reference.device)
        self.register_buffer('reference', reference.detach().clone())
        self.w = nn.Parameter(torch.zeros_like(self.reference))

    def forward(self):
        H, W, C = self.reference.shape
        return self.reference + inverse(self.w.reshape(H * W, C), self.solver).reshape(H, W, C)


def envmap_lookup(radiance, dirs, env_to_world):
    """Linear radiance of a lat-long envmap (H, W, 3) along world directions (N, 3), bilinear as Mitsuba's envmap
    emitter evaluates it (texel centres in longitude with wrap-around, texel rows on the pole-inclusive grid)."""
    inv = torch.as_tensor(np.linalg.inv(np.asarray(env_to_world, np.float64))[:3, :3], dtype=dirs.dtype, device=dirs.device)
    v = dirs @ inv.T
    u = torch.atan2(v[:, 0], -v[:, 2]) / (2 * math.pi)
    u = u - torch.floor(u)
    t = torch.acos(v[:, 1].clamp(-1, 1)) / math.pi
    H, W, _ = radiance.shape
    img = torch.cat((radiance[:, -1:], radiance, radiance[:, :1]), 1).permute(2, 0, 1)[None]
    grid = torch.stack(((u * W + 1) / (W + 2) * 2 - 1, (t * (H - 1) + .5) / H * 2 - 1), -1)[None, None]
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T


def pixel_directions(transform, fov_deg, size, ij):
    """World directions of the camera rays through pixel centres ij (N, 2: row, column) of a NeRF camera."""
    w, h = size
    tx = math.tan(math.radians(fov_deg) / 2)
    ty = tx * h / w
    x = (2 * (ij[:, 1] + .5) / w - 1) * tx
    y = (1 - 2 * (ij[:, 0] + .5) / h) * ty
    d = torch.stack((x, y, -torch.ones_like(x)), -1) @ torch.as_tensor(np.asarray(transform)[:3, :3], dtype=x.dtype).T
    return d / d.norm(dim=-1, keepdim=True)


class BackgroundObservations:
    """Training-view pixels that see the environment directly: outside the object mask dilated by `dilate` px,
    nonzero, on a `stride` grid. Each gives a world ray direction and a linear HDR radiance."""

    def __init__(self, views, stride=4, dilate=5, device='cuda'):
        if not views.hdr:
            raise ValueError('background observations need HDR frames that keep the surroundings')
        dirs, radiance = [], []
        for k, path in enumerate(views.paths):
            x = views.image(path).float()
            mask = views.batch(k, device='cpu')['mask'][..., 0].float()
            grown = F.max_pool2d(mask[None, None], 2 * dilate + 1, 1, dilate)[0, 0] > .5
            keep = ~grown & (x.sum(-1) > 0)
            grid = torch.zeros_like(keep)
            grid[stride // 2::stride, stride // 2::stride] = True
            keep &= grid
            dirs.append(pixel_directions(views.transforms[k], views.fov, views.size, keep.nonzero().float()))
            radiance.append(x[keep])
        self.dirs = torch.cat(dirs).to(device)
        self.radiance = torch.cat(radiance).to(device)

    def loss(self, log_envmap, env_to_world, samples, seed):
        """Mean sRGB L1 between the envmap and `samples` random observations (the data term's colour space)."""
        g = torch.Generator(device=self.dirs.device).manual_seed(int(seed))
        idx = torch.randint(len(self.dirs), (min(int(samples), len(self.dirs)),), generator=g, device=self.dirs.device)
        pred = envmap_lookup(log_envmap.exp(), self.dirs[idx], env_to_world)
        return (linear_to_srgb(pred) - linear_to_srgb(self.radiance[idx])).abs().mean()

    def average(self, env_to_world, height=128, width=256):
        """Mean observed radiance per lat-long texel (emitter frame); unobserved texels are zero."""
        inv = torch.as_tensor(np.linalg.inv(np.asarray(env_to_world, np.float64))[:3, :3], dtype=self.dirs.dtype, device=self.dirs.device)
        v = self.dirs @ inv.T
        u = torch.atan2(v[:, 0], -v[:, 2]) / (2 * math.pi)
        u = u - torch.floor(u)
        t = torch.acos(v[:, 1].clamp(-1, 1)) / math.pi
        flat = (t * (height - 1)).round().long().clamp(0, height - 1) * width + (u * width).floor().long().clamp(0, width - 1)
        total = torch.zeros(height * width, 3, device=self.dirs.device).index_add_(0, flat, self.radiance.float())
        count = torch.zeros(height * width, device=self.dirs.device).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        mean = torch.where((count > 0)[:, None], total / count.clamp_min(1)[:, None], torch.zeros_like(total))
        return mean.reshape(height, width, 3)
