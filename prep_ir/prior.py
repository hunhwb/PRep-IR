"""Fusion of per-view material predictions onto the mesh vertices (Sec. 4.2 of the paper).

Prediction layout (written by scripts/predict_mvinverse.py and scripts/convert_diffusion_renderer.py):
    <prior>/mapping.json                 {view key: {"npz": file name, ...}}
    <prior>/predictions/<key>.npz        albedo (h, w, 3) sRGB-encoded, roughness (h, w), metallic (h, w), alpha (h, w) uint8
View keys: `train_<stem>` for frames stored as train/<stem>, else the frame's directory name (TensoIR: train_000).

Fusion over the training views: a prediction pixel is valid where its alpha > .95, its values are finite and in
[0, 1], and the dataset mask > .95; the valid mask is eroded by a 5x5 min filter. Predictions are resized bilinearly
to the camera resolution and sampled at each vertex's projection (nearest). A vertex counts as seen by a view if it
projects onto a valid pixel, faces the camera (cosine > .3) and is its first hit (tolerance .002 x bounding-box
diagonal). Seen views are weighted by cosine^2; the result is the weighted mean of the 5 channels (albedo RGB,
roughness, metallic) and the weighted cross-view variance averaged over the channels.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Views
from .mesh import angle_weighted_normals
from .image import linear_to_srgb, srgb_to_linear, read_exr
from .refinement import refine_albedo


def guidance_image(views, index, mask, paths=None):
    """Linear input RGB, with foreground-percentile exposure for HDR guidance.

    Optional paths map view keys to original HDR captures when the reconstruction
    dataset contains display PNGs. This changes guidance only, not training data.
    """
    path = Path(paths[view_key(views.paths[index])]) if paths else views.paths[index]
    if path.suffix.lower() in ('.exr', '.hdr'):
        rgb = read_exr(path)
        fg = F.interpolate(mask[None, None].float(), rgb.shape[:2], mode='nearest')[0, 0].cpu().numpy() > .95
        if not fg.any():
            raise ValueError(f'no foreground for HDR guidance: {path}')
        exposure = .8 / max(float(np.quantile(rgb[fg], .99)), .01)
        rgb = torch.from_numpy((rgb * exposure).clip(0, 1))
    else:
        from PIL import Image
        rgb = srgb_to_linear(torch.from_numpy(np.array(Image.open(path).convert('RGB'), copy=True)).float() / 255)
    return rgb.cuda().permute(2, 0, 1)[None]


def view_key(path):
    path = Path(path)
    return f'train_{path.stem}' if path.parent.name in ('train', 'test') else path.parent.name


def load_prediction(prior_dir, key, mapping):
    with np.load(Path(prior_dir) / 'predictions' / mapping[key]['npz'], allow_pickle=False) as z:
        values = np.concatenate((z['albedo'], z['roughness'][..., None], z['metallic'][..., None]), axis=-1)
        return torch.tensor(values, device='cuda', dtype=torch.float32), torch.tensor(z['alpha'], device='cuda', dtype=torch.float32).squeeze() / 255


@torch.no_grad()
def fuse_prior(renderer, vertices, faces, views_root, prior_dir, min_facing=.3, erode_px=5, log=print,
               refinement=None, guidance_paths=None):
    """Returns dict(features (N, 5), observations (N,), variance (N,)) as float32 CPU tensors."""
    mapping = json.loads((Path(prior_dir) / 'mapping.json').read_text())
    views = Views(views_root, 'train')
    v = torch.as_tensor(vertices, device='cuda')
    f = torch.as_tensor(faces, device='cuda')
    normals = angle_weighted_normals(v, f)
    diagonal = (v.amax(0) - v.amin(0)).norm()
    sums = torch.zeros((len(v), 5), device='cuda')
    squares = sums.clone()
    mass = torch.zeros_like(v[:, :1])
    observations = mass.clone()
    w, h = views.size
    for i, path in enumerate(views.paths):
        r, alpha = load_prediction(prior_dir, view_key(path), mapping)
        valid = (alpha > .95) & torch.isfinite(r).all(-1) & (r >= 0).all(-1) & (r <= 1).all(-1)
        r = torch.where(valid[..., None], r, 0.)
        native = r
        r = F.interpolate(r.permute(2, 0, 1)[None], size=(h, w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        valid = F.interpolate(valid[None, None].float(), size=(h, w), mode='nearest')[0, 0] > .5
        mask = views.batch(i)['mask'][..., 0]
        valid &= mask > .95
        valid = -F.max_pool2d(-valid[None, None].float(), erode_px, 1, erode_px // 2)[0, 0] > .5
        refined = None
        if refinement is not None:
            native_valid = F.interpolate(valid[None, None].float(), native.shape[:2], mode='nearest')
            albedo = srgb_to_linear(native[..., :3]).permute(2, 0, 1)[None]
            refined = refine_albedo(guidance_image(views, i, mask, guidance_paths), albedo, native_valid, refinement)
        # project the vertices into the view (NeRF camera: -z forward, y up)
        pose = torch.tensor(views.transforms[i], device='cuda')
        camera = (v - pose[:3, 3]) @ pose[:3, :3]
        depth = -camera[:, 2]
        focal = .5 * w / np.tan(np.radians(views.fov) / 2)
        uv = torch.stack((focal * camera[:, 0] / depth.clamp_min(1e-8) + w / 2, -focal * camera[:, 1] / depth.clamp_min(1e-8) + h / 2), 1)
        grid = (2 * uv / uv.new_tensor([w, h]) - 1)[None, None]
        packed = torch.cat((r, valid[..., None].float()), -1).permute(2, 0, 1)[None]
        sampled = F.grid_sample(packed, grid, mode='nearest', align_corners=False)[0, :, 0].T
        if refined is not None:
            # Interpolate linear albedo first, then encode before cross-view fusion.
            value = F.grid_sample(refined, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T
            sampled[:, :3] = linear_to_srgb(value).clamp(0, 1)
        # visibility: in frame, on a valid pixel, facing the camera, and the first surface hit
        delta = v - pose[:3, 3]
        distance = delta.norm(dim=1)
        direction = delta / distance[:, None]
        hit, t = renderer.first_hits(pose[:3, 3], direction.T)
        facing = -(normals * direction).sum(-1)
        visible = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (sampled[:, 5] > .95) & (facing > min_facing)
        visible &= hit & ((t - distance).abs() < diagonal * .002)
        weight = (visible * facing.clamp_min(0).square())[:, None]
        sums += sampled[:, :5] * weight
        squares += sampled[:, :5].square() * weight
        mass += weight
        observations += visible[:, None]
        log(f'prior fusion {i + 1}/{len(views)}: coverage {float((mass > 0).float().mean()):.4f}')
    features = sums / mass.clamp_min(1e-8)
    variance = (squares / mass.clamp_min(1e-8) - features.square()).clamp_min(0).mean(-1)
    if not torch.isfinite(features).all() or not (observations > 0).any():
        raise RuntimeError('prior fusion produced no observations')
    result = dict(features=features.cpu(), observations=observations[:, 0].cpu(), variance=variance.cpu())
    if refinement is not None:
        result['refinement'] = dict(**refinement.as_dict(), variance_source='refined_per_view_material')
    return result
