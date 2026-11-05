"""Synthetic4Relight / TensoIR-Synthetic evaluation (supplement, "Evaluation Protocol"), adapted from the public
TensoIR implementation (https://github.com/Haian-Jin/TensoIR).

All test views at 800x800, rendered with the method's own renderer, environment hidden from camera rays:
  * NVS under the learned envmap (128 spp) and relighting under the dataset's test envmaps (512 spp), sRGB, scored on
    white backgrounds with the predicted coverage;
  * one RGB albedo scale per scene, the per-channel median GT/prediction ratio over the GT foreground of all test
    views, applied to the albedo images and to the material before relighting;
  * PSNR from the RGB-mean MSE of each view, averaged over views (relighting also over envmaps); albedo PSNR on
    x^(1/2.2), albedo SSIM in linear space;
  * roughness MAE over the GT foreground; TensoIR normal error over the whole image with a +z background.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import Views
from ..image import linear_to_srgb, read_exr, read_png, srgb_to_linear, write_png
from ..scene import dataset_info
from .metrics import score, ssim


class TestViews(Views):
    """Ground truth at its native bit depth and size (no resampling)."""

    def image(self, path, rgba=False):
        a = read_png(path, rgba)
        if tuple(a.shape[1::-1]) != tuple(self.size):
            raise ValueError(f'{path}: expected {self.size}')
        return torch.from_numpy(a.copy())

    def material_paths(self, index):
        path = self.paths[index]
        stem = path.stem.removesuffix('_rgba')
        for albedo, roughness in ((path.parent / 'albedo.png', path.parent / 'roughness.png'),
                                  (path.parent / f'{stem}_albedo.png', path.parent / f'{stem}_rough.png')):
            if albedo.is_file() and roughness.is_file():
                return albedo, roughness
        raise FileNotFoundError(f'material ground truth for {path}')

    def albedo(self, index, srgb_encoded):
        """Linear GT albedo composited onto white."""
        raw = self.image(self.material_paths(index)[0], True).cuda()
        rgb = srgb_to_linear(raw[..., :3]) if srgb_encoded else raw[..., :3]
        return rgb * raw[..., 3:] + 1 - raw[..., 3:]

    def roughness(self, index):
        return self.image(self.material_paths(index)[1])[..., :1].cuda()

    def normal_image(self, index):
        return self.image(self.paths[index].parent / 'normal.png', True).cuda()

    def relight_target(self, index, envmap_path):
        path, name = self.paths[index], Path(envmap_path).stem
        for p in (self.root / 'test_rli' / f'{name}_{path.stem.removesuffix("_rgba")}.png', path.parent / f'rgba_{name}.png'):
            if p.is_file():
                x = self.image(p, True).cuda()
                return x[..., :3] * x[..., 3:4] + self.background * (1 - x[..., 3:4])
        raise FileNotFoundError(f'relighting target of {path} under {name}')


def load_envmap(path):
    """A test envmap (texel-centred lat-long rows) resampled to the renderer's pole-inclusive rows, on the GPU."""
    rgb = torch.from_numpy(read_exr(path)).cuda()
    h, w, _ = rgb.shape
    y = torch.linspace(-1, 1, h, device=rgb.device)
    x = (torch.arange(w, device=rgb.device) + .5) * 2 / w - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None]
    return F.grid_sample(rgb.permute(2, 0, 1)[None], grid, align_corners=False, padding_mode='border')[0].permute(1, 2, 0).contiguous()


def load_result(path):
    with np.load(Path(path) / 'result.npz') as z:
        return {k: torch.from_numpy(z[k]).cuda() for k in ('vertices', 'faces', 'albedo', 'roughness', 'metallic', 'envmap')}


@torch.no_grad()
def evaluate(result_dir, data_dir, out_dir, view_stride=1, save_images=False, log=print):
    from ..render import Renderer
    start = time.time()
    info = dataset_info(data_dir)
    kind = info['dataset']
    if kind not in ('s4r', 'tir'):
        raise ValueError(f'{data_dir}: synthetic evaluation covers s4r and tir, not {kind}')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = load_result(result_dir)
    gt = TestViews(data_dir, 'test', 800, 'white')
    black = TestViews(data_dir, 'test', 800, 'black')
    # the training renderer's state (derivative sample counts only affect gradients, kept for an identical integrator)
    renderer = Renderer(r['vertices'], r['faces'], metallic=r['metallic'], envmap_shape=tuple(r['envmap'].shape[:2]), derivative_spp=(32, 8, 8))
    v, a, rough, env = r['vertices'], r['albedo'], r['roughness'], r['envmap']
    scored = list(range(0, len(gt), view_stride))
    rows = {i: dict(index=i) for i in scored}

    # pass 1, every test view: material buffers (the albedo scale always uses all views)
    albedo_maps, opacity_maps = [], []
    for i in range(len(gt)):
        aov = renderer.aovs(v, a, rough, env, gt, i, spp=16, seed=20000 + i)
        op = aov['opacity'].clamp(0, 1)
        albedo_maps.append(aov['albedo'].cpu())
        opacity_maps.append(op.cpu())
        if i in rows:
            mask = gt.batch(i)['mask']
            row = rows[i]
            row['roughness_mae'] = float(((aov['roughness'] - gt.roughness(i)).abs() * mask).sum() / mask.sum())
            if kind == 'tir':
                nm = gt.normal_image(i)
                z = torch.tensor([0., 0., 1.], device='cuda')
                tn = F.normalize((nm[..., :3] * 2 - 1) * nm[..., 3:] + z * (1 - nm[..., 3:]), dim=-1)
                pn = F.normalize(F.normalize(aov['normal'], dim=-1) * op + z * (1 - op), dim=-1)
                angle = (pn * tn).sum(-1).clamp(-1, 1).acos() * 180 / np.pi
                row.update(normal_degrees=float(angle.mean()), normal_degrees_foreground=float(angle[mask[..., 0] > 0].mean()))
            image = linear_to_srgb((renderer.render(v, a, rough, env, black, i, spp=128, seed=10000 + i)[..., :3] + 1 - op).clamp(0, 1)).clamp(0, 1)
            row.update({f'nvs_{k}': x for k, x in score(image, gt.batch(i)['rgb']).items()})
            if save_images:
                (out / 'nvs').mkdir(exist_ok=True)
                write_png(out / 'nvs' / f'{i:03d}.png', image)
        if i % 20 == 0:
            log(f'views {i}/{len(gt)}  {time.time() - start:.0f} s')

    # one RGB albedo scale: per-channel median of GT / prediction over the GT foreground of all views
    ratios = [[], [], []]
    for i in range(len(gt)):
        target = gt.albedo(i, srgb_encoded=kind == 's4r')
        pred = (albedo_maps[i].cuda() + 1 - opacity_maps[i].cuda()).clamp(0, 1)
        valid = gt.batch(i)['mask'][..., 0] > 0
        for c in range(3):
            support = valid & (target[..., c] > 0) & (pred[..., c] > 1e-6)   # zero-signal pixels do not constrain a scale
            ratios[c].append((target[..., c][support] / pred[..., c][support]).cpu())
    scale = torch.stack([torch.cat(x).median() for x in ratios]).cuda()
    log(f'albedo scale {scale.tolist()}')
    for i in scored:
        target = gt.albedo(i, srgb_encoded=kind == 's4r')
        pred = (albedo_maps[i].cuda() + 1 - opacity_maps[i].cuda()).clamp(0, 1)
        valid = gt.batch(i)['mask'][..., 0] > 0
        albedo = torch.ones_like(pred)
        albedo[valid] = (pred[valid] * scale).clamp(0, 1)
        s = score(albedo.pow(1 / 2.2), target.clamp_min(0).pow(1 / 2.2))
        rows[i].update(albedo_psnr=s['psnr'], albedo_mse=s['mse'], albedo_ssim=ssim(albedo, target))
        if save_images:
            (out / 'albedo').mkdir(exist_ok=True)
            write_png(out / 'albedo' / f'{i:03d}.png', albedo)

    # relighting with the scaled albedo
    envmaps = [Path(data_dir) / p for p in info['relight_envmaps']]
    for path in envmaps:
        light = load_envmap(path)
        for i in scored:
            op = opacity_maps[i].cuda()
            rgb = renderer.render(v, a * scale, rough, light, black, i, spp=512, seed=30000 + i)[..., :3]
            pred = torch.where(op > .5, linear_to_srgb((rgb / op.clamp_min(1e-6)).clamp(0, 1)).clamp(0, 1), torch.ones_like(rgb))
            s = score(pred, gt.relight_target(i, path))
            rows[i].update({f'relight_{path.stem}_{k}': s[k] for k in ('psnr', 'ssim', 'mse')})
            if save_images:
                (out / f'relight_{path.stem}').mkdir(exist_ok=True)
                write_png(out / f'relight_{path.stem}' / f'{i:03d}.png', pred)
        log(f'relight {path.stem} done  {time.time() - start:.0f} s')

    table = list(rows.values())
    mean = lambda k: float(np.mean([row[k] for row in table]))
    keys = ['nvs_psnr', 'nvs_ssim', 'albedo_psnr', 'albedo_ssim', 'roughness_mae'] + (['normal_degrees', 'normal_degrees_foreground'] if kind == 'tir' else [])
    metrics = {k: mean(k) for k in keys}
    for k in ('psnr', 'ssim'):
        for path in envmaps:
            metrics[f'relight_{path.stem}_{k}'] = mean(f'relight_{path.stem}_{k}')
        metrics[f'relight_{k}'] = float(np.mean([metrics[f'relight_{p.stem}_{k}'] for p in envmaps]))
    summary = dict(metrics=metrics, albedo_scale=scale.tolist(), views=scored, result=str(Path(result_dir).resolve()),
                   data=str(Path(data_dir).resolve()), seconds=time.time() - start, rows=table)
    (out / 'evaluation.json').write_text(json.dumps(summary, indent=1) + '\n')
    log(json.dumps(metrics, indent=1))
    return metrics
