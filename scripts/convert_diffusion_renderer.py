"""DiffusionRenderer G-buffers (scripts/predict_diffusion_renderer.sh) -> the prior layout of prep_ir.prior.

Each {basecolor,roughness,metallic}_single map (704x1280, letterboxed by DiffReg-PBIR's staging) is mapped back to the
training view's resolution with DiffReg-PBIR's inverse transform (neural_surface_recon/lib/diffrender_transforms.py).
Albedo stays sRGB-encoded like MVInverse's; roughness and metallic are the first channel; alpha is the view's mask.
Views follow the order of DiffReg-PBIR's train split, which equals the training views' order.

usage: python scripts/convert_diffusion_renderer.py --data data/s4r/hotdog --out work/s4r/hotdog/priors/diffusion_renderer
"""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(os.environ.get('PREP_IR_DIFFREG', ROOT / 'third_party/DiffReg-PBIR')).resolve()))
from neural_surface_recon.lib.diffrender_transforms import transform_diffrender_invrender as to_view  # noqa: E402


def view_alpha(data, dataset, index, stem):
    """(prediction key, alpha uint8) of the index-th training view."""
    if dataset == 's4r':
        return f'train_{index:03d}', iio.imread(data / 'train' / f'{index:03d}.png')[..., 3]
    if dataset == 'tir':
        return f'train_{index:03d}', iio.imread(data / f'train_{index:03d}' / 'rgba.png')[..., 3]
    alpha = iio.imread(data / 'train' / f'{stem}_mask.png')
    return f'train_{stem}', alpha[..., 0] if alpha.ndim == 3 else alpha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.data / 'dataset.json').read_text())
    dr = (a.data / info['diffreg_data']).resolve()
    cameras = json.loads((dr / 'cameras.json').read_text())
    mode = json.loads((dr / 'diffrender_mode.json').read_text())
    kw = dict(mode=mode['mode'], crop_offset=tuple(mode.get('crop_offset', (0, 0))))
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, fi in enumerate(cameras['split']['train']):
        stem = Path(cameras['frames'][fi]['path']).stem
        key, alpha = view_alpha(a.data, info['dataset'], i, stem)
        H, W = alpha.shape
        maps = {}
        for k in ('basecolor', 'roughness', 'metallic'):
            x = torch.tensor(iio.imread(dr / f'{k}_single' / f'{stem}.png')[..., :3].astype(np.float32) / 255)[None]
            maps[k] = to_view(x, H, W, **kw)[0].clamp(0, 1).numpy()
        np.savez(a.out / 'predictions' / f'{key}.npz', albedo=maps['basecolor'].astype(np.float16), roughness=maps['roughness'][..., 0].astype(np.float16),
                 metallic=maps['metallic'][..., 0].astype(np.float16), alpha=alpha.astype(np.uint8))
        mapping[key] = dict(npz=f'{key}.npz', orig_hw=[H, W], model_hw=[H, W])
    (a.out / 'mapping.json').write_text(json.dumps(mapping, indent=1))
    print(f'{len(mapping)} views -> {a.out}')


if __name__ == '__main__':
    main()
