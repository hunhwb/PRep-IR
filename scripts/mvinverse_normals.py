"""A DiffReg-PBIR data directory whose normal guidance (normal_single/) comes from MVInverse instead of
DiffusionRenderer, for the MVInverse configuration's surface reconstruction.

Every entry of the scene's DiffReg-PBIR data directory is linked except normal_single/, which is written from the
MVInverse predictions (camera-frame normals, the same frame as DiffusionRenderer's): resized bilinearly to the view
size, normalized, zero outside the alpha, letterboxed to 704x1280 with DiffReg-PBIR's forward transform and encoded
as n * .5 + .5. If the data directory has not been staged yet (no diffrender_mode.json), DiffReg-PBIR's staging step
is run first.

usage: python scripts/mvinverse_normals.py --data data/s4r/hotdog --prior work/s4r/hotdog/priors/mvinverse
                                           --out work/s4r/hotdog/nsr_data/mvinverse/hotdog   (the last part = scene name)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DIFFREG = Path(os.environ.get('PREP_IR_DIFFREG', ROOT / 'third_party/DiffReg-PBIR')).resolve()
sys.path.insert(0, str(DIFFREG))
from neural_surface_recon.lib.diffrender_transforms import transform_diffrender_invrender as to_view  # noqa: E402
from neural_surface_recon.lib.diffrender_transforms import transform_invrender_diffrender as to_diffrender  # noqa: E402


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True, help='MVInverse predictions (scripts/predict_mvinverse.py)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.data / 'dataset.json').read_text())
    if a.out.name != info['scene']:
        raise SystemExit(f'--out must end in the scene name ({info["scene"]})')
    dr = (a.data / info['diffreg_data']).resolve()
    if not (dr / 'diffrender_mode.json').is_file():   # DiffReg-PBIR's staging decides the letterbox mode
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=f'{DIFFREG}:{DIFFREG}/diffusion_reg/deps/irtk', CUDA_VISIBLE_DEVICES='')
            subprocess.run([sys.executable, '-m', 'scripts.preprocess.export_train_images', '--data_root', str(dr), '--output_dir', tmp],
                           cwd=DIFFREG, env=env, check=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for entry in dr.iterdir():
        if entry.name != 'normal_single' and not (a.out / entry.name).exists():
            os.symlink(entry.resolve(), a.out / entry.name)
    (a.out / 'normal_single').mkdir(exist_ok=True)
    cameras = json.loads((dr / 'cameras.json').read_text())
    mode = json.loads((dr / 'diffrender_mode.json').read_text())
    kw = dict(mode=mode['mode'], crop_offset=tuple(mode.get('crop_offset', (0, 0))))
    mapping = json.loads((a.prior / 'mapping.json').read_text())
    roundtrip = []
    for i, fi in enumerate(cameras['split']['train']):
        stem = Path(cameras['frames'][fi]['path']).stem
        key = f'train_{stem}' if info['dataset'] in ('orb', 'dtc') else f'train_{i:03d}'
        z = np.load(a.prior / 'predictions' / mapping[key]['npz'])
        H, W = mapping[key]['orig_hw']
        n = F.interpolate(torch.tensor(z['normal'].astype(np.float32)).permute(2, 0, 1)[None], size=(H, W), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        alpha = F.interpolate(torch.tensor(z['alpha'].astype(np.float32) / 255)[None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0] > .5
        n = unit(n) * alpha[..., None]
        png = to_diffrender(n[None] * .5 + .5, **kw)[0].clamp(0, 1)
        iio.imwrite(a.out / 'normal_single' / f'{stem}.png', (png * 255).round().to(torch.uint8).numpy())
        if i % max(1, len(cameras['split']['train']) // 8) == 0:   # check the encoding against DiffReg-PBIR's decoder
            got = unit(to_view(torch.tensor(iio.imread(a.out / 'normal_single' / f'{stem}.png')[..., :3].astype(np.float32) / 255 * 2 - 1)[None], H, W, **kw)[0])
            roundtrip.append(float((got[alpha] * unit(n)[alpha]).sum(-1).mean()))
    if np.mean(roundtrip) < .99:
        raise SystemExit(f'normal round trip cosine {np.mean(roundtrip):.4f} < .99')
    print(f'MVInverse normals -> {a.out / "normal_single"} (round trip cosine {np.mean(roundtrip):.4f})')


if __name__ == '__main__':
    main()
