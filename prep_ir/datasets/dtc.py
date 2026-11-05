"""DTC-Synthetic (DiffReg-PBIR's release, https://github.com/gerwang/DiffReg-PBIR/releases/tag/dtc-synthetic-v1, one
zip per scene, already in DiffReg-PBIR's format) -> the layout used here.

    <out>/<scene>/transforms_train.json, train/   the 800x800 linear HDR training frames and soft masks (links);
                                                  NeRF cameras = DiffReg to_world with x and z columns negated
    <out>/<scene>/mvinverse_input/                sRGB PNGs of the training frames (MVInverse input)
    <out>/diffreg/<scene>/                        DiffReg-PBIR input (the release, linked; ground-truth relit images)
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .common import IDENTITY, link, read_exr_rgb, write_json

SCENES = ['Block_B007GE75HY_RedBlue_scene002', 'Hammer_B000FK3VZ6_Wood_Lighting002', 'Mallard_B082D168CK_MintGreen_Lighting002',
          'Shoe_B000ZP6MIY_Navy7L_TU_Lighting002', 'TeaPot_B084G3K8TD_YellowBlackSunflowers_TU_scene002',
          'TeaPot_B08HSDHBM4_BlackGoldLeaves_Lighting002', 'TeaPot_B094FQW6Q4_EmeraldGoldTop_scene002']
FLIP = np.diag([-1., 1., -1., 1.]).astype(np.float32)


def mvinverse_png(exr, mask_png, out):
    f = np.clip(read_exr_rgb(exr), 0, None)
    rgb = np.where(f > .0031308, np.power(np.maximum(f, .0031308), 1 / 2.4) * 1.055 - .055, 12.92 * f)
    alpha = np.asarray(Image.open(mask_png).convert('L'), np.float32) / 255
    Image.fromarray((np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], -1) * 255).round().astype(np.uint8)).save(out)


def prepare(src, out_root, scene):
    release, out_root = Path(src) / scene, Path(out_root)
    out = out_root / scene
    cameras = json.loads((release / 'cameras.json').read_text())
    if abs(cameras['fx'] - cameras['fy']) > 1e-6 or cameras['split'].get('test'):
        raise ValueError(f'{release}: unexpected cameras.json')
    width = 2 * cameras['cx']
    frames = []
    for i in cameras['split']['train']:
        f = cameras['frames'][i]
        stem = Path(f['path']).stem
        link(release / f['path'], out / 'train' / f'{stem}.exr')
        link(release / f['mask'], out / 'train' / f'{stem}_mask.png')
        frames.append(dict(file_path=f'train/{stem}', transform_matrix=(np.array(f['to_world'], np.float64) @ FLIP).tolist()))
    meta = dict(camera_angle_x=float(2 * np.arctan(.5 * width / cameras['fx'])), frames=frames)
    write_json(out / 'transforms_train.json', meta)
    (out / 'mvinverse_input' / 'train').mkdir(parents=True, exist_ok=True)
    for fr in frames:
        stem = Path(fr['file_path']).name
        if not (out / 'mvinverse_input' / 'train' / f'{stem}.png').exists():
            mvinverse_png(out / 'train' / f'{stem}.exr', out / 'train' / f'{stem}_mask.png', out / 'mvinverse_input' / 'train' / f'{stem}.png')
    write_json(out / 'mvinverse_input' / 'transforms_train.json', meta)
    nsr = out_root / 'diffreg' / scene   # named exactly <scene>; files the G-buffer staging rewrites are copied
    nsr.mkdir(parents=True, exist_ok=True)
    for x in release.iterdir():
        if x.name in ('cameras.json', 'diffrender_mode.json'):
            shutil.copy(x, nsr / x.name)
        else:
            link(x, nsr / x.name)
    write_json(out / 'dataset.json', dict(dataset='dtc', scene=scene, nsr_to_world=IDENTITY, textured_eval=True,
                                          diffreg_data=f'../diffreg/{scene}', mvinverse_input='mvinverse_input'))
