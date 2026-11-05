"""Synthetic4Relight (InvRender release, https://github.com/zju3dv/InvRender) -> the layout used here.

    <out>/<scene>/train/NNN.png          training views: linear EXR clipped to [0, 1], sRGB, alpha = mask > .5,
                                         RGB multiplied by alpha, 8-bit
    <out>/<scene>/test/NNN.png, NNN_albedo.png, NNN_rough.png, test_rli/   test views and ground truth (links)
    <out>/<scene>/background/            the raw training EXRs, which show the training environment behind the
                                         object: direct observations of the illumination
    <out>/diffreg/<scene>/cameras.json   DiffReg-PBIR input for G-buffers and surface reconstruction
    <out>/envmaps/envmap{6,12}.exr       relighting envmaps (DiffReg-PBIR's envmap_mii)
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .common import NSR_TO_WORLD_MII, diffreg_cameras, link, mii_to_world, read_exr_rgb, srgb, write_json

SCENES = ['air_baloons', 'chair', 'hotdog', 'jugs']
ENVMAPS = ['envmap6', 'envmap12']


def training_png(exr, mask_png, out):
    rgb = srgb(read_exr_rgb(exr))
    mask = np.asarray(Image.open(mask_png))
    if mask.ndim == 3:
        mask = mask[..., 0]
    alpha = (mask.astype(np.float32) / 255. > .5).astype(np.float32)
    rgba = np.concatenate([rgb * alpha[..., None], alpha[..., None]], -1)
    Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8)).save(out)


def prepare(src, out_root, scene, diffreg):
    import json
    src, out_root = Path(src) / scene, Path(out_root)
    out = out_root / scene
    train = json.loads((src / 'transforms_train.json').read_text())
    test = json.loads((src / 'transforms_test.json').read_text())
    for split, meta in (('train', train), ('test', test)):
        write_json(out / f'transforms_{split}.json', meta)
    (out / 'train').mkdir(parents=True, exist_ok=True)
    for f in train['frames']:
        n = Path(f['file_path']).name
        if not (out / 'train' / f'{n}.png').exists():
            training_png(src / 'train' / f'{n}_rgb.exr', src / 'train' / f'{n}_mask.png', out / 'train' / f'{n}.png')
        link(src / 'train' / f'{n}_rgb.exr', out / 'background' / 'train' / f'{n}.exr')
        link(src / 'train' / f'{n}_mask.png', out / 'background' / 'train' / f'{n}_mask.png')
    write_json(out / 'background' / 'transforms_train.json', train)
    for f in test['frames']:
        n = Path(f['file_path']).name
        link(src / 'test' / f'{n}_rgba.png', out / 'test' / f'{n}.png')
        for k in ('albedo', 'rough'):
            link(src / 'test' / f'{n}_{k}.png', out / 'test' / f'{n}_{k}.png')
    link(src / 'test_rli', out / 'test_rli')
    for e in ENVMAPS:
        link(Path(diffreg) / 'envmap_mii' / f'{e}.exr', out_root / 'envmaps' / f'{e}.exr')

    height, width = np.asarray(Image.open(src / 'train' / '000_mask.png')).shape[:2]
    frames = [{'path': f"{f['file_path']}_rgb.exr", 'mask': f"{f['file_path']}_mask.png", 'to_world': mii_to_world(f['transform_matrix'])}
              for f in train['frames']]
    nsr = out_root / 'diffreg' / scene
    write_json(nsr / 'cameras.json', diffreg_cameras(frames, width, height, train['camera_angle_x']))
    link(src / 'train', nsr / 'train')
    write_json(out / 'dataset.json', dict(dataset='s4r', scene=scene, nsr_to_world=NSR_TO_WORLD_MII, textured_eval=False,
                                          relight_envmaps=[f'../envmaps/{e}.exr' for e in ENVMAPS],
                                          diffreg_data=f'../diffreg/{scene}', mvinverse_input='.'))
