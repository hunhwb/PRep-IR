"""Stanford-ORB (https://stanfordorb.github.io, blender_HDR + ground_truth) -> the layout used here.

    <out>/stanford_orb/{blender_HDR,ground_truth}/<capture>   writable mirror (links) of every capture of the requested
                                                             objects (novel-light frames come from an object's other
                                                             captures), plus DiffReg-PBIR's preprocessing: cameras.json,
                                                             {train,test}_512x512/ HDR frames, cameras_novel/, and the
                                                             ground-truth envmaps in Blender orientation
    <out>/<capture>/transforms_{train,test}.json, train/, test/  the 512x512 linear HDR frames and masks (links)
    <out>/<capture>/mvinverse_input/                         sRGB PNGs of the training frames (MVInverse input)
    <out>/diffreg/<capture>/                                 DiffReg-PBIR input for G-buffers and surface reconstruction

Runs DiffReg-PBIR's scripts/preprocess/stanford_orb.py and scripts/stanford_orb/postproc_envmap_gt.py unchanged.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .common import IDENTITY, link, read_exr_rgb, write_json

# the official relighting split (main paper) and the full benchmark (supplement), as in Stanford-ORB's orb/constant.py
LIGHT_SPLIT = ['teapot_scene001', 'grogu_scene002', 'gnome_scene003', 'car_scene004', 'pitcher_scene005', 'blocks_scene006', 'cactus_scene007']
FULL = [f'{o}_scene{n:03d}' for o, ns in (('baking', (1, 2, 3)), ('ball', (2, 3, 4)), ('blocks', (2, 5, 6)), ('cactus', (1, 5, 7)),
                                          ('car', (2, 4, 6)), ('chips', (2, 3, 4)), ('cup', (3, 6, 7)), ('curry', (1, 5, 7)),
                                          ('gnome', (3, 5, 7)), ('grogu', (1, 2, 3)), ('pepsi', (2, 3, 4)), ('pitcher', (1, 5, 7)),
                                          ('salt', (4, 5, 7)), ('teapot', (1, 2, 6))) for n in ns]


def mvinverse_png(exr, mask_png, out):
    f = read_exr_rgb(exr)
    rgb = np.where(f > .0031308, np.power(np.maximum(f, .0031308), 1 / 2.4) * 1.055 - .055, 12.92 * f)
    alpha = np.asarray(Image.open(mask_png).convert('L'), np.float32) / 255
    Image.fromarray((np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], -1) * 255).round().astype(np.uint8)).save(out)


def prepare(src, out_root, scenes, diffreg):
    src, out_root, diffreg = Path(src), Path(out_root), Path(diffreg).resolve()
    mirror = out_root / 'stanford_orb'
    objects = {s.split('_scene')[0] for s in scenes}
    captures = [d.name for d in sorted((src / 'blender_HDR').iterdir()) if d.name.split('_scene')[0] in objects]
    for top in ('blender_HDR', 'ground_truth'):
        for s in captures:
            for child in sorted((src / top / s).iterdir()):
                link(child, mirror / top / s / child.name)
    for scene in scenes:
        D = mirror / 'blender_HDR' / scene
        if not (D / 'cameras.json').is_file():
            subprocess.run([sys.executable, 'scripts/preprocess/stanford_orb.py', str(D)], cwd=diffreg, check=True)
        cameras = json.loads((D / 'cameras.json').read_text())
        out = out_root / scene
        for split in ('train', 'test'):
            meta = json.loads((D / f'transforms_{split}.json').read_text())
            frames = []
            for i, fr in zip(cameras['split'][split], meta['frames']):
                f = cameras['frames'][i]
                stem = Path(f['path']).stem
                if Path(fr['file_path']).stem != stem.removesuffix('_image'):
                    raise RuntimeError(f'frame order differs: {fr["file_path"]} vs {stem}')
                link(D / f['path'], out / split / f'{stem}.exr')
                link(D / f['mask'], out / split / f'{stem}_mask.png')
                frames.append(dict(file_path=f'{split}/{stem}', transform_matrix=fr['transform_matrix']))
            write_json(out / f'transforms_{split}.json', dict(camera_angle_x=meta['camera_angle_x'], frames=frames))
        train = json.loads((out / 'transforms_train.json').read_text())
        (out / 'mvinverse_input' / 'train').mkdir(parents=True, exist_ok=True)
        for fr in train['frames']:
            stem = Path(fr['file_path']).name
            if not (out / 'mvinverse_input' / 'train' / f'{stem}.png').exists():
                mvinverse_png(out / 'train' / f'{stem}.exr', out / 'train' / f'{stem}_mask.png', out / 'mvinverse_input' / 'train' / f'{stem}.png')
        write_json(out / 'mvinverse_input' / 'transforms_train.json', train)
        nsr = out_root / 'diffreg' / scene   # named exactly <capture>; cameras.json is copied (G-buffer staging writes next to it)
        nsr.mkdir(parents=True, exist_ok=True)
        for x in D.iterdir():
            if x.name == 'cameras.json':
                shutil.copy(x, nsr / x.name)
            else:
                link(x, nsr / x.name)
        write_json(out / 'dataset.json', dict(dataset='orb', scene=scene, nsr_to_world=IDENTITY, textured_eval=True,
                                              orb_root='../stanford_orb', diffreg_data=f'../diffreg/{scene}', mvinverse_input='mvinverse_input'))
    if any(not (mirror / 'ground_truth' / s / 'env_map_for_blender').is_dir() for s in captures):
        subprocess.run([sys.executable, 'scripts/stanford_orb/postproc_envmap_gt.py', '--gt_dir', str(mirror / 'ground_truth'),
                        '--im_dir', str(mirror / 'blender_HDR')], cwd=diffreg, check=True)
