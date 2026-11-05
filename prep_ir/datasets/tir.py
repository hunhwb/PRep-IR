"""TensoIR-Synthetic -> the layout used here.

armadillo, ficus and hotdog come from PTIR-GS's TensoIR_Material release (https://github.com/junkzhu/PTIR-GS), which
adds roughness ground truth to TensoIR's per-view directories (train_XXX/rgba.png; test_XXX/{rgba, albedo (linear),
roughness, normal, rgba_<envmap>}.png). lego uses the original TensoIR release (https://github.com/Haian-Jin/TensoIR)
with cameras from its metadata.json and the roughness of PTIR-GS's lego (PTIR-GS re-rendered lego's images).

    <out>/<scene>/transforms_{train,test}.json, train_XXX/, test_XXX/   (per-view directories, linked)
    <out>/diffreg/<scene>/                                             DiffReg-PBIR input: black-composited images, masks
    <out>/envmaps/{bridge,city,fireplace,forest,night}.hdr            relighting envmaps (TensoIR high_res_envmaps_2k)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import NSR_TO_WORLD_MII, diffreg_cameras, link, mii_to_world, write_json

SCENES = ['armadillo', 'ficus', 'hotdog', 'lego']
ENVMAPS = ['bridge', 'city', 'fireplace', 'forest', 'night']
LEGO_FILES = ['rgba.png', 'albedo.png', 'normal.png', 'metadata.json'] + [f'rgba_{e}.png' for e in ENVMAPS]


def link_views(src, out):
    for split in ('train', 'test', 'val'):
        if (src / f'transforms_{split}.json').is_file():
            write_json(out / f'transforms_{split}.json', json.loads((src / f'transforms_{split}.json').read_text()))
    for d in sorted(src.iterdir()):
        if d.is_dir() and d.name.split('_')[0] in ('train', 'test', 'val'):
            link(d, out / d.name)


def build_lego(tensoir_lego, ptir_lego, out):
    for split, n in (('train', 100), ('test', 200), ('val', 10)):
        frames = []
        for i in range(n):
            d = f'{split}_{i:03d}'
            for f in LEGO_FILES:
                if (tensoir_lego / d / f).exists():
                    link(tensoir_lego / d / f, out / d / f)
            link(ptir_lego / d / 'roughness.png', out / d / 'roughness.png')
            meta = json.loads((tensoir_lego / d / 'metadata.json').read_text())
            frames.append({'file_path': f'./{d}/rgba',
                           'transform_matrix': np.array(meta['cam_transform_mat'].split(','), float).reshape(4, 4).tolist()})
        write_json(out / f'transforms_{split}.json', {'camera_angle_x': meta['cam_angle_x'], 'frames': frames})


def prepare(ptir_material, tensoir, out_root, scene):
    """ptir_material: PTIR-GS's TensoIR_Material directory; tensoir: TensoIR-Synthetic (lego/, high_res_envmaps_2k/)."""
    ptir_material, tensoir, out_root = Path(ptir_material), Path(tensoir), Path(out_root)
    out = out_root / scene
    if scene == 'lego':
        build_lego(tensoir / 'lego', ptir_material / 'lego', out)
    else:
        link_views(ptir_material / scene, out)
    for e in ENVMAPS:
        link(tensoir / 'high_res_envmaps_2k' / f'{e}.hdr', out_root / 'envmaps' / f'{e}.hdr')

    train = json.loads((out / 'transforms_train.json').read_text())
    nsr = out_root / 'diffreg' / scene
    (nsr / 'images').mkdir(parents=True, exist_ok=True)
    (nsr / 'masks').mkdir(exist_ok=True)
    frames = []
    for f in train['frames']:
        src = (out / (f['file_path'] + '.png')).resolve()
        name = Path(f['file_path']).parent.name + '.png'
        x = np.asarray(Image.open(src)).astype(np.float32) / 255
        height, width = x.shape[:2]
        if not (nsr / 'masks' / name).exists():
            Image.fromarray((x[..., :3] * x[..., 3:] * 255 + .5).astype(np.uint8)).save(nsr / 'images' / name)
            Image.fromarray((x[..., 3] * 255 + .5).astype(np.uint8)).save(nsr / 'masks' / name)
        frames.append({'path': f'images/{name}', 'mask': f'masks/{name}', 'to_world': mii_to_world(f['transform_matrix']), 'source': str(src)})
    write_json(nsr / 'cameras.json', diffreg_cameras(frames, width, height, train['camera_angle_x']))
    write_json(out / 'dataset.json', dict(dataset='tir', scene=scene, nsr_to_world=NSR_TO_WORLD_MII, textured_eval=False,
                                          relight_envmaps=[f'../envmaps/{e}.hdr' for e in ENVMAPS],
                                          diffreg_data=f'../diffreg/{scene}', mvinverse_input='.'))
