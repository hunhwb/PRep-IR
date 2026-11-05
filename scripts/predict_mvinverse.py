"""MVInverse material and normal predictions for a scene's training views (Wu et al., "MVInverse: Feed-forward
Multi-view Inverse Rendering in Seconds"; code https://github.com/Maddog241/mvinverse, weights maddog241/mvinverse).

Preprocessing follows MVInverse's inference.py: RGBA composited on white, long side capped at 1024, both sides floored
to a multiple of 14. Views are processed in chunks of 10 (one forward pass and inter-view attention per chunk) under
float16 autocast. Output (prep_ir.prior layout):
    <out>/mapping.json, <out>/predictions/<key>.npz   albedo, roughness, metallic, camera-space normal (float16), alpha

usage: python scripts/predict_mvinverse.py --data data/s4r/hotdog --out work/s4r/hotdog/priors/mvinverse
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(os.environ.get('PREP_IR_MVINVERSE', ROOT / 'third_party/mvinverse'))))
from mvinverse.models.mvinverse import MVInverse  # noqa: E402


def load_view(scene_dir, file_path):
    """(key, image tensor (3, h, w), alpha at model size (uint8), original (H, W), model (h, w))."""
    split_dir = os.path.basename(os.path.dirname(file_path))
    if split_dir in ('train', 'test'):   # flat frames train/<stem>
        stem = os.path.basename(file_path)
        key, path = f'{split_dir}_{stem}', scene_dir / split_dir / f'{stem}.png'
    else:                                # per-view directories (TensoIR): train_000/rgba
        key, path = split_dir, scene_dir / split_dir / 'rgba.png'
    with Image.open(path) as img:
        rgba = np.asarray(img.convert('RGBA')).astype(np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    h, w = rgb.shape[:2]
    scale = min(1.0, 1024.0 / max(h, w))
    new_w, new_h = int(w * scale) // 14 * 14, int(h * scale) // 14 * 14
    image = Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h))
    alpha_img = Image.fromarray((alpha[..., 0] * 255).astype(np.uint8)).resize((new_w, new_h))
    return key, torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1), np.asarray(alpha_img), (h, w), (new_h, new_w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True, help='prepared scene directory (data/<dataset>/<scene>)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chunk-size', type=int, default=10)
    a = p.parse_args()
    info = json.loads((a.data / 'dataset.json').read_text())
    scene_dir = (a.data / info['mvinverse_input']).resolve()
    frames = json.loads((scene_dir / 'transforms_train.json').read_text())['frames']
    (a.out / 'predictions').mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260916)
    model = MVInverse.from_pretrained('maddog241/mvinverse').to('cuda').eval()
    mapping = {}
    for ci in range(0, len(frames), a.chunk_size):
        views = [load_view(scene_dir, f['file_path']) for f in frames[ci:ci + a.chunk_size]]
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            res = model(torch.stack([v[1] for v in views])[None].to('cuda'))
        out = {k: res[k][0].float().cpu().numpy() for k in ('albedo', 'roughness', 'metallic', 'normal')}
        for i, (key, _, alpha, orig_hw, model_hw) in enumerate(views):
            np.savez(a.out / 'predictions' / f'{key}.npz', albedo=out['albedo'][i].astype(np.float16),
                     roughness=out['roughness'][i, ..., 0].astype(np.float16), metallic=out['metallic'][i, ..., 0].astype(np.float16),
                     alpha=alpha.astype(np.uint8), normal=out['normal'][i].astype(np.float16))
            mapping[key] = dict(npz=f'{key}.npz', chunk=ci // a.chunk_size, orig_hw=list(orig_hw), model_hw=list(model_hw))
        print(f'{len(mapping)}/{len(frames)} views', flush=True)
    (a.out / 'mapping.json').write_text(json.dumps(mapping, indent=2))


if __name__ == '__main__':
    main()
