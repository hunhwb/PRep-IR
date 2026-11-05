"""Multi-view images in the NeRF-synthetic layout: transforms_<split>.json plus one image per frame.

Frames are 8-bit sRGB PNGs with alpha (Synthetic4Relight, TensoIR) or linear HDR EXRs (Stanford-ORB, DTC); an
optional `<stem>_mask.png` next to a frame replaces its alpha as the object mask. Training targets are sRGB images
composited onto the background colour; HDR frames are composited with their mask and sRGB-encoded without clipping.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .image import linear_to_srgb, read_exr


class Views:
    def __init__(self, root, split, resolution=None, background='black'):
        """resolution: image width to work at (the height follows the aspect ratio); None keeps the files' size."""
        self.root = Path(root).resolve()
        self.split = split
        self.background = {'black': 0., 'white': 1.}[background]
        meta = json.loads((self.root / f'transforms_{split}.json').read_text())
        if any(meta.get(k, 0) != 0 for k in ('k1', 'k2', 'p1', 'p2')):
            raise ValueError('distorted cameras are not supported')
        self.frames = meta['frames']
        self.paths = [self._path(f['file_path']) for f in self.frames]
        self.hdr = self.paths[0].suffix == '.exr'
        w, h = read_exr(self.paths[0]).shape[1::-1] if self.hdr else Image.open(self.paths[0]).size
        self.size = (w, h) if not resolution else (int(resolution), round(h * int(resolution) / w))
        self.fov = float(meta['camera_angle_x']) * 180 / np.pi   # horizontal, degrees
        self.transforms = [np.asarray(f['transform_matrix'], np.float32) for f in self.frames]   # camera-to-world, OpenGL axes
        self._cache = {}

    def _path(self, name):
        p = self.root / name
        if not p.suffix:
            p = p.with_suffix('.png') if p.with_suffix('.png').is_file() or not p.with_suffix('.exr').is_file() else p.with_suffix('.exr')
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def __len__(self):
        return len(self.frames)

    def image(self, path, rgba=False):
        """The frame at the working size as a float tensor: linear radiance (EXR) or sRGB in [0, 1] (PNG)."""
        if Path(path).suffix == '.exr':
            x = torch.from_numpy(read_exr(path))
            if (x.shape[1], x.shape[0]) != tuple(self.size):
                x = F.interpolate(x.permute(2, 0, 1)[None], size=(self.size[1], self.size[0]), mode='area')[0].permute(1, 2, 0)
            return torch.cat((x, torch.ones_like(x[..., :1])), -1) if rgba else x
        with Image.open(path) as im:
            im = im.convert('RGBA' if rgba else 'RGB').resize(self.size, Image.Resampling.BILINEAR)
            return torch.from_numpy(np.asarray(im).copy()).float() / 255

    def mask(self, index, default):
        path = self.paths[index].with_name(self.paths[index].stem + '_mask.png')
        if not path.is_file():
            return default
        with Image.open(path) as im:
            return torch.from_numpy(np.asarray(im.convert('L').resize(self.size, Image.Resampling.NEAREST)).copy()).float()[..., None] / 255

    def batch(self, index, device='cuda'):
        """{'rgb': sRGB target on the background colour, 'mask': object mask}, both (H, W, C)."""
        if (index, device) in self._cache:
            return self._cache[index, device]
        rgba = self.image(self.paths[index], True)
        alpha = rgba[..., 3:4]
        rgb = (rgba[..., :3] * alpha + self.background * (1 - alpha)).to(device)
        mask = self.mask(index, alpha).to(device)
        if self.hdr:   # an EXR has no alpha: composite with the mask, then encode (highlights above 1 are kept)
            rgb = linear_to_srgb(rgb * mask + self.background * (1 - mask))
        return {'rgb': rgb, 'mask': mask}

    def preload(self, device='cuda'):
        """Decode every frame once and keep it on `device`."""
        for i in range(len(self)):
            self._cache[i, device] = self.batch(i, device)
        return self
