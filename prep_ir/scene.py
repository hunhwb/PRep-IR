"""Scene preparation: reconstructed mesh + fused material prior -> scene.pt, the input of joint optimization.

A prepared dataset directory carries `dataset.json` (written by scripts/prepare_dataset.py):
    nsr_to_world   3x3 matrix mapping the surface reconstruction's frame to the dataset's world frame
    textured_eval  true when evaluation maps the result back onto the reconstruction's UV atlas (Stanford-ORB, DTC)
"""
import json
from pathlib import Path

import numpy as np
import torch

from .mesh import load_reconstruction
from .prior import fuse_prior
from .refinement import RefinementConfig


def dataset_info(data_dir):
    return json.loads((Path(data_dir) / 'dataset.json').read_text())


def prepare_scene(data_dir, mesh_path, prior_dir, out_path, log=print, refinement=RefinementConfig(), guidance_paths=None):
    """Refine albedo with the paper's defaults, then fuse; pass refinement=None for the unrefined control."""
    from .render import Renderer
    info = dataset_info(data_dir)
    vertices, faces, report = load_reconstruction(mesh_path, np.asarray(info['nsr_to_world']), repair_normals=not info['textured_eval'])
    log(f'mesh: {report}')
    prior = fuse_prior(Renderer(vertices, faces), vertices, faces, data_dir, prior_dir, log=log,
                       refinement=refinement, guidance_paths=guidance_paths)
    scene = dict(vertices=torch.from_numpy(vertices), faces=torch.from_numpy(faces), prior=prior,
                 data_dir=str(Path(data_dir).resolve()), mesh=str(Path(mesh_path).resolve()), prior_dir=str(Path(prior_dir).resolve()),
                 report=dict(report, supported=int((prior['observations'] > 0).sum())))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(scene, out_path)
    return scene


def load_scene(path):
    return torch.load(path, map_location='cpu', weights_only=False)
