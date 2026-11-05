"""Refine and re-fuse albedo on an existing fixed mesh, then re-estimate prior sigma.

python scripts/refine_scene.py --scene work/s4r/hotdog/scene_mvinverse.pt \
    --prior work/s4r/hotdog/priors/mvinverse --out work/s4r/hotdog/scene_mvinverse_refined.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from prep_ir.material import prior_sigma
from prep_ir.prior import fuse_prior
from prep_ir.refinement import RefinementConfig
from prep_ir.render import Renderer
from prep_ir.scene import load_scene

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene', type=Path, required=True)
p.add_argument('--prior', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--spatial-sigma', type=float, default=12.)
p.add_argument('--prior-sigma', type=float, default=.2)
p.add_argument('--guidance-paths', type=Path)
a = p.parse_args()
if a.scene.resolve() == a.out.resolve():
    p.error('use a separate output so the original prior remains available')
scene = load_scene(a.scene)
c = RefinementConfig(spatial_sigma=a.spatial_sigma, prior_sigma=a.prior_sigma)
guidance = json.loads(a.guidance_paths.read_text()) if a.guidance_paths else None
v, f = scene['vertices'], scene['faces']
renderer = Renderer(v, f)
# Cached features may precede mesh cleanup; compare fusion on the actual fixed mesh.
reference = fuse_prior(renderer, v, f, scene['data_dir'], a.prior)
prior = fuse_prior(renderer, v, f, scene['data_dir'], a.prior,
                   refinement=c, guidance_paths=guidance)
if not torch.equal(prior['observations'], reference['observations']):
    raise RuntimeError('refinement changed visibility/observation counts')
if not torch.allclose(prior['features'][:, 3:], reference['features'][:, 3:], atol=2e-5, rtol=0):
    raise RuntimeError('refinement changed roughness or metallic')
scene = dict(scene, prior=prior, prior_dir=str(a.prior.resolve()))
scene['report'] = dict(scene.get('report', {}), refinement=dict(
    **c.as_dict(), sigma_after_refinement=prior_sigma(prior['observations'], prior['variance']),
    variance_source='refined_per_view_material', source_scene=str(a.scene.resolve())))
a.out.parent.mkdir(parents=True, exist_ok=True)
temporary = a.out.with_suffix(a.out.suffix + '.tmp')
torch.save(scene, temporary)
temporary.replace(a.out)
print(json.dumps(scene['report']['refinement'], indent=2))
