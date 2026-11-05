"""Scene preparation: load and clean the reconstructed mesh, fuse the material prior onto its vertices -> scene.pt.

usage: python scripts/prepare_scene.py --data data/s4r/hotdog --mesh work/s4r/hotdog/mesh/mvinverse/mesh.obj
                                       --prior work/s4r/hotdog/priors/mvinverse --out work/s4r/hotdog/scene_mvinverse_refined.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prep_ir.scene import prepare_scene  # noqa: E402
from prep_ir.refinement import RefinementConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True, help='prepared scene directory (scripts/prepare_dataset.py)')
p.add_argument('--mesh', type=Path, required=True, help='mesh.obj of scripts/reconstruct_mesh.py')
p.add_argument('--prior', type=Path, required=True, help='predictions of scripts/predict_mvinverse.py or convert_diffusion_renderer.py')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--refine-albedo', action=argparse.BooleanOptionalAction, default=True,
               help='RGB-guided prior selection before fusion (default: enabled)')
p.add_argument('--spatial-sigma', type=float, default=12., help='spatial affinity width in prediction pixels')
p.add_argument('--prior-sigma', type=float, default=.2, help='sRGB prior-colour affinity width')
p.add_argument('--guidance-paths', type=Path, help='optional JSON mapping view keys to original RGB/HDR captures')
a = p.parse_args()
refinement = RefinementConfig(spatial_sigma=a.spatial_sigma, prior_sigma=a.prior_sigma) if a.refine_albedo else None
guidance = json.loads(a.guidance_paths.read_text()) if a.guidance_paths else None
scene = prepare_scene(a.data, a.mesh, a.prior, a.out, refinement=refinement, guidance_paths=guidance)
print(f"{a.out}: {scene['report']}")
