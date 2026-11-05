"""Initial mesh: DiffReg-PBIR's neural surface reconstruction (stage 2) guided by the prior's normals (Sec. 4.2, 5.1).

A dense-grid SDF is optimized for 25k iterations with DiffReg-PBIR's `*_regnormal` template at a normal-loss weight of
1e-6, then meshed by marching cubes, Botsch-remeshed and UV-unwrapped with Blender 3.6 (DiffReg-PBIR's export). Keeps
the largest connected component, except TensoIR ficus and Synthetic4Relight air_baloons (all components). Seed 776.
Needs Blender 3.6 ($BLENDER, DiffReg-PBIR's third_party copy, or `blender` on PATH).

usage: python scripts/reconstruct_mesh.py --data data/s4r/hotdog --nsr-data work/s4r/hotdog/nsr_data/mvinverse/hotdog
                                          --out work/s4r/hotdog/mesh/mvinverse
  --nsr-data: a DiffReg-PBIR data directory with normal_single/ (MVInverse: scripts/mvinverse_normals.py;
              DiffusionRenderer: the scene's own DiffReg-PBIR data directory after scripts/predict_diffusion_renderer.sh)
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFREG = Path(os.environ.get('PREP_IR_DIFFREG', ROOT / 'third_party/DiffReg-PBIR')).resolve()
TEMPLATE = dict(s4r='template_mii_regnormal', tir='template_mii_regnormal', orb='template_stanford_orb_regnormal', dtc='template_dtc_regnormal')
KEEP_ALL_COMPONENTS = {('tir', 'ficus'), ('s4r', 'air_baloons')}
NORMAL_WEIGHT = 1e-6
SEED = 776

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--nsr-data', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
info = json.loads((a.data / 'dataset.json').read_text())
blender = os.environ.get('BLENDER') or next(iter(sorted(DIFFREG.glob('third_party/blender-3.6*/blender'))), None)
if blender:   # DiffReg-PBIR's UV unwrap calls a bare `blender`
    os.environ['PATH'] = f"{Path(blender).parent}:{os.environ['PATH']}"

sys.path.insert(0, str(DIFFREG))
import mmcv  # noqa: E402
from neural_surface_recon.run import config_parser, run  # noqa: E402

cfg = mmcv.Config.fromfile(str(DIFFREG / 'neural_surface_recon/configs' / f"{TEMPLATE[info['dataset']]}.py"))
cfg.fine_train.weight_normal = NORMAL_WEIGHT
cfg.results_dir = a.out.resolve()
cfg.data.datadir = str(a.nsr_data.resolve())
args = ['--run_vis_period', '100', '--run_vis_factor', '2', '--render_video', '--render_normal', '--render_depth', '--render_fg_only',
        '--render_video_factor', '1', '--export_surface', '--no_reload', '--seed', str(SEED)]
if (info['dataset'], info['scene']) in KEEP_ALL_COMPONENTS:
    args.append('--mesh_no_cleanup')
os.chdir(DIFFREG)
run(config_parser().parse_args(args + ['--config', 'dummy']), cfg)
