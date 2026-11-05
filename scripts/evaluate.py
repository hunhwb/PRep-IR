"""Evaluate a trained run with its dataset's protocol (paper, supplement "Evaluation Protocol").

  Synthetic4Relight, TensoIR-Synthetic   prep_ir.evaluation.synthetic
  Stanford-ORB                           Blender 3.6 + Stanford-ORB's scoring (prep_ir.evaluation.orb)
  DTC-Synthetic                          texture bake, then DiffReg-PBIR's Mitsuba relighting protocol
                                         (prep_ir.evaluation.dtc, in a process with DiffReg-PBIR's Mitsuba fork)
External locations come from PREP_IR_DIFFREG, PREP_IR_ORB_BENCHMARK, PREP_IR_MITSUBA_SETPATH and BLENDER.

usage: python scripts/evaluate.py RUN_DIR [--view-stride 4] [--save-images]     -> RUN_DIR/eval[_strideN]/evaluation.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('--view-stride', type=int, default=1, help='synthetic: score every n-th of the 200 test views (the albedo scale always uses all)')
p.add_argument('--save-images', action='store_true', help='synthetic: also write the scored images')
a = p.parse_args()
run = json.loads((a.run / 'run.json').read_text())
data = Path(run['data_dir'])
kind = json.loads((data / 'dataset.json').read_text())['dataset']
result, out = a.run / 'result', a.run / ('eval' if a.view_stride == 1 else f'eval_stride{a.view_stride}')

if kind in ('s4r', 'tir'):
    from prep_ir.evaluation.synthetic import evaluate
    evaluate(result, data, out, view_stride=a.view_stride, save_images=a.save_images)
elif kind == 'orb':
    from prep_ir.evaluation.orb import evaluate
    evaluate(result, data, out)
elif kind == 'dtc':
    from prep_ir.evaluation.bake import bake
    stage = out / 'stage'
    if not (stage / 'envmap.exr').is_file():
        bake(result, run['mesh'], stage)
    diffreg = Path(os.environ.get('PREP_IR_DIFFREG', ROOT / 'third_party/DiffReg-PBIR')).resolve()
    setpath = Path(os.environ.get('PREP_IR_MITSUBA_SETPATH', diffreg / 'diffusion_reg/deps/mitsuba3/build/setpath.sh'))
    if not setpath.is_file():
        raise FileNotFoundError(f'{setpath}: build DiffReg-PBIR\'s Mitsuba fork with scripts/setup_environment.sh')
    # the fork's setpath.sh puts its mitsuba / drjit ahead of the pip packages, so it runs in a process of its own
    subprocess.run(['bash', '-c', 'source "$1" && shift && exec "$@"', 'prep-ir-dtc', setpath,
                    sys.executable, '-m', 'prep_ir.evaluation.dtc', stage, data, out],
                   cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), check=True)
else:
    raise SystemExit(f'unknown dataset {kind}')
