"""Convert a public dataset release into the layout used here (see prep_ir/datasets/*.py for each layout).

Stanford-ORB uses DiffReg-PBIR's preprocessing scripts (PREP_IR_DIFFREG, default third_party/DiffReg-PBIR).

  python scripts/prepare_dataset.py s4r --src /path/to/Synthetic4Relight --out data/s4r [--scenes hotdog ...]
  python scripts/prepare_dataset.py tir --src /path/to/PTIR-GS/TensoIR_Material --tensoir /path/to/TensoIR-Synthetic --out data/tir
  python scripts/prepare_dataset.py orb --src /path/to/stanford_orb --out data/orb [--scenes light | full | teapot_scene001 ...]
  python scripts/prepare_dataset.py dtc --src /path/to/extracted/DTC-Synthetic --out data/dtc
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prep_ir.datasets import dtc, orb, s4r, tir  # noqa: E402

DIFFREG = Path(os.environ.get('PREP_IR_DIFFREG', ROOT / 'third_party/DiffReg-PBIR'))

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('dataset', choices=['s4r', 'tir', 'orb', 'dtc'])
p.add_argument('--src', type=Path, required=True, help='the downloaded release')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--tensoir', type=Path, help='tir: TensoIR-Synthetic release (lego and high_res_envmaps_2k)')
p.add_argument('--scenes', nargs='+', help='default: every scene of the paper (orb: the light split; "full" for all 42)')
a = p.parse_args()

if a.dataset == 's4r':
    for scene in a.scenes or s4r.SCENES:
        s4r.prepare(a.src, a.out, scene, DIFFREG)
        print(scene, 'ok')
elif a.dataset == 'tir':
    if a.tensoir is None:
        p.error('tir needs --tensoir')
    for scene in a.scenes or tir.SCENES:
        tir.prepare(a.src, a.tensoir, a.out, scene)
        print(scene, 'ok')
elif a.dataset == 'orb':
    scenes = {None: orb.LIGHT_SPLIT, ('light',): orb.LIGHT_SPLIT, ('full',): orb.FULL}.get(tuple(a.scenes) if a.scenes else None, a.scenes)
    orb.prepare(a.src, a.out, scenes, DIFFREG)
    print(len(scenes), 'captures ok')
else:
    for scene in a.scenes or dtc.SCENES:
        dtc.prepare(a.src, a.out, scene)
        print(scene, 'ok')
