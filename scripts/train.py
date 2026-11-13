"""Joint optimization of one scene (the paper's method unless experiment configs or overrides say otherwise).

usage: python scripts/train.py --scene work/s4r/hotdog/scene_diffusion_renderer_refined.pt --out outputs/s4r/hotdog/diffusion_renderer_refined
         [--config configs/experiments/fast200.yaml ...] [--set seed=20261916 material.large_steps=0 ...]
         [--resume outputs/.../checkpoint_0500.pt]
Writes <out>/run.json, train_log.jsonl, checkpoint.pt and result/ (result.npz, envmap.exr), the input of evaluate.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prep_ir.config import as_plain, load_config  # noqa: E402
from prep_ir.scene import load_scene  # noqa: E402
from prep_ir.trainer import Trainer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--scene', type=Path, required=True, help='scene.pt of scripts/prepare_scene.py')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--config', type=Path, nargs='*', default=[], help='experiment files applied over configs/default.yaml')
p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides, e.g. seed=1 material.large_steps=0')
p.add_argument('--resume', type=Path, help='continue from a checkpoint of this run')
a = p.parse_args()

config = load_config(a.config, a.set)
scene = load_scene(a.scene)
a.out.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
trainer = Trainer(scene, config, a.out)
if a.resume:
    trainer.load_checkpoint(a.resume)
setup = time.perf_counter() - start
info = dict(scene=str(a.scene.resolve()), data_dir=scene['data_dir'], mesh=scene['mesh'], prior_dir=scene['prior_dir'],
            config=as_plain(config), experiment_files=[str(f) for f in a.config], overrides=a.set,
            lattice=trainer.lattice.report if trainer.lattice is not None else None, setup_seconds=setup)
(a.out / 'run.json').write_text(json.dumps(info, indent=1) + '\n')
info['optimization_seconds'] = trainer.train()
(a.out / 'run.json').write_text(json.dumps(info, indent=1) + '\n')
print(f"done: {info['optimization_seconds']:.1f} s of optimization -> {a.out / 'result'}")
