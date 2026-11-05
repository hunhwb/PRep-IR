"""Collect outputs/<dataset>/<scene>/<run>/eval/evaluation.json into per-dataset tables: the mean over scenes for every
run name (runs named <config>_seed<N> are first averaged per seed, then the seed means are reported as mean and sample
standard deviation).

usage: python scripts/summarize.py [--outputs outputs] [--scenes orb=light]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prep_ir.datasets.orb import LIGHT_SPLIT  # noqa: E402

COLUMNS = dict(
    s4r=['nvs_psnr', 'nvs_ssim', 'albedo_psnr', 'albedo_ssim', 'roughness_mae', 'relight_psnr', 'relight_ssim'],
    tir=['nvs_psnr', 'nvs_ssim', 'albedo_psnr', 'albedo_ssim', 'roughness_mae', 'normal_degrees', 'relight_psnr', 'relight_ssim'],
    orb=['nv_psnr_hdr', 'nv_psnr_ldr', 'nv_ssim', 'nv_lpips', 'nl_psnr_hdr', 'nl_psnr_ldr', 'nl_ssim', 'nl_lpips'],
    dtc=['relight_psnr', 'relight_ssim', 'relight_lpips'])

p = argparse.ArgumentParser()
p.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
p.add_argument('--orb', choices=['light', 'all'], default='light', help='Stanford-ORB captures to average')
a = p.parse_args()

table = defaultdict(lambda: defaultdict(dict))   # dataset -> (config, seed) -> scene -> metrics
for f in sorted(a.outputs.glob('*/*/*/eval/evaluation.json')):
    dataset, scene, run = f.parts[-5:-2]
    if dataset == 'orb' and a.orb == 'light' and scene not in LIGHT_SPLIT:
        continue
    m = re.fullmatch(r'(.*)_seed(\d+)', run)
    config, seed = (m.group(1), m.group(2)) if m else (run, 'default')
    table[dataset][config, seed][scene] = json.loads(f.read_text())['metrics']

for dataset in sorted(table):
    cols = COLUMNS[dataset]
    print(f'\n== {dataset}')
    print(f"{'run':44s} {'scenes':>6s} {'seeds':>5s} " + ' '.join(f'{c:>14s}' for c in cols))
    configs = sorted({c for c, _ in table[dataset]})
    for config in configs:
        seeds = [table[dataset][key] for key in table[dataset] if key[0] == config]
        scenes = set.intersection(*(set(s) for s in seeds))
        means = np.array([[np.mean([s[x][c] for x in scenes]) for c in cols] for s in seeds])
        cells = [f'{m:14.4f}' if len(means) == 1 else f'{m:8.4f}±{sd:5.3f}' for m, sd in zip(means.mean(0), means.std(0, ddof=1) if len(means) > 1 else means[0])]
        print(f'{config:44s} {len(scenes):6d} {len(seeds):5d} ' + ' '.join(cells))
