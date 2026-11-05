"""DTC-Synthetic relighting with DiffReg-PBIR's released protocol.

The baked asset (prep_ir.evaluation.bake) is rendered by DiffReg-PBIR's scripts/dtc/render_relit_images.py (its
Mitsuba fork, 1024 spp) under envmap12, envmap3 and envmap6 from the 20 orbit cameras of envmap_mii/cameras.json and
compared with the release's ground-truth relit images: masked sRGB PSNR, SSIM and VGG-LPIPS over the full 800x800
frame, averaged over the 60 frames. The paper reports the unaligned scores.

Runs with DiffReg-PBIR's Mitsuba fork on the path (its build/setpath.sh, as scripts/evaluate.py does) and
PREP_IR_DIFFREG pointing to the checkout (default: third_party/DiffReg-PBIR).
usage: python -m prep_ir.evaluation.dtc STAGE_DIR SCENE_DATA_DIR OUT_DIR
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIFFREG = Path(os.environ.get('PREP_IR_DIFFREG', ROOT / 'third_party/DiffReg-PBIR')).resolve()


def evaluate(stage, data_dir, out_dir):
    stage, data_dir, out = Path(stage).resolve(), Path(data_dir), Path(out_dir).resolve()
    info = json.loads((data_dir / 'dataset.json').read_text())
    release = (data_dir / info['diffreg_data']).resolve()   # the DTC-Synthetic release directory (GT relit images)
    envmaps = DIFFREG / 'envmap_mii'
    metrics_file = out / 'relit' / 'relighting_metrics.json'
    if not metrics_file.is_file():
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(DIFFREG), str(DIFFREG / 'diffusion_reg'), str(DIFFREG / 'diffusion_reg/deps/irtk'),
                                                            os.environ.get('PYTHONPATH', '')]))
        cmd = [sys.executable, '-m', 'scripts.dtc.render_relit_images', '--envmap_dir', envmaps, '--cameras_path', envmaps / 'cameras.json',
               '--output_dir', out / 'relit', '--texture_dir', stage, '--gt_relit_dir', release / 'relit', '--do_eval',
               '--dataroot', release, '--skip_render_if_exists']
        print('+', ' '.join(map(str, cmd)), flush=True)
        # from the output directory: the irtk connectors write scratch files into the working directory
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(x) for x in cmd], cwd=out, env=env, check=True)
    d = json.loads(metrics_file.read_text())
    names = [p.stem for p in sorted(envmaps.glob('*.exr'))]   # the evaluator's envmap order
    keys = ['psnr', 'ssim', 'lpips', 'psnr_aligned', 'ssim_aligned', 'lpips_aligned']
    metrics = {f'relight_{k}': d[k] for k in keys}
    for j, name in enumerate(names):
        rows = [r for r in d['details'] if r['envmap_index'] == j]
        metrics.update({f'relight_{name}_{k}': sum(r[k] for r in rows) / len(rows) for k in keys})
    metrics['relight_frames'] = d['count']
    (out / 'evaluation.json').write_text(json.dumps(dict(metrics=metrics), indent=1) + '\n')
    print(json.dumps({k: metrics[k] for k in ('relight_psnr', 'relight_ssim', 'relight_lpips')}))
    return metrics


if __name__ == '__main__':
    evaluate(*sys.argv[1:4])
