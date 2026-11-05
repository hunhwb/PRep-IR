"""Stanford-ORB evaluation of one capture with the official benchmark code (novel view and novel light).

The result is baked onto the reconstruction's UV atlas (prep_ir.evaluation.bake) and rendered by DiffReg-PBIR's
Stanford-ORB scripts (themselves from Neural-PBIR) with Blender 3.6 Cycles: the 10 novel views under the recovered
envmap and the 20 novel-light frames under the ground-truth envmap of each frame. The images are scored by
Stanford-ORB's `compute_metrics_image_similarity` (novel view: absolute; novel light: scale-invariant).

Needs Stanford-ORB's repository (its `lpips`, `kornia` requirements are in this environment). External locations
(environment variables):
    PREP_IR_DIFFREG        DiffReg-PBIR checkout              (default: third_party/DiffReg-PBIR)
    PREP_IR_ORB_BENCHMARK  Stanford-ORB checkout, rgba_exr.patch applied (default: third_party/Stanford-ORB)
    BLENDER                Blender 3.6 executable             (DiffReg-PBIR's relight script also finds its third_party copy)
usage: python -m prep_ir.evaluation.orb RESULT_DIR SCENE_DATA_DIR OUT_DIR
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIFFREG = Path(os.environ.get('PREP_IR_DIFFREG', ROOT / 'third_party/DiffReg-PBIR'))
BENCHMARK = Path(os.environ.get('PREP_IR_ORB_BENCHMARK', ROOT / 'third_party/Stanford-ORB'))


def run(*cmd, cwd=DIFFREG):
    print('+', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def render(stage, scene, orb_root):
    """DiffReg-PBIR's scripts: test-view geometry buffers, envmap to Blender orientation, Blender relighting."""
    py = sys.executable
    capture = orb_root / 'blender_HDR' / scene
    tex = ['--use_metallic', '--metallic_path', stage / 'metallic.exr']
    if len(list((stage / 'render_geo').glob('*.normal.exr'))) < 10:
        run(py, 'scripts/stanford_orb/render_geo.py', capture / 'cameras.json', stage / 'mesh.obj')
    if not (stage / 'envmap_for_blender.exr').is_file():
        run(py, 'scripts/stanford_orb/postproc_envmap_our.py', stage / 'envmap.exr')
    if len(list((stage / 'blender_relit').glob('*_envmap_for_blender.exr'))) < 10:
        run(py, 'scripts/relit/relit.py', stage / 'mesh.obj', stage / 'base_color.exr', stage / 'roughness.exr', capture / 'cameras.json',
            '--lgt_paths', stage / 'envmap_for_blender.exr', '--render_exr', '--with_bg', '--render_albedo', *tex)
    for cam in sorted((capture / 'cameras_novel').glob('*.json')):
        frame = json.load(open(cam))['frames'][0]
        stem, light_scene = frame['path'], frame['scene_name']
        env = orb_root / 'ground_truth' / light_scene / 'env_map_for_blender' / f'{light_scene}_{stem}_for_{scene}.exr'
        if not (stage / 'blender_relit' / f'im_{stem}_{env.stem}.exr').is_file():
            run(py, 'scripts/relit/relit.py', stage / 'mesh.obj', stage / 'base_color.exr', stage / 'roughness.exr', cam,
                '--lgt_paths', env, '--render_exr', '--with_bg', *tex)


def score(stage, scene, orb_root, out):
    """Stanford-ORB's view / light image metrics (material and geometry are not scored)."""
    ckpt = out / 'score'
    link = ckpt / scene / 'result'
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(stage.resolve())
    run(sys.executable, 'scripts/stanford_orb/eval_preprocess.py', '--stage', 'result', '--data_dir', orb_root, '--ckpt_dir', ckpt, '--scenes', scene)
    sys.path.insert(0, str(BENCHMARK))
    from orb.utils.test import compute_metrics_image_similarity
    info = json.loads((ckpt / 'eval_inputs_result.json').read_text())['info'][scene]
    view = compute_metrics_image_similarity(info['view'], scale_invariant=False)
    light = compute_metrics_image_similarity(info['light'], scale_invariant=True)
    metrics = {f'{prefix}_{k}': float(x) for prefix, m in (('nv', view), ('nl', light)) for k, x in m.items()}
    (out / 'evaluation.json').write_text(json.dumps(dict(metrics=metrics, scene=scene), indent=1) + '\n')
    print(json.dumps(metrics, indent=1))
    return metrics


def evaluate(result_dir, data_dir, out_dir):
    from .bake import bake
    data_dir, out = Path(data_dir), Path(out_dir)
    info = json.loads((data_dir / 'dataset.json').read_text())
    run_info = json.loads((Path(result_dir).parent / 'run.json').read_text())
    stage = out / 'stage'   # real files: Blender writes next to the textures
    if not (stage / 'envmap.exr').is_file():
        bake(result_dir, run_info['mesh'], stage)
    orb_root = (data_dir / info['orb_root']).resolve()
    render(stage, info['scene'], orb_root)
    return score(stage, info['scene'], orb_root, out)


if __name__ == '__main__':
    evaluate(*sys.argv[1:4])
