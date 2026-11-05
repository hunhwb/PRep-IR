"""Default refinement must reach preparation and avoid reusing unrefined caches."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from prep_ir import scene
from prep_ir.refinement import RefinementConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('disable', [False, True])
def test_scene_api_refines_by_default_with_explicit_opt_out(tmp_path, monkeypatch, disable):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'dataset.json').write_text(json.dumps(dict(nsr_to_world=np.eye(3).tolist(), textured_eval=False)))
    monkeypatch.setattr(scene, 'load_reconstruction', lambda *a, **kw: (np.zeros((3, 3)), np.array([[0, 1, 2]]), {}))
    monkeypatch.setitem(sys.modules, 'prep_ir.render', types.SimpleNamespace(Renderer=lambda *a: object()))
    received = {}

    def fuse(*args, **kwargs):
        received.update(kwargs)
        return dict(observations=torch.ones(3))

    monkeypatch.setattr(scene, 'fuse_prior', fuse)
    kwargs = dict(refinement=None) if disable else {}
    scene.prepare_scene(data, tmp_path / 'mesh.obj', tmp_path / 'prior', tmp_path / 'scene.pt', log=lambda *a: None, **kwargs)
    assert received['refinement'] == (None if disable else RefinementConfig())


@pytest.mark.parametrize('flags, expected', [([], RefinementConfig()), (['--no-refine-albedo'], None)])
def test_preparation_cli_defaults_and_opt_out(tmp_path, monkeypatch, flags, expected):
    received = {}

    def prepare(*args, **kwargs):
        received.update(kwargs)
        return dict(report={})

    monkeypatch.setattr(scene, 'prepare_scene', prepare)
    monkeypatch.setattr(sys, 'argv', ['prepare_scene.py', '--data', str(tmp_path), '--mesh', 'mesh.obj',
                                     '--prior', 'prior', '--out', str(tmp_path / 'scene.pt'), *flags])
    runpy.run_path(str(ROOT / 'scripts/prepare_scene.py'), run_name='__main__')
    assert received['refinement'] == expected


@pytest.mark.parametrize('flags, refined', [([], True), (['--no-refine-albedo'], False)])
def test_pipeline_uses_separate_refined_cache(tmp_path, flags, refined):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy(ROOT / 'scripts/run_scene.sh', scripts)
    data = tmp_path / 'data/s4r/hotdog'
    data.mkdir(parents=True)
    (data / 'dataset.json').write_text('{"diffreg_data": "unused"}')
    work = tmp_path / 'work/s4r/hotdog'
    for directory in ['priors/mvinverse', 'nsr_data/mvinverse/hotdog/normal_single', 'mesh/mvinverse']:
        (work / directory).mkdir(parents=True)
    (work / 'priors/mvinverse/mapping.json').write_text('{}')
    (work / 'mesh/mvinverse/to_worlds_after.txt').touch()
    # This cache predates refinement and must only be used by the explicit opt-out.
    (work / 'scene_mvinverse.pt').write_bytes(b'unrefined')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'python'
    stub.write_text(f'#!{sys.executable}\n' + '''import json, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    print('unused')
    sys.exit()
with open('calls.jsonl', 'a') as f:
    f.write(json.dumps(args) + '\\n')
if args[0].endswith('prepare_scene.py'):
    Path(args[args.index('--out') + 1]).write_bytes(b'refined')
elif args[0].endswith('train.py'):
    p = Path(args[args.index('--out') + 1]) / 'result/result.npz'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
elif args[0].endswith('evaluate.py'):
    p = Path(args[1]) / 'eval/evaluation.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{}')
''')
    stub.chmod(0o755)
    subprocess.run(['bash', str(scripts / 'run_scene.sh'), 's4r', 'hotdog', 'mvinverse', *flags],
                   env=dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH']), check=True)
    calls = [json.loads(x) for x in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    train = next(x for x in calls if x[0].endswith('train.py'))
    name = 'mvinverse_refined' if refined else 'mvinverse'
    assert Path(train[train.index('--scene') + 1]).name == f'scene_{name}.pt'
    assert Path(train[train.index('--out') + 1]).name == name
    preparation = [x for x in calls if x[0].endswith('prepare_scene.py')]
    assert bool(preparation) == refined
    if refined:
        assert '--refine-albedo' in preparation[0]
    assert (work / 'scene_mvinverse.pt').read_bytes() == b'unrefined'
