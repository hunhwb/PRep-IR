"""Configuration loading: defaults, experiment files, overrides, typo protection."""
from pathlib import Path

import pytest

from prep_ir.config import load_config

EXPERIMENTS = Path(__file__).resolve().parents[1] / 'configs' / 'experiments'


def test_defaults_are_the_paper_method():
    c = load_config()
    assert (c.steps, c.views_per_step, c.material.parameterization, c.material.lattice_spacing, c.material.large_steps) == (1000, 2, 'lattice', 1.0, 20.0)
    assert (c.geometry.large_steps, c.geometry.lr, c.envmap.sg_steps, c.envmap.large_steps, c.envmap.lr) == (19.0, 6e-4, 250, 6.0, .04)


def test_every_experiment_file_loads():
    for f in EXPERIMENTS.glob('*.yaml'):
        load_config([f])


def test_overrides_and_typos():
    c = load_config([EXPERIMENTS / 'learned_metallic.yaml'], ['seed=7', 'render.spp=64'])
    assert (c.material.learn_metallic, c.seed, c.render.spp) == (True, 7, 64)
    with pytest.raises(KeyError):
        load_config([], ['material.lr_albdo=1'])
