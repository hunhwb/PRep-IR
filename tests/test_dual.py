"""Dual-hypothesis selection: render schedule, residual comparison and lattice pooling."""
from pathlib import Path
from types import SimpleNamespace

import torch

from prep_ir.config import load_config
from prep_ir.dual import metal_weight, ramp, renders_both, select, temperature

EXPERIMENTS = Path(__file__).resolve().parents[1] / 'configs' / 'experiments'


def test_paper_settings_and_schedule():
    c = load_config([EXPERIMENTS / 'dual_hypothesis.yaml'])
    assert c.material.dual_hypothesis and not c.material.learn_metallic
    assert (c.dual.freeze_after, c.dual.every, c.dual.margin, c.dual.margin_from, c.dual.margin_until) == (400, 4, .02, 300, 600)
    both = [s for s in range(1, c.steps + 1) if renders_both(c.dual, s)]
    assert both[:400] == list(range(1, 401)) and all(s % 4 == 0 for s in both[400:])
    assert len(both) == 550   # 1.55 renders per view on average
    assert abs(temperature(c.dual, 1, 1000) - .3) < 1e-12 and abs(temperature(c.dual, 1000, 1000) - .01) < 1e-12
    assert (ramp(300, 300, 600), ramp(450, 300, 600), ramp(900, 300, 600), ramp(5, 0, 0)) == (0., .5, 1., 1.)


def test_binarization_files_learn_metallic():
    for name, weight in (('metallic_binarization_001', .01), ('metallic_binarization_01', .1)):
        c = load_config([EXPERIMENTS / f'{name}.yaml'])
        assert c.material.learn_metallic and c.loss.metallic_binarization == weight
        assert (c.loss.binarization_from, c.loss.binarization_until) == (300, 600)
    assert load_config().loss.metallic_binarization == 0 and not load_config().material.dual_hypothesis


def test_residuals_are_normalized_by_coverage():
    # metal fits better per pixel in row 0 despite less coverage; row 1 prefers dielectric; row 2 has no evidence
    residual = torch.tensor([[20., 1.], [1., 2.], [0., 0.]])
    coverage = torch.tensor([[100., 10.], [10., 10.], [0., 0.]])
    w = metal_weight(residual, coverage, coverage > 0, .1)
    assert w[0] > .9 and w[1] < .1 and w[2] == .5
    # old (decayed) evidence keeps its preference
    torch.testing.assert_close(metal_weight(residual * 1e-10, coverage * 1e-10, coverage > 0, .1), w)


def test_one_sided_evidence_and_margin_keep_dielectric():
    residual, coverage = torch.tensor([[.5, .1]]), torch.tensor([[1., 0.]])
    assert metal_weight(residual, coverage, coverage > 0, .1) == .5
    # d = (.51 - .49) / 1 = .02: no preference without a margin, dielectric with a larger one
    residual, coverage = torch.tensor([[.51, .49]]), torch.ones(1, 2)
    assert metal_weight(residual, coverage, coverage > 0, .01, .0) > .5
    assert metal_weight(residual, coverage, coverage > 0, .01, .05) < .5


def test_lattice_decision_is_sliced_back_and_unsupported_vertices_decide_alone():
    S = torch.tensor([[1., .2, 0.], [0., .8, 1.]])
    lattice = SimpleNamespace(S=S.to_sparse(), ST=S.T.to_sparse(), supported=torch.tensor([0, 1, 2]))
    residual = torch.tensor([[.2, .1], [.05, .15], [.3, .4], [.1, .5]])
    coverage = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [2., 2.]])
    observed = coverage > 0
    w = select(residual, coverage, observed, .1, lattice=lattice)
    cells = metal_weight(S @ residual[:3], S @ coverage[:3], (S @ observed[:3].float()) > 0, .1)
    expected = torch.cat(((S.T @ cells) / (S.T @ torch.ones_like(cells)), metal_weight(residual[3:], coverage[3:], observed[3:], .1)))
    torch.testing.assert_close(w, expected)
