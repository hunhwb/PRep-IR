"""Selection preserves albedo support and cannot invent colours from RGB guidance."""
import pytest
import torch
from prep_ir.refinement import RefinementConfig, refine_albedo


def test_constant_prediction_is_preserved_under_textured_guidance():
    torch.manual_seed(3)
    p = torch.full((1, 3, 12, 16), .27)
    result = refine_albedo(torch.rand(1, 3, 24, 32), p, torch.ones(1, 1, 12, 16))
    assert result.shape == (1, 3, 24, 32)
    assert torch.allclose(result, torch.full_like(result, .27), atol=2e-6)


def test_invalid_candidates_cannot_change_valid_interior():
    torch.manual_seed(2)
    p = torch.full((1, 3, 18, 18), .2)
    mask = torch.ones(1, 1, 18, 18)
    mask[..., :4] = 0
    changed = p.clone()
    changed[..., :4] = .9
    image = torch.rand_like(p)
    a = refine_albedo(image, p, mask)
    b = refine_albedo(image, changed, mask)
    assert torch.allclose(a[..., 7:-3, 7:-3], b[..., 7:-3, 7:-3], atol=2e-6)


def test_empty_support_uses_bilinear_prediction_and_no_nan():
    torch.manual_seed(1)
    p = torch.rand(1, 3, 8, 10)
    a = refine_albedo(torch.rand(1, 3, 16, 20), p, torch.zeros(1, 1, 8, 10))
    expected = torch.nn.functional.interpolate(p, (16, 20), mode='bilinear', align_corners=False)
    assert torch.equal(a, expected)


def test_invalid_bandwidths_are_rejected():
    with pytest.raises(ValueError):
        RefinementConfig(prior_sigma=0)
