"""Permutohedral lattice: barycentric weights, splat/slice, and the lattice graph."""
import numpy as np
from scipy import sparse

from prep_ir.lattice import build_lattice, simplex


def test_barycentric_weights_are_a_partition_of_unity():
    x = np.random.default_rng(0).uniform(0, 3, (500, 5))
    keys, bary = simplex(x)
    assert keys.shape == (500, 6, 5) and bary.shape == (500, 6)
    assert np.all(bary >= 0) and np.allclose(bary.sum(1), 1)


def test_identical_features_share_lattice_vertices_and_weights():
    x = np.repeat(np.random.default_rng(1).uniform(0, 2, (10, 5)), 3, axis=0)
    S = build_lattice(x)['splat'].toarray()
    assert np.allclose(S[:, 0::3], S[:, 1::3]) and np.allclose(S[:, 0::3], S[:, 2::3])
    assert np.allclose(S.sum(0), 1)   # every sample's weights sum to one


def test_laplacian_is_symmetric_with_zero_row_sums():
    L = build_lattice(np.random.default_rng(2).uniform(0, 4, (2000, 5)))['laplacian']
    assert abs(L - L.T).max() == 0
    assert np.allclose(np.asarray(L.sum(1)).ravel(), 0)
    off = L - sparse.diags(L.diagonal())
    assert off.max() <= 0   # unit-weight edges


def test_coarser_spacing_gives_fewer_cells():
    x = np.random.default_rng(3).uniform(0, 1, (5000, 5))
    assert build_lattice(x / .05)['splat'].shape[0] > build_lattice(x / .2)['splat'].shape[0]
