"""Permutohedral lattice over per-vertex prior features (Sec. 3.2 / 4.3 of the paper).

The simplex embedding follows Adams, Baek and Davis, "Fast High-Dimensional Filtering Using the Permutohedral
Lattice" (Eurographics 2010), and its reference implementation's conventions (e.g. rounding ties down), as in
https://github.com/MiguelMonteiro/permutohedral_lattice (MIT License, Copyright (c) 2018 Miguel Monteiro,
Andrew Adams, Jongmin Baek, Abe Davis). This is an independent NumPy implementation of the embedding; the graph
Laplacian on the occupied lattice vertices and its use as an optimization variable are ours.
"""
import numpy as np
from scipy import sparse


def simplex(features):
    """Enclosing-simplex keys (n, d+1, d) and barycentric weights (n, d+1) of each row of `features` (n, d)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('finite nonempty (n, d) features with d >= 2 required')
    n, d = x.shape
    # elevate onto the hyperplane x . 1 = 0 in R^(d+1)
    scale = (d + 1) * np.sqrt(2 / 3) / np.sqrt(np.arange(1, d + 1) * np.arange(2, d + 2))
    elevated = np.zeros((n, d + 1))
    acc = np.zeros(n)
    for i in range(d, 0, -1):
        cf = x[:, i - 1] * scale[i - 1]
        elevated[:, i] = acc - i * cf
        acc += cf
    elevated[:, 0] = acc
    # nearest remainder-zero lattice point (ties round down, as in the reference implementation)
    z = elevated / (d + 1)
    rem = np.where(np.ceil(z) - z < z - np.floor(z), np.ceil(z), np.floor(z)).astype(np.int64) * (d + 1)
    rank = np.zeros((n, d + 1), dtype=np.int64)
    delta = elevated - rem
    for i in range(d):
        for j in range(i + 1, d + 1):
            less = delta[:, i] < delta[:, j]
            rank[:, i] += less
            rank[:, j] += ~less
    rank += (rem.sum(1) // (d + 1))[:, None]
    low, high = rank < 0, rank > d
    rank += (d + 1) * (low.astype(int) - high.astype(int))
    rem += (d + 1) * (low.astype(int) - high.astype(int))
    # barycentric coordinates
    bary = np.zeros((n, d + 2))
    rows = np.arange(n)
    for i in range(d + 1):
        delta = (elevated[:, i] - rem[:, i]) / (d + 1)
        bary[rows, d - rank[:, i]] += delta
        bary[rows, d - rank[:, i] + 1] -= delta
    bary[:, 0] += 1 + bary[:, d + 1]
    bary = bary[:, :d + 1]
    if bary.min() < -1e-9:
        raise ArithmeticError('invalid barycentric coordinates')
    bary = np.maximum(bary, 0)
    bary /= bary.sum(1, keepdims=True)
    remainder = np.arange(d + 1)[None, :, None]
    keys = rem[:, None, :d] + remainder - (rank[:, None, :d] > d - remainder) * (d + 1)
    return keys, bary


def _records(keys):
    return np.ascontiguousarray(keys).view(np.dtype([(f'x{i}', np.int64) for i in range(keys.shape[1])])).ravel()


def _occupied(features):
    """Splat matrix S (M, n) over the M occupied lattice vertices, their sorted key records, and their keys."""
    keys, bary = simplex(features)
    n, width = bary.shape
    d = width - 1
    positive = bary.ravel() > 0   # no zero-weight entries at exact simplex boundaries
    unique, inverse = np.unique(_records(keys.reshape(-1, d)[positive]), return_inverse=True)
    sample = np.repeat(np.arange(n), width)[positive]
    S = sparse.coo_matrix((bary.ravel()[positive], (inverse.ravel(), sample)), shape=(len(unique), n)).tocsr()
    return S, unique, unique.view(np.int64).reshape(-1, d)


def _neighbors(unique, lattice_keys, axis):
    """(i, j) pairs of occupied lattice vertices that are neighbours along lattice direction `axis`."""
    d = lattice_keys.shape[1]
    neighbor = lattice_keys + 1
    if axis < d:
        neighbor[:, axis] -= d + 1
    idx = np.searchsorted(unique, _records(neighbor))
    safe = np.minimum(idx, len(unique) - 1)
    valid = (idx < len(unique)) & (unique[safe] == _records(neighbor))
    return np.flatnonzero(valid), idx[valid]


def build_lattice(features):
    """Lattice of `features` (n, d), already divided by the lattice spacing.

    Returns dict(splat=S (M, n) CSR barycentric weights, laplacian=L (M, M) CSR of the unit-weight graph that links
    lattice neighbours, L = D - A).
    """
    S, unique, lattice_keys = _occupied(features)
    m = len(unique)
    pairs = [_neighbors(unique, lattice_keys, axis) for axis in range(lattice_keys.shape[1] + 1)]
    a = np.concatenate([p[0] for p in pairs])
    b = np.concatenate([p[1] for p in pairs])
    A = sparse.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(m, m)).tocsr()
    L = sparse.diags(np.asarray(A.sum(1)).ravel()) - A
    return dict(splat=S, laplacian=L.tocsr())


def gaussian_blur_operators(features):
    """Splat S and the d+1 separable [1 2 1]/4 blurs along the lattice directions (absent neighbours dropped):
    Adams et al.'s approximate Gaussian filter, used by the prior-as-loss control (prep_ir.prior_loss)."""
    S, unique, lattice_keys = _occupied(features)
    m = len(unique)
    blurs = []
    for axis in range(lattice_keys.shape[1] + 1):
        a, b = _neighbors(unique, lattice_keys, axis)
        A = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
        blurs.append(((2 * sparse.identity(m) + A + A.T) / 4).tocsr())
    return S, blurs
