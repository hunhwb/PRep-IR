"""Large Steps systems and their sparse Cholesky solves."""
import numpy as np
import torch

from prep_ir.largesteps import SPDSolver, inverse, laplacian, sphere_grid_edges, system


def random_graph(n=60, seed=0):
    rng = np.random.default_rng(seed)
    edges = np.unique(np.sort(rng.integers(0, n, (4 * n, 2)), 1), axis=0)
    return edges[edges[:, 0] != edges[:, 1]]


def test_solve_matches_dense_inverse():
    n = 60
    B = system(laplacian(n, random_graph(n)), 20., 'mean_degree')
    rhs = torch.randn(n, 3, dtype=torch.float64)
    x = SPDSolver(B, 'cpu', torch.float64).solve(rhs)
    assert torch.allclose(x, torch.linalg.solve(torch.tensor(B.toarray()), rhs), atol=1e-10)


def test_inverse_is_self_adjoint():
    n = 40
    solver = SPDSolver(system(laplacian(n, random_graph(n, 1)), 5., 'none'), 'cpu', torch.float64)
    u = torch.randn(n, 2, dtype=torch.float64, requires_grad=True)
    g = torch.randn(n, 2, dtype=torch.float64)
    inverse(u, solver).backward(g)
    assert torch.allclose(u.grad, solver.solve(g))


def test_sphere_grid_weights_are_positive_and_wrap():
    edges, weights = sphere_grid_edges(8, 16)
    assert (weights > 0).all() and len(edges) == 8 * 16 + 7 * 16
    assert [0, 15] in edges[:, ::-1].tolist() or [15, 0] in edges.tolist()   # longitude wraps
