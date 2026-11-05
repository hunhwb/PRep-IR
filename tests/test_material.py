"""Lattice material field: forward Q = Q0 + S^T B^-1 W (+ R), adjoint B^-1 S g."""
import numpy as np
import torch

from prep_ir.material import LatticeMaterial, PriorLattice, prior_sigma


def toy_prior(n=400, seed=0):
    rng = np.random.default_rng(seed)
    features = torch.tensor(rng.uniform(0, 1, (n, 5)), dtype=torch.float32)
    observations = torch.tensor(rng.integers(0, 4, n), dtype=torch.float32)
    variance = torch.tensor(rng.uniform(1e-4, 1e-2, n), dtype=torch.float32)
    return dict(features=features, observations=observations, variance=variance)


def dense(lattice):
    return lattice.S.to_dense().double()


def test_sigma_is_the_median_cross_view_std_of_vertices_seen_twice():
    prior = toy_prior()
    seen = prior['observations'] >= 2
    assert np.isclose(prior_sigma(prior['observations'], prior['variance']), float(prior['variance'][seen].double().sqrt().median()))


def test_zero_variables_reproduce_the_reference():
    prior = toy_prior()
    lattice = PriorLattice(prior, large_steps=20., device='cpu')
    field = LatticeMaterial(torch.full((400, 3), .5), lattice)
    assert torch.equal(field(), torch.full((400, 3), .5))


def test_forward_and_adjoint_match_the_dense_formula():
    prior = toy_prior()
    lattice = PriorLattice(prior, large_steps=20., device='cpu')
    field = LatticeMaterial(torch.zeros(400, 1), lattice)
    with torch.no_grad():
        field.u.normal_()
    S = dense(lattice)
    M = lattice.cells
    # B from the solver's own matrix: B^-1 applied to the identity
    Binv = lattice.solver.solve(torch.eye(M, dtype=torch.float64))
    supported, unsupported = lattice.supported, field.unsupported
    q = field()
    expected = S.T @ Binv @ field.u[:M].double()
    assert torch.allclose(q[supported].double(), expected, atol=1e-5)
    assert torch.equal(q[unsupported], field.u[M:])
    g = torch.randn(400, 1)
    q.backward(g)
    assert torch.allclose(field.u.grad[:M].double(), Binv @ S @ g[supported].double(), atol=1e-5)
    assert torch.equal(field.u.grad[M:], g[unsupported])
