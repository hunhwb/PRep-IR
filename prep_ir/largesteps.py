"""Large Steps reparameterization x = x0 + B^-1 u with B = I + lambda L (Nicolet, Jacobson and Jakob, "Large
Steps in Inverse Rendering of Geometry", SIGGRAPH Asia 2021), and its UniformAdam optimizer.

Independent implementation of the paper's change of variables for three graphs: the mesh (geometry), the
occupied prior-space lattice (materials, Sec. 4.5) and the lat-long grid of the environment map (illumination).
Sparse Cholesky solves use cholespy (https://github.com/rgl-epfl/cholespy, BSD-3-Clause).
"""
import math

import numpy as np
import torch
from scipy import sparse


class SPDSolver:
    """Cholesky factorization of a fixed sparse SPD matrix B (n, n); solve() applies B^-1 to (n, C) tensors."""

    def __init__(self, matrix, device='cuda', dtype=torch.float32):
        from cholespy import CholeskySolverD, CholeskySolverF, MatrixType
        coo = sparse.coo_matrix(matrix)
        self.n, self.dtype = coo.shape[0], dtype
        t = lambda x, d: torch.as_tensor(np.asarray(x), device=device, dtype=d)
        factory = CholeskySolverD if dtype == torch.float64 else CholeskySolverF
        self._solver = factory(self.n, t(coo.row, torch.int64), t(coo.col, torch.int64), t(coo.data, dtype), MatrixType.COO)

    @torch.no_grad()
    def solve(self, rhs):
        if rhs.ndim != 2 or len(rhs) != self.n or not torch.isfinite(rhs).all():
            raise ValueError('finite (n, C) right-hand side required')
        b = rhs.to(self.dtype).contiguous()
        x = torch.empty_like(b)
        self._solver.solve(b, x)
        if not torch.isfinite(x).all():
            raise FloatingPointError('nonfinite solve')
        return x.to(rhs.dtype)


class _Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs, solver):
        ctx.solver = solver
        return solver.solve(rhs)

    @staticmethod
    def backward(ctx, grad):
        return ctx.solver.solve(grad.contiguous()), None   # B is symmetric


def inverse(rhs, solver):
    """B^-1 rhs, differentiable in rhs."""
    return _Inverse.apply(rhs, solver)


def laplacian(n, edges, weights=None):
    """Graph Laplacian D - A (scipy CSR) of undirected `edges` (E, 2) with nonnegative `weights` (default 1)."""
    e = np.asarray(edges.cpu() if torch.is_tensor(edges) else edges, np.int64)
    w = np.ones(len(e)) if weights is None else np.asarray(weights.cpu() if torch.is_tensor(weights) else weights, np.float64)
    A = sparse.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    return (sparse.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


def system(L, strength, normalize):
    """B = I + strength L / s, with s = 1 ('none'), the mean degree over all nodes ('mean_degree') or over the nodes
    with an edge ('mean_positive_degree')."""
    degree = L.diagonal()
    s = {'none': 1., 'mean_degree': degree.mean(), 'mean_positive_degree': degree[degree > 0].mean()}[normalize]
    return (sparse.identity(L.shape[0], format='csr') + L * (float(strength) / s)).tocsr()


def sphere_grid_edges(height, width):
    """Edges and finite-volume Laplace-Beltrami weights of the lat-long grid (row i at theta = i pi / (H - 1),
    column j at phi = 2 pi j / W, longitude wraps). Pole rows are a single point physically and are tied as a
    quarter-row ring, so a pole texel cannot move on its own."""
    dt, dp = np.pi / (height - 1), 2 * np.pi / width
    theta = np.arange(height) * dt
    ring = np.maximum(np.sin(theta), np.sin(dt / 4))
    idx = np.arange(height * width).reshape(height, width)
    east = np.stack((idx.ravel(), np.roll(idx, -1, 1).ravel()), 1)
    north = np.stack((idx[:-1].ravel(), idx[1:].ravel()), 1)
    w_east = np.repeat(dt / (ring * dp), width)
    w_north = np.repeat(np.sin(theta[:-1] + dt / 2) * dp / dt, width)
    return np.concatenate((east, north)), np.concatenate((w_east, w_north)).astype(np.float32)


class UniformAdam(torch.optim.Optimizer):
    """Adam with one shared second-moment scale per parameter tensor (the max over its entries), from Large Steps."""

    def __init__(self, params, lr=1e-3, betas=(.9, .999), eps=1e-8):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('finite positive learning rate required')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=0, first=torch.zeros_like(p), second=torch.zeros_like(p))
                state['step'] += 1
                state['first'].lerp_(p.grad, 1 - b1)
                state['second'].lerp_(p.grad.square(), 1 - b2)
                first = state['first'] / (1 - b1 ** state['step'])
                second = state['second'] / (1 - b2 ** state['step'])
                p.add_(first / (second.sqrt().max() + group['eps']), alpha=-group['lr'])
