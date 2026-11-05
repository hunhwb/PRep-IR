"""Per-vertex material logits parameterized through the prior-space lattice (Sec. 4.3-4.5 of the paper).

    Q_s = Q_0 + S^T B^-1 W     supported vertices (seen by at least one training view)
    Q_u = Q_0 + R              unsupported vertices (free per-vertex offsets)

S is the splat matrix of the supported vertices' prior features on a permutohedral lattice of spacing
h = spacing * sigma_prior, B = I + lambda L / mean degree the Large Steps system on the lattice graph, and W, R start
at zero. The gradient on W is B^-1 S g: photometric gradients pooled over vertices with similar predicted material.
The lattice, the graph and the vertex assignment stay fixed while the mesh moves.

`VertexMaterial` is the per-vertex control without a prior (Q = Q_0 + U), used by the paper's prior-use comparison.
"""
import time
import warnings

import numpy as np
import torch
from torch import nn

from .largesteps import SPDSolver, inverse, system
from .lattice import build_lattice


def prior_sigma(observations, variance):
    """sigma_prior: median over vertices seen in >= 2 views of the cross-view std (RMS over the feature channels)."""
    seen = torch.as_tensor(observations).reshape(-1) >= 2
    if not seen.any():
        raise ValueError('no vertex has two prior observations')
    return float(torch.as_tensor(variance).double().reshape(-1)[seen].clamp_min(0).sqrt().median())


def to_device_csr(matrix, device, dtype):
    c = matrix.tocsr()
    with warnings.catch_warnings():   # torch's beta notice for sparse CSR tensors
        warnings.simplefilter('ignore', UserWarning)
        return torch.sparse_csr_tensor(torch.tensor(c.indptr, device=device, dtype=torch.int64),
                                       torch.tensor(c.indices, device=device, dtype=torch.int64),
                                       torch.tensor(c.data, device=device, dtype=dtype), size=c.shape, device=device, dtype=dtype)


class _SparseMatmul(torch.autograd.Function):
    """y = A x for a fixed sparse A, with the precomputed transpose as the adjoint."""

    @staticmethod
    def forward(ctx, x, A, AT):
        ctx.AT = AT
        return torch.sparse.mm(A, x)

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.AT, g.contiguous()), None, None


class PriorLattice:
    """The fixed lattice shared by all material attributes of one scene."""

    def __init__(self, prior, spacing=1., large_steps=20., device='cuda'):
        """prior: dict(features (N, 5) fused albedo RGB / roughness / metallic, observations (N,), variance (N,))."""
        start = time.perf_counter()
        observations = torch.as_tensor(prior['observations']).reshape(-1)
        self.supported = torch.where(observations > 0)[0].to(device)
        self.sigma = prior_sigma(observations, prior['variance'])
        features = torch.as_tensor(prior['features']).double()[observations > 0].numpy()
        lattice = build_lattice(features / (float(spacing) * self.sigma))
        self.cells = lattice['splat'].shape[0]
        self.S = to_device_csr(lattice['splat'], device, torch.float32)       # (M, N_s): splat
        self.ST = to_device_csr(lattice['splat'].T, device, torch.float32)    # (N_s, M): slice
        self.solver = SPDSolver(system(lattice['laplacian'], large_steps, 'mean_degree'), device, torch.float64) if large_steps > 0 else None
        self.report = dict(sigma_prior=self.sigma, spacing=float(spacing) * self.sigma, cells=self.cells,
                           supported=len(self.supported), vertices=len(observations), large_steps=float(large_steps),
                           build_seconds=time.perf_counter() - start)


class LatticeMaterial(nn.Module):
    def __init__(self, reference, lattice):
        """reference: (N, C) initial logits Q_0; lattice: PriorLattice."""
        super().__init__()
        n = len(reference)
        self.register_buffer('reference', reference.detach().clone())
        self.lattice = lattice
        mask = torch.zeros(n, dtype=torch.bool, device=reference.device)
        mask[lattice.supported] = True
        self.unsupported = torch.where(~mask)[0]
        # u = [W (one row per lattice vertex); R (one row per unsupported vertex)]
        self.u = nn.Parameter(torch.zeros((lattice.cells + len(self.unsupported), reference.shape[1]), device=reference.device))

    def forward(self):
        M = self.lattice.cells
        w = self.u[:M] if self.lattice.solver is None else inverse(self.u[:M], self.lattice.solver)
        sliced = _SparseMatmul.apply(w, self.lattice.ST, self.lattice.S)
        delta = torch.zeros_like(self.reference).index_copy(0, self.lattice.supported, sliced)
        return self.reference + delta.index_copy(0, self.unsupported, self.u[M:])


class VertexMaterial(nn.Module):
    """Control: free per-vertex logit offsets."""

    def __init__(self, reference):
        super().__init__()
        self.register_buffer('reference', reference.detach().clone())
        self.u = nn.Parameter(torch.zeros_like(self.reference))

    def forward(self):
        return self.reference + self.u


def material_logits(value, low=0., high=1., count=None, channels=1, device='cuda'):
    """Constant logits of `value` for a sigmoid mapped onto [low, high]."""
    x = torch.full((count, channels), float(value), device=device)
    return torch.logit(((x - low) / (high - low)).clamp(1e-6, 1 - 1e-6))
