"""Dielectric-metallic dual hypothesis (supplement, Sec. "Dielectric-Metallic Dual Hypothesis").

Every vertex carries a dielectric hypothesis (base colour, roughness; metallic 0) and a metal hypothesis (metallic 1),
both on the material lattice. A binary label selects the hypothesis that represents the vertex. Per-pixel photometric
residuals of both hypotheses are pooled to vertices, accumulated with exponential decay, pooled again on the lattice
and smoothed with its Large Steps operator, and compared:

    d = (L^D - L^M) / (L^D + L^M),   w = sigmoid((d - margin) / tau),   metal where the sliced w exceeds 1/2.
"""
import torch


def renders_both(config, step):
    """Whether step `step` renders the unselected hypotheses too: every step up to `freeze_after`, then every
    `every`-th step."""
    return step <= config.freeze_after or step % config.every == 0


def ramp(step, start, end):
    """Linear ramp from 0 at `start` to 1 at `end` (constant 1 when end <= start)."""
    return 1. if end <= start else min(1., max(0., (step - start) / (end - start)))


def temperature(config, step, steps):
    """tau decays geometrically from tau0 at step 1 to tau1 at the last step."""
    return config.tau0 * (config.tau1 / config.tau0) ** ((step - 1) / max(1, steps - 1))


def metal_weight(residual, coverage, observed, tau, margin=0.):
    """w^M = sigmoid((d - margin) / tau) from residual and coverage sums (rows, 2) [dielectric, metal]. Rows without
    evidence for both hypotheses get w = 1/2, which leaves them dielectric."""
    mean = residual / coverage.clamp_min(1e-30)
    advantage = (mean[:, :1] - mean[:, 1:]) / (mean.sum(1, keepdim=True) + 1e-8)
    valid = observed.all(1, keepdim=True)
    return torch.where(valid, torch.sigmoid((advantage - margin) / tau), torch.full_like(advantage, .5))


def select(residual, coverage, observed, tau, margin=0., lattice=None, solver=None, slice_weight=None):
    """Per-vertex metal weight. Supported vertices take the lattice decision: the residual, coverage and observation
    sums are splatted to the lattice, the sums smoothed with `solver` (B^-1, as the material gradients), decided per
    lattice point and sliced back with normalized weights. Vertices outside the lattice decide from their own sums."""
    w = metal_weight(residual, coverage, observed, tau, margin)
    if lattice is None:
        return w
    sup = lattice.supported
    pooled = torch.sparse.mm(lattice.S, torch.cat((residual[sup], coverage[sup], observed[sup].float()), 1))
    sums = pooled[:, :4]
    if solver is not None:
        sums = solver.solve(sums.double()).float().clamp_min(0.)
    cells = metal_weight(sums[:, :2], sums[:, 2:4], pooled[:, 4:] > 0, tau, margin)
    if slice_weight is None:
        slice_weight = torch.sparse.mm(lattice.ST, torch.ones_like(cells)).clamp_min(1e-12)
    w[sup] = torch.sparse.mm(lattice.ST, cells) / slice_weight
    return w
