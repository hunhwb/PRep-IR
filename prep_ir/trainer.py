"""Joint optimization of geometry, materials and illumination with a purely photometric objective (Sec. 4).

One step: render `views_per_step` training views from the current state, back-propagate the photometric loss (plus
the background term when the frames observe the environment) to the decoded per-vertex positions, material logits and
log envmap, clip per-vertex gradient norms, pull the gradients back through each block's parameterization, and update
all blocks from the same renderings:

    geometry   v = v0 + diag * (I + 19 L_mesh)^-1 u                       UniformAdam
    materials  Q = Q0 + S^T (I + 20 L_lattice / mean degree)^-1 W  (+ R)   Adam
    envmap     SG lobes for 250 steps, then log E0 + (I + 6 K / mean degree)^-1 w   Adam, warm-up + cosine

With `material.dual_hypothesis` each step renders the selected and the unselected hypotheses instead (prep_ir/dual.py).
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import as_plain
from .data import Views
from .dual import ramp, renders_both, select, temperature
from .envmap import BackgroundObservations, SGEnvmap, TextureEnvmap
from .image import linear_to_srgb, write_exr
from .largesteps import SPDSolver, UniformAdam, inverse, laplacian, system
from .losses import clip_vertex_norms, photometric_loss
from .material import LatticeMaterial, PriorLattice, VertexMaterial, material_logits
from .mesh import unique_edges
from .prior_loss import PriorBilateralLoss
from .render import ENV_TO_WORLD, Renderer

MATERIALS = ('albedo', 'roughness', 'metallic', 'albedo_m', 'roughness_m')   # *_m: metal hypothesis


def background_views(data_dir, views):
    """Frames whose background pixels observe the environment: <data>/background (same cameras, HDR) if present,
    else the training frames themselves when they are HDR, else None."""
    root = Path(data_dir) / 'background'
    if not root.is_dir():
        return views if views.hdr else None
    bg = Views(root, 'train', views.size[0])
    if len(bg) != len(views) or bg.size != views.size or abs(bg.fov - views.fov) > 1e-6 or not bg.hdr \
            or any(not np.allclose(a, b, atol=1e-6) for a, b in zip(bg.transforms, views.transforms)):
        raise ValueError(f'{root}: background frames must be HDR and match the training cameras')
    return bg


class Trainer:
    def __init__(self, scene, config, out_dir, log=print):
        c = self.config = config
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = log
        torch.manual_seed(c.seed)
        self.views = Views(scene['data_dir'], 'train', c.resolution).preload()
        if c.views_per_step > len(self.views):
            raise ValueError('views_per_step exceeds the number of training views')
        v0, faces = scene['vertices'].cuda(), scene['faces'].cuda()
        n = len(v0)
        r = c.render
        # geometry.lr 0 holds the mesh fixed: no vertex gradients, so the boundary (edge) derivative terms, which only
        # contribute to vertex gradients, are not sampled either
        self.fixed_geometry = c.geometry.lr == 0
        edges = (0, 0) if self.fixed_geometry else (r.spp_primary_edge, r.spp_indirect_edge)
        self.renderer = Renderer(v0, faces, envmap_shape=(c.envmap.height, c.envmap.width), max_depth=r.max_depth,
                                 derivative_spp=(r.spp_interior, *edges), cos_floor=r.bsdf_cos_floor)

        # geometry: mesh Large Steps on world-space displacements, scaled by the bounding-box diagonal
        self.v0 = v0
        self.diagonal = (v0.amax(0) - v0.amin(0)).norm()
        self.geometry_solver = SPDSolver(system(laplacian(n, unique_edges(faces)), c.geometry.large_steps, 'none'))
        self.geometry_u = nn.Parameter(torch.zeros_like(v0))
        self.geometry_optimizer = UniformAdam([dict(params=[self.geometry_u], name='geometry', lr=c.geometry.lr)])

        # materials: logits on the prior-space lattice (ours) or per vertex (control)
        m = c.material
        if m.parameterization == 'lattice':
            self.lattice = PriorLattice(scene['prior'], m.lattice_spacing, m.large_steps)
            field = lambda reference: LatticeMaterial(reference, self.lattice)
            log(f'lattice: {self.lattice.report}')
        elif m.parameterization == 'vertex':
            self.lattice = None
            field = VertexMaterial
        else:
            raise ValueError(m.parameterization)
        self.albedo = field(material_logits(m.init_albedo, 0., 1., n, 3))
        self.roughness = field(material_logits(m.init_roughness, .09, .99, n, 1))
        self.metallic = field(torch.full((n, 1), float(m.init_metallic), device='cuda').logit()) if m.learn_metallic else None
        groups = [dict(params=[self.albedo.u], name='albedo', lr=m.lr_albedo),
                  dict(params=[self.roughness.u], name='roughness', lr=m.lr_roughness)]
        if self.metallic is not None:
            groups.append(dict(params=[self.metallic.u], name='metallic', lr=m.lr_roughness))
        # dual hypothesis: albedo / roughness are the dielectric hypothesis, albedo_m / roughness_m the metal one
        self.dual = bool(m.dual_hypothesis)
        self.albedo_m = self.roughness_m = None
        if self.dual:
            if m.learn_metallic or c.loss.prior_weight > 0:
                raise ValueError('material.dual_hypothesis excludes learn_metallic and the prior loss')
            self.albedo_m = field(material_logits(c.dual.init_metal, 0., 1., n, 3))
            self.roughness_m = field(material_logits(m.init_roughness, .09, .99, n, 1))
            groups += [dict(params=[self.albedo_m.u], name='albedo_m', lr=m.lr_albedo),
                       dict(params=[self.roughness_m.u], name='roughness_m', lr=m.lr_roughness)]
            self.residual = torch.zeros((n, 2), device='cuda')      # decayed per-vertex residual sums (dielectric, metal)
            self.coverage = torch.zeros((n, 2), device='cuda')      # decayed per-vertex pixel weights
            self.observed = torch.zeros((n, 2), dtype=torch.bool, device='cuda')
            self.selection = torch.full((n, 1), .5, device='cuda')  # metal weight w per vertex; metal where w > 1/2
            self.slice_weight = None if self.lattice is None else \
                torch.sparse.mm(self.lattice.ST, torch.ones((self.lattice.cells, 1), device='cuda')).clamp_min(1e-12)
        self.material_optimizer = torch.optim.Adam(groups)
        self.prior_loss = PriorBilateralLoss(scene['prior'], scene['vertices'], scene['faces']) if c.loss.prior_weight > 0 else None

        # envmap: spherical Gaussians fitted to the observed background (or to the constant start where there is none)
        bg_views = background_views(scene['data_dir'], self.views)
        self.background = BackgroundObservations(bg_views, stride=c.loss.background_stride) if bg_views is not None else None
        start = torch.full((c.envmap.height, c.envmap.width, 3), float(c.envmap.init), device='cuda').clamp_min(1e-8).log()
        target = self.background.average(ENV_TO_WORLD) if self.background is not None else start.exp()
        self.envmap = SGEnvmap((c.envmap.height, c.envmap.width), target, lobes=c.envmap.sg_lobes)
        self.envmap_optimizer = torch.optim.Adam(self.envmap.parameters(), lr=c.envmap.lr, betas=(.9, .99))
        self.step = 0

    # ----------------------------------------------------------------------------------------------- state
    def vertices(self):
        if self.fixed_geometry:
            return self.v0
        return self.v0 + inverse(self.geometry_u, self.geometry_solver) * self.diagonal

    def decoded(self):
        """Differentiable decoded state: world vertices, material logits and log envmap."""
        state = dict(geometry=self.vertices(), albedo=self.albedo(), roughness=self.roughness(), envmap=self.envmap())
        if self.metallic is not None:
            state['metallic'] = self.metallic()
        if self.dual:
            state['albedo_m'], state['roughness_m'] = self.albedo_m(), self.roughness_m()
        return state

    @staticmethod
    def physical(state):
        """Renderer inputs: vertices, linear albedo, perceptual roughness in [.09, .99], metallic or None, envmap."""
        metallic = state['metallic'].sigmoid() if 'metallic' in state else None
        return state['geometry'], state['albedo'].sigmoid(), .09 + .9 * state['roughness'].sigmoid(), metallic, state['envmap'].exp()

    def _set_learning_rates(self, step):
        c = self.config
        for g in self.material_optimizer.param_groups:
            g['lr'] = c.material.lr_albedo if g['name'] in ('albedo', 'albedo_m') else c.material.lr_roughness
        self.geometry_optimizer.param_groups[0]['lr'] = c.geometry.lr
        e = c.envmap
        progress = min(1., max(0, step - e.warmup) / max(1, c.steps - e.warmup))
        cosine = e.min_lr_factor + (1 - e.min_lr_factor) * .5 * (1 + math.cos(math.pi * progress))
        for g in self.envmap_optimizer.param_groups:
            g['lr'] = e.lr * min(1., step / max(1, e.warmup)) * cosine

    # ----------------------------------------------------------------------------------------------- one step
    def train_step(self, step):
        c = self.config
        if isinstance(self.envmap, SGEnvmap) and step > c.envmap.sg_steps:
            # second illumination phase: the SG envmap becomes the reference of a Large Steps texture
            self.envmap = TextureEnvmap(self.envmap().detach(), c.envmap.large_steps)
            self.envmap_optimizer = torch.optim.Adam(self.envmap.parameters(), lr=c.envmap.lr, betas=(.9, .99))
        start = time.perf_counter()
        for opt in (self.geometry_optimizer, self.material_optimizer, self.envmap_optimizer):
            opt.zero_grad(set_to_none=True)
        self._set_learning_rates(step)
        indices = np.random.default_rng(c.seed + step).choice(len(self.views), c.views_per_step, replace=False).tolist()

        decoded = self.decoded()
        leaves = {k: x.detach().requires_grad_(not (k == 'geometry' and self.fixed_geometry)) for k, x in decoded.items()}
        losses, fractions, extra = [], [], {}
        if self.dual:
            losses, fractions, extra = self._dual_step(leaves, indices, step)
        for j, index in enumerate([] if self.dual else indices):
            v, a, r, metallic, e = self.physical(leaves)   # a fresh activation graph per view (freed by its backward)
            image = self.renderer.render(v, a, r, e, self.views, index, spp=c.render.spp, seed=c.seed + step * 1000 + j, metallic=metallic)
            batch = self.views.batch(index)
            fractions.append(float(batch['mask'].mean()))
            loss = photometric_loss(image, batch['rgb'], c.loss.ssim_weight)
            (loss / len(indices)).backward()
            losses.append(float(loss.detach()))
        if self.prior_loss is not None:
            prior = self.prior_loss(leaves['albedo'].sigmoid(), .09 + .9 * leaves['roughness'].sigmoid(), float(np.mean(fractions)))
            (c.loss.prior_weight * prior).backward()
        if self.metallic is not None and c.loss.metallic_binarization > 0:
            weight = c.loss.metallic_binarization * ramp(step, c.loss.binarization_from, c.loss.binarization_until)
            m = leaves['metallic'].sigmoid().clamp(1e-6, 1 - 1e-6)
            if weight > 0:
                (weight * (m * (1 - m)).mean()).backward()
        background = None
        if self.background is not None and c.loss.background_weight > 0:
            background = self.background.loss(leaves['envmap'], ENV_TO_WORLD, c.loss.background_samples, c.seed + step)
            (c.loss.background_weight * background).backward()

        # clip per-vertex gradients, then pull them back through each parameterization
        for name, x in decoded.items():
            if name == 'geometry' and self.fixed_geometry:
                continue
            g = leaves[name].grad
            if g is None or not torch.isfinite(g).all():
                raise FloatingPointError(f'{name}: missing or nonfinite gradient at step {step}')
            if name == 'geometry':
                g = clip_vertex_norms(g, c.geometry.clip_quantile, nonzero_only=False)
            elif name in MATERIALS:
                g = clip_vertex_norms(g, c.material.clip_quantile, nonzero_only=True)
            x.backward(g)
        if not self.fixed_geometry:
            self.geometry_optimizer.step()
        self.material_optimizer.step()
        if c.envmap.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in self.envmap.parameters() if p.grad is not None], c.envmap.grad_clip)
        self.envmap_optimizer.step()
        torch.cuda.synchronize()
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise FloatingPointError(f'nonfinite parameters after step {step}')
        self.step = step
        return dict(step=step, views=indices, loss=float(np.mean(losses)), background=None if background is None else float(background.detach()),
                    seconds=time.perf_counter() - start, **extra)

    # ----------------------------------------------------------------------------------------------- dual hypothesis
    def _dual_step(self, leaves, indices, step):
        """Renders of one dual-hypothesis step. Per view, A renders each vertex's selected hypothesis (gradients to
        geometry, envmap and the selected fields; torch.where routes each vertex's gradient to its hypothesis), and on
        `renders_both` steps B renders the unselected ones with geometry and envmap detached. On those steps the
        per-pixel sRGB residuals of A and B are pooled to the vertices of the primary hits and the selection is updated."""
        c, d = self.config, self.config.dual
        self.residual *= d.residual_decay
        self.coverage *= d.residual_decay
        both = renders_both(d, step)
        losses, fractions, unselected = [], [], []
        k, n = len(indices), len(self.residual)
        for j, index in enumerate(indices):
            batch = self.views.batch(index)
            fractions.append(float(batch['mask'].mean()))
            seed = c.seed + step * 1000 + j
            metal = self.selection > .5
            hypotheses = lambda: (leaves['albedo'].sigmoid(), .09 + .9 * leaves['roughness'].sigmoid(),
                                  leaves['albedo_m'].sigmoid(), .09 + .9 * leaves['roughness_m'].sigmoid())
            residual_of = lambda image: (linear_to_srgb(image.detach()) - batch['rgb']).abs().mean(-1).reshape(-1)
            v, e = leaves['geometry'], leaves['envmap'].exp()
            aD, rD, aM, rM = hypotheses()
            image = self.renderer.render(v, torch.where(metal, aM, aD), torch.where(metal, rM, rD), e, self.views, index,
                                         spp=c.render.spp, seed=seed, metallic=metal.float())
            loss = photometric_loss(image, batch['rgb'], c.loss.ssim_weight)
            (loss / k).backward()
            losses.append(float(loss.detach()))
            if not both:
                continue
            selected = residual_of(image)
            aD, rD, aM, rM = hypotheses()   # fresh activations: A's graph is freed
            image = self.renderer.render(v.detach(), torch.where(metal, aD, aM), torch.where(metal, rD, rM), e.detach(),
                                         self.views, index, spp=c.render.spp, seed=seed, metallic=(~metal).float())
            loss = photometric_loss(image, batch['rgb'], c.loss.ssim_weight)
            (loss / k).backward()
            unselected.append(float(loss.detach()))
            other = residual_of(image)

            valid, tri, bary = self.renderer.pixel_hits(self.views, index)
            weight = (bary * (valid & (batch['mask'].reshape(-1) > .5))[:, None].float()).reshape(-1)
            ids = tri.reshape(-1)
            pool = lambda r_pix: torch.zeros(n, device='cuda').index_add_(0, ids, weight * r_pix.repeat_interleave(3))
            cov = torch.zeros(n, device='cuda').index_add_(0, ids, weight)
            seen, w = cov > 0, metal[:, 0]
            for r_pix, as_metal in ((pool(selected), w), (pool(other), ~w)):   # column 1: metal, column 0: dielectric
                self.residual[:, 1] += torch.where(as_metal, r_pix, 0.)
                self.residual[:, 0] += torch.where(as_metal, 0., r_pix)
                self.coverage[:, 1] += torch.where(as_metal, cov, 0.)
                self.coverage[:, 0] += torch.where(as_metal, 0., cov)
                self.observed[:, 1] |= seen & as_metal
                self.observed[:, 0] |= seen & ~as_metal
        if both:
            with torch.no_grad():
                self.selection = select(self.residual, self.coverage, self.observed, temperature(d, step, c.steps),
                                        d.margin * ramp(step, d.margin_from, d.margin_until), self.lattice,
                                        None if self.lattice is None else self.lattice.solver, self.slice_weight)
        return losses, fractions, dict(both_rendered=both, loss_unselected=float(np.mean(unselected)) if unselected else None,
                                       metal_fraction=float((self.selection > .5).float().mean()))

    @torch.no_grad()
    def dual_export(self):
        """The selected (binary) material and both hypotheses with the selection evidence."""
        state = self.decoded()
        aD, rD = state['albedo'].sigmoid(), .09 + .9 * state['roughness'].sigmoid()
        aM, rM = state['albedo_m'].sigmoid(), .09 + .9 * state['roughness_m'].sigmoid()
        m = (self.selection > .5).float()
        return dict(albedo=(1 - m) * aD + m * aM, roughness=(1 - m) * rD + m * rM, metallic=m,
                    albedo_dielectric=aD, roughness_dielectric=rD, albedo_metal=aM, roughness_metal=rM,
                    selection=self.selection, residual=self.residual / self.coverage.clamp_min(1e-30), coverage=self.coverage,
                    observed=self.observed)

    def parameters(self):
        fields = [f for f in (self.albedo, self.roughness, self.metallic, self.albedo_m, self.roughness_m) if f is not None]
        return [self.geometry_u] + [f.u for f in fields] + list(self.envmap.parameters())

    # ----------------------------------------------------------------------------------------------- loop and outputs
    def train(self):
        c = self.config
        keep = set(c.checkpoint_steps)
        total = 0.
        with (self.out / 'train_log.jsonl').open('a') as f:
            for step in range(self.step + 1, c.steps + 1):
                row = self.train_step(step)
                total += row['seconds']
                f.write(json.dumps(row) + '\n')
                f.flush()
                if step == 1 or step % 50 == 0:
                    metal = f'  metal {row["metal_fraction"]:.3f}' if self.dual else ''
                    self.log(f'step {step}/{c.steps}  loss {row["loss"]:.5f}{metal}  {total:.1f} s')
                if step in keep:
                    self.save_checkpoint(self.out / f'checkpoint_{step:04d}.pt')
        self.save_checkpoint(self.out / 'checkpoint.pt')
        self.export(self.out / 'result')
        return total

    @torch.no_grad()
    def export(self, out_dir):
        """Decoded result for evaluation: result.npz (vertices, faces, linear albedo, perceptual roughness, metallic,
        linear envmap in the renderer's lat-long layout) and envmap.exr."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        v, a, r, metallic, e = self.physical(self.decoded())
        if metallic is None:
            metallic = torch.zeros_like(r)
        if self.dual:   # the selected binary material; both hypotheses go to dual.npz
            dual = self.dual_export()
            a, r, metallic = dual['albedo'], dual['roughness'], dual['metallic']
            np.savez(out / 'dual.npz', **{k: x.detach().cpu().numpy() for k, x in dual.items()})
        arrays = dict(vertices=v, faces=self.renderer.faces, albedo=a, roughness=r, metallic=metallic, envmap=e)
        np.savez(out / 'result.npz', step=self.step, **{k: x.detach().cpu().numpy() for k, x in arrays.items()})
        write_exr(out / 'envmap.exr', e.cpu().numpy())

    def save_checkpoint(self, path):
        envmap_kind = 'sg' if isinstance(self.envmap, SGEnvmap) else 'texture'
        fields = {k: getattr(self, k).u.detach().cpu() for k in MATERIALS if getattr(self, k) is not None}
        state = dict(step=self.step, config=as_plain(self.config), geometry_u=self.geometry_u.detach().cpu(), materials=fields,
                     envmap_kind=envmap_kind, envmap={k: x.detach().cpu() for k, x in self.envmap.state_dict().items()},
                     dual={k: getattr(self, k).cpu() for k in ('residual', 'coverage', 'observed', 'selection')} if self.dual else None,
                     optimizers=dict(geometry=self.geometry_optimizer.state_dict(), material=self.material_optimizer.state_dict(),
                                     envmap=self.envmap_optimizer.state_dict()))
        tmp = Path(path).with_suffix('.tmp')
        torch.save(state, tmp)
        tmp.replace(path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location='cuda', weights_only=False)
        with torch.no_grad():
            self.geometry_u.copy_(state['geometry_u'])
            for k, u in state['materials'].items():
                getattr(self, k).u.copy_(u)
            if self.dual:
                for k, x in state['dual'].items():
                    setattr(self, k, x.cuda())
        if state['envmap_kind'] == 'texture':
            self.envmap = TextureEnvmap(state['envmap']['reference'], self.config.envmap.large_steps)
            self.envmap_optimizer = torch.optim.Adam(self.envmap.parameters(), lr=self.config.envmap.lr, betas=(.9, .99))
        self.envmap.load_state_dict(state['envmap'])
        self.geometry_optimizer.load_state_dict(state['optimizers']['geometry'])
        self.material_optimizer.load_state_dict(state['optimizers']['material'])
        self.envmap_optimizer.load_state_dict(state['optimizers']['envmap'])
        self.step = state['step']
