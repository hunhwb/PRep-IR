"""Joint optimization of geometry, materials and illumination with a purely photometric objective (Sec. 4).

One step: render `views_per_step` training views from the current state, back-propagate the photometric loss (plus
the background term when the frames observe the environment) to the decoded per-vertex positions, material logits and
log envmap, clip per-vertex gradient norms, pull the gradients back through each block's parameterization, and update
all blocks from the same renderings:

    geometry   v = v0 + diag * (I + 19 L_mesh)^-1 u                       UniformAdam
    materials  Q = Q0 + S^T (I + 20 L_lattice / mean degree)^-1 W  (+ R)   Adam
    envmap     SG lobes for 250 steps, then log E0 + (I + 6 K / mean degree)^-1 w   Adam, warm-up + cosine
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
from .envmap import BackgroundObservations, SGEnvmap, TextureEnvmap
from .image import write_exr
from .largesteps import SPDSolver, UniformAdam, inverse, laplacian, system
from .losses import clip_vertex_norms, photometric_loss
from .material import LatticeMaterial, PriorLattice, VertexMaterial, material_logits
from .mesh import unique_edges
from .prior_loss import PriorBilateralLoss
from .render import ENV_TO_WORLD, Renderer

MATERIALS = ('albedo', 'roughness', 'metallic')


def decay_factor(start, step, steps):
    """Exponential decay from `start` at step 1 to 1 at the last step."""
    return start * (1 / start) ** ((step - 1) / max(1, steps - 1))


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
        self.renderer = Renderer(v0, faces, envmap_shape=(c.envmap.height, c.envmap.width), max_depth=r.max_depth,
                                 derivative_spp=(r.spp_interior, r.spp_primary_edge, r.spp_indirect_edge), cos_floor=r.bsdf_cos_floor)

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
        return self.v0 + inverse(self.geometry_u, self.geometry_solver) * self.diagonal

    def decoded(self):
        """Differentiable decoded state: world vertices, material logits and log envmap."""
        state = dict(geometry=self.vertices(), albedo=self.albedo(), roughness=self.roughness(), envmap=self.envmap())
        if self.metallic is not None:
            state['metallic'] = self.metallic()
        return state

    @staticmethod
    def physical(state):
        """Renderer inputs: vertices, linear albedo, perceptual roughness in [.09, .99], metallic or None, envmap."""
        metallic = state['metallic'].sigmoid() if 'metallic' in state else None
        return state['geometry'], state['albedo'].sigmoid(), .09 + .9 * state['roughness'].sigmoid(), metallic, state['envmap'].exp()

    def _set_learning_rates(self, step):
        c = self.config
        d = lambda name: decay_factor(float(c.lr_decay[name]), step, c.steps)
        for g in self.material_optimizer.param_groups:
            base = c.material.lr_albedo if g['name'] == 'albedo' else c.material.lr_roughness
            g['lr'] = base * d('albedo' if g['name'] == 'albedo' else 'roughness')
        self.geometry_optimizer.param_groups[0]['lr'] = c.geometry.lr * d('geometry')
        e = c.envmap
        progress = min(1., max(0, step - e.warmup) / max(1, c.steps - e.warmup))
        cosine = e.min_lr_factor + (1 - e.min_lr_factor) * .5 * (1 + math.cos(math.pi * progress))
        for g in self.envmap_optimizer.param_groups:
            g['lr'] = e.lr * min(1., step / max(1, e.warmup)) * cosine * d('envmap')

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
        leaves = {k: x.detach().requires_grad_(True) for k, x in decoded.items()}
        losses, fractions = [], []
        for j, index in enumerate(indices):
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
        background = None
        if self.background is not None and c.loss.background_weight > 0:
            background = self.background.loss(leaves['envmap'], ENV_TO_WORLD, c.loss.background_samples, c.seed + step)
            (c.loss.background_weight * background).backward()

        # clip per-vertex gradients, then pull them back through each parameterization
        for name, x in decoded.items():
            g = leaves[name].grad
            if g is None or not torch.isfinite(g).all():
                raise FloatingPointError(f'{name}: missing or nonfinite gradient at step {step}')
            if name == 'geometry':
                g = clip_vertex_norms(g, c.geometry.clip_quantile, nonzero_only=False)
            elif name in MATERIALS:
                g = clip_vertex_norms(g, c.material.clip_quantile, nonzero_only=True)
            x.backward(g)
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
                    seconds=time.perf_counter() - start)

    def parameters(self):
        fields = [self.albedo, self.roughness] + ([self.metallic] if self.metallic is not None else [])
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
                    self.log(f'step {step}/{c.steps}  loss {row["loss"]:.5f}  {total:.1f} s')
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
        arrays = dict(vertices=v, faces=self.renderer.faces, albedo=a, roughness=r, metallic=metallic, envmap=e)
        np.savez(out / 'result.npz', step=self.step, **{k: x.detach().cpu().numpy() for k, x in arrays.items()})
        write_exr(out / 'envmap.exr', e.cpu().numpy())

    def save_checkpoint(self, path):
        envmap_kind = 'sg' if isinstance(self.envmap, SGEnvmap) else 'texture'
        fields = {k: getattr(self, k).u.detach().cpu() for k in MATERIALS if getattr(self, k) is not None}
        state = dict(step=self.step, config=as_plain(self.config), geometry_u=self.geometry_u.detach().cpu(), materials=fields,
                     envmap_kind=envmap_kind, envmap={k: x.detach().cpu() for k, x in self.envmap.state_dict().items()},
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
        if state['envmap_kind'] == 'texture':
            self.envmap = TextureEnvmap(state['envmap']['reference'], self.config.envmap.large_steps)
            self.envmap_optimizer = torch.optim.Adam(self.envmap.parameters(), lr=self.config.envmap.lr, betas=(.9, .99))
        self.envmap.load_state_dict(state['envmap'])
        self.geometry_optimizer.load_state_dict(state['optimizers']['geometry'])
        self.material_optimizer.load_state_dict(state['optimizers']['material'])
        self.envmap_optimizer.load_state_dict(state['optimizers']['envmap'])
        self.step = state['step']
