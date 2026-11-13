"""Differentiable rendering of the mesh with per-vertex materials and an environment map (Mitsuba 3 / Dr.Jit).

Scene model: one triangle mesh with Mitsuba's principled BSDF (base colour, perceptual roughness, metallic,
specular .5), each attribute a per-vertex field interpolated with the hit's barycentrics, angle-weighted vertex
normals, and a 64x128 lat-long environment emitter hidden from camera rays. Rendering uses the projective path
tracer `prb_projective` (max depth 4) with the estimator changes of `prep_ir.mitsuba_patches`.

Environment convention: Mitsuba lat-long layout (rows include the poles), emitter-to-world `ENV_TO_WORLD`, which
maps emitter (x, y, z) to world (z, x, y), i.e. the world z axis is up.
"""
import mitsuba as mi
import drjit as dr
import numpy as np
import torch

mi.set_variant('cuda_ad_rgb')

from . import mitsuba_patches  # noqa: E402
from .mesh import angle_weighted_normals  # noqa: E402

ENV_TO_WORLD = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], np.float32)


class VertexTexture(mi.Texture):
    """Per-vertex values interpolated with the primitive barycentrics of the hit (the mesh has no UVs)."""

    def __init__(self, values, faces, channels):
        super().__init__(mi.Properties())
        self.channels = channels
        self.values = mi.Float(np.asarray(values, np.float32).ravel())
        self.faces = dr.unravel(mi.Vector3u, mi.UInt32(np.asarray(faces, np.uint32).ravel()))

    def _value(self, si, active=True):
        indices = dr.gather(mi.Vector3u, self.faces, si.prim_index, active)
        u, v = si.uv.x, si.uv.y
        out = []
        for channel in range(self.channels):
            a, b, c = [dr.gather(mi.Float, self.values, indices[i] * self.channels + channel, active) for i in range(3)]
            out.append(dr.fma(u, b - a, dr.fma(v, c - a, a)))
        return out

    def eval(self, si, active=True):
        return self.eval_3(si, active)

    def eval_1(self, si, active=True):
        value = self._value(si, active)
        return value[0] if self.channels == 1 else .212671 * value[0] + .715160 * value[1] + .072169 * value[2]

    def eval_3(self, si, active=True):
        value = self._value(si, active)
        return mi.Color3f(value[0]) if self.channels == 1 else mi.Color3f(*value)

    def mean(self):
        return dr.mean(self.values)

    def is_spatially_varying(self):
        return True

    def traverse(self, callback):
        callback.put('values', self.values, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return f'VertexTexture[channels={self.channels}]'


def camera_sensor(views, index):
    """Pinhole sensor of a NeRF-convention camera (camera-to-world, -z forward, y up)."""
    transform = views.transforms[index] @ np.diag([-1, 1, -1, 1]).astype(np.float32)
    w, h = views.size
    return mi.load_dict({
        'type': 'perspective', 'fov': views.fov, 'fov_axis': 'x', 'to_world': mi.ScalarTransform4f(transform),
        'near_clip': 1e-4, 'far_clip': 1e5,
        'film': {'type': 'hdrfilm', 'width': w, 'height': h, 'pixel_format': 'rgba', 'rfilter': {'type': 'box'}, 'sample_border': True},
        'sampler': {'type': 'independent', 'sample_count': 32}})


class Renderer:
    def __init__(self, vertices, faces, metallic=None, envmap_shape=(64, 128), max_depth=4, derivative_spp=None, cos_floor=1e-6):
        """vertices (N, 3), faces (F, 3); metallic (N, 1) fixed values (None: zero) unless passed to render();
        derivative_spp: (interior, primary boundary, indirect boundary) samples per pixel of the derivative
        estimator, None for Mitsuba's defaults."""
        mitsuba_patches.apply(cos_floor)
        v = np.asarray(vertices.cpu() if torch.is_tensor(vertices) else vertices, np.float32)
        f = np.asarray(faces.cpu() if torch.is_tensor(faces) else faces, np.int64)
        self.faces = torch.as_tensor(f, device='cuda', dtype=torch.long)
        m = np.zeros((len(v), 1), np.float32) if metallic is None else np.asarray(metallic.cpu() if torch.is_tensor(metallic) else metallic, np.float32)
        props = mi.Properties()
        props['bsdf'] = mi.load_dict({'type': 'principled', 'specular': .5,
                                      # placeholders (every render loads the current values); the initial material
                                      'base_color': VertexTexture(np.full((len(v), 3), .3, np.float32), f, 3),
                                      'roughness': VertexTexture(np.full((len(v), 1), .6, np.float32), f, 1),
                                      'metallic': VertexTexture(m, f, 1)})
        mesh = mi.Mesh('object', len(v), len(f), props, has_vertex_normals=True)
        params = mi.traverse(mesh)
        params['vertex_positions'] = v.ravel()
        params['faces'] = f.astype(np.uint32).ravel()
        params.update()
        self.scene = mi.load_dict({
            'type': 'scene', 'object': mesh,
            'environment': {'type': 'envmap', 'bitmap': mi.Bitmap(np.full((*envmap_shape, 3), .5, np.float32)),
                            'to_world': mi.ScalarTransform4f(ENV_TO_WORLD)},
            'integrator': {'type': 'prb_projective', 'max_depth': max_depth, 'guiding': 'none', 'hide_emitters': True}})
        self.derivative_spp = derivative_spp
        if derivative_spp is not None:
            for key, value in zip(('sppc', 'sppp', 'sppi'), derivative_spp):
                setattr(self.scene.integrator(), key, int(value))
        self.params = mi.traverse(self.scene)
        self.keys = dict(albedo='object.bsdf.base_color.values', roughness='object.bsdf.roughness.values',
                         metallic='object.bsdf.metallic.values')
        self.params.keep(['object.vertex_positions', 'object.vertex_normals', *self.keys.values(), 'environment.data'])
        if tuple(self.params['environment.data'].shape) != (envmap_shape[0], envmap_shape[1] + 2, 3):
            raise RuntimeError('unexpected envmap texture layout (Mitsuba 3.9.1 stores one wrap column per side)')
        self.sensors = {}
        self._pending = False

        def load(v, n, a, r, e):
            self.params['object.vertex_positions'] = dr.ravel(v)
            self.params['object.vertex_normals'] = dr.ravel(n)
            self.params[self.keys['albedo']] = dr.ravel(a)
            self.params[self.keys['roughness']] = dr.ravel(r)
            self.params['environment.data'] = e

        @dr.wrap(source='torch', target='drjit')
        def render(v, n, a, r, e, sensor, spp, seed):
            load(v, n, a, r, e)
            self.params.update()
            return mi.render(self.scene, self.params, sensor=sensor, spp=spp, seed=seed, seed_grad=seed + 100000)

        @dr.wrap(source='torch', target='drjit')
        def render_metallic(v, n, a, r, m, e, sensor, spp, seed):
            load(v, n, a, r, e)
            self.params[self.keys['metallic']] = dr.ravel(m)
            self.params.update()
            return mi.render(self.scene, self.params, sensor=sensor, spp=spp, seed=seed, seed_grad=seed + 100000)
        self._render, self._render_metallic = render, render_metallic

    def sensor(self, views, index):
        key = (str(views.root), views.split, views.size, index)
        if key not in self.sensors:
            self.sensors[key] = camera_sensor(views, index)
        return self.sensors[key]

    def render(self, vertices, albedo, roughness, envmap, views, index, *, spp, seed, metallic=None):
        """Linear RGB image (H, W, 3) of view `index`, composited onto the views' background colour.

        albedo (N, 3) linear base colour, roughness (N, 1) perceptual, envmap (H, W, 3) linear radiance, metallic
        (N, 1) or None (the fixed values given at construction). With gradients enabled and derivative sample
        counts configured, `spp` is the primal sample count and the derivative counts apply to the adjoint pass.
        """
        if self._pending:
            raise RuntimeError('backpropagate the previous render before rendering again')
        n = angle_weighted_normals(vertices, self.faces)
        e = torch.cat((envmap[:, -1:], envmap, envmap[:, :1]), dim=1)   # wrap columns; their gradients flow back to the texels
        sensor = self.sensor(views, index)
        inputs = (vertices, albedo, roughness, envmap) + ((metallic,) if metallic is not None else ())
        if self.derivative_spp is not None and torch.is_grad_enabled() and any(x.requires_grad for x in inputs):
            sensor.sampler().set_sample_count(spp)
            spp = 0   # runtime spp 0: the integrator's own derivative counts are used for the adjoint pass
        if metallic is None:
            rgba = self._render(vertices, n, albedo, roughness, e, sensor, spp, seed)
        else:
            rgba = self._render_metallic(vertices, n, albedo, roughness, metallic, e, sensor, spp, seed)
        rgb = rgba[..., :3] + views.background * (1 - rgba[..., 3:4])
        if rgb.requires_grad:   # the Mitsuba scene is mutable: guard against a second render before backward
            self._pending = True

            def release(g):
                self._pending = False
                return g
            rgb.register_hook(release)
        return rgb

    @torch.no_grad()
    def aovs(self, vertices, albedo, roughness, envmap, views, index, *, spp=16, seed=1):
        """Pixel-averaged albedo, roughness, world normal and coverage from jittered primary rays (evaluation only)."""
        self.render(vertices, albedo, roughness, envmap, views, index, spp=1, seed=seed)   # load the state into the scene
        sensor = self.sensor(views, index)
        w, h = views.size
        gen = torch.Generator(device='cuda').manual_seed(seed)
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        base = torch.stack((xx, yy), -1).reshape(-1, 2)
        accum = torch.zeros((h * w, 8), device='cuda')
        for _ in range(spp):
            uv = (base + torch.rand((h * w, 2), device='cuda', generator=gen)) / vertices.new_tensor([w, h])
            ray, _ = sensor.sample_ray(0., 0., mi.Point2f(mi.Float(uv[:, 0]), mi.Float(uv[:, 1])), mi.Point2f(.5))
            si = self.scene.ray_intersect(ray)
            valid = si.is_valid().torch().bool()
            idx = torch.where(valid, si.prim_index.torch().long(), 0)
            hit_uv = si.uv.torch().T
            weight = torch.stack((1 - hit_uv[:, 0] - hit_uv[:, 1], hit_uv[:, 0], hit_uv[:, 1]), 1)
            weight = torch.where(valid[:, None], weight, 0.)
            tri = self.faces[idx]
            a = (albedo[tri] * weight[..., None]).sum(1)
            r = (roughness[tri] * weight[..., None]).sum(1)
            normal = torch.where(valid[:, None], si.sh_frame.n.torch().T, 0.)
            accum += torch.cat((a, r, normal, valid[:, None]), 1)
        accum = (accum / spp).reshape(h, w, 8)
        return {'albedo': accum[..., :3], 'roughness': accum[..., 3:4], 'normal': accum[..., 4:7], 'opacity': accum[..., 7:8]}

    def first_hits(self, origins, directions):
        """(valid, t) of rays against the mesh at its current vertex positions."""
        hit = self.scene.ray_intersect(mi.Ray3f(mi.Point3f(origins), mi.Vector3f(directions)))
        return hit.is_valid().torch(), hit.t.torch()

    @torch.no_grad()
    def pixel_hits(self, views, index):
        """Primary hits of the pixel-centre rays of view `index` against the mesh as loaded by the last render():
        (valid (H*W,), triangle vertex ids (H*W, 3), barycentric weights (H*W, 3)) in row-major pixel order."""
        sensor = self.sensor(views, index)
        w, h = views.size
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        uv = (torch.stack((xx, yy), -1).reshape(-1, 2) + .5) / torch.tensor([w, h], device='cuda', dtype=torch.float32)
        ray, _ = sensor.sample_ray(0., 0., mi.Point2f(mi.Float(uv[:, 0]), mi.Float(uv[:, 1])), mi.Point2f(.5))
        si = self.scene.ray_intersect(ray)
        valid = si.is_valid().torch().bool()
        idx = torch.where(valid, si.prim_index.torch().long(), 0)
        hit_uv = si.uv.torch().T
        weight = torch.stack((1 - hit_uv[:, 0] - hit_uv[:, 1], hit_uv[:, 0], hit_uv[:, 1]), 1)
        return valid, self.faces[idx], torch.where(valid[:, None], weight, 0.)
