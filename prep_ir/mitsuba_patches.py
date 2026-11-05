"""Derivative-estimator changes to Mitsuba 3.9.1's projective path tracer (`prb_projective`).

The forward (primal) images are unchanged; only the gradients differ from the stock integrator:

1. Symmetric visible-side clamp. `sample_radiance_difference` clamps the shading normal of the foreground
   boundary hit to the visible side of the boundary segment, but not the background hit. With interpolated
   normals this reports radiance jumps at creases where the image has none. The same clamp is applied to both.
2. Material-path parameterization. The primary path vertex keeps Mitsuba's camera-ray parameterization; every
   later vertex follows the surface (`RayFlags.FollowShape`) and its contribution carries the relative area
   stretch of its face. The indirect-visibility boundary term then moves both segment endpoints.
3. BSDF angle guard. At grazing angles the BSDF derivative is evaluated with local cosines floored at
   `cos_floor` (sign kept); the forward value is untouched (`dr.replace_grad`).

Mitsuba re-reads the source of `@dr.syntax` functions, so each patched method is compiled from a real file
written to a temporary directory. The patterns are tied to Mitsuba 3.9.1 and fail loudly on any other version.
The patched code is derived from Mitsuba 3 (https://github.com/mitsuba-renderer/mitsuba3, BSD-3-Clause).
"""
import inspect
import re
import tempfile
import textwrap
import typing
from pathlib import Path

import drjit as dr
import mitsuba as mi

MITSUBA_VERSION = '3.9.1'
_applied = False


def _patch_method(cls, name, replacements):
    """Replace each `old` snippet (whitespace-insensitive, must be unique) of cls.name by `new` and recompile."""
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    for old, new in replacements:
        pattern = r'\s+'.join(re.escape(token) for token in old.split())
        hits = list(re.finditer(pattern, src))
        if len(hits) != 1:
            raise RuntimeError(f'pattern found {len(hits)} times in {cls.__name__}.{name}: {old[:60]}')
        m = hits[0]
        indent = src[src.rfind('\n', 0, m.start()) + 1:m.start()]
        if indent.strip():
            indent = ''
        src = src[:m.start()] + ('\n' + indent).join(new.split('\n')) + src[m.end():]
    path = Path(tempfile.mkdtemp(prefix='prep_ir_mitsuba_')) / f'{cls.__name__}_{name}.py'
    path.write_text(src)
    namespace = dict(vars(inspect.getmodule(cls)))
    for t in ('Optional', 'Any', 'Tuple', 'List', 'Union'):
        namespace.setdefault(t, getattr(typing, t))
    exec(compile(src, str(path), 'exec'), namespace)
    setattr(cls, name, namespace[name])


def _clamp_to_visible_side(si, d, active):
    """The foreground surgery of stock `sample_radiance_difference`, as a function."""
    alpha = dr.dot(si.sh_frame.n, d)
    eps = dr.epsilon(alpha) * 100
    wrong_side = active & (alpha > -eps)
    new_n = dr.normalize(si.sh_frame.n - (alpha + eps) * d)
    si.sh_frame[wrong_side] = mi.Frame3f(new_n)
    si.wi[wrong_side] = si.to_local(-d)
    return si


def sample_radiance_difference(self, scene, ss, curr_depth, sampler, wavelengths, active):
    """Stock method with the visible-side clamp also applied to the background hit (patch 1)."""
    ray_bg = ss.spawn_ray(wavelengths)
    si_bg = scene.ray_intersect(ray_bg, ray_flags=mi.RayFlags.All, coherent=False, active=active)
    hit = active & si_bg.is_valid()
    si_bg = _clamp_to_visible_side(si_bg, ss.d, hit)
    radiance_bg_hit, _, _, _ = self.sample(
        dr.ADMode.Primal, scene, sampler, ray_bg, curr_depth, None, None, hit, False, si_bg)
    radiance_bg_miss, _, _, _ = self.sample(
        dr.ADMode.Primal, scene, sampler, ray_bg, curr_depth, None, None, active & ~hit, False, None)
    radiance_bg = dr.select(hit, radiance_bg_hit, radiance_bg_miss)

    pi_fg = dr.zeros(mi.PreliminaryIntersection3f)
    pi_fg.valid = active
    pi_fg.t = 1
    pi_fg.prim_index = ss.prim_index
    pi_fg.prim_uv = ss.uv
    pi_fg.shape = ss.shape
    dummy_ray = mi.Ray3f(ss.p - ss.d, ss.d)
    ray_bg.wavelengths = wavelengths
    si_fg = pi_fg.compute_surface_interaction(dummy_ray, mi.RayFlags.All, active)
    si_fg = _clamp_to_visible_side(si_fg, ss.d, active)
    radiance_fg, _, _, _ = self.sample(
        dr.ADMode.Primal, scene, sampler, ray_bg, curr_depth, None, None, active, False, si_fg)

    radiance_diff = radiance_fg - radiance_bg
    active_diff = active & (dr.max(dr.abs(radiance_diff)) > 0)
    return radiance_diff, active_diff


def sample_area_background(self, scene, ss, curr_depth, sampler, wavelengths, active):
    """Background radiance behind a boundary segment for the material-domain occlusion term (patch 2)."""
    ray = ss.spawn_ray(wavelengths)
    si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=False, active=active)
    self._area_background_si = dr.detach(si)
    value, _, _, _ = self.sample(dr.ADMode.Primal, scene, sampler, ray, curr_depth, None, None, active, False, None)
    return -value, active & (dr.max(dr.abs(value)) > 0)


# Patch 2, in `PathProjectiveIntegrator.sample`: later vertices follow the surface, area stretch in the continuous term.
_MATERIAL_PATH_SAMPLE = [
    ('si = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All, active=active_next & ~use_si_shade)',
     'si_ray = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All, active=active_next & ~use_si_shade)\n'
     'si_material = pi.compute_surface_interaction(ray, ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape, active=active_next & ~use_si_shade)\n'
     'si = dr.select(depth == 0, si_ray, si_material)'),
    ('si_prev_diff = pi_prev.compute_surface_interaction( ray_prev, ray_flags=mi.RayFlags.Minimal, active=active_next & ~use_si_shade )',
     'si_prev_ray = pi_prev.compute_surface_interaction(ray_prev, ray_flags=mi.RayFlags.Minimal, active=active_next & ~use_si_shade)\n'
     'si_prev_material = pi_prev.compute_surface_interaction(ray_prev, ray_flags=mi.RayFlags.Minimal | mi.RayFlags.FollowShape, active=active_next & ~use_si_shade)\n'
     'si_prev_diff = dr.select(depth == 1, si_prev_ray, si_prev_material)'),
    ('si_detached = dr.detach(si) # Ignore motion of current point',
     'si_detached = si  # both attached material endpoints and shading frame'),
    ('si_next = pi_next.compute_surface_interaction(ray_next, ray_flags=mi.RayFlags.Minimal, active=active_next)',
     'with dr.resume_grad():\n'
     '    si_next = pi_next.compute_surface_interaction(ray_next, ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape, active=active_next)'),
    ('Lr_ind = L * dr.relative_grad(bsdf_val) * dr.relative_grad(J)',
     'area = dr.norm(dr.cross(si_next.dp_du, si_next.dp_dv))\n'
     'area = dr.select(si_next.is_valid() & (dr.detach(area) > 0), area, 1.)\n'
     'Lr_ind = L * dr.relative_grad(bsdf_val) * dr.relative_grad(J) * dr.relative_grad(area)'),
]

# Patch 2, in `PathProjectiveIntegrator.sample_importance`: remember the receiver of the boundary segment.
_MATERIAL_PATH_IMPORTANCE = [
    ('return W, sensor_ds.uv, depth_cam, active_found',
     'self._area_receiver_si = dr.detach(si_boundary)\n'
     'self._area_receiver_wo = dr.detach(first_wo)\n'
     'self._area_receiver_primary = depth_cam == 2\n'
     'self._area_receiver_active = active_found\n'
     'return W, sensor_ds.uv, depth_cam, active_found'),
]

# Patch 2, in `ProjectiveDetail.eval_indirect_integrand`: boundary motion relative to both segment endpoints.
_MATERIAL_PATH_INDIRECT = [
    ('fE, active_e = parent.sample_radiance_difference(',
     'fE, active_e = parent.sample_area_background('),
    ('motion = dr.dot(dr.detach(ss.n), x_b_follow)',
     'receiver = parent._area_receiver_si\n'
     'background = parent._area_background_si\n'
     'ok = active & parent._area_receiver_active & receiver.is_valid()\n'
     'dp_receiver = receiver.shape.differential_motion(receiver, ok)\n'
     'v_receiver = dp_receiver - dr.detach(dp_receiver)\n'
     'omega = -parent._area_receiver_wo\n'
     'ng = dr.detach(receiver.n); cos = dr.dot(omega, ng)\n'
     'primary = ok & parent._area_receiver_primary\n'
     'v_primary = omega * dr.dot(v_receiver, ng) / dr.select(primary, cos, 1.)\n'
     'v_receiver = dr.select(primary, v_primary, v_receiver)\n'
     'has_bg = ok & background.is_valid()\n'
     'dp_bg = background.shape.differential_motion(background, has_bg)\n'
     'v_bg = dr.select(has_bg, dp_bg - dr.detach(dp_bg), mi.Vector3f(0.))\n'
     'distance_ab = dr.norm(dr.detach(ss.p - receiver.p))\n'
     'distance_ac = dr.norm(dr.detach(background.p - receiver.p))\n'
     'fraction = dr.select(has_bg, distance_ab / dr.select(has_bg, distance_ac, 1.), 0.)\n'
     'v_boundary = x_b_follow - dr.detach(x_b_follow)\n'
     'motion = dr.dot(dr.detach(ss.n), v_boundary - (1. - fraction)*v_receiver - fraction*v_bg)\n'
     'motion = dr.replace_grad(mi.Float(1.), motion)  # splat activity must not erase a zero-primal AD surrogate'),
]


def _bsdf_angle_guard(cos_floor):
    """Patch 3: replacements in `PathProjectiveIntegrator.sample` for the floored-cosine BSDF derivative."""
    t = repr(float(cos_floor))

    def floor(c):
        return f'dr.select(dr.abs({c}) < {t}, dr.select({c} >= 0, {t}, -{t}), {c})'

    safe = ('si_safe = mi.SurfaceInteraction3f(si)\n'
            f'si_safe.wi = mi.Vector3f(si.wi.x, si.wi.y, {floor("si.wi.z")})\n'
            f'wo_safe = mi.Vector3f(wo.x, wo.y, {floor("wo.z")})')
    direct = 'bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)'
    indirect = 'bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)'
    return [
        (direct, direct + '\n'
                 "if dr.hint(not primal, mode='scalar'):\n"
                 '    ' + safe.replace('\n', '\n    ') + '\n'
                 '    bsdf_value_safe, bsdf_pdf_safe = bsdf.eval_pdf(bsdf_ctx, si_safe, wo_safe, active_em)\n'
                 '    bsdf_value_em = dr.replace_grad(bsdf_value_em, bsdf_value_safe)\n'
                 '    bsdf_pdf_em = dr.replace_grad(bsdf_pdf_em, bsdf_pdf_safe)'),
        (indirect, indirect + '\n'
                   "if dr.hint(not primal, mode='scalar'):\n"
                   '    ' + safe.replace('\n', '\n    ') + '\n'
                   '    bsdf_val = dr.replace_grad(bsdf_val, bsdf.eval(bsdf_ctx, si_safe, wo_safe, active_next))'),
    ]


def apply(cos_floor=1e-6):
    """Install patches 1-3 once per process (idempotent)."""
    global _applied
    if _applied:
        return
    if mi.__version__ != MITSUBA_VERSION:
        raise RuntimeError(f'the estimator patches target Mitsuba {MITSUBA_VERSION}, found {mi.__version__}')
    if not 0 < float(cos_floor) <= 1e-3:
        raise ValueError('BSDF cosine floor must lie in (0, 1e-3]')
    from mitsuba.ad.integrators.prb_projective import PathProjectiveIntegrator as Integrator
    from mitsuba.ad.projective import ProjectiveDetail
    Integrator.sample_radiance_difference = sample_radiance_difference
    Integrator.sample_area_background = sample_area_background
    _patch_method(Integrator, 'sample', _MATERIAL_PATH_SAMPLE + _bsdf_angle_guard(cos_floor))
    _patch_method(Integrator, 'sample_importance', _MATERIAL_PATH_IMPORTANCE)
    _patch_method(ProjectiveDetail, 'eval_indirect_integrand', _MATERIAL_PATH_INDIRECT)
    _applied = True
