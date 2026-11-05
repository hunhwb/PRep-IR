"""Bake a result onto the surface reconstruction's UV atlas: the textured-asset layout that the Stanford-ORB and DTC
evaluation scripts of DiffReg-PBIR render (mesh.obj, base_color.exr, roughness.exr, metallic.exr, envmap.exr).

The reconstruction's mesh.obj carries the Blender UV atlas made during surface reconstruction. Scene preparation
welded its UV-seam duplicates and dropped zero-area faces and components with at most 4 faces (prep_ir.mesh.load_reconstruction
without normal repair); the same steps are repeated here to map every .obj vertex to its optimized position and
material, which are then rasterized into the atlas.

Rasterizes with pytorch3d; it does not import Mitsuba.
usage: python -m prep_ir.evaluation.bake RESULT_DIR MESH_OBJ OUT_DIR
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np  # noqa: E402
import torch  # noqa: E402


def read_obj(path):
    v, vt, f, ft = [], [], [], []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            v.append([float(x) for x in p[1:4]])
        elif p[0] == 'vt':
            vt.append([float(x) for x in p[1:3]])
        elif p[0] == 'f':
            idx = [q.split('/') for q in p[1:4]]
            f.append([int(q[0]) - 1 for q in idx])
            ft.append([int(q[1]) - 1 if len(q) > 1 and q[1] else -1 for q in idx])
    return np.array(v, np.float64), np.array(vt, np.float64), np.array(f, np.int64), np.array(ft, np.int64)


def write_obj(path, v, vt, f, ft):
    with open(path, 'w') as s:
        for x in v:
            s.write(f'v {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n')
        for x in vt:
            s.write(f'vt {x[0]:.6f} {x[1]:.6f}\n')
        for a, b in zip(f, ft):
            s.write(f'f {a[0] + 1}/{b[0] + 1} {a[1] + 1}/{b[1] + 1} {a[2] + 1}/{b[2] + 1}\n')


def rasterize(vt, ft, values, resolution, device='cuda'):
    """Interpolate per-corner values over the UV atlas; empty texels take their nearest covered texel."""
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.renderer.mesh.rasterizer import Fragments
    from pytorch3d.structures import Meshes
    from scipy import ndimage
    uv = torch.as_tensor(vt[ft.reshape(-1)], dtype=torch.float32, device=device)
    val = torch.as_tensor(values.reshape(-1, values.shape[-1]), dtype=torch.float32, device=device)
    faces = torch.arange(uv.shape[0], device=device).reshape(-1, 3)
    ndc = torch.cat((uv * 2 - 1, torch.ones_like(uv[:, :1])), -1)
    ndc[:, 0] *= -1
    mesh = Meshes(verts=[ndc], faces=[faces])
    pix, zbuf, bary, dists = rasterize_meshes(mesh, image_size=(resolution, resolution), faces_per_pixel=1, perspective_correct=True)
    mesh.textures = TexturesVertex(verts_features=val[None])
    img = mesh.sample_textures(Fragments(pix, zbuf, bary, dists))[0, :, :, 0, :]
    mask = (pix[0, :, :, 0] >= 0).cpu().numpy()
    if not mask.any():
        raise RuntimeError('empty UV atlas')
    near = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    flat = torch.as_tensor(near[0] * resolution + near[1], dtype=torch.long, device=device).reshape(-1)
    return img.reshape(-1, img.shape[-1])[flat].reshape(resolution, resolution, -1).cpu().numpy()


def kept_vertex_map(mesh_path, vo):
    """For each .obj vertex, the index of its welded, kept vertex in the scene (-1 if dropped)."""
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.load(mesh_path, process=True, merge_tex=True, merge_norm=True)
    vm = np.asarray(m.vertices, np.float64)
    dist, idx = cKDTree(vm).query(vo, k=1)
    if dist.max() >= 1e-9:
        raise RuntimeError(f'.obj vertex missing from the welded mesh ({dist.max():.2e})')
    fm = np.asarray(m.faces, np.int64)
    area = np.linalg.norm(np.cross(vm[fm[:, 1]] - vm[fm[:, 0]], vm[fm[:, 2]] - vm[fm[:, 0]]), axis=1)
    fm = fm[area > 0]
    labels = trimesh.graph.connected_component_labels(trimesh.Trimesh(vm, fm, process=False).face_adjacency, node_count=len(fm))
    fm = fm[~(np.bincount(labels) <= 4)[labels]]
    used = np.unique(fm)
    remap = np.full(len(vm), -1)
    remap[used] = np.arange(len(used))
    return remap[idx], len(used)


def envmap_to_orb(env, height=512):
    """Renderer lat-long envmap (pole-inclusive rows, world z up) -> the Neural-PBIR / Stanford-ORB envmap.exr layout:
    polar about world +y, indexed by d = (-cos t sin p, cos p, -sin t sin p), t in [-pi/2, 3pi/2], p in [0, pi],
    rolled by W/2. (Resampled with texel-centred rows, as for the paper's numbers.)"""
    import cv2
    h, w = env.shape[:2]
    W = 2 * height
    th, ph = np.meshgrid(np.linspace(-.5 * np.pi, 1.5 * np.pi, W), np.linspace(0, np.pi, height))
    dx, dy, dz = -np.cos(th) * np.sin(ph), np.cos(ph), -np.sin(th) * np.sin(ph)
    u = (.5 - np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(dz, -1, 1)) / np.pi
    wrapped = np.concatenate([env[:, -1:], env, env[:, :1]], 1)
    out = cv2.remap(wrapped, (u * w + .5).astype(np.float32), (v * h - .5).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.roll(out, W // 2, axis=1)


def bake(result_dir, mesh_path, out_dir, resolution=1024):
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = np.load(Path(result_dir) / 'result.npz')
    vo, vt, f, ft = read_obj(mesh_path)
    keep, kept = kept_vertex_map(mesh_path, vo)
    if kept != len(r['vertices']):
        raise RuntimeError(f'{kept} kept reconstruction vertices, {len(r["vertices"])} in the result')
    live = keep >= 0
    faces_live = live[f].all(1)
    v = vo.copy()
    v[live] = r['vertices'][keep[live]]
    f2, ft2 = f[faces_live], ft[faces_live]
    write_obj(out / 'mesh.obj', v, vt, f2, ft2)
    corner = lambda x: x[keep[f2.reshape(-1)].clip(0)].reshape(-1, 3, x.shape[-1])
    for name in ('base_color', 'roughness', 'metallic'):
        values = r['albedo' if name == 'base_color' else name]
        tex = rasterize(vt, ft2, corner(values), resolution)
        if tex.shape[-1] == 1:
            tex = np.repeat(tex, 3, -1)
        cv2.imwrite(str(out / f'{name}.exr'), tex[..., ::-1].astype(np.float32))
    cv2.imwrite(str(out / 'envmap.exr'), envmap_to_orb(r['envmap'].astype(np.float32))[..., ::-1].astype(np.float32))


if __name__ == '__main__':
    bake(*sys.argv[1:4])
