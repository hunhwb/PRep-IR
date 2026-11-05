"""Triangle-mesh helpers: loading the reconstructed mesh, vertex normals, edges and areas."""
import numpy as np
import torch


def angle_weighted_normals(vertices, faces):
    """Angle-weighted vertex normals in float64 (atan2 instead of acos for the corner angles). Part of the
    differentiable forward model; degenerate triangles or vertices are errors, never silently repaired."""
    v = vertices.double()
    tri = v[faces]
    cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = torch.linalg.vector_norm(cross, dim=1, keepdim=True)
    if (norm == 0).any():
        raise ValueError('zero-area triangle')
    unit = cross / norm
    normal = torch.zeros_like(v)
    for i in range(3):
        a = tri[:, (i + 1) % 3] - tri[:, i]
        b = tri[:, (i + 2) % 3] - tri[:, i]
        normal = normal.index_add(0, faces[:, i], unit * torch.atan2(norm, (a * b).sum(1, keepdim=True)))
    norm = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    if (norm == 0).any():
        raise ValueError('undefined vertex normal')
    return (normal / norm).to(vertices.dtype)


def unique_edges(faces):
    """Undirected edges (E, 2), each once with the smaller index first."""
    f = faces.long()
    return torch.unique(torch.sort(torch.cat((f[:, :2], f[:, 1:], f[:, [2, 0]])), dim=1).values, dim=0)


def vertex_areas(vertices, faces):
    """A third of the incident triangle areas per vertex (float64)."""
    v, f = vertices.double(), faces.long()
    area = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]).norm(dim=1) / 2
    return torch.zeros(len(v), dtype=torch.float64, device=v.device).index_add_(0, f.reshape(-1), area.repeat_interleave(3) / 3)


def _compact(v, f):
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def load_reconstruction(path, to_world=np.eye(3), repair_normals=True):
    """The surface-reconstruction mesh as (vertices float32, faces int64), cleaned for rendering.

    UV-seam duplicates are welded (trimesh process=True), the vertices are mapped to the dataset's world frame
    (`v @ to_world`), and zero-area faces and components with at most 4 faces are dropped. Faces at vertices
    without a defined angle-weighted normal are removed repeatedly when `repair_normals`; otherwise such vertices
    are an error (Stanford-ORB / DTC evaluation maps the vertices back onto the reconstruction's UV atlas, which
    assumes only the two drops above).
    """
    import trimesh
    m = trimesh.load(path, process=True, merge_tex=True, merge_norm=True)
    v = (np.asarray(m.vertices, dtype=np.float64) @ np.asarray(to_world, np.float64)).astype(np.float32)
    f = np.asarray(m.faces, dtype=np.int64)
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    f = f[area > 0]
    labels = trimesh.graph.connected_component_labels(trimesh.Trimesh(v, f, process=False).face_adjacency, node_count=len(f))
    tiny = np.bincount(labels) <= 4
    v, f = _compact(v, f[~tiny[labels]])
    report = dict(dropped_zero_area_faces=int((area == 0).sum()), dropped_small_components=int(tiny.sum()), normal_repair_faces=0)
    while True:
        normals = _vertex_normal_norms(v, f)
        bad = np.flatnonzero(normals == 0)
        if not len(bad):
            break
        if not repair_normals:
            raise ValueError(f'{len(bad)} vertices without a defined normal')
        incident = np.isin(f, bad).any(1)
        report['normal_repair_faces'] += int(incident.sum())
        v, f = _compact(v, f[~incident])
    report.update(vertices=len(v), faces=len(f))
    return v, f, report


def _vertex_normal_norms(v, f):
    """Length of the unnormalized angle-weighted normal sum (zero where angle_weighted_normals would fail)."""
    V, Fc = torch.from_numpy(v).double(), torch.from_numpy(f)
    tri = V[Fc]
    cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = torch.linalg.vector_norm(cross, dim=1, keepdim=True)
    unit = cross / norm
    normal = torch.zeros_like(V)
    for i in range(3):
        a = tri[:, (i + 1) % 3] - tri[:, i]
        b = tri[:, (i + 2) % 3] - tri[:, i]
        normal = normal.index_add(0, Fc[:, i], unit * torch.atan2(norm, (a * b).sum(1, keepdim=True)))
    return torch.linalg.vector_norm(normal, dim=1).numpy()
