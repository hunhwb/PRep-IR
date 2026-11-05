"""Scene preparation and UV export must retain exactly the same mesh vertices."""
import numpy as np
import trimesh

from prep_ir.mesh import load_reconstruction
from prep_ir.evaluation.bake import kept_vertex_map


def test_four_face_component_is_dropped_by_preparation_and_bake(tmp_path):
    cube = trimesh.creation.box()
    v = np.array([[3, 0, 0], [4, 0, 0], [3, 1, 0], [3, 0, 1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh = trimesh.util.concatenate([cube, trimesh.Trimesh(v, f, process=False)])
    path = tmp_path / 'mesh.obj'
    mesh.export(path)
    vertices, faces, _ = load_reconstruction(path, np.eye(3), repair_normals=False)
    remap, kept = kept_vertex_map(path, np.asarray(mesh.vertices))
    assert kept == len(vertices) == 8
    assert len(faces) == 12
    assert np.all(remap[-4:] == -1)
