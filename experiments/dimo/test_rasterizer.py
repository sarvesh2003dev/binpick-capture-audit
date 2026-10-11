import tempfile
import unittest
from pathlib import Path
import numpy as np
from rasterizer import read_ply, render_scene


def pose(z, x=0, obj_id=1):
    return {"obj_id": obj_id, "cam_R_m2c": np.eye(3).ravel().tolist(),
            "cam_t_m2c": [x, 0, z]}


class RasterizerTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.models = {1: (self.vertices, self.faces)}
        self.K = np.array([[20, 0, 16], [0, 20, 16], [0, 0, 1]])

    def test_near_object_occludes_far(self):
        depth, ids = render_scene(self.models, [pose(10), pose(5)], self.K, 32, 32)
        self.assertEqual(ids[16, 16], 2)
        self.assertEqual(depth[16, 16], 5)
        self.assertEqual(depth[0, 0], 0)

    def test_order_does_not_change_depth_or_foreground(self):
        a, ia = render_scene(self.models, [pose(10), pose(5)], self.K, 32, 32)
        b, ib = render_scene(self.models, [pose(5), pose(10)], self.K, 32, 32)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ia > 0, ib > 0)
        self.assertEqual(ib[16, 16], 1)

    def test_camera_translation_moves_projection(self):
        _, first = render_scene(self.models, [pose(5)], self.K, 32, 32)
        _, shifted = render_scene(self.models, [pose(5, x=1)], self.K, 32, 32)
        self.assertAlmostEqual(np.where(shifted)[1].mean() - np.where(first)[1].mean(), 4)

    def test_back_faces_are_rendered(self):
        _, first = render_scene(self.models, [pose(5)], self.K, 32, 32)
        reversed_models = {1: (self.vertices, self.faces[:, ::-1])}
        _, second = render_scene(reversed_models, [pose(5)], self.K, 32, 32)
        np.testing.assert_array_equal(first, second)

    def test_reciprocal_depth_interpolation(self):
        # Projected corners (0,0), (4,0), (0,4); sample(.5,.5) has
        # barycentric weights(.75,.125,.125), giving 1/(.75/2+.125/4+.125/8).
        vertices = np.array([[0, 0, 2], [16, 0, 4], [0, 32, 8]])
        depth, ids = render_scene({1: (vertices, [[0, 1, 2]])}, [pose(0)], np.eye(3), 4, 4)
        self.assertEqual(ids[0, 0], 1)
        self.assertAlmostEqual(float(depth[0, 0]), 1 / (.75 / 2 + .125 / 4 + .125 / 8), places=6)

    def test_behind_camera_and_degenerate_faces(self):
        depth, ids = render_scene(self.models, [pose(-5)], self.K, 32, 32)
        self.assertFalse(ids.any())
        self.assertFalse(depth.any())
        _, ids = render_scene({1: (self.vertices, [[0, 0, 1]])}, [pose(5)], self.K, 32, 32)
        self.assertFalse(ids.any())

    def test_ascii_ply_extra_properties_and_quad(self):
        text = """ply
format ascii 1.0
comment fixture
element vertex 4
property uchar red
property float z
property float x
property float y
element face 1
property list uchar int vertex_indices
end_header
255 0 -1 -1
255 0 1 -1
255 0 1 1
255 0 -1 1
4 0 1 2 3
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            path.write_text(text, encoding="ascii")
            vertices, faces = read_ply(path)
        np.testing.assert_array_equal(vertices, self.vertices)
        np.testing.assert_array_equal(faces, self.faces)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            render_scene(self.models, [], self.K, 0, 32)


if __name__ == "__main__":
    unittest.main()
