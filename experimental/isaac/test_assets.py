"""Geometry/material integrity checks using only the Python standard library."""

from collections import Counter
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import assets


def signed_volume(geometry):
    points = geometry["points"]
    total = 0.0
    for ia, ib, ic in geometry["faces"]:
        a, b, c = points[ia], points[ib], points[ic]
        total += (a[0] * (b[1] * c[2] - b[2] * c[1])
                  + a[1] * (b[2] * c[0] - b[0] * c[2])
                  + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6
    return total


def edge_counts(geometry):
    return Counter(tuple(sorted((face[j], face[(j + 1) % 3])))
                   for face in geometry["faces"] for j in range(3))


class AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = assets.generate_specs()

    def test_determinism_diversity_and_ids(self):
        self.assertEqual(self.specs, assets.generate_specs(50, 7))
        self.assertNotEqual(self.specs, assets.generate_specs(50, 8))
        self.assertEqual(len(self.specs), 50)
        self.assertEqual(len({s["object_id"] for s in self.specs}), 50)
        unique_shapes = {(s["outer_radius_m"], s["inner_radius_m"], s["height_m"])
                         for s in self.specs}
        self.assertEqual(len(unique_shapes), 50)
        self.assertEqual({s["family"] for s in self.specs},
                         {"bushing", "flanged_bushing", "gear"})

    def test_dimension_material_and_mass_bounds(self):
        for spec in self.specs:
            with self.subTest(part=spec["object_id"]):
                self.assertTrue(0.016 <= spec["outer_radius_m"] <= 0.028)
                self.assertTrue(0.005 <= spec["inner_radius_m"] <= 0.010)
                self.assertTrue(0.014 <= spec["height_m"] <= 0.028)
                self.assertIn(spec["density_kg_m3"], (2700, 7850))
                self.assertTrue(0.95 <= spec["mass_scale"] <= 1.05)
                self.assertGreaterEqual(spec["static_friction"], spec["dynamic_friction"])
                self.assertGreaterEqual(spec["dynamic_friction"], 0)
                for key in ("restitution", "metallic", "roughness"):
                    self.assertTrue(0 <= spec[key] <= 1)
                ideal_mass = spec["volume_m3"] * spec["density_kg_m3"]
                self.assertTrue(ideal_mass * .95 <= spec["mass_kg"] <= ideal_mass * 1.05)
                self.assertAlmostEqual(spec["mass_kg"], ideal_mass * spec["mass_scale"])

    def test_visual_watertight_outward_and_exact_volume(self):
        for spec in self.specs:
            mesh = assets.build_geometry(spec)
            with self.subTest(part=spec["object_id"]):
                self.assertEqual(set(edge_counts(mesh).values()), {2})
                self.assertGreater(signed_volume(mesh), 0)
                self.assertTrue(math.isclose(signed_volume(mesh), mesh["volume_m3"],
                                             rel_tol=1e-11))
                self.assertEqual(mesh["volume_m3"], spec["volume_m3"])
                self.assertEqual(len(mesh["uv"]), 3 * len(mesh["faces"]))
                self.assertTrue(all(0 <= u <= 1 and 0 <= v <= 1 for u, v in mesh["uv"]))
                # Every directed edge has exactly the reversed edge, certifying
                # consistent local orientation beyond just positive total volume.
                directed = Counter((f[i], f[(i + 1) % 3])
                                   for f in mesh["faces"] for i in range(3))
                self.assertTrue(all(n == directed[b, a] for (a, b), n in directed.items()))

    def test_colliders_convex_closed_disjoint_volume_and_open_bore(self):
        for spec in self.specs:
            mesh = assets.build_geometry(spec)
            total_volume = 0
            for wedge in mesh["colliders"]:
                self.assertEqual(set(edge_counts(wedge).values()), {2})
                vol = signed_volume(wedge)
                self.assertGreater(vol, 0)
                total_volume += vol
                points = wedge["points"]
                for ia, ib, ic in wedge["faces"]:
                    a, b, c = points[ia], points[ib], points[ic]
                    ab = [b[i] - a[i] for i in range(3)]
                    ac = [c[i] - a[i] for i in range(3)]
                    normal = (ab[1]*ac[2]-ab[2]*ac[1],
                              ab[2]*ac[0]-ab[0]*ac[2],
                              ab[0]*ac[1]-ab[1]*ac[0])
                    self.assertGreater(sum(x*x for x in normal), 1e-24)
                    # Convexity: all points behind each outward supporting plane.
                    self.assertTrue(all(sum(normal[j]*(p[j]-a[j]) for j in range(3))
                                        <= 1e-14 for p in points))
                n = (4 * spec["teeth"] if spec["family"] == "gear" else 64)
                # The polygonal bore's inscribed radius remains empty. Each
                # wedge's inner chord is a supporting plane separating it from 0.
                a, d = points[0], points[3]
                mid = [(a[j] + d[j]) / 2 for j in range(2)]
                dist = math.hypot(*mid)
                bore = spec["inner_radius_m"] * math.cos(math.pi / n)
                self.assertAlmostEqual(dist, bore, places=12)
                self.assertTrue(all(sum(p[j] * mid[j] for j in range(2))
                                    >= bore * bore - 1e-14 for p in points))
            self.assertTrue(math.isclose(total_volume, mesh["volume_m3"], rel_tol=1e-11))
            # Each layer tiles non-overlapping angular sectors. For each sector,
            # all vertices stay between its two rays and outside the bore.
            n = (4 * spec["teeth"] if spec["family"] == "gear" else 64)
            for index, wedge in enumerate(mesh["colliders"]):
                angle = 2 * math.pi * (index % n) / n
                for p in wedge["points"]:
                    relative = (math.atan2(p[1], p[0]) - angle) % (2 * math.pi)
                    self.assertTrue(relative <= 2 * math.pi / n + 1e-10
                                    or relative >= 2 * math.pi - 1e-10)

    def test_png_crc_dimensions_pixels_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.png" for i in range(3)]
            for i, seed in enumerate((8, 8, 9)):
                assets.write_roughness_png(paths[i], seed, .3, size=64)
            data = paths[0].read_bytes()
            self.assertEqual(data, paths[1].read_bytes())
            self.assertNotEqual(data, paths[2].read_bytes())
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            pos, chunks, compressed = 8, [], b""
            while pos < len(data):
                length = struct.unpack(">I", data[pos:pos+4])[0]
                kind = data[pos+4:pos+8]
                payload = data[pos+8:pos+8+length]
                crc = struct.unpack(">I", data[pos+8+length:pos+12+length])[0]
                self.assertEqual(crc, zlib.crc32(kind + payload) & 0xFFFFFFFF)
                chunks.append(kind)
                if kind == b"IHDR":
                    self.assertEqual(struct.unpack(">IIBBBBB", payload), (64,64,8,0,0,0,0))
                elif kind == b"IDAT":
                    compressed += payload
                pos += length + 12
            self.assertEqual(chunks, [b"IHDR", b"IDAT", b"IEND"])
            raw = zlib.decompress(compressed)
            self.assertEqual(len(raw), 64 * 65)
            self.assertTrue(all(raw[y * 65] == 0 for y in range(64)))
            pixels = [v for y in range(64) for v in raw[y*65+1:(y+1)*65]]
            self.assertGreater(len(set(pixels)), 20)
            self.assertLess(abs(sum(pixels) / len(pixels) / 255 - .3), .03)

    def test_obj_and_cli_export(self):
        with tempfile.TemporaryDirectory() as directory:
            assets.main(["--output", directory, "--count", "3", "--seed", "7"])
            folder = Path(directory)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(len(manifest["specs"]), 3)
            for spec in manifest["specs"]:
                obj = folder / (spec["object_id"] + ".obj")
                lines = obj.read_text().splitlines()
                mesh = assets.build_geometry(spec)
                self.assertEqual(sum(s.startswith("v ") for s in lines), len(mesh["points"]))
                self.assertEqual(sum(s.startswith("vt ") for s in lines), len(mesh["uv"]))
                self.assertEqual(sum(s.startswith("f ") for s in lines), len(mesh["faces"]))
                self.assertTrue((folder / (spec["object_id"] + "_roughness.png")).exists())
                saved = json.loads((folder / (spec["object_id"] + ".json")).read_text())
                self.assertEqual(saved["spec"], spec)

    def test_invalid_inputs(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                assets.generate_specs(value)
        self.assertEqual(assets.generate_specs(0), [])
        bad = dict(self.specs[0], inner_radius_m=.1)
        with self.assertRaises(ValueError):
            assets.build_geometry(bad)


if __name__ == "__main__":
    unittest.main()
