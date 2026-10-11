import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from customer_audit.core import AuditError, run_audit, write_report
from customer_audit.demo import create_demo


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class CustomerAuditTests(unittest.TestCase):
    def manifest(self, root: Path, frames: list[dict], *, registration="unknown",
                 invalid_values=None, scale=None) -> Path:
        value = {"schema_version": 1,
                 "camera": {"name": "unit-test", "invalid_values":
                            [0] if invalid_values is None else invalid_values,
                            "depth_scale_mm": scale, "registration_status": registration},
                 "frames": frames}
        path = root / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_demo_runs_and_pools_pixel_weighted_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = create_demo(root / "demo")
            result = run_audit(manifest)
            invalid = sum(f["whole_image"]["invalid_pixels"] for f in result["frames"])
            pixels = sum(f["whole_image"]["pixels"] for f in result["frames"])
            self.assertEqual(result["pooled_whole_image"]["pixels"], pixels)
            self.assertEqual(result["pooled_whole_image"]["invalid_pixels"], invalid)
            self.assertAlmostEqual(result["pooled_whole_image"]["invalid_fraction"], invalid / pixels)
            self.assertIsNotNone(result["pooled_roi"])
            self.assertAlmostEqual(
                result["frames"][0]["whole_image"]["positive_quantiles_mm_01_50_99"][1],
                result["frames"][0]["whole_image"]["positive_raw_quantiles_01_50_99"][1] * 0.25)
            write_report(result, root / "report")
            self.assertTrue((root / "report" / "report.md").is_file())
            self.assertTrue((root / "report" / "metrics.json").is_file())

    def test_unknown_scale_withholds_metric_claim_but_keeps_raw_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_png(root / "depth.png", np.array([[0, 100], [200, 300]], dtype=np.uint16))
            manifest = self.manifest(root, [{"id": "one", "depth": "depth.png"}])
            result = run_audit(manifest)
            self.assertEqual(result["pooled_whole_image"]["invalid_fraction"], 0.25)
            self.assertEqual(result["camera"]["depth_scale_mm"], None)
            self.assertEqual(result["frames"][0]["whole_image"]["positive_raw_quantiles_01_50_99"][1], 200)
            self.assertNotIn("positive_quantiles_mm_01_50_99", result["frames"][0]["whole_image"])

    def test_metric_scale_requires_positive_number(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_png(root / "depth.png", np.array([[20, 30]], dtype=np.uint16))
            manifest = self.manifest(root, [{"id": "one", "depth": "depth.png"}], scale=0)
            with self.assertRaises(AuditError):
                run_audit(manifest)

    def test_rgb_roi_requires_registration_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_png(root / "depth.png", np.full((4, 4), 200, dtype=np.uint16))
            save_png(root / "rgb.png", np.zeros((4, 4, 3), dtype=np.uint8))
            save_png(root / "mask.png", np.full((4, 4), 255, dtype=np.uint8))
            frame = {"id": "one", "depth": "depth.png", "rgb": "rgb.png",
                     "mask": "mask.png", "mask_alignment": "rgb"}
            manifest = self.manifest(root, [frame])
            with self.assertRaisesRegex(AuditError, "registered_to_rgb"):
                run_audit(manifest)

    def test_registered_rgb_mask_requires_matching_grids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_png(root / "depth.png", np.full((4, 4), 200, dtype=np.uint16))
            save_png(root / "rgb.png", np.zeros((8, 8, 3), dtype=np.uint8))
            save_png(root / "mask.png", np.full((8, 8), 255, dtype=np.uint8))
            frame = {"id": "one", "depth": "depth.png", "rgb": "rgb.png",
                     "mask": "mask.png", "mask_alignment": "rgb"}
            manifest = self.manifest(root, [frame], registration="registered_to_rgb")
            with self.assertRaisesRegex(AuditError, "matching RGB/depth grids"):
                run_audit(manifest)

    def test_manifest_rejects_parent_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); outside = root / "outside.png"
            save_png(outside, np.full((2, 2), 100, dtype=np.uint16))
            private = root / "private"; private.mkdir()
            manifest = self.manifest(private, [{"id": "one", "depth": "../outside.png"}])
            with self.assertRaisesRegex(AuditError, "inside the manifest directory"):
                run_audit(manifest)

    def test_manifest_requires_explicit_invalid_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_png(root / "depth.png", np.full((2, 2), 100, dtype=np.uint16))
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1,
                                        "camera": {"name": "camera"},
                                        "frames": [{"id": "one", "depth": "depth.png"}]}))
            with self.assertRaisesRegex(AuditError, "invalid_values"):
                run_audit(path)

    def test_output_must_be_new_to_prevent_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_png(root / "depth.png", np.array([[0, 1]], dtype=np.uint16))
            result = run_audit(self.manifest(root, [{"id": "one", "depth": "depth.png"}]))
            output = root / "report"; output.mkdir(); (output / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(AuditError, "already exists"):
                write_report(result, output)
            self.assertEqual((output / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
