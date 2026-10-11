import math
import unittest
from pipeline import bounds_errors, harness_specs, pose_motion, scene_plan


class PipelineTests(unittest.TestCase):
    def test_metal_envelopes_do_not_overlap(self):
        plan = scene_plan("metal", 50, 73)
        positions = [p["position_m"] for p in plan["placements"]]
        for i, first in enumerate(positions):
            for second in positions[i+1:]:
                self.assertGreater(math.dist(first, second), 2 * 0.043)
        for pose in plan["placements"]:
            self.assertAlmostEqual(sum(x*x for x in pose["quaternion_wxyz"]), 1.0)

    def test_harness_envelopes_do_not_overlap(self):
        poses = scene_plan("harness", 50, 73)["placements"]
        for i, first in enumerate(poses):
            for second in poses[i+1:]:
                delta = [abs(a-b) for a,b in zip(first["position_m"], second["position_m"])]
                self.assertTrue(any(d > size for d,size in zip(delta, (0.32,0.28,0.04))))

    def test_motion_quaternion_sign_invariance(self):
        old = {"a": ((0,0,0), (1,0,0,0))}
        new = {"a": ((0.1,0,0), (-1,0,0,0))}
        self.assertEqual(pose_motion(old, new, 0.1), (1.0,0.0))

    def test_reject_bad_states(self):
        self.assertTrue(bounds_errors({"bad": ((0,0,-.01),(1,0,0,0))},(.6,.5,.2)))
        self.assertTrue(bounds_errors({"bad": ((math.nan,0,0),(1,0,0,0))},(.6,.5,.2)))

    def test_specs_repeat_and_fit(self):
        a = harness_specs(50, 7)
        self.assertEqual(a, harness_specs(50, 7))
        self.assertEqual(len({s["object_id"] for s in a}),50)
        for s in a:
            self.assertLessEqual(s["segments"]*s["segment_length_m"],.216)
            self.assertGreaterEqual(s["static_friction"],s["dynamic_friction"])


if __name__ == "__main__":
    unittest.main()
