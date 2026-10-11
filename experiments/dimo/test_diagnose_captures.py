import unittest
import numpy as np
from diagnose_captures import summarize_depth, cutoff_hypothesis


class DiagnosisTests(unittest.TestCase):
    def test_separates_background_and_objects(self):
        ids=np.zeros((5,5),dtype=int);ids[1:4,1:4]=1
        raw=np.ones((5,5),dtype=np.uint16);raw[1:4,1:4]=0
        stats,_=summarize_depth(raw,ids,np.full((5,5),350.),erosion_px=0)
        self.assertEqual(stats['interior_zero_pixels'],9)
        self.assertEqual(stats['outside_zero_pixels'],0)

    def test_all_zero_is_not_evidence_of_cutoff_boundary(self):
        frames=[{'positive_raw_min':None,'cad_interior_z_max_mm':400}]*13
        self.assertEqual(cutoff_hypothesis(frames,.25)['status'],'unassessable_no_positive_depth')

    def test_boundary_heuristic_requires_parts_below_boundary(self):
        frames=[{'positive_raw_min':1960,'cad_interior_z_max_mm':400}]*12
        self.assertEqual(cutoff_hypothesis(frames,.25)['status'],'suspected_range_cutoff')
        # A raw minimum alone is insufficient: explicit scale changes the conclusion.
        self.assertEqual(cutoff_hypothesis(frames,.1)['status'],'not_flagged')

    def test_mixed_minima_do_not_trigger(self):
        frames=[{'positive_raw_min':n,'cad_interior_z_max_mm':300} for n in range(1960,1973)]
        self.assertEqual(cutoff_hypothesis(frames,.25)['status'],'not_flagged')


if __name__=='__main__':unittest.main()
