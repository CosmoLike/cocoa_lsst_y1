"""Unit tests 5-8: EXAMPLE_EVALUATE2 (3x2pt) on the frozen test data.

5. chi2 at the example fiducial point within 0.2 of the frozen reference.
6. No race condition (OMP_NUM_THREADS=2): the fiducial evaluated as the 10th
   of 10 cosmologies in a row matches a fresh single evaluation.
7. Same as 5 with the TATT IA model (IA_model: 1) and
   LSST_A2_1=0.05, LSST_BTA_1=0.05, LSST_A2_2=-1.51541.
8. Same as 6 with the TATT IA model.

Run from the Cocoa/ folder with the cocoa environment active and
start_cocoa.sh sourced:  python -m pytest ./projects/lsst_y1/tests -v
"""

import os

# must precede any cobaya/cosmolike import (OpenMP reads it at library load)
os.environ["OMP_NUM_THREADS"] = "2"

import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

EXAMPLE = "example2"


class TestExample2ThreeXTwo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_5_chi2_matches_frozen_reference(self):
        chi2 = u.single_model_chi2(EXAMPLE, tatt=False)
        ref = self.reference[f"{EXAMPLE}_nla"]
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_6_no_race_condition_ten_in_a_row(self):
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=False)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")

    def test_7_chi2_matches_frozen_reference_tatt(self):
        chi2 = u.single_model_chi2(EXAMPLE, tatt=True)
        ref = self.reference[f"{EXAMPLE}_tatt"]
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"TATT chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_8_no_race_condition_ten_in_a_row_tatt(self):
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=True)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"TATT 10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
