"""Unit tests 1-4: EXAMPLE_EVALUATE1 (cosmic shear) on the frozen test data.

1. chi2 at the example fiducial point within 0.2 of the frozen reference.
2. No race condition (OMP_NUM_THREADS=2): the fiducial evaluated as the 10th
   of 10 cosmologies in a row matches a fresh single evaluation.
3. Same as 1 with the TATT IA model (IA_model: 1) and
   LSST_A2_1=0.05, LSST_BTA_1=0.05, LSST_A2_2=-1.51541.
4. Same as 2 with the TATT IA model.

Run from the Cocoa/ folder with the cocoa environment active and
start_cocoa.sh sourced:  python -m pytest ./projects/lsst_y1/tests
"""

import os

# must precede any cobaya/cosmolike import (OpenMP reads it at library load)
os.environ["OMP_NUM_THREADS"] = "2"

import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

EXAMPLE = "example1"


class TestExample1CosmicShear(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_1_chi2_matches_frozen_reference(self):
        chi2 = u.single_model_chi2(EXAMPLE, tatt=False)
        ref = self.reference[f"{EXAMPLE}_nla"]
        u.report_chi2_test(
            1, "example1 (cosmic shear, NLA) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_2_no_race_condition_ten_in_a_row(self):
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=False)
        u.report_race_test(
            2, "example1 (cosmic shear, NLA) race check: 10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")

    def test_3_chi2_matches_frozen_reference_tatt(self):
        chi2 = u.single_model_chi2(EXAMPLE, tatt=True)
        ref = self.reference[f"{EXAMPLE}_tatt"]
        u.report_chi2_test(
            3, "example1 (cosmic shear, TATT) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"TATT chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_4_no_race_condition_ten_in_a_row_tatt(self):
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=True)
        u.report_race_test(
            4, "example1 (cosmic shear, TATT) race check: 10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"TATT 10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
