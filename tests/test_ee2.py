"""Unit tests 18-19: the Cocoa modifications of EuclidEmulator2.

Cocoa pins a modified EuclidEmulator2 (the EE2_GIT_COMMIT of
set_installation_options.sh): OpenMP threading, a 1,010-redshift
capacity, the get_boost2 API with a pre-built emulator, memory-leak
fixes, and a bilinear interpolation with a border fix (the
repository's README documents them). These tests pin the
modifications two ways:

 18. the installed, modified EE2 against the PRE-modification code
     (commit ff59f66), built side by side into a temporary prefix
     from the local clone (offline; --ignore-installed protects
     .local). Both builds evaluate the cosmic-shear NLA data vector
     with non_linear_emul: 1 at the ten shared cosmologies
     (NONLINEAR_COMPARISON_POINTS), and the per-cosmology
     delta^T C^-1 delta of the original against the modified build
     must stay below CHI2_TOLERANCE (0.2). The original cannot run
     inside Cocoa as-is - it lacks get_boost2 and overflows beyond
     101 redshifts - so its worker carries the compatibility patch
     (EE2_ORIGINAL_SHIM: a get_boost2 adapter plus 100-redshift
     chunking) that leaves its numerics untouched.
 19. the race check with EE2 on: the fiducial evaluated fresh and
     again as the 10th of 10 cosmologies on one model instance,
     with the nonlinear P(k) from EE2 (non_linear_emul: 1). EE2's
     compute is OpenMP-threaded, so leaked state or a thread race
     inside it shifts the second fiducial value; the two must agree
     within RACE_TOLERANCE (1e-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests/test_ee2.py

Test 18 compiles the original EE2 once per run (about half a
minute); it skips, rather than fails, when the local euclidemu2
clone does not carry the original commit.
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
# insert(0, ...) puts the folder FIRST in the search order, ahead of
# every other place a same-named module could hide.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


class TestEE2Modifications(unittest.TestCase):
    """Tests 18-19, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR and
    verifies every frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: test 18 compares the two EE2
    builds against each other and test 19 compares one build
    against itself, so the frozen state only supplies the
    configuration and the data files.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x18_ee2_modifications_vs_original(self):
        """The modified EE2 reproduces the original's data vectors.

        The per-cosmology delta chi2 of the pre-modification build
        against the installed one, at the ten shared cosmologies,
        must stay below the house comfort band.
        """
        try:
            dchi2s = u.ee2_original_vs_cocoa_dchi2s("example1")
        except RuntimeError as error:
            # a shallow local clone cannot supply the original
            # commit; that is an installation limitation, not an
            # EE2 regression
            self.skipTest(str(error))
        print("\n==== test 18: modified EE2 vs original EE2 "
              "(example1, NLA, 10 cosmologies) ====", flush=True)
        for i, d in enumerate(dchi2s, start=1):
            print(f"  cosmology {i:2d}: dchi2 = {d:.8f}", flush=True)
        largest = max(dchi2s)
        print(f"  max dchi2 = {largest:.8f}   (limit: < "
              f"{u.CHI2_TOLERANCE})", flush=True)
        self.assertLess(
            largest, u.CHI2_TOLERANCE,
            msg=f"max dchi2 of the original-vs-modified EE2 data "
                f"vectors = {largest:.8f} over the ten cosmologies "
                f"(limit {u.CHI2_TOLERANCE})")

    def test_x19_ee2_race_ten_in_a_row(self):
        """The fiducial with EE2 as 10th of 10 matches a fresh run.

        EE2's OpenMP-threaded compute runs inside every evaluation
        of the row, so a thread race or leaked state in it moves
        the second fiducial value.
        """
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2("example1", tatt=False,
                                           ee2=True)
        u.report_race_test(
            19, "example1 (cosmic shear, NLA+EE2) race check: 10 "
            "cosmologies in a row", fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
