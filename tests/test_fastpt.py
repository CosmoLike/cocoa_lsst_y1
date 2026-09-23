"""Unit tests 9, 10, and 15: the TATT terms computed with python FAST-PT.

Cosmolike offers two implementations of the perturbation-theory
integrals that the TATT intrinsic-alignment model needs: cfastpt, a C
implementation built into the cosmolike interface (`IA_code: 0`, the
default everywhere else in this suite), and the python FAST-PT package
used through the fastpt theory block (`IA_code: 1`). The two
implementations compute the same integrals with different numerics,
so their chi2 values at the same point differ by a small amount.

Tests 9 and 10 evaluate the frozen TATT point with FASTPT and:

  - compare the chi2 against its own frozen FASTPT reference within
    CHI2_TOLERANCE (0.2) of its own frozen reference, the same
    pass rule as tests 1-8;
  - print the FASTPT-minus-CFASTPT difference next to the frozen
    value of that difference, so a numerics change in either
    implementation is visible at a glance.

  9. example1 (cosmic shear), TATT with LSST_A2_1 = 0.05,
     LSST_BTA_1 = 0.05, LSST_A2_2 = -1.51541.
 10. example2 (3x2pt), same TATT point.
 15. example1 (cosmic shear): the SAME 30 hard-coded points across
     the intrinsic-alignment prior (FASTPT_COMPARISON_POINTS;
     cosmology fixed at the frozen fiducial) evaluated three times -
     with cfastpt, with FASTPT at the converged two-grid defaults
     (FASTPT_LOW_SETTINGS, hard-coded), and with FASTPT at the
     doubled boosts (FASTPT_HIGH_SETTINGS). Every block
     prints its theory vector at every point, and the CFASTPT vector
     is the fiducial of that point: its own chi2 against it is zero
     by construction, so the pass rule is the chi2 of the
     FASTPT(low) vector against it (delta^T C^-1 delta, a pure
     second-order deviation; a chi2 difference against the shipped
     data would ride the slope instead). FASTPT(high)'s deviation is
     printed as the advisory FAST-PT grid response. Each
     configuration runs in its own subprocess, so no cache survives
     from one block to the next; inside a block the shared cosmology
     makes CAMB run once and the 30 points cheap.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests

Test 15 repeats at the pushed numerical settings of the low-vs-high
accuracy checks (HIGH_ACCURACY_LIKELIHOOD and
HIGH_ACCURACY_CAMB_EXTRA_ARGS, applied to every block) when the
--high=1 option is given; the full comparison is one run without the
option and one with it, so the 30 points go through the FASTPT side
four times (fastpt low and high, under each camb/cosmolike setting):

    python -m pytest ./projects/lsst_y1/tests/test_fastpt.py --high=1
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


class TestFastptTatt(unittest.TestCase):
    """Tests 9-10, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR,
    verifies every frozen file against the SHA-256 manifest (an edited
    frozen state must fail loudly before any physics runs), and loads
    the frozen reference chi2 values.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def _run_fastpt_case(self, number, example, label):
        """Evaluate one FASTPT TATT case against its frozen reference.

        Shared by both tests: the only differences between them are
        the example evaluated and the report header.

        Arguments:
          number  = the test number shown in the report header.
          example = a key of cocoa_test_utils.EXAMPLES.
          label   = one line naming the example and probe.
        """
        chi2 = u.single_model_chi2(example, tatt=True, fastpt=True)
        ref = self.reference[f"{example}_tatt_fastpt"]
        cfastpt_ref = self.reference[f"{example}_tatt"]
        u.report_fastpt_test(number, label, chi2, ref, cfastpt_ref,
                             u.CHI2_TOLERANCE)
        frozen_diff = self.reference[f"{example}_fastpt_minus_cfastpt"]
        # the + in :+.6f forces a sign, so the direction of the
        # frozen difference always shows
        print(f"  frozen FASTPT - CFASTPT difference = {frozen_diff:+.6f}",
              flush=True)
        # assertLess(a, b) passes when a < b and fails with msg
        # otherwise; in that message, :.6f prints fixed six decimals
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"FASTPT chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_9_fastpt_example1_tatt(self):
        """example1 TATT with python FAST-PT stays on its reference."""
        self._run_fastpt_case(
            9, "example1", "example1 (cosmic shear, TATT+FASTPT) chi2 "
            "vs frozen reference")

    def test_x10_fastpt_example2_tatt(self):
        """example2 TATT with python FAST-PT stays on its reference.

        The method name carries the x prefix only so unittest's
        alphabetical ordering runs it after test_9.
        """
        self._run_fastpt_case(
            10, "example2", "example2 (3x2pt, TATT+FASTPT) chi2 "
            "vs frozen reference")


class TestCfastptVsFastptSweep(unittest.TestCase):
    """Test 15, the direct CFASTPT-vs-FASTPT comparison.

    setUpClass runs once before the test: it moves to ROOTDIR and
    verifies every frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: this test compares the two
    implementations against each other, so the frozen state only
    supplies the configuration and the data files.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x15_cfastpt_vs_fastpt_sweep(self):
        """Cosmic shear: cfastpt and FASTPT agree at 30 IA points.

        The method name carries the x prefix only so unittest's
        alphabetical ordering runs it after tests 9 and 10.
        """
        # the conftest copies the --high command line option into this
        # variable; .get with the "0" default keeps a run outside
        # pytest (plain `python test_fastpt.py`) on the default
        # settings unless the variable is exported by hand
        high = os.environ.get("COCOA_FASTPT_HIGH", "0") == "1"
        setting = "high accuracy" if high else "default settings"
        (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
         dchi2_low, dchi2_high) = u.cfastpt_vs_fastpt_chi2s(
            "example1", high=high)
        largest = u.report_fastpt_comparison(
            15,
            f"example1 (cosmic shear, TATT, camb/cosmolike {setting}): "
            "CFASTPT vs FASTPT at 30 hard-coded points",
            chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
            dchi2_low, dchi2_high, u.FASTPT_COMPARISON_TOLERANCE)
        self.assertLess(
            largest, u.FASTPT_COMPARISON_TOLERANCE,
            msg="max chi2 of the FASTPT(low)-vs-CFASTPT data-vector "
                f"difference = {largest:.6f} over the comparison "
                f"points ({setting}); limit "
                f"{u.FASTPT_COMPARISON_TOLERANCE}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
