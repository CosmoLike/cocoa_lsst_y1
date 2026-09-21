"""Unit tests 9-10: the TATT terms computed with python FAST-PT.

Cosmolike offers two implementations of the perturbation-theory
integrals that the TATT intrinsic-alignment model needs: cfastpt, a C
implementation built into the cosmolike interface (`IA_code: 0`, the
default everywhere else in this suite), and the python FAST-PT package
used through the fastpt theory block (`IA_code: 1`). The two
implementations compute the same integrals with different numerics,
so their chi2 values at the same point differ by a small amount.

Each test evaluates the frozen TATT point with FASTPT and:

  - compares the chi2 against its own frozen FASTPT reference within
    CHI2_TOLERANCE (0.2) of its own frozen reference, the same
    pass rule as tests 1-8;
  - prints the FASTPT-minus-CFASTPT difference next to the frozen
    value of that difference, so a numerics change in either
    implementation is visible at a glance.

  9. example1 (cosmic shear), TATT with LSST_A2_1 = 0.05,
     LSST_BTA_1 = 0.05, LSST_A2_2 = -1.51541.
 10. example2 (3x2pt), same TATT point.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


class TestFastptTatt(unittest.TestCase):
    """Tests 9-10, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR,
    verifies every frozen file against the SHA-256 manifest (an edited
    frozen state must fail loudly before any physics runs), and loads
    the frozen reference chi2 values.
    """

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
        print(f"  frozen FASTPT - CFASTPT difference = {frozen_diff:+.6f}",
              flush=True)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
