"""Unit tests 9, 10, and 15-17: the TATT terms computed with python FAST-PT.

Cosmolike offers two implementations of the perturbation-theory
integrals that the TATT intrinsic-alignment model needs: cfastpt, a C
implementation built into the cosmolike interface (`IA_code: 0`, the
default everywhere else in this suite), and the python FAST-PT package
used through the fastpt theory block (`IA_code: 1`). The two
implementations compute the same integrals with different numerics,
so their chi2 values at the same point differ by a small amount.

Tests 9 and 10 evaluate the frozen TATT point with FASTPT and:

  - compare the chi2 with its own frozen FASTPT reference: the
    difference must stay below CHI2_TOLERANCE (0.2), the same pass
    rule as tests 1-8;
  - print the FASTPT-minus-CFASTPT difference next to the frozen
    value of that difference, so a numerics change in either
    implementation is visible at a glance.

  9. example1 (cosmic shear), TATT with LSST_A2_1 = 0.05,
     LSST_BTA_1 = 0.05, LSST_A2_2 = -1.51541.
 10. example2 (3x2pt), same TATT point.
 15. example1 (cosmic shear): the same 30 hard-coded points across
     the intrinsic-alignment prior (FASTPT_COMPARISON_POINTS;
     cosmology fixed at the frozen fiducial) evaluated three times:
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
 16. example2 (3x2pt): the same three-block sweep as test 15 on the
     3x2pt likelihood, so the TATT terms are also scored inside
     galaxy-galaxy lensing and under the 3x2pt covariance. The
     frozen configuration fixes the one-loop bias amplitudes
     (LSST_B2_*) at zero, so the one-loop galaxy-bias tables both
     implementations compute multiply by zero here: this sweep
     compares the intrinsic-alignment tables only, on the wider
     data vector.
 17. example2_2x2pt (2x2pt): the same sweep on the 2x2pt likelihood
     (galaxy clustering plus galaxy-galaxy lensing, cosmic shear
     dropped). Clustering carries no intrinsic alignment, so here
     the TATT tables are scored through galaxy-galaxy lensing
     alone, under the 2x2pt covariance; LSST_B2_* stays zero as in
     test 16.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests/data_vector

Tests 15-17 repeat at the pushed numerical settings of the
low-vs-high accuracy checks (HIGH_ACCURACY_LIKELIHOOD and
HIGH_ACCURACY_CAMB_EXTRA_ARGS, applied to every block) when the
--high=1 option is given; the full comparison is one run without the
option and one with it, so the 30 points go through the FASTPT side
four times (fastpt low and high, under each camb/cosmolike setting):

    python -m pytest ./projects/lsst_y1/tests/data_vector/test_fastpt.py --high=1

Tests 15-17 also read the --mask option (see conftest.py and
FASTPT_MASK_DATASETS): --mask=frozen (the default) keeps the M1
mask of the frozen contract, --mask=M2 .. --mask=M6 select the
other shipped scale cuts, and --mask=ones keeps every data point
(no scale cuts), the strictest comparison; the 0.2 pass rule
applies unchanged:

    python -m pytest ./projects/lsst_y1/tests/data_vector/test_fastpt.py --mask=ones
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factories (cosmolike_core/cocoa_testing_suites.py) build
# the classes; assigning each return value to the usual module-level
# name is what makes pytest collect it here, with unchanged test ids.
TestFastptTatt = u.fastpt_reference_suite(
    u, name="TestFastptTatt",
    cases=(
        ('test_9_fastpt_example1_tatt', 9, 'example1', 'example1 (cosmic shear, TATT+FASTPT) chi2 vs frozen reference'),
        ('test_x10_fastpt_example2_tatt', 10, 'example2', 'example2 (3x2pt, TATT+FASTPT) chi2 vs frozen reference'),
    ))

TestCfastptVsFastptSweep = u.fastpt_sweep_suite(
    u, name="TestCfastptVsFastptSweep",
    cases=(
        ('test_x15_cfastpt_vs_fastpt_sweep', 15, 'example1', 'example1 (cosmic shear'),
        ('test_x16_cfastpt_vs_fastpt_sweep_3x2pt', 16, 'example2', 'example2 (3x2pt'),
        ('test_x17_cfastpt_vs_fastpt_sweep_2x2pt', 17, 'example2_2x2pt', 'example2_2x2pt (2x2pt'),
    ))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
