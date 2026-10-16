"""Unit tests 18-19: the Cocoa modifications of EuclidEmulator2.

Cocoa pins a modified EuclidEmulator2 (the EE2_GIT_COMMIT of
set_installation_options.sh): OpenMP threading, a 1,010-redshift
capacity, the get_boost2 API with a pre-built emulator, memory-leak
fixes, and a bilinear interpolation with a border fix (the
repository's README documents them). These tests pin the
modifications two ways:

 18. the installed, modified EE2 against the pre-modification code
     (commit ff59f66), built side by side into a temporary prefix
     from the local clone (offline; --ignore-installed protects
     .local). Both builds evaluate the cosmic-shear NLA data vector
     with non_linear_emul: 1 at the ten shared cosmologies
     (NONLINEAR_COMPARISON_POINTS), and the per-cosmology
     delta^T C^-1 delta of the original against the modified build
     must stay below CHI2_TOLERANCE (0.2). The original cannot run
     inside Cocoa as-is (it lacks get_boost2 and overflows beyond
     101 redshifts), so its worker carries the compatibility patch
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

    python -m pytest ./projects/lsst_y1/tests/data_vector/test_ee2.py

Test 18 compiles the original EE2 once per run (about half a
minute); it skips, rather than fails, when the local euclidemu2
clone does not carry the original commit.
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
TestEE2Modifications = u.ee2_suite(
    u, name="TestEE2Modifications",
    race=('test_x19_ee2_race_ten_in_a_row', 19, 'example1', 'example1 (cosmic shear, NLA+EE2)'),
    original=('test_x18_ee2_modifications_vs_original', 18, 'example1'))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
