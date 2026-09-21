"""Accuracy advisory checks A1-A6: default vs high-accuracy settings.

Every reference in this suite is computed with the examples' default
numerical settings. These checks answer: how much numerical error do
those defaults carry? Each one re-evaluates a frozen configuration at
its frozen point with the numerical knobs pushed far beyond the
defaults (cosmolike: accuracyboost 2, integration_accuracy 10,
lmax 200000, kmax_boltzmann 40; CAMB: AccuracyBoost 2.0,
k_per_logint 50, kmax 50; the exact values live in
cocoa_test_utils.HIGH_ACCURACY_*) and reports

    delta chi2 = chi2(high accuracy) - chi2(default, frozen)

There is NO pass/fail: how much numerical error an analysis tolerates
is a judgment call. The six checks cover the three probes with both
IA models:

  A1. cosmic shear (example1), NLA      A2. cosmic shear, TATT
  A3. 2x2pt (example2_2x2pt), NLA       A4. 2x2pt, TATT
  A5. 3x2pt (example2), NLA             A6. 3x2pt, TATT

The TATT checks evaluate against the TATT-generated data vector
(frozen/data/tatt_lsst_y1.dataset, written at freeze time), so the
chi2 sits at a minimum and the delta is a stable, quadratic response
instead of a linear one.

Before the all-knobs checks, one scan (K) evaluates each accuracy
knob ALONE on the example2 NLA configuration, so a large all-knobs
delta can be attributed to the knob causing it. The scan includes
accuracyboost 5 as a stress knob: past experience (desy1xplanck) is
that extreme boosts can break an interface rather than refine it,
and the one-at-a-time delta is what tells those cases apart.

One opt-in check extends the scan beyond the fiducial: the
N-random-models check evaluates the same delta at N reproducible
random points across the prior, each against a synthetic data vector
generated at that point (see test_ax99_nmodels). It runs only when
the COCOA_ACCURACY_NMODELS environment variable is a positive
integer:

    COCOA_ACCURACY_NMODELS=10 python -m pytest \\
        ./projects/lsst_y1/tests/test_accuracy.py -k nmodels

When this file runs as a script, --nmodels N sets the same option:

    python ./projects/lsst_y1/tests/test_accuracy.py --nmodels 10

With the variable unset (or 0) the check prints how to enable it and
passes without evaluating anything.

A high-accuracy evaluation takes minutes, not seconds: the whole file
is far slower than the rest of the suite. To run only this file (from
the Cocoa/ folder, cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests/test_accuracy.py

and to run the rest of the suite without it:

    python -m pytest ./projects/lsst_y1/tests --ignore \\
        ./projects/lsst_y1/tests/test_accuracy.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import math
import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
# insert(0, ...) puts the folder FIRST in the search order, ahead of
# every other place a same-named module could hide.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


class TestAccuracyAdvisory(unittest.TestCase):
    """Advisory checks A1-A6, sharing one frozen-state verification.

    setUpClass runs once: it moves to ROOTDIR, verifies every frozen
    file against the SHA-256 manifest (an edited frozen state must
    fail loudly before any physics runs), and loads the frozen
    reference chi2 values.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def _accuracy_check(self, name, example, tatt, label):
        """Evaluate one configuration at high accuracy and report.

        Arguments:
          name    = the advisory label (A1-A6) for the report.
          example = a key of cocoa_test_utils.EXAMPLES (exact-physics
                    configurations only).
          tatt    = True evaluates the TATT variant against the
                    TATT-generated data vector, False the NLA one.
          label   = one line naming the probe and IA model.
        """
        chi2_high = u.single_model_chi2(example, tatt, high_accuracy=True)
        # ternary: the reference key ends in "tatt" or "nla", the
        # naming the frozen reference file uses
        suffix = "tatt" if tatt else "nla"
        default_ref = self.reference[f"{example}_{suffix}"]
        u.report_accuracy(f"{name}: {label}", chi2_high, default_ref)

    def test_a0_one_knob_at_a_time(self):
        """K scan: each accuracy knob alone on example2, NLA.

        Advisory: each knob's chi2 and its difference to the frozen
        default reference print as the scan runs. A knob whose delta
        rivals the all-knobs delta is the driver; a knob whose delta
        explodes (orders of magnitude beyond the others) signals an
        interface breakdown, not a numerics improvement.
        """
        default_ref = self.reference["example2_nla"]
        print("", flush=True)
        # each knob entry is (label, likelihood overrides, camb
        # overrides); the two _ discard the override tables here,
        # single_model_chi2 looks them up again by label
        for label, _, _ in u.ACCURACY_KNOBS:
            chi2 = u.single_model_chi2("example2", False, knob=label)
            u.report_knob(label, chi2, default_ref)

    def test_a1_cosmic_shear_nla(self):
        """A1: cosmic shear, NLA, default vs high accuracy."""
        self._accuracy_check("A1", "example1", False,
                             "example1 (cosmic shear, NLA)")

    def test_a2_cosmic_shear_tatt(self):
        """A2: cosmic shear, TATT, default vs high accuracy."""
        self._accuracy_check("A2", "example1", True,
                             "example1 (cosmic shear, TATT)")

    def test_a3_2x2pt_nla(self):
        """A3: 2x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A3", "example2_2x2pt", False,
                             "example2_2x2pt (2x2pt, NLA)")

    def test_a4_2x2pt_tatt(self):
        """A4: 2x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A4", "example2_2x2pt", True,
                             "example2_2x2pt (2x2pt, TATT)")

    def test_a5_3x2pt_nla(self):
        """A5: 3x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A5", "example2", False,
                             "example2 (3x2pt, NLA)")

    def test_a6_3x2pt_tatt(self):
        """A6: 3x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A6", "example2", True,
                             "example2 (3x2pt, TATT)")

    def test_ax99_nmodels(self):
        """N-random-models check across the prior. Opt-in, advisory.

        A1-A6 measure the numerical error of the default settings at
        the one frozen fiducial point. This check measures it at N
        reproducible random points across the prior of the example2
        (3x2pt, NLA) configuration instead. For each point a
        synthetic data vector is generated AT that point with the
        default settings, so the default chi2 against it is zero by
        construction and the high-accuracy chi2 against it is the
        delta directly (the mechanism lives in
        cocoa_test_utils.random_model_accuracy). Nothing fails on a
        large delta; the deltas only have to be finite.

        Opt-in because each model costs a default build+evaluation
        plus a high-accuracy build+evaluation, minutes per model.
        Enable it with the COCOA_ACCURACY_NMODELS environment
        variable, or with --nmodels N when this file runs as a
        script:

            COCOA_ACCURACY_NMODELS=10 python -m pytest \\
                ./projects/lsst_y1/tests/test_accuracy.py -k nmodels

        The ax99 in the method name sorts this check after A1-A6
        (unittest runs methods in name order), so the cheap fiducial
        checks always report before the expensive multi-model loop
        starts.
        """
        # .get returns its second argument, "0", when the variable
        # is unset, so an untouched environment means a clean skip
        n_models_text = os.environ.get("COCOA_ACCURACY_NMODELS", "0")
        # int() raises ValueError on non-numeric text; the except
        # turns the stack trace into an instruction
        try:
            n_models = int(n_models_text)
        except ValueError:
            raise AssertionError(
                f"COCOA_ACCURACY_NMODELS={n_models_text!r} is not an "
                "integer; set it to the number of random models, for "
                "example COCOA_ACCURACY_NMODELS=10")
        if n_models <= 0:
            # the default: report how to enable the check and pass
            # without building anything
            print("\n  N-random-models check skipped (set "
                  "COCOA_ACCURACY_NMODELS=<N>, or run this file with "
                  "--nmodels <N>, to enable it).", flush=True)
            return
        print(f"\n  N-random-models check: {n_models} models, example2 "
              f"(3x2pt, NLA), seed base {u.RANDOM_MODEL_SEED}",
              flush=True)
        deltas = u.random_model_accuracy("example2", n_models)
        # enumerate pairs each delta with its position, counted
        # from 0, so a broken model is named by number
        for index, delta in enumerate(deltas):
            # advisory: the size of the delta is a judgment call, but
            # a non-finite one means the evaluation itself broke
            self.assertTrue(
                math.isfinite(delta),
                msg=f"model {index}: non-finite delta chi2 {delta!r}")
        u.report_random_model_summary(deltas)


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    # --nmodels N is the script-run spelling of COCOA_ACCURACY_NMODELS;
    # it must leave sys.argv before unittest.main parses the arguments.
    # `in` scans the argument list for the flag, and .index returns
    # the position of its first occurrence
    if "--nmodels" in sys.argv:
        flag_index = sys.argv.index("--nmodels")
        if flag_index + 1 >= len(sys.argv):
            sys.exit("--nmodels requires a value, for example "
                     "--nmodels 10")
        os.environ["COCOA_ACCURACY_NMODELS"] = sys.argv[flag_index + 1]
        # del on the slice removes the flag and its value from the
        # list in place, so unittest.main never sees them
        del sys.argv[flag_index:flag_index + 2]
    unittest.main(verbosity=2)
