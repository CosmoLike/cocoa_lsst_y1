"""Shared harness for the lsst_y1 unit tests: the project's data
bound to the shared Cocoa test machinery.

The machinery itself (frozen-state verification, the chi2 pipeline,
worker-subprocess isolation, the race and baryon checks, the
CFASTPT-vs-FASTPT comparison, the N-random-models check, the EE2
modification comparison, and the terminal reports) lives in
external_modules/code/cosmolike_core/cocoa_testing.py. This file
carries what is lsst_y1's alone: the examples table (cosmic shear,
the 3x2pt/2x2pt combinations, and the two EMUL2 emulator
configurations), the TATT point, the accuracy knobs, the dataset
names, the scale-cut mask variants, and the CFASTPT-vs-FASTPT
comparison contract (tolerance and pinned FAST-PT settings). It binds
them to a single cocoa_testing.CocoaTestHarness instance and
re-exports the instance's methods, and the core's functions, as
module-level names, so the test modules and generate_frozen_reference.py
import everything from this module.

Frozen state: everything a test evaluates lives under tests/frozen/,
pinned byte for byte by tests/manifest_sha256.json and verified before
any model is built; refreshing the frozen state is a deliberate
maintainer action (generate_frozen_reference.py --overwrite).
"""

import os
import sys

# ---- tests/ paths -----------------------------------------------------------

# Everything the tests read or write lives relative to this folder, so
# the suite works no matter which directory pytest is launched from
# (__file__ is this module's own path; dirname strips the file name).
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# ---- the shared machinery ---------------------------------------------------

# The import is path-based (tests/ is three levels below Cocoa/, which
# holds external_modules/code/cosmolike_core) so it works before
# start_cocoa.sh's python-path setup runs.
_CORE_DIR = os.path.abspath(os.path.join(
    TESTS_DIR, "..", "..", "..", "external_modules", "code",
    "cosmolike_core"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
import cocoa_testing as _cct
# cocoa_testing_suites holds the factories that build whole shared
# test classes bound to this module; the project's test files assign
# their return values to the module-level names pytest collects
import cocoa_testing_suites as _cct_suites
# cocoa_testing_freeze is the shared machinery behind the
# maintainer tool generate_frozen_reference.py
import cocoa_testing_freeze as _cct_freeze

# ---- the project data ------------------------------------------------------

# ---- TATT -------------------------------------------------------------------

# The TATT (Tidal Alignment and Tidal Torquing, an intrinsic-alignment
# model with tidal second-order terms) tests replace these values in
# the frozen point. In the NLA reference point A2 and BTA are zero, so
# the nonzero values here make the TATT reference genuinely exercise
# the second-order terms.
TATT_POINT = {
    "LSST_A2_1": 0.05,
    "LSST_BTA_1": 0.05,
    "LSST_A2_2": -1.51541,
}

# The TATT variants evaluate against a data vector generated with TATT
# itself at the fiducial point (written by the generator into
# frozen/data/tatt_lsst_y1.modelvector and named on the data_file line
# of this ".dataset" file). Reason: against the shipped NLA-based
# vector the TATT chi2 sits away from its minimum, where it responds
# linearly (not quadratically) to tiny numerical changes: harmless
# rounding-level shifts would eat much of the 0.2 chi2 band the
# reference tests allow. Both combinations share one data set, so a
# single full-length vector generated from the example2 TATT model
# serves every configuration (the other probes' masks select their
# sections).
TATT_DATASET = "tatt_lsst_y1.dataset"

# The comparison sweeps can rerun under a different scale-cut mask
# (the --mask option): each entry names a frozen TATT ".dataset" file
# identical to TATT_DATASET except for its mask_file line, following
# the harness naming rule "<tatt_dataset stem>_<name>.dataset".
# "frozen" is the M1 mask of the frozen contract (959 of the 1,560
# data points kept); M2-M5 cut progressively more points (854, 749,
# 659, 569), M6 cuts fewer (1,378), and "ones" keeps every point (no
# scale cuts), the strictest comparison: the small angular scales the
# shipped masks remove are exactly where the perturbation-theory
# tables matter most.
FASTPT_MASK_DATASETS = {
    "frozen": TATT_DATASET,
    "M2": "tatt_lsst_y1_M2.dataset",
    "M3": "tatt_lsst_y1_M3.dataset",
    "M4": "tatt_lsst_y1_M4.dataset",
    "M5": "tatt_lsst_y1_M5.dataset",
    "M6": "tatt_lsst_y1_M6.dataset",
    "ones": "tatt_lsst_y1_ones.dataset",
}

# ---- configurations ---------------------------------------------------------

# The frozen configurations. Field meanings:
#   "likelihood"        = the cobaya component name, needed to reach that
#                         block inside the loaded info dictionary;
#   "provenance"        = the human-readable snapshot (never loaded);
#   "frozen_module"     = the file, under frozen/, holding the frozen
#                         configuration and point;
#   "source_likelihood" = when the frozen configuration is derived from an
#                         example that ships a different likelihood (the
#                         2x2pt entry reuses example2 with the likelihood
#                         swapped), the generator renames this block;
#   "fastpt_reference"  = also freeze a TATT reference computed with the
#                         python FAST-PT theory block (IA_code: 1) so the
#                         FASTPT tests have their own baseline;
#   "emulator"          = an EMUL2 configuration: machine-learning
#                         emulators replace the Boltzmann code. These are
#                         advisory (no pass/fail; see test_emul2.py);
#                         "exact_example" names the exact-physics
#                         counterpart whose frozen layout the harness
#                         copies, and "exact_reference" the reference
#                         chi2 their accuracy is judged against.
EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "lsst_y1.cosmic_shear",
        "tatt_dataset": TATT_DATASET,
        "fastpt_reference": True,
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "lsst_y1.combo_3x2pt",
        "tatt_dataset": TATT_DATASET,
        "fastpt_reference": True,
    },
    "example2_2x2pt": {
        "frozen_module": "frozen_config_example2_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "source_likelihood": "lsst_y1.combo_3x2pt",
        "likelihood": "lsst_y1.combo_2x2pt",
        "tatt_dataset": TATT_DATASET,
    },
    "emul2_example1": {
        "frozen_module": "frozen_config_emul2_example1.py",
        "provenance": "EXAMPLE_EMUL2_EVALUATE1.yaml",
        "likelihood": "lsst_y1.cosmic_shear",
        "emulator": True,
        "exact_example": "example1",
        "exact_reference": "example1_nla",
    },
    "emul2_example2": {
        "frozen_module": "frozen_config_emul2_example2.py",
        "provenance": "EXAMPLE_EMUL2_EVALUATE2.yaml",
        "likelihood": "lsst_y1.combo_3x2pt",
        "emulator": True,
        "exact_example": "example2",
        "exact_reference": "example2_nla",
    },
}

# ---- accuracy knobs ---------------------------------------------------------

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py): the same physics evaluated with the numerical
# knobs pushed far beyond the defaults.
HIGH_ACCURACY_LIKELIHOOD = {
    # boost 3 is the highest value that stays healthy in every project
    # scanned (desy1xplanck breaks down above it), so the all-knobs
    # check compares the default against 3; the one-at-a-time scan
    # keeps 5 as a deliberate stress knob
    "accuracyboost": 3.0,       # default 1.0
    "internal_accuracyboost": 2.0, # default 1.0 (denser convolution grid)
    "integration_accuracy": 10,  # default 0
    "lmax": 200000,             # default 50000-65000
    "kmax_boltzmann": 40.0,     # default 5.0
}

# The one-at-a-time scan of test_accuracy.py: each entry is (label,
# likelihood overrides, camb extra_args overrides), evaluated alone on
# the example2 NLA configuration before the all-knobs checks, so a
# large all-knobs delta can be attributed to the knob causing it. The
# accuracyboost=5 entry is a stress knob: it exceeds what measuring
# the default numerics needs, and it is kept because it exposed an
# interface breakdown (a suspected fixed-size table) in desy1xplanck.
# Investigation order when several knobs move the chi2: raise the
# cosmolike accuracyboost first (cheap), then camb k_per_logint, and
# only then camb AccuracyBoost (expensive at run time): an apparent
# CAMB sensitivity can masquerade as unresolved cosmolike-side
# resolution, so the cheap knobs must be settled before the expensive
# one is blamed. kmax_boltzmann and camb kmax are one physical cutoff
# seen from the two sides, so the scan moves them together.
ACCURACY_KNOBS = [
    ("accuracyboost 1->3", {"accuracyboost": 3.0}, {}),
    ("accuracyboost 1->5 (stress)", {"accuracyboost": 5.0}, {}),
    ("internal_accuracyboost 1->2", {"internal_accuracyboost": 2.0}, {}),
    ("integration_accuracy 0->10", {"integration_accuracy": 10}, {}),
    ("lmax 50000->200000", {"lmax": 200000}, {}),
    ("kmax_boltzmann -> 40 + camb kmax -> 50",
     {"kmax_boltzmann": 40.0}, {"kmax": 50.0}),
    ("camb AccuracyBoost 1.05->2", {}, {"AccuracyBoost": 2.0}),
    ("camb k_per_logint 10->50", {}, {"k_per_logint": 50}),
]

# Pass limit on the covariance-weighted difference of the two TATT
# implementations: at each point both blocks print their theory data
# vector, and the tested number is delta^T C^-1 delta with
# delta = dv(FASTPT low) - dv(CFASTPT) and C^-1 the masked inverse
# covariance: the chi2 of the implementation difference, zero when the
# vectors agree. The raw chi2 values are printed only as information:
# across the IA prior they are large, so their difference rides the
# local chi2 slope and measures the distance from the data, not the
# numerics. 0.2 is the chi2 band of the reference tests
# (CHI2_TOLERANCE), reachable because FASTPT_LOW_SETTINGS carries the
# converged two-grid configuration (this project's sweep: max delta
# chi2 0.0082 with it, 21.40 across the prior with a single grid).
FASTPT_COMPARISON_TOLERANCE = 0.2

# The python FAST-PT side has numerical settings of its own, read by
# the fastpt theory block from its extra_args block
# (external_modules/code/PyFAST-PT/fastpt.py, symlinked into cobaya
# as theories/fastpt). The block computes on two grids: accuracyboost
# multiplies the density of the output table cosmolike reads with
# linear interpolation (the accuracy driver), and
# internal_accuracyboost the density of the internal grid the FFTLog
# convolutions run on; a cubic spline in log k upsamples the terms
# from one grid onto the other. Both boosts default to 1.0 = the
# converged configuration, so the low settings are the defaults; they
# are written out here so the test keeps evaluating this exact
# configuration even if the defaults later move. High doubles both
# boosts, so the advisory column shows the residual grid response of
# low.
FASTPT_LOW_SETTINGS = {
    "accuracyboost": 1.0,
    "internal_accuracyboost": 1.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

FASTPT_HIGH_SETTINGS = {
    "accuracyboost": 2.0,
    "internal_accuracyboost": 2.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

# ---- project-independent constants ------------------------------------------

# These are identical in every project and live in the core module.
REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
RACE_TOLERANCE = _cct.RACE_TOLERANCE
RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
HIGH_ACCURACY_CAMB_EXTRA_ARGS = _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS
BARYON_METHODS = _cct.BARYON_METHODS
BARYON_POINT_OVERRIDES = _cct.BARYON_POINT_OVERRIDES
RANDOM_MODEL_SEED = _cct.RANDOM_MODEL_SEED
FASTPT_FROZEN_SETTINGS = _cct.FASTPT_FROZEN_SETTINGS

# The 30 CFASTPT-vs-FASTPT comparison points under this project's
# sampled-parameter prefix; the values are identical in every project.
FASTPT_COMPARISON_POINTS = _cct.fastpt_comparison_points("LSST")

# The 10 Halofit-vs-EE2 comparison cosmologies (omegam/ns/As draws,
# no project prefix); the values are identical in every project.
NONLINEAR_COMPARISON_POINTS = _cct.NONLINEAR_COMPARISON_POINTS

# ---- the harness -----------------------------------------------------------

# A single instance binds the shared machinery to this project's data;
# the assignments below re-export its methods, and the core's
# functions, under the module-level names the test modules import.
_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_lsst_y1_interface",
    examples=EXAMPLES,
    tatt_point=TATT_POINT,
    accuracy_knobs=ACCURACY_KNOBS,
    high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
    fastpt_low_settings=FASTPT_LOW_SETTINGS,
    fastpt_high_settings=FASTPT_HIGH_SETTINGS,
    fastpt_points=FASTPT_COMPARISON_POINTS,
    # the masks the CFASTPT-vs-FASTPT sweeps can run under (the --mask
    # option of the tests), keys of FASTPT_MASK_DATASETS above; every
    # non-"frozen" name selects the harness's dataset-variant rule
    # "<tatt_dataset stem>_<name>.dataset"
    fastpt_masks=tuple(FASTPT_MASK_DATASETS),
)

# ---- module functions re-exported from the core (no project state) ----------
require_cocoa_environment = _cct.require_cocoa_environment
assert_omp_threads = _cct.assert_omp_threads
sha256_of = _cct.sha256_of
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
_evaluate_cached = _cct._evaluate_cached
_load_datavector = _cct._load_datavector
_baryon_method = _cct._baryon_method
_baryon_dataset = _cct._baryon_dataset
build_original_ee2 = _cct.build_original_ee2
report_chi2_test = _cct.report_chi2_test
report_race_test = _cct.report_race_test
report_accuracy = _cct.report_accuracy
report_knob = _cct.report_knob
report_fastpt_comparison = _cct.report_fastpt_comparison
report_fastpt_test = _cct.report_fastpt_test
report_nonlinear_comparison = _cct.report_nonlinear_comparison
report_random_model = _cct.report_random_model
report_random_model_summary = _cct.report_random_model_summary
report_emul2_advisory = _cct.report_emul2_advisory
report_emul2_race = _cct.report_emul2_race

# ---- bound methods of the harness (the machinery, project-bound) ------------
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
_frozen_module = _H._frozen_module
load_frozen_info = _H.load_frozen_info
load_frozen_point = _H.load_frozen_point
build_point = _H.build_point
_single_model_chi2_impl = _H._single_model_chi2_impl
_ten_in_a_row_impl = _H._ten_in_a_row_impl
_baryon_accuracy_delta_impl = _H._baryon_accuracy_delta_impl
_baryon_drift_chi2_impl = _H._baryon_drift_chi2_impl
single_model_chi2 = _H.single_model_chi2
ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
baryon_accuracy_delta = _H.baryon_accuracy_delta
baryon_drift_chi2 = _H.baryon_drift_chi2
draw_uniform_point = _H.draw_uniform_point
random_model_accuracy = _H.random_model_accuracy
ee2_original_vs_cocoa_dchi2s = _H.ee2_original_vs_cocoa_dchi2s
_ee2_comparison_block = _H._ee2_comparison_block
_worker = _H._worker
_run_isolated = _H._run_isolated
_fastpt_comparison_info = _H._fastpt_comparison_info
_fastpt_comparison_block = _H._fastpt_comparison_block
_run_fastpt_comparison_worker = _H._run_fastpt_comparison_worker
cfastpt_vs_fastpt_chi2s = _H.cfastpt_vs_fastpt_chi2s
_nonlinear_comparison_block = _H._nonlinear_comparison_block
halofit_vs_ee2_dchi2s = _H.halofit_vs_ee2_dchi2s

# ---- shared test-class factories (bound by the test files) ------------------
example_suite = _cct_suites.example_suite
accuracy_suite = _cct_suites.accuracy_suite
fastpt_reference_suite = _cct_suites.fastpt_reference_suite
fastpt_sweep_suite = _cct_suites.fastpt_sweep_suite
ee2_suite = _cct_suites.ee2_suite
nonlinear_suite = _cct_suites.nonlinear_suite
photoz_conventions_suite = _cct_suites.photoz_conventions_suite
nonlimber_suite = _cct_suites.nonlimber_suite
cache_ladder_suite = _cct_suites.cache_ladder_suite
baryon_accuracy_suite = _cct_suites.baryon_accuracy_suite
baryon_drift_suite = _cct_suites.baryon_drift_suite
freeze_main = _cct_freeze.freeze_main
