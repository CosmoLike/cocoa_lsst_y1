"""Shared harness for the lsst_y1 unit tests.

The tests answer two questions about the lsst_y1 likelihoods:

  1. Does the chi2 at a fixed reference point still match the value
     recorded when the tests were created (tests 1, 3, 5, 7)?
  2. Does evaluating several cosmologies in a row on one model change
     the answer for a point, compared to evaluating that point alone
     (tests 2, 4, 6, 8)? Cosmolike keeps internal state in C between
     evaluations, and OpenMP splits loops across threads; a bug in
     either would make the 10th evaluation of a sequence differ from a
     fresh evaluation of the same point. That class of bug is called a
     race condition or a state leak.

Beyond the eight pass/fail tests, the suite carries: two FASTPT
comparison tests (the TATT terms computed by the python FAST-PT
package instead of the C implementation cfastpt, with the difference
between the two printed and saved), a quartet of tests on the 2x2pt
likelihood (example2 with the probe selection reduced to clustering
plus galaxy-galaxy lensing), and ADVISORY checks of the EMUL2
examples (machine-learning emulators in place of the Boltzmann code;
see test_emul2.py: no pass/fail, only the measured accuracy and a
recommendation).

Everything a test evaluates is FROZEN: stored under tests/frozen/ and
pinned by a SHA-256 hash (a 64-character fingerprint that changes when
any byte of the file changes) in tests/manifest_sha256.json. The live
project configuration is never read, so a user can edit the examples,
the likelihood default yaml files, or ../data without touching these
tests. The frozen state has three parts:

  - frozen/frozen_config_*.py: one auto-generated module per
    configuration in EXAMPLES, holding (a) the complete cobaya
    configuration as a yaml string, with every likelihood option and
    every parameter written out, including the ones that normally come
    from the likelihood default files (cosmic_shear.yaml,
    combo_3x2pt.yaml, params_source.yaml, params_lens.yaml), and
    (b) the exact sampled-parameter point the reference chi2 was
    evaluated at. Because every default is materialized in the frozen
    copy, a later edit to a live default file is shadowed and cannot
    reach the test.
  - frozen/data/: the tests' own copy of the data vectors, covariance,
    n(z), and masks. (The EMUL2 trained-network files are NOT copied:
    they live in external_modules/data/emultrf, pinned by the EMULTRF
    keys in set_installation_options.sh. When a retrained network
    replaces them the emulator chi2 changes, and measuring that
    change is part of the advisory checks' job.)
  - frozen/EXAMPLE_*.yaml: snapshots of the example yaml files at
    freeze time, kept only so a human can diff what changed in the live examples
    since the freeze; no test reads them.

Every test first verifies the manifest and refuses to run when any
frozen file changed. Refreshing the frozen state is a deliberate
maintainer action: generate_frozen_reference.py --overwrite.

Call flow, top to bottom (the MAP OF THIS FILE below the imports says
where each definition lives):

    test method (test_example1.py and the other test modules)
      |
      |  setUpClass, once per test class:
      |    require_cocoa_environment()  chdir to ROOTDIR, or refuse
      |    verify_frozen()              hash frozen/ against the manifest
      |    load_reference()             read the frozen reference chi2
      |
      +-> single_model_chi2(example, tatt, ...)      tests 1, 3, 5, 7
      |     +-> load_frozen_info(...)  frozen yaml -> cobaya input dict
      |     +-> make_model(info)       input dict -> evaluable Model
      |     +-> build_point(...)       frozen point, names checked
      |     +-> evaluate_chi2(...)     one point -> chi2 (-2 ln L)
      |
      +-> ten_in_a_row_chi2(example, tatt)           tests 2, 4, 6, 8
      |     the same four steps, with evaluate_chi2 run 11 times on
      |     one shared model instance (the race check)
      |
      +-> random_model_accuracy(example, n_models)   opt-in accuracy
      |     N random prior points; at each point a synthetic data
      |     vector is generated with the default settings, then the
      |     HIGH_ACCURACY chi2 against that vector is the delta
      |     (test_accuracy.py, enabled by COCOA_ACCURACY_NMODELS)
      |
      +-> report_*(...)   print the numbers; the test itself asserts

Glossary:

  frozen state   = the copy of configurations, data files, and points
                   under tests/frozen/ that the tests read instead of
                   the live project files.
  manifest       = tests/manifest_sha256.json, the {path: SHA-256}
                   table that defines "untouched" for every frozen
                   file.
  reference      = a chi2 recorded at freeze time in
                   frozen/reference_chi2.json; each test compares its
                   freshly computed chi2 against one reference.
  variant        = one configuration under one intrinsic-alignment
                   model, for example example1_nla or example2_tatt;
                   the keys of the reference file name the variants.
  knob           = one numerical-accuracy setting (an ACCURACY_KNOBS
                   entry) pushed beyond its default to measure the
                   numerical error the default carries.
  fiducial point = the frozen sampled-parameter point the references
                   were evaluated at; the TATT variants replace three
                   of its values with TATT_POINT.

To run the tests: activate the cocoa conda environment, source
start_cocoa.sh from the Cocoa/ folder (this exports ROOTDIR), then

    python -m pytest ./projects/lsst_y1/tests
"""

import hashlib
import json
import os
import shutil
import tempfile

# =============================================================================
# MAP OF THIS FILE
# =============================================================================
# Section 1: CONSTANTS
#   paths               TESTS_DIR, FROZEN_DIR, MANIFEST_FILE, REFERENCE_FILE
#   tolerances          REQUIRED_OMP_THREADS, CHI2_TOLERANCE, RACE_TOLERANCE
#   TATT                TATT_POINT, TATT_DATASET
#   configurations      EXAMPLES
#   race perturbations  RACE_PERTURBATIONS
#   accuracy knobs      HIGH_ACCURACY_LIKELIHOOD,
#                       HIGH_ACCURACY_CAMB_EXTRA_ARGS, ACCURACY_KNOBS
#   random models       RANDOM_MODEL_SEED
#
# Section 2: ENVIRONMENT CHECKS
#   require_cocoa_environment  refuse to run outside a started Cocoa shell
#   assert_omp_threads         refuse a race test that is not multi-threaded
#
# Section 3: FROZEN-STATE INTEGRITY
#   sha256_of         fingerprint one file with SHA-256
#   compute_manifest  hash every file currently under tests/frozen/
#   verify_frozen     fail every test up front when the frozen state changed
#   load_reference    read the frozen reference chi2 values
#
# Section 4: MODEL CONSTRUCTION AND EVALUATION (the chi2 pipeline)
#   _frozen_module     load one frozen configuration module by file path
#   load_frozen_info   frozen yaml string -> cobaya input dictionary
#   make_model         cobaya input dictionary -> evaluable Model
#   load_frozen_point  read the frozen evaluation point of one example
#   build_point        frozen point, cross-checked against the model
#   evaluate_chi2      one point on one model -> chi2 (-2 ln L)
#   draw_uniform_point     one random point across the uniform priors
#   random_model_accuracy  N random points, one synthetic vector each,
#                          delta chi2 = chi2(high accuracy) per point
#
# Section 5: TEST QUANTITIES (what the test methods call)
#   single_model_chi2  chi2 of the fiducial point on a fresh model
#   ten_in_a_row_chi2  fiducial fresh vs fiducial as 10th of 10 (race)
#
# Section 6: TERMINAL REPORTS (printers; the assertions live in the tests)
#   report_chi2_test       pass/fail block for tests 1, 3, 5, 7
#   report_race_test       pass/fail block for tests 2, 4, 6, 8
#   report_fastpt_test     pass/fail block for the FASTPT comparison tests
#   report_emul2_advisory  advisory block: emulator accuracy + verdict
#   report_emul2_race      advisory block: emulator race check
#   report_accuracy        advisory block: default vs high-accuracy chi2
#   report_knob            one line of the one-knob-at-a-time scan
#   report_random_model         advisory block: one random-model delta
#   report_random_model_summary min/median/max of the random deltas
# =============================================================================


# =============================================================================
# SECTION 1: CONSTANTS
# =============================================================================

# ---- paths ------------------------------------------------------------------

# Everything the tests read or write lives relative to this folder, so
# the suite works no matter which directory pytest is launched from.
# __file__ is this module's own file path; abspath + dirname reduce it
# to the tests/ folder.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# ---- tolerances -------------------------------------------------------------

# The race tests must run multi-threaded: with one thread there is no
# thread scheduling, so an OpenMP race could never show up.
REQUIRED_OMP_THREADS = "4"

# Tests 1, 3, 5, 7: |chi2(now) - chi2(frozen reference)| must stay
# below this. The bound tolerates compiler and library-version noise
# but catches a real physics change.
CHI2_TOLERANCE = 0.2

# Tests 2, 4, 6, 8: |chi2(10th of a row) - chi2(fresh model)|. The two
# numbers come from the same code on the same inputs, so only float
# noise is allowed; a state leak produces a much larger shift.
RACE_TOLERANCE = 1.0e-4

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

# The TATT variants evaluate against a data vector GENERATED WITH TATT
# at the fiducial point (written by the generator into
# frozen/data/tatt_lsst_y1.modelvector and named on the data_file
# line of this ".dataset" file). Reason: with the shipped NLA-based data vector the TATT
# chi2 at the fiducial is well above zero, and away from a minimum the
# chi2 responds linearly (not quadratically) to tiny numerical
# changes: harmless rounding-level shifts would eat much of the 0.2
# chi2 band the reference tests allow. Against its own data vector
# the TATT chi2 sits at the minimum, where it is stable.
TATT_DATASET = "tatt_lsst_y1.dataset"

# ---- configurations ---------------------------------------------------------

# The frozen configurations. Field meanings:
#   "likelihood"        = the cobaya component name, needed to reach that
#                         block inside the loaded info dictionary;
#   "provenance"        = the human-readable snapshot (never loaded);
#   "source_likelihood" = when the frozen configuration is derived from an
#                         example that ships a DIFFERENT likelihood (the
#                         2x2pt entry reuses example2 with the likelihood
#                         swapped), the generator renames this block;
#   "fastpt_reference"  = also freeze a TATT reference computed with the
#                         python FAST-PT theory block (IA_code: 1) so the
#                         FASTPT tests have their own baseline;
#   "emulator"          = an EMUL2 configuration: machine-learning
#                         emulators replace the Boltzmann code. These are
#                         ADVISORY (no pass/fail; see test_emul2.py), and
#                         "exact_reference" names the exact-physics
#                         reference chi2 their accuracy is judged against.
EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "lsst_y1.cosmic_shear",
        "fastpt_reference": True,
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "lsst_y1.combo_3x2pt",
        "fastpt_reference": True,
    },
    "example2_2x2pt": {
        "frozen_module": "frozen_config_example2_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "source_likelihood": "lsst_y1.combo_3x2pt",
        "likelihood": "lsst_y1.combo_2x2pt",
    },
    "emul2_example1": {
        "frozen_module": "frozen_config_emul2_example1.py",
        "provenance": "EXAMPLE_EMUL2_EVALUATE1.yaml",
        "likelihood": "lsst_y1.cosmic_shear",
        "emulator": True,
        "exact_reference": "example1_nla",
    },
    "emul2_example2": {
        "frozen_module": "frozen_config_emul2_example2.py",
        "provenance": "EXAMPLE_EMUL2_EVALUATE2.yaml",
        "likelihood": "lsst_y1.combo_3x2pt",
        "emulator": True,
        "exact_reference": "example2_nla",
    },
}

# ---- race perturbations -----------------------------------------------------

# The nine cosmologies evaluated before the fiducial point in a race
# test. Each entry replaces the named parameters in the frozen point.
# They stay inside the priors of the frozen configuration (an
# out-of-prior point would evaluate to -inf and abort the test), and
# they change the chi2 by orders of magnitude, so state leaked from
# any of them would visibly move the final fiducial evaluation.
RACE_PERTURBATIONS = [
    {"As_1e9": 1.95},
    {"As_1e9": 2.25},
    {"omegam": 0.28},
    {"omegam": 0.33},
    {"H0": 64.0},
    {"H0": 71.0},
    {"ns": 0.95},
    {"w": -1.1, "w0pwa": -1.1},
    {"omegab": 0.052, "mnu": 0.15},
]

# ---- accuracy knobs ---------------------------------------------------------

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py): the same physics evaluated with the numerical
# knobs pushed far beyond the defaults. The difference to the default
# reference chi2 measures the numerical error of the DEFAULT settings.
HIGH_ACCURACY_LIKELIHOOD = {
    # boost 3 is the highest value that stays healthy in every project
    # scanned (desy1xplanck breaks down above it), so the all-knobs
    # check compares the default against 3; the one-at-a-time scan
    # keeps 5 as a deliberate stress knob
    "accuracyboost": 3.0,       # default 1.0
    "integration_accuracy": 10,  # default 0
    "lmax": 200000,             # default 50000-65000
    "kmax_boltzmann": 40.0,     # default 5.0
}
HIGH_ACCURACY_CAMB_EXTRA_ARGS = {
    "halofit_version": "takahashi",
    "AccuracyBoost": 2.0,       # default 1.05
    "dark_energy_model": "ppf",
    "accurate_massive_neutrino_transfers": False,
    "k_per_logint": 50,         # default 10
    "kmax": 50.0,               # default 5.0
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
# ---- baryonic feedback methods (bfmt theory block) --------------------------

# One entry per feedback method the bfmt theory block implements:
# (label, theory-block options selecting it, fixed evaluation point
# for its sampled parameters). The SP(k) points are pyspk's
# documented examples; the emulator points are the fiducial values
# quoted in the example yamls. test_accuracy_baryons.py evaluates
# each method at the default and the pushed numerical settings.
BARYON_METHODS = [
    ("spk power law", {"baryon_model": 1, "spk_fb_model": 1},
     {"fb_a_spk": 0.4, "fb_pow_spk": 0.3}),
    ("spk akino", {"baryon_model": 1, "spk_fb_model": 2},
     {"alpha_spk": 4.189, "beta_spk": 1.273, "gamma_spk": 0.298}),
    ("spk double power law", {"baryon_model": 1, "spk_fb_model": 3},
     {"epsilon_spk": 0.3, "alpha_spk": 1.1, "beta_spk": 0.2,
      "gamma_spk": 0.5}),
    ("bcemu", {"baryon_model": 2},
     {"log10Mc_bcemu": 13.32, "mu_bcemu": 0.93, "thej_bcemu": 4.235,
      "gamma_bcemu": 2.25, "delta_bcemu": 6.40, "eta_bcemu": 0.15,
      "deta_bcemu": 0.14}),
    ("flamingo", {"baryon_model": 3},
     {"fgas_sigma_flamingo": 0.0, "mstar_sigma_flamingo": 0.0,
      "jet_frac_flamingo": 0.0}),
    ("baccoemu", {"baryon_model": 4},
     {"M_c_baccoemu": 14.0, "eta_baccoemu": -0.3,
      "beta_baccoemu": -0.22, "M1_z0_cen_baccoemu": 10.5,
      "theta_inn_baccoemu": -0.86}),
    ("bcemu2025", {"baryon_model": 5},
     {"Theta_co_bcemu25": 0.3, "log10Mc_bcemu25": 13.1,
      "mu_bcemu25": 1.0, "delta_bcemu25": 6.0, "eta_bcemu25": 0.10,
      "deta_bcemu25": 0.22, "Nstar_bcemu25": 0.028}),
]


def _baryon_method(label):
    """Look one BARYON_METHODS entry up by its label.

    Arguments:
      label = the first field of one BARYON_METHODS entry.

    Returns:
      the (label, theory options, parameter point) tuple.

    Raises:
      ValueError when label names no entry.
    """
    matches = [b for b in BARYON_METHODS if b[0] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown baryon method {label!r}")
    return matches[0]


ACCURACY_KNOBS = [
    ("accuracyboost 1->3", {"accuracyboost": 3.0}, {}),
    ("accuracyboost 1->5 (stress)", {"accuracyboost": 5.0}, {}),
    ("integration_accuracy 0->10", {"integration_accuracy": 10}, {}),
    ("lmax 50000->200000", {"lmax": 200000}, {}),
    ("kmax_boltzmann -> 40 + camb kmax -> 50",
     {"kmax_boltzmann": 40.0}, {"kmax": 50.0}),
    ("camb AccuracyBoost 1.05->2", {}, {"AccuracyBoost": 2.0}),
    ("camb k_per_logint 10->50", {}, {"k_per_logint": 50}),
]

# ---- random models ----------------------------------------------------------

# Base of the per-model random seeds of the N-random-models accuracy
# check (random_model_accuracy): model m draws its point from
# numpy.random.default_rng(RANDOM_MODEL_SEED + m). The value 137 is
# arbitrary; what matters is that it is fixed, so two runs (on any
# machine) evaluate the same points, and that each model has its own
# seed, so model m's point does not depend on how many models a run
# asks for.
RANDOM_MODEL_SEED = 137


# =============================================================================
# SECTION 2: ENVIRONMENT CHECKS
# =============================================================================
def require_cocoa_environment():
    """Refuse to run outside a started Cocoa shell, then move to ROOTDIR.

    start_cocoa.sh exports ROOTDIR (the absolute path of the Cocoa/
    folder) and prepares the library paths the compiled cosmolike
    interface needs. Without it, importing the likelihood would fail
    with a confusing linker error, so this check turns that failure
    into an instruction. The chdir matters because component paths in
    the frozen configuration (for example CAMB's
    ./external_modules/code/CAMB) are relative to ROOTDIR.

    Returns:
      nothing; on success the process working directory is ROOTDIR.

    Raises:
      RuntimeError telling the user to activate the cocoa environment
      and source start_cocoa.sh when ROOTDIR is not exported.
    """
    if "ROOTDIR" not in os.environ:
        raise RuntimeError(
            "ROOTDIR is not set. Activate the cocoa conda environment and run "
            "`source start_cocoa.sh` from the Cocoa/ folder before running "
            "these tests."
        )
    os.chdir(os.environ["ROOTDIR"])


def assert_omp_threads():
    """Refuse a race test that would not actually run multi-threaded.

    OpenMP reads OMP_NUM_THREADS once, when the compiled library is
    first loaded, so the value must be in the environment before any
    cobaya or cosmolike import. The test modules set it at their first
    line; this check catches a run that imported the stack some other
    way first (for example from an interactive session).

    Returns:
      nothing when OMP_NUM_THREADS equals REQUIRED_OMP_THREADS.

    Raises:
      RuntimeError naming the observed value and the required one.
    """
    # .get returns None when the variable is unset, where indexing
    # would raise a KeyError; None then fails the comparison below
    # with the same readable message. In that message, !r prints the
    # value as python source (None without quotes, a string with
    # them), telling unset apart from empty.
    observed = os.environ.get("OMP_NUM_THREADS")
    if observed != REQUIRED_OMP_THREADS:
        raise RuntimeError(
            f"OMP_NUM_THREADS={observed!r}; the race-condition tests require "
            f"OMP_NUM_THREADS={REQUIRED_OMP_THREADS} and it must be set "
            "before cobaya/cosmolike are imported."
        )


# =============================================================================
# SECTION 3: FROZEN-STATE INTEGRITY
# =============================================================================
def sha256_of(path):
    """Fingerprint one file with SHA-256.

    Arguments:
      path = absolute path of the file to hash.

    Returns:
      the 64-character lowercase hexadecimal SHA-256 digest of the
      file's bytes. Reading happens in 1 MiB blocks so the 80 MB
      covariance never sits in memory at once.
    """
    hasher = hashlib.sha256()
    # "rb" reads raw bytes; the with closes the file when the block
    # ends, even if reading raises
    with open(path, "rb") as f:
        # two-argument iter(callable, sentinel) calls the lambda
        # again and again until it returns b"" (end of file);
        # 1 << 20 is 2 to the 20th = 1 MiB, the size of each read
        for block in iter(lambda: f.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def compute_manifest():
    """Hash every file currently under tests/frozen/.

    __pycache__ folders and .pyc files are skipped: Python writes them
    as a side effect of importing the frozen modules, so hashing them
    would make the manifest fail after the first run. .DS_Store files
    (macOS Finder metadata) are skipped for the same reason.

    Returns:
      a dictionary {relative path: sha256 digest}, with paths relative
      to the tests/ folder using "/" separators, sorted by path so the
      manifest file is stable across platforms.
    """
    files = {}
    # os.walk visits frozen/ and every folder below it, handing back
    # (folder, subfolder names, file names) one folder at a time
    for base, dirs, names in os.walk(FROZEN_DIR):
        # the comprehension keeps every name except __pycache__, and
        # assigning to dirs[:] rewrites the list os.walk is holding
        # in place, so the walk never descends into the dropped folder
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(names):
            if name == ".DS_Store" or name.endswith(".pyc"):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, TESTS_DIR).replace(os.sep, "/")
            files[rel] = sha256_of(full)
    # sorted(files.items()) orders the (path, digest) pairs by path;
    # dict() rebuilds the table in that order (dicts remember
    # insertion order), so the manifest writes identically everywhere
    return dict(sorted(files.items()))


def verify_frozen():
    """Fail every test up front when the frozen state was edited.

    Compares the stored manifest with a fresh hash of tests/frozen/ in
    both directions, so an edited file (CHANGED), a deleted file
    (MISSING), and a new file (EXTRA) are all reported. This runs
    before any model is built: a tampered frozen state must not
    produce a plausible-looking chi2.

    Returns:
      nothing when every frozen file matches the manifest.

    Raises:
      AssertionError listing every mismatched path and pointing to
      generate_frozen_reference.py --overwrite for a deliberate
      refresh; AssertionError also when the manifest file itself is
      absent (the frozen state was never generated).
    """
    if not os.path.isfile(MANIFEST_FILE):
        raise AssertionError(
            "tests/manifest_sha256.json is missing; run "
            "generate_frozen_reference.py --overwrite to create the frozen "
            "test state."
        )
    # expected = the {relative path: sha256 digest} table written at
    # freeze time; it is the definition of "untouched". json.load
    # turns the file's JSON text back into nested dictionaries, and
    # the with closes the file once the block ends, error or not
    with open(MANIFEST_FILE) as f:
        expected = json.load(f)["files"]
    # actual = the same table computed from the files on disk right now
    # (compute_manifest walks frozen/ and fingerprints each file)
    actual = compute_manifest()
    # collect every discrepancy before raising: a report naming all
    # problem files at once beats failing on the first one
    problems = []
    # .items() hands out (path, digest) pairs, unpacked into the two
    # loop names
    for rel, digest in expected.items():
        if rel not in actual:
            # the manifest lists it but the file is gone from disk
            problems.append(f"MISSING  {rel}")
        elif actual[rel] != digest:
            # the file exists but at least one byte differs
            problems.append(f"CHANGED  {rel}")
    # both directions matter: a file ADDED to frozen/ is as suspicious
    # as an edited one, so the reverse scan runs too
    for rel in actual:
        if rel not in expected:
            problems.append(f"EXTRA    {rel}")
    if problems:
        # "\n  ".join(problems) glues the collected lines into one
        # indented block, one mismatch per line
        raise AssertionError(
            "Frozen test data does not match tests/manifest_sha256.json "
            "(someone edited the frozen copies; the tests refuse to run):\n  "
            + "\n  ".join(problems)
            + "\nIf the change is deliberate, regenerate with "
            "generate_frozen_reference.py --overwrite."
        )


def load_reference():
    """Read the frozen reference chi2 values.

    Returns:
      the dictionary stored in frozen/reference_chi2.json: one entry
      per configuration ("example1_nla", "example1_tatt",
      "example2_nla", "example2_tatt") plus a "_meta" entry recording
      when and how the references were generated. The file sits inside
      frozen/, so verify_frozen() also protects it from editing.
    """
    # json.load parses the file's JSON text back into the dictionary
    # json.dump wrote; the with closes the file on every exit
    with open(REFERENCE_FILE) as f:
        return json.load(f)


# =============================================================================
# SECTION 4: MODEL CONSTRUCTION AND EVALUATION (the chi2 pipeline)
# =============================================================================
# cobaya and numpy are imported inside the functions below, not at the
# top of this module. The reason is OpenMP: OMP_NUM_THREADS must be in
# the environment before the compiled libraries load, and it is the
# TEST modules that set it, on their first line, before importing this
# module's callers.
def _frozen_module(example):
    """Load one frozen configuration module from its file path.

    importlib is used instead of a plain import statement because the
    frozen modules live inside frozen/, which is data, not a package:
    it has no __init__.py and is never on sys.path. Loading by path
    also guarantees the file that verify_frozen() hashed is exactly
    the file being executed.

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES).

    Returns:
      the loaded module, carrying the attributes `yaml_string` (the
      complete configuration) and `point` (the frozen evaluation
      point).
    """
    import importlib.util

    path = os.path.join(FROZEN_DIR, EXAMPLES[example]["frozen_module"])
    # the three importlib steps mirror what `import` does under the
    # hood: build a loading recipe (spec) for this one file, make an
    # empty module from it, then run the file's code inside the
    # module so its top-level assignments become module attributes
    spec = importlib.util.spec_from_file_location(f"frozen_{example}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_info(example, tatt, fastpt=False, high_accuracy=False,
                     overrides=None, baryon=None):
    """Build the cobaya input dictionary for one frozen configuration.

    Starts from the frozen module's yaml string and applies the only
    three run-time adjustments the tests need:

      - the likelihood `path` is pointed at the absolute location of
        frozen/data on this machine (the frozen string stores the
        ROOTDIR-relative form, which would also resolve, but the
        absolute path is independent of the working directory);
      - `IA_model` selects the intrinsic-alignment model: 0 keeps NLA,
        1 selects TATT;
      - cobaya's log level is raised to WARNING (debug: 30) so the
        component-loading chatter does not bury the test reports.

    A fourth adjustment exists for the FASTPT comparison tests:
    `fastpt=True` sets `IA_code: 1` (the likelihood then asks the
    python FAST-PT package for the TATT perturbation-theory terms
    instead of the C implementation cfastpt built into cosmolike) and
    adds the fastpt theory block that computation requires. IA_code
    only matters under TATT, so fastpt=True is combined with
    tatt=True.

    Arguments:
      example = a key of EXAMPLES.
      tatt    = True selects the TATT IA model, False keeps NLA.
      fastpt  = True computes the TATT terms with python FAST-PT
                (IA_code: 1) instead of cfastpt (IA_code: 0).
      high_accuracy = True applies HIGH_ACCURACY_LIKELIHOOD and
                HIGH_ACCURACY_CAMB_EXTRA_ARGS on top of the frozen
                configuration (accuracy advisory checks only; not
                available for the emulator configurations, which have
                no camb block).
      overrides = None, or a pair (likelihood overrides, camb
                extra_args overrides) applied on top of the frozen
                configuration; single_model_chi2 builds this pair
                from one ACCURACY_KNOBS entry.

    Returns:
      the input dictionary ready for cobaya's get_model.
    """
    from cobaya.yaml import yaml_load

    cfg = EXAMPLES[example]
    # _frozen_module loads frozen/<frozen_module>.py by path and hands
    # back its yaml_string attribute: the complete configuration with
    # every option and parameter written out at freeze time
    info = yaml_load(_frozen_module(example).yaml_string)
    # the tests drive the model directly, so a sampler or output block
    # left in the info would only confuse cobaya. pop(key, None)
    # removes the key when present and does nothing (no error) when
    # the frozen configuration never had it
    info.pop("sampler", None)
    info.pop("output", None)
    # log level WARNING (30): component-loading chatter would bury the
    # test reports
    info["debug"] = 30
    info["timing"] = False
    likelihood_block = info["likelihood"][cfg["likelihood"]]
    # the frozen string stores the ROOTDIR-relative data path; the
    # absolute path is independent of the working directory
    likelihood_block["path"] = os.path.join(FROZEN_DIR, "data")
    # intrinsic-alignment model selection: 0 = NLA, 1 = TATT (the
    # ternary `1 if tatt else 0` reads: 1 when tatt is True, else 0)
    likelihood_block["IA_model"] = 1 if tatt else 0
    # cfg.get("emulator") is None when the key is absent, and None
    # counts as false: only the exact-physics entries take the branch
    if tatt and not cfg.get("emulator"):
        # TATT evaluates against its own generated data vector so the
        # chi2 sits at a minimum (see the TATT_DATASET comment)
        likelihood_block["data_file"] = TATT_DATASET
    if high_accuracy:
        if cfg.get("emulator"):
            raise ValueError(
                "high_accuracy applies to the exact-physics "
                "configurations only (the emulators have no accuracy "
                "knobs to push)")
        # .update copies every entry of the high-accuracy table into
        # the block, overwriting the frozen value of any shared key
        likelihood_block.update(HIGH_ACCURACY_LIKELIHOOD)
        info["theory"]["camb"]["extra_args"].update(
            HIGH_ACCURACY_CAMB_EXTRA_ARGS)
    if overrides is not None:
        # one knob at a time (an ACCURACY_KNOBS entry): the same
        # mechanism as high_accuracy, restricted to a single setting
        like_over, camb_over = overrides
        likelihood_block.update(like_over)
        info["theory"]["camb"]["extra_args"].update(camb_over)
    if fastpt:
        likelihood_block["IA_code"] = 1
        # the frozen configurations carry no fastpt block (the live
        # examples ship it commented out), so it is added here with the
        # same path the examples use. setdefault hands back the
        # existing "theory" dictionary, or first inserts the empty {}
        # and hands that back, so the assignment lands inside info
        # either way
        info.setdefault("theory", {})["fastpt"] = {
            "path": "./external_modules/code/FAST-PT",
        }
    if baryon is not None:
        _, theory_options, baryon_point = _baryon_method(baryon)
        # the likelihood requests the suppression product only when
        # this switch is on (see external_baryon_suppression in
        # likelihood/_cosmolike_prototype_base.py)
        likelihood_block["external_baryon_suppression"] = True
        # dict(a, **b) builds a new dictionary with a's entries plus
        # b's: python_path tells cobaya where the bfmt class lives
        info["theory"]["bfmt"] = dict(
            {"python_path": os.path.join(
                os.environ["ROOTDIR"], "external_modules", "code",
                "baryon_suppression")},
            **theory_options)
        for name, value in baryon_point.items():
            info["params"][name] = value
    return info


def make_model(info):
    """Build a cobaya model (theory plus likelihood, ready to evaluate).

    Arguments:
      info = a cobaya input dictionary from load_frozen_info.

    Returns:
      the cobaya Model. Building one loads CAMB and the compiled
      cosmolike interface and reads the frozen data files, which takes
      a few seconds; the callers print a progress line first.
    """
    from cobaya.model import get_model

    return get_model(info)


def load_frozen_point(example):
    """Read the frozen evaluation point of one example.

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES).

    Returns:
      a fresh {parameter name: value} dictionary (copied, so a caller
      may modify it without affecting later calls).
    """
    # dict(...) builds a NEW dictionary with the same entries: the
    # caller may edit the copy without touching the frozen module
    return dict(_frozen_module(example).point)


def build_point(model, example, tatt):
    """Assemble the exact point a test evaluates, with a safety check.

    The frozen point must cover the model's sampled parameters one to
    one. When likelihood or theory code changes its parameter set (a
    new nuisance parameter appears, or one is removed), evaluating
    would either fail cryptically or silently pick up a new default,
    so the mismatch is reported here by name instead.

    Arguments:
      model   = the cobaya Model the point will be evaluated on.
      example = "example1" or "example2" (a key of EXAMPLES).
      tatt    = True replaces the TATT_POINT values (nonzero A2/BTA)
                in the frozen point; False evaluates it unchanged.

    Returns:
      a {parameter name: value} dictionary covering every sampled
      parameter of the model.

    Raises:
      AssertionError listing the parameters that appeared or vanished
      when the model's sampled set differs from the frozen point;
      ValueError when a TATT parameter is not sampled by the model.
    """
    # load_frozen_point returns a copy of the frozen module's point:
    # the exact {parameter: value} table the references were computed at
    point = load_frozen_point(example)
    # sampled = the parameters THIS model, built from today's code,
    # expects to receive; the frozen point must cover them exactly.
    # set() collects the names for comparison by content, order
    # ignored, and set(point) is the set of the dictionary's KEYS
    sampled = set(model.parameterization.sampled_params())
    if sampled != set(point):
        # sampled - set(point) is set difference: the names in the
        # first set only, here the parameters the code gained; the
        # mirrored expression lists the ones it lost
        raise AssertionError(
            "sampled-parameter set differs from the frozen point (the "
            "likelihood/theory code changed its parameters):\n"
            f"  new since freeze: {sorted(sampled - set(point))}\n"
            f"  gone since freeze: {sorted(set(point) - sampled)}"
        )
    if tatt:
        for name, value in TATT_POINT.items():
            if name not in point:
                raise ValueError(f"TATT parameter {name} is not sampled")
            point[name] = value
    return point


def evaluate_chi2(model, point):
    """Evaluate one point and return the likelihood chi2.

    cached=False forces a full recomputation: the race tests evaluate
    the same point twice on one model, and letting cobaya return a
    cached value would compare a number with itself.

    Arguments:
      model = the cobaya Model to evaluate on.
      point = {parameter name: value} covering the sampled parameters.

    Returns:
      chi2 = -2 ln L of the single likelihood, as a plain float.

    Raises:
      RuntimeError when the model holds more than one likelihood
      (the -2*loglikes[0] extraction would then be ambiguous);
      AssertionError when the chi2 is not finite, which is how an
      out-of-prior or rejected point shows up.
    """
    import numpy as np

    # logposterior runs the full pipeline (theory + likelihood) at the
    # point; cached=False forces recomputation (see docstring)
    posterior = model.logposterior(point, cached=False)
    # loglikes = one ln L per likelihood component, in model order
    if len(posterior.loglikes) != 1:
        raise RuntimeError(
            f"expected exactly one likelihood: {posterior.loglikes}")
    chi2 = -2.0 * posterior.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    return float(chi2)


def draw_uniform_point(info, example, rng):
    """Draw one random sampled-parameter point across the priors.

    In this project every freely varied parameter carries a uniform
    box prior (a `prior` block with `min` and `max`, for example
    As_1e9 in [0.5, 5]), so drawing each of them uniformly across its
    box IS a draw from the prior. Two kinds of sampled parameters are
    not drawn and keep their frozen point value instead: those whose
    prior is a Gaussian rather than a box (`dist: norm`, no min/max;
    the photo-z shifts LSST_DZ_* and the shear calibrations LSST_M*),
    and any parameter without a prior block at all. Keeping their
    frozen values leaves the draw well inside those Gaussian priors
    while the box-prior parameters explore the full volume.

    Arguments:
      info    = the cobaya input dictionary of the frozen
                configuration (a load_frozen_info result); its params
                block holds every parameter's prior.
      example = a key of EXAMPLES; names the frozen point that fills
                the parameters no box prior covers.
      rng     = a numpy random Generator; the caller seeds it, which
                is what makes the draw reproducible.

    Returns:
      {parameter name: value} with the same names as the frozen
      point, every box-prior parameter replaced by a uniform draw.
    """
    # load_frozen_point returns a copy of the frozen evaluation
    # point; its keys are exactly the sampled parameters
    point = load_frozen_point(example)
    drawn = {}
    # sorted() fixes the order the rng is consumed in: dict order
    # would also be stable here (the generator writes the point
    # sorted), but reproducibility must not hang on that detail
    for name in sorted(point):
        spec = info["params"][name]
        prior = spec.get("prior")
        # `"min" in prior` asks whether the dictionary carries that
        # key; a prior dict holding min and max is a uniform box
        if isinstance(prior, dict) and "min" in prior and "max" in prior:
            # a uniform box prior: the draw is a prior sample
            drawn[name] = float(rng.uniform(prior["min"], prior["max"]))
        else:
            # Gaussian prior or no prior: keep the frozen value
            drawn[name] = point[name]
    return drawn


def random_model_accuracy(example, n_models, seed=RANDOM_MODEL_SEED):
    """Delta chi2 at N random prior points, each against its own vector.

    The A1-A6 checks of test_accuracy.py measure the numerical error
    of the default settings at ONE frozen fiducial point. This
    measures it at n_models random points across the prior instead.
    Per model m, in order:

      1. draw a point with draw_uniform_point, seeded with seed + m
         (see RANDOM_MODEL_SEED for why per model);
      2. generate a synthetic data vector AT that point: a
         DEFAULT-settings model with print_datavector enabled writes
         the full-length theory vector during its evaluation. The
         vector goes into a temporary directory, never into frozen/
         (the manifest pins every byte there, so a write into it
         would fail every later test). Because the default model
         itself produced the vector, the default-settings chi2
         against it is zero by construction;
      3. write a ".dataset" file for that vector into the same
         temporary directory. A ".dataset" file is the small text
         file a likelihood reads first: one "key = filename" line
         per ingredient (data_file = the data vector to fit, plus
         the covariance, n(z) tables, masks, baryon files). The one
         written here is the frozen ".dataset" text with a single
         change: its data_file line now names the new vector. The
         filenames in it are joined onto the likelihood's `path`
         option, so the temporary directory must look like a
         complete data folder: every file of frozen/data is
         symlinked in;
      4. evaluate a HIGH_ACCURACY model at the same point against the
         new descriptor. Since the vector is exact for the default
         settings, that chi2 IS
         delta chi2 = chi2(high accuracy) - chi2(default);
      5. remove the temporary directory, also when a step failed.

    Every configuration built here shares example2's data-vector
    dimensions, so building the models one after another inside one
    process is safe in this project. (Where configurations differ in
    dimensions, cosmolike's C globals keep the first size and each
    build would need its own process.)

    One report block per model streams as it finishes
    (report_random_model), so a long run shows progress; the caller
    receives the deltas for the finiteness assertion and the summary.

    Arguments:
      example  = a key of EXAMPLES (exact-physics configurations
                 only); test_accuracy.py uses "example2" (3x2pt, NLA).
      n_models = how many random points to evaluate. Each one costs a
                 default build+evaluation plus a high-accuracy
                 build+evaluation, minutes per model.
      seed     = base of the per-model rng seeds (seed + m);
                 RANDOM_MODEL_SEED unless a caller needs a second,
                 different reproducible set.

    Returns:
      the list of per-model delta chi2 values (floats), in model
      order.

    Raises:
      RuntimeError when print_datavector wrote no file, when the
      generated vector's line count differs from the frozen data
      vector's (a masking or probe mismatch: the covariance and the
      masks would select the wrong entries), or when the frozen
      descriptor does not contain exactly one data_file line;
      AssertionError when the drawn point and the model disagree on
      which parameters are sampled, meaning the likelihood or theory
      code gained or lost a sampled parameter since the freeze (the
      same mismatch build_point reports for the frozen point), or
      when a chi2 comes out non-finite (evaluate_chi2).
    """
    import numpy as np

    cfg = EXAMPLES[example]
    frozen_data_dir = os.path.join(FROZEN_DIR, "data")
    deltas = []
    for m in range(n_models):
        rng = np.random.default_rng(seed + m)
        # the DEFAULT-settings configuration; its params block also
        # supplies the prior boxes the draw reads
        info = load_frozen_info(example, tatt=False)
        likelihood_block = info["likelihood"][cfg["likelihood"]]
        point = draw_uniform_point(info, example, rng)
        # mkdtemp creates a fresh private directory; this model's
        # vector, descriptor, and symlinks live and die inside it
        workdir = tempfile.mkdtemp(prefix="cocoa_random_model_")
        try:
            # the likelihood joins path + filename for EVERY file a
            # descriptor names, so the temporary directory must look
            # like a complete data folder: symlink each frozen data
            # file in (a symlink reads as the original file)
            for name in sorted(os.listdir(frozen_data_dir)):
                os.symlink(os.path.join(frozen_data_dir, name),
                           os.path.join(workdir, name))
            # names absent from frozen/data, so writing them can
            # never follow a symlink back into the pinned folder
            vector_name = f"random_model_{m}.modelvector"
            descriptor_name = f"random_model_{m}.dataset"
            vector_path = os.path.join(workdir, vector_name)
            # step 2: the evaluation of the default model at the
            # drawn point is what makes cosmolike write the theory
            # vector; the chi2 of that evaluation (against the frozen
            # data vector) plays no role
            likelihood_block["print_datavector"] = True
            likelihood_block["print_datavector_file"] = vector_path
            print(f"  model {m + 1}/{n_models}: building the "
                  "default-settings model ...", flush=True)
            model = make_model(info)
            # Before evaluating, confirm the drawn point and the
            # model agree on WHICH parameters are sampled. The point's
            # names come from the frozen configuration; the model was
            # just built from today's code. If the code gained or lost
            # a sampled parameter since the freeze, evaluating would
            # either fail with a cryptic cobaya error or silently fill
            # the new parameter with a default value, so the mismatch
            # is reported by name instead (build_point makes this same
            # check for the frozen-point tests).
            sampled = set(model.parameterization.sampled_params())
            if sampled != set(point):
                raise AssertionError(
                    "sampled-parameter set differs from the frozen point "
                    "(the likelihood/theory code changed its parameters):\n"
                    f"  new since freeze: {sorted(sampled - set(point))}\n"
                    f"  gone since freeze: {sorted(set(point) - sampled)}"
                )
            print(f"  model {m + 1}/{n_models}: evaluating the drawn "
                  "point (writes the synthetic vector) ...", flush=True)
            evaluate_chi2(model, point)
            if not os.path.isfile(vector_path):
                raise RuntimeError(
                    f"print_datavector wrote no file at {vector_path}; "
                    "the likelihood accepted print_datavector_file but "
                    "produced nothing (did its path handling change?)")
            # the generated vector must be full length: the frozen
            # descriptor names the original vector, and that file's
            # line count is the definition of full length
            with open(vector_path) as f:
                # sum(1 for _ in f) walks the file line by line and
                # adds 1 per line: a line count that never loads the
                # whole file into memory (_ is the throwaway name
                # for the line itself)
                generated_lines = sum(1 for _ in f)
            with open(os.path.join(frozen_data_dir,
                                   likelihood_block["data_file"])) as f:
                # .read() with no size returns the whole file as one
                # string (the descriptors are small text files)
                descriptor = f.read()
            original_vector = None
            # splitlines() cuts the text into a list of lines with
            # the newline characters removed
            for line in descriptor.splitlines():
                if line.strip().startswith("data_file"):
                    # split("=", 1) cuts at the FIRST "=" only, so a
                    # value containing "=" survives whole; [1] is the
                    # part after the cut, and strip() drops the
                    # blanks around it
                    original_vector = line.split("=", 1)[1].strip()
            with open(os.path.join(frozen_data_dir, original_vector)) as f:
                # the same load-nothing line count as above
                original_lines = sum(1 for _ in f)
            if generated_lines != original_lines:
                raise RuntimeError(
                    f"model {m}: generated vector has {generated_lines} "
                    f"lines; the original {original_vector} has "
                    f"{original_lines}")
            # step 3: write the ".dataset" file for the synthetic
            # vector: the frozen ".dataset" text, unchanged except for
            # one line, so the likelihood reads the same covariance,
            # n(z), and masks but fits the vector from step 2 instead
            # of the shipped measurement
            replaced = 0
            out_lines = []
            # keepends=True leaves the newline on the end of every
            # line, so joining the pieces rebuilds the file byte for
            # byte and only the retyped line differs
            for line in descriptor.splitlines(keepends=True):
                if line.strip().startswith("data_file"):
                    out_lines.append(f"data_file = {vector_name}\n")
                    replaced += 1
                else:
                    out_lines.append(line)
            if replaced != 1:
                raise RuntimeError(
                    f"{likelihood_block['data_file']}: expected exactly "
                    f"one data_file line, found {replaced}")
            with open(os.path.join(workdir, descriptor_name), "w") as f:
                # "".join(out_lines) glues the list into one string
                # with nothing between the pieces; each piece still
                # ends in its own newline
                f.write("".join(out_lines))
            # step 4: the high-accuracy model reads the temporary
            # directory as its data folder and the new descriptor as
            # its dataset; at the same point its chi2 is the delta
            info_high = load_frozen_info(example, tatt=False,
                                         high_accuracy=True)
            block_high = info_high["likelihood"][cfg["likelihood"]]
            block_high["path"] = workdir
            block_high["data_file"] = descriptor_name
            print(f"  model {m + 1}/{n_models}: building the "
                  "high-accuracy model ...", flush=True)
            model_high = make_model(info_high)
            print(f"  model {m + 1}/{n_models}: evaluating the same "
                  "point at high accuracy ...", flush=True)
            chi2_high = evaluate_chi2(model_high, point)
        finally:
            # a finally block runs on EVERY exit from the try: after
            # success and while an exception is on its way out alike,
            # so no failure mode leaves the directory behind to pile
            # up across runs
            shutil.rmtree(workdir, ignore_errors=True)
        report_random_model(m, n_models, point, chi2_high)
        deltas.append(chi2_high)
    return deltas


# =============================================================================
# SECTION 5: TEST QUANTITIES (what the test methods call)
# =============================================================================
def single_model_chi2(example, tatt, fastpt=False, high_accuracy=False,
                      knob=None, baryon=None):
    """chi2 of the frozen fiducial point on a freshly built model.

    This is the quantity tests 1, 3, 5, and 7 compare against the
    frozen reference, and the quantity the generator stores as that
    reference. It chains the Section 4 pipeline: load_frozen_info,
    make_model, build_point, evaluate_chi2.

    Arguments:
      example = a key of EXAMPLES.
      tatt    = True evaluates the TATT variant, False the NLA one.
      fastpt  = True computes the TATT terms with python FAST-PT
                (see load_frozen_info).
      high_accuracy = True evaluates with the pushed numerical
                settings (see load_frozen_info); expect the
                evaluation to take minutes instead of seconds.
      knob    = None, or the label of one ACCURACY_KNOBS entry; that
                knob's overrides are applied alone (the one-at-a-time
                scan of test_accuracy.py).

    Returns:
      the chi2 as a float.

    Raises:
      ValueError when knob names no ACCURACY_KNOBS entry.
    """
    # the ternary a if c else b picks "TATT" when tatt is True and
    # "NLA" otherwise; the label only feeds the progress line
    ia_label = "TATT" if tatt else "NLA"
    if fastpt:
        ia_label += "+FASTPT"
    if high_accuracy:
        ia_label += ", high accuracy"
    overrides = None
    if knob is not None:
        # knob = a label from ACCURACY_KNOBS; the comprehension keeps
        # the entries whose label matches, so a right name yields a
        # one-entry list and a wrong one an empty list
        matches = [k for k in ACCURACY_KNOBS if k[0] == knob]
        if len(matches) != 1:
            raise ValueError(f"unknown accuracy knob {knob!r}")
        overrides = (matches[0][1], matches[0][2])
        ia_label += f", knob: {knob}"
    if baryon is not None:
        ia_label += f", baryons: {baryon}"
    print(f"  building model ({example}, {ia_label}) ...", flush=True)
    # load_frozen_info returns the frozen configuration dictionary with
    # the run-time adjustments applied; make_model turns it into an
    # evaluable cobaya Model (loads CAMB or the emulators + cosmolike)
    info = load_frozen_info(example, tatt, fastpt=fastpt,
                            high_accuracy=high_accuracy,
                            overrides=overrides, baryon=baryon)
    model = make_model(info)
    # build_point returns the frozen evaluation point after checking
    # that the point and the model name the same sampled parameters:
    # if the likelihood or theory code gained or lost a sampled
    # parameter since the freeze, the mismatch is reported by name
    # instead of failing deep inside cobaya
    point = build_point(model, example, tatt)
    print("  evaluating the fiducial point ...", flush=True)
    if baryon is not None:
        # With the feedback on, a non-finite chi2 means the method
        # REJECTED the frozen fiducial (a training-box violation; the
        # warning above names the offending parameter). The B-checks
        # report that as documented behavior, so hand back None
        # instead of dying on the assertion.
        try:
            return evaluate_chi2(model, point)
        except AssertionError:
            return None
    return evaluate_chi2(model, point)


def ten_in_a_row_chi2(example, tatt):
    """Race check: the fiducial evaluated fresh and as 10th of a row.

    On ONE model instance, in order: the fiducial point (the fresh
    value), then the nine RACE_PERTURBATIONS cosmologies, then the
    fiducial again as the 10th point of the row. State leaked between
    evaluations, or an OpenMP race under REQUIRED_OMP_THREADS threads,
    shifts the second fiducial value away from the first; correct
    code reproduces it to float noise. Each evaluation prints its
    chi2, so a stuck or slow run is visible line by line.

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES).
      tatt    = True runs the TATT variant, False the NLA one.

    Returns:
      (fresh, tenth): chi2 of the first fiducial evaluation and chi2
      of the fiducial as the 10th point of the row, both floats.
    """
    # ternary: "TATT" when tatt is True, "NLA" otherwise
    ia_label = "TATT" if tatt else "NLA"
    print(f"  building model ({example}, {ia_label}) ...", flush=True)
    # one model instance for the whole sequence: sharing the instance
    # is the point, since leaked state lives inside it
    info = load_frozen_info(example, tatt)
    model = make_model(info)
    point = build_point(model, example, tatt)
    # the fresh value: the fiducial evaluated before anything else
    # touched this model instance
    fresh = evaluate_chi2(model, point)
    # the format spec :.8f prints fixed-point with eight decimals,
    # enough to see a float-noise difference against the 1e-4 band
    print(f"  fresh model, fiducial point:  chi2 = {fresh:.8f}", flush=True)
    # enumerate pairs each perturbation with a counter; start=1 makes
    # the printed rows read 1..9 instead of 0..8
    for i, perturbation in enumerate(RACE_PERTURBATIONS, start=1):
        # the EMUL2 configurations sample fewer parameters than the
        # exact ones (mnu is fixed inside the emulator training), so a
        # perturbation key the model does not sample is dropped rather
        # than kept in a separate perturbation table per configuration;
        # the dict comprehension rebuilds the table with only the
        # keys the point carries
        applied = {k: v for k, v in perturbation.items() if k in point}
        # {**point, **applied} builds a NEW dict: point's entries
        # first, then applied's on top of any shared key; point
        # itself stays untouched for the final fiducial evaluation
        chi2 = evaluate_chi2(model, {**point, **applied})
        # join feeds on a generator: one "name=value" string per
        # changed parameter, glued with ", " between them
        changed = ", ".join(f"{k}={v}" for k, v in applied.items())
        # {i:2d} pads the counter to two characters so the rows line
        # up; :.4f prints four fixed decimals
        print(f"  row {i:2d}/10 ({changed}):  chi2 = {chi2:.4f}", flush=True)
    tenth = evaluate_chi2(model, point)
    print(f"  row 10/10 (fiducial again):  chi2 = {tenth:.8f}", flush=True)
    return fresh, tenth


# =============================================================================
# SECTION 6: TERMINAL REPORTS
# =============================================================================
# Each printer returns the difference it printed; the assertion on
# that difference lives in the calling test method, not here.
def report_chi2_test(number, label, chi2, ref, tol):
    """Print one reference-comparison test as a readable block.

    A bare pytest PASSED does not say what was compared, so each test
    prints its own numbers: the freshly computed chi2, the frozen
    reference, their absolute difference, and the limit the assertion
    uses. flush=True makes the block appear immediately (pytest runs
    with -s, so nothing buffers it).

    Arguments:
      number = the test number (1-8) shown in the header.
      label  = one line naming the example, probe, and IA model.
      chi2   = the chi2 computed in this run.
      ref    = the frozen reference chi2.
      tol    = the pass limit on |chi2 - ref| (CHI2_TOLERANCE).

    Returns:
      |chi2 - ref|, the printed difference.
    """
    delta = abs(chi2 - ref)
    # in the f-string: '-' * 66 repeats the dash into a 66-character
    # rule, :.6f prints fixed six decimals, and the a-if-c-else-b at
    # the arrow picks the verdict word from the comparison
    print(f"""
{'-' * 66}
TEST {number}: {label}
  chi2 (this run)     = {chi2:.6f}
  frozen reference    = {ref:.6f}
  |delta chi2|        = {delta:.6f}   (limit: < {tol})
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_race_test(number, label, fresh, tenth, tol):
    """Print one race-condition test as a readable block.

    Arguments:
      number = the test number (1-8) shown in the header.
      label  = one line naming the example, probe, and IA model.
      fresh  = chi2 of the fiducial point evaluated first on the model.
      tenth  = chi2 of the same point evaluated as the 10th of a row.
      tol    = the pass limit on |tenth - fresh| (RACE_TOLERANCE).

    Returns:
      |tenth - fresh|, the printed difference.
    """
    delta = abs(tenth - fresh)
    # :.8f = eight fixed decimals; the 1e-4 race band needs more
    # digits than the six the reference blocks print
    print(f"""
{'-' * 66}
TEST {number}: {label}
  fresh-model chi2    = {fresh:.8f}
  10th of 10 in a row = {tenth:.8f}
  |delta chi2|        = {delta:.8f}   (limit: < {tol})
  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_fastpt_test(number, label, chi2, ref, cfastpt_ref, tol):
    """Print one FASTPT comparison test as a readable block.

    Two numbers matter here: how far the FASTPT chi2 moved from its
    own frozen reference (the pass/fail criterion, same rule as every
    other reference test), and the physical difference between the
    FASTPT and cfastpt implementations of the TATT terms at the same
    point (informational; it was measured and saved at freeze time).

    Arguments:
      number      = the test number shown in the header.
      label       = one line naming the example and IA model.
      chi2        = the FASTPT chi2 computed in this run.
      ref         = the frozen FASTPT reference chi2.
      cfastpt_ref = the frozen cfastpt (IA_code: 0) reference chi2.
      tol         = the pass limit on |chi2 - ref| (CHI2_TOLERANCE).

    Returns:
      |chi2 - ref|, the printed pass/fail difference.
    """
    delta = abs(chi2 - ref)
    # :+.6f = fixed six decimals with the + forcing a sign, so the
    # direction of the FASTPT-minus-CFASTPT difference always shows
    print(f"""
{'-' * 66}
TEST {number}: {label}
  chi2 (this run, FASTPT)  = {chi2:.6f}
  frozen FASTPT reference  = {ref:.6f}
  |delta chi2|             = {delta:.6f}   (limit: < {tol})
  frozen CFASTPT reference = {cfastpt_ref:.6f}
  FASTPT - CFASTPT         = {chi2 - cfastpt_ref:+.6f}   (informational)
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_emul2_advisory(label, chi2, frozen_ref, exact_ref, limit):
    """Print one EMUL2 accuracy check: measurements and a recommendation.

    There is no pass/fail here. An emulator is an approximation, so
    the useful outputs are the numbers themselves: the change against
    the frozen emulator reference (nonzero: the installed emulator no
    longer reproduces the chi2 it gave at freeze time),
    the difference against the exact-physics chi2 at the same
    cosmology (how accurate the emulator is), and the recommendation
    derived from that accuracy.

    Arguments:
      label      = one line naming the emulated configuration.
      chi2       = the emulator chi2 computed in this run.
      frozen_ref = the frozen emulator reference chi2.
      exact_ref  = the exact-physics reference chi2 (from the matching
                   example's frozen reference).
      limit      = the recommendation threshold on |chi2 - exact_ref|.

    Returns:
      |chi2 - exact_ref|, the accuracy difference the recommendation
      is based on.
    """
    drift = chi2 - frozen_ref
    delta_exact = abs(chi2 - exact_ref)
    if delta_exact < limit:
        verdict = "RECOMMENDED for actual data analysis"
    else:
        verdict = ("NOT recommended for actual data analysis "
                   f"(|delta chi2| >= {limit})")
    # the + in {drift:+.6f} forces a sign: the direction of the
    # drift matters as much as its size
    print(f"""
{'-' * 66}
EMUL2 ADVISORY: {label}
  chi2 (this run, emulator)   = {chi2:.6f}
  frozen emulator reference   = {frozen_ref:.6f}  (drift {drift:+.6f})
  exact-physics reference     = {exact_ref:.6f}
  |emulator - exact| chi2     = {delta_exact:.6f}   (threshold: {limit})
  -> {verdict}
{'-' * 66}""", flush=True)
    return delta_exact


def report_emul2_race(label, fresh, tenth):
    """Print one EMUL2 race check, advisory only.

    Arguments:
      label = one line naming the emulated configuration.
      fresh = chi2 of the point evaluated first on the model.
      tenth = chi2 of the same point as the 10th of a row.

    Returns:
      |tenth - fresh|, the printed difference. A value above
      RACE_TOLERANCE is flagged as a possible race or state leak, but
      nothing fails: this file only alerts.
    """
    delta = abs(tenth - fresh)
    # ternary: the note reads "consistent" below the race band and
    # the warning at or above it; nothing asserts either way
    note = ("consistent" if delta < RACE_TOLERANCE
            else "WARNING: possible race condition or state leak")
    print(f"""
{'-' * 66}
EMUL2 ADVISORY: {label}
  fresh-model chi2    = {fresh:.8f}
  10th of 10 in a row = {tenth:.8f}
  |delta chi2|        = {delta:.8f}   ({note})
  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}
{'-' * 66}""", flush=True)
    return delta


def report_accuracy(label, chi2_high, default_ref,
                    default_name="default, frozen"):
    """Print one default-vs-high-accuracy check. Advisory only.

    The default-settings chi2 is the frozen reference (recorded at
    freeze time); the high-accuracy chi2 is computed in this run. The
    difference is the numerical error of the default settings at this
    point: there is no pass/fail because how much numerical error an
    analysis tolerates is a judgment call, not a fixed bound.

    Arguments:
      label       = one line naming the probe and IA model.
      chi2_high   = chi2 with HIGH_ACCURACY settings, this run.
      default_ref = the frozen default-settings reference chi2.

    Returns:
      chi2_high - default_ref, the printed difference.
    """
    delta = chi2_high - default_ref
    line_high = "  chi2 (high accuracy)".ljust(28)
    line_default = f"  chi2 ({default_name})".ljust(28)
    line_delta = "  delta chi2 (high-default) "
    print(f"""
{'-' * 66}
ACCURACY: {label}
{line_high}= {chi2_high:.6f}
{line_default}= {default_ref:.6f}
{line_delta}= {delta:+.6f}
{'-' * 66}""", flush=True)
    return delta


def report_knob(label, chi2, default_ref):
    """Print one entry of the one-knob-at-a-time scan. Advisory only.

    Arguments:
      label       = the ACCURACY_KNOBS entry evaluated.
      chi2        = chi2 with only that knob changed, this run.
      default_ref = the frozen default-settings reference chi2.

    Returns:
      chi2 - default_ref, the printed difference.
    """
    delta = chi2 - default_ref
    # :30s pads the label to 30 characters so the scan lines land in
    # columns; :12.6f = six decimals in a 12-wide field, and the +
    # variant forces a sign on the delta
    print(f"  KNOB {label:30s} chi2 = {chi2:12.6f}  "
          f"delta = {delta:+12.6f}", flush=True)
    return delta


def report_random_model(index, n_models, point, chi2_high):
    """Print one N-random-models block. Advisory only.

    No subtraction happens here: the synthetic data vector was
    generated by the default-settings model at this exact point, so
    the default chi2 against it is zero by construction and the
    high-accuracy chi2 already IS
    delta chi2 = chi2(high accuracy) - chi2(default). The three
    printed parameters locate the point inside the prior at a glance;
    the full point is reproducible from RANDOM_MODEL_SEED + index.

    Arguments:
      index     = the model counter, 0-based (printed 1-based).
      n_models  = how many models the run evaluates in total.
      point     = the drawn {parameter name: value} point.
      chi2_high = chi2 with the HIGH_ACCURACY settings against the
                  point's own synthetic vector (= the delta).

    Returns:
      chi2_high, the printed delta.
    """
    # inside the triple-quoted f-string, a backslash at the end of a
    # source line eats the newline: As_1e9, omegam, and ns print on
    # one line
    print(f"""
{'-' * 66}
RANDOM MODEL {index + 1}/{n_models}
  As_1e9 = {point['As_1e9']:.6f}   omegam = {point['omegam']:.6f}   \
ns = {point['ns']:.6f}
  delta chi2 = chi2(high accuracy) = {chi2_high:+.6f}
{'-' * 66}""", flush=True)
    return chi2_high


def report_random_model_summary(deltas):
    """Print the spread of the N-random-models deltas. Advisory only.

    The median rather than the mean: one point near a prior edge can
    carry a delta far above the rest, and the median keeps the
    typical numerical error readable next to that outlier (which the
    max still shows).

    Arguments:
      deltas = the per-model delta chi2 list from
               random_model_accuracy, one float per model.

    Returns:
      (minimum, median, maximum) of the deltas as floats.
    """
    import numpy as np

    lowest = float(np.min(deltas))
    middle = float(np.median(deltas))
    highest = float(np.max(deltas))
    print(f"  RANDOM MODELS SUMMARY ({len(deltas)} models): delta chi2 "
          f"min = {lowest:+.6f}  median = {middle:+.6f}  "
          f"max = {highest:+.6f}", flush=True)
    return lowest, middle, highest
