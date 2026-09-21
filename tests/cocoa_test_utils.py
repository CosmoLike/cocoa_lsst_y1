"""Shared harness for the lsst_y1 unit tests.

The tests never read ./data or the live EXAMPLE_EVALUATE yaml files: they run
on the byte-frozen copies stored in tests/frozen/, whose SHA-256 hashes are
pinned in tests/manifest_sha256.json. If any frozen file changes, every test
fails before evaluating anything. To refresh the frozen state deliberately,
run generate_frozen_reference.py --overwrite (maintainers only).

Requirements to run: the cocoa conda environment active and
`source start_cocoa.sh` done (ROOTDIR must be exported).
"""

import hashlib
import json
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

REQUIRED_OMP_THREADS = "2"

# Tests 1, 3, 5, 7: |chi2 - frozen reference| must stay below this
CHI2_TOLERANCE = 0.2
# Tests 2, 4, 6, 8: |chi2(10th of a 10-in-a-row run) - chi2(fresh model)|
RACE_TOLERANCE = 1.0e-4

# TATT point requested for tests 3, 4, 7, 8 (on top of the example override)
TATT_POINT = {
    "LSST_A2_1": 0.05,
    "LSST_BTA_1": 0.05,
    "LSST_A2_2": -1.51541,
}

EXAMPLES = {
    "example1": {
        "yaml": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "lsst_y1.cosmic_shear",
    },
    "example2": {
        "yaml": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "lsst_y1.combo_3x2pt",
    },
}

# Nine deterministic cosmologies evaluated before the fiducial point in the
# race tests (all inside the priors of the example yaml files)
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


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
def require_cocoa_environment():
    if "ROOTDIR" not in os.environ:
        raise RuntimeError(
            "ROOTDIR is not set. Activate the cocoa conda environment and run "
            "`source start_cocoa.sh` from the Cocoa/ folder before running "
            "these tests."
        )
    # cobaya component paths in the yaml files (e.g. CAMB) are ROOTDIR-relative
    os.chdir(os.environ["ROOTDIR"])


def assert_omp_threads():
    n = os.environ.get("OMP_NUM_THREADS")
    if n != REQUIRED_OMP_THREADS:
        raise RuntimeError(
            f"OMP_NUM_THREADS={n!r}; the race-condition tests require "
            f"OMP_NUM_THREADS={REQUIRED_OMP_THREADS} and it must be set "
            "before cobaya/cosmolike are imported."
        )


# -----------------------------------------------------------------------------
# Frozen-state integrity
# -----------------------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compute_manifest():
    files = {}
    for base, _, names in os.walk(FROZEN_DIR):
        for name in sorted(names):
            if name == ".DS_Store":
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, TESTS_DIR).replace(os.sep, "/")
            files[rel] = sha256_of(full)
    return dict(sorted(files.items()))


def verify_frozen():
    if not os.path.isfile(MANIFEST_FILE):
        raise AssertionError(
            "tests/manifest_sha256.json is missing; run "
            "generate_frozen_reference.py --overwrite to create the frozen "
            "test state."
        )
    with open(MANIFEST_FILE) as f:
        expected = json.load(f)["files"]
    actual = compute_manifest()
    problems = []
    for rel, sha in expected.items():
        if rel not in actual:
            problems.append(f"MISSING  {rel}")
        elif actual[rel] != sha:
            problems.append(f"CHANGED  {rel}")
    for rel in actual:
        if rel not in expected:
            problems.append(f"EXTRA    {rel}")
    if problems:
        raise AssertionError(
            "Frozen test data does not match tests/manifest_sha256.json "
            "(someone edited the frozen copies; the tests refuse to run):\n  "
            + "\n  ".join(problems)
            + "\nIf the change is deliberate, regenerate with "
            "generate_frozen_reference.py --overwrite."
        )


def load_reference():
    with open(REFERENCE_FILE) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Model construction and evaluation (imports cobaya lazily so that
# OMP_NUM_THREADS can be set by the caller first)
# -----------------------------------------------------------------------------
def load_frozen_info(example, tatt):
    from cobaya.yaml import yaml_load_file

    cfg = EXAMPLES[example]
    info = yaml_load_file(os.path.join(FROZEN_DIR, cfg["yaml"]))
    override = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler", None)
    info.pop("output", None)
    info["debug"] = False
    info["timing"] = False
    like = info["likelihood"][cfg["likelihood"]]
    # point the likelihood at the frozen data copy, not the live ./data
    like["path"] = os.path.join(FROZEN_DIR, "data")
    like["IA_model"] = 1 if tatt else 0  # NLA (0) or TATT (1)
    return info, override


def make_model(info):
    from cobaya.model import get_model

    return get_model(info)


def build_point(model, info, override, tatt):
    point = {}
    for p in model.parameterization.sampled_params():
        if p in override:
            point[p] = override[p]
        else:  # fall back on the ref center declared in the yaml
            ref = info["params"][p].get("ref")
            point[p] = ref["loc"] if isinstance(ref, dict) else ref
            if point[p] is None:
                raise ValueError(f"parameter {p}: not in override and no ref")
    if tatt:
        for p, v in TATT_POINT.items():
            if p not in point:
                raise ValueError(f"TATT parameter {p} is not sampled")
            point[p] = v
    return point


def evaluate_chi2(model, point):
    import numpy as np

    lp = model.logposterior(point, cached=False)
    if len(lp.loglikes) != 1:
        raise RuntimeError(f"expected exactly one likelihood: {lp.loglikes}")
    chi2 = -2.0 * lp.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    return float(chi2)


def single_model_chi2(example, tatt):
    """chi2 of the example's fiducial point on a freshly built model."""
    info, override = load_frozen_info(example, tatt)
    model = make_model(info)
    point = build_point(model, info, override, tatt)
    return evaluate_chi2(model, point)


def ten_in_a_row_chi2(example, tatt):
    """(fresh chi2, chi2 of the fiducial as 10th of 10 evaluations in a row).

    On one model instance: evaluate the fiducial point (fresh), then nine
    perturbed cosmologies, then the fiducial again as the 10th point of the
    row. Any state leaked between evaluations (or an OpenMP race with
    OMP_NUM_THREADS=2) shifts the second fiducial chi2.
    """
    info, override = load_frozen_info(example, tatt)
    model = make_model(info)
    point = build_point(model, info, override, tatt)
    fresh = evaluate_chi2(model, point)
    for pert in RACE_PERTURBATIONS:
        evaluate_chi2(model, {**point, **pert})
    tenth = evaluate_chi2(model, point)
    return fresh, tenth
