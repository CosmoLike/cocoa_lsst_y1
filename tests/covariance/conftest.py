"""Locate this project's compiled interface for covariance-only pytest runs."""

from pathlib import Path
import sys

# Resolve from this file, so the documented command works from Cocoa and
# from tests/. The ordinary project build owns the imported extension.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

# Match the CAMB source path requested by the project's Cobaya likelihood.
# Otherwise a covariance test importing the installed package first makes
# a later data-vector test reject its location, even with identical physics.
sys.path.insert(0, str(project.parents[1]/"external_modules/code/CAMB"))

import pytest
import cosmolike_lsst_y1_interface as ci


# autouse=True applies the fixture to every test of this folder without
# the tests naming it; scope="session" runs it once per pytest run
@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip this sector when covariance generation was intentionally omitted.

    The compiled module exposes has_covariance = True only when it was
    built with the covariance code; otherwise every covariance test is
    skipped with the rebuild instruction.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_LSST_Y1_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
