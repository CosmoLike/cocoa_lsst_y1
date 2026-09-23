"""Command line options for these tests (bound from cosmolike_core).

pytest requires a conftest.py inside each project's tests folder (it
discovers the file by walking up from the collected tests), so this
file cannot move; its content is the shared implementation in
cosmolike_core/cocoa_testing.py, bound here. This project's --mask
choices come from its FASTPT_MASK_DATASETS table: the frozen M1
mask, the M2-M6 variants, and the all-ones mask.
"""

import os
import sys

# The tests folder first (cocoa_test_utils), then cosmolike_core
# (cocoa_testing), both computed from this file's location so the
# imports work no matter where pytest was launched from. This
# project does not use the shared harness, so the core path is added
# here rather than by the shim.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(
    _TESTS_DIR, "..", "..", "..", "external_modules", "code",
    "cosmolike_core")))
import cocoa_test_utils as u
import cocoa_testing as _cct


def pytest_addoption(parser):
    """Register --high and --mask on pytest's parser.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_addoption. tuple(...) of the mask table
    hands over its keys in insertion order, "frozen" first.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the options become readable through config.getoption.
    """
    _cct.conftest_addoption(parser, tuple(u.FASTPT_MASK_DATASETS))


def pytest_configure(config):
    """Copy the option values where the test classes read them.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_configure.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains the variables.
    """
    _cct.conftest_configure(config)
