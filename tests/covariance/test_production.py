"""Check the production interface against the notebook calculation.

Small grids exercise both measurement spaces and one/eight-thread execution.
The project catalogs and all internal field pairs remain present; these
checks establish API equivalence, not survey numerical convergence.
"""

from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"covariance"))

import cosmolike_lsst_y1_interface as ci
import lsst_y1_covariance as survey
from cocoa_covariance_testing import check_project_forecast


def test_production_matches_notebook(tmp_path):
    """Both entry points retain the same components, axes and saved settings."""
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(1560, 675),
        directory=tmp_path,
    )


def test_production_array_ownership_and_layout():
    """The direct boundary borrows C-order inputs and returns independent values."""
    import numpy as np
    import pytest

    left = np.arange(24, dtype=np.float64).reshape(4, 6)/10.0
    right = np.arange(18, dtype=np.float64).reshape(3, 6)/7.0
    weight = np.linspace(0.2, 1.0, 6)
    saved = left.copy()
    left.flags.writeable = False
    result = ci.covariance.covariance_project(
        left=left, right=right, weight=weight,
    )
    expected = (left*weight) @ right.T
    np.testing.assert_allclose(result, expected, rtol=5.e-16)
    np.testing.assert_array_equal(left, saved)
    ci.covariance.covariance_project(left=right, right=left, weight=weight)
    np.testing.assert_allclose(result, expected, rtol=5.e-16)
    assert result.flags.c_contiguous
    assert result.flags.owndata

    # No hidden layout copy in production. Notebook wrappers still accept
    # Fortran and sliced arrays; callers of this direct path must pack them.
    with pytest.raises(TypeError):
        ci.covariance.covariance_project(
            left=np.asfortranarray(left), right=right, weight=weight,
        )
