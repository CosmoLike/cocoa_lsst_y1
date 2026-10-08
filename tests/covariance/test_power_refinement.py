"""Check the covariance-only preparation of globally dense power tables.

The preparation interpolates existing CAMB log powers; it must not change
redshifts, distances or the original samples. These small array tests check
nested nodes, field/axis conventions and input ownership without running
CAMB or a covariance calculation. Survey tests separately exercise the
prepared arrays through production and notebook interfaces.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from cosmolike_notebook_utils.covariance.power import refine_power_tables


@pytest.fixture
def tables():
    """Return distinct linear, nonlinear and cb rows in CAMB's interchange layout.

    Each field has a different shape in k and redshift. Using unequal axis
    lengths exposes accidental transposes or replacing cb with matter power.
    The actual halo prescription is irrelevant to this interpolation test.
    """
    # ln P tables on 3 redshifts x 7 log10 k nodes, flattened in Fortran
    # order (ravel(order="F"): redshift index fastest), the layout
    # set_cosmology reads; np.log1p(z) = ln(1 + z)
    log10k = np.linspace(-4.0, 1.0, 7)
    redshift = np.array([0.0, 0.5, 2.0])
    growth = -2*np.log1p(redshift[:, None])
    linear = 8+0.9*log10k[None, :]-0.12*log10k[None, :]**2+growth
    nonlinear = linear+0.3*np.exp(log10k[None, :]/2)/(1+redshift[:, None])
    cb = linear+0.02*np.sin(log10k[None, :])*(1+redshift[:, None])
    return {
        "log10k_2D": log10k,
        "z_2D": redshift,
        "lnP_linear": linear.ravel(order="F"),
        "lnP_nonlinear": nonlinear.ravel(order="F"),
        "lnP_linear_cb": cb.ravel(order="F"),
        "G": np.array([1.0, 1.1, 1.2, 1.3]),
        "z_G": np.array([0.0, 0.5, 1.0, 2.0]),
        "z_1D": np.array([0.0, 0.5, 1.0, 2.0]),
        "chi": np.array([0.0, 1200.0, 2400.0, 3600.0]),
        "omegan2": 0.0,
    }


def test_refinement_one_preserves_every_input(tables):
    """Disabling densification retains the original arrays and scalar values."""
    result = refine_power_tables(tables=tables, refinement=1)
    assert set(result) == set(tables)
    for name in tables:
        np.testing.assert_array_equal(result[name], tables[name])


@pytest.mark.parametrize("refinement", [8, 16])
def test_nested_natural_cubic_all_fields(tables, refinement):
    """Every original node survives, and only the inserted powers use cubics."""
    original_k = tables["log10k_2D"]
    original_nk = len(original_k)
    nz = len(tables["z_2D"])
    result = refine_power_tables(tables=tables, refinement=refinement)
    dense_k = result["log10k_2D"]
    # refinement r splits each k interval into r: r (n - 1) + 1 nodes,
    # and every r-th node ([::refinement]) is an original node
    assert len(dense_k) == refinement*(original_nk-1)+1
    np.testing.assert_array_equal(dense_k[::refinement], original_k)
    for name in ("lnP_linear", "lnP_nonlinear", "lnP_linear_cb"):
        original = tables[name].reshape((nz, original_nk), order="F")
        actual = result[name].reshape((nz, len(dense_k)), order="F")
        spline = CubicSpline(
            x=original_k, y=original, axis=1, bc_type="natural",
        )
        expected = spline(dense_k)
        np.testing.assert_array_equal(actual[:, ::refinement], original)
        np.testing.assert_allclose(actual, expected, rtol=2.e-15, atol=2.e-15)
    for name in ("z_2D", "G", "z_G", "z_1D", "chi", "omegan2"):
        np.testing.assert_array_equal(result[name], tables[name])


def test_refinement_keeps_inputs_and_boost_eight_nodes(tables):
    """Repeated preparation neither mutates its inputs nor moves coarser nodes."""
    before = {}
    for name, value in tables.items():
        before[name] = np.array(value, copy=True)
    coarse = refine_power_tables(tables=tables, refinement=8)
    fine = refine_power_tables(tables=tables, refinement=16)
    for name in tables:
        np.testing.assert_array_equal(tables[name], before[name])
    np.testing.assert_array_equal(fine["log10k_2D"][::2], coarse["log10k_2D"])
    nz = len(tables["z_2D"])
    for name in ("lnP_linear", "lnP_nonlinear", "lnP_linear_cb"):
        coarse_values = coarse[name].reshape((nz, -1), order="F")
        fine_values = fine[name].reshape((nz, -1), order="F")
        np.testing.assert_array_equal(fine_values[:, ::2], coarse_values)
