"""Check survey assembly against independently expanded small matrices.

These checks test interpolation, angular/radial contraction and row coverage.
They do not establish numerical convergence of a physical survey forecast.
"""

import numpy as np

from cosmolike_notebook_utils.covariance.survey import (
    compress_operators,
    observable_rows,
    project_connected,
)


def test_observable_layout():
    """Retain the actual survey dimensions and only measured pair exclusions."""
    lsst = observable_rows(nlens=5, nsource=5)
    roman = observable_rows(
        nlens=8, nsource=8, excluded_gammat=((6, 0), (7, 0), (7, 1))
    )
    assert lsst.shape == (60, 3)
    assert roman.shape == (141, 3)
    assert np.sum(roman[:, 0] == 2) == 61


def test_compressed_transform():
    """Compare B T B^T with a dense explicitly interpolated signed spectrum."""
    rng = np.random.default_rng(seed=729)
    ell = np.arange(2., 60.)
    coarse = np.exp(np.linspace(np.log(2.5), np.log(59.5), 7))-0.5
    coarse[[0, -1]] = ell[[0, -1]]
    kernels = rng.normal(size=(4, 3, len(ell)))
    raw = rng.normal(size=(7, 7))
    trispectrum = raw+raw.T
    hats = np.empty((len(ell), len(coarse)))
    for node in range(len(coarse)):
        basis = np.zeros(len(coarse))
        basis[node] = 1.0
        hats[:, node] = np.interp(np.log(ell+0.5), np.log(coarse+0.5), basis)
    dense = hats @ trispectrum @ hats.T
    spin = (ell-1)*(ell+2)/(ell+0.5)**2
    physical = kernels.copy()
    for probe, count in enumerate((2, 2, 1, 0)):
        physical[probe] *= spin**count
    physical = physical.reshape(12, -1)
    expected = physical @ dense @ physical.T
    compressed = compress_operators(
        operators=kernels, ell=ell, coarse_ell=coarse
    ).reshape(12, -1)
    actual = compressed @ trispectrum @ compressed.T
    np.testing.assert_allclose(actual, expected, rtol=3.e-12, atol=2.e-12)


def test_complete_connected_projection():
    """All cross-lens and angular blocks agree with one NumPy tensor sum."""
    import cosmolike_lsst_y1_interface as ci

    rng = np.random.default_rng(seed=829)
    rows = observable_rows(nlens=2, nsource=2)
    ntheta = 3
    nradial = 9
    pair = rng.uniform(size=(len(rows), nradial))
    raw = rng.normal(size=(4*ntheta, 4*ntheta, nradial))
    projected = raw+raw.transpose(1, 0, 2)
    measure = rng.uniform(size=nradial)
    expected = np.empty((len(rows)*ntheta, len(rows)*ntheta))
    # Expand one observable at a time, independently of the grouped C calls.
    for first, left in enumerate(rows):
        for second, right in enumerate(rows):
            for i in range(ntheta):
                for j in range(ntheta):
                    integrand = projected[left[0]*ntheta+i, right[0]*ntheta+j]
                    value = np.sum(pair[first]*pair[second]*measure*integrand)
                    expected[first*ntheta+i, second*ntheta+j] = value
    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(threads)
        actual = project_connected(
            interface=ci, rows=rows, pair_window=pair,
            projected=projected, measure=measure,
        )
        np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=1.e-14)
        np.testing.assert_array_equal(actual, actual.T)
        if baseline is None:
            baseline = actual
        np.testing.assert_array_equal(actual.view(np.uint64), baseline.view(np.uint64))
