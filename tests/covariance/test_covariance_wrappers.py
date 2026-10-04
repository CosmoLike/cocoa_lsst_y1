"""Check notebook-level Gaussian wrappers against independent block assembly.

Real-space checks retain the already tested analytic pair-noise primitive.
Fourier checks explicitly form the Wick products and band sums in NumPy.
The arrays contain complete internal cross spectra, while the measured
observable list contains only a subset of field pairs.
"""

import numpy as np
import pytest

from cosmolike_notebook_utils.covariance.gaussian import realspace_block


def inputs():
    """Return positive field spectra, white noise and a small multipole grid."""
    ell = np.arange(2., 41.)
    amplitude = np.array([1.0, 0.7, 0.5, 0.9])
    field_power = np.outer(amplitude, amplitude)+0.2*np.eye(4)
    spectra = np.ascontiguousarray(field_power[None, :, :]/(ell[:, None, None]+1))
    noise = np.array([0.02, 0.03, 0.01, 0.015])
    return spectra, noise, ell


def test_realspace_wrapper_matches_blocks():
    """The whole-matrix call preserves per-block arithmetic at every thread count."""
    import cosmolike_lsst_y1_interface as ci

    spectra, noise, ell = inputs()
    rows = np.array([
        [0, 2, 2],
        [1, 3, 3],
        [2, 0, 2],
        [3, 1, 1],
        [3, 0, 0],
    ], dtype=np.int32)
    operators = ci.covariance_realspace_operator(
        edges_rad=np.array([0.01, 0.02, 0.04, 0.08]),
        ell_max=int(ell[-1]), nquad=64,
    )
    kernels = np.ascontiguousarray(operators[:, :, 2:])
    pair_area = np.array([0.001, 0.004, 0.016])
    area_sr = 0.8
    expected = np.empty((15, 15))
    for first, (left_probe, a, b) in enumerate(rows):
        for second in range(first, len(rows)):
            right_probe, c, d = rows[second]
            block = realspace_block(
                interface=ci, spectra=spectra, noise=noise,
                fields=np.array([a, b, c, d], dtype=np.int32),
                operators=operators, probe_left=left_probe,
                probe_right=right_probe, pair_area_sr2=pair_area,
                area_sr=area_sr,
            )
            if first == second:
                block = np.triu(block)+np.triu(block, 1).T
            i = slice(first*3, (first+1)*3)
            j = slice(second*3, (second+1)*3)
            expected[i, j] = block
            expected[j, i] = block.T

    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(threads)
        actual = ci.covariance_gaussian_real(
            spectra=spectra, noise=noise, rows=rows, operators=kernels,
            ell_min=2, area_sr=area_sr, pair_area_sr2=pair_area,
        )
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        if baseline is None:
            baseline = actual.copy()
        np.testing.assert_array_equal(actual.view(np.uint64), baseline.view(np.uint64))
        np.testing.assert_array_equal(actual, actual.T)

    # A subsequent call cannot mutate a previously returned owned matrix.
    noise *= 2
    changed = ci.covariance_gaussian_real(
        spectra=spectra, noise=noise, rows=rows, operators=kernels,
        ell_min=2, area_sr=area_sr, pair_area_sr2=pair_area,
    )
    assert not np.array_equal(changed, baseline)
    np.testing.assert_array_equal(actual, baseline)


def test_fourier_wrapper_matches_numpy():
    """Check overlapping bands, all Wick crossings and harmonic pure noise."""
    import cosmolike_lsst_y1_interface as ci

    spectra, noise, ell = inputs()
    pairs = np.array([[2, 2], [2, 3], [0, 2], [1, 1]], dtype=np.int32)
    operators = ci.covariance_bandpower_operator(
        first=np.array([2, 12, 20], dtype=np.int32),
        last=np.array([15, 31, 40], dtype=np.int32),
        ell_min=2, nell=len(ell),
    )
    area_sr = 0.8
    observed = spectra+np.diag(noise)[None, :, :]
    expected = np.empty((12, 12))
    for first, (a, b) in enumerate(pairs):
        for second, (c, d) in enumerate(pairs):
            wick = observed[:, a, c]*observed[:, b, d]
            wick += observed[:, a, d]*observed[:, b, c]
            harmonic = wick/((2*ell+1)*area_sr/(4*np.pi))
            block = (operators*harmonic) @ operators.T
            i = slice(first*3, (first+1)*3)
            j = slice(second*3, (second+1)*3)
            expected[i, j] = block
    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(threads)
        actual = ci.covariance_gaussian_fourier(
            spectra=spectra, noise=noise, pairs=pairs, operators=operators,
            ell_min=2, area_sr=area_sr,
        )
        np.testing.assert_allclose(actual, expected, rtol=3.e-15, atol=1.e-18)
        np.testing.assert_array_equal(actual, actual.T)
        if baseline is None:
            baseline = actual.copy()
        np.testing.assert_array_equal(actual.view(np.uint64), baseline.view(np.uint64))


def test_wrapper_rejects_invalid_shapes_and_fields():
    """Malformed notebook arrays raise Python errors before entering C."""
    import cosmolike_lsst_y1_interface as ci

    spectra, noise, ell = inputs()
    kwargs = {
        "spectra": spectra,
        "noise": noise,
        "pairs": np.array([[0, 2]], dtype=np.int32),
        "operators": np.ones((2, len(ell))),
        "ell_min": 2,
        "area_sr": 0.8,
    }
    for key, value in (
        ("pairs", np.array([[0, 4]], dtype=np.int32)),
        ("operators", np.ones((2, len(ell)-1))),
        ("noise", np.array([-1., 0., 0., 0.])),
        ("area_sr", 0.0),
        ("spectra", np.full_like(spectra, np.nan)),
    ):
        invalid = dict(kwargs)
        invalid[key] = value
        with pytest.raises(ValueError):
            ci.covariance_gaussian_fourier(**invalid)
