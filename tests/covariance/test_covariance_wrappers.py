"""Check notebook-level Gaussian wrappers against independent block assembly.

The whole-matrix functions covariance_gaussian_real and
covariance_gaussian_fourier of the compiled module must equal the
block-by-block assembly of the same Gaussian covariance. Real-space
checks retain the already tested analytic pair-noise primitive. Fourier
checks explicitly form the Wick products and band sums in NumPy. The
arrays contain complete internal cross spectra, while the measured
observable list contains only a subset of field pairs.
"""

import numpy as np
from abort_check import assert_aborts
import pytest

from cosmolike_notebook_utils.covariance.gaussian import realspace_block


def inputs():
    """Return positive field spectra, white noise and a small multipole grid.

    Returns:
      (spectra [n_ell, 4, 4], noise [4], ell [n_ell]): four fields with
      a positive-definite field matrix (outer product plus 0.2 on the
      diagonal) falling as 1/(l + 1), for l = 2..40.
    """
    ell = np.arange(2., 41.)
    amplitude = np.array([1.0, 0.7, 0.5, 0.9])
    field_power = np.outer(amplitude, amplitude)+0.2*np.eye(4)
    spectra = np.ascontiguousarray(field_power[None, :, :]/(ell[:, None, None]+1))
    noise = np.array([0.02, 0.03, 0.01, 0.015])
    return spectra, noise, ell


@pytest.mark.parametrize("nobs", [1, 5])
@pytest.mark.parametrize("nbin", [1, 3, 5, 20])
def test_realspace_wrapper_matches_blocks(nobs, nbin):
    """The whole-matrix call preserves per-block arithmetic at every thread count."""
    import cosmolike_lsst_y1_interface as ci

    spectra, noise, ell = inputs()
    # observable rows (probe, A, B): xi+ of field 2, xi- of field 3,
    # gamma_t of fields 0 and 2, w of field 1, w of field 0
    rows = np.array([
        [0, 2, 2],
        [1, 3, 3],
        [2, 0, 2],
        [3, 1, 1],
        [3, 0, 0],
    ], dtype=np.int32)[:nobs].copy()
    operators = ci.covariance_realspace_operator(
        edges_rad=np.geomspace(0.01, 0.08, nbin+1),
        ell_max=int(ell[-1]), nquad=64,
    )
    kernels = np.ascontiguousarray(operators[:, :, 2:])
    pair_area = np.geomspace(0.001, 0.016, nbin)
    area_sr = 0.8
    expected = np.empty((nobs*nbin, nobs*nbin))

    # One observable exercises C's bin-level threading; several observables
    # exercise whole-block threading. Bin counts cover full and partial
    # four-bin SIMD tiles, including the ordinary survey's twenty bins.
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
            # a diagonal block is symmetric: rebuild it from its upper
            # triangle (np.triu with 1 excludes the diagonal)
            if first == second:
                block = np.triu(block)+np.triu(block, 1).T
            i = slice(first*nbin, (first+1)*nbin)
            j = slice(second*nbin, (second+1)*nbin)
            expected[i, j] = block
            expected[j, i] = block.T

    # .view(np.uint64) compares the 64-bit patterns: bit-for-bit equality
    # with the block assembly and across thread counts
    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
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
    # four field pairs and three overlapping bands [2, 15], [12, 31],
    # [20, 40]; the expected covariance is the Wick product of the
    # observed spectra (signal plus noise on the diagonal) divided by
    # (2l+1) f_sky, projected onto the bands
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
        def attempt():
            ci.covariance_gaussian_fourier(**invalid)
        assert_aborts(attempt)


def notebook_layout(values, layout):
    """Represent the same physical axes in common notebook array layouts.

    Arguments:
      values = numpy array.
      layout = "c" (row-major copy), "fortran" (column-major copy),
               "sliced" (a strided view taking every second element of a
               larger array) or "readonly" (a row-major copy marked not
               writable).

    Returns:
      an array with the values of values in the requested memory layout.
    """
    if layout == 'fortran':
        return np.array(values, order='F', copy=True)
    if layout == 'sliced':
        shape = tuple(2*size for size in values.shape)
        parent = np.zeros(shape=shape, dtype=values.dtype)
        selection = tuple(slice(None, None, 2) for size in values.shape)
        view = parent[selection]
        view[...] = values
        return view
    result = np.array(values, order='C', copy=True)
    if layout == 'readonly':
        result.setflags(write=False)
    return result


@pytest.mark.parametrize('layout', ['c', 'fortran', 'sliced', 'readonly'])
def test_armadillo_axes_and_input_ownership(layout):
    """Conversions retain axes and leave both arrays and existing views intact.

    The compiled functions convert numpy arrays to Armadillo (the C++
    matrix library) objects; whatever the memory layout of the input,
    the result must use the same axes, the inputs must stay unchanged,
    and the output must not share memory with them.
    """
    import cosmolike_lsst_y1_interface as ci

    left = notebook_layout(np.arange(14.).reshape(2, 7), layout)
    right = notebook_layout(np.arange(21.).reshape(3, 7)-4.0, layout)
    weight = notebook_layout(np.linspace(0.1, 0.7, 7), layout)
    old_left = left.copy()
    left_view = left[:, ::2]
    old_view = left_view.copy()
    strides = left.strides
    expected = (left*weight) @ right.T
    actual = ci.covariance_project(left=left, right=right, weight=weight)
    np.testing.assert_allclose(actual, expected, rtol=3.e-15, atol=1.e-14)
    np.testing.assert_array_equal(left, old_left)
    np.testing.assert_array_equal(left_view, old_view)
    assert left.strides == strides
    assert not np.shares_memory(actual, left)
    assert not np.shares_memory(actual, right)

    # A cube of 48 elements exercises Armadillo's small-cube storage.
    # Its two transform axes differ from its three radial nodes.
    rng = np.random.default_rng(seed=772)
    matter = rng.normal(size=(4, 4, 3))
    matter += matter.transpose(1, 0, 2).copy()
    matter = notebook_layout(matter, layout)
    windows = notebook_layout(rng.normal(size=(2, 3)), layout)
    measure = notebook_layout(np.array([0.2, 0.5, 0.3]), layout)
    probes = notebook_layout(np.array([0, 3], dtype=np.int32), layout)
    saved_matter = matter.copy()
    projected = ci.covariance_project_connected(
        probes=probes, pair_window=windows, projected=matter, measure=measure,
    )
    expected = np.empty(shape=(2, 2))
    for first in range(2):
        for second in range(2):
            expected[first, second] = np.sum(
                windows[first]*windows[second]*measure
                *matter[probes[first], probes[second]]
            )
    np.testing.assert_allclose(projected, expected, rtol=3.e-15, atol=1.e-15)
    np.testing.assert_array_equal(matter, saved_matter)
    assert not np.shares_memory(projected, matter)
