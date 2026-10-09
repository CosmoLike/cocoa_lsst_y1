"""Check complete connected projections with supplied signed radial tables.

The connected (non-Gaussian) covariance between two observable rows is a
sum over radial integration nodes,

    C[left, right] = sum_n window[o_left, n] window[o_right, n]
                     projected[p_left*nbin + b_left, p_right*nbin + b_right, n]
                     measure[n],

where o is the observable a row belongs to, p its probe, b its bin,
window the row's radial weight, projected the projected trispectrum of
the two (probe, bin) pairs, and measure the quadrature weight. The block
reference calls the weighted-sum primitive covariance_project once per
angular block. A separate NumPy contraction checks the equation itself.
No cosmology initialization is needed: random but fixed inputs isolate
catalog/bin mapping, sum order, symmetry and thread scheduling from halo
modeling.
"""

import numpy as np
from abort_check import assert_aborts
import pytest


def connected_inputs(nbin, nnode, one_probe=False):
    """Return mixed probe groups and signed windows/trispectrum samples.

    The three nontrivial group lengths are 1, 3 and 5, exercising incomplete
    four-row SIMD tiles (SIMD = vector instructions that process four
    rows at once). Probe order is intentionally different from matrix
    row order. With one_probe, three groups are empty. All dimensions are
    small so the independent per-entry check is inexpensive.

    Arguments:
      nbin  = bins per probe.
      nnode = radial nodes.
      one_probe = True keeps only the first observable.

    Returns:
      (probes [n_obs] int32, window [n_obs, nnode], projected
      [4 nbin, 4 nbin, nnode], symmetric in its first two axes,
      measure [nnode]).
    """
    # probes[o] = the probe (0-3) of observable row o; the seed 721 only
    # fixes the random draws
    probes = np.array([3, 0, 2, 1, 0, 0, 2, 2, 2, 2], dtype=np.int32)
    if one_probe:
        probes = probes[:1].copy()
    rng = np.random.default_rng(seed=721)
    window = rng.normal(size=(len(probes), nnode))
    projected = rng.normal(size=(4*nbin, 4*nbin, nnode))
    # adding the transpose of the first two axes makes the table
    # symmetric, like a covariance
    projected = projected+projected.transpose(1, 0, 2)
    measure = np.linspace(0.3, 1.1, nnode)
    return probes, window, projected, measure


def connected_block_reference(interface, probes, window, projected, measure):
    """Assemble the matrix block by block with the covariance_project primitive.

    One call per (probe, bin) block pair, with the primitive's upper
    triangle convention for diagonal blocks. This keeps the C primitive's
    tested fused sums, allowing a bitwise scheduling comparison. It does
    not use the complete-matrix function covariance_project_connected.

    Arguments:
      interface = the compiled module cosmolike_lsst_y1_interface.
      probes, window, projected, measure = as connected_inputs returns.

    Returns:
      the symmetric matrix [n_obs nbin, n_obs nbin].
    """
    nbin = projected.shape[0]//4
    result = np.empty((len(probes)*nbin, len(probes)*nbin))
    # groups[p] = the observable rows of probe p (np.flatnonzero lists
    # the positions where the condition is true)
    groups = []
    for probe in range(4):
        groups.append(np.flatnonzero(probes == probe))
    for left_probe in range(4):
        left_rows = groups[left_probe]
        if len(left_rows) == 0:
            continue
        for right_probe in range(left_probe, 4):
            right_rows = groups[right_probe]
            if len(right_rows) == 0:
                continue
            for first in range(nbin):
                start = first if left_probe == right_probe else 0
                for second in range(start, nbin):
                    weight = measure*projected[
                        left_probe*nbin+first, right_probe*nbin+second
                    ]
                    block = interface.covariance_project(
                        left=window[left_rows].copy(),
                        right=window[right_rows].copy(), weight=weight,
                    )
                    # a diagonal block is symmetric: rebuild it from its
                    # upper triangle (np.triu with k=1 excludes the
                    # diagonal, so it is not counted twice)
                    if left_probe == right_probe and first == second:
                        block = np.triu(block)+np.triu(block, k=1).T
                    i = left_rows*nbin+first
                    j = right_rows*nbin+second
                    # np.ix_(i, j) selects the submatrix with rows i and
                    # columns j; the transposed block fills the mirror
                    result[np.ix_(i, j)] = block
                    result[np.ix_(j, i)] = block.T
    return result


# pytest runs the test once for every combination of these values
# (2 x 2 x 3 = 12 runs); 1, 7 and 8 nodes cover the scalar path, an
# incomplete SIMD tile and a complete one
@pytest.mark.parametrize("one_probe", [False, True])
@pytest.mark.parametrize("nbin", [1, 3])
@pytest.mark.parametrize("nnode", [1, 7, 8])
def test_connected_blocks_and_threads(one_probe, nbin, nnode):
    """Whole matrices equal the block-by-block sums bit for bit, at any thread count."""
    import cosmolike_lsst_y1_interface as ci
    from cosmolike_notebook_utils.covariance.survey import project_connected

    probes, window, projected, measure = connected_inputs(
        nbin=nbin, nnode=nnode, one_probe=one_probe
    )
    ci.set_omp_threads(n=1)
    expected = connected_block_reference(
        interface=ci, probes=probes, window=window,
        projected=projected, measure=measure,
    )

    # Evaluate the covariance equation independently with NumPy sums.
    # Their rounding need not equal the C fused sums, so use a tight
    # absolute scale based on this entire supplied matrix.
    direct = np.empty_like(expected)
    for left in range(len(probes)*nbin):
        # divmod returns (quotient, remainder): row = observable*nbin + bin
        observable_left, bin_left = divmod(left, nbin)
        first = probes[observable_left]*nbin+bin_left
        for right in range(len(probes)*nbin):
            observable_right, bin_right = divmod(right, nbin)
            second = probes[observable_right]*nbin+bin_right
            integrand = window[observable_left]*window[observable_right]
            integrand *= projected[first, second]*measure
            direct[left, right] = np.sum(integrand)
    np.testing.assert_allclose(
        actual=expected, desired=direct, rtol=1.e-14,
        atol=1.e-14*np.max(np.abs(direct)),
    )

    # the rows table of project_connected: column 0 = probe (the bin
    # columns stay 0 here)
    rows = np.zeros((len(probes), 3), dtype=np.int32)
    rows[:, 0] = probes
    # .view(np.uint64) reinterprets each double's 64 bits as an
    # integer, so the comparison below demands identical bit patterns
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        actual = project_connected(
            interface=ci, rows=rows, pair_window=window,
            projected=projected, measure=measure,
        )
        np.testing.assert_array_equal(
            x=actual.view(np.uint64), y=expected.view(np.uint64)
        )
        np.testing.assert_array_equal(x=actual, y=actual.T)
        # CARMA (the library converting the C++ Armadillo matrices to
        # numpy) hands back an array whose memory is held by a Python
        # capsule object. NumPy's OWNDATA flag need not be set;
        # independence from the inputs is the contract.
        assert not np.shares_memory(actual, window)
        assert not np.shares_memory(actual, projected)

    # Later calls cannot change an already returned owned matrix.
    changed = ci.covariance_project_connected(
        probes=probes, pair_window=window, projected=projected*2,
        measure=measure,
    )
    assert not np.array_equal(changed, actual)
    np.testing.assert_array_equal(x=actual, y=expected)


def test_connected_rejects_invalid_inputs():
    """Array shapes and invalid values fail before parallel pointer access."""
    import cosmolike_lsst_y1_interface as ci
    from cosmolike_notebook_utils.covariance.survey import project_connected

    probes, window, projected, measure = connected_inputs(nbin=3, nnode=7)
    valid = {
        "probes": probes,
        "pair_window": window,
        "projected": projected,
        "measure": measure,
    }
    invalid = [
        ("probes", np.full(len(probes), 4, dtype=np.int32)),
        ("pair_window", window[:, :-1].copy()),
        ("projected", projected[:-1].copy()),
        ("measure", np.full(7, np.nan)),
    ]
    for name, value in invalid:
        arguments = dict(valid)
        arguments[name] = value
        def attempt():
            ci.covariance_project_connected(**arguments)
        assert_aborts(attempt)

    # The high-level helper must not truncate fractional or overflowing
    # probe IDs while converting an integer array to the C int32 type.
    for dtype, value in ((float, 0.5), (np.int64, 2**32)):
        rows = np.zeros((len(probes), 3), dtype=dtype)
        rows[:, 0] = value
        def attempt():
            project_connected(
                interface=ci, rows=rows, pair_window=window,
                projected=projected, measure=measure,
            )
        assert_aborts(attempt)
