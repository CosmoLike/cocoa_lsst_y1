"""Check complete connected projections with supplied signed radial tables.

The block reference calls the established weighted-sum primitive once per
angular block. A separate NumPy contraction checks the equation itself.
No cosmology initialization is needed: these tests isolate catalog/bin
mapping, sum order, symmetry and thread scheduling from halo modeling.
"""

import numpy as np
import pytest


def connected_inputs(nbin, nnode, one_probe=False):
    """Return mixed probe groups and signed windows/trispectrum samples.

    The three nontrivial group lengths are 1, 3 and 5, exercising incomplete
    four-row SIMD tiles. Probe order is intentionally different from matrix
    row order. With one_probe, three groups are empty. All dimensions are
    small so the independent per-entry check is inexpensive.
    """
    probes = np.array([3, 0, 2, 1, 0, 0, 2, 2, 2, 2], dtype=np.int32)
    if one_probe:
        probes = probes[:1].copy()
    rng = np.random.default_rng(seed=721)
    window = rng.normal(size=(len(probes), nnode))
    projected = rng.normal(size=(4*nbin, 4*nbin, nnode))
    projected = projected+projected.transpose(1, 0, 2)
    measure = np.linspace(0.3, 1.1, nnode)
    return probes, window, projected, measure


def connected_block_reference(interface, probes, window, projected, measure):
    """Reproduce the previous separate-block calls and triangle convention.

    This retains the C primitive's tested fused sums, allowing a bitwise
    scheduling comparison. It does not use the new complete-matrix wrapper.
    """
    nbin = projected.shape[0]//4
    result = np.empty((len(probes)*nbin, len(probes)*nbin))
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
                    if left_probe == right_probe and first == second:
                        block = np.triu(block)+np.triu(block, k=1).T
                    i = left_rows*nbin+first
                    j = right_rows*nbin+second
                    result[np.ix_(i, j)] = block
                    result[np.ix_(j, i)] = block.T
    return result


@pytest.mark.parametrize("one_probe", [False, True])
@pytest.mark.parametrize("nbin", [1, 3])
@pytest.mark.parametrize("nnode", [1, 7, 8])
def test_connected_blocks_and_threads(one_probe, nbin, nnode):
    """Whole blocks retain exact old sums, including scalar/SIMD edge cases."""
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

    rows = np.zeros((len(probes), 3), dtype=np.int32)
    rows[:, 0] = probes
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
        # CARMA's capsule owns the returned Armadillo storage. NumPy's
        # OWNDATA flag need not be set; independence is the contract.
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
        with pytest.raises(ValueError):
            ci.covariance_project_connected(**arguments)

    # The high-level helper must not truncate fractional or overflowing
    # probe IDs while converting an integer array to the C int32 type.
    for dtype, value in ((float, 0.5), (np.int64, 2**32)):
        rows = np.zeros((len(probes), 3), dtype=dtype)
        rows[:, 0] = value
        with pytest.raises(ValueError):
            project_connected(
                interface=ci, rows=rows, pair_window=window,
                projected=projected, measure=measure,
            )
