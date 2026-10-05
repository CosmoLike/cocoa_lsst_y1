"""Check the covariance Bessel transform against analytic integrals.

These tests use the project's ordinary compiled C library. They do not need
survey inputs, CAMB, a developer's external library, or an FFT reference.
"""

import ctypes

import numpy as np
import pytest
from scipy.special import hyp1f1

import cosmolike_lsst_y1_interface as ci


@pytest.fixture
def transform_library():
    """Describe the C array boundary and restore the caller's OpenMP team."""
    library = ctypes.CDLL(ci.__file__)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    rows = ctypes.POINTER(double_pointer)
    library.fftlog_create_cov.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        rows,
    ]
    library.fftlog_create_cov.restype = ctypes.c_void_p
    library.fftlog_execute_cov.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        rows,
        ctypes.POINTER(rows),
    ]
    library.fftlog_free_cov.argtypes = [ctypes.c_void_p]
    library.omp_get_max_threads.restype = ctypes.c_int
    original = library.omp_get_max_threads()
    yield library
    ci.set_omp_threads(original)


def row_addresses(values):
    """Expose contiguous NumPy rows only for this test of the public C API."""
    pointer = ctypes.POINTER(ctypes.c_double)
    result = (pointer*len(values))()
    for index, row in enumerate(values):
        result[index] = row.ctypes.data_as(pointer)
    return result


def project(library, chi, radial, powers, first, count, threads):
    """Execute twice with one workspace; outputs remain owned by NumPy."""
    ci.set_omp_threads(threads)
    intervals = len(chi)-1
    padding = intervals
    extra = intervals//4
    nfft = 4*intervals
    nk = len(chi)+2*extra
    spacing = np.log(chi[1]/chi[0])
    powers = np.asarray(powers, dtype=np.int32)
    workspace = library.fftlog_create_cov(
        len(chi), padding, nfft, extra, chi[0], spacing, len(radial),
        powers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        row_addresses(values=radial),
    )
    wave = np.empty((count, nk))
    transfer = np.empty((count, len(radial), nk))
    rows = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
    planes = (rows*count)()
    owners = []
    for index in range(count):
        addresses = row_addresses(values=transfer[index])
        owners.append(addresses)
        planes[index] = addresses
    try:
        # A different preceding multipole must not leave stale kernels in
        # the workspace, and inverse FFTs must not overwrite saved forwards.
        library.fftlog_execute_cov(
            workspace, first+19, count, row_addresses(values=wave), planes,
        )
        library.fftlog_execute_cov(
            workspace, first, count, row_addresses(values=wave), planes,
        )
    finally:
        library.fftlog_free_cov(workspace)
    return wave, transfer


@pytest.mark.parametrize("first", [2, 32, 96])
def test_gaussian_bessel_integrals(transform_library, first):
    """Check normalization, both kernels, parity recurrence and team reuse."""
    chi = np.geomspace(1.e-4, 1.e4, 2049)
    count = 16
    multipoles = (first, first+1, first+15)
    radial = []
    powers = []
    for ell in multipoles:
        for power in (0, 2):
            alpha = (ell+3+power)/2
            radial.append(np.exp((ell+3+power)*np.log(chi)-alpha*(chi**2-1)))
            powers.append(power)
    radial = np.asarray(radial)
    reference = None
    for threads in (1, 8):
        wave, transfer = project(
            library=transform_library, chi=chi, radial=radial, powers=powers,
            first=first, count=count, threads=threads,
        )
        if reference is None:
            reference = transfer.copy()
        else:
            np.testing.assert_array_equal(transfer, reference)
        field = 0
        for ell in multipoles:
            for power in (0, 2):
                k = wave[ell-first]
                alpha = (ell+3+power)/2
                logarithm = (
                    alpha+0.5*np.log(np.pi)+(ell-power)*np.log(k)
                    -(ell+2)*np.log(2)-(ell+1.5)*np.log(alpha)
                    -k**2/(4*alpha)
                )
                exact = np.exp(logarithm)
                error = np.max(np.abs(transfer[ell-first, field]-exact))
                print(f"ell={ell}, p={power}: peak-scaled error {error/np.max(exact):.3e}")
                assert error/np.max(exact) < 1.e-8
                field += 1


def test_lensing_observer_endpoint(transform_library):
    """A chi^2 lensing weight stays regular when the Bessel kernel holds x^-2."""
    chi = np.geomspace(1.e-6, 10.0, 2049)
    radial = (chi**2*np.exp(-chi**2))[None, :]
    wave, transfer = project(
        library=transform_library, chi=chi, radial=radial, powers=[2],
        first=2, count=1, threads=8,
    )
    exact = hyp1f1(1.0, 3.5, -wave[0]**2/4)/30
    assert np.max(np.abs(transfer[0, 0]-exact))/np.max(exact) < 2.e-9
