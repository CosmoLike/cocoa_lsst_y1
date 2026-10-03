"""Independent checks of full-sky spin-bin and discrete Fourier operators.

The external reference integrates Jacobi polynomials and independently
checks their spin convention with high-precision factorial rotation sums.
This test does not certify a survey covariance or its ell cutoff.
"""

import ctypes
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


def row_pointers(array):
    """Address each live padded NumPy row without assuming flat storage."""
    pointer = ctypes.POINTER(ctypes.c_double)
    result = (pointer*len(array))()
    for index in range(len(array)):
        result[index] = array[index].ctypes.data_as(pointer)
    return result


class CovarianceOperators(unittest.TestCase):
    """Check actual production operators, including output-row padding."""

    @classmethod
    def setUpClass(cls):
        """Load an explicitly supplied isolated library and external reference."""
        reference = os.environ.get("COSMOLIKE_COVARIANCE_REFERENCE")
        library = os.environ.get("COSMOLIKE_OPERATORS_LIBRARY")
        if not reference or not library:
            raise unittest.SkipTest("set covariance reference and operator library paths")
        source = Path(reference)/"operators_reference.py"
        if not source.is_file() or not Path(library).is_file():
            raise unittest.SkipTest("operator reference or library is absent")
        specification = importlib.util.spec_from_file_location(
            name="operators_reference", location=source
        )
        cls.reference = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.reference)
        cls.library = ctypes.CDLL(library)
        integer = ctypes.c_int
        real_pointer = ctypes.POINTER(ctypes.c_double)
        integer_pointer = ctypes.POINTER(integer)
        rows = ctypes.POINTER(real_pointer)
        cls.library.realspace_operator_cov.argtypes = [
            integer, real_pointer, integer, integer, rows,
        ]
        cls.library.realspace_operator_cov.restype = None
        cls.library.bandpower_operator_cov.argtypes = [
            integer, integer, integer, integer_pointer, integer_pointer, rows,
        ]
        cls.library.bandpower_operator_cov.restype = None
        cls.library.omp_set_num_threads.argtypes = [integer]
        cls.library.omp_set_num_threads.restype = None
        cls.library.omp_set_num_threads(4)

    def angular(self, edges, ell_max, nquad):
        """Build four operators with sentinel columns after every output row."""
        edges = np.ascontiguousarray(edges, dtype=float)
        nbin = len(edges)-1
        storage = np.full((4*nbin, ell_max+4), 713.25)
        self.library.realspace_operator_cov(
            nbin, edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ell_max, nquad, row_pointers(storage),
        )
        np.testing.assert_array_equal(storage[:, ell_max+1:], 713.25)
        return storage[:, :ell_max+1].reshape(4, nbin, ell_max+1).copy()

    def bands(self, first, last, ell_min, nell):
        """Build inclusive integer bands, preserving padding and input arrays."""
        first = np.ascontiguousarray(first, dtype=np.int32)
        last = np.ascontiguousarray(last, dtype=np.int32)
        storage = np.full((len(first), nell+3), -719.5)
        pointer = ctypes.POINTER(ctypes.c_int)
        self.library.bandpower_operator_cov(
            len(first), ell_min, nell, first.ctypes.data_as(pointer),
            last.ctypes.data_as(pointer), row_pointers(storage),
        )
        np.testing.assert_array_equal(storage[:, nell:], -719.5)
        return storage[:, :nell].copy()

    def test_factorial_rotation_averages(self):
        """Check low degrees, tiny xi- and broad bins at 60-digit precision."""
        edges = np.array([0.0007, 0.001, 0.1, 0.5])
        actual = self.angular(edges=edges, ell_max=9, nquad=64)
        pairs = [(2, 2), (2, -2), (2, 0), (0, 0)]
        for probe, (first, second) in enumerate(pairs):
            for row in range(3):
                for ell in (2, 3, 5, 9):
                    expected = self.reference.high_precision_bin(
                        lower=edges[row], upper=edges[row+1], ell=ell,
                        first=first, second=second,
                    )
                    np.testing.assert_allclose(
                        actual[probe, row, ell], expected, rtol=5.e-13, atol=0,
                    )
        np.testing.assert_array_equal(actual[:3, :, :2], 0.0)
        np.testing.assert_allclose(actual[3, :, 0], 1/(4*np.pi), rtol=3.e-15)

    def test_high_ell_independent_polynomials(self):
        """Check every bin through ell=50000 with SciPy's polynomial evaluator."""
        edges = np.geomspace(2.5, 250.0, 21)*np.pi/(180*60)
        multipoles = np.array([2, 3, 10, 100, 1000, 10000, 30000, 50000])
        actual = self.angular(edges=edges, ell_max=50000, nquad=512)
        expected = self.reference.angular_operator(
            edges=edges, multipoles=multipoles, nquad=512,
        )
        normalization = (2*multipoles+1)/(4*np.pi)
        error = np.abs(actual[:, :, multipoles]-expected)/normalization
        print(f"  spin-bin polynomial error / mode density: {error.max():.9e}")
        self.assertLess(error.max(), 1.e-10)

    def test_narrow_bins_and_endpoints(self):
        """Retain tiny-angle shear and resolve endpoints without edge subtraction."""
        edges = np.array([0.0, 1.e-8, 1.e-8+1.e-14, np.pi])
        actual = self.angular(edges=edges, ell_max=7, nquad=64)
        pairs = [(2, 2), (2, -2), (2, 0), (0, 0)]
        for probe, (first, second) in enumerate(pairs):
            for row in range(3):
                expected = self.reference.high_precision_bin(
                    lower=edges[row], upper=edges[row+1], ell=7,
                    first=first, second=second,
                )
                # The first two bins have tiny but nonzero xi-: an absolute
                # tolerance would hide losing it. The almost full-sky last
                # bin can instead have a vanishing scalar integral.
                absolute_tolerance = 0.0
                if row == 2:
                    absolute_tolerance = 5.e-15
                np.testing.assert_allclose(actual[probe, row, 7], expected,
                                           rtol=2.e-12, atol=absolute_tolerance)

    def test_angular_refinement(self):
        """Measure quadrature separately from polynomial recurrence error."""
        edges = np.geomspace(2.5, 250.0, 21)*np.pi/(180*60)
        coarse = self.angular(edges=edges, ell_max=50000, nquad=256)
        fine = self.angular(edges=edges, ell_max=50000, nquad=512)
        normalization = (2*np.arange(50001)+1)/(4*np.pi)
        error = np.abs(coarse-fine)/normalization
        print(f"  spin-bin 256 -> 512 / mode density: {error.max():.9e}")
        self.assertLess(error.max(), 2.e-10)

    def test_thread_and_geometry_roundtrip(self):
        """Bitwise outputs at 1/4/8 threads, with changed/restored angular bins."""
        edges = np.array([0.001, 0.003, 0.01, 0.07])
        outputs = []
        for threads in (1, 4, 8):
            self.library.omp_set_num_threads(threads)
            outputs.append(self.angular(edges=edges, ell_max=2048, nquad=128))
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])
        shifted = self.angular(edges=edges*1.01, ell_max=2048, nquad=128)
        self.assertGreater(np.max(np.abs(shifted-outputs[0])), 1.e-3)
        restored = self.angular(edges=edges, ell_max=2048, nquad=128)
        np.testing.assert_array_equal(restored, outputs[0])
        self.library.omp_set_num_threads(4)

    def test_discrete_band_gaussian_and_connected(self):
        """Exact mode count, overlapping-band covariance and constant cNG."""
        first = np.array([2, 5, 11, 8])
        last = np.array([4, 10, 11, 15])
        ell = np.arange(2, 19)
        operators = self.bands(first=first, last=last, ell_min=2, nell=len(ell))
        expected = np.zeros_like(operators)
        counts = np.empty(len(first))
        for band in range(len(first)):
            inside = (ell >= first[band]) & (ell <= last[band])
            counts[band] = np.sum(2*ell[inside]+1)
            expected[band, inside] = (2*ell[inside]+1)/counts[band]
        # SIMD multiplies by the shared reciprocal; NumPy divides directly.
        # Their one-rounding difference can reach one binary64 ulp.
        np.testing.assert_allclose(operators, expected, rtol=5.e-16, atol=0)
        np.testing.assert_allclose(operators.sum(axis=1), 1, rtol=3.e-16)
        fsky = 0.3
        wick = 7.0
        covariance = np.einsum("il,l,jl->ij", operators, wick/((2*ell+1)*fsky),
                               operators, optimize=False)
        np.testing.assert_allclose(np.diag(covariance), wick/(fsky*counts),
                                   rtol=3.e-16)
        self.assertEqual(covariance[0, 1], 0.0)
        self.assertGreater(covariance[1, 3], 0.0)
        connected = np.einsum("il,lm,jm->ij", operators,
                              np.full((len(ell), len(ell)), 2.5), operators)
        np.testing.assert_allclose(connected, 2.5, rtol=6.e-16)


if __name__ == "__main__":
    unittest.main()
