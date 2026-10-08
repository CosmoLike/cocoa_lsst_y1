"""Independent checks of full-sky spin-bin and discrete Fourier operators.

An operator turns harmonic quantities into binned observables. The
real-space operator K_l(bin) gives a bin-averaged correlation function
from the angular power spectrum, xi(bin) = sum_l K_l(bin) C_l; its four
probes use spin-weighted functions of the angle with spin pairs (2, 2)
for xi+, (2, -2) for xi-, (2, 0) for gamma_t and (0, 0) for w(theta),
written as Jacobi polynomials. The bandpower operator averages C_l over
an integer band [first, last] with weights (2l + 1)/sum(2l + 1). The
same operators project the harmonic covariance onto binned observables.

The independent reference integrates Jacobi polynomials and independently
checks their spin convention with high-precision factorial rotation sums.
This test does not certify a survey covariance or its ell cutoff. The C
functions are called through ctypes (argtypes and restype declare their
C signatures).
"""

import ctypes
import unittest

import numpy as np
from scipy.special import eval_legendre


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
        """Load the project C symbols and independent reference."""
        import cosmolike_lsst_y1_interface as ci
        from cosmolike_notebook_utils.covariance.reference import operators_reference

        cls.reference = operators_reference
        cls.library = ctypes.CDLL(ci.__file__)
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
        """Build four operators with sentinel columns after every output row.

        Each output row has three extra columns prefilled with the sentinel
        713.25, which the C code must leave in place (a write past the row
        would overwrite it).

        Arguments:
          edges   = angular bin edges [radians].
          ell_max = largest multipole.
          nquad   = Gauss-Legendre nodes per angular integration panel.

        Returns:
          array [4 probes, nbin, ell_max + 1], probes in the order xi+,
          xi-, gamma_t, w.
        """
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
        """Build inclusive integer bands, preserving padding and input arrays.

        Arguments:
          first, last = first and last multipole of each band (inclusive).
          ell_min = the multipole of column 0.
          nell    = number of multipole columns.

        Returns:
          array [n_band, nell] of band weights; the three sentinel
          columns (-719.5) must survive the call.
        """
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
        # the spin pairs of the four probes: xi+, xi-, gamma_t, w
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
        # spin-2 probes have no l = 0, 1 modes; the scalar monopole of a
        # bin average is 1/(4 pi)
        np.testing.assert_array_equal(actual[:3, :, :2], 0.0)
        np.testing.assert_allclose(actual[3, :, 0], 1/(4*np.pi), rtol=3.e-15)

    def test_high_ell_independent_polynomials(self):
        """Check every bin through ell=50000 with SciPy's polynomial evaluator."""
        # 20 log-spaced bins from 2.5 to 250 arcmin, in radians; errors
        # are measured relative to the mode density (2l+1)/(4 pi)
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

    def test_high_multipole_wide_bin_against_antiderivative(self):
        """Resolve the widest LSST bin through ell=100000 independently of GL."""
        # the last of the 26 LSST-Y1 bins (2.5 to 900 arcmin), in radians
        edges = np.geomspace(2.5, 900.0, 27)[-2:]*np.pi/(180*60)
        ell = np.arange(20000, 100001, 500)
        lower, upper = np.cos(edges)
        # Integrate P_ell(x) analytically between the two cosines. The
        # antiderivative is (P_(ell+1)-P_(ell-1))/(2*ell+1); multiplying by
        # the harmonic mode density cancels its denominator. These large
        # angles avoid the small-angle endpoint cancellation of xi-.
        expected = (
            eval_legendre(ell+1, lower)-eval_legendre(ell-1, lower)
            -eval_legendre(ell+1, upper)+eval_legendre(ell-1, upper)
        )/(4*np.pi*(lower-upper))
        for count in (64, 96, 128):
            actual = self.angular(edges=edges, ell_max=100000, nquad=count)[3, 0, ell]
            error = np.max(np.abs(actual-expected))/np.max(np.abs(expected))
            self.assertLess(error, 1.e-6)

    def test_precomputed_rule_floor_and_polynomial_integrals(self):
        """Accept only tabulated rules >=64, checking their nodes and weights."""
        import cosmolike_lsst_y1_interface as ci
        # each tabulated Gauss-Legendre rule on [-1, 1] must integrate
        # x^degree exactly: 2/(degree+1) for even degrees; other node
        # counts are refused
        for count in (64, 96, 128, 256, 512, 1024):
            nodes, weights = ci.covariance_integration_rule(nquad=count)
            self.assertTrue(np.all(np.diff(nodes) > 0.0))
            self.assertTrue(np.all(weights > 0.0))
            for degree in (0, 2, 10, 20):
                integral = np.dot(weights, nodes**degree)
                np.testing.assert_allclose(integral, 2/(degree+1), rtol=2.e-14)
        for count in (16, 32, 65, 384, 2048):
            with self.assertRaises(ValueError):
                ci.covariance_integration_rule(nquad=count)

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
        # a Gaussian harmonic covariance wick/((2l+1) f_sky) projected
        # onto the bands: np.einsum("il,l,jl->ij", A, w, A) is
        # sum_l A[i,l] w[l] A[j,l], the matrix A diag(w) A^T (optimize=False
        # keeps a fixed summation order). Bands that share no multipole
        # (bands 0 and 1) are uncorrelated; overlapping ones (1 and 3)
        # are positively correlated.
        fsky = 0.3
        wick = 7.0
        covariance = np.einsum("il,l,jl->ij", operators, wick/((2*ell+1)*fsky),
                               operators, optimize=False)
        np.testing.assert_allclose(np.diag(covariance), wick/(fsky*counts),
                                   rtol=3.e-16)
        self.assertEqual(covariance[0, 1], 0.0)
        self.assertGreater(covariance[1, 3], 0.0)
        # a constant connected covariance (2.5 for every l, m) stays 2.5
        # after projection, because every band's weights sum to one
        connected = np.einsum("il,lm,jm->ij", operators,
                              np.full((len(ell), len(ell)), 2.5), operators)
        np.testing.assert_allclose(connected, 2.5, rtol=6.e-16)


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead
if __name__ == "__main__":
    unittest.main()
