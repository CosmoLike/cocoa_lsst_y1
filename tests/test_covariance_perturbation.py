"""Validate cNG tree-level angular averages before adding halo moments.

The external reference enumerates Wick diagrams and the EdS recursion.
The C code uses reduced formulas, precomputed input powers and SIMDe.
No halo or full covariance prediction is certified by these angular tests.
"""

import ctypes
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


def row_pointers(array):
    """Return ctypes pointers to live float64 rows, allowing padding."""
    pointer = ctypes.POINTER(ctypes.c_double)
    rows = (pointer*len(array))()
    for row in range(len(array)):
        rows[row] = array[row].ctypes.data_as(pointer)
    return rows


def power(k):
    """A positive test spectrum with a finite peak and regular small-k limit."""
    return k/(1+k*k)**2


class PerturbationCovariance(unittest.TestCase):
    """Check diagram multiplicity, diagonal limits, angular accuracy and units."""

    @classmethod
    def setUpClass(cls):
        """Load explicitly supplied external reference and isolated C library."""
        reference = os.environ.get("COSMOLIKE_COVARIANCE_REFERENCE")
        library = os.environ.get("COSMOLIKE_PERTURBATION_LIBRARY")
        if not reference or not library:
            raise unittest.SkipTest("set covariance reference and perturbation library")
        source = Path(reference)/"cng_reference.py"
        if not source.is_file() or not Path(library).is_file():
            raise unittest.SkipTest("cNG reference or isolated library is absent")
        specification = importlib.util.spec_from_file_location(
            name="cng_reference", location=source
        )
        cls.reference = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.reference)
        cls.library = ctypes.CDLL(library)
        integer = ctypes.c_int
        pointer = ctypes.POINTER(ctypes.c_double)
        rows = ctypes.POINTER(pointer)
        cls.library.tree_averages_cov.argtypes = [
            integer, integer, rows, rows, pointer, pointer, rows, rows,
        ]
        cls.library.tree_averages_cov.restype = None
        cls.library.omp_set_num_threads.argtypes = [integer]
        cls.library.omp_get_max_threads.restype = integer
        cls.original_threads = cls.library.omp_get_max_threads()

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's thread count after the reproducibility sweep."""
        cls.library.omp_set_num_threads(cls.original_threads)

    def averages(self, pairs, nquad=64, npanel=16, scale=1.0):
        """Prepare supplied power samples, then call only the C angular algebra.

        Arguments: pairs [npair,2] positive K,Q; rule sizes; distance scale
            converts k -> k/scale and P -> P*scale^3 for the units check.
        Returns: [3,npair] P/B/T averages, with input/output canaries checked.
        """
        pairs = np.asarray(pairs, dtype=float)
        theta, weights, corner = self.reference.angle_rule(nquad=nquad, npanel=npanel)
        k = np.ascontiguousarray(pairs.T/scale)
        pk = np.ascontiguousarray(power(pairs.T)*scale**3)
        magnitude = np.sqrt((pairs[:, 0, None]-pairs[:, 1, None])**2
                            +2*pairs[:, 0, None]*pairs[:, 1, None]*corner)
        ps = power(magnitude)*scale**3
        output = np.full((3, len(pairs)+3), np.nan)
        pointer = ctypes.POINTER(ctypes.c_double)
        self.library.tree_averages_cov(
            len(pairs), len(theta), row_pointers(k), row_pointers(pk),
            corner.ctypes.data_as(pointer), weights.ctypes.data_as(pointer),
            row_pointers(ps), row_pointers(output),
        )
        self.assertTrue(np.all(np.isnan(output[:, len(pairs):])))
        return output[:, :len(pairs)].copy()

    def test_explicit_wick_diagrams(self):
        """Direct F2/F3 diagrams reproduce each of the three C averages."""
        pairs = np.array([[0.2, 0.3], [1., 1.], [2., 0.3], [10., 10.], [0.3, 2.]])
        actual = self.averages(pairs=pairs)
        for index, (first, second) in enumerate(pairs):
            expected = self.reference.direct_averages(
                first_k=first, second_k=second, power=power, nquad=256
            )
            np.testing.assert_allclose(actual[:, index], expected, rtol=2.e-12)

    def test_closed_planar_f3(self):
        """The closed F3 average agrees with the six-permutation EdS recursion."""
        theta, weights, unused = self.reference.angle_rule(nquad=512, npanel=1)
        paired = np.array([1.0, 0.0])
        for ratio in (0.01, 0.3, 1.0, 2.0, 100.0):
            total = 0.0
            for angle, weight in zip(theta, weights):
                other = ratio*np.array([np.cos(angle), np.sin(angle)])
                total += weight*self.reference.third_order(
                    vectors=np.array([paired, -paired, other])
                )
            expected = self.reference.paired_f3_average(paired_k=1.0, other_k=ratio)
            self.assertAlmostEqual(total/expected, 1.0, delta=1.e-12)

    def test_corner_against_high_precision(self):
        """Resolve the narrow internal-power peak on and near the diagonal."""
        import mpmath as mp

        pairs = np.array([[1000., 1000.], [1000., 1000.001], [1., 1.0001]])
        actual = self.averages(pairs=pairs, nquad=64, npanel=20)
        refined = self.averages(pairs=pairs, nquad=128, npanel=20)
        np.testing.assert_allclose(actual, refined, rtol=1.e-12)
        with mp.workdps(60):
            for index, (first, second) in enumerate(pairs):
                k = mp.mpf(first)
                q = mp.mpf(second)
                pk = k/(1+k*k)**2
                pq = q/(1+q*q)**2

                def integrand(gap, role):
                    # Integrate the original vector F2 bracket at high precision,
                    # not the rearranged double-precision expression used by C.
                    mu = -mp.cos(gap)
                    s2 = (k-q)**2+4*k*q*mp.sin(gap/2)**2
                    if s2 == 0:
                        return mp.mpf(0)  # P(0)=0; bracket has a finite limit
                    ps = mp.sqrt(s2)/(1+s2)**2
                    dot_k = -(k*k+k*q*mu)
                    dot_q = -(q*q+k*q*mu)
                    f_k = mp.mpf(5)/7+dot_k*(1/s2+1/(k*k))/2
                    f_k += mp.mpf(2)/7*dot_k**2/(s2*k*k)
                    f_q = mp.mpf(5)/7+dot_q*(1/s2+1/(q*q))/2
                    f_q += mp.mpf(2)/7*dot_q**2/(s2*q*q)
                    bracket = f_k*pk+f_q*pq
                    return ps*bracket**role/mp.pi

                edges = [0, mp.mpf("0.000001"), mp.mpf("0.001"),
                         mp.mpf("0.01"), mp.mpf("0.1"), mp.pi]
                integrals = []
                for role in range(3):
                    integrals.append(mp.quad(lambda gap: integrand(gap, role), edges))
                # Closed F3 is independently checked against the recursion above.
                f3_k = self.reference.paired_f3_average(paired_k=first, other_k=second)
                f3_q = self.reference.paired_f3_average(paired_k=second, other_k=first)
                expected = np.array([
                    float(integrals[0]),
                    float(12*pk*pq/7+2*integrals[1]),
                    float(12*f3_k*pk*pk*pq+12*f3_q*pq*pq*pk+8*integrals[2]),
                ])
                np.testing.assert_allclose(actual[:, index], expected, rtol=2.e-11)

    def test_units_and_pair_exchange(self):
        """P,B,T transform with length^3, length^6, length^9, and K,Q exchange."""
        pairs = np.array([[0.1, 0.3], [0.3, 0.1], [1., 1.0001], [10., 0.5]])
        actual = self.averages(pairs=pairs)
        np.testing.assert_allclose(actual[:, 0], actual[:, 1], rtol=2.e-13)
        scale = 4282.7494
        converted = self.averages(pairs=pairs, scale=scale)
        for role in range(3):
            np.testing.assert_allclose(converted[role]/scale**(3*(role+1)),
                                       actual[role], rtol=1.e-12)

    def test_thread_and_repeat_determinism(self):
        """Retain the same angular-node summation order for odd pair counts."""
        first = np.geomspace(0.01, 1000., 41)
        pairs = np.column_stack([first, first*(1+1.e-4*np.cos(first))])
        original = None
        for threads in (1, 4, 8, 1, 8):
            self.library.omp_set_num_threads(threads)
            actual = self.averages(pairs=pairs)
            if original is None:
                original = actual
            np.testing.assert_array_equal(actual, original)


if __name__ == "__main__":
    unittest.main()
