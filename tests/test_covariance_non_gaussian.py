"""Validate halo response conventions and the five trispectrum partitions.

Independent checks use explicit labelled halo partitions and EdS Wick
diagrams. The response test differentiates a specified halo-power model;
it does not certify that model's nonlinear calibration.
"""

import ctypes
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


def rows(array):
    """Return C row pointers for live, possibly padded NumPy storage."""
    pointer = ctypes.POINTER(ctypes.c_double)
    output = (pointer*len(array))()
    for row in range(len(array)):
        output[row] = array[row].ctypes.data_as(pointer)
    return output


def power(k):
    """A regular linear test spectrum with a turnover near k=1."""
    return k/(1+k*k)**2


class NonGaussianCovariance(unittest.TestCase):
    """Check assembly independently of the mass and angular integrators."""

    @classmethod
    def setUpClass(cls):
        """Load explicitly supplied references and isolated production code."""
        reference = os.environ.get("COSMOLIKE_COVARIANCE_REFERENCE")
        library = os.environ.get("COSMOLIKE_NON_GAUSSIAN_LIBRARY")
        if not reference or not library:
            raise unittest.SkipTest("set covariance reference and non-Gaussian library")
        directory = Path(reference)
        cls.modules = {}
        for name in ("cng_reference", "non_gaussian_reference"):
            path = directory/f"{name}.py"
            if not path.is_file():
                raise unittest.SkipTest(f"external reference {name} is absent")
            specification = importlib.util.spec_from_file_location(name=name, location=path)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            cls.modules[name] = module
        if not Path(library).is_file():
            raise unittest.SkipTest("isolated non-Gaussian library is absent")
        cls.library = ctypes.CDLL(library)
        integer = ctypes.c_int
        real = ctypes.c_double
        pointer = ctypes.POINTER(ctypes.POINTER(real))
        cls.library.halo_response_cov.argtypes = [integer, real, real, integer,
                                                 pointer, pointer]
        cls.library.halo_response_cov.restype = None
        cls.library.halo_trispectrum_cov.argtypes = [integer]+[pointer]*5
        cls.library.halo_trispectrum_cov.restype = None
        cls.library.omp_set_num_threads.argtypes = [integer]
        cls.library.omp_get_max_threads.restype = integer
        cls.original_threads = cls.library.omp_get_max_threads()

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's thread setting after repeated evaluations."""
        cls.library.omp_set_num_threads(cls.original_threads)

    def response(self, inputs, growth, dilation, fractional):
        """Call the C response builder and check padded output canaries."""
        count = inputs.shape[1]
        output = np.full((2, count+3), np.nan)
        self.library.halo_response_cov(count, growth, dilation, int(fractional),
                                       rows(inputs), rows(output))
        self.assertTrue(np.all(np.isnan(output[:, count:])))
        return output[:, :count].copy()

    def trispectrum(self, pk, i11, moments, tree):
        """Call the C assembler, retaining all five contributions separately."""
        count = pk.shape[1]
        output = np.full((5, count+3), np.nan)
        self.library.halo_trispectrum_cov(count, rows(pk), rows(i11), rows(moments),
                                          rows(tree), rows(output))
        self.assertTrue(np.all(np.isnan(output[:, count:])))
        return output[:, :count].copy()

    def test_labelled_halo_partitions(self):
        """Enumerate the four 1+3 partitions and every other halo grouping."""
        tree_reference = self.modules["cng_reference"]
        reference = self.modules["non_gaussian_reference"]
        count = np.array([0.2, 0.3, 0.1])
        volume = np.array([0.7, 1.1, 1.8])
        bias = np.array([0.8, 1.3, 1.9])
        radii = np.array([0.1, 0.3, 0.5])
        for first, second in ((0.3, 0.7), (1., 1.), (0.7, 0.3)):
            uk = np.exp(-first**2*radii**2/2)
            uq = np.exp(-second**2*radii**2/2)
            product = uk*uq
            moments = np.array([
                np.sum(count*volume**2*product),
                np.sum(count*bias*volume**2*product),
                np.sum(count*bias*volume**3*product*uq),
                np.sum(count*bias*volume**3*product*uk),
                np.sum(count*volume**4*product**2),
            ])[:, None]
            i11 = np.array([np.sum(count*bias*volume*uk),
                            np.sum(count*bias*volume*uq)])[:, None]
            pk = np.array([power(first), power(second)])[:, None]
            tree = tree_reference.reduced_averages(
                first_k=first, second_k=second, power=power
            )[:, None]
            actual = self.trispectrum(pk=pk, i11=i11, moments=moments, tree=tree)
            expected = reference.partition_terms(
                first_k=first, second_k=second, power=power, count=count,
                volume=volume, bias=bias, profile_k=uk, profile_q=uq,
                tree_reference=tree_reference, nquad=256,
            )
            np.testing.assert_allclose(actual[:, 0], expected, rtol=2.e-12)

    def test_response_by_power_differentiation(self):
        """Use the published two-halo slope, including the I11 scale dependence."""
        k = np.geomspace(0.1, 2., 13)
        exponent = -1.3
        halo_exponent = -0.15
        linear = k**exponent
        i11 = k**halo_exponent
        one_halo = np.full(len(k), 0.07)
        biased_one_halo = np.full(len(k), 0.11)
        target = linear+0.8
        slope = np.full(len(k), exponent+2*halo_exponent)
        inputs = np.array([linear, target, i11, one_halo, biased_one_halo, slope])
        actual = self.response(inputs=inputs, growth=47/21, dilation=1/3,
                                fractional=False)
        step = 1.e-5
        values = []
        for sign in (-1, 1):
            delta = sign*step
            shifted_k = k*np.exp(-delta/3)
            values.append(np.exp(47*delta/21)*shifted_k**(exponent+2*halo_exponent)
                          +one_halo+delta*biased_one_halo)
        derivative = (values[1]-values[0])/(2*step)
        np.testing.assert_allclose(actual[1], derivative, rtol=3.e-9)
        np.testing.assert_allclose(actual[0], i11**2*linear+one_halo, rtol=2.e-15)
        fractional = self.response(inputs=inputs, growth=47/21, dilation=1/3,
                                    fractional=True)
        np.testing.assert_allclose(fractional[1], actual[1]/actual[0]*target,
                                   rtol=3.e-15)

    def test_projected_tree_limit(self):
        """R1+RK/6 equals 17/7-n/2 in the linear regime, independently of code."""
        slopes = np.array([-2.5, -2., -1.5, -1., 0., 1., 2.])
        pk = np.linspace(0.1, 1., len(slopes))
        inputs = np.array([pk, pk, np.ones_like(pk), np.zeros_like(pk),
                           np.zeros_like(pk), slopes])
        actual = self.response(inputs=inputs, growth=17/7, dilation=1/2,
                                fractional=False)
        isotropic = 1+26/21-slopes/3
        tidal = 8/7-slopes
        np.testing.assert_allclose(actual[1]/pk, isotropic+tidal/6, rtol=3.e-15)

    def test_length_dimensions_and_negative_terms(self):
        """All halo terms have length^9, and a negative tree term is retained."""
        generator = np.random.default_rng(seed=1302699425)
        pk = generator.uniform(0.1, 0.9, size=(2, 17))
        i11 = generator.uniform(0.5, 1.1, size=(2, 17))
        moments = generator.uniform(0.01, 0.1, size=(5, 17))
        tree = generator.normal(size=(3, 17))
        actual = self.trispectrum(pk=pk, i11=i11, moments=moments, tree=tree)
        self.assertTrue(np.any(actual[4] < 0.0))
        scale = 4282.7494
        converted_moments = moments.copy()
        for role, exponent in enumerate((3, 3, 6, 6, 9)):
            converted_moments[role] *= scale**exponent
        converted_tree = tree.copy()
        for role, exponent in enumerate((3, 6, 9)):
            converted_tree[role] *= scale**exponent
        converted = self.trispectrum(pk=pk*scale**3, i11=i11,
                                     moments=converted_moments, tree=converted_tree)
        np.testing.assert_allclose(converted/scale**9, actual, rtol=1.e-12)

    def test_repeated_threads(self):
        """Odd SIMD groups and all five terms are bitwise stable across threads."""
        generator = np.random.default_rng(seed=2026100325)
        pk = generator.uniform(0.1, 1., size=(2, 65))
        i11 = generator.uniform(0.5, 1., size=(2, 65))
        moments = generator.uniform(0.01, 1., size=(5, 65))
        tree = generator.normal(size=(3, 65))
        inputs = generator.uniform(0.2, 1., size=(6, 65))
        original = None
        for threads in (1, 4, 8, 1, 8):
            self.library.omp_set_num_threads(threads)
            response = self.response(inputs=inputs, growth=17/7, dilation=1/2,
                                      fractional=True)
            terms = self.trispectrum(pk=pk, i11=i11, moments=moments, tree=tree)
            if original is None:
                original = (response, terms)
            for actual, expected in zip((response, terms), original):
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
