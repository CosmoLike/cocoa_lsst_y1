"""Validate halo response conventions and the five trispectrum partitions.

Two C routines of cosmolike/covariances/non_gaussian_cov.c are checked:

  halo_trispectrum_cov: the connected matter trispectrum T(k, q) (the
    four-point function the non-Gaussian covariance needs) as five
    halo-model terms, by how the four density legs (k, -k, q, -q) are
    shared among halos: one halo (T_1h), two halos as 1+3 (T_13) or 2+2
    (T_22), three halos (T_3h) and four halos (T_4h, the tree-level
    perturbation-theory term);
  halo_response_cov: the response D = dP/d(delta_b) of the power spectrum
    to a long-wavelength background density mode delta_b, which the
    super-sample covariance (SSC) needs; growth and dilation are the
    coefficients of D = (growth - dilation x slope) P_2h + I12.

Independent checks use explicit labelled halo partitions and EdS
(Einstein-de Sitter perturbation theory) Wick diagrams. The response test
differentiates a specified halo-power model; it does not certify that
model's nonlinear calibration. The C routines are called through ctypes
(argtypes and restype declare their C signatures).
"""

import ctypes
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
    """A regular linear test spectrum with a turnover near k=1.

    Arguments:
      k = wavenumber (arbitrary units).

    Returns:
      k/(1 + k^2)^2.
    """
    return k/(1+k*k)**2


class NonGaussianCovariance(unittest.TestCase):
    """Check assembly independently of the mass and angular integrators."""

    @classmethod
    def setUpClass(cls):
        """Load the repository references and project C symbols.

        Declares halo_response_cov(n, growth, dilation, fractional,
        inputs[][], output[][]) and halo_trispectrum_cov(n, pk[][],
        i11[][], moments[][], tree[][], output[][]), and saves the
        caller's OpenMP thread count.
        """
        import cosmolike_lsst_y1_interface as ci
        from cosmolike_notebook_utils.covariance.reference import cng_reference
        from cosmolike_notebook_utils.covariance.reference import non_gaussian_reference

        cls.modules = {
            "cng_reference": cng_reference,
            "non_gaussian_reference": non_gaussian_reference,
        }
        cls.library = ctypes.CDLL(ci.__file__)
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
        """Call the C response builder and check padded output canaries.

        The output gets three extra NaN columns (canaries) the C code
        must not touch.

        Arguments:
          inputs   = six rows [P_lin, P_target, I11, I02, I12, slope],
                     slope = dlnP/dlnk.
          growth, dilation = the response coefficients.
          fractional = True rescales the halo response to P_target.

        Returns:
          array [2, n]: P_halo and the response D.
        """
        count = inputs.shape[1]
        output = np.full((2, count+3), np.nan)
        self.library.halo_response_cov(count, growth, dilation, int(fractional),
                                       rows(inputs), rows(output))
        self.assertTrue(np.all(np.isnan(output[:, count:])))
        return output[:, :count].copy()

    def trispectrum(self, pk, i11, moments, tree):
        """Call the C assembler, retaining all five contributions separately.

        Arguments:
          pk      = linear power at K and Q [2, n].
          i11     = the halo moment I11 at K and Q [2, n].
          moments = rows I02, I12, I13(K,Q,Q), I13(K,K,Q), I04 [5, n].
          tree    = angle-averaged tree-level power, bispectrum and
                    trispectrum [3, n].

        Returns:
          array [5, n]: T_1h, T_13, T_22, T_3h, T_4h.
        """
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
        # three labelled halo species: number density, volume (mass over
        # mean density), bias and radius of a Gaussian profile
        # u(k) = exp(-k^2 r^2/2); the moments below are the halo-model
        # mass integrals written as sums over the species
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
        """Use the published two-halo slope, including the I11 scale dependence.

        growth = 47/21 and dilation = 1/3 are the isotropic halo response
        of Takada & Hu (2013). The expected derivative is a centered finite
        difference (step 1e-5) of the power after a background mode
        delta: amplitude exp(47 delta/21), wavenumbers dilated by
        exp(-delta/3), one-halo term shifted by delta I12.
        """
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
        """R1+RK/6 equals 17/7-n/2 in the linear regime, independently of code.

        R1 = 1 + 26/21 - n/3 (isotropic) and RK = 8/7 - n (tidal) are the
        tree-level responses for a spectrum of slope n; with growth 17/7
        and dilation 1/2 the C response must reproduce R1 + RK/6.
        """
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
        """All halo terms have length^9, and a negative tree term is retained.

        scale = 4282.7494 is c/H0 in Mpc for h = 0.7 (2997.92458/0.7): a
        change of length unit multiplies each input by scale to its length
        power (P: 3; moments: 3, 3, 6, 6, 9; tree: 3, 6, 9), and every
        trispectrum term must then change by scale^9. The seeds of this
        test and the next only fix the random draws.
        """
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


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead
if __name__ == "__main__":
    unittest.main()
