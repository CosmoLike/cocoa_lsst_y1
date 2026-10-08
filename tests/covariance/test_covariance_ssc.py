"""Check the mask normalization and supplied-response SSC calculation.

Super-sample covariance (SSC): density modes larger than the survey
change the mean density delta_b inside the footprint, and every measured
power spectrum responds to it coherently. In the Limber form,

    Cov_SSC[A, B] = integral dchi R_A(chi) R_B(chi) sigma_b^2(chi),

with R the response of each observable to delta_b in a radial shell and
sigma_b^2 the variance of delta_b over the footprint, computed from the
harmonic spectrum C_L of the mask (ssc_mask_variance_cov). Galaxy
statistics are normalized by the measured mean density, which subtracts
-(b_A + b_B) P from their response (ssc_shell_response_cov). The
projection reuses gaussian_project_cov; a general background covariance
between shells (S K S^T) is checked as well. The C functions live in
cosmolike/covariances/ssc_cov.c and are called through ctypes (argtypes
and restype declare their C signatures).

The independent NumPy/mpmath reference is independent of the compiled code.
These checks validate mask and response algebra, including a correlated
radial-kernel hook. They do not certify a halo response, nonlinear tidal
model, non-Limber background calculation, or production survey accuracy.
"""

import ctypes
import unittest

import numpy as np


def row_pointers(array):
    """Return ctypes pointers to live, possibly padded float64 array rows."""
    pointer = ctypes.POINTER(ctypes.c_double)
    rows = (pointer*len(array))()
    for index in range(len(array)):
        rows[index] = array[index].ctypes.data_as(pointer)
    return rows


def vector_pointer(array):
    """Return a double pointer; the caller keeps the NumPy array alive."""
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class SuperSampleCovariance(unittest.TestCase):
    """Independent normalization, estimator differentiation and matrix checks."""

    @classmethod
    def setUpClass(cls):
        """Load only explicitly supplied independent reference and C library.

        Declares the C signatures of ssc_mask_variance_cov,
        ssc_shell_response_cov and gaussian_project_cov, and saves the
        caller's OpenMP thread count.
        """
        import cosmolike_lsst_y1_interface as ci
        from cosmolike_notebook_utils.covariance.reference import ssc_reference

        cls.reference = ssc_reference
        cls.library = ctypes.CDLL(ci.__file__)
        integer = ctypes.c_int
        real = ctypes.c_double
        pointer = ctypes.POINTER(real)
        rows = ctypes.POINTER(pointer)
        cls.library.ssc_mask_variance_cov.argtypes = [
            integer, integer, real, pointer, pointer, rows, pointer,
        ]
        cls.library.ssc_mask_variance_cov.restype = None
        cls.library.ssc_shell_response_cov.argtypes = [
            integer, integer, pointer, pointer, rows, rows, rows, rows,
        ]
        cls.library.ssc_shell_response_cov.restype = None
        cls.library.gaussian_project_cov.argtypes = [
            integer, integer, integer, rows, rows, pointer, rows, rows,
        ]
        cls.library.gaussian_project_cov.restype = None
        cls.library.omp_set_num_threads.argtypes = [integer]
        cls.library.omp_get_max_threads.restype = integer
        cls.original_threads = cls.library.omp_get_max_threads()

    @classmethod
    def tearDownClass(cls):
        """Restore the thread setting that was active before these tests."""
        cls.library.omp_set_num_threads(cls.original_threads)

    def variance(self, mask, area, distance, power):
        """Call C for an explicit power table and preserve an output canary.

        Arguments:
          mask     = harmonic spectrum C_L of the mask, L = 0..L_max.
          area     = footprint area [sr].
          distance = comoving distance of each radial node.
          power    = linear power [n_node, L_max + 1] at k = (L + 1/2)/chi.

        Returns:
          sigma_b^2 at each node; two extra NaN slots must survive.
        """
        output = np.full(len(distance)+2, np.nan)
        self.library.ssc_mask_variance_cov(
            len(distance), len(mask), area, vector_pointer(mask),
            vector_pointer(distance), row_pointers(power), vector_pointer(output),
        )
        self.assertTrue(np.all(np.isnan(output[len(distance):])))
        return output[:len(distance)].copy()

    def response(self, distance, signal, pair, mean, dp):
        """Call C with padded writable rows; return only the physical entries.

        Arguments:
          distance = comoving distance of each radial node [n_node].
          signal   = the projected signal of each row [n_row].
          pair     = pair window of each row [n_row, n_node].
          mean     = mean-density window, sum of the two galaxy legs.
          dp       = power response dP/d(delta_b) [n_row, n_node].

        Returns:
          the shell response [n_row, n_node].
        """
        output = np.full((len(signal), len(distance)+3), np.nan)
        self.library.ssc_shell_response_cov(
            len(signal), len(distance), vector_pointer(distance),
            vector_pointer(signal), row_pointers(pair), row_pointers(mean),
            row_pointers(dp), row_pointers(output),
        )
        self.assertTrue(np.all(np.isnan(output[:, len(distance):])))
        return output[:, :len(distance)].copy()

    def project(self, left, right, weights):
        """Reuse the production SIMD projection for any common integration axis.

        Arguments: left/right [nrow,nnode], weights [nnode].
        Returns: integral left_i * right_j * weights, with no BLAS call.
        """
        scratch = np.empty_like(left)
        output = np.empty((len(left), len(right)))
        self.library.gaussian_project_cov(
            len(left), len(right), len(weights), row_pointers(left),
            row_pointers(right), vector_pointer(weights), row_pointers(scratch),
            row_pointers(output),
        )
        return output

    def test_cap_harmonics_and_raw_normalization(self):
        """Check the cap coefficients against 60-digit direct angular integrals."""
        import mpmath as mp

        # a spherical cap of the LSST Y1 area (12300 deg^2, in sr): its
        # mask harmonics are C_L = pi [integral_edge^1 P_L(x) dx]^2, with
        # edge = cos(cap radius); mp.workdps(60) sets mpmath to 60 digits
        area = 12300*(np.pi/180)**2
        mask = self.reference.cap_mask(area_sr=area, ell_max=128)
        with mp.workdps(60):
            edge = 1-mp.mpf(area)/(2*mp.pi)
            for ell in (0, 1, 2, 7, 31, 64, 128):
                integral = mp.quad(lambda x: mp.legendre(ell, x), [edge, 1])
                expected = float(mp.pi*integral**2)
                self.assertAlmostEqual(mask[ell]/expected, 1.0, delta=1.e-10)
        distance = np.array([0.2, 0.5, 1.1])
        power = np.ones((3, len(mask)))
        actual = self.variance(mask=mask, area=area, distance=distance, power=power)
        expected = self.reference.mask_variance(
            mask_cl=mask, area_sr=area, distance=distance, power=power
        )
        np.testing.assert_allclose(actual, expected, rtol=3.e-14)
        # Multiplying W by c multiplies both C_L by c^2 and area by c.
        # The normalized survey average and its variance must not change.
        rescaled = self.variance(mask=mask*9, area=area*3,
                                 distance=distance, power=power)
        np.testing.assert_allclose(actual, rescaled, rtol=3.e-14)

    def test_white_power_mask_closed_form_and_units(self):
        """A band-limited mask has an exact sky integral of W^2.

        W=1+alpha*P_2 has only L=0 and L=2. With constant P the spherical
        mask sum is P*integral(W^2)/[area^2*f_K^2]. This tests the (2L+1)
        multiplicity independently of the cap recurrence.
        """
        alpha = 0.6
        area = 4*np.pi
        mask = np.array([4*np.pi, 0.0, 4*np.pi*alpha**2/25])
        distance = np.array([0.1, 0.6, 1.0, 1.2, 1.7])
        power = np.full((len(distance), 3), 0.004)
        actual = self.variance(mask=mask, area=area, distance=distance, power=power)
        expected = 0.004*(1+alpha**2/5)/(4*np.pi*distance**2)
        np.testing.assert_allclose(actual, expected, rtol=3.e-15)
        scale = 4282.7494  # example c/H0 in Mpc
        converted = self.variance(mask=mask, area=area, distance=distance*scale,
                                  power=power*scale**3)
        np.testing.assert_allclose(converted/scale, actual, rtol=3.e-15)

    def test_estimator_derivative_and_unit_invariance(self):
        """Differentiate an actual projected estimator, including its two means."""
        generator = np.random.default_rng(seed=13026994)
        nrow = 7
        nnode = 19  # exercises the scalar tail and padded output
        distance = np.linspace(0.1, 1.4, nnode)
        weights = np.full(nnode, 0.07)
        pair = generator.normal(size=(nrow, nnode))
        mean_a = generator.normal(size=(nrow, nnode))
        mean_b = generator.normal(size=(nrow, nnode))
        mean = mean_a+mean_b
        power = generator.uniform(0.1, 0.3, size=(nrow, nnode))
        dp = generator.uniform(0.2, 0.8, size=(nrow, nnode))
        signal = np.sum(pair*power/distance**2*weights, axis=1)
        actual = self.response(distance=distance, signal=signal, pair=pair,
                                mean=mean, dp=dp)
        expected = self.reference.shell_response(
            distance=distance, signal=signal, pair_window=pair,
            mean_window=mean, response_power=dp,
        )
        np.testing.assert_allclose(actual, expected, rtol=1.e-13, atol=1.e-14)
        direction = generator.normal(size=nnode)
        step = 1.e-5
        values = []
        for sign in (-1, 1):
            delta = sign*step*direction
            numerator = np.sum(pair*(power+dp*delta)/distance**2*weights, axis=1)
            normalization_a = 1+np.sum(mean_a*weights*delta, axis=1)
            normalization_b = 1+np.sum(mean_b*weights*delta, axis=1)
            values.append(numerator/(normalization_a*normalization_b))
        finite_difference = (values[1]-values[0])/(2*step)
        predicted = np.sum(actual*weights*direction, axis=1)
        np.testing.assert_allclose(predicted, finite_difference, rtol=2.e-8)
        # a change of length unit (c/H0 in Mpc for h = 0.7) rescales each
        # input by its length dimension; the response must follow
        scale = 4282.7494
        converted = self.response(distance=distance*scale, signal=signal,
                                   pair=pair/scale**2, mean=mean/scale,
                                   dp=dp*scale**3)
        np.testing.assert_allclose(converted*scale, actual, rtol=2.e-13, atol=1.e-13)

    def test_narrow_slice_mean_subtraction(self):
        """Two biased galaxies in one constant slice recover -(b_A+b_B)*P."""
        width = 0.15
        distance = np.full(9, 0.7)
        bias_a = 1.4
        bias_b = 1.8
        power = 0.004
        dp = np.full((1, 9), 0.013)
        pair = np.full((1, 9), bias_a*bias_b/width**2)
        mean = np.full((1, 9), (bias_a+bias_b)/width)
        signal = np.array([bias_a*bias_b*power/(distance[0]**2*width)])
        actual = self.response(distance=distance, signal=signal, pair=pair,
                                mean=mean, dp=dp)
        expected = pair*(dp-(bias_a+bias_b)*power)/distance**2
        np.testing.assert_allclose(actual, expected, rtol=2.e-14)

    def test_limber_and_correlated_projection(self):
        """Check S K S^T, signed correlations and non-Limber-compatible factors."""
        generator = np.random.default_rng(seed=171107467)
        response = generator.normal(size=(13, 31))
        radial = generator.uniform(0.01, 0.03, size=31)
        sigma2 = generator.uniform(0.02, 0.04, size=31)
        weights = radial*sigma2
        actual = self.project(left=response, right=response, weights=weights)
        expected = self.reference.project_limber(
            response=response, radial_weights=radial, sigma2=sigma2
        )
        np.testing.assert_allclose(actual, expected, rtol=2.e-13, atol=1.e-16)
        # The diagonal shell kernel has sigma^2/dchi, not sigma^2*dchi:
        # both radial integrations are explicit in the general formula.
        general = self.reference.project_correlated(
            response=response, radial_weights=radial,
            background=np.diag(sigma2/radial),
        )
        np.testing.assert_allclose(actual, general, rtol=2.e-13, atol=1.e-16)
        factor = generator.normal(size=(17, 31))  # K = factor.T @ factor
        modes = self.project(left=response, right=factor, weights=radial)
        correlated = self.project(left=modes, right=modes, weights=np.ones(17))
        expected = self.reference.project_correlated(
            response=response, radial_weights=radial,
            background=factor.T @ factor,
        )
        np.testing.assert_allclose(correlated, expected, rtol=2.e-13, atol=1.e-16)
        # both matrices are positive semidefinite correlation matrices
        # that still contain negative (anti-correlated) entries
        for covariance in (actual, correlated):
            scale = np.sqrt(np.diag(covariance))
            correlation = covariance/np.outer(scale, scale)
            self.assertGreaterEqual(np.linalg.eigvalsh(correlation)[0], -1.e-12)
            self.assertLess(np.min(correlation), 0.0)

    def test_subblocks_preserve_cross_correlations(self):
        """Separate rectangular calls must reconstruct one common SSC matrix.

        A Python survey driver may assign these blocks to different MPI
        processes. The C projection receives ordinary arrays and uses only
        OpenMP. Uneven block boundaries exercise the SIMD edge groups.
        Cross blocks must be computed even if their catalogs differ.
        """
        generator = np.random.default_rng(seed=20261004)
        response = generator.normal(size=(13, 31))
        weights = generator.uniform(low=0.01, high=0.03, size=31)
        expected = self.reference.project_limber(
            response=response,
            radial_weights=weights,
            sigma2=np.ones(shape=31),
        )
        boundaries = (0, 3, 7, 13)
        for threads in (1, 2, 4, 8):
            self.library.omp_set_num_threads(threads)
            complete = self.project(
                left=response, right=response, weights=weights
            )
            assembled = np.full_like(a=complete, fill_value=np.nan)

            # Both covariance indices need full coverage. Computing only
            # matching diagonal blocks discards shared background modes.
            for left in range(len(boundaries)-1):
                rows = slice(boundaries[left], boundaries[left+1])
                for right in range(len(boundaries)-1):
                    columns = slice(boundaries[right], boundaries[right+1])
                    assembled[rows, columns] = self.project(
                        left=response[rows],
                        right=response[columns],
                        weights=weights,
                    )
            np.testing.assert_array_equal(x=assembled, y=complete)
            np.testing.assert_allclose(
                actual=assembled, desired=expected, rtol=2.e-13, atol=1.e-16
            )
            # the Cholesky factorization exists only for a positive
            # definite matrix; numpy raises LinAlgError otherwise
            np.linalg.cholesky(a=assembled)

    def test_repetition_and_threads(self):
        """Odd shapes exercise SIMD tails; all outputs are bitwise repeatable."""
        generator = np.random.default_rng(seed=20261003)
        mask = self.reference.cap_mask(area_sr=1.7, ell_max=1024)
        distance = np.linspace(0.05, 1.3, 33)
        power = generator.uniform(0.01, 0.03, size=(33, len(mask)))
        pair = generator.normal(size=(11, 33))
        mean = generator.normal(size=(11, 33))
        dp = generator.uniform(0.01, 0.03, size=(11, 33))
        signal = generator.uniform(0.01, 0.03, size=11)
        first = None
        for threads in (1, 4, 8, 1, 8):
            self.library.omp_set_num_threads(threads)
            sigma = self.variance(mask=mask, area=1.7, distance=distance, power=power)
            phi = self.response(distance=distance, signal=signal, pair=pair,
                                 mean=mean, dp=dp)
            cov = self.project(left=phi, right=phi, weights=sigma*0.01)
            result = (sigma, phi, cov)
            if first is None:
                first = result
            for actual, expected in zip(result, first):
                np.testing.assert_array_equal(actual, expected)


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead
if __name__ == "__main__":
    unittest.main()
