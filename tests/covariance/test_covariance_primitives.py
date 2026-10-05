"""Check the isolated covariance algebra before a survey driver is integrated.

The repository supplies independent NumPy and mpmath oracles. Tests call
symbols from the normally built project interface, including padded output
checks at the C boundary. No separate development library is required.
"""

import ctypes
import unittest

import numpy as np


def row_pointers(array):
    """Return C double row pointers for a live float64 2D NumPy array.

    Arguments:
        array = float64 [nrow, ncolumn], with contiguous individual rows.
                Distinct rows may be separated by padding.
    Returns:
        ctypes array of nrow double pointers; the caller retains array.
    """
    double_pointer = ctypes.POINTER(ctypes.c_double)
    pointer_array = (double_pointer*array.shape[0])()
    for row in range(array.shape[0]):
        pointer_array[row] = array[row].ctypes.data_as(double_pointer)
    return pointer_array


class CovariancePrimitives(unittest.TestCase):
    """Compare the C primitives with closed forms and independent NumPy algebra."""

    @classmethod
    def setUpClass(cls):
        """Load the repository oracle and normally built project interface."""
        import cosmolike_lsst_y1_interface as ci
        from cosmolike_notebook_utils.covariance.reference import gaussian_reference

        cls.reference = gaussian_reference
        cls.library = ctypes.CDLL(ci.__file__)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        row_pointer = ctypes.POINTER(double_pointer)
        cls.library.gaussian_wick_cov.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            row_pointer,
            double_pointer,
            ctypes.c_int,
            double_pointer,
        ]
        cls.library.gaussian_wick_cov.restype = None
        cls.library.gaussian_project_cov.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            row_pointer,
            row_pointer,
            double_pointer,
            row_pointer,
            row_pointer,
        ]
        cls.library.gaussian_project_cov.restype = None
        cls.library.annulus_pair_area_cov.argtypes = [ctypes.c_double]*3
        cls.library.annulus_pair_area_cov.restype = ctypes.c_double
        cls.library.gaussian_noise_pair_cov.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            double_pointer,
            ctypes.c_double,
        ]
        cls.library.gaussian_noise_pair_cov.restype = ctypes.c_double
        cls.library.omp_set_num_threads.argtypes = [ctypes.c_int]
        cls.library.omp_set_num_threads.restype = None
        cls.library.omp_get_max_threads.argtypes = []
        cls.library.omp_get_max_threads.restype = ctypes.c_int
        cls.original_threads = cls.library.omp_get_max_threads()

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's OpenMP setting after the thread-count sweep."""
        cls.library.omp_set_num_threads(cls.original_threads)

    def wick(self, signal, noise, left, right, ell_min=0, include_nn=False):
        """Call the C Wick primitive for two catalog pairs at fsky=0.2.

        Arguments: signal [nfield,nfield,nell], diagonal noise [nfield],
            left/right catalog pairs, first multipole, and noise-product switch.
        Returns: float64 [nell], with the supplied signal arrays unchanged.
        """
        field_a, field_b = left
        field_c, field_d = right
        cross_signal = np.ascontiguousarray([
            signal[field_a, field_c],
            signal[field_b, field_d],
            signal[field_a, field_d],
            signal[field_b, field_c],
        ])
        noise_matrix = np.diag(noise)
        cross_noise = np.array([
            noise_matrix[field_a, field_c],
            noise_matrix[field_b, field_d],
            noise_matrix[field_a, field_d],
            noise_matrix[field_b, field_c],
        ])
        output = np.empty(signal.shape[-1])
        pointer = ctypes.POINTER(ctypes.c_double)
        # ctypes follows the C signature positionally: grid, sky fraction,
        # four signal rows, four noise powers, NN switch, output buffer.
        self.library.gaussian_wick_cov(
            ell_min, len(output), 0.2, row_pointers(cross_signal),
            cross_noise.ctypes.data_as(pointer), int(include_nn),
            output.ctypes.data_as(pointer),
        )
        return output

    def project(self, left, right, harmonic):
        """Return the C projection using padded scratch and output rows.

        Arguments: left [nleft,nell], right [nright,nell], harmonic [nell].
        Returns: float64 [nleft,nright]. Padding canaries are checked to
            detect writes beyond the physical columns in house-style rows.
        """
        nleft, nell = left.shape
        nright = len(right)
        scratch = np.full((nleft, nell+3), np.nan)
        output = np.full((nleft, nright+3), np.nan)
        pointer = ctypes.POINTER(ctypes.c_double)
        self.library.gaussian_project_cov(
            nleft, nright, nell, row_pointers(left), row_pointers(right),
            harmonic.ctypes.data_as(pointer), row_pointers(scratch),
            row_pointers(output),
        )
        self.assertTrue(np.all(np.isnan(scratch[:, nell:])))
        self.assertTrue(np.all(np.isnan(output[:, nright:])))
        return output[:, :nright].copy()

    def test_all_pairs_wick_and_psd(self):
        """Keep every field pair, signed correlations, both NN choices, and PSD."""
        generator = np.random.default_rng(seed=60105779)
        factors = generator.normal(size=(4, 4, 37))
        signal = np.einsum("ikl,jkl->ijl", factors, factors)*1.e-8
        noise = np.array([2.e-8, 3.e-8, 4.e-9, 7.e-9])
        pairs = []
        for first in range(4):
            for second in range(first, 4):
                pairs.append((first, second))
        for include_nn in (False, True):
            expected = self.reference.harmonic_covariance(
                signal=signal, noise=noise, pairs=pairs, ell_min=2,
                fsky=0.2, include_nn=include_nn,
            )
            actual = np.empty_like(expected)
            for left, pair_left in enumerate(pairs):
                for right, pair_right in enumerate(pairs):
                    actual[left, right] = self.wick(
                        signal=signal, noise=noise, left=pair_left,
                        right=pair_right, ell_min=2, include_nn=include_nn,
                    )
            np.testing.assert_allclose(actual, expected, rtol=3.e-14, atol=1.e-29)
            np.testing.assert_allclose(
                actual, actual.swapaxes(0, 1), rtol=3.e-14, atol=1.e-29
            )
            for node in range(signal.shape[-1]):
                matrix = actual[:, :, node]
                scale = np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
                minimum = np.linalg.eigvalsh(matrix/scale)[0]
                self.assertGreaterEqual(minimum, -1.e-12)

    def test_integer_band_mode_count(self):
        """Disjoint bands recover the Gaussian variance from their exact mode counts."""
        signal = np.full((1, 1, 30), 2.e-7)
        noise = np.array([3.e-8])
        harmonic = self.wick(
            signal=signal, noise=noise, left=(0, 0), right=(0, 0),
            ell_min=2, include_nn=True,
        )
        ell = np.arange(2, 32)
        operators = np.zeros((2, len(ell)))
        expected_diagonal = []
        for row, (low, high) in enumerate(((2, 9), (10, 31))):
            selected = (ell >= low) & (ell <= high)
            mode_weights = 2*ell[selected]+1
            mode_count = np.sum(mode_weights)
            operators[row, selected] = mode_weights/mode_count
            expected_diagonal.append(2*(2.e-7+3.e-8)**2/(0.2*mode_count))
        actual = self.project(left=operators, right=operators, harmonic=harmonic)
        np.testing.assert_allclose(actual, np.diag(expected_diagonal), rtol=3.e-15)

    def test_projection_shapes_and_threads(self):
        """Exercise signed operators, SIMD edge groups, padding, and 1/4/8 workers."""
        generator = np.random.default_rng(seed=201208568)
        for nleft, nright, nell in ((1, 1, 1), (3, 5, 7), (9, 7, 103), (20, 20, 50001)):
            left = generator.normal(size=(nleft, nell))
            right = generator.normal(size=(nright, nell))
            harmonic = generator.normal(size=nell)*1.e-12
            expected = self.reference.project_block(
                kernel_left=left, kernel_right=right, harmonic=harmonic
            )
            first = None
            for threads in (1, 4, 8):
                self.library.omp_set_num_threads(threads)
                actual = self.project(left=left, right=right, harmonic=harmonic)
                # Cancellation can make one element nearly zero. Bound the
                # error by the absolute sum of its products, not by that zero.
                absolute_sum = self.reference.project_block(
                    kernel_left=np.abs(left), kernel_right=np.abs(right),
                    harmonic=np.abs(harmonic),
                )
                self.assertTrue(np.all(np.abs(actual-expected) <= 3.e-13*absolute_sum))
                if first is None:
                    first = actual
                else:
                    np.testing.assert_array_equal(actual, first)
                repeated = self.project(left=left, right=right, harmonic=harmonic)
                np.testing.assert_array_equal(actual, repeated)

    def test_pure_noise_catalogs_and_components(self):
        """Audit ordered pairs, shear components, reversed pairs and zero cross blocks."""
        pointer = ctypes.POINTER(ctypes.c_double)
        noise = np.array([0.09/2.e7, 0.04/3.e7])
        area = 2.e-6
        direct_ids = np.array([2, 3, 2, 3], dtype=np.int32)
        reversed_ids = np.array([2, 3, 3, 2], dtype=np.int32)
        auto_ids = np.array([2, 2, 2, 2], dtype=np.int32)
        # The coefficients count independent shear components and catalog
        # pairings: xi± has two components, gamma_t one, and w no shear.
        direct_coefficients = (2, 2, 1, 1)
        reversed_coefficients = (2, 2, 0, 1)
        auto_coefficients = (4, 4, 1, 2)
        for ids, coefficients in (
            (direct_ids, direct_coefficients),
            (reversed_ids, reversed_coefficients),
            (auto_ids, auto_coefficients),
        ):
            pair_noise = noise.copy()
            if ids[0] == ids[1]:
                pair_noise[1] = pair_noise[0]
            for left in range(4):
                for right in range(4):
                    # An auto pair cannot have both a lens and a source ID.
                    # gamma_t is only defined with lens-first/source-second IDs.
                    if np.array_equal(ids, auto_ids) and (left == 2 or right == 2):
                        continue
                    actual = self.library.gaussian_noise_pair_cov(
                        left, right, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        pair_noise.ctypes.data_as(pointer), area,
                    )
                    expected = 0.0
                    if left == right:
                        expected = coefficients[left]*np.prod(pair_noise)/area
                    coefficient = actual/np.prod(pair_noise)*area
                    expected_coefficient = expected/np.prod(pair_noise)*area
                    self.assertAlmostEqual(coefficient, expected_coefficient)

    def test_noise_dominated_signal_survives(self):
        """Compare with arbitrary precision when subtracting NN would erase signal."""
        import mpmath as mp

        for signal_value in (0.0, 1.e-30, 1.e-15, 1.e-7):
            signal = np.full((1, 1, 5), signal_value)
            noise = np.array([1.e-4])
            actual = self.wick(
                signal=signal, noise=noise, left=(0, 0), right=(0, 0),
                ell_min=0, include_nn=False,
            )
            with mp.workdps(70):
                total = mp.mpf(signal_value) + mp.mpf(noise[0])
                moment = 2*(total**2 - mp.mpf(noise[0])**2)
                expected = []
                for ell in range(5):
                    expected.append(float(moment/((2*ell+1)*mp.mpf(0.2))))
            np.testing.assert_allclose(actual, expected, rtol=5.e-15, atol=0.0)

    def test_spherical_white_noise_completeness(self):
        """A resolved integer-ell sum approaches the analytic clustering noise."""
        # Broad rings converge quickly enough to test the all-ell identity.
        # Small production bins require many more ell modes: their noise
        # should therefore use the analytic term, not this truncated sum.
        edges = np.array([0.2, 0.35, 0.6])
        kernels = self.reference.spherical_annulus_kernel(edges_rad=edges, ell_max=40000)
        ell = np.arange(kernels.shape[1])
        harmonic = 2.0/((2.0*ell+1)*0.2)
        projected = self.project(left=kernels, right=kernels, harmonic=harmonic)
        expected_diagonal = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            area = self.library.annulus_pair_area_cov(4*np.pi*0.2, lower, upper)
            expected_diagonal.append(2.0/area)
        # A finite cutoff misses positive tail power. This check is a
        # completeness/convention test at 0.03%, not an ell-default choice.
        np.testing.assert_allclose(np.diag(projected), expected_diagonal, rtol=3.e-4)
        scale = np.sqrt(np.outer(expected_diagonal, expected_diagonal))
        self.assertLess(abs(projected[0, 1])/scale[0, 1], 3.e-4)

    def test_annulus_area_units_and_narrow_bins(self):
        """Check solid-angle geometry against high precision, including tiny annuli."""
        import mpmath as mp

        with mp.workdps(60):
            for lower, upper in ((0.0, np.pi), (1.e-8, 1.000001e-8), (0.01, 0.02)):
                actual = self.library.annulus_pair_area_cov(0.7, lower, upper)
                expected = 0.7*2*mp.pi*(mp.cos(lower)-mp.cos(upper))
                self.assertLess(abs(actual/float(expected)-1), 1.e-14)
        degree_to_rad = np.pi/180
        area_sr = 1200*degree_to_rad**2
        lower_rad = 2.5*degree_to_rad/60
        upper_rad = 3.0*degree_to_rad/60
        pair_area = self.library.annulus_pair_area_cov(area_sr, lower_rad, upper_rad)
        density_per_arcmin2 = 3.7
        density_per_sr = density_per_arcmin2/(degree_to_rad/60)**2
        ordered_pairs_sr = density_per_sr**2*pair_area
        annulus_arcmin2 = pair_area/area_sr/(degree_to_rad/60)**2
        ordered_pairs_arcmin = density_per_arcmin2**2*(1200*60**2)*annulus_arcmin2
        self.assertLess(abs(ordered_pairs_sr/ordered_pairs_arcmin-1), 1.e-12)


if __name__ == "__main__":
    unittest.main()
