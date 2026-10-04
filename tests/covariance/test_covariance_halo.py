"""Check covariance-owned halo mass integration against independent sums.

The physical sigma, multiplicity, bias, concentration and profile samples
are shared inputs from public core readers. The NumPy reference performs
its own quadrature construction and moment contraction. These tests do
not independently calibrate the halo fits or a massive-neutrino response.
"""

from pathlib import Path
import unittest

import numpy as np



class HaloCovariance(unittest.TestCase):
    """Check all five moments, low-mass completion and high-k convergence."""

    @classmethod
    def setUpClass(cls):
        """Initialize the pinned massless input and explicitly selected library."""
        import tempfile
        import cosmolike_lsst_y1_interface as ci
        from cosmolike_notebook_utils.covariance.reference import halo_reference
        import survey_inputs as setup

        temporary = tempfile.TemporaryDirectory(prefix="cocoa_covariance_")
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        setup.create_inputs(directory=directory/"inputs")
        cls.ci = ci
        cls.reference = halo_reference
        cls.configuration = setup.initialize(ci=ci, directory=directory/"inputs")
        import halo_inputs as adapter
        cls.inputs = adapter.HaloInputs(
            project_library=ci.__file__, covariance_library=ci.__file__
        )
        cls.original_threads = cls.inputs.core.omp_get_max_threads()
        cls.a = np.array([0.35, 0.7, 0.95])
        wave = np.array([0., 0.01, 0.1, 1., 10., 100., 300.])*2997.92458
        cls.k = np.tile(wave, (len(cls.a), 1))
        cls.edges = np.linspace(np.log(1.e6), np.log(1.e17), 9)

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's OpenMP setting after the reproducibility sweep."""
        cls.ci.set_omp_threads(cls.original_threads)

    def test_public_notebook_halo_components(self):
        """Public owned arrays preserve moments and assemble finite responses.

        The C moment sums already have an independent NumPy oracle below.
        This check exercises the C++ shapes and the shared Python batching,
        including finite-difference responses and all angular halo terms.
        """
        from cosmolike_notebook_utils import covariance as cov

        single, moments = self.inputs.compute(
            a=self.a, k=self.k, edges=self.edges, nquad=64
        )
        public = self.ci.covariance_halo_moments(
            a=self.a, k=self.k, lnm_edges=self.edges, nquad=64
        )
        np.testing.assert_array_equal(x=public[0], y=single)
        np.testing.assert_array_equal(x=public[1], y=moments)

        response = cov.halo_power_response(
            interface=self.ci, a=self.a, k=self.k[:, 1:4],
            lnm_edges=self.edges, accuracy_boost=1, mnu=0.0,
        )
        refined = cov.halo_power_response(
            interface=self.ci, a=self.a, k=self.k[:, 1:4],
            lnm_edges=self.edges, accuracy_boost=2, mnu=0.0,
        )
        self.assertEqual(response.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(response)))
        np.testing.assert_allclose(actual=response, desired=refined, rtol=5.e-4)

        trispectrum = cov.halo_trispectrum(
            interface=self.ci, a=0.7, k=self.k[1, 1:4],
            lnm_edges=self.edges, accuracy_boost=1, mnu=0.0,
        )
        self.assertEqual(trispectrum["terms"].shape, (5, 6))
        self.assertTrue(np.all(np.isfinite(trispectrum["terms"])))
        with self.assertRaises(ValueError):
            cov.halo_power_response(
                interface=self.ci, a=self.a, k=self.k[:, 1:4],
                lnm_edges=self.edges, accuracy_boost=1, mnu=0.06,
            )

    def test_independent_mass_contraction(self):
        """Use NumPy GL nodes and independent sums of supplied physical samples."""
        actual = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=64)
        samples = self.inputs.sample(a=self.a, k=self.k, edges=self.edges, nquad=64)
        expected = self.reference.moments(**samples)
        for measured, reference in zip(actual, expected):
            np.testing.assert_allclose(measured, reference, rtol=2.e-11)

    def test_unresolved_mass_and_diagonal_identity(self):
        """I11(0)=1 and the two I13 orderings agree for identical wavenumbers."""
        for minimum in (1.e6, 1.e9, 1.e11):
            edges = np.linspace(np.log(minimum), np.log(1.e17), 9)
            single, moments = self.inputs.compute(a=self.a, k=self.k,
                                                   edges=edges, nquad=128)
            np.testing.assert_allclose(single[:, 0], 1.0, rtol=0, atol=4.e-15)
            pair = 0
            for first in range(self.k.shape[1]):
                for second in range(first, self.k.shape[1]):
                    if first == second:
                        np.testing.assert_array_equal(moments[2, :, pair],
                                                      moments[3, :, pair])
                    pair += 1

    def test_mass_refinement_at_large_wavenumber(self):
        """Resolve profile oscillations through 300 h/Mpc before choosing a rule."""
        coarse = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=512)
        fine = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=1024)
        for first, second in zip(coarse, fine):
            relative = np.max(np.abs(first/second-1))
            print(f"  halo moment mass refinement: {relative:.9e}", flush=True)
            self.assertLess(relative, 1.e-6)

    def test_reference_length_dimensions(self):
        """Shared samples preserve each moment's dimension under length conversion."""
        samples = self.inputs.sample(a=self.a, k=self.k, edges=self.edges, nquad=64)
        original = self.reference.moments(**samples)
        scale = 4282.7494
        converted = dict(samples)
        converted["number_density"] = samples["number_density"]/scale**3
        converted["rho_cb"] = samples["rho_cb"]/scale**3
        result = self.reference.moments(**converted)
        np.testing.assert_allclose(original[0], result[0], rtol=1.e-13)
        for role, exponent in enumerate((3, 3, 6, 6, 9)):
            np.testing.assert_allclose(original[1][role], result[1][role]/scale**exponent,
                                       rtol=1.e-13)

    def test_small_batches_match_shared_tables(self):
        """Small k batches retain the corresponding complete-table moments.

        These small calls expose loops with too few outer iterations for
        eight workers. One, two and three k rows exercise different mass
        partitions, including an uneven split among workers. Every mass
        sum and the completion at zero wavenumber must stay bitwise equal.
        """
        self.ci.set_omp_threads(1)
        single, moments = self.inputs.compute(
            a=self.a, k=self.k, edges=self.edges, nquad=128
        )
        first, second = np.triu_indices(n=self.k.shape[1])
        for threads in (1, 2, 4, 8):
            self.ci.set_omp_threads(threads)
            for nk in (1, 2, 3):
                selected_pairs = np.flatnonzero(a=(first < nk) & (second < nk))
                for row in range(len(self.a)):
                    actual_single, actual_moments = self.inputs.compute(
                        a=self.a[row:row+1],
                        k=self.k[row:row+1, :nk],
                        edges=self.edges,
                        nquad=128,
                    )
                    expected_single = single[row:row+1, :nk]
                    expected_moments = moments[:, row:row+1, selected_pairs]
                    np.testing.assert_array_equal(
                        x=actual_single.view(np.uint64),
                        y=expected_single.view(np.uint64),
                    )
                    np.testing.assert_array_equal(
                        x=actual_moments.view(np.uint64),
                        y=expected_moments.view(np.uint64),
                    )

    def test_repeated_threads_and_state_refresh(self):
        """Repeat complete builds at 1/4/8 threads; no static covariance state."""
        original = None
        for threads in (1, 4, 8, 1, 8):
            self.ci.set_omp_threads(threads)
            actual = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=128)
            if original is None:
                original = actual
            for measured, reference in zip(actual, original):
                np.testing.assert_array_equal(measured, reference)
        changed = self.inputs.compute(a=self.a*0.99, k=self.k,
                                       edges=self.edges, nquad=128)
        self.assertFalse(np.array_equal(changed[1], original[1]))
        restored = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=128)
        for measured, reference in zip(restored, original):
            np.testing.assert_array_equal(measured, reference)


if __name__ == "__main__":
    unittest.main()
