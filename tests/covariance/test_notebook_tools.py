"""Test the portable notebook boundary and shared plotting conventions.

The notebook boundary is the set of functions EXAMPLE_EVALUATE_COVARIANCE
calls: the Python helpers of cosmolike_notebook_utils.covariance (cov
below: accuracy settings, mass panels, Gaussian blocks, tables), the
covariance_* bindings of the compiled module (ci), and the plots of
cosmolike_notebook_utils.plot_covariances. Numeric references use direct
NumPy sums, analytic polynomials and explicit Wick contractions. Figures
are drawn with a noninteractive canvas (matplotlib's "Agg" backend,
chosen before pyplot is imported) so their artists and masked ratios can
be inspected without opening a window.
"""

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from abort_check import assert_aborts
import numpy as np

import cosmolike_lsst_y1_interface as ci
from cosmolike_notebook_utils import covariance as cov
from cosmolike_notebook_utils import plot_covariances as plots
from cosmolike_notebook_utils.covariance.reference import gaussian_reference
from cosmolike_notebook_utils.covariance.reference import ssc_reference
from cosmolike_notebook_utils.covariance.accuracy import non_gaussian_multipoles


class NotebookCovariance(unittest.TestCase):
    """Check owned-array bindings and the physical meaning of plotted values."""

    def tearDown(self):
        """Release figures so independent test runs do not accumulate canvases."""
        plt.close("all")

    def test_low_mass_extension_preserves_upper_panels(self):
        """Adding small halos must not move the established mass quadrature.

        halo_mass_edges: 22 edges in ln(M/[Msun/h]) from 10^-40; the first
        eleven panels span four decades each (the low-mass tail), and the
        last nine edges are the linear-in-ln M panels from 10^6 to 10^17.
        """
        edges = cov.halo_mass_edges()
        self.assertEqual(len(edges), 22)
        self.assertEqual(edges[0], np.log(1.e-40))
        np.testing.assert_allclose(np.diff(edges[:12]), 4*np.log(10),
                                   rtol=0, atol=2.e-14)
        upper = np.linspace(start=np.log(1.e6), stop=np.log(1.e17), num=9)
        np.testing.assert_array_equal(edges[13:], upper)
        self.assertTrue(np.all(np.diff(edges) > 0.0))

    def test_single_accuracy_boost(self):
        """Refinement keeps every old interpolation node, even as cutoffs grow.

        Each doubling of accuracy_boost doubles the multipole cutoffs and
        the window intervals, halves the ln(l + 1/2) step of the
        non-Gaussian multipole grid (keeping every coarser node) and the
        response step, and leaves the quadrature rules alone; only
        boosts 1, 2, 4 and 8 are accepted.
        """
        previous = cov.covariance_accuracy(accuracy_boost=1)
        for boost in (2, 4, 8):
            current = cov.covariance_accuracy(accuracy_boost=boost)
            for key in ("ell_max", "mask_ell_max", "core_accuracyboost"):
                self.assertEqual(current[key], 2*previous[key])
            old_grid = previous["ng_ell"]
            new_grid = current["ng_ell"]
            np.testing.assert_array_equal(old_grid, new_grid[:2*len(old_grid):2])
            self.assertGreaterEqual(new_grid[-1], current["ell_max"])
            old_steps = np.diff(np.log(old_grid+0.5))
            new_steps = np.diff(np.log(new_grid+0.5))
            np.testing.assert_allclose(new_steps, old_steps[0]/2, rtol=1.e-12)
            self.assertEqual(current["nwindow"]-1, 2*(previous["nwindow"]-1))
            for key in ("radial_nquad", "angle_nquad", "halo_mass_nquad", "tree_nquad"):
                self.assertEqual(current[key], previous[key])
            self.assertEqual(current["tree_npanel"], previous["tree_npanel"])
            self.assertEqual(current["response_step"], previous["response_step"]/2)
            previous = current
        for value in (0, 3, 16, 1.5):
            with self.assertRaises(ValueError):
                cov.covariance_accuracy(accuracy_boost=value)

    def test_global_boost_multiplies_internal_refinements(self):
        """Project factors survive global refinement, including a factor three."""
        internal = {
            "ell_max": 75000,
            "non_gaussian_accuracyboost": 3,
            "window_accuracyboost": 3,
            "core_accuracyboost": 3,
            "integration_accuracy": 2,
        }
        base = cov.covariance_accuracy(accuracy_boost=1.0, **internal)
        refined = cov.covariance_accuracy(accuracy_boost=2.0, **internal)
        for name in ("radial_nquad", "angle_nquad", "halo_mass_nquad", "tree_nquad"):
            self.assertEqual(base[name], 256)
            self.assertEqual(refined[name], 256)
        for name in ("ell_max", "mask_ell_max"):
            self.assertEqual(refined[name], 2*base[name])
        self.assertEqual(refined["nwindow"]-1, 2*(base["nwindow"]-1))
        self.assertEqual(refined["response_step"], base["response_step"]/2)
        self.assertEqual(refined["core_accuracyboost"], 2*base["core_accuracyboost"])
        self.assertEqual(base["integration_accuracy"], 2)
        self.assertEqual(refined["integration_accuracy"], 2)
        self.assertEqual(refined["tree_npanel"], base["tree_npanel"])
        np.testing.assert_array_equal(
            base["ng_ell"], refined["ng_ell"][:2*len(base["ng_ell"]):2]
        )
        self.assertEqual(base["accuracy_parameters"], refined["accuracy_parameters"])

    def test_yaml_accuracy_baseline_and_explicit_overrides(self):
        """Changing just the global control retains the file's tuned baseline."""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/"default.yaml"
            filename.write_text(
                "accuracy_boost: 1\nell_max: 100000\n"
                "non_gaussian_accuracyboost: 3\nwindow_accuracyboost: 2\n"
                "integration_accuracy: 1\n"
            )
            base = cov.load_covariance_accuracy(filename=filename)
            refined = cov.load_covariance_accuracy(filename=filename, accuracy_boost=2)
            self.assertEqual(refined["radial_nquad"], base["radial_nquad"])
            self.assertEqual(refined["angle_nquad"], base["angle_nquad"])
            restored = cov.covariance_accuracy(
                accuracy_boost=refined["accuracy_boost"],
                **refined["accuracy_parameters"],
            )
            np.testing.assert_array_equal(restored["ng_ell"], refined["ng_ell"])
            with self.assertRaises(TypeError):
                cov.load_covariance_accuracy(filename=filename, radial_boost_typo=2)
            with self.assertRaises(ValueError):
                cov.load_covariance_accuracy(filename=filename, window_accuracyboost=0)

    def test_integration_level_is_independent_of_global_boost(self):
        """Every integrated sector follows the precomputed rule ladder only."""
        for level, count in enumerate((96, 128, 256, 512, 1024)):
            for boost in (1, 2, 4, 8):
                settings = cov.covariance_accuracy(
                    accuracy_boost=boost, integration_accuracy=level,
                )
                for name in ("radial_nquad", "angle_nquad", "halo_mass_nquad",
                             "tree_nquad"):
                    self.assertEqual(settings[name], count)
        for level in (-1, 5, 0.5, True):
            with self.assertRaises(ValueError):
                cov.covariance_accuracy(integration_accuracy=level)

    def test_non_gaussian_grid_does_not_move_band_endpoints(self):
        """A fixed measured cutoff trims the table without stretching cells."""
        for boost in (1, 2, 4, 8):
            settings = cov.covariance_accuracy(accuracy_boost=boost)
            grid = settings["ng_ell"]
            for cutoff in (2, 4000, settings["ell_max"]):
                retained = non_gaussian_multipoles(samples=grid, ell_max=cutoff)
                np.testing.assert_array_equal(retained, grid[:len(retained)])
                self.assertGreaterEqual(retained[-1], cutoff)
                self.assertGreaterEqual(len(retained), 2)
                if cutoff > 2:
                    self.assertLess(retained[-2], cutoff)
                self.assertFalse(np.shares_memory(retained, grid))

        for bad in ([2.0], [2., 3., np.nan], [2., 4., 3.], [2., 4., 10.]):
            with self.assertRaises(ValueError):
                non_gaussian_multipoles(samples=bad, ell_max=3)
        with self.assertRaises(ValueError):
            non_gaussian_multipoles(samples=[2., 3.], ell_max=4)

    def test_public_wick_and_rectangular_blocks(self):
        """Cross-bin spectra and mixed-noise terms match a separate contraction."""
        random = np.random.default_rng(seed=2026)
        factors = random.normal(size=(17, 4, 4))
        # einsum labels l=ell and a,b=field. A A^T is a physical PSD
        # field matrix, with nonzero cross spectra on every multipole.
        spectra = np.einsum("lak,lbk->lab", factors, factors)
        noise = np.array([0.1, 0.3, 0.2, 0.4])
        left = random.normal(size=(3, 17))
        right = random.normal(size=(5, 17))
        pairs = np.array([[0, 1], [1, 3]])
        harmonic = gaussian_reference.harmonic_covariance(
            signal=np.moveaxis(a=spectra, source=0, destination=-1),
            noise=noise, pairs=pairs, ell_min=2, fsky=0.2, include_nn=True,
        )
        expected = gaussian_reference.project_block(
            kernel_left=left, kernel_right=right, harmonic=harmonic[0, 1]
        )
        result = cov.gaussian_block(
            interface=ci, spectra=spectra, noise=noise, fields=[0, 1, 1, 3],
            left=left, right=right, ell_min=2, area_sr=0.8*np.pi,
        )
        np.testing.assert_allclose(actual=result, desired=expected, rtol=2.e-14)
        saved = result.copy()
        cov.gaussian_block(
            interface=ci, spectra=spectra*2, noise=noise, fields=[0, 1, 1, 3],
            left=left, right=right, ell_min=2, area_sr=0.8*np.pi,
        )
        np.testing.assert_array_equal(x=result, y=saved)
        def attempt():
            ci.covariance_project(left=left,
                                  right=np.ascontiguousarray(right[:, :-1]),
                                  weight=np.ones(shape=17))
        assert_aborts(attempt)

    def test_public_operators_and_pair_noise(self):
        """The Python operator ordering gives analytic full-sky pair variance."""
        edges = np.array([0.02, 0.04, 0.08])
        operators = ci.covariance_realspace_operator(
            edges_rad=edges, ell_max=30, nquad=128
        )
        area = 4*np.pi
        mask = cov.cap_mask(area_sr=area, ell_max=30)
        pair_area = ci.covariance_mask_pair_area(
            edges_rad=edges, mask_cl=mask, area_sr=area,
            scalar_kernel=np.ascontiguousarray(operators[3]),
        )
        expected_area = area*2*np.pi*(np.cos(edges[:-1])-np.cos(edges[1:]))
        np.testing.assert_allclose(actual=pair_area, desired=expected_area, rtol=1.e-12)
        # pure shape noise of an auto xi+: variance 4 N^2/pair area (two
        # shear components and two catalog pairings)
        result = cov.realspace_block(
            interface=ci, spectra=np.zeros(shape=(29, 1, 1)), noise=np.array([0.3]),
            fields=[0, 0, 0, 0], operators=operators, probe_left=0, probe_right=0,
            pair_area_sr2=pair_area, area_sr=area,
        )
        np.testing.assert_allclose(actual=result, desired=np.diag(4*0.3**2/pair_area))
        bands = ci.covariance_bandpower_operator(
            first=np.array([2, 8], dtype=np.int32),
            last=np.array([7, 30], dtype=np.int32), ell_min=2, nell=29,
        )
        np.testing.assert_allclose(actual=bands.sum(axis=1), desired=1.0)

    def test_observed_shear_and_units(self):
        """Apply one spin factor per source leg, leaving independent noise alone."""
        ell = np.array([2.0, 10.0])
        spectra = np.ones(shape=(2, 2, 2))
        converted = cov.observed_spectra(spectra=spectra, ell=ell, nlens=1)
        spin_squared = (ell-1)*(ell+2)/(ell*(ell+1))
        np.testing.assert_allclose(actual=converted[:, 1, 1], desired=spin_squared)
        np.testing.assert_allclose(actual=converted[:, 0, 1], desired=np.sqrt(spin_squared))
        np.testing.assert_array_equal(x=spectra, y=1.0)
        # white noise N_g = 1/n_g and N_s = sigma_e^2/n_s, with the
        # densities per arcmin^2 converted to per sr
        white = cov.noise_powers(lens_density=[3.6], source_density=[2.0],
                                 sigma_component=[0.26])
        radians_per_arcmin = np.pi/(180*60)
        expected = np.array([1/3.6, 0.26**2/2])*radians_per_arcmin**2
        np.testing.assert_allclose(actual=white, desired=expected)

    def test_dense_linear_lookup_off_grid(self):
        """A cubic ln-k function isolates dense linear error from coarse error.

        DenseLogTable resamples a coarse table (21 nodes) onto ndense
        nodes uniform in ln k and then interpolates linearly; doubling
        ndense must cut the error by more than three (linear
        interpolation error falls as the square of the spacing). A query
        outside the tabulated k range is refused.
        """
        k = np.geomspace(start=0.01, stop=10, num=21)
        x = np.log(k)
        values = 2+x-0.3*x**2+0.02*x**3
        query = np.geomspace(start=0.011, stop=9.9, num=251)
        coordinate = np.log(query)
        expected = 2+coordinate-0.3*coordinate**2+0.02*coordinate**3
        table = cov.DenseLogTable(k=k, values=values, ndense=1025)
        finer = cov.DenseLogTable(k=k, values=values, ndense=2049)
        error = np.max(np.abs(table(k=query)-expected))
        fine_error = np.max(np.abs(finer(k=query)-expected))
        self.assertLess(error, 1.e-5)
        self.assertLess(fine_error, error/3)
        with self.assertRaises(ValueError):
            table(k=np.array([0.001]))
        positive = cov.DenseLogTable(k=k, values=k**1.5, ndense=1025,
                                     logarithmic=True)
        np.testing.assert_allclose(actual=positive(k=query), desired=query**1.5,
                                   rtol=3.e-15)

    def test_ssc_public_projection_and_negative_modes(self):
        """A shared response preserves PSD; deleting a cross block can break it."""
        distance = np.array([0.2, 0.5, 0.9])
        power = np.ones(shape=(3, 3))*0.2
        mask = cov.cap_mask(area_sr=1.0, ell_max=2)
        variance = ci.covariance_ssc_mask_variance(
            mask_cl=mask, area_sr=1.0, distance=distance, power=power
        )
        expected = ssc_reference.mask_variance(
            mask_cl=mask, area_sr=1.0, distance=distance, power=power
        )
        np.testing.assert_allclose(actual=variance, desired=expected)
        # a covariance built as R R^T is positive definite; "bad" (an
        # arbitrary symmetric matrix) has a negative eigenvalue; the
        # generalized eigenvalues of (1.001 C, C) are all 1.001
        response = np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])
        matrix = ci.covariance_project(left=response, right=response,
                                       weight=np.ones(shape=3))
        self.assertTrue(cov.covariance_modes(matrix=matrix)["positive_definite"])
        bad = np.array([[1., .9, .9], [.9, 1., 0.], [.9, 0., 1.]])
        self.assertLess(cov.covariance_modes(matrix=bad)["eigenvalues"][0], 0)
        comparison = cov.compare_covariances(matrix=1.001*matrix, reference=matrix)
        np.testing.assert_allclose(actual=comparison["generalized_eigenvalues"],
                                   desired=1.001)

    def test_positivity_is_independent_of_observable_units(self):
        """A unit change cannot create a mode or hide a real negative one."""
        # This correlation has one eigenvalue 1.6 and three eigenvalues 0.8.
        # Vastly different units imitate joint shear/count/cluster blocks;
        # they also expose overflow in multiplying two variances first.
        correlation = 0.8*np.eye(N=4)+0.2*np.ones(shape=(4, 4))
        for deviation in (np.ones(shape=4),
                          np.array([1.e-90, 1.e-30, 1.e30, 1.e90])):
            matrix = deviation[:, None]*correlation*deviation[None, :]
            saved = matrix.copy()
            result = cov.covariance_modes(matrix=matrix)
            self.assertTrue(result["positive_definite"])
            np.testing.assert_allclose(
                actual=result["correlation_eigenvalues"],
                desired=[0.8, 0.8, 0.8, 1.6], rtol=2.e-14,
            )
            np.testing.assert_array_equal(x=matrix, y=saved)

            # Correlation > 1 violates Cauchy-Schwarz. Rescaling must not
            # turn this physically negative mode into an accepted matrix.
            negative = correlation.copy()
            negative[0, 1] = 1.2
            negative[1, 0] = 1.2
            matrix = deviation[:, None]*negative*deviation[None, :]
            result = cov.covariance_modes(matrix=matrix)
            self.assertFalse(result["positive_definite"])
            self.assertLess(result["correlation_eigenvalues"][0], -0.19)

        for diagonal in ([0., 1.], [-1., 1.]):
            result = cov.covariance_modes(matrix=np.diag(v=diagonal))
            self.assertFalse(result["positive_definite"])
            self.assertIsNone(result["correlation_eigenvalues"])

    def test_plot_ratio_masks_and_returned_artists(self):
        """Undefined ratios stay masked; signed correlations and labels survive."""
        # show=None makes each plotting function return (figure, axes)
        # instead of drawing; the checks read back the drawn arrays
        matrix = np.array([[2., -0.2], [-0.2, 1.]])
        figure, axis = plots.plot_correlation(
            covariance=matrix, covariance_ref=np.eye(N=2),
            block_sizes=[1, 1], block_labels=["A", "B"], show=None,
        )
        figure.canvas.draw()
        image = axis.images[0].get_array()
        self.assertLess(image[1, 0], 0)
        self.assertEqual(image[0, 1], 0)
        figure, axes = plots.plot_covariance_components(
            total=np.eye(N=2), components={"Half": 0.5*np.eye(N=2)},
            normalization="element", show=None,
        )
        figure.canvas.draw()
        image = axes["maps"][0].images[0].get_array()
        self.assertEqual(np.count_nonzero(np.ma.getmaskarray(image)), 2)
        figure, axes = plots.plot_covariance_diagonal(
            theta_arcmin=np.array([1., 2.]), covariances={"Test": matrix},
            panel_labels=["Shear"], covariance_ref=matrix, show=None,
            coordinate_label=r"$\ell$",
        )
        figure.canvas.draw()
        self.assertEqual(axes[0].get_xlabel(), r"$\ell$")
        np.testing.assert_array_equal(x=axes[0].lines[0].get_ydata(), y=0.0)
        with self.assertRaises(ValueError):
            plots.plot_correlation(covariance=np.diag([1., -1.]), show=None)


def test_forecast_archive_preserves_arrays_and_resolved_settings(tmp_path):
    """A saved forecast loads without pickle or substitution of current defaults."""
    from cosmolike_notebook_utils.covariance.forecast import save_forecast

    matrix = np.array([[2., -0.2], [-0.2, 1.]])
    result = {
        "gaussian": matrix,
        "ssc": 0.2*matrix,
        "cng": 0.1*matrix,
        "total": 1.3*matrix,
        "signal": np.array([[1., 2.]]),
        "ssc_normalization_signal": np.array([[0.9, 1.8]]),
        "rows": np.array([[0, 1, 1]], dtype=np.int32),
        "coordinate": np.array([10., 20.]),
        "coordinate_label": r"$\ell$",
        "geometry": np.ones((4, 3)),
        "coarse_ell": np.array([2., 40.]),
        "pair_area_sr2": np.empty(0),
        "stages_s": {"total": 1.0},
        "settings": {
            "accuracy_boost": 2,
            "band_first": np.array([5, 15], dtype=np.int32),
            "cosmology": {"mnu": 0.0},
            "space": "fourier",
        },
    }
    # allow_pickle=False refuses stored Python objects: every entry must
    # be a plain array, and the settings travel as JSON text
    output = tmp_path/"forecast.npz"
    save_forecast(result=result, filename=output)
    with np.load(output, allow_pickle=False) as saved:
        for name in ("gaussian", "ssc", "cng", "total", "signal", "rows",
                     "ssc_normalization_signal"):
            np.testing.assert_array_equal(saved[name], result[name])
        settings = json.loads(str(saved["settings_json"]))
        assert settings["band_first"] == [5, 15]
        assert settings["cosmology"]["mnu"] == 0.0
        assert settings["space"] == "fourier"
        assert settings["accuracy_boost"] == 2

    del result["ssc_normalization_signal"]
    save_forecast(result=result, filename=output)
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ssc_normalization_signal"],
                                      saved["signal"])


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead
if __name__ == "__main__":
    unittest.main()
