"""Test the portable notebook boundary and shared plotting conventions.

Numeric references use direct NumPy sums, analytic polynomials and explicit
Wick contractions. Figures are drawn with a noninteractive canvas so their
artists and masked ratios can be inspected without opening a window.
"""

import json
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

import cosmolike_lsst_y1_interface as ci
from cosmolike_notebook_utils import covariance as cov
from cosmolike_notebook_utils import plot_covariances as plots
from cosmolike_notebook_utils.covariance.reference import gaussian_reference
from cosmolike_notebook_utils.covariance.reference import ssc_reference


class NotebookCovariance(unittest.TestCase):
    """Check owned-array bindings and the physical meaning of plotted values."""

    def tearDown(self):
        """Release figures so independent test runs do not accumulate canvases."""
        plt.close("all")

    def test_single_accuracy_boost(self):
        """Every boost raises all controls, retaining a nested window grid."""
        previous = cov.covariance_accuracy(accuracy_boost=1)
        for boost in (2, 4, 8):
            current = cov.covariance_accuracy(accuracy_boost=boost)
            for key in ("ell_max", "mask_ell_max", "radial_nquad", "angle_nquad"):
                self.assertEqual(current[key], 2*previous[key])
            self.assertEqual(current["ng_ell_nodes"], 2*previous["ng_ell_nodes"])
            self.assertEqual(current["nwindow"]-1, 2*(previous["nwindow"]-1))
            self.assertLessEqual(current["angle_nquad"], 1024)
            self.assertEqual(current["halo_mass_nquad"], 2*previous["halo_mass_nquad"])
            self.assertEqual(current["tree_nquad"], 2*previous["tree_nquad"])
            self.assertEqual(current["tree_npanel"], previous["tree_npanel"]+1)
            self.assertEqual(current["response_step"], previous["response_step"]/2)
            previous = current
        for value in (0, 3, 16, 1.5):
            with self.assertRaises(ValueError):
                cov.covariance_accuracy(accuracy_boost=value)

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
        with self.assertRaises(ValueError):
            ci.covariance_project(left=left, right=np.ascontiguousarray(right[:, :-1]),
                                  weight=np.ones(shape=17))

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
        white = cov.noise_powers(lens_density=[3.6], source_density=[2.0],
                                 sigma_component=[0.26])
        radians_per_arcmin = np.pi/(180*60)
        expected = np.array([1/3.6, 0.26**2/2])*radians_per_arcmin**2
        np.testing.assert_allclose(actual=white, desired=expected)

    def test_dense_linear_lookup_off_grid(self):
        """A cubic ln-k function isolates dense linear error from coarse error."""
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
    output = tmp_path/"forecast.npz"
    save_forecast(result=result, filename=output)
    with np.load(output, allow_pickle=False) as saved:
        for name in ("gaussian", "ssc", "cng", "total", "signal", "rows"):
            np.testing.assert_array_equal(saved[name], result[name])
        settings = json.loads(str(saved["settings_json"]))
        assert settings["band_first"] == [5, 15]
        assert settings["cosmology"]["mnu"] == 0.0
        assert settings["space"] == "fourier"
        assert settings["accuracy_boost"] == 2


if __name__ == "__main__":
    unittest.main()
