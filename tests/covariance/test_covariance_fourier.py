"""Check complete bandpower forecasts and the linearity of band averaging.

The synthetic survey has overlapping lens bins and distinct source windows.
Combining two adjacent measured bands is an independent check of the whole
G/SSC/cNG projection: covariance must transform on both matrix axes.
These small grids test assembly, not numerical accuracy for a survey.
G = Gaussian, SSC = super-sample covariance, cNG = connected
non-Gaussian; a bandpower is the (2l+1)-weighted average of C_l over an
integer multipole band [band_first, band_last].
"""

import numpy as np

from cosmolike_notebook_utils import covariance as cov


def test_band_rebinning_and_thread_determinism(tmp_path):
    """Every component obeys band rebinning and repeats at one/eight threads."""
    import cosmolike_lsst_y1_interface as ci
    import survey_inputs as setup

    # tmp_path = a fresh temporary folder pytest creates for this test
    setup.create_inputs(directory=tmp_path)
    configuration = setup.initialize(ci=ci, directory=tmp_path)
    # the survey's magnification coefficients are set to zero here
    ci.set_nuisance_bias(
        B1=configuration["survey"]["bias"], B2=[0., 0.],
        B_MAG=[0., 0.], B3nl=[0., 0.], BK=[0., 0.],
    )
    # deliberately small numerical settings (few nodes, two bands
    # [10, 39] and [40, 120]): the test checks assembly, not convergence
    settings = {
        "mnu": 0.0,
        "ell_max": 120,
        "ng_ell": np.geomspace(2.5, 120.5, 16)-0.5,
        "mask_ell_max": 128,
        "radial_nquad": 64,
        "angle_nquad": 64,
        "nwindow": 1025,
        "halo_mass_nquad": 64,
        "tree_nquad": 64,
        "tree_npanel": 16,
        "response_step": 0.005,
        "area_sr": 1.0,
        "a_edges": np.linspace(0.25, 1.-1.e-6, 3),
        "lnm_edges": np.linspace(np.log(1.e6), np.log(1.e17), 5),
        "band_first": np.array([10, 40], dtype=np.int32),
        "band_last": np.array([39, 120], dtype=np.int32),
    }
    # observable rows (probe, A, B) with probe 0 xi+, 1 xi-, 2 gamma_t,
    # 3 w; Fourier space has one shear spectrum per pair, so the xi-
    # rows (probe 1) are dropped
    rows = cov.observable_rows(nlens=2, nsource=2)
    rows = np.ascontiguousarray(rows[rows[:, 0] != 1])
    noise = cov.noise_powers(
        lens_density=[3., 4.], source_density=[5., 6.],
        sigma_component=[0.26, 0.29],
    )
    baseline = None
    for threads in (1, 8):
        ci.set_omp_threads(threads)
        result = cov.fourier_covariance(
            interface=ci, settings=settings, rows=rows, noise=noise,
        )
        for name in ("gaussian", "ssc", "cng", "total"):
            assert np.all(np.isfinite(result[name]))
            np.testing.assert_array_equal(result[name], result[name].T)
            # .view(np.uint64) compares the 64-bit patterns: the 8-thread
            # matrices must equal the 1-thread ones bit for bit
            if baseline is not None:
                np.testing.assert_array_equal(
                    result[name].view(np.uint64), baseline[name].view(np.uint64)
                )
        baseline = result
    # Fourier means must average the C spectra directly with (2l+1)
    # weights; the conversion of the source leg that the real-space
    # kernels apply is not part of the Fourier mean.
    ell = np.arange(2., 121.)
    snapshot = ci.covariance_limber_spectra(
        ell=ell, a_edges=settings["a_edges"], nquad=settings["radial_nquad"],
        nwindow=settings["nwindow"], include_ia=False, include_rsd=False,
        linear=False,
    )
    for band, (first, last) in enumerate(zip(
            settings["band_first"], settings["band_last"])):
        selected = (ell >= first) & (ell <= last)
        weights = 2*ell[selected]+1
        weights /= np.sum(weights)
        for row, (probe, a, b) in enumerate(rows):
            expected = np.sum(snapshot["spectra"][selected, a, b]*weights)
            np.testing.assert_allclose(result["signal"][row, band], expected,
                                       rtol=3.e-14, atol=0.)
    assert result["pair_area_sr2"].size == 0
    assert cov.covariance_modes(matrix=result["total"])["positive_definite"]

    # A band's normalization is its number of harmonic modes. Averaging
    # adjacent band estimates therefore uses these mode-count fractions,
    # not the geometric centers or equal weights for the two bands.
    # sum of (2l+1) from l = first to last = (last+1)^2 - first^2.
    # np.kron(identity, weights row) builds the matrix that averages the
    # two bands of every row: shape [n_rows, 2 n_rows]
    modes = (settings["band_last"]+1)**2-settings["band_first"]**2
    weights = modes/np.sum(modes)
    transform = np.kron(np.eye(len(rows)), weights[None, :])
    merged = dict(settings)
    merged["band_first"] = np.array([10], dtype=np.int32)
    merged["band_last"] = np.array([120], dtype=np.int32)
    combined = cov.fourier_covariance(
        interface=ci, settings=merged, rows=rows, noise=noise,
    )
    for name in ("gaussian", "ssc", "cng", "total"):
        expected = transform @ result[name] @ transform.T
        np.testing.assert_allclose(combined[name], expected, rtol=3.e-12, atol=0.)
    expected_signal = result["signal"] @ weights
    np.testing.assert_allclose(combined["signal"][:, 0], expected_signal,
                               rtol=3.e-13, atol=0.)
