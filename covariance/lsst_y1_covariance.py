"""Supplies the LSST Y1 survey choices to the shared covariance code.

The covariance matrix C of the data vector gives the expected scatter of
each measured correlation function and the correlations between entries;
the likelihood's chi2 uses its inverse. The shared package
cosmolike_notebook_utils.covariance computes C as the sum of a Gaussian term
(G: cosmic variance, shape noise and shot noise), the super-sample
covariance (SSC: the response to density modes larger than the survey) and
the connected non-Gaussian term (cNG, from the halo-model trispectrum).
This module is the project's adapter to that package (the "survey" argument
of its functions): it states the survey (the five lens and five source n(z)
of data/, area, number densities, shape noise, angular and Fourier binning,
fiducial cosmology) and forwards the calculation. compute_covariance.py,
the notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb and tests/covariance use it.

The example is a forecast with massless neutrinos and explicit choices for
the Gaussian term (non-Limber projection, intrinsic alignments, number
densities); it is not the covariance file (lsst_y1_cov) that the
likelihood examples and the frozen tests read.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return resolved survey, cosmology and YAML accuracy choices.

    Densities are equally allocated across the project's five bins. The
    SRD totals (the LSST DESC Science Requirements Document: 18 lenses and
    10 sources per arcmin^2) do not fix that allocation: the shipped n(z)
    columns are individually normalized. "Resolved" means that every value
    the calculation uses is written out, defaults included, so an archive
    saved with these settings records exactly what ran.

    Arguments:
        accuracy_boost = None uses the value in default.yaml; 1, 2, 4 or 8
            refines every boosted control of that file.
        gaussian = optional dictionary of Gaussian-term choices: nonlimber
            (bool, default True), ia ("none", "NLA" or "TATT", default
            "none"), A1, A2 and B_TA (IA amplitudes, default 0; a scalar or
            one value per source bin). gaussian_model checks it.
        accuracy_overrides = keyword arguments that replace single entries
            of default.yaml (for example window_accuracyboost=2); an unknown
            name raises TypeError.
    Returns:
        dict of fully resolved settings: the survey and cosmology entries
        below, the accuracy entries of default.yaml (boosted, and also
        their unboosted values) and "gaussian".
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )
    # Inclusive integer bands cover 30..4000 without gaps or overlap: 16
    # log-spaced edges give 15 Fourier bands [band_first, band_last]. With 15
    # source pairs, 25 lens-source pairs and 5 lens bins the Fourier data
    # vector has 15*(15 + 25 + 5) = 675 entries.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)
    settings = {
        # Fiducial cosmology and CAMB settings: As_1e9 = 10^9 A_s,
        # w0pwa = w0 + wa (-1 with w = -1 means wa = 0), mnu = 0 eV
        # (massless neutrinos), kmax [1/Mpc] and k_per_logint = CAMB's k
        # range and sampling, non_linear_emul = 2 = CAMB's Halofit
        # (halofit_version) for the nonlinear P(k).
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.05,
            "H0": 70.0,
            "ns": 0.965,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },
        # n(z) files, relative to projects/lsst_y1. photoz_interpolation = 1
        # interpolates n(z) linearly and photoz_zmid = 1 reads the file's z
        # column as bin midpoints (the likelihood yaml files default to 0
        # and 0: cubic spline, left bin edges).
        "lens_file": "data/lsst_y1_lens.nz",
        "source_file": "data/lsst_y1_source.nz",
        "photoz_interpolation": 1,
        "photoz_zmid": 1,
        # zero-based (lens, source) pairs left out of gamma_t: none
        "excluded_gammat": [],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        # ln(M/[Msun/h]) edges of the halo-mass integration panels
        "lnm_edges": cov.halo_mass_edges(),
        # LSST Y1 area [deg^2]; the SRD densities split equally over five
        # bins (18/5 = 3.6 lenses, 10/5 = 2.0 sources per arcmin^2); shape
        # noise per ellipticity component; the linear bias of each lens bin
        # (the LSST_B1_<i> values of the EXAMPLE yaml files)
        "area_deg2": 12300.0,
        "lens_density_arcmin2": [3.6]*5,
        "source_density_arcmin2": [2.0]*5,
        "sigma_e_component": [0.26]*5,
        "bias": [1.72716, 1.65168, 1.61423, 1.92886, 2.11633],
        # source bin (zero-based) of the single-bin example shear_gaussian
        "source_bin": 0,
        # 27 log-spaced edges: the 26 angular bins of the .dataset files
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=900.0, num=27),
        # scale-factor edges of the line-of-sight integration panels, at
        # z = 3.5, 2, 1.5, 1, 0.7, 0.4, 0.2 and 1e-5 (the last edge stops
        # just short of the observer at z = 0)
        "a_edges": 1.0/(1.0+np.array([3.5, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    Arguments:
        interface = imported cosmolike_lsst_y1_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the LSST forecast with G, SSC, connected and total matrices.

    Arguments:
        interface = the compiled project module, after initialize().
        settings = the dictionary returned by configuration().
        space = "real" (angular bins) or "fourier" (E-mode bandpowers).
        rows = optional subset of measured rows, an int32 [n_rows, 3] table
            of (type, A, B); None selects the full layout.
        progress = optional function called as progress(stage,
            elapsed_seconds) while the calculation runs.
        backend = None for the notebook wrappers, interface.covariance for
            the direct production bindings (compute_covariance.py).
    Returns:
        the shared forecast dict (compute_forecast): the G, SSC, cNG and
        total matrices, mean signals, the coordinate of each row (bin
        centers in arcmin or band centers in multipole) and the resolved
        settings. The full real layout has 1560 entries; Fourier has 675.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )



def shear_gaussian(interface, settings):
    """Bind LSST survey choices to the shared single-source shear calculation.

    Computes the Gaussian covariance of xi_+ and xi_- for the one source bin
    settings["source_bin"] (Limber spectra, no intrinsic alignments, no SSC
    or cNG), a small example for checking the Gaussian term. The survey
    area is converted from deg^2 to steradians and the angular edges from
    arcmin to radians, the units of the shared function.

    Arguments:
        interface = the compiled project module, after initialize().
        settings = the dictionary returned by configuration().
    Returns:
        the dict of cosmolike_notebook_utils.covariance.shear_gaussian
        (matrices [2*n_theta, 2*n_theta], xi_+ rows first, and theta_rad
        bin centers), plus theta_arcmin, the same centers in arcmin.
    """
    noise = cov.noise_powers(
        lens_density=settings["lens_density_arcmin2"],
        source_density=settings["source_density_arcmin2"],
        sigma_component=settings["sigma_e_component"],
    )
    result = cov.shear_gaussian(
        interface=interface,
        source=settings["source_bin"],
        ell_max=settings["ell_max"],
        area=settings["area_deg2"]*(np.pi/180.0)**2,
        edges_rad=settings["theta_edges_arcmin"]*np.pi/(180.0*60.0),
        a_edges=settings["a_edges"],
        radial_nquad=settings["radial_nquad"],
        nwindow=settings["nwindow"],
        angle_nquad=settings["angle_nquad"],
        mask_ell_max=settings["mask_ell_max"],
        noise=noise,
    )
    result["theta_arcmin"] = result["theta_rad"]*(180.0*60.0)/np.pi
    return result
