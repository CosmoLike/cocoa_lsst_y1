"""LSST Y1 survey choices for the shared covariance notebook workflow.

This module initializes five lens and five source distributions from the
project. Numerical algorithms live in cosmolike_notebook_utils.covariance.
The example is a massless-neutrino forecast with explicit Gaussian
non-Limber/IA choices and
number densities; it does not reproduce the project's frozen likelihood.
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
    SRD totals (18 lenses and 10 sources per arcmin^2) do not fix that
    allocation: the shipped n(z) columns are individually normalized.
    Arguments:
        accuracy_boost = None uses default.yaml; 1, 2, 4 or 8 refines it.
        gaussian = optional nonlimber/ia/A1/A2/B_TA model mapping.
        accuracy_overrides = named internal controls from default.yaml.
    Returns:
        Fully resolved settings, including the unboosted accuracy parameters.
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )
    # Inclusive integer bands cover 30..4000 without gaps or overlap.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)
    settings = {
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
        "lens_file": "data/lsst_y1_lens.nz",
        "source_file": "data/lsst_y1_source.nz",
        "photoz_interpolation": 1,
        "photoz_zmid": 1,
        "excluded_gammat": [],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),
        "area_deg2": 12300.0,
        "lens_density_arcmin2": [3.6]*5,
        "source_density_arcmin2": [2.0]*5,
        "sigma_e_component": [0.26]*5,
        "bias": [1.72716, 1.65168, 1.61423, 1.92886, 2.11633],
        "source_bin": 0,
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=900.0, num=27),
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

    Arguments: interface = initialized compiled project; settings = configuration();
        space = "real" or "fourier"; rows = optional measured row subset;
        progress = optional (stage, elapsed_seconds) callback;
        backend = None for notebook wrappers, interface.covariance for CLI.
    Returns: shared forecast dict, including resolved settings and coordinates.
    The full real layout has 1560 entries; Fourier has 675 entries.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )



def shear_gaussian(interface, settings):
    """Bind LSST survey choices to the shared single-source shear calculation.

    Arguments: interface = initialized project; settings = configuration().
    Returns: the shared result dict, plus theta_arcmin bin centers.
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
