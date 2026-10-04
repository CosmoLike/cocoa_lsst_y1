"""LSST Y1 survey choices for the shared covariance notebook workflow.

This module initializes five lens and five source distributions from the
project. Numerical algorithms live in cosmolike_notebook_utils.covariance.
The example is a massless-neutrino, zero-IA forecast with explicitly chosen
number densities; it does not reproduce the project's frozen likelihood.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.camb_cosmology import get_camb_cosmology
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=1):
    """Return resolved survey, cosmology and pilot-integration choices.

    Densities are equally allocated across the project's five bins. The
    SRD totals (18 lenses and 10 sources per arcmin^2) do not fix that
    allocation: the shipped n(z) columns are individually normalized.
    Arguments:
        accuracy_boost = 1, 2, 4 or 8; raises covariance settings together.
    Returns:
        Fully resolved settings. Boost 1 is a pilot, not a certified FoM target.
    """
    numerical = cov.covariance_accuracy(accuracy_boost=accuracy_boost)
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
    project = Path(__file__).resolve().parents[1]
    lens_file = project/"data/lsst_y1_lens.nz"
    source_file = project/"data/lsst_y1_source.nz"
    if not lens_file.is_file() or not source_file.is_file():
        raise FileNotFoundError("LSST Y1 lens/source n(z) files are required")
    cosmology = settings["cosmology"]
    # CAMB tables are built before mutating the interface. The tuple's order
    # is the documented get_camb_cosmology / set_cosmology interchange format.
    arrays = get_camb_cosmology(**cosmology)
    names = (
        "log10k_2D",
        "z_2D",
        "lnP_linear",
        "lnP_nonlinear",
        "G",
        "z_G",
        "z_1D",
        "chi",
        "omegan2",
        "lnP_linear_cb",
    )
    tables = dict(zip(names, arrays))
    interface.initial_setup()
    interface.init_probes(possible_probes="3x2pt")
    interface.init_IA(ia_model=0, ia_redshift_evolution=2, ia_code=0)
    interface.init_bias(bias_model=[0, 0, 0, 0, 0])
    interface.init_photoz_conventions(interpolation_type=1, zmid_convention=1)
    interface.init_cosmo_runmode(is_linear=False)
    interface.init_redshift_distributions_from_files(
        lens_multihisto_file=str(lens_file),
        lens_ntomo=5,
        source_multihisto_file=str(source_file),
        source_ntomo=5,
    )
    interface.set_cosmology(
        omegam=cosmology["omegam"],
        omegab=cosmology["omegab"],
        H0=cosmology["H0"],
        **tables,
    )
    zero = [0.0]*5
    interface.set_nuisance_bias(
        B1=settings["bias"], B2=zero, B_MAG=zero, B3nl=zero, BK=zero
    )
    interface.set_nuisance_ia(A1=zero, A2=zero, B_TA=zero)
    interface.set_nuisance_shear_photoz(bias=zero)
    interface.set_nuisance_clustering_photoz(bias=zero)
    interface.set_nuisance_shear_calib(M=zero)
    return tables



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
