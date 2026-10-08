"""Pin a two-lens, two-source covariance configuration for independent integration tests.

The two lens distributions overlap in redshift, and their magnification
coefficients have opposite signs. This exposes cross-bin terms that would
vanish for separated lenses or zero magnification. CAMB runs only when the
input archive is generated. Tests reuse those exact arrays through the
ordinary project setters. This setup module is separate from the independent
NumPy integration in spectra_reference.py
(cosmolike_notebook_utils/covariance/reference/).

Tests create these files in a temporary directory and remove them afterward.
The order of use: create_inputs writes the files once per test class,
initialize installs them in the compiled interface, and panel_edges
returns the radial integration panels the spectrum tests share.
"""

import json

import numpy as np


def create_inputs(directory):
    """Write the fully specified configuration, n(z), and one CAMB archive.

    Arguments:
        directory = temporary destination Path chosen by the test.
    Returns:
        Nothing; writes configuration.json, lens.nz, source.nz, camb.npz.
    """
    from cosmolike_notebook_utils.camb_cosmology import get_camb_cosmology

    # The fiducial cosmology and CAMB settings (the forecast cosmology of
    # covariance/lsst_y1_covariance.py: massless neutrinos, w = -1).
    configuration = {
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
    }
    # The test survey, two bins per sample. *_support_z = the [z_low,
    # z_high] range where each bin's n(z) is nonzero (the lens ranges
    # overlap); bias = linear bias per lens bin; magnification = the
    # magnification-bias coefficient per lens bin (opposite signs);
    # ia_amplitude = NLA amplitude per source bin; densities in objects
    # per arcmin^2; sigma_e_per_component = shape noise per ellipticity
    # component; area_deg2 = the LSST Y1 area; 20 log-spaced angular
    # bins between 2.5 and 250 arcmin; linear n(z) interpolation with
    # the z column read as sample points; nz_profile documents the
    # n(z) shape written below, tabulated every nz_step up to nz_max.
    survey = {
        "lens_support_z": [[0.2, 0.7], [0.4, 0.9]],
        "source_support_z": [[0.35, 1.1], [0.75, 2.0]],
        "bias": [1.4, 1.8],
        "magnification": [-0.6, 1.2],
        "ia_amplitude": [0.6, 0.8],
        "density_lens_arcmin2": [3.0, 4.0],
        "density_source_arcmin2": [5.0, 6.0],
        "sigma_e_per_component": [0.26, 0.29],
        "area_deg2": 12300.0,
        "theta_edges_arcmin": np.geomspace(2.5, 250.0, 21).tolist(),
        "ell_max": 50000,
        "photoz_interpolation_type": 1,
        "photoz_zmid_convention": 1,
        "nz_profile": "30 t^2 (1-t)^2/(zhi-zlo), 0<t<1",
        "nz_step": 0.001,
        "nz_max": 3.0,
    }
    # The compact beta(3,3) density has zero value and slope at both
    # support edges. Its normalization is analytic, so there is no fitted
    # normalization hidden in the generated files.
    z = np.arange(0.0, survey["nz_max"]+survey["nz_step"]/2,
                  survey["nz_step"])
    directory.mkdir(parents=True, exist_ok=True)
    for sample in ("lens", "source"):
        columns = [z]
        for lower, upper in survey[f"{sample}_support_z"]:
            fraction = (z-lower)/(upper-lower)
            density = np.zeros_like(z)
            inside = (fraction > 0.0) & (fraction < 1.0)
            t = fraction[inside]
            density[inside] = 30*t**2*(1-t)**2/(upper-lower)
            columns.append(density)
        # one z column, then one n(z) column per bin; "%.17e" (17
        # significant digits) writes every double exactly, so the file
        # reads back bit for bit
        np.savetxt(directory/f"{sample}.nz", np.column_stack(columns), fmt="%.17e")

    # **configuration passes the dictionary entries as keyword
    # arguments; names lists get_camb_cosmology's returned tuple in
    # order, under the keyword names set_cosmology expects
    tables = get_camb_cosmology(**configuration)
    names = (
        "log10k_2D", "z_2D", "lnP_linear", "lnP_nonlinear", "G", "z_G",
        "z_1D", "chi", "omegan2", "lnP_linear_cb",
    )
    arrays = {}
    for name, values in zip(names, tables):
        arrays[name] = values
    np.savez(directory/"camb.npz", **arrays)
    document = {"cosmology": configuration, "survey": survey}
    (directory/"configuration.json").write_text(
        json.dumps(document, indent=2)+"\n", encoding="utf-8"
    )


def initialize(ci, directory):
    """Install the archived fiducial through an explicitly supplied interface.

    Arguments:
        ci = imported project interface; no interface is imported here.
        directory = Path containing the four generated input files.
    Returns:
        The resolved configuration dict. Mutates the supplied core state.
    """
    configuration = json.loads(
        (directory/"configuration.json").read_text(encoding="utf-8")
    )
    survey = configuration["survey"]
    cosmology = configuration["cosmology"]
    ci.initial_setup()
    ci.init_probes(possible_probes="3x2pt")
    ci.init_IA(ia_model=0, ia_redshift_evolution=2, ia_code=0)
    ci.init_bias(bias_model=[0, 0, 0, 0, 0])
    ci.init_photoz_conventions(
        interpolation_type=survey["photoz_interpolation_type"],
        zmid_convention=survey["photoz_zmid_convention"],
    )
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=str(directory/"lens.nz"),
        lens_ntomo=2,
        source_multihisto_file=str(directory/"source.nz"),
        source_ntomo=2,
    )
    # copy every array out of the .npz archive before the with-block
    # closes the file
    with np.load(directory/"camb.npz") as archive:
        inputs = {}
        for name in archive.files:
            inputs[name] = archive[name].copy()
    inputs["omegam"] = cosmology["omegam"]
    inputs["omegab"] = cosmology["omegab"]
    inputs["H0"] = cosmology["H0"]
    inputs["omegan2"] = float(inputs["omegan2"])
    ci.set_cosmology(**inputs)
    ci.set_nuisance_bias(
        B1=survey["bias"], B2=[0.0, 0.0], B_MAG=survey["magnification"],
        B3nl=[0.0, 0.0], BK=[0.0, 0.0],
    )
    ci.set_nuisance_ia(
        A1=survey["ia_amplitude"], A2=[0.0, 0.0], B_TA=[0.0, 0.0]
    )
    ci.set_nuisance_shear_photoz(bias=[0.0, 0.0])
    ci.set_nuisance_clustering_photoz(bias=[0.0, 0.0])
    ci.set_nuisance_shear_calib(M=[0.0, 0.0])
    return configuration


def panel_edges(configuration):
    """Return common scale-factor edges including every sample support edge.

    The radial integrals are split into panels at a = 1/(1+z) for
    z = 1e-5 (just short of the observer), z = 3, and the lower edge,
    midpoint and upper edge of every bin's support, so no panel straddles
    the kink of an n(z) edge. np.unique sorts the values and removes
    repeats.

    Arguments: configuration = fully specified dict from initialize.
    Returns: ascending float64 edges, with a foreground endpoint at z=1e-5.
    """
    redshifts = [1.e-5, 3.0]
    for sample in ("lens", "source"):
        for lower, upper in configuration["survey"][f"{sample}_support_z"]:
            redshifts.extend([lower, (lower+upper)/2.0, upper])
    return np.unique(1.0/(1.0+np.array(redshifts)))
