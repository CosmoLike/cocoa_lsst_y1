"""Notebook wrappers for the lsst_y1 Cosmolike interface.

The EXAMPLE_EVALUATE notebooks all drive the same compiled interface
(cosmolike_lsst_y1_interface) through the same steps: run CAMB once
(cnu.get_camb_cosmology), push the resulting power spectra and
distances into the interface with set_cosmology, set the nuisance
parameters, and read off a spectrum, a correlation function, or the
masked chi2. This module holds those steps once, so every notebook of
this project imports the same wrappers instead of redefining them:

    import cosmolike_lsst_y1_notebook_wrappers as nw
    nw.configure()                    # per-notebook yaml overrides
    nw.init_cosmolike(CLprobe="xi", with_data=True)
    nw.get_chi2(omegam=0.31)

Three layers of state matter here:

- The compiled interface is a C library with global state: every
  init_* and set_* call above replaces part of it, and the spectrum
  functions read whatever was set last. Each wrapper therefore
  resets everything it depends on (tables, accuracy, cosmology,
  nuisances) on every call, so no call depends on which wrapper ran
  before it.
- The project fiducial point lives in this module as plain
  constants (LSST_B1_1, ...), shared by every notebook; a notebook
  overrides any of them per call (nw.C_ss_tomo_limber(ell=ell,
  omegam=x)) or imports the names for its own sweeps.
- The few values that differ between notebooks because each mirrors
  its own yaml (the lmax of the internal C_ell tables, the angular
  binning, the nonlinear emulator choice) live in _CONFIG and are
  set once per notebook with configure().

Every wrapper accepts the same accuracy arguments and applies the
same folds: CLAccuracyBoost multiplies by AccuracyBoost, the
integration accuracy grows as |3 (CLAccuracyBoost - 1)|, and the
C_ell table reaches lmax + 20000 (CLAccuracyBoost - 1).
"""

import os
import sys

import numpy as np
from getdist import IniFile

# the shared notebook utilities live in cosmolike_core; the compiled
# interface is on the path already (each project's interface/
# directory is part of the Cocoa PYTHONPATH)
sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
import cosmolike_notebook_utils as cnu
import cosmolike_lsst_y1_interface as ci


# ----------------------------------------------------------------------
# Project fiducial point (the evaluate override of the example yamls)
# ----------------------------------------------------------------------
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
mnu = 0.06
w = -0.9
w0pwa = -0.9
LSST_A1_1 = 0.606102   # NLA amplitude
LSST_A1_2 = -1.51541   # NLA redshift power-law index
LSST_DZ_S1 = 0.0414632
LSST_DZ_S2 = 0.00147332
LSST_DZ_S3 = 0.0237035
LSST_DZ_S4 = -0.0773436
LSST_DZ_S5 = -8.67127e-05
LSST_M1 = 0.0191832
LSST_M2 = -0.0431752
LSST_M3 = -0.034961
LSST_M4 = -0.0158096
LSST_M5 = -0.0158096
LSST_DZ_L1 = 0.00457604
LSST_DZ_L2 = 0.000309875
LSST_DZ_L3 = 0.00855907
LSST_DZ_L4 = -0.00316269
LSST_DZ_L5 = -0.0146753
LSST_B1_1 = 1.72716
LSST_B1_2 = 1.65168
LSST_B1_3 = 1.61423
LSST_B1_4 = 1.92886
LSST_B1_5 = 2.11633
LSST_PM_1 = 0.0
LSST_PM_2 = 0.0
LSST_PM_3 = 0.0
LSST_PM_4 = 0.0
LSST_PM_5 = 0.0

# default nuisance vectors built from the constants above; wrappers
# take None and fall back to these, so a call overrides one vector
# without retyping the rest
A1_FID = [LSST_A1_1, LSST_A1_2, 0, 0, 0]
A2_FID = [0, 0, 0, 0, 0]
BTA_FID = [0, 0, 0, 0, 0]
SHEAR_PHOTOZ_FID = [LSST_DZ_S1, LSST_DZ_S2, LSST_DZ_S3, LSST_DZ_S4,
                    LSST_DZ_S5]
M_FID = [LSST_M1, LSST_M2, LSST_M3, LSST_M4, LSST_M5]
LENS_PHOTOZ_FID = [LSST_DZ_L1, LSST_DZ_L2, LSST_DZ_L3, LSST_DZ_L4,
                   LSST_DZ_L5]
B1_FID = [LSST_B1_1, LSST_B1_2, LSST_B1_3, LSST_B1_4, LSST_B1_5]
ZEROS5 = [0, 0, 0, 0, 0]
PM_FID = [LSST_PM_1, LSST_PM_2, LSST_PM_3, LSST_PM_4, LSST_PM_5]

# ----------------------------------------------------------------------
# Per-notebook configuration
# ----------------------------------------------------------------------
# Every notebook mirrors its own yaml; the current three agree on
# every entry here (lmax = 50000 for the shear and 3x2pt notebooks
# alike), so configure() exists for per-notebook overrides the same
# way it does in the other projects.
_CONFIG = {
    "lmax": 50000,              # base of the internal C_ell tables
    "ntheta": 26,               # angular bins of the real-space vector
    "theta_min_arcmin": 2.5,
    "theta_max_arcmin": 900.0,
    "non_linear_emul": 2,       # 1 = EuclidEmulator2, 2 = halofit
    "path": "../../external_modules/data/lsst_y1",
    "data_file": "lsst_y1_M1_GGL0.05.dataset",
    "ggl_exclude": [],          # this project keeps every ggl pair
    "IA_model": 0,
    "IA_redshift_evolution": 3,
    "IA_code": 0,               # 0 = C FASTPT (NLA always uses 0)
    "bias_model": [0, 0, 0, 1, 0],    # n(z) photo-z conventions (mirror the likelihood yaml keys):
    # interpolation 0 = cspline, 1 = linear, 2+ = Steffen monotone;
    # z column 0 = Z_LOW (left bin edges), 1 = Z_MID (sample points)
    "photoz_interpolation_type": 0,
    "photoz_zmid_convention": 0,
    # C-FAST-PT internal (convolution) grid / output grid; 1.0 = equal
    "internal_accuracyboost": 1.0,

}

# filled by init_cosmolike: the HDF5 file with every hydro simulation
# used by init_baryons_contamination
allsims = None


def configure(**overrides):
    """Sets this notebook's yaml-mirroring values, once per notebook.

    Every keyword must already exist in _CONFIG; an unknown name is
    almost always a typo, so it raises instead of being stored
    silently.

    Arguments:
      overrides = keyword form of any _CONFIG entry, e.g.
                  configure(lmax=70000).

    Returns:
      nothing; later wrapper calls read the stored values.

    Raises:
      KeyError naming the unknown keyword and the valid names.
    """
    for name, value in overrides.items():
        if name not in _CONFIG:
            raise KeyError(
                f"configure() got unknown option '{name}'; valid options: "
                + ", ".join(sorted(_CONFIG)))
        _CONFIG[name] = value


def init_cosmolike(CLprobe=None, with_data=False, lmax=None):
    """One-time interface setup for a notebook session.

    Reads the project's .dataset file (the small text file listing
    the n(z), covariance, mask, and data-vector files), then runs
    the interface init sequence every notebook shares: the excluded
    ggl pairs, the angular binning, the n(z) tables, and the IA
    model. The chi2 machinery (probes, covariance, mask, data
    vector) only loads when asked, because the plotting-only
    notebooks never need it.

    Arguments:
      CLprobe   = "xi", "3x2pt", ... to select the probe set and
                  (except for "xi") the galaxy-bias model, or None
                  to skip init_probes for a plotting-only session.
      with_data = True also loads covariance, mask, and data vector,
                  which get_chi2 and compute_probes need.
      lmax      = base lmax of the internal C_ell tables, or None
                  for the configure()d value.

    Returns:
      the parsed IniFile, for notebooks that read extra entries.

    Side effects:
      replaces the compiled interface's global state and sets this
      module's allsims to the hydro-simulation file of the dataset.
    """
    global allsims
    if lmax is None:
        lmax = _CONFIG["lmax"]
    ini = IniFile(os.path.normpath(
        os.path.join(_CONFIG["path"], _CONFIG["data_file"])))
    allsims = ini.relativeFileName('all_sims_hdf5_file')
    ci.initial_setup()
    if _CONFIG["ggl_exclude"]:
        # flatten() turns the [[lens, source], ...] pair list into
        # the flat [l0, s0, l1, s1, ...] array the C layer expects
        ci.init_ggl_exclude(np.array(_CONFIG["ggl_exclude"]).flatten())
    if CLprobe is not None:
        ci.init_probes(possible_probes=CLprobe)
    ci.init_binning(int(ini.int("n_theta")),
                    ini.float("theta_min_arcmin"),
                    ini.float("theta_max_arcmin"))
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_IA(ia_model=int(_CONFIG["IA_model"]),
               ia_redshift_evolution=int(_CONFIG["IA_redshift_evolution"]),
               ia_code=int(_CONFIG["IA_code"]))
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ini.relativeFileName('nz_lens_file'),
        lens_ntomo=int(ini.int("lens_ntomo")),
        source_multihisto_file=ini.relativeFileName('nz_source_file'),
        source_ntomo=int(ini.int("source_ntomo")))
    if with_data:
        ci.init_data_real(ini.relativeFileName('cov_file'),
                          ini.relativeFileName('mask_file'),
                          ini.relativeFileName('data_file'))
    if CLprobe is not None and CLprobe != "xi":
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    ci.init_ntable_lmax(lmax=int(lmax))
    ci.init_accuracy_boost(1.0, int(1))
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    return ini


def _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=None, M=None, shear_photoz_bias=None,
               A1=None, A2=None, BTA=None,
               lens_photoz_bias=None, B1=None, B2=None,
               B_MAG=None, B3nl=None, BK=None, PM=None,
               baryon_sims=None, allsims_file=None):
    """Runs CAMB and pushes one complete state into the interface.

    This is the body every wrapper shares. The compiled interface
    keeps global state, so the sequence rebuilds everything a
    spectrum call reads: the accuracy folds and lookup tables, the
    binning when a real-space probe asked for it, the cosmology
    (power spectra, growth, distances from one CAMB run), and each
    nuisance group whose vectors were passed. A group passed as None
    is skipped, which leaves that part of the state at whatever the
    interface holds, exactly as the per-probe notebook definitions
    did (a cosmic-shear wrapper never touched galaxy bias).

    Arguments:
      omegam ... non_linear_emul = the cosmology and accuracy
                 arguments, forwarded to cnu.get_camb_cosmology
                 (kmax in h/Mpc; see its docstring for the grids).
      binning  = (ntheta, theta_min_arcmin, theta_max_arcmin) to
                 re-run init_binning (the real-space wrappers), or
                 None to keep the current binning.
      M, shear_photoz_bias, A1, A2, BTA = shear nuisance vectors
                 (M and the photo-z shifts gate the shear setters;
                 A1 gates the IA setter).
      lens_photoz_bias, B1, B2, B_MAG, B3nl, BK = clustering
                 nuisance vectors; passing them also selects the
                 configured galaxy-bias model.
      PM       = point-mass amplitudes, one per lens bin, or None.
      baryon_sims = a hydro simulation name to contaminate the
                 matter power with, or None to reset that state.
      allsims_file = HDF5 file for baryon_sims, or None for the one
                 init_cosmolike recorded.

    Returns:
      nothing; the interface state is the result.
    """
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_interp_1D, chi) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, AccuracyBoost=AccuracyBoost,
        kmax=kmax, k_per_logint=k_per_logint,
        CAMBAccuracyBoost=CAMBAccuracyBoost,
        CLAccuracyBoost=CLAccuracyBoost,
        non_linear_emul=non_linear_emul)
    # the house accuracy folds: the overall boost multiplies the
    # cosmolike boost, and the integration accuracy and the C_ell
    # table length grow with it
    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(
        0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost - 1.0)))
    ci.init_ntable_lmax(int(_CONFIG["lmax"] + 20000*(CLAccuracyBoost - 1)))
    ci.init_accuracy_boost(CLAccuracyBoost, int(CLIntegrationAccuracy))
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    if binning is not None:
        ci.init_binning(int(binning[0]), binning[1], binning[2])
    if B1 is not None:
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    ci.set_cosmology(omegam=omegam,
                     H0=H0,
                     log10k_2D=log10k_interp_2D,
                     z_2D=z_interp_2D,
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_1D=z_interp_1D,
                     chi=chi)
    if M is not None:
        ci.set_nuisance_shear_calib(M=M)
    if shear_photoz_bias is not None:
        ci.set_nuisance_shear_photoz(bias=shear_photoz_bias)
    if lens_photoz_bias is not None:
        ci.set_nuisance_clustering_photoz(bias=lens_photoz_bias)
    if B1 is not None:
        ci.set_nuisance_bias(B1=B1, B2=B2, B_MAG=B_MAG, B3nl=B3nl, BK=BK)
    if A1 is not None:
        ci.set_nuisance_ia(A1=A1, A2=A2, B_TA=BTA)
    if PM is not None:
        ci.set_point_mass(PMV=PM)
    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        if allsims_file is None:
            allsims_file = allsims
        ci.init_baryons_contamination(sim=baryon_sims, allsims=allsims_file)


def _shear_defaults(M, shear_photoz_bias, A1, A2, BTA):
    """Replaces None shear vectors with the fiducial ones."""
    if M is None:
        M = M_FID
    if shear_photoz_bias is None:
        shear_photoz_bias = SHEAR_PHOTOZ_FID
    if A1 is None:
        A1 = A1_FID
    if A2 is None:
        A2 = A2_FID
    if BTA is None:
        BTA = BTA_FID
    return M, shear_photoz_bias, A1, A2, BTA


def _clustering_defaults(lens_photoz_bias, B1, B2, B_MAG, B3nl, BK):
    """Replaces None clustering vectors with the fiducial ones."""
    if lens_photoz_bias is None:
        lens_photoz_bias = LENS_PHOTOZ_FID
    if B1 is None:
        B1 = B1_FID
    if B2 is None:
        B2 = ZEROS5
    if B_MAG is None:
        B_MAG = ZEROS5
    if B3nl is None:
        B3nl = ZEROS5
    if BK is None:
        BK = ZEROS5
    return lens_photoz_bias, B1, B2, B_MAG, B3nl, BK


def C_ss_tomo_limber(ell, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
                     k_per_logint=10, CAMBAccuracyBoost=1.0,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """Cosmic-shear angular power spectra (EE, BB) at multipoles ell.

    Rebuilds the full interface state (see _set_state) and evaluates
    ci.C_ss_tomo_limber. The nuisance vectors default to the module
    fiducials when passed as None.

    Arguments:
      ell = 1D array of multipoles; the rest as in _set_state, with
      the shear group only (this wrapper never touches clustering).

    Returns:
      (EE, BB): two 3D arrays (n_ell, n_bin, n_bin).
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_ss_tomo_limber(l=ell)


def xi(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
       omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
       w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
       shear_photoz_bias=None, M=None, baryon_sims=None,
       AccuracyBoost=1.0, kmax=5.0, k_per_logint=10,
       CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
       CLIntegrationAccuracy=0, non_linear_emul=None, allsims=None):
    """Real-space shear correlations xi_plus/minus on a theta grid.

    Same state build as C_ss_tomo_limber plus a re-binning, so the
    binning can change between calls without restarting the kernel;
    the binning arguments default to the configure()d values.

    Returns:
      (theta, xi_plus, xi_minus): theta in arcmin, xi 3D arrays
      (n_theta, n_bin, n_bin).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    (xip, xim) = ci.xi_pm_tomo()
    return (ci.get_binning_real_space(), xip, xim)


def C_gs_tomo_limber(ell, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     lens_photoz_bias=None, galaxy_bias_b1=None,
                     galaxy_bias_b2=None, galaxy_bias_bmag=None,
                     galaxy_bias_b3nl=None, galaxy_bias_bk=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
                     k_per_logint=10, CAMBAccuracyBoost=1.0,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """Galaxy-galaxy lensing spectra C_gs at multipoles ell.

    Builds the shear AND clustering state (both samples enter ggl)
    and evaluates ci.C_gs_tomo_limber. Pairs dropped at init time
    via ggl_exclude come back as identically zero.

    Returns:
      3D array (n_ell, n_lens, n_source).
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_gs_tomo_limber(l=ell)


def C_gg_tomo(ell, limber, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
              As_1e9=As_1e9, w=w, w0pwa=w0pwa,
              lens_photoz_bias=None, galaxy_bias_b1=None,
              galaxy_bias_b2=None, galaxy_bias_bmag=None,
              galaxy_bias_b3nl=None, galaxy_bias_bk=None,
              baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
              k_per_logint=10, CAMBAccuracyBoost=1.0,
              CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
              non_linear_emul=None, allsims=None):
    """Galaxy-clustering spectra C_gg at multipoles ell.

    Clustering state only. limber = 1 evaluates the Limber
    approximation, anything else the non-Limber computation, so the
    two can be compared on the same state.

    Returns:
      3D array (n_ell, n_lens, n_lens).
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    if limber == 1:
        return ci.C_gg_tomo_limber(l=ell)
    else:
        return ci.C_gg_tomo(l=ell)


def gamma_t(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
            omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
            w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
            shear_photoz_bias=None, M=None,
            lens_photoz_bias=None, galaxy_bias_b1=None,
            galaxy_bias_b2=None, galaxy_bias_bmag=None,
            galaxy_bias_b3nl=None, galaxy_bias_bk=None, PM=None,
            baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
            k_per_logint=10, CAMBAccuracyBoost=1.0,
            CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
            non_linear_emul=None, allsims=None):
    """Real-space tangential shear gamma_t on a theta grid.

    Full 3x2pt nuisance state (shear, clustering, point masses) plus
    a re-binning, then ci.w_gammat_tomo.

    Returns:
      (theta, gammat): theta in arcmin, gammat a 3D array
      (n_theta, n_lens, n_source).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    if PM is None:
        PM = PM_FID
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk, PM=PM,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return (ci.get_binning_real_space(), ci.w_gammat_tomo())


def w_theta(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
            omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
            w=w, w0pwa=w0pwa,
            lens_photoz_bias=None, galaxy_bias_b1=None,
            galaxy_bias_b2=None, galaxy_bias_bmag=None,
            galaxy_bias_b3nl=None, galaxy_bias_bk=None,
            baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
            k_per_logint=10, CAMBAccuracyBoost=1.0,
            CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
            non_linear_emul=None, allsims=None):
    """Real-space clustering w(theta) on a theta grid.

    Clustering state plus a re-binning, then ci.w_gg_tomo.

    Returns:
      (theta, wtheta): theta in arcmin, wtheta a 3D array
      (n_theta, n_lens, n_lens); the panels read the diagonal.
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return (ci.get_binning_real_space(), ci.w_gg_tomo())


def get_chi2(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
             w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
             shear_photoz_bias=None, M=None,
             lens_photoz_bias=None, galaxy_bias_b1=None,
             galaxy_bias_b2=None, galaxy_bias_bmag=None,
             galaxy_bias_b3nl=None, galaxy_bias_bk=None, PM=None,
             baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
             k_per_logint=10, CAMBAccuracyBoost=1.0,
             CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
             non_linear_emul=None, allsims=None):
    """chi2 of the masked data vector against the loaded data.

    Requires init_cosmolike(CLprobe=..., with_data=True) first: the
    probe selection fixes which blocks enter the masked vector, and
    with_data loads the covariance, mask, and data vector this chi2
    compares against. The full 3x2pt nuisance state is set every
    call; blocks outside the selected probes simply never read
    theirs (a "xi" run ignores the clustering state).

    Returns:
      float chi2.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    if PM is None:
        PM = PM_FID
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk, PM=PM,
               baryon_sims=baryon_sims, allsims_file=allsims)
    datavector = np.array(ci.compute_data_vector_masked())
    return ci.compute_chi2(datavector)


# ----------------------------------------------------------------------
# Fisher forecasting (cosmic shear)
# ----------------------------------------------------------------------
# One flat parameter vector drives the Fisher machinery: the sampled
# cosmology, the two NLA numbers, the eight source photo-z shifts,
# and the eight shear calibrations, in this order. The names, LaTeX
# labels, and priors below index into the same vector.
FISHER_PARAM_FID = np.array(
    [As_1e9, ns, H0, omegab, omegam, LSST_A1_1, LSST_A1_2,
     LSST_DZ_S1, LSST_DZ_S2, LSST_DZ_S3, LSST_DZ_S4, LSST_DZ_S5,
     LSST_M1, LSST_M2, LSST_M3, LSST_M4, LSST_M5], dtype="float64")

FISHER_PARAM_NAMES = [
    "As_1e9", "ns", "H0", "omegab", "omegam", "A1_1", "A1_2",
    "zshift_1", "zshift_2", "zshift_3", "zshift_4", "zshift_5",
    "M_1", "M_2", "M_3", "M_4", "M_5",
]

FISHER_PARAM_LABELS = [
    r"10^{9} A_s", r"n_s", r"H_0", r"\Omega_b", r"\Omega_m",
    r"A_\mathrm{1IA,LSST}^1", r"A_\mathrm{1IA,LSST}^2",
    r"\Delta z_\mathrm{s,LSST}^1", r"\Delta z_\mathrm{s,LSST}^2",
    r"\Delta z_\mathrm{s,LSST}^3", r"\Delta z_\mathrm{s,LSST}^4",
    r"\Delta z_\mathrm{s,LSST}^5",
    r"m_\mathrm{LSST}^1", r"m_\mathrm{LSST}^2", r"m_\mathrm{LSST}^3",
    r"m_\mathrm{LSST}^4", r"m_\mathrm{LSST}^5",
]

# flat priors clip the getdist contours of the unbounded parameters
FISHER_FLAT_PRIORS = {
    0: (0.5, 5),
    1: (0.87, 1.07),
    2: (55.0, 91.0),
    3: (0.03, 0.07),
    4: (0.1, 0.9),
    5: (-5.0, 5.0),
    6: (-5.0, 5.0),
}

# Gaussian priors on the calibration nuisances, centered on the
# fiducial point of this project (index: (mean, sigma))
FISHER_GAUSSIAN_PRIORS = {
    12: (0.0191832, 0.005),
    13: (-0.0431752, 0.005),
    14: (-0.034961, 0.005),
    15: (-0.0158096, 0.005),
    16: (-0.0158096, 0.005),
    7: (0.0414632, 0.002),
    8: (0.00147332, 0.002),
    9: (0.0237035, 0.002),
    10: (-0.0773436, 0.002),
    11: (-8.67127e-05, 0.002),
}

# getdist samples for the Fisher contours come from this generator;
# module-level so repeated plot calls consume one stream, as the
# in-notebook implementation did with its own seeded generator
_FISHER_RNG = np.random.default_rng(0)


def fisher_fiducial_point():
    """A fresh copy of the Fisher fiducial vector.

    Returns a copy so a notebook can shift entries for a forecast
    without editing the module's fiducial in place.
    """
    return FISHER_PARAM_FID.copy()


def get_dv(param=None, AccuracyBoost=1.0):
    """Masked cosmic-shear data vector at a flat parameter vector.

    The Fisher derivatives evaluate this at shifted copies of the
    fiducial vector; the layout is the one FISHER_PARAM_NAMES
    documents. w0wa is pinned to the cosmological constant here: the
    forecast of this notebook family does not open the dark-energy
    parameters (the project fiducial w = -0.9 applies to the
    likelihood evaluations, not this forecast).

    Arguments:
      param = 1D float array in the FISHER_PARAM_NAMES layout, or
              None for the fiducial vector.
      AccuracyBoost = the house boost, forwarded to the state build.

    Returns:
      1D float64 array: the masked data vector.
    """
    if param is None:
        param = FISHER_PARAM_FID
    A1 = [param[5], param[6], 0, 0, 0]
    shear_photoz_bias = [param[7], param[8], param[9], param[10],
                         param[11]]
    M = [param[12], param[13], param[14], param[15], param[16]]
    _set_state(param[4], param[3], param[2], param[1], param[0],
               -1.0, -1.0,
               AccuracyBoost, 5.0, 10, 1.0,
               AccuracyBoost, 0, _CONFIG["non_linear_emul"],
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2_FID, BTA=BTA_FID)
    return np.array(ci.compute_data_vector_masked(), dtype=np.float64)


def get_ddv(index=0, h=0.02, CV=None, AccuracyBoost=1.0):
    """Derivative of get_dv along one parameter (cnu.get_ddv)."""
    if CV is None:
        CV = FISHER_PARAM_FID
    return cnu.get_ddv(get_dv, index=index, h=h, CV=CV,
                       AccuracyBoost=AccuracyBoost)


def get_Fisher(CV=None, h=0.02, AccuracyBoost=3.1, ddv=None,
               priors=None, invcov=None):
    """Fisher matrix from the 5-point-stencil derivatives.

    Thin binding of cnu.get_Fisher to this project's data vector,
    priors, and masked inverse covariance (fetched from the
    interface when not passed, so init_cosmolike with data must
    have run).
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    if ddv is None:
        ddv = get_ddv
    if priors is None:
        priors = FISHER_GAUSSIAN_PRIORS
    if invcov is None:
        invcov = ci.get_inv_cov_masked()
    return cnu.get_Fisher(ddv, CV=CV, h=h, AccuracyBoost=AccuracyBoost,
                          priors=priors, invcov=invcov)


def get_ddv_dkit(index=0, CV=None, AccuracyBoost=1.0, min_samples=7,
                 fallback_mode="poly_at_floor"):
    """Derivative of get_dv via derivkit (cnu.get_ddv_dkit)."""
    if CV is None:
        CV = FISHER_PARAM_FID
    return cnu.get_ddv_dkit(get_dv, index=index, CV=CV,
                            AccuracyBoost=AccuracyBoost,
                            min_samples=min_samples,
                            fallback_mode=fallback_mode)


def get_Fisher2(CV=None, AccuracyBoost=3.0, priors=None, invcov=None,
                min_samples=7, fallback_mode="poly_at_floor"):
    """Fisher matrix from the derivkit derivatives (cnu.get_Fisher2)."""
    if CV is None:
        CV = FISHER_PARAM_FID
    if priors is None:
        priors = FISHER_GAUSSIAN_PRIORS
    if invcov is None:
        invcov = ci.get_inv_cov_masked()
    return cnu.get_Fisher2(get_dv, CV=CV, AccuracyBoost=AccuracyBoost,
                           priors=priors, invcov=invcov,
                           min_samples=min_samples,
                           fallback_mode=fallback_mode)


def plot_Fisher(F, mu, F2=None, root=None, select=None, labels=None,
                names=None, filled=True, flat_priors=None,
                chain_names=None):
    """Fisher contour triangle via getdist (cnu.plot_Fisher)."""
    if labels is None:
        labels = FISHER_PARAM_LABELS
    if names is None:
        names = FISHER_PARAM_NAMES
    if flat_priors is None:
        flat_priors = FISHER_FLAT_PRIORS
    return cnu.plot_Fisher(F, mu, F2=F2, root=root, select=select,
                           labels=labels, names=names, filled=filled,
                           flat_priors=flat_priors,
                           chain_names=chain_names, rng=_FISHER_RNG)


# ----------------------------------------------------------------------
# Baryonic feedback via the bfmt theory block
# ----------------------------------------------------------------------
def get_baryon_suppression(theory_options, point, z_grid, log10k_grid):
    """Suppression S(k, z) from the bfmt theory block, on our grid.

    Builds a minimal Cobaya model (CAMB + bfmt + the unit
    likelihood), requests the baryon_suppression product at the
    given grid (k in 1/Mpc, as the Cosmolike likelihoods send it;
    the block converts to h/Mpc internally), evaluates it at this
    module's fiducial cosmology, and returns {z: S array over k}.

    Arguments:
      theory_options = bfmt options dict, e.g. {"baryon_model": 2}.
      point   = {parameter name: value} for the method's feedback
                parameters, fixed in the model.
      z_grid, log10k_grid = the evaluation grid.

    Returns:
      {z: 1D S array over k}, one entry per z_grid value.
    """
    from cobaya.model import get_model
    info = {
        "likelihood": {"one": None},
        "theory": {
            # no "path" for camb: the session already imported it,
            # and cobaya accepts the loaded module as is
            "camb": {"extra_args": {"halofit_version": "takahashi",
                                    "dark_energy_model": "ppf"}},
            "bfmt": dict({"python_path": os.environ["ROOTDIR"]
                          + "/external_modules/code/baryon_suppression"},
                         **theory_options),
        },
        "params": dict({
            "As": {"value": As_1e9*1e-9},
            "ns": ns, "H0": H0, "mnu": mnu, "tau": 0.0543,
            "w": w,
            "ombh2": omegab*(H0/100)**2,
            "omch2": (omegam-omegab)*(H0/100)**2
                     - (mnu*(3.046/3)**0.75)/94.0708,
            "omegam": {"derived": True, "latex": r"\Omega_m"},
        }, **point),
        "debug": 50,
    }
    model = get_model(info)
    model.add_requirements({"baryon_suppression": {
        "z": z_grid, "k": np.power(10.0, log10k_grid)}})
    model.logposterior({})
    return model.provider.get_baryon_suppression()


def compute_probes(sup=None, ell=None):
    """The four 3x2pt probes and the masked data vector, with
    optional baryonic suppression folded into the nonlinear power.

    Requires init_cosmolike(CLprobe="3x2pt", with_data=True). sup =
    None computes the dark-matter-only prediction; otherwise sup is
    the {z: S array} dictionary from get_baryon_suppression, applied
    exactly as the Cosmolike likelihoods apply it:
    lnPNL[i :: len(z_grid)] += ln S(z_i).

    Arguments:
      sup = suppression dictionary on the CAMB interpolation grid,
            or None.
      ell = multipoles of the returned harmonic spectra, or None
            for np.arange(25, 3000, 15).

    Returns:
      dict with ell, C_ss, C_gs, theta, xip, xim, gammat, dv, chi2,
      and the z/log10k interpolation grids (for feeding
      get_baryon_suppression).
    """
    if ell is None:
        ell = np.arange(25., 3000., 15.)
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_interp_1D, chi) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, kmax=5.0, k_per_logint=10,
        CAMBAccuracyBoost=1.0,
        non_linear_emul=_CONFIG["non_linear_emul"])
    lnPNL = np.array(lnPNL, copy=True)
    if sup is not None:
        for i, z_val in enumerate(z_interp_2D):
            # every k row of redshift z_i sits at stride len(z) in
            # the flattened table, the layout set_cosmology expects
            lnPNL[i :: len(z_interp_2D)] += np.log(sup[z_val])
    ci.init_ntable_lmax(int(_CONFIG["lmax"]))
    ci.init_accuracy_boost(1.0, 0)
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    ci.set_cosmology(omegam=omegam, H0=H0,
                     log10k_2D=log10k_interp_2D, z_2D=z_interp_2D,
                     lnP_linear=lnPL, lnP_nonlinear=lnPNL,
                     G=G_growth, z_1D=z_interp_1D, chi=chi)
    ci.set_nuisance_shear_calib(M=M_FID)
    ci.set_nuisance_shear_photoz(bias=SHEAR_PHOTOZ_FID)
    ci.set_nuisance_clustering_photoz(bias=LENS_PHOTOZ_FID)
    ci.set_nuisance_ia(A1=A1_FID, A2=A2_FID, B_TA=BTA_FID)
    ci.set_nuisance_bias(B1=B1_FID, B2=ZEROS5, B_MAG=ZEROS5,
                         B3nl=ZEROS5, BK=ZEROS5)
    ci.set_point_mass(PMV=PM_FID)
    ci.reset_bary_struct()
    (C_ss, tmp) = ci.C_ss_tomo_limber(l=ell)
    C_gs = ci.C_gs_tomo_limber(l=ell)
    (xip, xim) = ci.xi_pm_tomo()
    gt = ci.w_gammat_tomo()
    theta = ci.get_binning_real_space()
    dv = np.array(ci.compute_data_vector_masked())
    chi2 = ci.compute_chi2(dv)
    return {"ell": ell, "C_ss": C_ss, "C_gs": C_gs,
            "theta": theta, "xip": xip, "xim": xim, "gammat": gt,
            "dv": dv, "chi2": chi2,
            "z_grid": z_interp_2D, "log10k_grid": log10k_interp_2D}
