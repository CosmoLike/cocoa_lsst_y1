"""Gaussian NLA/TATT spectra and parity-aware real-space covariance checks.

Intrinsic alignments (IA) add to the shear spectra. NLA (nonlinear
alignment) adds E-mode power only; TATT (tidal alignment and tidal
torquing) also adds B-mode power through its quadratic (A2) and
density-weighting (B_TA) terms, and reduces to NLA when both are zero.
In real space xi+ combines E + B power and xi- combines E - B, so B-mode
power enters the covariance with sign +1 in the (xi+, xi+) and (xi-, xi-)
blocks and -1 in the cross block. The survey fixture and the spectra
helper come from test_covariance_nonlimber.py.
"""

import ctypes

import numpy as np

import cosmolike_lsst_y1_interface as ci
from test_covariance_fftlog import row_addresses
from test_covariance_nonlimber import survey, spectra


def set_alignment(model, a2=(0.0, 0.0), bta=(0.0, 0.0)):
    """Use independent per-bin amplitudes in the core's established convention.

    ia_redshift_evolution = 2 is one amplitude per source bin; ia_code =
    0 is the C FAST-PT; A1 = (0.6, 0.8) are the NLA amplitudes of
    survey_inputs.py.

    Arguments:
      model = 0 (NLA) or 1 (TATT).
      a2, bta = per-bin TATT amplitudes A2 and B_TA.

    Returns:
      nothing; cosmolike's IA state is changed.
    """
    ci.init_IA(ia_model=model, ia_redshift_evolution=2, ia_code=0)
    ci.set_nuisance_ia(A1=[0.6, 0.8], A2=list(a2), B_TA=list(bta))


def test_tatt_nla_limit_and_nonlimber(survey):
    """Zero quadratic/density-weighting amplitudes recover NLA exactly."""
    edges, unused = survey
    ell = [2., 8., 30., 1000.]
    set_alignment(model=0)
    nla = spectra(edges=edges, ell=ell, cutoff=30)
    set_alignment(model=1)
    tatt = spectra(edges=edges, ell=ell, cutoff=30)
    np.testing.assert_array_equal(tatt["spectra"], nla["spectra"])
    np.testing.assert_array_equal(tatt["b_spectra"], 0.0)
    set_alignment(model=0)


def test_tatt_against_data_vector_projection(survey):
    """Compare E corrections and B power with the independent data-vector integrator."""
    edges, library = survey
    ell = np.array([10., 50., 200., 1000., 5000.])
    ci.init_accuracy_boost(accuracy_boost=2, integration_accuracy=2)
    # the data-vector integrator, called through ctypes:
    # C_ss_tomo_limber_nointerp_ells(ell[], n_ell, n_pairs, EE[][], BB[][])
    pointer = ctypes.POINTER(ctypes.c_double)
    library.C_ss_tomo_limber_nointerp_ells.argtypes = [
        pointer, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(pointer), ctypes.POINTER(pointer),
    ]
    base = None
    for a2, bta in (([0., 0.], [0., 0.]), ([0.4, -0.3], [0.7, 1.1])):
        set_alignment(model=1, a2=a2, bta=bta)
        actual = spectra(edges=edges, ell=ell, nquad=256)
        ee = np.empty((3, len(ell)))
        bb = np.empty_like(ee)
        library.C_ss_tomo_limber_nointerp_ells(
            ell.ctypes.data_as(pointer), len(ell), 3,
            row_addresses(values=ee), row_addresses(values=bb),
        )
        # the three source pairs, as field indices of the covariance
        # spectra (lenses are fields 0-1, sources 2-3); the comparison
        # error is relative to each pair's largest |value| (5e-4 allows
        # for the different radial grids of the two integrators)
        source_pairs = ((2, 2), (2, 3), (3, 3))
        measured_e = np.array([actual["spectra"][:, i, j] for i, j in source_pairs])
        measured_b = np.array([actual["b_spectra"][:, i, j] for i, j in source_pairs])
        if base is None:
            base = (measured_e.copy(), ee.copy())
        else:
            # Subtract NLA from both projections to isolate the TATT
            # contribution, independently of their lensing-window grids.
            expected_e = ee-base[1]
            change_e = measured_e-base[0]
            scale = np.max(np.abs(expected_e), axis=1)[:, None]
            error = np.max(np.abs(change_e-expected_e)/scale)
            print(f"TATT E correction vs data vector: {error:.3e}")
            assert error < 5.e-4
            scale_b = np.max(np.abs(bb), axis=1)[:, None]
            error_b = np.max(np.abs(measured_b-bb)/scale_b)
            print(f"TATT BB vs data vector: {error_b:.3e}")
            assert error_b < 5.e-4
            ci.set_omp_threads(1)
            repeated = spectra(edges=edges, ell=ell, nquad=256, backend=ci.covariance)
            np.testing.assert_array_equal(repeated["spectra"], actual["spectra"])
            np.testing.assert_array_equal(repeated["b_spectra"], actual["b_spectra"])
            ci.set_omp_threads(8)
    set_alignment(model=0)
    ci.init_accuracy_boost(accuracy_boost=1, integration_accuracy=0)


def test_b_covariance_signs_and_noise_once():
    """Compare the added BB Wick covariance with a direct finite ell sum.

    B-mode power B_l adds 2 (B_l^2 + 2 B_l N)/((2l+1) f_sky) per multipole
    (the Gaussian, or Wick, covariance of a B field with shape noise N;
    f_sky = area/4pi), projected by the angular-bin operators with sign +1
    in the (xi+, xi+) and (xi-, xi-) blocks and -1 across. The pure-noise
    N^2 term belongs to the E part and must not be added a second time
    ("noise once").
    """
    ell = np.arange(2, 21, dtype=float)
    spectra_e = np.zeros((len(ell), 3, 3))
    spectra_b = np.zeros_like(spectra_e)
    spectra_e[:, 1, 1] = 2.e-5/(ell+1)
    spectra_b[:, 1, 1] = 3.e-6/(ell+1)
    noise = np.array([1.e-8, 4.e-7, 5.e-7])
    # rows (probe, A, B): xi+ and xi- of field 1 (a source), and gamma_t
    # of field 0 (a lens) with field 1; operators = per-probe angular-bin
    # weights [probe, angular bin, multipole]
    rows = np.array([[0, 1, 1], [1, 1, 1], [2, 0, 1]], dtype=np.int32)
    operators = np.ones((4, 2, len(ell)))
    operators[:, 1] *= np.linspace(0.5, 1.2, len(ell))
    area = 2.0
    kwargs = dict(
        spectra=spectra_e, noise=noise, rows=rows, operators=operators,
        ell_min=2, area_sr=area, pair_area_sr2=np.array([0.01, 0.02]),
    )
    baseline = ci.covariance_gaussian_real(**kwargs)
    actual = ci.covariance_gaussian_real(**kwargs, b_spectra=spectra_b)
    production = ci.covariance.covariance_gaussian_real(**kwargs, b_spectra=spectra_b)
    np.testing.assert_array_equal(production, actual)
    b = spectra_b[:, 1, 1]
    harmonic = 2*(b*b+2*b*noise[1])/((2*ell+1)*area/(4*np.pi))
    expected = np.zeros_like(actual)
    for first in range(2):
        for second in range(2):
            sign = 1 if first == second else -1
            block = (operators[first]*harmonic) @ operators[second].T
            expected[2*first:2*first+2, 2*second:2*second+2] = sign*block
    np.testing.assert_allclose(actual-baseline, expected, rtol=2.e-12, atol=1.e-25)
    zero_b = ci.covariance_gaussian_real(**kwargs, b_spectra=np.zeros_like(spectra_b))
    np.testing.assert_array_equal(zero_b, baseline)


def test_gaussian_options_leave_ssc_cng_unchanged(survey):
    """The full assembler changes only G when its Gaussian model is changed."""
    from cosmolike_notebook_utils.covariance.accuracy import covariance_accuracy
    from cosmolike_notebook_utils.covariance.forecast import gaussian_model
    from cosmolike_notebook_utils.covariance.survey import realspace_covariance

    edges, unused = survey
    ci.set_nuisance_bias(
        B1=[1.4, 1.8], B2=[0., 0.], B_MAG=[0., 0.],
        B3nl=[0., 0.], BK=[0., 0.],
    )
    settings = covariance_accuracy(ell_max=64, mask_ell_max=64, ng_ell_intervals=3)
    settings.update({
        "mnu": 0.0,
        "a_edges": edges,
        "lnm_edges": np.linspace(np.log(1.e6), np.log(1.e17), 3),
        "area_sr": 2.0,
        "edges_rad": np.geomspace(0.001, 0.01, 4),
        "nonlimber_lmax": 30,
    })
    rows = np.array([[0, 2, 2], [1, 2, 2], [2, 0, 2], [3, 0, 0]], dtype=np.int32)
    noise = np.full(4, 1.e-7)
    reference = None
    choices = (
        {"nonlimber": False, "ia": "none"},
        {"nonlimber": True, "ia": "NLA", "A1": [0.6, 0.8]},
        {"nonlimber": True, "ia": "TATT", "A1": [0.6, 0.8],
         "A2": [0.4, -0.3], "B_TA": [0.7, 1.1]},
    )
    for choice in choices:
        model = gaussian_model(gaussian=choice, nsource=2)
        settings["gaussian"] = model
        ci.init_IA(ia_model=1 if model["ia"] == "TATT" else 0,
                   ia_redshift_evolution=2, ia_code=0)
        ci.set_nuisance_ia(A1=model["A1"], A2=model["A2"], B_TA=model["B_TA"])
        actual = realspace_covariance(
            interface=ci.covariance, settings=settings, rows=rows, noise=noise,
        )
        if reference is None:
            reference = actual
        else:
            np.testing.assert_array_equal(actual["ssc"], reference["ssc"])
            np.testing.assert_array_equal(actual["cng"], reference["cng"])
            np.testing.assert_array_equal(actual["ssc_normalization_signal"],
                                          reference["ssc_normalization_signal"])
            assert np.any(actual["gaussian"] != reference["gaussian"])
        # the total, normalized to unit diagonal (a correlation matrix),
        # must be positive definite: its smallest eigenvalue is positive
        scale = np.sqrt(np.outer(np.diag(actual["total"]), np.diag(actual["total"])))
        assert np.linalg.eigvalsh(actual["total"]/scale)[0] > 0.0
    set_alignment(model=0)
