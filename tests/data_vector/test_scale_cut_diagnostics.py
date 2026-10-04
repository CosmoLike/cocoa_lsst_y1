"""Unit test: the scale-cut diagnostic functions (cosmo2D_scuts).

These are the notebook-facing derivative diagnostics of 2011.06469
eq 17 - the normalized Fourier derivative dlnC_ss/dlnk, its real-space
sibling dlnxi_pm/dlnk, and the response functions rf_C_ss and rf_xi
that cumulate |dln X/dlnk| up to a cutoff wavenumber. They ride the
batch _work machinery (dC_ss_dlnk_tomo_limber_work and the
Gauss-Legendre node arrays of the RF functions), and this test is the
regression guard for the 2026-09 refactor that moved them there.

History this test pins down: before the refactor, calling rf_C_ss at
any multipole l <= 20 was fatal - the exact-scalar low-l branch
underflowed k to 0 inside its normalization integrand and the k > 0
guard called exit(1), which also killed any jupyter kernel above it
(the "notebook derivative functions crash" symptom). The low
multipoles below are therefore load-bearing, not decoration.

Checks, all in one process on the frozen TATT cosmic-shear fiducial:
  1. every scalar and array overload returns finite values;
  2. the scalar overloads agree with the matching array-overload
     entries (they run the same batch engines);
  3. the response functions lie in [0, 1] up to quadrature slack and
     grow with the cutoff wavenumber.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest

import cocoa_test_utils as u

KK = np.array([0.05, 0.2, 1.0])       # wavenumbers in (Mpc/h)^-1
ELL = np.array([3.0, 10.0, 100.0, 1000.0])  # includes the old fatal l <= 20
RTOL = 1e-10


@pytest.fixture(scope="module")
def shear_state():
    info = u.load_frozen_info("example1", tatt=True)
    model = u.make_model(info)
    point = u.build_point(model, "example1", tatt=True)
    u.evaluate_chi2(model, point)
    import cosmolike_lsst_y1_interface as ci
    return ci


def test_dlnC_ss_dlnk(shear_state):
    ci = shear_state
    (EE, BB) = ci.dlnC_ss_dlnk_tomo_limber(k=KK, l=ELL)
    EE, BB = np.asarray(EE), np.asarray(BB)
    assert np.isfinite(EE).all() and np.isfinite(BB).all()
    (ee, bb) = ci.dlnC_ss_dlnk_tomo_limber(k=float(KK[1]), l=float(ELL[2]),
                                           ni=0, nj=1)
    assert np.isclose(ee, EE[1, 2, 0, 1], rtol=RTOL)
    assert np.isclose(bb, BB[1, 2, 0, 1], rtol=RTOL)


def test_rf_C_ss(shear_state):
    ci = shear_state
    (EE, tmp) = ci.rf_C_ss_tomo_limber(k=KK, l=ELL)
    EE = np.asarray(EE)
    assert np.isfinite(EE).all()
    # a normalized cumulative fraction, monotone in the cutoff
    assert (EE > -1e-6).all() and (EE < 1.0 + 1e-2).all()
    filled = np.abs(EE[-1]) > 1e-12
    assert (EE[-1][filled] >= EE[0][filled] - 1e-6).all()
    (ee, bb) = ci.rf_C_ss_tomo_limber(k=float(KK[1]), l=float(ELL[0]),
                                      ni=0, nj=1)  # l = 3: the old fatal
    assert np.isclose(ee, EE[1, 0, 0, 1], rtol=RTOL)


def test_dlnxi_dlnk(shear_state):
    ci = shear_state
    (XP, XM) = ci.dlnxi_dlnk_pm_tomo_limber(k=KK)
    XP, XM = np.asarray(XP), np.asarray(XM)
    assert np.isfinite(XP).all() and np.isfinite(XM).all()
    (xp, xm) = ci.dlnxi_dlnk_pm_tomo_limber(k=float(KK[1]))
    assert np.allclose(np.asarray(xp), XP[1], rtol=RTOL)
    assert np.allclose(np.asarray(xm), XM[1], rtol=RTOL)


def test_rf_xi(shear_state):
    ci = shear_state
    (XP, XM) = ci.rf_xi_tomo_limber(k=KK[:2])
    XP, XM = np.asarray(XP), np.asarray(XM)
    assert np.isfinite(XP).all() and np.isfinite(XM).all()
    assert (XP > -1e-6).all() and (XP < 1.0 + 1e-2).all()
    (xp, xm) = ci.rf_xi_tomo_limber(k=float(KK[0]), nt=5, ni=0, nj=1)
    assert np.isclose(xp, XP[0, 5, 0, 1], rtol=RTOL)
    assert np.isclose(xm, XM[0, 5, 0, 1], rtol=RTOL)
