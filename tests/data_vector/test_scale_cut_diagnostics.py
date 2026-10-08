"""Unit test: the scale-cut diagnostic functions (cosmo2D_scuts).

These are the notebook-facing derivative diagnostics of
arXiv:2011.06469, eq. 17: the normalized Fourier derivative
dlnC_ss/dlnk (how much of the shear spectrum comes from the matter
power at wavenumber k), its real-space sibling dlnxi_pm/dlnk, and the
response functions rf_C_ss and rf_xi that accumulate |dln X/dlnk| up
to a cutoff wavenumber. They are computed by the batch engines of
cosmo2D_scuts.c (dC_ss_dlnk_tomo_limber_work and the Gauss-Legendre
node arrays of the RF functions).

The multipoles l <= 20 take a separate low-l branch of rf_C_ss: if k
underflowed to 0 inside its normalization integrand there, the k > 0
guard would stop the process with exit(1), which also kills a Jupyter
kernel running the function. The low multipoles in ELL keep that
branch under test.

Checks, all in one process on the frozen TATT cosmic-shear fiducial:
  1. every scalar and array overload returns finite values;
  2. the scalar overloads agree with the matching array-overload
     entries (they run the same batch engines);
  3. the response functions lie in [0, 1] up to quadrature slack and
     grow with the cutoff wavenumber.
"""
import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load;
# setdefault sets 4 only when the variable is not already set
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest

# cocoa_test_utils resolves because tests/conftest.py puts the tests
# folder on the import path before pytest imports this module
import cocoa_test_utils as u

# the probed grid: three wavenumbers (cutoffs, for the response
# functions) and four multipoles, two of them on the low-l branch
# (l <= 20); RTOL = the agreement required between the scalar and
# array overloads, which run the same batch engines (rounding only)
KK = np.array([0.05, 0.2, 1.0])       # wavenumbers in (Mpc/h)^-1
ELL = np.array([3.0, 10.0, 100.0, 1000.0])  # includes the old fatal l <= 20
RTOL = 1e-10


@pytest.fixture(scope="module")
def shear_state():
    """Build the frozen TATT cosmic-shear state once for this module.

    A pytest fixture: pytest calls it once per module (scope="module")
    and hands its return value to every test that names shear_state as
    an argument. It builds the example1 TATT model and evaluates the
    frozen fiducial, which leaves that cosmology and those nuisance
    parameters inside the compiled interface the tests then query.
    This module does not call verify_frozen.

    Returns:
      the compiled module cosmolike_lsst_y1_interface.
    """
    info = u.load_frozen_info("example1", tatt=True)
    model = u.make_model(info)
    point = u.build_point(model, "example1", tatt=True)
    u.evaluate_chi2(model, point)
    import cosmolike_lsst_y1_interface as ci
    return ci


def test_dlnC_ss_dlnk(shear_state):
    """dlnC_ss/dlnk is finite, and its scalar and array overloads agree."""
    ci = shear_state
    # array overload: EE, BB of shape (n_k, n_ell, n_source, n_source);
    # [1, 2, 0, 1] = k = 0.2, l = 100, source bins (0, 1), the entry the
    # scalar call below computes
    (EE, BB) = ci.dlnC_ss_dlnk_tomo_limber(k=KK, l=ELL)
    EE, BB = np.asarray(EE), np.asarray(BB)
    assert np.isfinite(EE).all() and np.isfinite(BB).all()
    (ee, bb) = ci.dlnC_ss_dlnk_tomo_limber(k=float(KK[1]), l=float(ELL[2]),
                                           ni=0, nj=1)
    assert np.isclose(ee, EE[1, 2, 0, 1], rtol=RTOL)
    assert np.isclose(bb, BB[1, 2, 0, 1], rtol=RTOL)


def test_rf_C_ss(shear_state):
    """rf_C_ss is a finite fraction in [0, 1] that grows with the cutoff."""
    ci = shear_state
    (EE, tmp) = ci.rf_C_ss_tomo_limber(k=KK, l=ELL)
    EE = np.asarray(EE)
    assert np.isfinite(EE).all()
    # a normalized cumulative fraction, monotone in the cutoff; the
    # slack (1e-6 below 0, 1% above 1) absorbs quadrature error
    assert (EE > -1e-6).all() and (EE < 1.0 + 1e-2).all()
    # filled = the entries the array overload computes (bin pairs with
    # i <= j; the others stay 0); EE[-1] (largest cutoff) must not lie
    # below EE[0] (smallest cutoff) on them
    filled = np.abs(EE[-1]) > 1e-12
    assert (EE[-1][filled] >= EE[0][filled] - 1e-6).all()
    (ee, bb) = ci.rf_C_ss_tomo_limber(k=float(KK[1]), l=float(ELL[0]),
                                      ni=0, nj=1)  # l = 3: the old fatal
    assert np.isclose(ee, EE[1, 0, 0, 1], rtol=RTOL)


def test_dlnxi_dlnk(shear_state):
    """dlnxi_pm/dlnk is finite, and its two overloads agree."""
    ci = shear_state
    (XP, XM) = ci.dlnxi_dlnk_pm_tomo_limber(k=KK)
    XP, XM = np.asarray(XP), np.asarray(XM)
    assert np.isfinite(XP).all() and np.isfinite(XM).all()
    (xp, xm) = ci.dlnxi_dlnk_pm_tomo_limber(k=float(KK[1]))
    assert np.allclose(np.asarray(xp), XP[1], rtol=RTOL)
    assert np.allclose(np.asarray(xm), XM[1], rtol=RTOL)


def test_rf_xi(shear_state):
    """rf_xi is finite, within [0, 1], and its two overloads agree."""
    ci = shear_state
    # XP, XM of shape (n_k, n_theta, n_source, n_source); [0, 5, 0, 1]
    # = k = 0.05, angular bin 5, source bins (0, 1)
    (XP, XM) = ci.rf_xi_tomo_limber(k=KK[:2])
    XP, XM = np.asarray(XP), np.asarray(XM)
    assert np.isfinite(XP).all() and np.isfinite(XM).all()
    assert (XP > -1e-6).all() and (XP < 1.0 + 1e-2).all()
    (xp, xm) = ci.rf_xi_tomo_limber(k=float(KK[0]), nt=5, ni=0, nj=1)
    assert np.isclose(xp, XP[0, 5, 0, 1], rtol=RTOL)
    assert np.isclose(xm, XM[0, 5, 0, 1], rtol=RTOL)
