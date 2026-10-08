"""Check low-mass extrapolation away from the fiducial cosmology.

The halo moment I11(k) is a mass integral down to M -> 0. The default
mass panels (halo_mass_edges) end in eleven four-decade panels below
10^4 Msun/h, and the Wynn epsilon algorithm (a sequence-acceleration
method) extrapolates their partial integrals to the full integral. A
high-order Wynn estimate can be finite but unstable. Comparing against
deep finite integrals exposes that failure; checking only I11(0)=1 does
not, because the additive completion enforces that identity by construction.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

# the project's covariance/ folder holds the survey adapter
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"covariance"))
import lsst_y1_covariance as survey
from cosmolike_notebook_utils.covariance import halo_mass_edges
import cosmolike_lsst_y1_interface as ci


# three cosmologies: the fiducial, a low-amplitude one, and a
# high-amplitude one with w = -0.8 (pytest runs the test once for each)
@pytest.mark.parametrize("changes", [
    {},
    {"omegam": 0.25, "ns": 0.92, "As_1e9": 1.7},
    {"omegam": 0.35, "ns": 1.0, "As_1e9": 2.5,
     "w": -0.8, "w0pwa": -0.8},
])
def test_wynn_against_deep_finite_integral(changes):
    """Retain the 1e-5 diagnostic tolerance through z=39 and k=300 h/Mpc.

    Split the first tail panel to request the ordinary finite integral.
    Both paths retain the same mass interval and zero-k completion. The
    control's 256-node rule agrees with a 512-node rule; where the Wynn
    extrapolation fails, extra nodes do not cure it, so this comparison
    is the check.
    """
    settings = survey.configuration(gaussian={"nonlimber": False, "ia": "none"})
    settings["cosmology"].update(changes)
    survey.initialize(interface=ci, settings=settings)

    # six redshifts up to z = 39 and k from 0 to 300 h/Mpc, converted to
    # H0/c units (times c/H0 = 2997.92458 Mpc/h); np.r_ joins the 0 and
    # the log-spaced values into one array, np.tile repeats the row per a
    a = 1/(1+np.array([0.01, 0.5, 1.0, 3.0, 10.0, 39.0]))
    wave = np.r_[0.0, np.geomspace(start=0.001, stop=300, num=31)]
    k = np.tile(wave*2997.92458, (len(a), 1))
    edges = halo_mass_edges()
    finite_edges = np.sort(np.r_[edges, (edges[0]+edges[1])/2])
    expected, unused = ci.covariance.covariance_halo_moments(
        a=a, k=k, lnm_edges=finite_edges, nquad=256, pair_moments=False,
    )

    for nquad in (96, 128, 256):
        actual, unused = ci.covariance.covariance_halo_moments(
            a=a, k=k, lnm_edges=edges, nquad=nquad, pair_moments=False,
        )
        np.testing.assert_allclose(actual=actual[:, 0], desired=1.0,
                                   rtol=0, atol=4.e-15)
        np.testing.assert_allclose(actual=actual, desired=expected,
                                   rtol=1.e-5, atol=0)
