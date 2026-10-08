"""Validate all-pairs Gaussian non-Limber spectra with direct Bessel sums.

The Gaussian covariance needs the angular power spectrum C_l of every
pair of fields (here the two lens and two source bins of
survey_inputs.py, fields 0-1 lenses, 2-3 sources). The Limber
approximation is poor at low l, so below the cutoff nonlimber_lmax the
covariance spectra add a non-Limber correction: the exact projection of
the linear power spectrum,

    C_l^AB = (2/pi) integral dk k^2 P_lin(k) T_l^A(k) T_l^B(k),
    T_l(k) = integral dchi W(chi) D(chi) j_l(k chi) (times the l factors
             of the lensing and magnification terms),

minus its Limber value. The fixture survey builds the test survey once
per module; spectra calls the covariance spectrum function;
direct_projection recomputes the correction with scipy's spherical
Bessel functions on an explicit k grid (no FFTLog).
"""

import ctypes

import numpy as np
import pytest
from scipy.special import spherical_jn

import cosmolike_lsst_y1_interface as ci
import survey_inputs


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    """Prepare overlapping lenses, signed magnification and nonzero NLA.

    A module-scoped pytest fixture: built once, handed to every test of
    this module (and of test_covariance_ia.py, which imports it) that
    names survey as an argument; the lines after the yield restore the
    OpenMP thread count when the module's tests end.

    Yields:
      (edges, library): the scale-factor panel edges of panel_edges and
      the compiled library opened with ctypes, with the C signatures of
      growfac(a) and p_lin_at_a(a, k[], n, out[]) declared.
    """
    directory = tmp_path_factory.mktemp("nonlimber")/"inputs"
    survey_inputs.create_inputs(directory=directory)
    configuration = survey_inputs.initialize(ci=ci, directory=directory)
    edges = survey_inputs.panel_edges(configuration=configuration)
    library = ctypes.CDLL(ci.__file__)
    library.growfac.argtypes = [ctypes.c_double]
    library.growfac.restype = ctypes.c_double
    pointer = ctypes.POINTER(ctypes.c_double)
    library.p_lin_at_a.argtypes = [ctypes.c_double, pointer, ctypes.c_int, pointer]
    library.omp_get_max_threads.restype = ctypes.c_int
    original = library.omp_get_max_threads()
    yield edges, library
    ci.set_omp_threads(original)


def spectra(edges, ell, cutoff=0, nchi=4097, nquad=128, backend=ci):
    """Use the shared production/notebook API with RSD explicitly disabled.

    Arguments:
      edges  = scale-factor panel edges.
      ell    = multipoles.
      cutoff = nonlimber_lmax: below it the non-Limber correction is
               added; 0 = Limber only.
      nchi   = radial samples of the non-Limber FFTLog.
      nquad  = Gauss-Legendre nodes per radial panel.
      backend = ci (notebook binding) or ci.covariance (production
               binding); both run the same C code.

    Returns:
      the spectrum snapshot: spectra [n_ell, n_field, n_field],
      b_spectra, geometry [4, n_radial] (rows a, chi, f_K, dchi; chi in
      c/H0), windows [3, n_field, n_radial] (rows density, lensing or
      magnification, signed NLA).
    """
    return backend.covariance_spectra(
        ell=np.asarray(ell, dtype=float), a_edges=edges, nquad=nquad,
        nwindow=16385, include_ia=True, include_rsd=False, linear=False,
        nonlimber_lmax=cutoff, nonlimber_nchi=nchi,
    )


def direct_projection(snapshot, library, ell, nk):
    """Integrate spherical Bessels directly; no FFT or Mellin kernel is used.

    Input windows carry the tested cosmology/catalog conventions. This
    reference independently integrates them in both distance and k and
    constructs the matched subtraction from P(k,1)*D(a)^2.
    """
    geometry = snapshot["geometry"]
    windows = snapshot["windows"]
    distance = geometry[1]
    growth = np.array([library.growfac(float(a)) for a in geometry[0]])
    # measure = dchi D(a) at each radial node (linear growth carries P(k,
    # a=1) to the node's time)
    measure = geometry[3]*growth
    # k grid in units of H0/c (distances are in c/H0)
    wave = np.geomspace(0.001, 3000.0, nk)
    power = np.empty_like(wave)
    pointer = ctypes.POINTER(ctypes.c_double)
    library.p_lin_at_a(
        1.0, wave.ctypes.data_as(pointer), nk, power.ctypes.data_as(pointer),
    )
    argument = wave[:, None]*distance[None, :]
    bessel = spherical_jn(ell, argument)
    # transfers [nk, n_field]: lens fields = density term plus the
    # magnification term (l(l+1) j_l/x^2); source fields = lensing plus
    # NLA times the spin-2 factor sqrt((l-1)l(l+1)(l+2)) j_l/x^2
    density = (bessel*measure) @ windows[0, :2].T
    spin = ((bessel/argument**2)*measure) @ (windows[1]+windows[2]).T
    transfers = spin*np.sqrt((ell-1)*ell*(ell+1)*(ell+2))
    transfers[:, :2] = density+ell*(ell+1)*spin[:, :2]
    # trapezoid weights in ln k for (2/pi) integral dk k^2 P(k) =
    # (2/pi) integral dlnk k^3 P(k); the two end points get half weight
    weights = (2/np.pi)*np.log(wave[1]/wave[0])*wave**3*power
    weights[[0, -1]] *= 0.5
    exact = transfers.T @ (weights[:, None]*transfers)

    # Matched Limber prediction of this same separable field, including
    # the geometric spin and magnification factors before pairing fields.
    wave_limber = np.ascontiguousarray((ell+0.5)/distance)
    power_limber = np.empty_like(wave_limber)
    library.p_lin_at_a(
        1.0, wave_limber.ctypes.data_as(pointer), len(distance),
        power_limber.ctypes.data_as(pointer),
    )
    lens = windows[0, :2]+ell*(ell+1)/(ell+0.5)**2*windows[1, :2]
    source = ((windows[1, 2:]+windows[2, 2:])
              *np.sqrt((ell-1)*ell*(ell+1)*(ell+2))/(ell+0.5)**2)
    fields = np.concatenate((lens, source), axis=0)
    weight_limber = geometry[3]*growth**2*power_limber/distance**2
    matched = (fields*weight_limber) @ fields.T
    return exact-matched


def test_direct_bessel_reference(survey):
    """Refine the independent k sum before comparing signed crossed pairs.

    The errors are measured in units of sqrt(C_ii C_jj) (scale below), so
    cross spectra near zero do not inflate them. Lens rows only ([:2])
    enter the direct comparison; the source-source block must not change
    at all (the correction applies to galaxy clustering and galaxy-galaxy
    lensing only).
    """
    edges, library = survey
    ell = np.array([2., 8., 30.])
    baseline = spectra(edges=edges, ell=ell, nquad=256)
    actual = spectra(edges=edges, ell=ell, cutoff=30, nquad=256)
    fine = spectra(edges=edges, ell=ell, cutoff=30, nchi=8193, nquad=256)
    diagonal = np.diagonal(baseline["spectra"], axis1=1, axis2=2)
    scale = np.sqrt(diagonal[:, :, None]*diagonal[:, None, :])
    error = np.max(np.abs(actual["spectra"]-fine["spectra"])/scale)
    print(f"non-Limber nested chi refinement: {error:.3e}")
    assert error < 2.e-4
    for index, mode in enumerate(ell):
        coarse = direct_projection(snapshot=baseline, library=library, ell=mode, nk=2049)
        reference = direct_projection(snapshot=baseline, library=library, ell=mode, nk=4097)
        change = np.max(np.abs(reference[:2]-coarse[:2])/scale[index, :2])
        assert change < 2.e-5
        correction = actual["spectra"][index]-baseline["spectra"][index]
        error = np.max(np.abs(correction[:2]-reference[:2])/scale[index, :2])
        print(f"ell={mode:g}, direct Bessel comparison: {error:.3e}")
        assert error < 2.e-4
    np.testing.assert_array_equal(actual["spectra"][:, 2:, 2:],
                                  baseline["spectra"][:, 2:, 2:])


def test_backends_threads_and_high_ell(survey):
    """Preserve all crosses, model limits, symmetry and immutable snapshots.

    The production and notebook bindings agree bit for bit; each C_l
    matrix is symmetric and, normalized to unit diagonal, positive
    definite (its smallest eigenvalue, from np.linalg.eigvalsh, is
    positive); l = 400, above the cutoff 300, is pure Limber.
    """
    edges, unused = survey
    ell = np.array([2., 8., 30., 100., 300., 400.])
    baseline = spectra(edges=edges, ell=ell)
    ci.set_omp_threads(1)
    result = spectra(edges=edges, ell=ell, cutoff=300)
    ci.set_omp_threads(8)
    production = spectra(edges=edges, ell=ell, cutoff=300, backend=ci.covariance)
    np.testing.assert_array_equal(result["spectra"], production["spectra"])
    np.testing.assert_array_equal(result["spectra"], result["spectra"].swapaxes(1, 2))
    np.testing.assert_array_equal(result["spectra"][-1], baseline["spectra"][-1])
    assert np.any(result["spectra"][:3, 0, 1] != baseline["spectra"][:3, 0, 1])
    assert np.any(result["spectra"][:3, :2, 2:] != baseline["spectra"][:3, :2, 2:])
    for matrix in result["spectra"]:
        norm = np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
        assert np.linalg.eigvalsh(matrix/norm)[0] > 0.0
    norm = np.sqrt(np.outer(np.diag(baseline["spectra"][-2]),
                           np.diag(baseline["spectra"][-2])))
    assert np.max(np.abs(result["spectra"][-2]-baseline["spectra"][-2])/norm) < 0.01
