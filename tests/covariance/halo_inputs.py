"""Read explicitly supplied core physics and call the covariance moment builder.

The halo model writes the matter trispectrum that the non-Gaussian
covariance needs as integrals over halo mass M of the halo abundance
dn/dlnM, the linear halo bias b(M) and the Fourier transform u(k|M) of
the halo density profile (NFW). The C function halo_moments_cov
(cosmolike/covariances/halo_cov.c) computes these mass integrals, the
moments I11(k) and the five pair moments I02, I12, I13 (two orderings)
and I04, on a Gauss-Legendre rule in ln M.

This is an input adapter, not the independent reference. HaloInputs.compute
calls the C builder; HaloInputs.sample reads the same physical inputs
(abundance, bias, profile) from the core at the nodes of a mass rule
built here with NumPy. The independent NumPy calculation in
halo_reference.py (cosmolike_notebook_utils/covariance/reference/)
receives only those numeric arrays and recomputes the moment sums, so
the sums are computed twice while the inputs are shared.

ctypes is Python's interface to C shared libraries: ctypes.CDLL(path)
opens a compiled library, library.name finds its C function called
name, and the function's argtypes and restype attributes declare the C
argument and return types, so Python converts numbers and pointers
correctly. A wrong declaration is not detected; it corrupts the call.
"""

import ctypes

import numpy as np


def row_pointers(array):
    """Return double row pointers while the caller retains the NumPy storage.

    Builds the C type double** for a 2D array: a ctypes array whose entry
    r points at the first element of row r of array's own memory, so C
    code reads and writes the numpy array directly. The caller must keep
    array alive, C-contiguous and float64 while the pointers are used.

    Arguments:
      array = 2D float64 numpy array in C (row-major) order.

    Returns:
      ctypes array of len(array) pointers to double.
    """
    pointer = ctypes.POINTER(ctypes.c_double)
    rows = (pointer*len(array))()
    for row in range(len(array)):
        rows[row] = array[row].ctypes.data_as(pointer)
    return rows


class CosmologyPrefix(ctypes.Structure):
    """Read-only prefix of the public cosmology struct, through rho_crit.

    A ctypes.Structure lays its _fields_ out in memory like a C struct
    with the same members, so CosmologyPrefix.in_dll(library,
    "cosmology") reads the live values of cosmolike's global variable
    `cosmology`. Only the first members are declared (a prefix); their
    order and types must match the struct in cosmolike/structs.h.
    rho_crit is the critical density, stored per (c/H0)^3.
    """

    _fields_ = [
        ("random", ctypes.c_uint64),
        ("Omega_b", ctypes.c_double),
        ("Omega_m", ctypes.c_double),
        ("Omega_v", ctypes.c_double),
        ("h0", ctypes.c_double),
        ("Omega_nu", ctypes.c_double),
        ("coverH0", ctypes.c_double),
        ("rho_crit", ctypes.c_double),
    ]


class HaloInputs:
    """Keep the explicitly supplied project and covariance C symbols."""

    def __init__(self, project_library, covariance_library):
        """Bind public core readers and the new moment entry, without setters.

        Declares the C signatures: halo_moments_cov(na, a[], nk, k[][],
        npanel, lnm_edges[], nquad, i11[][], moments[][][]) returns
        nothing; the core readers sigma2, dlognudlogm, fnu, hb1nu and
        conc take two doubles (mass or peak height, scale factor) and
        return a double; u_nfw_c takes four doubles (concentration, k,
        mass, scale factor). No setter is bound: the cosmology must
        already be installed in the library.

        Arguments:
          project_library = path of the compiled project interface
                     (the core readers and the cosmology struct).
          covariance_library = path of the library holding
                     halo_moments_cov (the same file in this project).
        """
        self.core = ctypes.CDLL(str(project_library))
        self.library = ctypes.CDLL(str(covariance_library))
        pointer = ctypes.POINTER(ctypes.c_double)
        rows = ctypes.POINTER(pointer)
        cube = ctypes.POINTER(rows)
        integer = ctypes.c_int
        self.library.halo_moments_cov.argtypes = [
            integer, pointer, integer, rows, integer, pointer, integer, rows, cube,
        ]
        self.library.halo_moments_cov.restype = None
        for name in ("sigma2", "dlognudlogm", "fnu", "hb1nu", "conc"):
            function = getattr(self.core, name)
            function.argtypes = [ctypes.c_double, ctypes.c_double]
            function.restype = ctypes.c_double
        self.core.u_nfw_c.argtypes = [ctypes.c_double]*4
        self.core.u_nfw_c.restype = ctypes.c_double
        self.cosmology = CosmologyPrefix.in_dll(self.core, "cosmology")

    def compute(self, a, k, edges, nquad):
        """Return production I11 and pair moments, checking padded output rows.

        The output arrays get three extra columns filled with NaN; the C
        function must leave them untouched, so any number found there
        reveals a write past the end of a row.

        Arguments:
          a     = 1D float64 array of scale factors [na].
          k     = 2D float64 array of wavenumbers [na, nk], one grid
                  per scale factor, in units of H0/c (k in h/Mpc times
                  c/H0 = 2997.92458 Mpc/h).
          edges = ln(M/[Msun/h]) panel edges [npanel + 1].
          nquad = Gauss-Legendre nodes per mass panel.

        Returns:
          (I11 [na, nk], moments [5, na, npair]) with npair =
          nk(nk+1)/2 pairs in the order (0,0), (0,1), ..., (1,1), ...

        Raises:
          AssertionError when the C function wrote into a padding column.
        """
        na, nk = k.shape
        npair = nk*(nk+1)//2
        single = np.full((na, nk+3), np.nan)
        moments = np.full((5, na, npair+3), np.nan)
        pointer = ctypes.POINTER(ctypes.c_double)
        rows = ctypes.POINTER(pointer)
        # cube = the C type double***: five row-pointer arrays, one per
        # moment role
        cube = (rows*5)()
        for role in range(5):
            cube[role] = row_pointers(moments[role])
        self.library.halo_moments_cov(
            len(a), a.ctypes.data_as(pointer), nk, row_pointers(k),
            len(edges)-1, edges.ctypes.data_as(pointer), nquad,
            row_pointers(single), cube,
        )
        if not np.all(np.isnan(single[:, nk:])):
            raise AssertionError("I11 wrote beyond its physical rows")
        if not np.all(np.isnan(moments[:, :, npair:])):
            raise AssertionError("moments wrote beyond their physical rows")
        return single[:, :nk].copy(), moments[:, :, :npair].copy()

    def sample(self, a, k, edges, nquad):
        """Sample physical halo inputs on an independently generated NumPy rule.

        Returns a dict consumable by halo_reference.moments. Public core
        readers are shared physics inputs, not a second moment algorithm.
        The mass rule maps nquad Gauss-Legendre nodes from [-1, 1] onto
        each ln M panel [lower, upper]: ln M = lower + (upper -
        lower)(x + 1)/2, weight (upper - lower) w/2. At each node the
        abundance is dn/dlnM = (rho_cb/M) f(nu) nu dln(nu)/dlnM with the
        peak height nu = 1.686/sigma_cb(M, a) (cold dark matter +
        baryon field, rho_cb = rho_crit (Omega_m - Omega_nu)).

        Arguments:
          a, k, edges, nquad = as in compute.

        Returns:
          {"mass": M at the nodes [n_node], "dlnmass": quadrature
          weights in ln M [n_node], "number_density": dn/dlnM
          [na, n_node], "bias": b [na, n_node], "profile": u(k|M)
          [na, nk, n_node], "profile_min": u(k|M_min) at the lowest
          edge [na, nk], "rho_cb": the density above}. u = 1 at k = 0.
        """
        nodes, measure = np.polynomial.legendre.leggauss(nquad)
        masses = []
        measures = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            masses.extend(np.exp(lower+(upper-lower)*(nodes+1)/2))
            measures.extend((upper-lower)*measure/2)
        mass = np.array(masses)
        weights = np.array(measures)
        density = self.cosmology.rho_crit*(self.cosmology.Omega_m
                                          -self.cosmology.Omega_nu)
        abundance = np.empty((len(a), len(mass)))
        bias = np.empty_like(abundance)
        profile = np.empty((len(a), k.shape[1], len(mass)))
        profile_min = np.empty_like(k)
        mass_min = np.exp(edges[0])
        for row, scale_factor in enumerate(a):
            for node, halo_mass in enumerate(mass):
                nu = 1.686/np.sqrt(self.core.sigma2(halo_mass, scale_factor))
                slope = self.core.dlognudlogm(halo_mass, scale_factor)
                abundance[row, node] = (density/halo_mass)*nu*slope
                abundance[row, node] *= self.core.fnu(nu, scale_factor)
                bias[row, node] = self.core.hb1nu(nu, scale_factor)
                concentration = self.core.conc(halo_mass, scale_factor)
                for index, wavenumber in enumerate(k[row]):
                    value = 1.0
                    if wavenumber > 0:
                        value = self.core.u_nfw_c(concentration, wavenumber,
                                                  halo_mass, scale_factor)
                    profile[row, index, node] = value
            concentration = self.core.conc(mass_min, scale_factor)
            for index, wavenumber in enumerate(k[row]):
                value = 1.0
                if wavenumber > 0:
                    value = self.core.u_nfw_c(concentration, wavenumber,
                                              mass_min, scale_factor)
                profile_min[row, index] = value
        return {
            "mass": mass,
            "dlnmass": weights,
            "number_density": abundance,
            "bias": bias,
            "profile": profile,
            "profile_min": profile_min,
            "rho_cb": density,
        }
