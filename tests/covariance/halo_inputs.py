"""Read explicitly supplied core physics and call the covariance moment builder.

This is an input adapter, not the independent reference. The independent
NumPy calculation in halo_reference.py receives only numeric arrays.
"""

import ctypes

import numpy as np


def row_pointers(array):
    """Return double row pointers while the caller retains the NumPy storage."""
    pointer = ctypes.POINTER(ctypes.c_double)
    rows = (pointer*len(array))()
    for row in range(len(array)):
        rows[row] = array[row].ctypes.data_as(pointer)
    return rows


class CosmologyPrefix(ctypes.Structure):
    """Read-only prefix of the public cosmology struct, through rho_crit."""

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
        """Bind public core readers and the new moment entry, without setters."""
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
        """Return production I11 and pair moments, checking padded output rows."""
        na, nk = k.shape
        npair = nk*(nk+1)//2
        single = np.full((na, nk+3), np.nan)
        moments = np.full((5, na, npair+3), np.nan)
        pointer = ctypes.POINTER(ctypes.c_double)
        rows = ctypes.POINTER(pointer)
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
