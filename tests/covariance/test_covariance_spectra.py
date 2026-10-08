"""Validate covariance-owned all-pairs Limber spectra on shared survey inputs.

The Gaussian covariance needs the Limber angular spectrum of every pair of
fields, C_l^AB = integral dchi W_A(chi) W_B(chi) P((l + 1/2)/chi)/chi^2
(with the radial windows W of galaxy density, magnification, lensing,
intrinsic alignment and, optionally, redshift-space distortions, RSD).
covariance_limber_spectra computes all ten pairs of the four fields
(fields 0-1 lenses, 2-3 sources) on one shared set of radial nodes.

Tests create a two-lens, two-source artificial survey and its CAMB tables
in a temporary directory (survey_inputs.py). This exposes overlapping
lens bins independently of the LSST forecast configuration. The
independent reference (spectra_reference.py in
cosmolike_notebook_utils/covariance/reference/) receives the archived CAMB
tables and the C window samples; quadrature refinement is checked
separately. A few core C functions are called directly through ctypes
(argtypes and restype declare their C signatures).
"""

import ctypes
from pathlib import Path
import unittest

import numpy as np



class CovarianceSpectra(unittest.TestCase):
    """Audit shared-node integration, signed weights, units, and determinism."""

    @classmethod
    def setUpClass(cls):
        """Install the archived fiducial through the normal project setters."""
        import tempfile
        import cosmolike_lsst_y1_interface as ci
        from cosmolike_notebook_utils.covariance.reference import spectra_reference
        import survey_inputs as setup

        # a temporary folder removed by the class cleanup (addClassCleanup
        # runs it after the last test of the class, pass or fail)
        temporary = tempfile.TemporaryDirectory(prefix="cocoa_covariance_")
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        setup.create_inputs(directory=directory/"inputs")
        cls.ci = ci
        cls.reference = spectra_reference
        cls.configuration = setup.initialize(ci=ci, directory=directory/"inputs")
        cls.edges = setup.panel_edges(configuration=cls.configuration)
        with np.load(directory/"inputs/camb.npz") as archive:
            cls.tables = {}
            for name in archive.files:
                cls.tables[name] = archive[name].copy()
        # The exact shared input readers are public C APIs, not diagnostic
        # wrappers added to the core. They supply an independent RSD input.
        cls.library = ctypes.CDLL(ci.__file__)
        cls.library.W_RSD.argtypes = [ctypes.c_double]*3+[ctypes.c_int]
        cls.library.W_RSD.restype = ctypes.c_double
        cls.library.a_chi.argtypes = [ctypes.c_double]
        cls.library.a_chi.restype = ctypes.c_double
        cls.library.chi.argtypes = [ctypes.c_double]
        cls.library.chi.restype = ctypes.c_double
        for name in ("nz_lens_photoz", "nz_source_photoz"):
            function = getattr(cls.library, name)
            function.argtypes = [ctypes.c_double, ctypes.c_int]
            function.restype = ctypes.c_double
        cls.library.omp_get_max_threads.restype = ctypes.c_int
        cls.original_threads = cls.library.omp_get_max_threads()
        cls.ell = np.array([1., 2., 3., 10., 30., 100., 1000., 10000., 50000.])

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's requested OpenMP worker count."""
        cls.ci.set_omp_threads(cls.original_threads)

    def compute(self, nquad=128, include_rsd=False, linear=False):
        """Return the public covariance spectrum snapshot for this configuration."""
        return self.ci.covariance_limber_spectra(
            ell=self.ell, a_edges=self.edges, nquad=nquad,
            include_ia=True, include_rsd=include_rsd, linear=linear,
        )

    def test_independent_contraction_and_units(self):
        """Check all ten pairs, both P choices, RSD, and a length-unit change."""
        for linear in (False, True):
            for include_rsd in (False, True):
                result = self.compute(linear=linear, include_rsd=include_rsd)
                geometry = result["geometry"]
                power = self.reference.power_at_nodes(
                    archive=self.tables, ell=self.ell, geometry=geometry,
                    linear=linear,
                )
                rsd = None
                # the RSD window of each lens at each multipole and node,
                # read from the core: W_RSD(l + 1/2, a, a_shift, lens),
                # with a_shift the scale factor at chi (l + 3/2)/(l + 1/2)
                if include_rsd:
                    rsd = np.empty((len(self.ell), 2, geometry.shape[1]))
                    for index, ell in enumerate(self.ell):
                        for node, a in enumerate(geometry[0]):
                            shifted_chi = geometry[1, node]*(ell+1.5)/(ell+0.5)
                            a_shift = self.library.a_chi(shifted_chi)
                            for lens in range(2):
                                rsd[index, lens, node] = self.library.W_RSD(
                                    ell+0.5, a, a_shift, lens
                                )
                expected = self.reference.limber_matrix(
                    ell=self.ell, geometry=geometry, windows=result["windows"],
                    nlens=2, power=power, rsd=rsd,
                )
                # scale[l, i, j] = sqrt(C_ii C_jj) at each multipole (the
                # einsum is an outer product per l): the errors are relative
                # to the auto spectra, so near-zero crosses do not inflate them
                scale = np.sqrt(np.einsum(
                    "ei,ej->eij", np.diagonal(expected, axis1=1, axis2=2),
                    np.diagonal(expected, axis1=1, axis2=2),
                ))
                np.testing.assert_array_less(
                    np.abs(result["spectra"]-expected), 2.e-12*scale+1.e-30
                )
                # the same reference in Mpc (c/H0 = 2997.92458/0.7 Mpc for
                # h = 0.7): the dimensionless C_l must not change
                converted = self.reference.limber_matrix(
                    ell=self.ell, geometry=geometry, windows=result["windows"],
                    nlens=2, power=power, rsd=rsd, distance_unit=2997.92458/0.7,
                )
                np.testing.assert_array_less(
                    np.abs(converted-expected), 1.e-12*scale+1.e-30
                )

    def test_thread_counts_symmetry_and_psd(self):
        """Require bitwise repetition and PSD before adding positive noise."""
        reference = None
        for threads in (1, 4, 8, 4):
            self.ci.set_omp_threads(threads)
            spectra = self.compute(include_rsd=True)["spectra"]
            np.testing.assert_array_equal(spectra, spectra.swapaxes(1, 2))
            if reference is None:
                reference = spectra.copy()
            else:
                np.testing.assert_array_equal(spectra, reference)
            # ell=1 has identically zero shear; inspect only nonzero fields.
            for matrix in spectra:
                # np.ix_(active, active) keeps the rows and columns of the
                # fields with nonzero auto spectra
                active = np.diag(matrix) > 0.0
                selected = matrix[np.ix_(active, active)]
                normalization = np.sqrt(np.outer(np.diag(selected), np.diag(selected)))
                self.assertGreaterEqual(
                    np.linalg.eigvalsh(selected/normalization)[0], -1.e-12
                )

    def test_signed_foreground_and_overlapping_lenses(self):
        """Retain foreground magnification and a nonzero cross-lens spectrum."""
        result = self.compute()
        redshift = 1.0/result["geometry"][0]-1.0
        foreground = redshift < 0.2
        self.assertTrue(np.any(result["windows"][1, 0, foreground] < 0.0))
        self.assertTrue(np.any(result["windows"][1, 1, foreground] > 0.0))
        self.assertTrue(np.all(result["spectra"][:, 0, 1] > 0.0))
        self.assertEqual(result["spectra"].shape, (len(self.ell), 4, 4))

    def test_refinement(self):
        """Measure the radial-rule floor on the archived piecewise input tables."""
        coarse = self.compute(nquad=256, include_rsd=True)["spectra"]
        fine = self.compute(nquad=512, include_rsd=True)["spectra"]
        diagonal = np.diagonal(fine, axis1=1, axis2=2)
        scale = np.sqrt(np.einsum("ei,ej->eij", diagonal, diagonal))
        error = np.max(np.abs(coarse-fine)/(scale+1.e-30))
        print(f"  common-node 256 -> 512 normalized spectrum change = {error:.6e}")
        # This checks the spectra alone; it is not the covariance
        # convergence contract or a production node setting.
        self.assertLess(error, 2.e-5)

    def test_lensing_efficiency_and_window_refinement(self):
        """Compare cumulative efficiencies with direct integration, including z<0.003."""
        result = self.compute(nquad=128)
        fine = self.ci.covariance_limber_spectra(
            ell=self.ell, a_edges=self.edges, nquad=128, nwindow=8193,
            include_ia=True, include_rsd=False,
        )
        delta = np.max(np.abs(fine["windows"][1]-result["windows"][1]))
        scale = np.max(np.abs(fine["windows"][1]))
        self.assertLess(delta/scale, 2.e-6)

        nodes, weights = np.polynomial.legendre.leggauss(128)
        geometry = fine["geometry"]
        redshift = 1.0/geometry[0]-1.0
        for target_z in (0.0001, 0.002, 0.1, 0.5):
            node = np.argmin(np.abs(redshift-target_z))
            foreground_z = redshift[node]
            distance = geometry[1, node]
            # lensing-window prefactor (3/2) Omega_m chi/a in c/H0 units,
            # Omega_m = 0.3 of survey_inputs.py; a lens window carries its
            # magnification coefficient on top. The expected efficiency
            # integrates n(z) on 24 panels of 128 Gauss-Legendre nodes.
            prefactor = 1.5*0.3*distance/geometry[0, node]
            for field in range(4):
                sample = "source"
                catalog = field-2
                amplitude = 1.0
                reader = self.library.nz_source_photoz
                if field < 2:
                    sample = "lens"
                    catalog = field
                    amplitude = self.configuration["survey"]["magnification"][field]
                    reader = self.library.nz_lens_photoz
                support = self.configuration["survey"][f"{sample}_support_z"][catalog]
                low = max(foreground_z, support[0]-0.005)
                high = support[1]+0.005
                edges = np.linspace(low, high, 25)
                expected = 0.0
                for lower, upper in zip(edges[:-1], edges[1:]):
                    z = (lower+upper)/2+(upper-lower)*nodes/2
                    dz = weights*(upper-lower)/2
                    density = np.array([reader(value, catalog) for value in z])
                    source_chi = np.array([self.library.chi(1/(1+value)) for value in z])
                    expected += self.reference.lensing_efficiency(
                        distance=distance, source_distance=source_chi,
                        source_density=density, redshift_weights=dz,
                    )
                actual = fine["windows"][1, field, node]/(prefactor*amplitude)
                self.assertLess(abs(actual-expected), 3.e-6)

    def test_invalid_arrays_raise(self):
        """Reject malformed Python input before entering the C allocation path."""
        for ell, edges, nquad in (
            (np.array([np.nan]), self.edges, 128),
            (self.ell, self.edges[::-1].copy(), 128),
            (self.ell, self.edges, 63),
        ):
            with self.assertRaises(ValueError):
                self.ci.covariance_limber_spectra(ell=ell, a_edges=edges, nquad=nquad)

    def test_multipole_batches_preserve_every_spectrum(self):
        """Batch boundaries and worker counts leave all radial sums unchanged."""
        from cosmolike_notebook_utils.covariance.gaussian import limber_spectra

        for linear in (False, True):
            for include_rsd in (False, True):
                self.ci.set_omp_threads(n=1)
                expected = self.ci.covariance_limber_spectra(
                    ell=self.ell, a_edges=self.edges, nquad=64, nwindow=1025,
                    include_ia=True, include_rsd=include_rsd, linear=linear,
                )
                for threads in (1, 8):
                    self.ci.set_omp_threads(n=threads)
                    for batch_size in (1, 4, len(self.ell)+1):
                        actual = limber_spectra(
                            interface=self.ci, ell=self.ell, a_edges=self.edges,
                            nquad=64, nwindow=1025, include_ia=True,
                            include_rsd=include_rsd, linear=linear,
                            batch_size=batch_size,
                        )
                        # .view(np.uint64) compares the 64-bit patterns:
                        # bit-for-bit equality
                        for name in ("spectra", "windows", "geometry"):
                            np.testing.assert_array_equal(actual[name].view(np.uint64),
                                                          expected[name].view(np.uint64))
                        self.assertEqual(actual["nlens"], expected["nlens"])
                        self.assertEqual(actual["nsource"], expected["nsource"])


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead
if __name__ == "__main__":
    unittest.main()
