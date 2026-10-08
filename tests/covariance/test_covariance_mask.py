"""Check mask-derived pair counts against direct spherical geometry.

The pure shot/shape-noise part of the Gaussian covariance of a
correlation function is inversely proportional to the pair area of its
angular bin: the area, in sr^2, of all ordered pairs of points inside the
survey footprint whose separation falls in the bin. On the full sky it is
4 pi x 2 pi (cos theta_low - cos theta_high); a finite footprint loses
the pairs that cross its boundary, which mask_pair_area_cov
(cosmolike/covariances/mask_cov.c) computes from the harmonic spectrum of
the mask. The tests compare it with the full-sky formula and with an
independent high-precision calculation for a spherical cap (a circular
footprint) in cosmolike_notebook_utils/covariance/reference/. The C
functions are called through ctypes (argtypes and restype declare each
function's C signature).
"""

import ctypes
import unittest

import numpy as np


def row_pointers(array):
    """Expose each independently addressable double row to the C interface."""
    pointer = ctypes.POINTER(ctypes.c_double)
    result = (pointer*len(array))()
    for row in range(len(array)):
        result[row] = array[row].ctypes.data_as(pointer)
    return result


class CovarianceMask(unittest.TestCase):
    """Use the same raw mask convention as SSC, with a separate geometric oracle."""

    @classmethod
    def setUpClass(cls):
        """Load the repository references and project C symbols.

        Declares the C signatures of mask_pair_area_cov,
        gaussian_noise_pair_cov and realspace_operator_cov, and starts
        with four OpenMP threads.
        """
        import cosmolike_lsst_y1_interface as ci
        from cosmolike_notebook_utils.covariance.reference import mask_reference
        from cosmolike_notebook_utils.covariance.reference import ssc_reference

        cls.references = {
            "mask_reference": mask_reference,
            "ssc_reference": ssc_reference,
        }
        cls.library = ctypes.CDLL(ci.__file__)
        cls.operators = cls.library
        integer = ctypes.c_int
        real = ctypes.c_double
        pointer = ctypes.POINTER(real)
        rows = ctypes.POINTER(pointer)
        cls.library.mask_pair_area_cov.argtypes = [integer, integer, real,
                                                   pointer, pointer, rows, pointer]
        cls.library.mask_pair_area_cov.restype = None
        cls.library.gaussian_noise_pair_cov.argtypes = [integer, integer,
                                                         ctypes.POINTER(integer), pointer, real]
        cls.library.gaussian_noise_pair_cov.restype = real
        cls.operators.realspace_operator_cov.argtypes = [integer, pointer, integer, integer, rows]
        cls.operators.realspace_operator_cov.restype = None
        cls.library.omp_set_num_threads.argtypes = [integer]
        cls.library.omp_set_num_threads.restype = None
        cls.library.omp_set_num_threads(4)

    def kernels(self, edges, ell_max):
        """Build scalar bin operators with mask monopole and dipole retained.

        realspace_operator_cov writes four blocks of nbin rows (one per
        probe; 256 quadrature nodes per bin); the last block is the
        scalar (spin-0) bin average that the pair area needs.

        Arguments:
          edges   = angular bin edges [radians].
          ell_max = largest multipole of the operators.

        Returns:
          array [nbin, ell_max + 1].
        """
        edges = np.ascontiguousarray(edges, dtype=float)
        nbin = len(edges)-1
        output = np.empty((4*nbin, ell_max+1))
        self.operators.realspace_operator_cov(
            nbin, edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ell_max, 256, row_pointers(output),
        )
        return output[3*nbin:].copy()

    def pair_areas(self, area, edges, mask, kernel):
        """Return C pair areas with sentinel-protected output and live inputs.

        The output array has three more slots than bins, prefilled with
        the sentinel 887.5 (a value the C function never writes); they
        must keep it, which detects a write past the last bin.

        Arguments:
          area   = footprint area [sr].
          edges  = angular bin edges [radians].
          mask   = harmonic spectrum of the mask, multipoles 0..L.
          kernel = the scalar operators of kernels().

        Returns:
          pair areas [sr^2], one per bin.
        """
        pointer = ctypes.POINTER(ctypes.c_double)
        edges = np.ascontiguousarray(edges, dtype=float)
        mask = np.ascontiguousarray(mask, dtype=float)
        output = np.full(len(edges)+2, 887.5)
        self.library.mask_pair_area_cov(
            len(edges)-1, len(mask), area, edges.ctypes.data_as(pointer),
            mask.ctypes.data_as(pointer), row_pointers(kernel),
            output.ctypes.data_as(pointer),
        )
        np.testing.assert_array_equal(output[len(edges)-1:], 887.5)
        return output[:len(edges)-1].copy()

    def test_full_sky_and_noise_normalization(self):
        """All-sky pair area and the auto-clustering Wick factor are analytic."""
        edges = np.array([0.0, 0.001, 0.002, 0.1, 0.5, np.pi])
        kernel = self.kernels(edges=edges, ell_max=2)
        # the full-sky mask spectrum: monopole 4 pi, nothing else; then
        # pair area = 8 pi^2 (cos lower - cos upper), written as
        # 2 sin(mid) sin(width/2) to avoid cancellation
        actual = self.pair_areas(area=4*np.pi, edges=edges,
                                 mask=np.array([4*np.pi, 0.0, 0.0]), kernel=kernel)
        width = 2*np.sin((edges[1:]+edges[:-1])/2)*np.sin(np.diff(edges)/2)
        expected = 8*np.pi**2*width
        np.testing.assert_allclose(actual, expected, rtol=3.e-15)
        # one lens catalog (all four field IDs 0): the auto-clustering
        # noise covariance is 2 N^2/pair area (two Wick pairings)
        fields = np.zeros(4, dtype=np.int32)
        noise = np.array([2.e-7, 2.e-7])
        result = self.library.gaussian_noise_pair_cov(
            3, 3, fields.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            noise.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), actual[1],
        )
        self.assertAlmostEqual(result/(2*noise[0]**2/expected[1]), 1.0, places=14)

    def test_cap_geometry_and_mask_refinement(self):
        """Harmonic mask sums converge to independent 60-digit cap intersections."""
        # 20 log-spaced bins from 2.5 to 250 arcmin, in radians, and the
        # LSST Y1 area of 12300 deg^2 in sr; the error must fall below
        # 2e-7 and 100 times below its value at L = 1024 as the mask's
        # multipole cutoff L grows
        edges = np.geomspace(2.5, 250.0, 21)*np.pi/(180*60)
        area = 12300*(np.pi/180)**2
        kernel = self.kernels(edges=edges, ell_max=32768)
        full_mask = self.references["ssc_reference"].cap_mask(area_sr=area, ell_max=32768)
        expected = np.empty(len(edges)-1)
        for bin_index in range(len(expected)):
            expected[bin_index] = self.references["mask_reference"].cap_pair_area(
                area_sr=area, lower=edges[bin_index], upper=edges[bin_index+1],
            )
        errors = []
        for ell_max in (1024, 4096, 16384, 32768):
            actual = self.pair_areas(area=area, edges=edges,
                                     mask=full_mask[:ell_max+1], kernel=kernel)
            error = float(np.max(np.abs(actual/expected-1)))
            errors.append(error)
            print(f"  mask Lmax={ell_max}: direct cap pair error {error:.9e}")
        self.assertLess(errors[-1], 2.e-7)
        self.assertLess(errors[-1], errors[0]/100)

    def test_thread_determinism_and_mask_roundtrip(self):
        """Odd bin counts, repeated calls and a changed footprint remain deterministic."""
        edges = np.geomspace(0.001, 0.07, 12)
        kernel = self.kernels(edges=edges, ell_max=4096)
        area = 3.0
        mask = self.references["ssc_reference"].cap_mask(area_sr=area, ell_max=4096)
        outputs = []
        for threads in (1, 4, 8):
            self.library.omp_set_num_threads(threads)
            outputs.append(self.pair_areas(area=area, edges=edges, mask=mask, kernel=kernel))
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])
        changed_mask = self.references["ssc_reference"].cap_mask(area_sr=4.0, ell_max=4096)
        changed = self.pair_areas(area=4.0, edges=edges, mask=changed_mask, kernel=kernel)
        self.assertGreater(np.max(np.abs(changed/outputs[0]-1)), 0.3)
        restored = self.pair_areas(area=area, edges=edges, mask=mask, kernel=kernel)
        np.testing.assert_array_equal(restored, outputs[0])
        self.library.omp_set_num_threads(4)


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead
if __name__ == "__main__":
    unittest.main()
