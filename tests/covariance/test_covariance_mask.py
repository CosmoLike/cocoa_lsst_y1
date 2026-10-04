"""Check mask-derived pair counts against direct spherical geometry."""

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
        """Load the repository references and project C symbols."""
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
        """Build scalar bin operators with mask monopole and dipole retained."""
        edges = np.ascontiguousarray(edges, dtype=float)
        nbin = len(edges)-1
        output = np.empty((4*nbin, ell_max+1))
        self.operators.realspace_operator_cov(
            nbin, edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ell_max, 256, row_pointers(output),
        )
        return output[3*nbin:].copy()

    def pair_areas(self, area, edges, mask, kernel):
        """Return C pair areas with sentinel-protected output and live inputs."""
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
        actual = self.pair_areas(area=4*np.pi, edges=edges,
                                 mask=np.array([4*np.pi, 0.0, 0.0]), kernel=kernel)
        width = 2*np.sin((edges[1:]+edges[:-1])/2)*np.sin(np.diff(edges)/2)
        expected = 8*np.pi**2*width
        np.testing.assert_allclose(actual, expected, rtol=3.e-15)
        fields = np.zeros(4, dtype=np.int32)
        noise = np.array([2.e-7, 2.e-7])
        result = self.library.gaussian_noise_pair_cov(
            3, 3, fields.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            noise.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), actual[1],
        )
        self.assertAlmostEqual(result/(2*noise[0]**2/expected[1]), 1.0, places=14)

    def test_cap_geometry_and_mask_refinement(self):
        """Harmonic mask sums converge to independent 60-digit cap intersections."""
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


if __name__ == "__main__":
    unittest.main()
