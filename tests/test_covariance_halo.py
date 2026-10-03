"""Check covariance-owned halo mass integration against independent sums.

The physical sigma, multiplicity, bias, concentration and profile samples
are shared inputs from public core readers. The NumPy reference performs
its own quadrature construction and moment contraction. These tests do
not independently calibrate the halo fits or a massive-neutrino response.
"""

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


def load_external(directory, name):
    """Import one named external input/reference module from a resolved path."""
    specification = importlib.util.spec_from_file_location(
        name=name, location=directory/f"{name}.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class HaloCovariance(unittest.TestCase):
    """Check all five moments, low-mass completion and high-k convergence."""

    @classmethod
    def setUpClass(cls):
        """Initialize the pinned massless input and explicitly selected library."""
        reference = os.environ.get("COSMOLIKE_COVARIANCE_REFERENCE")
        library = os.environ.get("COSMOLIKE_HALO_COVARIANCE_LIBRARY")
        if not reference or not library:
            raise unittest.SkipTest("set covariance reference and halo library paths")
        directory = Path(reference)
        for name in ("halo_inputs.py", "halo_reference.py", "inputs/camb.npz"):
            if not (directory/name).is_file():
                raise unittest.SkipTest(f"missing external covariance input {name}")
        if not Path(library).is_file():
            raise unittest.SkipTest("isolated halo covariance library is absent")
        import cosmolike_lsst_y1_interface as ci

        cls.ci = ci
        cls.reference = load_external(directory=directory, name="halo_reference")
        setup = load_external(directory=directory, name="survey_inputs")
        adapter = load_external(directory=directory, name="halo_inputs")
        setup.initialize(ci=ci, directory=directory/"inputs")
        cls.inputs = adapter.HaloInputs(project_library=ci.__file__,
                                       covariance_library=library)
        cls.original_threads = cls.inputs.core.omp_get_max_threads()
        cls.a = np.array([0.35, 0.7, 0.95])
        wave = np.array([0., 0.01, 0.1, 1., 10., 100., 300.])*2997.92458
        cls.k = np.tile(wave, (len(cls.a), 1))
        cls.edges = np.linspace(np.log(1.e6), np.log(1.e17), 9)

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's OpenMP setting after the reproducibility sweep."""
        cls.ci.set_omp_threads(cls.original_threads)

    def test_independent_mass_contraction(self):
        """Use NumPy GL nodes and independent sums of supplied physical samples."""
        actual = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=64)
        samples = self.inputs.sample(a=self.a, k=self.k, edges=self.edges, nquad=64)
        expected = self.reference.moments(**samples)
        for measured, reference in zip(actual, expected):
            np.testing.assert_allclose(measured, reference, rtol=2.e-11)

    def test_unresolved_mass_and_diagonal_identity(self):
        """I11(0)=1 and the two I13 orderings agree for identical wavenumbers."""
        for minimum in (1.e6, 1.e9, 1.e11):
            edges = np.linspace(np.log(minimum), np.log(1.e17), 9)
            single, moments = self.inputs.compute(a=self.a, k=self.k,
                                                   edges=edges, nquad=128)
            np.testing.assert_allclose(single[:, 0], 1.0, rtol=0, atol=4.e-15)
            pair = 0
            for first in range(self.k.shape[1]):
                for second in range(first, self.k.shape[1]):
                    if first == second:
                        np.testing.assert_array_equal(moments[2, :, pair],
                                                      moments[3, :, pair])
                    pair += 1

    def test_mass_refinement_at_large_wavenumber(self):
        """Resolve profile oscillations through 300 h/Mpc before choosing a rule."""
        coarse = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=512)
        fine = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=1024)
        for first, second in zip(coarse, fine):
            relative = np.max(np.abs(first/second-1))
            print(f"  halo moment mass refinement: {relative:.9e}", flush=True)
            self.assertLess(relative, 1.e-6)

    def test_reference_length_dimensions(self):
        """Shared samples preserve each moment's dimension under length conversion."""
        samples = self.inputs.sample(a=self.a, k=self.k, edges=self.edges, nquad=64)
        original = self.reference.moments(**samples)
        scale = 4282.7494
        converted = dict(samples)
        converted["number_density"] = samples["number_density"]/scale**3
        converted["rho_cb"] = samples["rho_cb"]/scale**3
        result = self.reference.moments(**converted)
        np.testing.assert_allclose(original[0], result[0], rtol=1.e-13)
        for role, exponent in enumerate((3, 3, 6, 6, 9)):
            np.testing.assert_allclose(original[1][role], result[1][role]/scale**exponent,
                                       rtol=1.e-13)

    def test_repeated_threads_and_state_refresh(self):
        """Repeat complete builds at 1/4/8 threads; no static covariance state."""
        original = None
        for threads in (1, 4, 8, 1, 8):
            self.ci.set_omp_threads(threads)
            actual = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=128)
            if original is None:
                original = actual
            for measured, reference in zip(actual, original):
                np.testing.assert_array_equal(measured, reference)
        changed = self.inputs.compute(a=self.a*0.99, k=self.k,
                                       edges=self.edges, nquad=128)
        self.assertFalse(np.array_equal(changed[1], original[1]))
        restored = self.inputs.compute(a=self.a, k=self.k, edges=self.edges, nquad=128)
        for measured, reference in zip(restored, original):
            np.testing.assert_array_equal(measured, reference)


if __name__ == "__main__":
    unittest.main()
