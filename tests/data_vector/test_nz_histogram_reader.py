"""The Akima reconstruction is fixed when only lookup accuracy is increased.

Synthetic uniform and irregular histograms exercise source/lens tables and
one/eight-thread preparation. The survey test covers the live mode round
trip. Runtime values come from the compiled production readers, with no diagnostic C API.
"""
import ctypes
from pathlib import Path

import numpy as np
import pytest

import cosmolike_lsst_y1_interface as ci


@pytest.mark.parametrize("preparation_factor", (3, 5, 7))
@pytest.mark.parametrize("irregular", (False, True))
def test_histogram_reader_refinement(irregular, preparation_factor, tmp_path):
    """Lookup refinement and thread count preserve the same reconstructed curve."""
    library = ctypes.CDLL(str(Path(ci.__file__).resolve()))
    library.omp_set_num_threads.argtypes = [ctypes.c_int]
    library.omp_get_max_threads.restype = ctypes.c_int
    previous_threads = library.omp_get_max_threads()
    z = np.arange(9, dtype=float) * .1 + .05
    if irregular:
        z = np.array([.05, .11, .18, .24, .39, .42, .61, .76, .85])
    heights = np.array([0., .2, .7, 1., .4, .6, .1, 0., 0.])
    table = np.column_stack((z, heights, heights[::-1]))
    filename = tmp_path / "histogram.nz"
    np.savetxt(fname=filename, X=table)
    query = np.linspace(start=.025, stop=1.0, num=2001)
    results = {}
    try:
        for boost, threads in ((1, 1), (2, 1), (1, 8)):
            ci.initial_setup()
            library.omp_set_num_threads(threads)
            ci.init_accuracy_boost(accuracy_boost=boost, integration_accuracy=0)
            ci.init_photoz_histogram_akima(
                enabled=1, preparation_factor=preparation_factor)
            ci.init_redshift_distributions_from_files(
                lens_multihisto_file=str(filename), lens_ntomo=2,
                source_multihisto_file=str(filename), source_ntomo=2)
            values = []
            for sample in ("source", "lens"):
                reader = getattr(library, "nz_" + sample + "_photoz")
                reader.argtypes = [ctypes.c_double, ctypes.c_int]
                reader.restype = ctypes.c_double
                for tomo in range(2):
                    values.append(np.fromiter((reader(float(x), tomo) for x in query),
                                               dtype=float, count=query.size))
            results[boost, threads] = np.asarray(values)
        np.testing.assert_allclose(actual=results[2, 1], desired=results[1, 1],
                                   rtol=2.e-13, atol=2.e-13)
        np.testing.assert_array_equal(results[1, 8], results[1, 1])
    finally:
        library.omp_set_num_threads(previous_threads)
        ci.initial_setup()
