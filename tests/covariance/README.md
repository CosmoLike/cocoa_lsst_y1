# Covariance tests

These tests check covariance construction separately from the data-vector
and likelihood tests in `../data_vector/`. A covariance describes the
joint scatter of measurements; getting each predicted mean right does
not by itself validate their covariance.

We assume users have run `conda activate cocoa`, use Bash, and are in
`cocoa/Cocoa`, with the LSST Y1 interface compiled.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run the covariance tests.

    python -m pytest projects/lsst_y1/tests/covariance

| Test file | Calculation checked |
| --- | --- |
| `test_covariance_primitives.py` | Gaussian field pairings, projection and analytic count/shape noise. |
| `test_covariance_spectra.py` | All lens/source Limber cross spectra on common radial nodes. |
| `test_covariance_halo.py` | Halo mass integrals, mass completion and small batched calls. |
| `test_covariance_perturbation.py` | Angular averages of gravitational mode coupling. |
| `test_covariance_non_gaussian.py` | Halo trispectrum terms and background-density responses. |
| `test_covariance_ssc.py` | Background fluctuations, mean subtraction and complete cross-block projection. |
| `test_covariance_operators.py` | Full-sky angular-bin and integer-band operators. |
| `test_notebook_tools.py` | Owned-array Python calls, interpolation, mode diagnostics and plotting normalization. |
| `test_covariance_mask.py` | Pair area for a supplied survey footprint. |

The normally built LSST interface contains all tested C components. The
independent references ship in `cosmolike_notebook_utils.covariance.reference`;
no external reference folder or separately compiled test library is needed.
The halo and spectrum checks create a small artificial survey and CAMB tables
in temporary directories, removed after the tests. `halo_inputs.py` only
samples core physics for an independently summed mass integral;
`survey_inputs.py` defines those reproducible two-bin test inputs.

Independent NumPy and high-precision calculations check the C algebra.
Thread comparisons check repeatability, while quadrature refinements
measure numerical convergence. Positive-semidefinite inputs test that
projection preserves nonnegative variances, including correlations
between different lens bins and separately computed matrix blocks.

An individual connected contribution need not be positive definite.
Positivity of a few component tests also does not certify the full survey
matrix: its modeling choices, all cross blocks and parameter-error
convergence must be checked together. These tests never clip eigenvalues
or replace the covariance used by the likelihood.
