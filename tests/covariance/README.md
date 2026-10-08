# Covariance tests

These tests require the optional covariance build. Follow the
[project build instructions](../../README.md#computing_covariances): unset
`IGNORE_COSMOLIKE_LSST_Y1_COVARIANCE` after activating Cocoa and rebuild.
With the default data-vector-only build this sector reports skips; the
separate `tests/data_vector` tests remain available.

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
| `test_covariance_wrappers.py` | Whole real/Fourier Gaussian matrices, overlapping bands, array ownership and thread determinism. |
| `test_power_refinement.py` | Natural-cubic preparation of linear, nonlinear and cb powers; nested grids and preservation of original inputs. |
| `test_production.py` | Production/notebook agreement at one and eight threads, direct-array ownership and layout requirements. |
| `test_command_line.py` | Cobaya YAML cosmology, explicit evaluate overrides, independent accuracy controls and unsupported-input errors. |
| `test_covariance_primitives.py` | Gaussian field pairings, projection and analytic count/shape noise. |
| `test_covariance_spectra.py` | All lens/source Limber cross spectra on common radial nodes. |
| `test_covariance_halo.py` | Halo mass integrals, mass completion, small batches and threaded power reads. |
| `test_covariance_perturbation.py` | Angular averages of gravitational mode coupling. |
| `test_covariance_non_gaussian.py` | Halo trispectrum terms and background-density responses. |
| `test_covariance_ssc.py` | Background fluctuations, mean subtraction and complete cross-block projection. |
| `test_covariance_operators.py` | Full-sky angular-bin and integer-band operators. |
| `test_likelihood_comparison.py` | Read the text and packed-triangle covariance formats; preserve signs and apply the same scale-cut indices to every component; reject retained Y null rows. |
| `test_notebook_tools.py` | Owned-array Python calls, interpolation, mode diagnostics and plotting normalization. |
| `test_covariance_mask.py` | Pair area for a supplied survey footprint. |
| `test_covariance_fourier.py` | Complete bandpower G/SSC/cNG assembly, two-sided band rebinning, Fourier means and thread determinism. |
| `test_covariance_survey.py` | Survey row layouts, compressed angular projection and complete cross-lens cNG assembly against independent NumPy sums. |
| `test_covariance_connected.py` | Connected projections over radial nodes: whole matrices equal block-by-block sums bit for bit at any thread count; invalid shapes are rejected before threaded access. |
| `test_covariance_fftlog.py` | FFTLog spherical-Bessel transforms against closed-form integrals, including the lensing kernel near the observer. |
| `test_covariance_nonlimber.py` | All-pairs Gaussian non-Limber spectra against direct spherical-Bessel sums; crossed pairs, symmetry and thread repeatability. |
| `test_covariance_ia.py` | Gaussian NLA/TATT spectra: the TATT-to-NLA limit, E and B spectra against the data-vector integrator, B-mode covariance signs, and SSC/cNG unchanged by Gaussian options. |
| `test_wynn_cosmologies.py` | Wynn low-mass extrapolation of the halo moment I11 against deep finite mass integrals, away from the fiducial cosmology. |

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
