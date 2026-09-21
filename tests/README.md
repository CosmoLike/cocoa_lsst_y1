# Unit tests for the lsst_y1 likelihoods

These tests evaluate the cosmic-shear (example1) and 3x2pt (example2)
likelihoods in Python and compare the chi2 at the frozen fiducial point
against stored references. They are completely independent of the live
project configuration: each test loads
`frozen/frozen_config_example{1,2}.py`, an auto-generated module holding the
FULLY EXPANDED cobaya configuration as a yaml string (the
`EXAMPLE_EMUL_NAUTILUS1.py` idiom) plus the exact evaluated point. Every
option and every parameter — including the ones that normally come from the
likelihood default yaml files — is written out explicitly, and the data is
the tests' own copy in `frozen/data/`. `EXAMPLE_EVALUATE1/2.yaml`, the
likelihood defaults (`cosmic_shear.yaml`, `combo_3x2pt.yaml`,
`params_source.yaml`, `params_lens.yaml`), and `../data` can therefore all
change without affecting these tests. Everything under `frozen/` is pinned
by SHA-256 in `manifest_sha256.json`; if a frozen file is edited, every test
fails before evaluating anything. The `frozen/EXAMPLE_EVALUATE{1,2}.yaml`
copies are provenance snapshots for humans to diff — the tests never load
them.

The eight tests:

1. `test_1` chi2 of example1 within 0.2 of the frozen reference.
2. `test_2` no race condition (`OMP_NUM_THREADS=2`): the fiducial evaluated
   as the 10th of 10 cosmologies in a row on one model instance matches a
   fresh single evaluation (tolerance 1e-4).
3. `test_3` same as 1 with the TATT IA model (`IA_model: 1`) and
   `LSST_A2_1=0.05`, `LSST_BTA_1=0.05`, `LSST_A2_2=-1.51541`.
4. `test_4` same as 2 with the TATT IA model.
5. -8. the same four tests for example2 (3x2pt).

## Running

From the `Cocoa/` folder, with the cocoa environment active and
`start_cocoa.sh` sourced:

    python -m pytest ./projects/lsst_y1/tests

or, without pytest:

    python -m unittest discover -s ./projects/lsst_y1/tests -v

Each test streams a report to the terminal with the computed chi2, the
frozen reference, the |delta chi2|, and the limit (the bundled `pytest.ini`
passes `-v -s` so the reports are not swallowed by pytest's capture), plus
progress lines while models build and the race rows evaluate. The full
suite takes a few minutes (each race test performs 11 likelihood
evaluations). `OMP_NUM_THREADS=2` is forced inside the test modules.

The suite is fully non-interactive and never waits for a keypress. If the
terminal stops between tests asking for space/enter, the output is going
through a pager: run the command exactly as above, with no `| less`,
`| more`, or pager alias after it.

## Refreshing the frozen state (maintainers only)

Changing the data vectors, n(z), covariance, or the examples on purpose
requires re-freezing; review the printed chi2 values before committing:

    python ./projects/lsst_y1/tests/generate_frozen_reference.py --overwrite
