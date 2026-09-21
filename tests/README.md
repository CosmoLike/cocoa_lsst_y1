# Unit tests for the lsst_y1 likelihoods

These tests evaluate `EXAMPLE_EVALUATE1.yaml` (cosmic shear) and
`EXAMPLE_EVALUATE2.yaml` (3x2pt) in Python and compare the chi2 at the
example fiducial point against stored references. They do NOT read `../data`
or the live example yaml files: they run on the byte-frozen copies in
`frozen/`, pinned by SHA-256 in `manifest_sha256.json`. The live `../data`
and examples can therefore change without affecting these tests — and if a
frozen copy itself is edited, every test fails before evaluating anything.

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

    python -m pytest ./projects/lsst_y1/tests -v

or, without pytest:

    python -m unittest discover -s ./projects/lsst_y1/tests -v

The full suite takes a few minutes (each race test performs 11 likelihood
evaluations). `OMP_NUM_THREADS=2` is forced inside the test modules.

## Refreshing the frozen state (maintainers only)

Changing the data vectors, n(z), covariance, or the examples on purpose
requires re-freezing; review the printed chi2 values before committing:

    python ./projects/lsst_y1/tests/generate_frozen_reference.py --overwrite
