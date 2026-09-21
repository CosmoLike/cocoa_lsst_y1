# Unit tests for the lsst_y1 likelihoods

These tests catch two kinds of silent breakage: a chi2 that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads). The
suite also measures the accuracy of the EMUL2 emulated pipelines and
reports whether they are accurate enough for data analysis (advisory:
no pass/fail).

## Running the tests

From the `Cocoa/` folder, with the cocoa conda environment active and
`start_cocoa.sh` sourced:

    python -m pytest ./projects/lsst_y1/tests

Without pytest:

    python -m unittest discover -s ./projects/lsst_y1/tests -v

The suite changes no project files. Each test streams a progress line
per model build and per evaluation, then a report block with the
computed chi2, the stored reference, the difference, and the pass
limit. A full run performs about 50 likelihood evaluations and takes a
few minutes. The test modules force `OMP_NUM_THREADS=4` internally.
The suite never waits for a keypress: a space/enter prompt between
tests means the output is being piped through a pager such as `less`,
so run the command with nothing piped after it.

## The eight tests

1. `test_1`: chi2 of the cosmic-shear likelihood at a fixed reference
   point must stay within 0.2 of the value stored in
   `frozen/reference_chi2.json`.
2. `test_2`: on one model, that point is evaluated fresh and then
   again as the 10th of 10 cosmologies in a row; the two chi2 values
   must agree to 1e-4. Leftover state or an OpenMP race breaks the
   agreement.
3. `test_3`: same as test 1 with the TATT intrinsic-alignment model
   (`IA_model: 1`) and `LSST_A2_1=0.05`, `LSST_BTA_1=0.05`,
   `LSST_A2_2=-1.51541`.
4. `test_4`: same as test 2 with the TATT model.
5. -8. the same four tests for the 3x2pt likelihood.
9. -10. `test_fastpt.py`: the TATT chi2 computed with the python
   FAST-PT package (`IA_code: 1` plus the fastpt theory block) instead
   of the C implementation cfastpt. Pass/fail against its own frozen
   FASTPT reference (0.2); the FASTPT-minus-CFASTPT difference is
   saved in `frozen/reference_chi2.json` and printed by the test.
11. -14. `test_example2_2x2pt.py`: the four standard tests on
   `lsst_y1.combo_2x2pt` (example2 with the probe selection reduced to
   galaxy clustering plus galaxy-galaxy lensing).

Advisory checks (`test_emul2.py`, E1-E4): the EXAMPLE_EMUL2 examples,
where trained machine-learning emulators replace the Boltzmann code.
No pass/fail: each check prints the emulator chi2, its drift against
the frozen emulator reference, the difference against the
exact-physics chi2 at the same cosmology, and the recommendation
(RECOMMENDED for actual data analysis when |emulator - exact| chi2
< 0.2, NOT recommended otherwise), plus a race check that warns
instead of failing. The trained-network files are read from
external_modules/data/emultrf, not from the frozen state; the network
device is frozen to `cpu` so the numbers do not depend on GPU
availability.

Accuracy checks (`test_accuracy.py`): first a one-knob-at-a-time
scan on the 3x2pt NLA configuration (so a large delta can be
attributed to the knob causing it; the scan includes accuracyboost 5
as a stress knob, which in other projects exposed interface
breakdowns), then the all-knobs checks A1-A6: the three probes with
both IA models re-evaluated with the numerical settings pushed
beyond the defaults (cosmolike accuracyboost 2, integration_accuracy
10, lmax 200000, kmax_boltzmann 40 paired with CAMB kmax 50; CAMB
AccuracyBoost 2, k_per_logint 50). Each check reports delta chi2 =
chi2(high accuracy) - chi2(default, frozen): the numerical error of
the default settings. No pass/fail. A high-accuracy evaluation takes
minutes; run this file on its own, or skip it with
`--ignore ./projects/lsst_y1/tests/test_accuracy.py`.

All TATT variants evaluate against `frozen/data/tatt_lsst_y1.dataset`,
a data vector GENERATED WITH TATT at the fiducial point during the
freeze. Reason: against the shipped NLA-based vector the TATT chi2
sits away from its minimum, where it responds linearly to tiny
numerical changes; at its own minimum the response is quadratic and
the drift bounds stay meaningful.

## Why the tests keep their own copy of everything

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

- `frozen_config_example{1,2}.py`: the complete cobaya configuration
  as a yaml string plus the exact evaluation point. Every option and
  every parameter is written out, including the ones that normally
  come from `params_source.yaml` and the other default files, so
  editing those files cannot change what the tests evaluate.
- `data/`: the tests' own copy of the data vectors, covariance, n(z),
  and masks.
- `EXAMPLE_EVALUATE{1,2}.yaml`: snapshots kept only so a human can
  diff how the live examples drifted since the freeze.

`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every frozen file. Each test
verifies the manifest first and refuses to run when a frozen file was
edited, naming the file. The result: users may change the live data
and examples freely, and nobody can quietly edit the frozen state
either.

## Refreshing the frozen state (maintainers only)

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze:

    python ./projects/lsst_y1/tests/generate_frozen_reference.py --overwrite

Run it from the `Cocoa/` folder with the environment set up as above.
It rebuilds `frozen/` from the current project, prints the four new
reference chi2 values, and rewrites the manifest. Review the printed
chi2 values against the old references before committing: they define
what every later test run compares against.
