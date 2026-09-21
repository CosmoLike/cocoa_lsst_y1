# Unit tests for the lsst_y1 likelihoods

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads). These
tests also measure the accuracy of the EMUL2 emulated pipelines and
report whether they are accurate enough for data analysis (advisory:
no pass/fail).

Contents:

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [Advisory checks](#advisory_checks)
    2. [Accuracy checks](#accuracy_checks)
    3. [Synthetic data vectors](#synthetic_vectors)
3. [Tests keep their own copy of configurations and data](#frozen_copy)
4. [Refreshing the frozen state (maintainers only)](#refreeze)

## Running the tests <a name="run_tests"></a>

**Step :one:**: from the `Cocoa/` folder, activate the cocoa Conda
environment and source `start_cocoa.sh`

    conda activate cocoa

and

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/lsst_y1/tests

Without pytest:

    python -m unittest discover -s ./projects/lsst_y1/tests -v

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the
computed $\chi^2$, the stored reference, the difference, and the pass
limit.

A full run performs about 50 likelihood evaluations and takes a
few minutes. The test files force `OMP_NUM_THREADS=4` internally.

> [!NOTE]
> The tests never stop to ask for input. If the terminal pauses
> until space or enter is pressed, something sent the output through
> `less` (a program that stops after each full screen): run the
> commands exactly as written above, with nothing added after them.

## The tests <a name="the_tests"></a>

The standard configurations get four tests each: a $\chi^2$ drift check
and a race check, both in the NLA and in the TATT intrinsic-alignment
model. The TATT variants set

    IA_model: 1
    LSST_A2_1: 0.05
    LSST_BTA_1: 0.05
    LSST_A2_2: -1.51541

| check | pass limit                                        | a failure means                    |
|-------|---------------------------------------------------|------------------------------------|
| $\chi^2$  | within 0.2 of `frozen/reference_chi2.json`        | code or data changed the numbers   |
| race  | fresh vs 10th of 10 cosmologies in a row, to $10^{-4}$ | leftover state or an OpenMP race   |

| tests | file | configuration | checks |
|-------|------|---------------|--------|
| 1-4   | `test_example1.py` | cosmic shear (example1) | $\chi^2$ + race, NLA and TATT |
| 5-8   | `test_example2.py` | 3x2pt (example2) | $\chi^2$ + race, NLA and TATT |
| 9-10  | `test_fastpt.py` | 3x2pt TATT with python FAST-PT (`IA_code: 1` plus the fastpt theory block) instead of the C cfastpt | $\chi^2$ vs its own frozen FASTPT reference (0.2); the FASTPT-minus-CFASTPT difference is stored in `frozen/reference_chi2.json` and printed |
| 11-14 | `test_example2_2x2pt.py` | 2x2pt (`lsst_y1.combo_2x2pt`: example2 reduced to galaxy clustering plus galaxy-galaxy lensing) | $\chi^2$ + race, NLA and TATT |

### Advisory checks (`test_emul2.py`, E1-E4) <a name="advisory_checks"></a>

The EXAMPLE_EMUL2 examples,
where trained machine-learning emulators replace the Boltzmann code.
No pass/fail: each check prints the emulator $\chi^2$, its drift against
the frozen emulator reference, the difference against the
exact-physics $\chi^2$ at the same cosmology, and the recommendation
(RECOMMENDED for actual data analysis when
$\lvert\chi^2_\text{emulator} - \chi^2_\text{exact}\rvert < 0.2$, NOT recommended otherwise), plus a race check that warns
instead of failing. The trained-network files are read from
external_modules/data/emultrf, not from the frozen state; the network
device is frozen to `cpu` so the numbers do not depend on GPU
availability.

### Accuracy checks (`test_accuracy.py`, A1-A6) <a name="accuracy_checks"></a>

First a one-knob-at-a-time scan on the 3x2pt NLA configuration, so a large delta can be
attributed to the knob causing it (the scan includes
`accuracyboost: 5` as a stress knob, which in other projects exposed
interface breakdowns). Then the all-knobs checks A1-A6 re-evaluate
the three probes with both IA models with every setting pushed
beyond the defaults at once:

    # cosmolike likelihood settings
    accuracyboost: 2
    integration_accuracy: 10
    lmax: 200000
    kmax_boltzmann: 40
    # CAMB extra_args (kmax moves with kmax_boltzmann: one physical cutoff)
    AccuracyBoost: 2
    k_per_logint: 50
    kmax: 50

Each check reports $\Delta\chi^2 = \chi^2(\text{high accuracy}) -
\chi^2(\text{default})$: the numerical error of the default
settings. No pass/fail. High-accuracy evaluations take minutes; run
the file on its own, or skip it with

    python -m pytest ./projects/lsst_y1/tests --ignore ./projects/lsst_y1/tests/test_accuracy.py

#### The N-random-models check (opt-in) <a name="nmodels_check"></a>

Instead of the one frozen fiducial point, this check repeats the
accuracy measurement at N reproducible random points drawn across the prior
of the 3x2pt NLA configuration. Per point:

1. draw the point (seeded, so every run draws the same points);
2. generate a synthetic data vector at it with the DEFAULT settings,
   so the default $\chi^2$ against that vector is zero by
   construction;
3. evaluate the high-accuracy $\chi^2$ against the same vector: that
   number is the $\Delta\chi^2$ directly.

The report streams one block per model and ends with the
min/median/max $\Delta\chi^2$. Advisory: the deltas only have to be
finite.

Each model costs a default build+evaluation plus a high-accuracy
build+evaluation, minutes per model, so the check is off by default:
with `COCOA_ACCURACY_NMODELS` unset (or 0) it prints how to enable
it and passes.

**Step :one:**: with the environment of
[Running the tests](#run_tests), enable and run the check

    COCOA_ACCURACY_NMODELS=10 python -m pytest \
        ./projects/lsst_y1/tests/test_accuracy.py -k nmodels

Running the file as a script accepts `--nmodels N` in place of the
environment variable.

### Synthetic data vectors <a name="synthetic_vectors"></a>

All TATT variants evaluate against `frozen/data/tatt_lsst_y1.dataset`,
a data vector GENERATED WITH TATT at the fiducial point during the
freeze. Reason: against the shipped NLA-based vector the TATT $\chi^2$
sits away from its minimum, where it responds linearly to tiny
numerical changes; at its own minimum the response is quadratic and
the drift bounds stay meaningful.

## Tests keep their own copy of configurations and data <a name="frozen_copy"></a>

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_example{1,2}.py` | the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vectors, covariance, n(z), and masks |
| `EXAMPLE_EVALUATE{1,2}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

In the configuration modules every option and every parameter is
written out, including the ones that normally come from
`params_source.yaml` and the other default files, so editing those
files cannot change what the tests evaluate.


`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every frozen file. Each test
verifies the manifest first and refuses to run when a frozen file was
edited, naming the file. The result: users may change the live data
and examples freely, and nobody can quietly edit the frozen state
either.

## Refreshing the frozen state (maintainers only) <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

**Step :one:**: set up the environment as in
[Running the tests](#run_tests).

**Step :two:**: rebuild the frozen state

    python ./projects/lsst_y1/tests/generate_frozen_reference.py --overwrite

It rebuilds `frozen/` from the current project, prints the four new
reference $\chi^2$ values, and rewrites the manifest. Review the printed
$\chi^2$ values against the old references before committing: they define
what every later test run compares against.
