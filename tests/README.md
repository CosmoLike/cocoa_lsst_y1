# Unit tests for the likelihoods

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads). These
tests also measure the accuracy of the hybrid emulated pipelines and
report whether they are accurate enough for data analysis (advisory:
no pass/fail).

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [Advisory checks](#advisory_checks)
    2. [Accuracy checks](#accuracy_checks)
    3. [The N-random-models check](#nmodels_check)
    4. [Why the TATT tests use their own data vector](#synthetic_vectors)
3. [Appendix](#appendix)
    1. [FAQ: Do the tests keep their own data?](#frozen_copy)
    2. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

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

## The tests <a name="the_tests"></a>

The standard configurations get four tests each: a $\chi^2$ drift check
and a race-condition check, both in the NLA and in the TATT intrinsic-alignment
model. The TATT variants set

    IA_model: 1
    LSST_A2_1: 0.05
    LSST_BTA_1: 0.05
    LSST_A2_2: -1.51541

The two checks and their pass limits:

| check | pass limit                                        | a failure means                    |
|-------|---------------------------------------------------|------------------------------------|
| $\Delta\chi^2$ | the recomputed $\chi^2$ must stay within 0.2 of the value stored in `frozen/reference_chi2.json` | code or data changed the numbers |
| race condition | the fiducial evaluated on its own vs evaluated again after nine other cosmologies; the two must agree within $10^{-4}$ | leftover state or an OpenMP race   |

Everything the tests compare against lives under `frozen/`: one
snapshot of configurations, data, and reference values, captured
together when the references were generated and unchanged since. The
[Appendix](#appendix) explains how the snapshot is protected.

The test files and the configurations they cover:

| test | file | configuration | what it checks |
|---|---|---|---|
| 1 | `test_example1.py` | cosmic shear; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 2 | `test_example1.py` | cosmic shear; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 3 | `test_example1.py` | cosmic shear; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 4 | `test_example1.py` | cosmic shear; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 5 | `test_example2.py` | 3x2pt; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 6 | `test_example2.py` | 3x2pt; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 7 | `test_example2.py` | 3x2pt; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 8 | `test_example2.py` | 3x2pt; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 9 | `test_fastpt.py` | cosmic shear; IA modeling: TATT with python FAST-PT (`IA_code: 1` plus the fastpt theory block) instead of the C cfastpt | $\chi^2$ vs its own stored FASTPT reference; prints the FASTPT-minus-CFASTPT difference next to its stored value |
| 10 | `test_fastpt.py` | 3x2pt; IA modeling: TATT with python FAST-PT (`IA_code: 1` plus the fastpt theory block) instead of the C cfastpt | $\chi^2$ vs its own stored FASTPT reference; prints the FASTPT-minus-CFASTPT difference next to its stored value |
| 11 | `test_example2_2x2pt.py` | 2x2pt (`lsst_y1.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 12 | `test_example2_2x2pt.py` | 2x2pt (`lsst_y1.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 13 | `test_example2_2x2pt.py` | 2x2pt (`lsst_y1.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 14 | `test_example2_2x2pt.py` | 2x2pt (`lsst_y1.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |

### Advisory checks (`test_emul2.py`, E1-E4) <a name="advisory_checks"></a>

These checks test the hybrid emulators: trained machine-learning
networks replace the Boltzmann code. There is no pass/fail; each
check prints four quantities:

| printed quantity | meaning |
|---|---|
| emulator $\chi^2$ | the emulated pipeline evaluated at the fiducial point |
| drift | change against the stored emulator reference; nonzero means the installed emulator no longer reproduces its stored $\chi^2$ |
| $\lvert\chi^2_\text{emulator} - \chi^2_\text{exact}\rvert$ | the emulator error against the exact-physics $\chi^2$ at the same cosmology |
| recommendation | RECOMMENDED for actual data analysis when $\lvert\chi^2_\text{emulator} - \chi^2_\text{exact}\rvert < 0.2$, NOT recommended otherwise |

A race-condition check (OpenMP threading) runs as well, warning
instead of failing.

> [!NOTE]
> The trained-network files are read from
> `external_modules/data/emultrf`, not from the `frozen/` snapshot; the
> network device is pinned to `cpu` so the numbers do not depend on
> GPU availability.

#### Running Advisory checks <a name="run_advisory"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the advisory checks on their own

    python -m pytest ./projects/lsst_y1/tests/test_emul2.py

### Accuracy checks (`test_accuracy.py`, A1-A6) <a name="accuracy_checks"></a>

First we change one accuracy parameter at a time on the 3x2pt NLA
configuration, so a large $\Delta\chi^2$ can be attributed to the
parameter causing it (the scan includes `accuracyboost: 5` as a
stress test). Then checks A1-A6 re-evaluate cosmic shear, 3x2pt, and
2x2pt, with NLA and TATT, with every setting pushed beyond the
defaults at once:

| setting | raised to | what it controls |
|---------|-----------|------------------|
| `accuracyboost` (cosmolike) | 2 | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables |
| `integration_accuracy` (cosmolike) | 10 | extra refinement passes of cosmolike's numerical integrals |
| `lmax` (cosmolike) | 200000 | highest multipole of the internal harmonic-space $C_\ell$ tables that cosmolike transforms into the real-space correlation functions; arcminute scales need very high $\ell$ |
| `kmax_boltzmann` (cosmolike) | 40 | the k cutoff of the power spectrum the likelihood requests from CAMB |
| `AccuracyBoost` (CAMB) | 2 | CAMB's overall accuracy multiplier: denser sampling in every internal CAMB grid, the most expensive setting |
| `k_per_logint` (CAMB) | 50 | k samples CAMB computes per logarithmic interval of the transfer functions |
| `kmax` (CAMB) | 50 | highest k of CAMB's matter power spectrum; one physical cutoff with `kmax_boltzmann`, seen from the CAMB side |

Each check reports the $\Delta\chi^2$ between the high-accuracy and
the default evaluations: the numerical error of the default
settings. No pass/fail.

> [!NOTE]
> High-accuracy evaluations take minutes.

#### Running Accuracy checks <a name="run_accuracy"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the accuracy checks on their own

    python -m pytest ./projects/lsst_y1/tests/test_accuracy.py

To run every other test while skipping these:

    python -m pytest ./projects/lsst_y1/tests --ignore ./projects/lsst_y1/tests/test_accuracy.py

### The N-random-models check (opt-in) <a name="nmodels_check"></a>

This check repeats the accuracy measurement at N reproducible random
points drawn across the prior of the 3x2pt NLA configuration. Per
point:

1. draw the point;
2. generate a synthetic data vector at that point with the default
   settings;
3. evaluate the high-accuracy $\chi^2$ against that vector.

Again, we set the data vector at the point itself, so the
$\Delta\chi^2$ at the default settings is zero by construction; the
number step 3 computes is the $\Delta\chi^2$ directly.

> [!NOTE]
> The draws are reproducible: the check specifies the seed (the
> starting state) of the random number generator, so every run draws
> exactly the same points and the numbers can be compared across
> reruns and machines.

The report streams one block per model and ends with the
min/median/max $\Delta\chi^2$. Advisory: the deltas only have to be
finite.

Each model costs a default build+evaluation plus a high-accuracy
build+evaluation, minutes per model, so the check is off by default:
with `COCOA_ACCURACY_NMODELS` unset (or 0) it prints how to enable
it and passes.

#### Running the N-random-models check <a name="run_nmodels"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: enable and run the check

    COCOA_ACCURACY_NMODELS=10 python -m pytest \
        ./projects/lsst_y1/tests/test_accuracy.py -k nmodels

Running the file as a script accepts `--nmodels N` in place of the
environment variable.

### Why the TATT tests use their own data vector <a name="synthetic_vectors"></a>

All TATT variants evaluate against `frozen/data/tatt_lsst_y1.dataset`,
a data vector generated with TATT at the fiducial point when the
snapshot was created. Reason: against the shipped NLA-based vector the TATT $\chi^2$
sits away from its minimum, where it responds linearly to tiny
numerical changes; at its own minimum the response is quadratic and
the drift bounds stay meaningful.

# Appendix <a name="appendix"></a>

## :interrobang: FAQ: Do the tests keep their own data? <a name="frozen_copy"></a>

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
changes when any byte changes) of every file under `frozen/`. Each test
verifies the manifest first and refuses to run when a file under `frozen/` was
edited, naming the file. The result: users may change the live data
and examples freely, and nobody can quietly edit the snapshot
either.

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/lsst_y1/tests/generate_frozen_reference.py --overwrite

It rebuilds `frozen/` from the current project, prints the four new
reference $\chi^2$ values, and rewrites the manifest. Review the printed
$\chi^2$ values against the old references before committing: they define
what every later test run compares against.
