"""Check that the YAML entry point evaluates the cosmology it was given.

These are reader/configuration checks: CAMB and covariance kernels do not
run. Tests keep malformed input from silently selecting a different model
or a random prior point on an HPC job. The reader is
load_run_configuration (cosmolike_notebook_utils/covariance/command_line.py),
the function compute_covariance.py calls first; each test edits the
shipped EXAMPLE_EVALUATE_COVARIANCE.yaml in memory, writes it into
tmp_path (a fresh temporary folder pytest creates for every test that
names it as an argument) and reads it back.
"""

import os
from pathlib import Path
import sys

import pytest
from cobaya.yaml import yaml_dump, yaml_load_file

# project = projects/lsst_y1; its covariance/ folder holds the survey
# adapter lsst_y1_covariance.py, imported below
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"covariance"))

import lsst_y1_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import load_run_configuration


def input_info():
    """Read the actual project example without depending on the process cwd."""
    info = yaml_load_file(file_name=str(project/"EXAMPLE_EVALUATE_COVARIANCE.yaml"))
    info["theory"]["camb"]["path"] = str(project.parents[1]/"external_modules/code/CAMB")
    return info


def write_input(tmp_path, info):
    """Write a test input using Cobaya's matching YAML serializer."""
    filename = tmp_path/"evaluate.yaml"
    filename.write_text(yaml_dump(info=info))
    return filename


def test_fixed_cosmology_and_independent_boosts(tmp_path):
    """Theory, numerical refinements and the requested point retain their meaning."""
    info = input_info()
    info["params"]["H0"] = {"value": 68.0, "latex": "H_0"}
    info["params"]["w0pwa"] = {"value": "lambda w: w+0.2"}
    info["theory"]["camb"]["extra_args"]["AccuracyBoost"] = 1.5
    info["covariance"]["accuracy_boost"] = 2
    info["covariance"]["window_accuracyboost"] = 3
    info["covariance"]["integration_accuracy"] = 1
    settings, run = load_run_configuration(
        filename=write_input(tmp_path=tmp_path, info=info), survey=survey,
    )
    assert settings["cosmology"]["H0"] == 68.0
    assert settings["cosmology"]["w0pwa"] == -0.8
    assert settings["cosmology"]["CAMBAccuracyBoost"] == 1.5
    # covariance_accuracy's rules: nwindow = 16384 x window boost (3) x
    # global boost (2) + 1 samples; integration_accuracy 1 selects the
    # 128-node quadrature rule
    assert settings["nwindow"] == 16384*3*2+1
    assert settings["radial_nquad"] == 128
    assert run["space"] == "real"
    assert run["threads"] == int(os.environ["OMP_NUM_THREADS"])


def test_evaluate_override_is_explicit(tmp_path):
    """A prior parameter takes its specified override, never a random ref draw."""
    info = input_info()
    info["params"]["H0"] = {"prior": {"min": 55, "max": 91}, "ref": 70}
    filename = write_input(tmp_path=tmp_path, info=info)
    with pytest.raises(ValueError, match="override must supply"):
        load_run_configuration(filename=filename, survey=survey)
    info["sampler"]["evaluate"]["override"] = {"H0": 67.0}
    settings, run = load_run_configuration(
        filename=write_input(tmp_path=tmp_path, info=info), survey=survey,
    )
    assert settings["cosmology"]["H0"] == 67.0


def test_unsupported_physics_or_misspelled_options_fail(tmp_path):
    """Unsupported theory and sampled cosmologies fail before numerical setup."""
    info = input_info()
    info["params"]["mnu"] = 0.06
    with pytest.raises(ValueError, match="requires mnu: 0"):
        load_run_configuration(
            filename=write_input(tmp_path=tmp_path, info=info), survey=survey,
        )
    info = input_info()
    info["theory"]["camb"]["extra_args"]["AccuracyBost"] = 2
    with pytest.raises(ValueError, match="unsupported CAMB options"):
        load_run_configuration(
            filename=write_input(tmp_path=tmp_path, info=info), survey=survey,
        )
    info = input_info()
    info["sampler"] = {"mcmc": {}}
    with pytest.raises(ValueError, match="sampler: evaluate"):
        load_run_configuration(
            filename=write_input(tmp_path=tmp_path, info=info), survey=survey,
        )


def test_gaussian_choices_and_environment_threads(tmp_path):
    """Keep Gaussian physics and shell resource choices explicit in the YAML."""
    info = input_info()
    info["covariance"]["gaussian"] = {"nonlimber": True, "ia": "TATT", "A1": 0.6,
                                       "A2": 0.2, "B_TA": 0.5}
    info["covariance"]["accuracy_boost"] = 2
    info["covariance"]["nonlimber_accuracyboost"] = 2
    settings, unused = load_run_configuration(
        filename=write_input(tmp_path=tmp_path, info=info), survey=survey,
    )
    # a scalar A1 becomes one value per source bin; nonlimber_nchi =
    # 4096 x non-Limber boost (2) x global boost (2) + 1; nonlimber_lmax
    # = 1000 x global boost
    assert settings["gaussian"]["A1"] == [0.6]*5
    assert settings["nonlimber_nchi"] == 4096*2*2+1
    assert settings["nonlimber_lmax"] == 2000
    info["covariance"]["threads"] = 8
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        load_run_configuration(
            filename=write_input(tmp_path=tmp_path, info=info), survey=survey,
        )
