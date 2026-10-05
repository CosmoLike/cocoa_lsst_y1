"""Compute and save the lsst_y1 covariance through the production interface.

From an activated Cocoa installation:
    python projects/lsst_y1/covariance/compute_covariance.py \
        projects/lsst_y1/EXAMPLE_EVALUATE_COVARIANCE.yaml

See --help and covariance/README.md for space and accuracy options.
The numerical model and survey settings are shared with the notebook.
"""

import os
from pathlib import Path
import sys

# Set external numerical libraries to one worker before their first import.
# CosmoLike's own OpenMP team is controlled by OMP_NUM_THREADS.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; it does not launch MPI workers or a sampler for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_lsst_y1_interface as ci
import lsst_y1_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="real", joint=False,
    )
