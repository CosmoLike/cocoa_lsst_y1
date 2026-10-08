"""Runs the lsst_y1 hybrid-emulator profile mode for configuration 1
(lsst_y1.cosmic_shear, NLA): one sampled parameter is fixed at each value of
a grid while the remaining ones are minimized (requires --minfile, a saved
minimization).

In the hybrid examples (use_emulator: 2) trained emulators replace CAMB for
the background expansion and the matter power spectra, while cosmolike still
computes the survey projections. This file binds the shared driver run() of
cosmolike_core/cocoa_hybrid_sampling.py (its module docstring explains the
three modes and every option) to this project and to
EXAMPLE_EMUL2_EVALUATE1.yaml, whose sampler.evaluate.override block is the
fiducial point.

From the Cocoa/ folder, with Cocoa activated (start_cocoa.sh), check the
setup without sampling:

    python ./projects/lsst_y1/EXAMPLE_EMUL2_PROFILE1.py --check

The results go to projects/lsst_y1/chains/EXAMPLE_EMUL2_PROFILE1.*
(--outroot changes the name; an existing result is never overwritten). The
project README gives the MPI commands.
"""

from pathlib import Path
import sys

# project = the folder of this file (projects/lsst_y1); core = the
# cosmolike_core checkout, which holds cocoa_hybrid_sampling.py and is
# put first on the import path (sys.path)
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# __name__ is "__main__" only when this file runs as a script; run()
# parses the command line and performs the whole run
if __name__ == "__main__":
    run(mode="profile", project=project, example=1)
