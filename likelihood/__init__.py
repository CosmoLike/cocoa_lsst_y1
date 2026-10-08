# Marks this directory as a Python package. Cocoa links
# projects/lsst_y1/likelihood into Cobaya's likelihoods directory under the
# name lsst_y1, so Python imports it as cobaya.likelihoods.lsst_y1. The
# package holds the five likelihood modules (cosmic_shear, combo_xi_ggl,
# combo_xi_gg, combo_2x2pt, combo_3x2pt), each with a yaml file of option
# defaults, the nuisance-parameter priors (params_source.yaml,
# params_lens.yaml) and their shared code, _cosmolike_prototype_base.py.
