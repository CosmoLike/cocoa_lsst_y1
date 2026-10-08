"""Defines the Cobaya likelihood lsst_y1.combo_xi_ggl (shear and galaxy-galaxy lensing).

The entry `lsst_y1.combo_xi_ggl` in the likelihood block of a yaml file
makes Cobaya import this module and build the class of the same name, with
the options and defaults of combo_xi_ggl.yaml (next to this file). The base
class _cosmolike_prototype_base does all the work (its module docstring
explains the data vector, the probes and the order of the calls); this
subclass selects the probe "xi_ggl": the xi_+- and gamma_t blocks of the
data vector (cosmic shear and galaxy-galaxy lensing).
"""
# Cocoa links this likelihood directory into Cobaya as the package
# cobaya.likelihoods.lsst_y1, hence the import path below.
from cobaya.likelihoods.lsst_y1._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_lsst_y1_interface as ci
import numpy as np

class combo_xi_ggl(_cosmolike_prototype_base):
  """Evaluates the shear plus galaxy-galaxy lensing likelihood (probe "xi_ggl")."""
  def initialize(self):
    """Configure the base likelihood for probe "xi_ggl".

    Cobaya calls initialize once, when it builds the likelihood;
    super(...) reaches the method of the base class, which reads the
    .dataset file and sets up cosmolike.
    """
    super(combo_xi_ggl,self).initialize(probe="xi_ggl")