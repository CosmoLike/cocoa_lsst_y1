"""Defines the Cobaya likelihood lsst_y1.combo_2x2pt (LSST-Y1 2x2pt).

The entry `lsst_y1.combo_2x2pt` in the likelihood block of a yaml file makes
Cobaya import this module and build the class of the same name, with the
options and defaults of combo_2x2pt.yaml (next to this file). The base class
_cosmolike_prototype_base does all the work (its module docstring explains
the data vector, the probes and the order of the calls); this subclass
selects the probe "2x2pt": the gamma_t and w(theta) blocks of the data
vector (galaxy-galaxy lensing and galaxy clustering).
"""
# Cocoa links this likelihood directory into Cobaya as the package
# cobaya.likelihoods.lsst_y1, hence the import path below.
from cobaya.likelihoods.lsst_y1._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_lsst_y1_interface as ci
import numpy as np

class combo_2x2pt(_cosmolike_prototype_base):
  """Evaluates the 2x2pt likelihood (probe "2x2pt")."""
  def initialize(self):
    """Configure the base likelihood for probe "2x2pt".

    Cobaya calls initialize once, when it builds the likelihood;
    super(...) reaches the method of the base class, which reads the
    .dataset file and sets up cosmolike.
    """
    super(combo_2x2pt,self).initialize(probe="2x2pt")