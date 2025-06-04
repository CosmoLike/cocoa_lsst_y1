from cobaya.likelihoods.lsst_y1._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_lsst_y1_interface as ci
import numpy as np

class combo_xi_ggl(_cosmolike_prototype_base):
  def initialize(self):
    super(combo_xi_ggl,self).initialize(probe="xi_ggl")