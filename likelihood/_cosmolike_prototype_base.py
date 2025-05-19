# Python 2/3 compatibility - must be first line
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time

# Local
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

import euclidemu2
import math

import cosmolike_lsst_y1_interface as ci

#dark emulator
from scipy.interpolate import InterpolatedUnivariateSpline as ius # KI add
from dark_emulator import darkemu
from dark_emulator import model_hod
from dark_emulator import fftlog
from demu_pk_gx import demu_pk_gx
survey = "LSST"

class _cosmolike_prototype_base(DataSetLikelihood):

  def initialize(self, probe):

    # ------------------------------------------------------------------------
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))

    self.probe = probe

    self.data_vector_file = ini.relativeFileName('data_file')

    self.cov_file = ini.relativeFileName('cov_file')

    self.mask_file = ini.relativeFileName('mask_file')

    self.lens_file = ini.relativeFileName('nz_lens_file')

    self.source_file = ini.relativeFileName('nz_source_file')

    self.lens_ntomo = ini.int("lens_ntomo") #5

    self.source_ntomo = ini.int("source_ntomo") #4

    self.ntheta = ini.int("n_theta")

    self.theta_min_arcmin = ini.float("theta_min_arcmin")

    self.theta_max_arcmin = ini.float("theta_max_arcmin")
    
    # ------------------------------------------------------------------------
 
    self.nz_interp_1d=int(500 + 250*self.accuracyboost)
    # EUCLID EMULATOR CAN ONLY HANDLE 100 Z's BELOW Z=10
    max_nz_interp_2d=150
    if self.non_linear_emul == 1:
      max_nz_interp_2d=99
    else: 
      max_nz_interp_2d=150
    self.nz_interp_2d=int(min(60 + 15*self.accuracyboost,150))
    
    self.nk_interp_2d=int(500 + 250*self.accuracyboost)

    self.z_interp_1D = np.linspace(0, 3.0, max(100,int(0.80*self.nz_interp_2d)))
    self.z_interp_1D = np.concatenate(
      (self.z_interp_1D,
      np.linspace(3.0,10.1,max(100,int(0.20*self.nz_interp_2d)))),
      axis=0)
    self.z_interp_1D = np.concatenate(
      (self.z_interp_1D,
      np.linspace(1080, 2000, 20)),
      axis=0) #CMB 6x2pt g_CMB (possible in the future)
    self.z_interp_1D[0] = 0
    
    self.z_interp_2D = np.linspace(0,3.0,max(10,int(0.85*self.nz_interp_2d)))
    self.z_interp_2D = np.concatenate(
      (self.z_interp_2D, 
       np.linspace(3.01, 10, max(10,int(0.15*self.nz_interp_2d)))), 
       axis=0)
    self.z_interp_2D[0] = 0

    self.len_z_interp_2D = len(self.z_interp_2D)
    self.len_log10k_interp_2D = self.nk_interp_2d
    self.log10k_interp_2D = np.linspace(-4.2,2.0,self.len_log10k_interp_2D)

    # Cobaya wants k in 1/Mpc
    self.k_interp_2D = np.power(10.0,self.log10k_interp_2D)
    self.len_k_interp_2D = len(self.k_interp_2D)
    self.len_pkz_interp_2D = self.len_log10k_interp_2D*self.len_z_interp_2D
    self.extrap_kmax = 2.5e2 * self.accuracyboost

    # ------------------------------------------------------------------------

    ci.initial_setup()
    
    ci.init_accuracy_boost(
      self.accuracyboost, 
      self.samplingboost, 
      self.integration_accuracy)

    ci.init_probes(possible_probes=self.probe)

    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

    ci.init_cosmo_runmode(is_linear=False)

    ci.init_IA(
      ia_model = int(self.IA_model), 
      ia_redshift_evolution = int(self.IA_redshift_evolution))

    ci.init_redshift_distributions_from_files(
      lens_multihisto_file=self.lens_file,
      lens_ntomo=int(self.lens_ntomo), 
      source_multihisto_file=self.source_file,
      source_ntomo=int(self.source_ntomo))  

    ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)
     
    if self.probe != "xi":
      # (b1, b2, bs2, b3, bmag). 0 = one amplitude per bin
      ci.init_bias(bias_model=self.bias_model)

    if self.create_baryon_pca:
      self.use_baryon_pca = False
    else:
      if ini.string('baryon_pca_file', default=''):
        baryon_pca_file = ini.relativeFileName('baryon_pca_file')
        self.baryon_pcs = np.loadtxt(baryon_pca_file)
        self.log.info('use_baryon_pca = True')
        self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
        self.use_baryon_pca = True
        ci.set_baryon_pcs(eigenvectors=self.baryon_pcs)
      else:
        self.log.info('use_baryon_pca = False')
        self.use_baryon_pca = False

    self.npcs = 4
    self.baryon_pcs_qs = np.zeros(self.npcs)
        
    if self.non_linear_emul == 1:
      self.emulator = ee2=euclidemu2.PyEuclidEmulator()
    elif self.non_linear_emul == 3: # KI add
      self.emulator = demu = darkemu.base_class() 

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    return {
      "As": None,
      "H0": None,
      "omegam": None,
      "omegab": None,
      "mnu": None,
      "w": None,
      "Pk_interpolator": {
        "z": self.z_interp_2D,
        "k_max": self.kmax_boltzmann * self.accuracyboost,
        "nonlinear": (True,False),
        "vars_pairs": ([("delta_tot", "delta_tot")])
      },
      "comoving_radial_distance": {
        "z": self.z_interp_1D
      # Get comoving radial distance from us to redshift z in Mpc.
      },
      "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
      }
    }

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_cosmo_related(self):
    h = self.provider.get_param("H0")/100.0

    # Compute linear matter power spectrum
    PKL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
      nonlinear=False, extrap_kmax = self.extrap_kmax)

    # Compute non-linear matter power spectrum
    PKNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
      nonlinear=True, extrap_kmax = self.extrap_kmax)

    lnPL  = np.empty(self.len_pkz_interp_2D)
    lnPNL = np.empty(self.len_pkz_interp_2D)
    
    t1 = PKNL.logP(self.z_interp_2D, self.k_interp_2D).flatten()
    t2 = PKL.logP(self.z_interp_2D, self.k_interp_2D).flatten()
    
    # Cosmolike wants k in h/Mpc
    log10k_interp_2D = self.log10k_interp_2D - np.log10(h)
    
    for i in range(self.len_z_interp_2D):
      lnPL[i::self.len_z_interp_2D]  = t2[i*self.len_k_interp_2D:(i+1)*self.len_k_interp_2D]
    lnPL  += np.log((h**3))

    if self.non_linear_emul == 1:

      params = {
        'Omm'  : self.provider.get_param("omegam"),
        'As'   : self.provider.get_param("As"),
        'Omb'  : self.provider.get_param("omegab"),
        'ns'   : self.provider.get_param("ns"),
        'h'    : h,
        'mnu'  : self.provider.get_param("mnu"), 
        'w'    : self.provider.get_param("w"),
        'wa'   : 0.0
      }

      kbt = np.power(10.0, np.linspace(-2.0589, 0.973, self.len_k_interp_2D))
      kbt, tmp_bt = self.emulator.get_boost(params, self.z_interp_2D, kbt)
      logkbt = np.log10(kbt)

      for i in range(self.len_z_interp_2D):    
        interp = interp1d(logkbt, 
            np.log(tmp_bt[i]), 
            kind = 'linear', 
            fill_value = 'extrapolate', 
            assume_sorted = True
          )

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0
    
        lnPNL[i::self.len_z_interp_2D]  = lnPL[i::self.len_z_interp_2D] + lnbt
      
    elif self.non_linear_emul == 2:

      for i in range(self.len_z_interp_2D):
        lnPNL[i::self.len_z_interp_2D]  = t1[i*self.len_k_interp_2D:(i+1)*self.len_k_interp_2D]  
      lnPNL += np.log((h**3))      

    elif self.non_linear_emul == 3:

      def pkmm_demu(k, z):
        xs = np.logspace(-3, 3, 2000)
        xinl = self.emulator.get_xinl(xs, z)
        # N_extrap_high=1024 is better than 0
        ks, pk = fftlog.xi2pk(xs,xinl,nu=1.01,N_extrap_low=1024,N_extrap_high=1024)
        return ius(ks, pk)(k)

      # read cosmo. params.
      omc = self.provider.get_param("omegam")-self.provider.get_param("omegab")
      cparam = np.array([
        self.provider.get_param("omegab")*h**2.,
        omc*h**2.,
        1-self.provider.get_param("omegam"),
        np.log(self.provider.get_param("As")*10**10),
        self.provider.get_param("ns"),
        self.provider.get_param("w")
      ])
      self.emulator.set_cosmology(cparam)

      for i, z_2D in enumerate(self.z_interp_2D):
        if z_2D > 1.48:
          lnPNL[i::self.len_z_interp_2D] = \
            t1[i*self.len_k_interp_2D:(i+1)*self.len_k_interp_2D] + np.log((h**3))
          continue

        log10k_interp = np.linspace(np.log10(2e-3),1.,self.len_k_interp_2D)
        pknl = pkmm_demu(10**log10k_interp, z_2D)
        lnpknl = interp1d(log10k_interp, 
                          np.log(pknl), 
                          fill_value='extrapolate', 
                          assume_sorted=True
                         )(log10k_interp_2D)
        lnPNL[i::self.len_z_interp_2D] = lnpknl
    
    else:
    
      raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)

    G_growth = np.sqrt(PKL.P(self.z_interp_2D,0.0005)/PKL.P(0,0.0005))
    G_growth = G_growth*(1 + self.z_interp_2D)    # do not merge these lines PI
    G_growth = G_growth/G_growth[len(G_growth)-1] # do not merge these lines PII

    ci.set_cosmology(
      omegam=self.provider.get_param("omegam"),
      H0=self.provider.get_param("H0"),
      log10k_2D=log10k_interp_2D,
      z_2D=self.z_interp_2D,
      lnP_linear=lnPL, 
      lnP_nonlinear=lnPNL, 
      G=G_growth,
      z_1D=self.z_interp_1D,
      chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
    )
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_source_related(self, **params_values):
    ci.set_nuisance_shear_calib(
      M = [
        params_values.get(p, None) for p in [
          survey+"_M"+str(i+1) for i in range(self.source_ntomo)
        ]
      ]
    )
    ci.set_nuisance_shear_photoz(
      bias = [
        params_values.get(p, None) for p in [
          survey+"_DZ_S"+str(i+1) for i in range(self.source_ntomo)
        ]
      ]
    )
    ci.set_nuisance_ia(
      A1 = [
        params_values.get(p, None) for p in [
          survey+"_A1_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
      A2 = [
        params_values.get(p, None) for p in [
          survey+"_A2_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
      B_TA = [
        params_values.get(p, None) for p in [
          survey+"_BTA_"+str(i+1) for i in range(self.source_ntomo)
        ]
      ],
    )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_lens_related(self, **params_values):
    ci.set_nuisance_bias(
      B1 = [
        params_values.get(p, None) for p in [
          survey+"_B1_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ],
      B2 = [
        params_values.get(p, None) for p in [
          survey+"_B2_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ],
      B_MAG = [
        params_values.get(p, None) for p in [
          survey+"_BMAG_"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )
    ci.set_nuisance_clustering_photoz(
      bias = [
        params_values.get(p, None) for p in [
          survey+"_DZ_L"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )
    ci.set_point_mass(
      PMV = [
        params_values.get(p, None) for p in [
          survey+"_PM"+str(i+1) for i in range(self.lens_ntomo)
        ]
      ]
    )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
