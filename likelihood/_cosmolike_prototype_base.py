"""Defines the Cobaya likelihood base class of the five LSST-Y1 real-space likelihoods.

Cobaya, the sampler framework, builds one likelihood object for each entry of
the likelihood block of a yaml file (for example lsst_y1.cosmic_shear) and
calls its logp method at every sampled point. The five likelihood modules of
this directory (cosmic_shear, combo_xi_ggl, combo_xi_gg, combo_2x2pt,
combo_3x2pt) are subclasses of _cosmolike_prototype_base that only name their
probe. The physics runs in the compiled cosmolike library, imported below as
ci (cosmolike_lsst_y1_interface, built from interface/interface.cpp).

Terms used in this file:
  data vector  = the 1D array of every measured two-point correlation function,
                 in a fixed block order: cosmic shear xi_+ then xi_- (one
                 entry per source-source bin pair and angle), galaxy-galaxy
                 lensing gamma_t (lens-source bin pairs), galaxy clustering
                 w(theta) (one per lens bin). With 5 source bins, 5 lens bins
                 and 26 angular bins it has 780 + 650 + 130 = 1560 entries.
  mask         = one 0/1 flag per data-vector entry; entries flagged 0 (scale
                 cuts) are excluded from the chi2 and set to 0 in the theory.
  .dataset     = small text file with one "key = value" line per setting: the
                 data-vector, covariance, mask and n(z) file names, the bin
                 counts and the angular range (example:
                 external_modules/data/lsst_y1/lsst_y1_M1_GGL0.05.dataset).
  probe        = the data-vector blocks entering the chi2: "xi" (cosmic shear),
                 "xi_ggl", "xi_gg", "2x2pt" (gamma_t and w) or "3x2pt" (all).
  nuisance parameters = survey systematics sampled together with cosmology,
                 named LSST_<kind><bin>: photo-z shifts LSST_DZ_S1, LSST_DZ_L1,
                 shear calibration LSST_M1, linear bias LSST_B1_1, intrinsic
                 alignment amplitude LSST_A1_1, point mass LSST_PM1, ...

What happens at one sampled point (logp):
  get_datavector
    -> set_cosmo_related : P(k, z), growth factor and distances to cosmolike
    -> set_lens_related  : lens-sample nuisance parameters to cosmolike
    -> set_source_related: source-sample nuisance parameters to cosmolike
    -> ci.compute_data_vector_masked: the theory data vector
  compute_logp: ln L = -chi2/2, with the covariance read by initialize

The yaml key use_emulator selects where the theory comes from:
  0 = CAMB computes P(k, z) and cosmolike the data vector;
  1 = machine-learning emulators return the xi, gamma_t and w blocks directly;
      cosmolike only adds shear calibration, point masses and baryon PCs;
  2 = "hybrid" mode: emulators replace CAMB for P(k, z) and the distances,
      and cosmolike computes the data vector as in mode 0.
"""
# The three __future__ imports are Python 2 compatibility switches; Python 3
# already behaves this way. Python requires a __future__ import before every
# other statement of the module (only the docstring and comments may precede it).
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools

# Cobaya (the sampler framework) and getdist's IniFile (the .dataset reader)
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

# euclidemu2 = EuclidEmulator2, an emulator of the nonlinear boost
# B(k, z) = P_nonlinear/P_linear (used when non_linear_emul = 1)
import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print the wall-clock time spent inside a with-block (a debugging aid).

  Usage: `with timer("set_cosmology"): ...` prints "set_cosmology: 0.0123s"
  when the block ends. The @contextmanager decorator turns this generator
  function into a context manager: the line before `yield` runs when the
  with-block starts, the line after it when the block finishes. A block that
  raises an exception prints nothing (there is no try/finally around yield).

  Arguments:
    label = text printed before the elapsed time

  Side effects: prints one line to standard output.
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

# ci = the compiled cosmolike library of this project
# (interface/cosmolike_lsst_y1_interface.so, built from interface/interface.cpp
# with pybind11); every ci.<name> call runs C/C++ code and changes or reads
# cosmolike's global state.
import cosmolike_lsst_y1_interface as ci

# OpenMP thread count of cosmolike's parallel loops, read once when this module
# is imported: the environment variable OMP_NUM_THREADS, or 1 (serial) when it
# is unset. with_omp_threads re-applies it before each cosmolike call.
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Return a version of fn that first restores cosmolike's thread count.

    Cosmolike's hot loops are parallelized with OpenMP and use as many
    threads as omp_get_max_threads() returns, which starts at the value of
    OMP_NUM_THREADS. Some Python libraries call omp_set_num_threads(1)
    without saying so; that call changes the thread count of the whole
    process, and cosmolike's parallel loops would then run on one core.
    The wrapper calls ci.set_omp_threads(COSMOLIKE_OMP_THREADS) before every
    call of fn. The same C function also limits OpenBLAS to one thread, so
    BLAS threads do not compete with cosmolike's OpenMP threads.

    Usage: the line @with_omp_threads above `def set_cosmo_related(self)`
    replaces that method by `wrapper` when the class is defined. wrapper
    accepts any positional (*args) and keyword (**kwargs) arguments and
    passes them unchanged to fn; functools.wraps copies the name and
    docstring of fn onto wrapper, so logs and help() still show them.

    Arguments:
      fn = the function or method to wrap

    Returns:
      wrapper, a function with the arguments and return value of fn.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of every nuisance-parameter name this likelihood reads from Cobaya,
# e.g. LSST_DZ_S1 (photo-z shift of source bin 1) or LSST_B1_1 (linear bias of
# lens bin 1). It must match the names in params_source.yaml and
# params_lens.yaml; the subclass modules import it too.
survey = "LSST"

class _cosmolike_prototype_base(DataSetLikelihood):
  """Evaluates the LSST-Y1 real-space likelihood of one probe choice.

  A subclass names the probe (the data-vector blocks in the chi2) and passes
  it to initialize. Cobaya then calls, once, get_requirements (the theory
  quantities to compute at every point) and, at every sampled point,
  logp(**params) with the nuisance-parameter values.

  Options come from the likelihood yaml (likelihood/cosmic_shear.yaml and
  the others, overridden by the likelihood block of an EXAMPLE_*.yaml);
  Cobaya stores each key as an attribute of self before initialize runs:
    path, data_file  = directory and name of the .dataset file
    accuracyboost    = multiplies the z and k sampling of the tables below
                       and cosmolike's internal sampling (1 = default)
    integration_accuracy = cosmolike's integration accuracy level (0 = default)
    pk_z_refinement  = extra integer refinement of the z nodes of P(k, z)
    lmax             = largest multipole of the Legendre sums that turn C_ell
                       into real-space correlation functions
    kmax_boltzmann   = k_max [1/Mpc] requested from CAMB, times accuracyboost
    non_linear_emul  = nonlinear P(k): 1 = EuclidEmulator2 boost times the
                       linear P(k) below z = 10, 2 = CAMB's nonlinear model
    IA_model, IA_redshift_evolution, IA_code = intrinsic-alignment model,
                       its redshift dependence and the FAST-PT implementation
                       (C = 0, Python theory block = 1)
    bias_model       = redshift-evolution code of each galaxy-bias term
    use_emulator     = 0 CAMB, 1 data-vector emulators, 2 hybrid emulators
    external_nz_modeling = send the n(z) arrays from Python at every point
    use_baryon_pca, create_baryon_pca, add_baryons_on_dv,
    external_baryon_suppression = baryonic-feedback treatment (initialize)
    print_datavector, print_datavector_file = write every theory data vector
                       to a text file
    debug            = cosmolike log level debug (True) or info (False)
  """

  def initialize(self, probe):
    """Read the .dataset file and configure cosmolike for one probe.

    Cobaya calls initialize once, when it builds the likelihood. In order:
      1. read the .dataset file (file names, bin counts, angular range);
      2. build the z and k grids of the tables sent to cosmolike at every
         point (z_interp_1D, z_interp_2D, z_interp_2D_camb,
         log10k_interp_2D);
      3. configure cosmolike (probes, angular binning, model options, n(z))
         and load the data vector, the covariance and the mask;
      4. select the baryonic-feedback treatment.
    Cosmolike keeps its configuration in C global variables shared by every
    likelihood object of the Python process, so each option is set here
    explicitly, even when it has its default value.

    Arguments:
      probe = "xi", "xi_ggl", "xi_gg", "2x2pt" or "3x2pt"

    Raises:
      LoggedError when pk_z_refinement is not a positive integer.

    Side effects: changes cosmolike's global state; sets the grid
    attributes above and their lengths; may reset IA_code, use_baryon_pca,
    add_baryons_on_dv and external_baryon_suppression (comments below).
    """
    # ini = the .dataset file parsed by getdist's IniFile; relativeFileName
    # returns the file named by a key, with a relative name resolved against
    # the directory of the .dataset file
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
    # z_interp_1D = redshifts of the 1D tables, the comoving distance chi(z)
    # and the growth factor G(z). Three uniform blocks: [0, 3) for the galaxy
    # samples, [3, 50.1) and [1070, 1100], which brackets recombination
    # (z ~ 1090) so the distance to the CMB last-scattering surface is
    # tabulated for CMB-lensing kernels. At accuracyboost = 1 (tmp = 1250) the
    # blocks have 1000, 500 and 125 nodes (dz = 0.003, 0.094 and 0.24); the
    # max() calls keep at least 100, 100 and 50 nodes.
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # z_interp_2D = the z nodes of the 2D tables ln P(k, z) sent to cosmolike.
    # Cosmolike interpolates linearly in z between exactly these nodes (it
    # indexes each uniform block directly and never regrids). Linear
    # interpolation leaves a residual of order dz^2 shaped like a sawtooth:
    # zero at the nodes, largest between them. Two grids that do not share
    # nodes therefore differ by the full residual. A node count that is not
    # nested across boost values, such as min(120 + 20*boost, 250), moves the
    # sawtooth at every boost instead of shrinking it: in roman_kl the
    # clustering chi2 then jumps by order unity, and in this project by less,
    # without converging as the boost grows.
    #
    # The dyadic factor m = 2^ceil(log2(boost)), capped at 16, refines each
    # uniform block by an integer factor with fixed endpoints, so
    #   (a) every block stays uniform (cosmolike keeps its direct indexing of
    #       two uniform blocks, with no search);
    #   (b) the nodes of a coarser grid are a subset of the nodes of every
    #       finer grid, and raising the boost is a true refinement (the error
    #       falls like 1/m^2);
    #   (c) at boost 1 (m = 1) the grid has 105 + 35 = 140 nodes, the grid
    #       sent to CAMB (z_interp_2D_camb below).
    # The low block multiplies its node count (endpoint=False, spacing
    # 3/(105 m)); the high block multiplies its interval count (endpoint=True:
    # 35 nodes = 34 intervals -> 34 m + 1 nodes). The table stops at
    # z = 49.99, inside the z <= 50 range of the P(k) emulator of the hybrid
    # mode (use_emulator = 2); only CMB lensing reads P(k, z) at such high z.
    #
    # pk_z_refinement multiplies m once more, independently of the boost. A
    # Fourier-space data vector reads P(k, z) at fixed multipoles, where the
    # linear-in-z residual does not average out as it does in the real-space
    # transform: roman_fourier's 3x2pt chi2 changes by 0.25, 0.030 and 0.002
    # from m = 1 to 2, 4 and 8; roman_real's and lsst_y1's by at most 0.004
    # from m = 1 to 2.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False), 
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # z_interp_2D_camb = the redshifts requested from CAMB through the
    # Pk_interpolator requirement. CAMB's transfer module accepts at most 256
    # redshifts, so this list is the boost-independent 140-node grid (the
    # m = 1 grid above). The denser nested nodes of z_interp_2D only
    # re-evaluate the smooth z-spline CAMB builds through these redshifts when
    # the cosmolike tables are filled: raising the boost refines that
    # resampling, where the linear-in-z residual arises, and CAMB's limit is
    # never exceeded.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False),
                                            np.linspace(3.0,49.99,35)),axis=0)

    # log10k_interp_2D = log10 of k [1/Mpc, CAMB's unit] at the k nodes of the
    # P(k, z) tables: 1500 nodes at boost 1 between k = 1.0e-5 and 100/Mpc.
    # set_cosmo_related subtracts log10(h) to send k in h/Mpc to cosmolike.
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    # initial_setup resets cosmolike's global configuration to its defaults
    # (every probe off); init_probes turns on the blocks of this probe, and
    # init_binning sets n_theta log-spaced angular bins between
    # theta_min_arcmin and theta_max_arcmin.
    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

#    ci.init_ggl_exclude(np.array(self.ggl_exclude).flatten())

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    # The model options below are set at every initialize, each from its yaml
    # key or, when the key is absent, from the default written in the
    # getattr call: cosmolike's options are C global variables, so a value
    # left unset would be inherited from the previous likelihood built in
    # the same process. The meaning of each value is in likelihood/*.yaml.
    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # the non-Limber FFTLog chi grid, refined on top of the accuracy boost
    # (narrow lens bins need it: see init_nonlimber_accuracy_boost)
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 0)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model (HOD) galaxy power
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # Cosmolike's halo statistics (mass variance, mass function) use the
    # linear spectrum of cold dark matter + baryons, P_cb. The hybrid
    # emulators provide no P_cb, so get_neutrino_inputs approximates it from
    # the linear total-matter spectrum; the log line records that choice.
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

    if self.use_emulator == 1:
      # Emulated data vector: cosmolike computes only the point-mass term of
      # gamma_t (from the n(z) and the distances) and applies the shear
      # calibration and the mask, so a low accuracy (boost 0.35, integration
      # level -1) replaces the yaml accuracy settings.
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo),
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)
      ci.init_accuracy_boost(accuracy_boost=0.35,
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      # init_ntable_lmax = largest multipole of the Legendre sums that turn
      # C_ell into xi(theta); init_accuracy_boost = sampling and integration
      # accuracy of cosmolike's tables; init_cosmo_runmode(is_linear=False)
      # keeps the nonlinear P(k) (True would force the linear one everywhere).
      ci.init_ntable_lmax(lmax=int(self.lmax))
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      ci.init_cosmo_runmode(is_linear=False)

      # external_nz_modeling: the n(z) tables are read into Python arrays
      # (self.lens_nz, self.source_nz) and sent to cosmolike at every point by
      # set_lens_related and set_source_related, so a user function can
      # modify them there; otherwise cosmolike reads and keeps the files.
      if self.external_nz_modeling:
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file = self.lens_file,
          lens_ntomo = int(self.lens_ntomo), 
          source_multihisto_file = self.source_file,
          source_ntomo = int(self.source_ntomo)) 

      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
        # NLA (IA_model = 0) needs no FAST-PT intrinsic-alignment terms, so
        # the C FAST-PT (IA_code = 0) replaces the Python FAST-PT theory
        # block, which get_requirements then does not request; the C code
        # also computes the one-loop galaxy-bias terms.
        self.IA_code = 0
      ci.init_IA(ia_model = int(self.IA_model),
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      if self.probe != "xi":
        # bias_model = one redshift-evolution code per bias term, in the
        # order [b1, b2, bs2, b3, bmag, bK] (cosmolike's bias.h); code 0 =
        # one free amplitude per lens bin (b3: 1 = computed from b1)
        ci.init_bias(bias_model=self.bias_model)

      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # Baryonic-feedback options:
      #   external_baryon_suppression = a Cobaya theory block returns the
      #     ratio S(k, z) of P(k) with and without baryons, applied to the
      #     nonlinear P(k) in set_cosmo_related;
      #   use_baryon_pca = the theory data vector receives sum_i Q_i PC_i,
      #     with the amplitudes Q_i = LSST_BARYON_Q<i> sampled;
      #   create_baryon_pca = compute the principal components (PCs) from the
      #     hydrodynamical simulations listed in baryon_pca_select_sims and
      #     write them to filename_baryon_pca (internal_get_datavector);
      #   add_baryons_on_dv = multiply P(k) by the ratio measured in the
      #     simulation which_bsims_add_on_dv (contaminated synthetic data).
      # The if-blocks below switch conflicting options off:
      # external_baryon_suppression turns off use_baryon_pca and
      # add_baryons_on_dv; create_baryon_pca turns off
      # external_baryon_suppression and use_baryon_pca; add_baryons_on_dv
      # turns off external_baryon_suppression and may run with use_baryon_pca.
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    if self.use_baryon_pca:
      # npcs = number of sampled PC amplitudes (LSST_BARYON_Q1 to Q4). The
      # PCA file holds one row per data-vector entry and one column per PC;
      # cosmolike uses its first npcs columns.
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Return the theory quantities Cobaya must compute at every point.

    Cobaya calls get_requirements once, after initialize, and later hands the
    requested quantities to the likelihood through self.provider
    (get_param, get_Pk_interpolator, get_comoving_radial_distance, ...).
    Each key of the returned dictionary names a quantity; its value is None
    (a parameter, or a quantity without options) or a dictionary of options,
    such as the redshifts at which to tabulate it. Cobaya routes each request
    to the theory code that provides it (CAMB, an emulator, the FAST-PT or
    baryon theory blocks).

    What each use_emulator value requests:
      1: the emulated blocks of the probe (cosmic_shear, ggl, wtheta); with
         gamma_t also H0 and the comoving distance chi(z) [Mpc] on
         z_interp_1D, which the point-mass term needs;
      2: As, H0, omegam, omegab, mnu, the linear and nonlinear total-matter
         P(k, z) at the z_interp_2D_camb redshifts up to k_max [1/Mpc], and
         chi(z) [Mpc] on z_interp_1D;
      0: as 2, plus w, wa, CAMB's omnuh2, the linear P_cb (the delta_nonu
         pair), the baryon suppression when external_baryon_suppression is
         on, and the "Cl" entry. That entry asks for the CMB temperature C_ell
         up to ell = 0, which carries no data: it makes Cobaya's CAMB wrapper
         switch on CAMB's CMB calculation (WantCls) and keep the yaml's
         set_for_lmax arguments, which the wrapper otherwise drops.
    In modes 0 and 2 the Python FAST-PT tables (IA_PS, bias_PS) are requested
    when IA_code = 1.

    Returns:
      dict {quantity name: None or dict of options}; None (no return
      statement reached) when use_emulator = 1 and the probe is unknown.
    """
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
      }
      # IA_code = 1: the Python FAST-PT theory block supplies the one-loop
      # IA and galaxy-bias tables
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 (non_linear_emul = 1) takes these as inputs
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # mnu gives Omega_nu h^2 on this path (get_neutrino_inputs)
      _requirements_["mnu"] = None
      return _requirements_
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "omegab": None,
        "mnu": None,
        "w": None,
        "wa": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D 
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # The baryon theory block computes the suppression S(k, z) at exactly
      # these nodes: the z nodes of the cosmolike tables and k in 1/Mpc
      # (10^log10k_interp_2D); a block that works in h/Mpc must convert.
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # IA_code = 1: the Python FAST-PT theory block supplies the one-loop
      # IA and galaxy-bias tables
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      # Two spectra: total matter (delta_tot) and cold dark matter + baryons
      # (delta_nonu, CAMB's name for all matter except massive neutrinos).
      # set_cosmo_related sends the second to cosmolike at every point
      # (get_neutrino_inputs reads it), so code that calls cosmolike's halo
      # functions directly finds it even after an evaluation without halo
      # terms. CAMB computes both from one transfer-function run.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Send the cosmology of this point to cosmolike: P(k, z), growth, chi(z).

    Hot path: runs at every sampled point. In modes use_emulator = 0 and 2
    it builds and sends
      lnPL, lnPNL = ln of the linear and nonlinear total-matter P(k, z)
                    [(Mpc/h)^3] on the grid z_interp_2D x log10k_interp_2D,
                    each a 1D array of n_z*n_k values in Fortran order (z
                    index fast: entry iz + n_z*ik);
      lnPL_cb     = ln of the linear cold dark matter + baryon P(k, z), same
                    layout (get_neutrino_inputs);
      G_growth    = growth factor G(z) = D(z)(1+z), normalized to 1 at
                    z_interp_2D[-1], on z_growth (n_G values);
      chi         = comoving distance [Mpc/h] on z_interp_1D;
    with Omega_m, Omega_b, Omega_nu h^2 and H0 [km/s/Mpc]. CAMB and Cobaya
    work in 1/Mpc and Mpc^3: the - log10(h) and + ln(h^3) shifts convert to
    cosmolike's h/Mpc units. With IA_code = 1 it also sends the Python
    FAST-PT tables. Mode 1 (emulated data vector) sends only chi(z), which
    the point-mass term needs.

    legend: n_z = len(z_interp_2D), n_k = len(log10k_interp_2D),
            n_G = len(z_growth)

    Raises:
      an exception when non_linear_emul is neither 1 nor 2.

    Side effects: replaces cosmolike's cosmology tables.
    """
    h = self.provider.get_param("H0")/100.0
    if not (self.use_emulator == 1):
      # PKL = Cobaya's interpolator of the linear P(k, z) [Mpc^3, k in 1/Mpc],
      # extrapolated from 1e-6/Mpc up to 250*accuracyboost /Mpc beyond the
      # range CAMB computed. PKL.logP(z, k) returns ln P as an (n_z, n_k)
      # array; flatten(order='F') lays it out with the z index fast, the
      # order cosmolike reads.
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        # EuclidEmulator2 inputs, under its own parameter names (mnu in eV)
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # EuclidEmulator2 covers z <= 10 and 8.73e-3 <= k <= 9.41 h/Mpc
        # (log10 k from -2.0589 to 0.973); it is asked for the redshifts
        # z < 10 of the table and as many k as the table has. It returns
        # kbt [h/Mpc] and the boost B = P_nonlinear/P_linear, one row per z.
        kbt, tmp_bt = ee2.get_boost2(params,
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # ln B is interpolated linearly in log10 k (along axis 1, the k axis)
        # onto the table's k [h/Mpc], extrapolated linearly above 9.41 h/Mpc;
        # below 8.73e-3 h/Mpc ln B is set to 0 (B = 1 on linear scales).
        # lnbt = ln B on the full (n_z, n_k) table, 0 at z >= 10.
        tmp = interp1d(np.log10(kbt),
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # CAMB's nonlinear P(k) (the halofit_version of the CAMB theory
        # block) first fills every redshift ...
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # ... then, below z = 10, ln P_nonlinear = ln P_linear + ln B. The
        # reshapes view the flat Fortran-order arrays as (n_z, n_k) tables;
        # np.where picks whole rows, because the (n_z, 1) boolean column
        # repeats across k; ravel(order='F') flattens the result back.
        lnPNL = np.where((self.z_interp_2D<10)[:,None],
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        # CAMB's nonlinear P(k) at every redshift
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)

      # z_growth = the dense 1D z grid up to the last P(k) node: the
      # redshifts of the growth table. Cosmolike reads G linearly in z: on
      # the coarse 2D grid (dz ~ 0.03) that linear read misses D by up to
      # 9e-5 and the growth rate f = 1 - (1+z) dlnG/dz (the slope of the
      # table) by 1%; on the 1D grid (dz = 0.003) by 1e-6 and 0.2%. PKL is a
      # cubic spline in z through CAMB's transfer redshifts, so the dense
      # grid asks CAMB for no extra redshifts (about 0.1 ms per evaluation).
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # growth_k = the wavenumber [1/Mpc] at which D(z) is read (yaml key
      # growth_k, default 0.05/Mpc), a sub-horizon scale. Near the horizon,
      # at k = 5e-4/Mpc (about 2 H0/c), CAMB's dark-energy perturbations
      # change the growth by 0.5-0.9% when w != -1 (z = 0.5 to 2), while
      # every user of G (IA amplitudes, one-loop D^4 terms, sigma(M, z), the
      # growth rate f) describes sub-horizon modes; with 0.06 eV neutrinos
      # the growth varies by 0.03% above 0.05/Mpc. The measurements are in
      # cosmolike_core/.claude/skills/cosmolike-dev/references/
      # growth_factor_measurements.md.
      growth_k = float(getattr(self, "growth_k", 0.05))
      # G(z) = D(z)(1+z)/D(0), with D(z)/D(0) = sqrt(P(z, k)/P(0, k)) at
      # k = growth_k; the second line divides by the same quantity at
      # z_norm = z_interp_2D[-1], so G = 1 there (z_growth ends just below
      # it). Cosmolike's growfac divides by G(0), so this normalization
      # cancels and D(z = 0) = 1.
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)
      # external_baryon_suppression: the theory block returns a dictionary
      # {z: S(k) array}, with S = P_baryons/P_dark-matter-only on the k nodes
      # requested in get_requirements (the block handles k and z outside its
      # calibrated range itself). ln S is added to ln P_nonlinear at each z
      # node: lnPNL[i::n_z] takes every n_z-th entry from i, which in
      # Fortran order is every k at z index i. The z lookup compares floats
      # exactly. A z missing from the dictionary is skipped with a warning;
      # any exception skips the whole suppression with an error message, and
      # the evaluation continues without baryonic feedback.
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA_code = 1: send the one-loop tables of the Python FAST-PT theory
      # block, z = 0 spectra with k in h/Mpc. FPTIA has 12 rows (10 IA
      # spectra, the k grid in row 10 = FPTIA[-2], the linear P in the last)
      # and FPTbias 8 rows; N = number of k columns. This comes after
      # ci.set_cosmology, which replaces cosmolike's cosmology and draws a
      # new cosmology.random (the number cosmolike's caches compare to
      # detect a changed cosmology), so the tables belong to the new one.
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
    else:
      ci.set_distances(
        z=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of massive neutrinos today, part of omegam.
    Halo variances use the cold dark matter + baryon spectrum P_cb at
    each redshift. Their mass-radius relation and mass-function density
    use rho_crit (Omega_m - Omega_nu). Total matter remains available
    for lensing and for the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k,z) grid and in the same units as
    lnPL. Both spectra are provided so that code calling cosmolike's halo
    functions directly (notebook wrappers, tests) finds P_cb even after a
    likelihood evaluation that computed no halo terms.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708 [mnu in eV], the
        neutrino density the EXAMPLE_EMUL2 yaml files subtract from
        omegam h^2 when they derive omegach2, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra at wavenumbers far above the neutrino
        free-streaming wavenumber, where the neutrinos do not cluster; an
        approximation on cluster scales. Its measured size is in
        projects/des_cluster/README.md.

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 where the neutrinos no longer
      # cluster (delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    """Send the source-sample nuisance parameters of this point to cosmolike.

    For each slot i = 1 to source_ntomo, params.get(name, 0) returns the
    sampled value of the parameter called name, or 0 (no systematic) when
    the run does not sample it:
      LSST_M<i>    = multiplicative shear calibration m of source bin i;
      LSST_DZ_S<i> = photo-z shift of the source n(z) of bin i;
      LSST_A1_<i>, LSST_A2_<i>, LSST_BTA_<i> = intrinsic-alignment
                     parameters (tidal alignment, tidal torquing, density
                     weighting). The meaning of slot i follows
                     IA_redshift_evolution: with 2 (binning) it is source bin
                     i; with 3 (power law) slot 1 is the amplitude and slot 2
                     the redshift exponent (cosmolike's IA.c).
    Each argument list is a nested list comprehension: the inner brackets
    build the names [LSST_M1, LSST_M2, ...], the outer ones read their values
    in that order. With use_emulator = 1 only the shear calibration is sent.

    Arguments:
      **params = the sampled parameters, as keyword arguments (name = value);
                 the ** collects them into the dictionary params

    Side effects: changes cosmolike's nuisance parameters; with
    external_nz_modeling also replaces the source n(z).
    """
    ntomo = self.source_ntomo
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling:
        # external_nz_modeling: the source n(z) is sent at every point, so a
        # user function of the nuisance parameters can change it first (for
        # example to add outlier populations). Modify a copy: self.source_nz
        # must keep the n(z) read from the file for the next point. The
        # commented line below marks where such a function goes.
        source_nz_local = self.source_nz.copy()

        #source_nz_local = f(source_nz_local, nuisance parameters)

        ci.set_source_sample(source_nz_local)

        # the photo-z shifts LSST_DZ_S<i> are applied on top of the n(z) above
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    """Send the lens-sample nuisance parameters of this point to cosmolike.

    For each lens bin i = 1 to lens_ntomo, params.get(name, default) returns
    the sampled value of the parameter called name, or the default when the
    run does not sample it:
      LSST_PM<i>   = point mass of lens bin i, the amplitude of a gamma_t
                     term that absorbs the unmodeled mass enclosed within
                     the smallest scales (sent in all modes);
      LSST_B1_<i>  = linear bias b1 (default 1, an unbiased tracer);
      LSST_B2_<i>, LSST_B3NL_<i>, LSST_BK_<i> = quadratic, third-order and
                     nonlocal bias (default 0);
      LSST_BMAG_<i> = magnification-bias amplitude (default 0);
      LSST_DZ_L<i> = photo-z shift of the lens n(z) of bin i (default 0).
    The nested list comprehensions build the names [LSST_B1_1, LSST_B1_2,
    ...] and read their values in that order (set_source_related explains
    them). With use_emulator = 1 only the point masses are sent.

    Arguments:
      **params = the sampled parameters, as keyword arguments (name = value)

    Side effects: changes cosmolike's nuisance parameters; with
    external_nz_modeling also replaces the lens n(z).
    """
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling:
        # external_nz_modeling: the lens n(z) is sent at every point, so a
        # user function of the nuisance parameters can change it first (for
        # example to add outlier populations). Modify a copy: self.lens_nz
        # must keep the n(z) read from the file for the next point. The
        # commented line below marks where such a function goes.
        lens_nz_local = self.lens_nz.copy()

        #lens_nz_local = f(lens_nz_local, nuisance parameters)

        ci.set_lens_sample(lens_nz_local)

        # the photo-z shifts LSST_DZ_L<i> are applied on top of the n(z) above
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    """Return ln L = -chi2/2 of a theory data vector.

    ci.compute_chi2 forms delta = theory - data on the unmasked entries only
    and returns chi2 = delta^T C^-1 delta, with the data vector, the
    covariance C and the mask that initialize loaded (ci.init_data_real).

    Arguments:
      datavector = theory data vector, 1D float64 array of the full length
                   (1560 for the LSST-Y1 binning, masked entries included)

    Returns:
      float, the log-likelihood up to an additive constant.
    """
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params):
    """Return the log-likelihood of one sampled point (Cobaya calls this).

    Cobaya passes the values of the parameters this likelihood declares
    (the nuisance parameters of params_source.yaml and params_lens.yaml);
    the cosmology arrives separately, through self.provider.

    Arguments:
      **params = parameter values as keyword arguments (name = value)

    Returns:
      float, ln L = -chi2/2.
    """
    return self.compute_logp(self.get_datavector(**params))

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):
    """Compute the theory data vector of one sampled point.

    Chooses the emulated data vector (use_emulator = 1) or cosmolike's own
    calculation (use_emulator = 0 or 2).

    Arguments:
      **params = parameter values as keyword arguments (name = value)

    Returns:
      1D float64 numpy array of the full data-vector length; masked entries
      are 0.
    """
    if self.use_emulator == 1:
      dv = self.internal_get_datavector_emulator(**params)
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector_emulator(self, **params):
    """Assemble the theory data vector from emulated blocks (use_emulator = 1).

    The emulator theory blocks return the xi_+- (get_cosmic_shear), gamma_t
    (get_ggl) and w(theta) (get_wtheta) blocks without the shear
    calibration and point-mass terms. They are copied into a zero vector of
    the full length at their block offsets:

      dv = [ xi_+- (sizes[0]) | gamma_t (sizes[1]) | w (sizes[2]) ]

    Blocks outside the probe stay 0. Cosmolike then adds the point-mass term
    to gamma_t (skipped when every LSST_PM<i> is 0), multiplies by the shear
    calibration factors (1 + m), adds the baryon PCs when use_baryon_pca is
    on, and sets the masked entries to 0.

    Arguments:
      **params = parameter values as keyword arguments (name = value)

    Returns:
      1D float64 numpy array of the full data-vector length.

    Raises:
      ValueError when an emulated block does not have the length cosmolike
      expects, or when the probe is unknown.

    Side effects: with print_datavector, overwrites print_datavector_file
    with two columns (entry index, value).
    """
    # ---------------------------------------------------------------
    # The shear calibration m and the point masses PM are never emulated:
    # cosmolike applies them below. The point-mass term needs the lens
    # nuisance parameters and the distances (set_cosmo_related sends only
    # chi(z) in this mode), so they are sent only when gamma_t is in the
    # probe and some PM is nonzero.
    PM = [params.get(p,0) for p in [survey+"_PM"+str(i+1) for i in range(self.lens_ntomo)]]
    if self.probe not in ("xi", "xi_gg") and not all(v == 0 for v in PM):
      self.set_lens_related(**params)
      self.set_cosmo_related()
    self.set_source_related(**params)
    # ---------------------------------------------------------------

    # sizes = [length of the xi_+- block, of gamma_t, of w], from cosmolike
    sizes = ci.compute_data_vector_3x2pt_real_sizes()
    total_size = int(np.sum(sizes))
    dv = np.zeros(total_size, dtype='float64') 
    
    if self.probe == "xi":
      tmp = self.provider.get_cosmic_shear()
      if (len(tmp) != sizes[0]):
        raise ValueError(f'Incompatible Sizes (Emulator Cosmic Shear)')
      dv[0:sizes[0]] = tmp[0:sizes[0]]
    elif self.probe == "xi_ggl":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1]):
        raise ValueError(f'Incompatible Sizes (Emulator xi_ggl)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
    elif self.probe == "3x2pt":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "xi_gg":
      tmp1 = self.provider.get_cosmic_shear()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "2x2pt": 
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    else:
      raise ValueError(f'Unknown probe')

    # force_exclude_pm = 1 skips the point-mass term when every PM is 0
    if not self.use_baryon_pca:
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=1)
    else:
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=1)
    dv = np.array(dv, dtype='float64')
    
    if self.print_datavector:
      size = len(dv)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = dv
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return dv

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Compute the theory data vector with cosmolike (use_emulator = 0 or 2).

    Sends the cosmology and the nuisance parameters (cosmic shear alone
    needs no lens parameters), then asks cosmolike for the data vector; the
    masked entries come back as 0. With use_baryon_pca the sampled PC
    amplitudes LSST_BARYON_Q<i> add sum_i Q_i PC_i.

    Arguments:
      **params = parameter values as keyword arguments (name = value)

    Returns:
      the theory data vector, a list of floats of the full length
      (get_datavector converts it to a numpy array).

    Side effects: with create_baryon_pca, computes the baryon PCs and
    overwrites filename_baryon_pca at every call; with print_datavector,
    overwrites print_datavector_file with two columns (entry index, value).
    """
    self.set_cosmo_related()
    if self.probe != "xi":
        self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else:  
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
