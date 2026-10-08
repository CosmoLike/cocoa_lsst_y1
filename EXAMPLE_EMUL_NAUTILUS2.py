"""Samples the CMB + BAO + SN + LSST-Y1 shear posterior with Nautilus (emulated theory).

The likelihoods are Planck 2018 CMB (high-l TTTEEE, low-l TT and EE),
DESI DR2 BAO, DES-Y5 supernovae and LSST-Y1 cosmic shear. Emulators
replace every Boltzmann-code product: theta* (emultheta), the drag-epoch
sound horizon (emulrdrag), the CMB spectra (emulcmb), the BAO and SN
distances (emulbaosn) and the 780 xi_+ and xi_- entries of the
cosmic-shear data vector (emul_cosmic_shear).

In these examples the cosmolike likelihood runs with use_emulator: 1:
it adds only the shear calibration, the point masses and the mask to the
emulated data vector (likelihood/_cosmolike_prototype_base.py). The
complete Cobaya configuration is the yaml_string below. Every chi2 in
this script is -2 (log prior + log likelihood), the -2 log posterior.

Nautilus is a nested sampler: a set of live points shrinks from the
prior volume toward high likelihood, estimating the Bayesian evidence Z
and weighted posterior samples; neural networks learn the likelihood
boundary to propose new points. Its prior here is uniform inside
Cobaya's 99.9999% prior bounds of every sampled parameter; the
Gaussian priors of the nuisance parameters enter through chi2, which
includes the log prior.

Run from the Cocoa/ folder with Cocoa activated (start_cocoa.sh), under
MPI, for example (the project README lists the full commands):

    mpirun -n 5 python -m mpi4py.futures \\
        ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS2.py \\
        --nlive 1000 --outroot EXAMPLE_EMUL_NAUTILUS2

Options: --nlive = live points; --nnetworks = neural networks;
--maxfeval = maximum likelihood evaluations; --neff = minimum effective
sample size; --flive = live-set evidence fraction that ends the
exploration; --root and --outroot = output folder and name. Output in
<root>chains/, in getdist's format: <outroot>.1.txt (weight, log
posterior, parameters, chi2*), .ranges, .paramnames, .covmat, and the
checkpoint <outroot>_checkpoint.hdf5 (a rerun resumes from it).
"""
import warnings
import os
from sklearn.exceptions import InconsistentVersionWarning
# Silence warnings that are expected here, so the sampler output stays
# readable: a scikit-learn version mismatch when the pickled emulator
# models load, a deprecation message of the sacc package, numpy
# invalid-value and overflow warnings at points far from the emulator
# training range (their chi2 becomes 1e20 below), and known UserWarnings
# matched by their message text.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*column is deprecated.*",
    module=r"sacc\.sacc"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*overflow encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Function not smooth or differentiabl*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Hartlap correction*"
)
import argparse, random
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from nautilus import Prior, Sampler
from getdist import loadMCSamples
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_PROJECT_NAUTILUS1')
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/lsst_y1/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="example_nautilus1")
parser.add_argument("--nlive",
                    dest="nlive",
                    help="Number of live points ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1000)
parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100000)
parser.add_argument("--neff",
                    dest="neff",
                    help="Minimum effective sample size. ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=10000)
parser.add_argument("--flive",
                    dest="flive",
                    help="Maximum fraction of the evidence contained in the live set before building the initial shells terminates",
                    type=float,
                    nargs='?',
                    const=1,
                    default=0.01)
parser.add_argument("--nnetworks",
                    dest="nnetworks",
                    help="Number of Neural Networks",
                    type=int,
                    nargs='?',
                    const=1,
                    default=4)
# parse_known_args (unlike parse_args) ignores options it does not define,
# such as those an MPI launcher adds, instead of stopping
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# yaml_string = the complete Cobaya configuration: likelihoods, sampled and
# derived parameters with their priors, and the emulator theory blocks
# (file = the trained network, extra = its normalization data, ord = the
# order of the network inputs, extrapar = the network architecture). The
# lines starting with # inside the string are YAML comments.
yaml_string=r"""
likelihood:
  planck_2018_highl_plik.TTTEEE:
    path: ./external_modules/
    clik_file: plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE_33_396.clik
  # choose only one low ell EE likelihood
  #planck_2018_lowl.EE:
  #  path: ./external_modules
  planck_2018_lowl.EE_sroll2: null 
  #planck_2020_lollipop.lowlE:
  #  data_folder: planck/lollipop
  planck_2018_lowl.TT:
    path: ./external_modules
  bao.desi_dr2.desi_bao_all:
    path: ./external_modules/data/
  sn.desy5: 
    path: ./external_modules/data/sn_data 
  lsst_y1.cosmic_shear:
    path: ./external_modules/data/lsst_y1
    data_file: lsst_y1_M1_GGL0.05.dataset
    use_emulator: 1
params:
  logA:
    prior:
      min: 1.61
      max: 3.91
    ref:
      dist: norm
      loc: 3.0448
      scale: 0.05
    proposal: 0.05
    latex: \log(10^{10} A_\mathrm{s}
  ns:
    prior:
      min: 0.92
      max: 1.05
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.005
    proposal: 0.005
    latex: n_\mathrm{s}
  thetastar:
    prior:
      min: 1
      max: 1.2
    ref:
      dist: norm
      loc: 1.04109
      scale: 0.0004
    proposal: 0.0002
    latex: 100\theta_\mathrm{*}
    renames: theta
  omegabh2:
    prior:
      min: 0.01
      max: 0.04
    ref:
      dist: norm
      loc: 0.022383
      scale: 0.005
    proposal: 0.005
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    prior:
      min: 0.06
      max: 0.2
    ref:
      dist: norm
      loc: 0.12011
      scale: 0.03
    proposal: 0.03
    latex: \Omega_\mathrm{c} h^2
  tau:
    prior:
      min: 0.04
      max: 0.1
    ref:
      dist: norm
      loc: 0.055
      scale: 0.006
    proposal: 0.003
    latex: \tau_\mathrm{reio}
  As:
    derived: 'lambda logA: 1e-10*np.exp(logA)'
    latex: A_\mathrm{s}
  A:
    derived: 'lambda As: 1e9*As'
    latex: 10^9 A_\mathrm{s}
  mnu:
    value: 0.06
  w0pwa:
    value: -1.0
    latex: w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}
    drop: true
  w:
    value: -1.0
    latex: w_{0,\mathrm{DE}}
  wa:
    value: 'lambda w0pwa, w: w0pwa - w'
    derived: false
    latex: w_{a,\mathrm{DE}}
  H0:
    derived: true
    latex: H_0
  omegamh2:
    derived: true
    value: 'lambda omegach2, omegabh2, mnu: omegach2+omegabh2+(mnu*(3.046/3)**0.75)/94.0708'
    latex: \Omega_\mathrm{m} h^2
  omegam:
    derived: true
    latex: \Omega_\mathrm{m}
  rdrag:
    derived: true
    latex: r_\mathrm{drag}
  # WL photo-z errors
  LSST_DZ_S1:
    prior:
      dist: norm
      loc: 0.0414632
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0414632
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^1
  LSST_DZ_S2:
    prior:
      dist: norm
      loc: 0.00147332
      scale: 0.002
    ref:
      dist: norm
      loc: 0.00147332
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^2
  LSST_DZ_S3:
    prior:
      dist: norm
      loc: 0.0237035
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0237035
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^3
  LSST_DZ_S4:
    prior:
      dist: norm
      loc: -0.0773436
      scale: 0.002
    ref:
      dist: norm
      loc: -0.0773436
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^4
  LSST_DZ_S5:
    prior:
      dist: norm
      loc: -8.67127e-05
      scale: 0.002
    ref:
      dist: norm
      loc: -8.67127e-05
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^5
  # Intrinsic alignment
  LSST_A1_1:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: 0.7
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{1-IA,LSST}^1
  LSST_A1_2:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: -1.7
      scale: 0.5
    proposal: 0.5
  # Shear calibration parameters
  LSST_M1:
    prior:
      dist: norm
      loc: 0.0191832
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0191832
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^1
  LSST_M2:
    prior:
      dist: norm
      loc: -0.0431752
      scale: 0.005
    ref:
      dist: norm
      loc: -0.0431752
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^2
  LSST_M3:
    prior:
      dist: norm
      loc: -0.034961
      scale: 0.005
    ref:
      dist: norm
      loc: -0.034961
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^3
  LSST_M4:
    prior:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    ref:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^4
  LSST_M5:
    prior:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    ref:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^5
theory:
  emultheta:
    path: ./cobaya/cobaya/theories/
    provides: ['H0', 'omegam']
    extra_args:
      file: ['external_modules/data/emultrf/CMB_TRF/emul_lcdm_thetaH0_GP.joblib']
      extra: ['external_modules/data/emultrf/CMB_TRF/extra_lcdm_thetaH0.npy']
      ord: [['omegabh2','omegach2','thetastar']]
      extrapar: [{'MLA' : "GP"}]
  emulrdrag:
    path: ./cobaya/cobaya/theories/
    provides: ['rdrag']
    extra_args:
      file: ['external_modules/data/emultrf/BAO_SN_RES/emul_lcdm_rdrag_GP.joblib'] 
      extra: ['external_modules/data/emultrf/BAO_SN_RES/extra_lcdm_rdrag.npy'] 
      ord: [['omegabh2','omegach2']]
  emulcmb:
    path: ./cobaya/cobaya/theories/
    extra_args:
      # This version of the emul was not trained with CosmoRec
      eval: [True, True, True, False] #TT,TE,EE,PHIPHI
      device: "cuda"
      ord: [['omegabh2','omegach2','H0','tau','ns','logA','mnu','w','wa'],
            ['omegabh2','omegach2','H0','tau','ns','logA','mnu','w','wa'],
            ['omegabh2','omegach2','H0','tau','ns','logA','mnu','w','wa'],
            None]
      file: ['external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBTT_CNN.pt',
             'external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBTE_CNN.pt',
             'external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBEE_CNN.pt', 
             None]
      extra: ['external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBTT_CNN.npy',
              'external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBTE_CNN.npy',
              'external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBEE_CNN.npy', 
              None]
      extrapar: [{'ellmax' : 5000, 'MLA': 'CNN', 'INTDIM': 4, 'INTCNN': 5120},
                 {'ellmax' : 5000, 'MLA': 'CNN', 'INTDIM': 4, 'INTCNN': 5120},
                 {'ellmax' : 5000, 'MLA': 'CNN', 'INTDIM': 4, 'INTCNN': 5120}, 
                 None]
      #file: ['external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBTT_TRF.pt',
      #       'external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBTE_TRF.pt',
      #       'external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBEE_TRF.pt',
      #       None] 
      #extra: ['external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBTT_TRF.npy',
      #        'external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBTE_TRF.npy',
      #        'external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBEE_TRF.npy',
      #        None]
      #extrapar: [{'ellmax' : 5000, 'MLA': 'TRF', 'NCTRF': 16, 'INTDIM': 4, 'INTTRF': 5120},
      #           {'ellmax' : 5000, 'MLA': 'TRF', 'NCTRF': 16, 'INTDIM': 4, 'INTTRF': 5120},
      #           {'ellmax' : 5000, 'MLA': 'TRF', 'NCTRF': 16, 'INTDIM': 4, 'INTTRF': 5120},
      #           None]
  emulbaosn:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    provides: ['comoving_radial_distance', 'angular_diameter_distance', 'Hubble']
    extra_args:
      device: "cuda"
      file:  [None, 'external_modules/data/emultrf/BAO_SN_RES/emul_lcdm_H.pt']
      extra: [None, 'external_modules/data/emultrf/BAO_SN_RES/extra_lcdm_H.npy']    
      ord: [None, ['omegam','H0']]
      extrapar: [{'MLA': 'INT', 'ZMIN' : 0.0001, 'ZMAX' : 3, 'NZ' : 600},
                 {'MLA': 'ResMLP', 'offset' : 0.0, 'INTDIM' : 1, 'NLAYER' : 1,
                  'TMAT': 'external_modules/data/emultrf/BAO_SN_RES/PCA_lcdm_H.npy',
                  'ZLIN': 'external_modules/data/emultrf/BAO_SN_RES/z_lin_lcdm.npy'}]
  emul_cosmic_shear:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args: 
      device: 'cuda'
      file:  ['projects/lsst_y1/emulators/lcdm_nla_halofit_cosmic_shear_trf/transformer.emul']
      extra: ['projects/lsst_y1/emulators/lcdm_nla_halofit_cosmic_shear_trf/transformer.h5']
      ord:   [['logA','ns','H0','omegabh2','omegach2',
               'LSST_DZ_S1','LSST_DZ_S2','LSST_DZ_S3','LSST_DZ_S4','LSST_DZ_S5',
               'LSST_A1_1','LSST_A1_2']]
      extrapar: [{'MLA': 'TRF', 'INT_DIM_RES': 256, 
                  'INT_DIM_TRF': 1024, 'NC_TRF': 32, 'OUTPUT_DIM': 780}]
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# model = the Cobaya model of the configuration: model.logprior,
# model.loglike and model.logposterior evaluate one point
model = get_model(yaml_load(yaml_string))
def chi2(p):
    """Return -2 (log prior + log likelihood) at one parameter point.

    Arguments:
      p = sampled-parameter values in Cobaya's sampled order (a list or
          array), or a {name: value} dictionary.

    Returns:
      float; 1e20 when the prior or the likelihood is infinite or NaN
      (outside the prior, or an evaluation that failed), so samplers and
      minimizers treat the point as forbidden.

    Raises:
      ValueError when a parameter value is infinite or NaN.
    """
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    if np.any(np.isinf(p)) or  np.any(np.isnan(p)):
      raise ValueError(f"At least one parameter value was infinite (CoCoa) param = {p}")
    point = dict(zip(model.parameterization.sampled_params(), p))
    res1 = model.logprior(point,make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1.e20
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=False)
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)

def likelihood(params):
  """Return the log posterior that Nautilus samples, -chi2/2.

  Arguments:
    params = sampled-parameter values in Cobaya's sampled order.

  Returns:
    float, or -inf for a forbidden point (chi2 above 1e19, infinite or
    NaN).
  """
  res = chi2(params)
  if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
    return -np.inf
  else:
    return -0.5*res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# MPIPoolExecutor: with the script started as python -m mpi4py.futures,
# the other MPI ranks evaluate the likelihood calls Nautilus submits
from mpi4py.futures import MPIPoolExecutor

# The block below runs only when this file is executed as a script
if __name__ == '__main__':
    print(f"nlive={args.nlive}, output={args.root}chains/{args.outroot}")
    # Build the Nautilus prior from Cobaya's: uniform in the 99.9999% prior
    # range of every sampled parameter
    NautilusPrior = Prior()                                       # Nautilus Call 
    dim    = model.prior.d()                                      # Cobaya call
    bounds = model.prior.bounds(confidence=0.999999)              # Cobaya call
    names  = list(model.parameterization.sampled_params().keys()) # Cobaya Call
    for b, name in zip(bounds, names):
      NautilusPrior.add_parameter(name, dist=(b[0], b[1]))
    
    sampler = Sampler(NautilusPrior, 
                      likelihood,  
                      filepath=f"{args.root}chains/{args.outroot}_checkpoint.hdf5", 
                      n_dim=dim,
                      pool=MPIPoolExecutor(),
                      n_live=args.nlive,
                      n_networks=args.nnetworks,
                      resume=True)
    sampler.run(f_live=args.flive,
                n_eff=args.neff,
                n_like_max=args.maxfeval,
                verbose=True,
                discard_exploration=True)
    points, log_w, log_l = sampler.posterior()
    
    # Save the weighted samples in getdist's text format: <outroot>.1.txt --
    os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
    np.savetxt(f"{args.root}chains/{args.outroot}.1.txt",
               np.column_stack((np.exp(log_w), log_l, points, -2*log_l)),
               fmt="%.5e",
               header=f"nlive={args.nlive}, maxfeval={args.maxfeval}, log-Z ={sampler.log_z}\n"+' '.join(names),
               comments="# ")
    
    # Save the .ranges file (prior bounds, read by getdist) -----------------
    rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bounds[:,0],bounds[:,1])]
    with open(f"{args.root}chains/{args.outroot}.ranges", "w") as f: 
      f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

    # Save the .paramnames file (name and LaTeX label per column) -----------
    param_info = model.info()['params']
    latex  = [param_info[x]['latex'] for x in names]
    names.append("chi2*")
    latex.append("\\chi^2")
    np.savetxt(f"{args.root}chains/{args.outroot}.paramnames", 
               np.column_stack((names,latex)),
               fmt="%s")

    # Save the sample covariance (.covmat), read back with getdist ----------
    samples = loadMCSamples(f"{args.root}chains/{args.outroot}",
                            settings={'ignore_rows': u'0.0'})
    np.savetxt(f"{args.root}chains/{args.outroot}.covmat",
               np.array(samples.cov(), dtype='float64'),
               fmt="%.5e",
               header=' '.join(names),
               comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------