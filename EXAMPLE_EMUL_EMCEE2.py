"""Samples the CMB + BAO + SN + LSST-Y1 shear posterior with emcee (emulated theory).

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

emcee is an ensemble sampler: a set of walkers (3 per sampled
parameter, or one per MPI process when there are more processes) moves
through parameter space, and each proposed move is built from the
positions of other walkers (differential-evolution moves, 80%, and
snooker moves, 20%). The walkers start at points drawn from the
reference distributions of the yaml. After the run, the first 5
autocorrelation times are discarded as burn-in and the chain is
thinned by half the shortest autocorrelation time.

Run from the Cocoa/ folder with Cocoa activated (start_cocoa.sh), under
MPI, for example (the project README lists the full commands):

    mpirun -n 5 python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE2.py --maxfeval 1000000 \\
        --root ./projects/lsst_y1/ --outroot EXAMPLE_EMUL_EMCEE2

Options: --maxfeval = total likelihood evaluations, shared among the
walkers; --root = folder whose chains/ subfolder receives the output;
--outroot = output name; --progress = show emcee's progress bar.
Output in <root>chains/, in getdist's format: <outroot>.1.txt (weight,
log posterior, parameters, chi2*), <outroot>.ranges (prior bounds),
<outroot>.paramnames, <outroot>.covmat (the chain covariance), and the
emcee checkpoint <outroot>.h5.
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
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
import numpy as np
from emcee.autocorr import AutocorrError
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from getdist import IniFile
from getdist import loadMCSamples
from schwimmbad import MPIPool
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_EMUL_EMCEE')

parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=5000)
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/example/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="test.dat")
parser.add_argument("--progress",
                    dest="progress",
                    help="Show Emcee Progress",
                    nargs='?',
                    type=bool,
                    default=False)
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
  planck_2018_lowl.TT:
    path: ./external_modules
  # choose only one low ell EE likelihood
  #planck_2018_lowl.EE:
  #  path: ./external_modules
  planck_2018_lowl.EE_sroll2: null 
  #planck_2020_lollipop.lowlE:
  #  data_folder: planck/lollipop
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
    latex: \log(10^{10} A_\mathrm{s})
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
  emulbaosn:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
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
    res1 = model.logprior(point, make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1.e20
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=False)
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)
def chi2v2(p):
    """Return the -2 log posterior split into its parts at one point.

    Arguments:
      p = sampled-parameter values, as in chi2.

    Returns:
      1D array: -2 log likelihood of each likelihood of the model, in the
      order of the yaml likelihood block, followed by -2 log prior.
    """
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    point = dict(zip(model.parameterization.sampled_params(), p))
    logposterior = model.logposterior(point, as_dict=True)
    chi2likes=-2*np.array(list(logposterior["loglikes"].values()))
    chi2prior=-2*np.atleast_1d(model.logprior(point,make_finite=False))
    return np.concatenate((chi2likes, chi2prior))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def chain(x0,
          ndim,
          nwalkers,
          cov,
          names,
          maxfeval=3000, 
          pool=None,
          checkpoint=None):  
    """Run emcee and return the thinned, burned-in chain.

    Arguments:
      x0       = starting points [nwalkers, ndim].
      ndim     = number of sampled parameters.
      nwalkers = number of walkers.
      cov      = parameter covariance (not used by the moves here).
      names    = sampled-parameter names.
      maxfeval = steps per walker.
      pool     = MPI pool that evaluates the walkers, or None.
      checkpoint = HDF5 file where emcee stores the chain as it runs.

    Returns:
      [rows, tau]: rows [n, ndim + 3] = weight 1, log posterior, the
      parameters and chi2 = -2 log posterior; tau = the integrated
      autocorrelation time of each parameter, in steps.
    """
  
    def logprob(params, *args):
        """Return the log probability -chi2/2 emcee samples (-inf if forbidden)."""
        res = chi2(params)
        if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
          return -np.inf
        else:
          return -0.5*res

    backend = emcee.backends.HDFBackend(checkpoint)

    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                    ndim=ndim, 
                                    log_prob_fn=logprob, 
                                    parameter_names=names,
                                    moves=[(emcee.moves.DEMove(), 0.8),
                                           (emcee.moves.DESnookerMove(), 0.2)],
                                    pool=pool,
                                    backend=backend)
    sampler.run_mcmc(x0, 
                     maxfeval, 
                     skip_initial_state_check=True, 
                     progress=args.progress)
    
    tau = sampler.get_autocorr_time(quiet=True, has_walkers=True)
    print(f"Partial Result: tau = {tau}, nwalkers={nwalkers}")

    burn_in = int(5*np.max(tau))
    thin    = int(0.5*np.min(tau))
    xf      = sampler.get_chain(flat=True, discard=burn_in, thin=thin)
    lnpf    = sampler.get_log_prob(flat=True, discard=burn_in, thin=thin)
    weights = np.ones((len(xf),1), dtype='float64')
    local_chi2 = -2*lnpf
    
    return [np.concatenate([weights,
                           lnpf[:,None], 
                           xf, 
                           local_chi2[:,None]], axis=1), 
            tau]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The block below runs only when this file is executed as a script.
# MPIPool (schwimmbad) makes MPI rank 0 the master, which runs the block;
# every other rank waits in pool.wait(), evaluating the walkers the master
# sends, until the master closes the pool, and then exits.
if __name__ == '__main__':
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        # dim = number of sampled parameters; bounds = the 99.9999% prior
        # ranges; nwalkers = 3 per parameter, at least one per MPI process;
        # maxevals = steps per walker
        dim      = model.prior.d()                                      # Cobaya call
        bounds   = model.prior.bounds(confidence=0.999999)              # Cobaya call
        names    = list(model.parameterization.sampled_params().keys()) # Cobaya Call
        nwalkers = max(3*dim,pool.comm.Get_size())
        maxevals = int(args.maxfeval/(nwalkers))
        print(f"\n\n\n"
              f"maxfeval={args.maxfeval}, "
              f"nwalkers={nwalkers}, "
              f"maxfeval per walker = {maxevals}"
              f"\n\n\n")
        # get initial points: one draw per walker from the yaml ref -----------
        x0 = [] # Initial point x0
        for j in range(nwalkers):
          (tmp_x0, tmp) = model.get_valid_point(max_tries=10000, 
                                                ignore_fixed_ref=False,
                                                logposterior_as_dict=True)
          x0.append(tmp_x0[0:dim])
        x0 = np.array(x0, dtype='float64')
        
        # get the covariance of the prior -------------------------------------
        cov = model.prior.covmat(ignore_external=True) # cov from prior
        
        # the emcee checkpoint: an HDF5 file that stores the chain as it runs
        checkpoint = f"{args.root}chains/{args.outroot}.h5"

        # run the chains (chain() returns [rows, autocorrelation times]) ------
        res = chain(x0=np.array(x0, dtype='float64'),
                    ndim=dim,
                    nwalkers=nwalkers,
                    cov=cov, 
                    names=names,
                    maxfeval=maxevals,
                    pool=pool,
                    checkpoint=checkpoint)

        # save the chain in getdist's text format: <outroot>.1.txt -----------
        os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
        hd=f"nwalkers={nwalkers}, maxfeval={args.maxfeval}, max tau={res[1]}\n"
        np.savetxt(f"{args.root}chains/{args.outroot}.1.txt",
                   res[0],
                   fmt="%.7e",
                   header=hd + ' '.join(names),
                   comments="# ")
        # save the .ranges file (prior bounds, read by getdist) --------------
        hd = ["weights","lnp"] + names + ["chi2*"]
        rows = [(str(n),float(l),float(h)) for n,l,h in zip(names, bounds[:,0], bounds[:,1])]
        with open(f"{args.root}chains/{args.outroot}.ranges", "w") as f: 
          f.write(f"# {' '.join(hd)}\n")
          f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

        # save the .paramnames file (name and LaTeX label per column) --------
        param_info = model.info()['params']
        latex  = [param_info[x]['latex'] for x in names]
        names.append("chi2*")
        latex.append("\\chi^2")
        np.savetxt(f"{args.root}chains/{args.outroot}.paramnames", 
                   np.column_stack((names,latex)),
                   fmt="%s")
    
        # save the chain covariance (.covmat), read back with getdist --------
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