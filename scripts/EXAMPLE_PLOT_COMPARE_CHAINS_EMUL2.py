"""Compares the posteriors of four samplers for CMB + BAO + SN + LSST-Y1.

Example 2 adds the Planck 2018 CMB likelihoods (l < 396; high-l TTTEEE,
low-l TT and EE), DESI DR2 BAO and DES Y5 supernovae to LSST-Y1 cosmic
shear (EXAMPLE_EMUL_MCMC2.yaml); every theory prediction (CMB spectra, BAO
and SN distances, the shear data vector) comes from an emulator. One
triangle plot compares four chains:
  EXAMPLE_EMUL_MCMC2      Cobaya Metropolis-Hastings (MH);
  EXAMPLE_EMUL_NAUTILUS2  Nautilus nested sampling (EXAMPLE_EMUL_NAUTILUS2.py);
  EXAMPLE_EMUL_EMCEE2     emcee ensemble sampling (EXAMPLE_EMUL_EMCEE2.py);
  EXAMPLE_EMUL_POLY2      PolyChord nested sampling (EXAMPLE_EMUL_POLY2.yaml).
The parameters are logA = log(10^10 A_s), ns, thetastar = 100 theta_*
(the angular size of the sound horizon at last scattering), omegabh2 and
omegach2 (Omega_b h^2 and Omega_c h^2), tau (the optical depth to
reionization), the intrinsic-alignment amplitude LSST_A1_1 and its
redshift exponent LSST_A1_2, and chi2v2 = -2 log posterior, built as in
EXAMPLE_PLOT_COMPARE_CHAINS_EMUL.py. The README shows the figure in its
Sampler Comparison section.

Input: the four chains in $ROOTDIR/projects/lsst_y1/chains/ (ROOTDIR = the
Cocoa/ folder, set by start_cocoa.sh). Output, in the same folder: the
hidden files .VM_P2_TMP2 to .VM_P2_TMP5 and
example_compare_chains_emul2.pdf. Run from any folder with Cocoa
activated.
"""
import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# General plot options: matplotlib.rcParams is matplotlib's table of default
# styles (fonts, ticks, grid, output format); the entries below apply to
# every figure of this script, saved as PDF with the white margins cropped.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

# parameter = the chain columns to plot, by their Cobaya names (chi2v2 is
# added below); chaindir = the project's chains folder, from ROOTDIR (the
# Cocoa/ folder, set by start_cocoa.sh)
parameter = [u'logA', u'ns', u'thetastar', u'omegabh2', u'omegach2', u'tau', 
             u'LSST_A1_1', u'LSST_A1_2', u'chi2v2']
chaindir  = os.environ['ROOTDIR'] + "/projects/lsst_y1/chains/"

# getdist analysis settings:
#   ignore_rows = fraction of each chain removed from its start as burn-in,
#       applied when the chain is loaded;
#   smooth_scale_1D, smooth_scale_2D = width of the Gaussian kernel that
#       smooths the 1D and 2D densities, in standard deviations of each
#       parameter;
#   range_confidence = tail probability that sets the plotted range of each
#       parameter;
#   fine_bins_2D = points per axis of the 2D density grid (getdist's key for
#       the 1D grid is fine_bins).
# analysissettings (ignore_rows = 0.3) is for the Metropolis-Hastings chains.
analysissettings={'smooth_scale_1D':0.25, 
                  'smooth_scale_2D':0.25,
                  'ignore_rows': u'0.3',
                  'range_confidence' : u'0.005',
                  'fine_bins_2D': 1024,
                  'fine_bins_1D': 1024}

# analysissettings2 (ignore_rows = 0) is for samples without burn-in: the
# nested samplers (Nautilus, PolyChord), the output of the emcee script
# (which removes burn-in itself), and the files saved below, whose burn-in
# was removed when their chain was loaded.
analysissettings2={'smooth_scale_1D':0.25,
                   'smooth_scale_2D':0.25,
                   'ignore_rows': u'0.0',
                   'range_confidence' : u'0.005',
                   'fine_bins_2D': 1024,
                   'fine_bins_1D': 1024}

# root_chains = chain names (file prefixes) in chaindir, in the order of the
# legend labels below
root_chains = (
  'EXAMPLE_EMUL_MCMC2',
  'EXAMPLE_EMUL_NAUTILUS2',
  'EXAMPLE_EMUL_EMCEE2',
  'EXAMPLE_EMUL_POLY2',
)

# --------------------------------------------------------------------------------
# Each block loads one chain (burn-in removed by its settings), adds the
# derived column chi2v2 = -2 log posterior (p.<name> is the array of column
# <name>, one entry per sample) and saves the chain as hidden getdist text
# files (names starting with a dot) in chaindir.
# EXAMPLE_EMUL_MCMC2, a Cobaya Metropolis-Hastings chain: chi2 = -2 log
# likelihood and minuslogprior = -log prior, so chi2v2 = chi2 + 2 minuslogprior
samples=loadMCSamples(chaindir + root_chains[0],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.chi2+2*p.minuslogprior,name='chi2v2',label='{\\chi^2_{\\rm post}}')
samples.saveAsText(chaindir + '/.VM_P2_TMP2')
# --------------------------------------------------------------------------------
# EXAMPLE_EMUL_NAUTILUS2, written by EXAMPLE_EMUL_NAUTILUS2.py: no burn-in, and
# its chi2 column already holds -2 log posterior
samples=loadMCSamples(chaindir+ root_chains[1], settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2,name='chi2v2',label='{\\chi^2_{\\rm post}}')
samples.saveAsText(chaindir + '/.VM_P2_TMP3')
# --------------------------------------------------------------------------------
# EXAMPLE_EMUL_EMCEE2, written by EXAMPLE_EMUL_EMCEE2.py: burn-in already
# removed, and its chi2 column already holds -2 log posterior
samples=loadMCSamples(chaindir+ root_chains[2],settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2, name='chi2v2', label='{\\chi^2_{\\rm post}}')
samples.saveAsText(chaindir + '/.VM_P2_TMP4')
# --------------------------------------------------------------------------------
# EXAMPLE_EMUL_POLY2, a Cobaya PolyChord chain (no burn-in): chi2v2 = chi2 + 2
# minuslogprior
samples=loadMCSamples(chaindir+ root_chains[3],settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2+2*p.minuslogprior,name='chi2v2',label='{\\chi^2_{\\rm post}}')
samples.saveAsText(chaindir + '/.VM_P2_TMP5')
# --------------------------------------------------------------------------------

# getdist plotter: it reads the hidden files with analysissettings2
# (ignore_rows = 0), since their burn-in is already removed; width_inch =
# figure width in inches; g.settings sets fonts, line widths, the rotation of
# the x tick labels and the legend style.
g=gplot.getSubplotPlotter(chain_dir=chaindir,
                          analysis_settings=analysissettings2,
                          width_inch=10.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour=1.0
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 15.0
g.settings.legend_fontsize = 16.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

# triangle_plot draws the 1D marginalized posterior of each parameter on the
# diagonal and the 68% and 95% contours of each pair below it. line_args
# (1D curves), contour_colors, contour_ls, contour_lws and filled hold one
# style per chain, in the order of roots; extra entries are not used. The
# legend labels are plain text written for these runs.
# In them, R-1 = the Gelman-Rubin convergence statistic of the MH chains (the
# Rminus1_stop and Rminus1_cl_stop limits of the yaml); n_live = live points
# and log(Z) = log evidence of the nested samplers; n_repeat = 3D =
# PolyChord's num_repeats, 3 times the number of parameters; n_eval =
# posterior evaluations.
g.triangle_plot(
  params=parameter,
  roots=[chaindir + '/.VM_P2_TMP2',
         chaindir + '/.VM_P2_TMP3',
         chaindir + '/.VM_P2_TMP4',
         chaindir + '/.VM_P2_TMP5'],
  plot_3d_with_param=None,
  line_args=[ 
              {'lw': 1.0,'ls': 'solid', 'color': 'lightcoral'},
              {'lw': 1.2,'ls': '--', 'color': 'black'},
              {'lw': 2.1,'ls': 'dotted', 'color': 'maroon'},
              {'lw': 1.6,'ls': '-.', 'color': 'indigo'}
            ],
  contour_colors=['lightcoral','black','maroon', 'indigo'],
  contour_ls=['solid', 'solid','--','dotted','-.'], 
  contour_lws=[1.4,0.7,2.1,1.6],
  filled=[True,False,False,True],
  shaded=False,
  legend_labels=[
    'MH, 4-walkers, $(R-1)_{\\rm median}$=0.02, $(R-1)_{\\rm std dev}$ = 0.2, burn-in=0.3, $n_{\\rm eval} \\sim 600,000$',
    'Nautilus, $n_{\\rm live}=3072$, $\\log(Z)=-1088.75$, $n_{\\rm eval} \\sim 436,000$',
    'EMCEE $n_{\\rm walkers}=114$, $n_{\\rm eval} \\sim 2,000,000$',
    'PolyChord $n_{\\rm live}=1024$, $n_{\\rm repeat}={\\rm 3D}$, $\\log(Z)=-1094.243 \\pm 0.22$',
  ],
  legend_loc=(0.375, 0.8))

g.export(os.path.join(chaindir,"example_compare_chains_emul2.pdf"))