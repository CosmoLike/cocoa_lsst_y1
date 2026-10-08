"""Compares two LSST-Y1 3x2pt chains (old developer script).

Draws one triangle plot of two Cobaya Metropolis-Hastings chains:
  EXAMPLE_MCMC4  legend "LSST-Y1 3x2pt EE2";
  EXAMPLE_MCMC2  legend "LSST-Y1 3x2pt".
3x2pt = cosmic shear, galaxy-galaxy lensing and galaxy clustering
(lsst_y1.combo_3x2pt).
"EE2" in a legend means the Euclid Emulator 2 nonlinear power spectrum
(non_linear_emul: 1 in the yaml).

The chains are read from ../chains/ relative to the folder the script runs
from, so run it from a folder next to chains/ (for example scripts/). The
first 50% of each chain is removed as burn-in. Output, in the folder the
script runs from: the hidden getdist files .VM_P2_TMP1 and .VM_P2_TMP2 (the
chains with derived columns added) and the figure plot2.pdf (g.export()
with no file name uses the name of the script).
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

# parameter = the chain columns to plot (Cobaya names, and SS8 added below);
# chaindir = the folder the script runs from
parameter = [u'omegam', u'sigma8', u'As_1e9', u'ns', u'SS8', u'omegab', u'H0', u'w', u'LSST_A1_1', u'LSST_A1_2']
chaindir=os.getcwd()

# getdist analysis settings. analysissettings removes the first 50% of each
# chain as burn-in (ignore_rows = 0.5) when the chain is loaded;
# analysissettings2 (ignore_rows = 0) is for the plotter, which reads the
# saved files, whose burn-in is already removed. smooth_scale_1D and
# smooth_scale_2D = width of the Gaussian smoothing kernel, in standard
# deviations of each parameter; range_confidence = tail probability that
# sets the plotted range of each parameter.
analysissettings={'smooth_scale_1D':0.35, 'smooth_scale_2D':0.3,'ignore_rows': u'0.5',
'range_confidence' : u'0.005'}

analysissettings2={'smooth_scale_1D':0.35,'smooth_scale_2D':0.3,'ignore_rows': u'0.0',
'range_confidence' : u'0.005'}

# root_chains = the two chain names, in the order of the legend labels
root_chains = (
  'EXAMPLE_MCMC4',
  'EXAMPLE_MCMC2',
)


# --------------------------------------------------------------------------------
# Each block loads one chain and adds derived columns (p.<name> is the array
# of column <name>, one entry per sample): gamma = Omega_m h; SS8 = S_8 =
# sigma_8 (Omega_m/0.3)^0.5, from s8omegamp5 = sigma_8 Omega_m^0.5 and
# 0.5477225575 = 0.3^0.5; om10, ob100 and ns10 = 10 Omega_m, 100 Omega_b
# and 10 n_s. Only SS8 is plotted. saveAsText writes the chain as hidden
# getdist text files (names starting with a dot) in the current folder.
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[0],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.addDerived(10*p.omegam,name='om10',label='{10 \\Omega_m}')
samples.addDerived(100*p.omegab,name='ob100',label='{100 \\Omega_b}')
samples.addDerived(10*p.ns,name='ns10',label='{10 n_s}')
samples.saveAsText(chaindir + '/.VM_P2_TMP1')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[1],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.addDerived(10*p.omegam,name='om10',label='{10 \\Omega_m}')
samples.addDerived(100*p.omegab,name='ob100',label='{100 \\Omega_b}')
samples.addDerived(10*p.ns,name='ns10',label='{10 n_s}')
samples.saveAsText(chaindir + '/.VM_P2_TMP2')
# --------------------------------------------------------------------------------


# getdist plotter: it reads the hidden files with analysissettings2;
# width_inch = figure width in inches; g.settings sets fonts, line widths,
# the rotation of the x tick labels and the legend style.
g=gplot.getSubplotPlotter(chain_dir=chaindir,
  analysis_settings=analysissettings2,width_inch=12.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 13.0
g.settings.legend_fontsize = 13.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

print(chaindir)

# triangle_plot draws the 1D marginalized posterior of each parameter on the
# diagonal and the 68% and 95% contours of each pair below it; the style
# lists hold one entry per chain, in the order of the roots, and extra
# entries are not used. param_3d = None: no third parameter shown as
# colored points.
param_3d = None
g.triangle_plot([chaindir + '/.VM_P2_TMP1',chaindir + '/.VM_P2_TMP2'],
parameter,
plot_3d_with_param=param_3d,line_args=[
{'lw': 1.2,'ls': 'solid', 'color':'lightcoral'},
{'lw': 1.2,'ls': '--', 'color':'black'},
{'lw': 1.6,'ls': '-.', 'color': 'maroon'},
{'lw': 1.6,'ls': 'solid', 'color': 'indigo'},
],
contour_colors=['lightcoral','black','maroon','indigo'],
contour_ls=['solid','--','-.'], 
contour_lws=[1.0,1.5,1.5,1.0],
filled=[True,False,False,True],
shaded=False,
legend_labels=[
'LSST-Y1 3x2pt EE2',
'LSST-Y1 3x2pt',
],
legend_loc=(0.48, 0.80))


g.export()