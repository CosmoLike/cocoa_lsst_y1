"""Plots the parameter distribution of a chain used for emulator training.

An emulator of the data vector learns from cosmolike data vectors computed
at a sample of parameter points. This script draws the triangle plot (the
1D and 2D marginalized distributions) of such a sample, stored as a getdist
chain, to show the region of parameter space it covers: As_1e9 = 10^9 A_s,
ns, H0, omegam, omegab, the photo-z shifts LSST_DZ_S1 to LSST_DZ_S5 of the
n(z) of the five source bins, and the intrinsic-alignment parameters
LSST_A1_1 and LSST_A1_2.

Input: the chain w0wa_takahashi_params_train_cs_10 in
$ROOTDIR/projects/lsst_y1/chains/ (ROOTDIR = the Cocoa/ folder, set by
start_cocoa.sh); no script of this project writes it, and the legend text
records how it was made. Output: plot_dv_generation_for_ML_training.pdf in
the same folder. Run from any folder with Cocoa activated.
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

# parameter = the chain columns to plot, by their Cobaya names; chaindir =
# the project's chains folder, from ROOTDIR
parameter = [u'As_1e9', u'ns', u'H0', u'omegam', u'omegab', 
             u'LSST_DZ_S1', u'LSST_DZ_S2', u'LSST_DZ_S3', 
             u'LSST_DZ_S4', u'LSST_DZ_S5', 
             u'LSST_A1_1', u'LSST_A1_2']
chaindir  = os.environ['ROOTDIR'] + "/projects/lsst_y1/chains/"

# getdist analysis settings:
#   ignore_rows = fraction of the chain removed from its start as burn-in;
#   smooth_scale_1D, smooth_scale_2D = width of the Gaussian kernel that
#       smooths the 1D and 2D densities, in standard deviations of each
#       parameter;
#   range_confidence = tail probability that sets the plotted range of each
#       parameter;
#   fine_bins_2D = points per axis of the 2D density grid (getdist's key for
#       the 1D grid is fine_bins).
analysissettings={'smooth_scale_1D':0.25, 
                  'smooth_scale_2D':0.25,
                  'ignore_rows': u'0.3',
                  'range_confidence' : u'0.005',
                  'fine_bins_2D': 1024,
                  'fine_bins_1D': 1024}

# root_chains = chain names (file prefixes) in chaindir. A tuple of one
# element needs a trailing comma, ('name',): without it the parentheses only
# group, root_chains is the string itself, and root_chains[0] below is its
# first letter.
root_chains = (
  'w0wa_takahashi_params_train_cs_10'
)

# getdist plotter: chain_dir = the folder of the chains; analysis_settings
# apply to every chain it loads; width_inch = figure width in inches;
# g.settings sets fonts, line widths, the rotation of the x tick labels and
# the legend style.
g=gplot.getSubplotPlotter(chain_dir=chaindir,
                          analysis_settings=analysissettings,
                          width_inch=20.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour=1.0
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 15.0
g.settings.legend_fontsize = 16.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

# triangle_plot draws the 1D marginalized distribution of each parameter on
# the diagonal and the 68% and 95% contours of each pair below it. The style
# lists (line_args, contour_colors, contour_ls, contour_lws, filled) hold one
# entry per chain, in the order of roots; extra entries are not used.
g.triangle_plot(
  params=parameter,
  roots=[chaindir + root_chains[0]],
  plot_3d_with_param=None,
  line_args=[ {'lw': 1.0,'ls': 'solid', 'color': 'cornflowerblue'},
              {'lw': 1.0,'ls': 'solid', 'color': 'lightcoral'},
              {'lw': 1.2,'ls': '--', 'color': 'black'},
              {'lw': 2.1,'ls': 'dotted', 'color': 'maroon'},
              {'lw': 1.6,'ls': '-.', 'color': 'indigo'}
            ],
  contour_colors=['cornflowerblue', 'lightcoral','black','maroon', 'indigo'],
  contour_ls=['solid', 'solid','--','dotted','-.'], 
  contour_lws=[1.0,1.0,1.2,2.1,1.6],
  filled=[True,True,False,False,True],
  shaded=False,
  legend_labels=[
    'T=24, burn-in=0.3',
  ],
  legend_loc=(0.375, 0.8))

# ----------------------------------------------------
# ----------------------------------------------------
# axarr[row, column] = the panels of the triangle; all panels of a column
# share its x axis, so set_xlim on panel [2, 0] sets the range of the first
# parameter, 10^9 A_s, in the whole first column.
axarr = g.subplots
# ----------------------------------------------------
axarr[2,0].set_xlim([0.5,5.0])
# ----------------------------------------------------
# ----------------------------------------------------

g.export(os.path.join(chaindir,"plot_dv_generation_for_ML_training.pdf"))