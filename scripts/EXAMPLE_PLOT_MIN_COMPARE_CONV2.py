#!/usr/bin/env python3

"""Plots how the minimum found by EXAMPLE_EMUL_MINIMIZE2.py converges.

The annealed minimizer of EXAMPLE_EMUL_MINIMIZE2.py (example 2, Planck 2018
CMB + DESI DR2 BAO + DES Y5 supernovae + LSST-Y1 cosmic shear) runs n_STW
steps per walker at each temperature (its --nstw option). Longer runs cost
more evaluations and get closer to the true minimum of
chi2 = -2 log posterior; this plot shows where the minimum stops improving.
It reads 30 minimizer outputs,
$ROOTDIR/projects/lsst_y1/chains/EXAMPLE_EMUL_MIN2_TEST_CONV<i>.txt for
i = 0 to 29 (ROOTDIR = the Cocoa/ folder, set by start_cocoa.sh), and
assumes run i used --nstw 100 + 25 i (100 to 825). The curve is
|chi2_min(n_STW) - chi2_min(825)|, the distance from the minimum of the
longest run, for runs 1 to 28, on a log scale. The README shows the figure
in its Global Minimizer section.

Output: EXAMPLE_PLOT_MIN2_COMPARE_CONV.pdf in the same folder. Run from any
folder with Cocoa activated.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

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

# rt = common prefix of the minimizer outputs: run i wrote rt<i>.txt
rt = os.environ['ROOTDIR']+"/projects/lsst_y1/chains/EXAMPLE_EMUL_MIN2_TEST_CONV"
plt.figure(figsize=(8, 5))
# Style lists for several curves; this plot draws one (color 0, marker and
# line style 5). A tuple (offset, (on, off, ...)) is a dash pattern: an
# offset, then alternating dash and gap lengths.
colors = ['royalblue','lightcoral','black', 'purple']
markers = ['o', 's', '^', 'v', 'D', '*', 'x', 'P', '<', '>']
linestyles = ['solid',
              '-', 
              '--', 
              '-.', 
              ':', 
              (0,(3,1,1,1)), 
              (0,(5,2)), 
              (0,(1,1)), 
              (0,(3,5,1,5)), 
              (0,(1,10)), 
              (0,(5,1))]
# sz = number of runs. data[i] = [n_STW, chi2_min] of run i, assuming run i
# used --nstw 100 + 25 i. A minimizer output holds one row, which np.loadtxt
# returns as a 1D array; [-1] is its last column, the total chi2.
sz=30
data = np.array([[100+25*i,np.loadtxt(f"{rt}{i}.txt")[-1]] for i in range(sz)])
# Distance of each minimum from that of the last run (the largest n_STW,
# the reference), for runs 1 to sz-2; run 0 is left out.
plt.plot(data[1:sz-1,0], 
         abs(data[1:sz-1,1]-data[-1,1]), 
         marker=markers[5],
         linestyle=linestyles[5],
         color=colors[0], 
         label="$\\Lambda$CDM, Planck (l<396) + DES-SN + BAO + LSST-Y1 Cosmic Shear")

# ax = the axes plt.plot drew on: a faint dashed grid on the minor ticks, and
# a log y axis from 1e-5 to 10
ax = plt.gca()
ax.grid(True)
ax.grid(True, 
        which='minor', 
        color='grey', 
        linestyle='--', 
        linewidth=0.25, 
        alpha=0.1)
ax.minorticks_on()
ax.tick_params(axis='both', which='major',labelsize=15)
ax.tick_params(axis='both', which='minor',labelsize=15)
plt.yscale('log')
plt.ylim(1e-5, 10)
# Axis labels: n_STW = steps per walker per temperature (the --nstw option)
plt.xlabel("$n_{\\rm STW}$")
plt.ylabel("$\\Delta \\chi_{\\rm min}^2$")
ax.legend(fontsize=13, frameon=False)
plt.savefig(os.environ['ROOTDIR']+
            "/projects/lsst_y1/chains/EXAMPLE_PLOT_MIN2_COMPARE_CONV.pdf", 
            bbox_inches='tight')