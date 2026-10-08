#!/usr/bin/env python3

"""Plots profiles of seven parameters of example 2 (CMB, BAO, SN, LSST-Y1).

Example 2 combines the Planck 2018 CMB (l < 396), DESI DR2 BAO, DES Y5
supernova and LSST-Y1 cosmic-shear likelihoods. A profile fixes one
parameter at each value of a grid and minimizes -2 log posterior over all
the others; EXAMPLE_EMUL_PROFILE2.py computes it with the emulated theory.
For logA = log(10^10 A_s), ns, omegabh2 = Omega_b h^2, omegach2 =
Omega_c h^2, thetastar = 100 theta_* (the angular size of the sound horizon
at last scattering), LSST_A1_1 and LSST_A1_2 (the intrinsic-alignment
amplitude and its redshift exponent), this script reads that output,
$ROOTDIR/projects/lsst_y1/chains/EXAMPLE_EMUL_PROFILE2.<name>.txt
(ROOTDIR = the Cocoa/ folder, set by start_cocoa.sh; column 0 = the fixed
value, column 1 = the minimum chi2 there). Each panel shows
Delta chi2 = chi2 - min(chi2) against the value, a least-squares parabola
through the points, and the values where the parabola crosses
Delta chi2 = 1, 4 and 9 (vertical lines and the text box): for one
parameter with a Gaussian profile these bound the 68.3%, 95.4% and 99.7%
confidence intervals (1, 2 and 3 sigma).

Each input comes from one EXAMPLE_EMUL_PROFILE2.py run with --outroot
EXAMPLE_EMUL_PROFILE2 and --profile = the position of <name> among the
sampled parameters. Output: EXAMPLE_PLOT_PROFILE2.pdf in the same folder;
the README shows the figure in its Profile section. Run from any folder
with Cocoa activated.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(15.1, 12.1))
# ------------------------------------------------------------------------------
# Layout: a master grid of two rows (the second 1.1 times taller); the top
# row holds four panels and the bottom row three, ax[0] to ax[6], one per
# parameter. hspace and wspace = gaps between rows and between panels, as
# fractions of the mean panel height and width.
master = gridspec.GridSpec(2, 
                           1, 
                           height_ratios=[1,1.1], 
                           hspace=0.225) # Master grid
top = gridspec.GridSpecFromSubplotSpec(1, 
                                       4,
                                       subplot_spec=master[0],
                                       wspace=0.275)
ax = [fig.add_subplot(top[0,i]) for i in range(4)]
bottom = gridspec.GridSpecFromSubplotSpec(1,
                                          3,
                                          subplot_spec=master[1],
                                          wspace=0.35)
ax += [fig.add_subplot(bottom[0,i]) for i in range(3)]
# ------------------------------------------------------------------------------
# root = common prefix of the input files; params = the profiled parameters
# (file suffixes); latex = their axis labels, in the same order
root = os.environ['ROOTDIR'] + "/projects/lsst_y1/chains/EXAMPLE_EMUL_PROFILE2."
params = ['logA', 'ns', 'omegabh2', 'omegach2', 'thetastar', 'LSST_A1_1', 'LSST_A1_2' ]
latex  = ["$\\log(10^{10} A_\\mathrm{s})$", "$n_\\mathrm{s}$", 
          "$100\\Omega_\\mathrm{b} h^2$", "$10\\Omega_\\mathrm{c} h^2$", 
          "$100\\theta_*$", "$A_\\mathrm{1-IA,LSST}^1$", "$A_\\mathrm{1-IA,LSST}^2$" ]
# ------------------------------------------------------------------------------
for i in range(7):
    data = np.loadtxt(root + params[i] + '.txt', comments="#",)
    # Panels 2 and 3 show 100 x omegabh2 and 10 x omegach2, as their axis
    # labels state; the parabola and the interval text use the scaled values
    if i == 2:
        data[:,0] = 100*data[:,0]
    if i == 3:
        data[:,0] = 10*data[:,0]
    # x = the fixed values; y = Delta chi2, the minimum chi2 at each value
    # minus the smallest of them
    x  = data[:, 0]
    y  = data[:, 1]-np.min(data[:,1])

    ax[i].plot(x, y, 
               marker='D',c='black', linestyle='None', markersize=4,
               alpha=1.0,lw=1.0,
               label=params[i])
    
    # Least-squares parabola y = a x^2 + b x + c through the points
    # (coeffs = [a, b, c]), drawn on 300 points
    coeffs = np.polyfit(x, y, deg=2)
    xfit = np.linspace(np.min(x), np.max(x), 300)
    yfit = np.polyval(coeffs, xfit)
    ax[i].plot(xfit, yfit, color='blue', lw=1.5, alpha=0.7, label='Parabola fit')

    # Axes style: a faint dashed grid on the minor ticks; axis labels (the y
    # label on the first panel of each row); y from 0 to the largest Delta
    # chi2; x over the scanned range plus 7.5% margins
    ax[i].grid(True)
    ax[i].grid(True, 
               which='minor', 
               color='black',
               linestyle='--', 
               linewidth=0.25, 
               alpha=0.1)
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', 
                      which='major', 
                      labelsize=15)
    ax[i].tick_params(axis='both', 
                      which='minor', 
                      labelsize=15)
    ax[i].set_xlabel(latex[i],fontsize = 19)
    if i == 0 or i==4:
        ax[i].set_ylabel('$\\Delta \\chi^2$',fontsize = 19)
    ax[i].set_ylim(np.min(y),np.max(y))
    ax[i].set_xlim(data[0,0]-0.075*(data[-1,0]-data[0,0]),
                   x[-1]+0.075*(x[-1]-x[0]))

    # Intervals: for y0 = 1, 4 and 9 (1, 2 and 3 sigma) the two real roots of
    # a x^2 + b x + c = y0 are the ends of the interval, drawn as vertical
    # lines; a level without two real roots is skipped
    sigma_lines = {}
    for y0 in [1, 4, 9]:
        a, b, c = coeffs
        roots = np.roots([a, b, c - y0])
        real_roots = [np.real(r) for r in roots if np.isreal(r)]
        if len(real_roots) == 2:
            sigma_lines[y0] = sorted(real_roots)
            for r in real_roots:
                ax[i].axvline(x=r, 
                              linestyle='--', 
                              color='grey', 
                              alpha=0.5, 
                              lw=1.0)
    # Text box listing the intervals; prec = decimals shown: 1 for a scanned
    # range between 1 and 10, 2 for one between 0.1 and 1, and so on
    lmap = {1: "1σ", 4: "2σ", 9: "3σ"}
    tlines = []
    prec = max(0, int(-math.floor(math.log10(x[-1] - x[0]))) + 1)
    for y0 in [1, 4, 9]:
        if y0 in sigma_lines:
            lo, hi = sigma_lines[y0]
            tlines.append(f"{lmap[y0]}: [{lo:.{prec}f}, {hi:.{prec}f}]")    
    ax[i].text(
        0.5, 
        0.90, 
        "\n".join(tlines),
        transform=ax[i].transAxes,
        fontsize=11,
        va='top', ha='center',
        bbox=dict(boxstyle="round", 
                  facecolor='white', 
                  alpha=0.8, 
                  edgecolor='black'))
# ------------------------------------------------------------------------------
plt.subplots_adjust(bottom=0.25, left = 0.2)
# The file name has no extension, so savefig appends .pdf (savefig.format)
plt.savefig(os.environ['ROOTDIR'] + "/projects/lsst_y1/chains/EXAMPLE_PLOT_PROFILE2")