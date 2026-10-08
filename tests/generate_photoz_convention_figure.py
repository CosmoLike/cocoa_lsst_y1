"""Regenerate the photo-z convention figures stored in tests/.

Evaluates the frozen cosmic-shear fiducial under the four runtime
photo-z settings (cspline/Z_LOW default, linear, Steffen, Z_MID; see
test_photoz_conventions.py for what they mean) and plots the
fractional data-vector differences against the default,

    delta xi/xi = xi(setting)/xi(default) - 1,

per tomographic pair and per angular bin, for xi_plus (solid) and
xi_minus (dashed). Masked angular bins are left out. Two figures,
because the two knobs live on different scales:

    photoz_zmid_dxi.png   - the Z_LOW vs Z_MID reading of the n(z)
                            file z column (percent level),
    photoz_interp_dxi.png - linear and Steffen vs cubic spline
                            (1e-4 level).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/lsst_y1/tests/generate_photoz_convention_figure.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process
# (4 threads, the count the frozen references were generated with)
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import shutil
import tempfile

# "Agg" is matplotlib's file-only backend: figures are written to PNG
# files without opening a window, so the script also runs on a machine
# without a display. It must be chosen before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# tests/ is not a package; put it first on the import path so
# cocoa_test_utils resolves from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# The frozen cosmic-shear configuration and its binning: NTOMO source
# bins, NTHETA log-spaced angular bins between THETA_MIN and THETA_MAX
# [arcmin], and the M1 scale-cut mask of the frozen contract.
EXAMPLE = "example1"
NTOMO = 5
NTHETA = 26
THETA_MIN, THETA_MAX = 2.5, 900.0
MASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "frozen", "data",
                         "lsst_y1_M1_GGLOLAP0.05.mask")

# (report tag, photoz_interpolation_type, photoz_zmid_convention); the
# first entry is the default every other setting is compared with
SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))


def datavectors():
    """Return the printed theory vector under each setting, keyed by tag.

    Builds one model per SETTINGS entry on the frozen fiducial, with
    print_datavector writing the theory vector into a temporary folder
    that is removed afterwards, also after an error.

    Returns:
      {tag: 1D float array of the full data-vector length}.
    """
    vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_fig_")
    out = {}
    try:
        for tag, interp, zmid in SETTINGS:
            print(f"building model ({EXAMPLE}, NLA, {tag}) ...", flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            block = info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
            block["photoz_interpolation_type"] = interp
            block["photoz_zmid_convention"] = zmid
            path = os.path.join(vectors_dir, f"dv_{interp}_{zmid}.modelvector")
            block["print_datavector"] = True
            block["print_datavector_file"] = path
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            u.evaluate_chi2(model, point)
            out[tag] = u._load_datavector(path)
    finally:
        shutil.rmtree(vectors_dir, ignore_errors=True)
    return out


def plot(curves, fname, title, scale=100.0, unit="%", ylim=None):
    """Draw one 3x5 grid of panels, one per source pair, and save it.

    The layout of the notebook plotters: 15 source pairs (i <= j),
    panels glued edge to edge, solid xi_+ and dashed xi_- curves.

    Arguments:
      curves = {label: (dxi_plus, dxi_minus)}, each a (npair, NTHETA)
               fractional-difference array with NaN at masked bins.
      fname  = PNG file name, written next to this script (tests/).
      title  = figure title.
      scale  = factor applied to the fractions before plotting (100
               for percent).
      unit   = text of the y-axis unit.
      ylim   = None, or the half-width of a fixed y range in plotted
               units.

    Returns:
      nothing; the PNG file is written and the figure closed.
    """
    # theta = the area-weighted centers of the log-spaced bins,
    # (2/3)(t_max^3 - t_min^3)/(t_max^2 - t_min^2) for each bin, the
    # convention of cosmolike's angular bins
    theta = np.geomspace(THETA_MIN, THETA_MAX, NTHETA + 1)
    theta = (2.0 / 3.0) * (theta[1:]**3 - theta[:-1]**3) \
                        / (theta[1:]**2 - theta[:-1]**2)
    # every source pair (i, j) with i <= j, in data-vector order
    pairs = [(i, j) for i in range(NTOMO) for j in range(i, NTOMO)]
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 9),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    cm = plt.get_cmap("gist_rainbow")
    # axes.ravel() lists the 15 panels row by row; panel p shows pair p.
    # Only panel 0 labels its curves, so the legend lists each once;
    # panels 10-14 form the bottom row (x labels), and p % 5 == 0 is
    # the first column (y labels).
    for p, (i, j) in enumerate(pairs):
        ax = axes.ravel()[p]
        for q, (label, (dp, dm)) in enumerate(curves.items()):
            color = cm(q / max(len(curves) - 1, 1) * 0.8)
            ax.semilogx(theta, scale * dp[p], color=color, lw=1.6,
                        label=rf"$\xi_+$: {label}" if p == 0 else None)
            ax.semilogx(theta, scale * dm[p], color=color, lw=1.6, ls="--",
                        label=rf"$\xi_-$: {label}" if p == 0 else None)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.text(0.08, 0.85, f"$({i+1},{j+1})$", transform=ax.transAxes,
                fontsize=15)
        if p >= 10:
            ax.set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
        if p % 5 == 0:
            ax.set_ylabel(rf"$\Delta\xi_\pm/\xi_\pm$ [{unit}]", fontsize=16)
    if ylim is not None:
        # fractional differences blow up where xi_plus crosses zero at
        # the largest angles; a fixed range keeps the flat offsets,
        # which are the physics, readable (the spikes run off-panel)
        axes.ravel()[0].set_ylim(-ylim, ylim)
    axes.ravel()[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=17)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


def main():
    """Evaluate the four settings and write the two figures.

    Returns:
      nothing; photoz_zmid_dxi.png and photoz_interp_dxi.png are
      written into tests/.

    Raises:
      RuntimeError when Cocoa is not activated; AssertionError when
      the frozen state does not match the manifest.
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = datavectors()

    # the mask file has two columns (entry index, 0/1); keep the second
    mask = np.loadtxt(MASK_FILE)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    # npair = 15 source pairs; nxi = entries of one xi block (xi_+ or
    # xi_-), 15 x 26 = 390
    npair = NTOMO * (NTOMO + 1) // 2
    nxi = npair * NTHETA

    def frac(tag):
        """Return [dxi_plus, dxi_minus] of one setting, each (npair, NTHETA)."""
        # first the xi_plus block, then xi_minus; fractional difference
        # against the default, NaN where the mask removes the point
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
        out = []
        for s in range(2):
            sl = slice(s * nxi, (s + 1) * nxi)
            # errstate silences numpy's divide-by-zero warnings inside
            # the with-block: masked entries are 0 in both vectors, and
            # np.where replaces their 0/0 by NaN
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(mask[sl] > 0, cur[sl] / ref[sl] - 1.0, np.nan)
            out.append(d.reshape(npair, NTHETA))
        return out

    plot({"Z_MID": frac("Z_MID")},
         "photoz_zmid_dxi.png",
         "lsst_y1 cosmic shear: n(z) z-column read as Z_MID instead of "
         "Z_LOW (frozen fiducial)", scale=100.0, unit="%", ylim=4.0)
    plot({"linear": frac("linear"), "steffen": frac("steffen")},
         "photoz_interp_dxi.png",
         "lsst_y1 cosmic shear: linear and Steffen n(z) interpolation "
         "vs cubic spline (frozen fiducial)", scale=1.0e4,
         unit=r"$10^{-4}$", ylim=8.0)


# __name__ is "__main__" only when this file runs directly as a
# script, not when it is imported
if __name__ == "__main__":
    main()
