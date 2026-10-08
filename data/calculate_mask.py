"""Writes the six LSST-Y1 scale-cut masks LSST_Y1_M<n>_GGLOLAP0.05.mask.

A mask has one entry per data-vector element: 1 keeps the element in the
chi2, 0 removes it (a scale cut). With 5 source bins, 5 lens bins and 26
angular bins the data vector has 1560 elements, in the order xi_+ (15
source pairs x 26 angles), xi_- (15 x 26), gamma_t (25 lens-source pairs x
26) and w(theta) (5 lens bins x 26). The masks differ in the cuts:
  M1 to M5: xi_+ and xi_- keep angles above theta_min = j/l_max, where j is
      the first zero of the Bessel function in the Hankel transform of xi_+
      (J0, j = 2.4048) or of xi_- (J4, j = 7.5883), for l_max = 3000, 1500,
      750, 375 and 187.5; gamma_t and w keep angles above 21 Mpc/h divided
      by the angular-diameter distance to the mean redshift of the lens bin;
  M6: no angular cut.
In every mask gamma_t also drops each lens-source pair whose lensing
efficiency is at or below ggl_efficiency_cut = 0.05 (the GGLOLAP0.05 of the
file name); the efficiency is low when the source bin overlaps with or lies
in front of the lens bin.

Run from any folder: python calculate_mask.py. The six files are written
to the current folder, two columns each (element index, 0/1); the
project's data folder ships them as lsst_y1_M<n>_GGLOLAP0.05.mask.
"""
import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
import math as mt

# gamma_t lens-source pairs with lensing efficiency at or below this value
# are removed (see ggl_efficiency below)
ggl_efficiency_cut = [0.05]

# Scale-cut choices of each mask ----------------------------------------------
# xi_+ and xi_- relate to the shear power spectrum C_l through Bessel
# functions: xi_+(theta) uses J0(l theta), xi_-(theta) uses J4(l theta).
# Removing the multipoles above l_max corresponds to cutting angles below
# the first zero of the Bessel function: l_max x theta_min = 2.4048 (first
# zero of J0) for xi_+ and 7.5883 (first zero of J4) for xi_-. Example: for
# l_max = 3000, theta_min = 2.4048/3000 rad = 2.756 arcmin. Each mask halves
# l_max, so the cutoffs double from M1 to M5.
for Year in [1]:
  for mask_choice in [1,2,3,4,5,6]:
    if (mask_choice == 1):
      # LSST_Y1_M1.mask (l_max = 3000) -------------------------------------
      ξp_CUTOFF = 2.756  # cutoff scale in arcminutes
      ξm_CUTOFF = 8.6955 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 2):
      # LSST_Y1_M2.mask (l_max = 1500) -------------------------------------
      ξp_CUTOFF = 5.512  # cutoff scale in arcminutes
      ξm_CUTOFF = 17.391 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 3):
      # LSST_Y1_M3.mask (l_max = 750) --------------------------------------
      ξp_CUTOFF = 11.024  # cutoff scale in arcminutes
      ξm_CUTOFF = 34.782 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 4):
      # LSST_Y1_M4.mask (l_max = 375) --------------------------------------
      ξp_CUTOFF = 22.048 # cutoff scale in arcminutes
      ξm_CUTOFF = 69.564 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 5):
      # LSST_Y1_M5.mask (l_max = 187.5) ------------------------------------
      ξp_CUTOFF = 44.096  # cutoff scale in arcminutes
      ξm_CUTOFF = 139.128 # cutoff scale in arcminutes
      gc_CUTOFF = 21      # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 6):
      # LSST_Y1_M6.mask: no angular cut; gamma_t still drops the
      # low-efficiency lens-source pairs, so not every entry is 1 ----------
      ξp_CUTOFF = 0 # cutoff scale in arcminutes
      ξm_CUTOFF = 0 # cutoff scale in arcminutes
      gc_CUTOFF = 0 # Galaxy clustering cutoff in Mpc/h
    # gc_CUTOFF = 21 Mpc/h is close to 2 pi/k_max = 20.9 Mpc/h for
    # k_max = 0.3 h/Mpc; ang_cut below turns it into an angle for gamma_t and
    # w(theta).
    # End of the scale-cut choices -------------------------------------------

    # Binning of the LSST-Y1 data vector (the .dataset files) ----------------
    THETA_MIN  = 2.5    # Minimum angular scale (in arcminutes)
    THETA_MAX  = 900.  # Maximum angular scale (in arcminutes)
    N_ANG_BINS = 26    # Number of angular bins
    N_LENS = 5  # Number of lens tomographic bins
    N_SRC  = 5  # Number of source tomographic bins
    # N_XI_PS = distinct source-bin pairs (i <= j): 5*6/2 = 15; N_XI = length
    # of one xi block, 15*26 = 390 (not used below)
    N_XI_PS = int(N_SRC * (N_SRC + 1) / 2)
    N_XI    = int(N_XI_PS * N_ANG_BINS)
    

    # theta[i] = the angle [arcmin] assigned to angular bin i: the bins are
    # log-spaced between THETA_MIN and THETA_MAX, and each bin is represented
    # by its area-weighted mean radius (2/3)(tmax^3 - tmin^3)/(tmax^2 - tmin^2),
    # the convention of cosmolike's angular bins. 2.90888208665721580e-4 is
    # one arcminute in radians. theta has one spare final entry; the masks
    # use theta[:-1], the first N_ANG_BINS values.
    vtmin = THETA_MIN * 2.90888208665721580e-4;
    vtmax = THETA_MAX * 2.90888208665721580e-4;
    logdt = (mt.log(vtmax) - mt.log(vtmin))/N_ANG_BINS;
    theta = np.zeros(N_ANG_BINS+1)

    for i in range(N_ANG_BINS):
      tmin = mt.exp(mt.log(vtmin) + (i + 0.0) * logdt);
      tmax = mt.exp(mt.log(vtmin) + (i + 1.0) * logdt);
      x = 2./ 3.
      theta[i] = x * (tmax**3 - tmin**3) / (tmax**2- tmin**2)
      theta[i] = theta[i]/2.90888208665721580e-4

    # Flat LambdaCDM with Omega_m = 0.3; H0 = 100 km/s/Mpc makes astropy's
    # distances in Mpc equal to distances in Mpc/h.
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    def ang_cut(z):
      """Return the angular scale cut [arcmin] of gamma_t and w at redshift z.

      theta_min = gc_CUTOFF/D_A(z), with D_A the angular-diameter distance
      [Mpc/h] of cosmo; D_A = chi/(1+z), where chi is the comoving distance.
      gc_CUTOFF and cosmo are read from the enclosing loop when the function
      runs (Python looks names up at call time), so each mask uses its own
      gc_CUTOFF.

      Arguments:
        z = redshift (the mean redshift of a lens bin)

      Returns:
        float, the cut in arcminutes.
      """
      theta_rad = gc_CUTOFF / cosmo.angular_diameter_distance(z).value
      return theta_rad * 180. / np.pi * 60.

    # zavg = mean redshift of each of the five Y1 lens bins
    if (Year == 1):
      zavg = [0.3269670081723307,
              0.5086885453137051,
              0.6699437575466684,
              0.848472949839094,
              1.0712458524571165]

    # Cosmic-shear masks ------------------------------------------------------
    # (theta[:-1] > cutoff) is a boolean array over the 26 angles; the list
    # comprehension repeats it once per source pair (N_XI_PS = 15) and
    # np.hstack joins the copies into one 1D array of 15*26 entries: the same
    # angular cut for every source pair.
    ξp_mask = np.hstack([(theta[:-1] > ξp_CUTOFF) for i in range(N_XI_PS)])
    ξm_mask = np.hstack([(theta[:-1] > ξm_CUTOFF) for i in range(N_XI_PS)])

    # Galaxy-galaxy lensing mask ----------------------------------------------
    # ggl_efficiency[i][j] = lensing efficiency of source bin j for the lenses
    # of bin i (dimensionless, between 0 and 1). The pairs are visited lens
    # bin first, source bin fastest, the gamma_t order of the data vector; a
    # pair at or below ggl_efficiency_cut is removed at every angle.
    if (Year == 1):
      ggl_efficiency = [
        [0.4456315654,0.8790767396,0.9389947229,0.8354926862,0.631370374],
        [0.0334525378,0.3739779295,0.8338207849,0.9821921200,0.8639203163],
        [0.0004551936,0.0536064628,0.4178650771,0.8715050829,0.9711497877],
        [0.0000006072,0.0015727852,0.0818817759,0.5363472954,0.9782246343],
        [0.0000000000,0.0000024903,0.0025740396,0.1465300465,0.8300740215]
      ]

    γt_mask = [] 
    if (Year == 1):
      for i in range(N_LENS): 
        for j in range(N_SRC):
          if ggl_efficiency[i][j] > ggl_efficiency_cut[0]:
            γt_mask.append((theta[:-1] > ang_cut(zavg[i])))
          else:
            γt_mask.append(np.zeros(N_ANG_BINS))
    γt_mask = np.hstack(γt_mask) 

    # Galaxy clustering mask: one block of 26 angles per lens bin ------------
    w_mask = np.hstack([(theta[:-1] > ang_cut(zavg[i])) for i in range(N_LENS)])

    # Output: the four blocks in data-vector order, written as two columns
    # (element index, 0.0 or 1.0) to the current folder ----------------------
    mask = np.hstack([ξp_mask, ξm_mask, γt_mask, w_mask])
    if (Year == 1):
      np.savetxt("LSST_Y" + str(Year) + "_M" + str(mask_choice) +
        "_GGLOLAP" + str(ggl_efficiency_cut[0]) + ".mask", 
        np.column_stack((np.arange(0,len(mask)), mask)),
        fmt='%d %1.1f')

