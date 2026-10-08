"""Unit test: sector-wise cache invalidation (the parameter ladder).

cosmolike keeps the result of every expensive stage in memory (a
cache) and recomputes it only when an input it depends on changed.
Each group of parameters (a sector) has its own change marker, a key
the cached tables compare: cosmology (distances, growth,
power-spectrum tables), intrinsic alignment (FAST-PT tables under
TATT), photo-z shifts (n(z) splines, lens efficiencies), and the
shear calibrations (a pure data-vector rescale). Invalidation is the
step that marks a cached table stale when its sector changes. A
partial-invalidation bug (one sector's update path failing to rebuild
a cached C static variable that another sector reads) produces wrong
data vectors only in mixed update sequences, which the per-point
tests never exercise.

The test walks a deterministic ladder in one process, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point):

    3 x cosmology-only steps   (omegam, H0, As_1e9)
    3 x IA-only steps          (A1; under TATT also A2 and BTA)
    3 x source-photo-z steps   (every DZ_S shift)
    3 x lens-photo-z steps     (every DZ_L shift)
    3 x shear-calibration steps (every M)

(a sector the configuration samples no parameters in drops out of the
ladder: separate DZ_L shifts when the lenses are the source sample, IA
amplitudes where the configuration fixes them; this harness is shared
with projects where those cases occur)

It records the final data vector, then evaluates one scramble point
(every sector moved at once, galaxy bias included; the chi2 is
discarded) and returns to the ladder's final point: the pipeline must
reproduce the recorded vector bit for bit. A second model instance
walks the mirrored ladder (M -> DZ_L -> DZ_S -> IA -> cosmology) to
the same final point: the answer must depend on the point, never on
the invalidation history.

Assertions, in each intrinsic-alignment model (NLA and TATT; the
TATT ladder exercises the FAST-PT rebuild machinery NLA never
touches):
  1. every ladder step changes the data vector (a dead sector flag
     would pass the later checks vacuously);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative (cosmic shear by
     both bins' factors, gamma_t by the source factor, w by nothing);
  3. a no-op update (re-sending the current M values) leaves the
     vector bitwise unchanged;
  4. after the scramble, returning to the ladder's final point
     reproduces the recorded vector and chi2 bit for bit;
  5. the mirrored-order instance lands on the same final vector bit
     for bit.

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation, not
cobaya's memoization.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests/data_vector/test_cache_consistency.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# the 3x2pt configuration: every sector of the ladder enters its
# data vector
EXAMPLE = "example2"

# Sector membership by sampled-parameter name. re.compile builds a
# regular-expression matcher: ^(a|b)$ matches exactly the names a or b,
# "_A1_|_A2_" matches a name containing either text, and "_M[0-9]+$" a
# name ending in _M plus digits (LSST_M1). _sector_of returns the first
# sector whose pattern matches, so the order matters and "other" (the
# pattern "." matches any name) collects the rest. The bias and other
# sectors move only in the scramble step.
SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("other", re.compile(r".")),
)

# Ladder phases (sector order of the forward walk) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic.
# A key of DELTAS is either an exact parameter name (a string) or a
# compiled pattern matched against the names. The offsets are small
# moves inside the priors that still change the data vector visibly.
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}
# NSTEP = steps per sector on the ladder; SCRAMBLE_STEP = the step
# count of every sector at the scramble point, beyond the ladder's end;
# RESCALE_RTOL = relative tolerance of the analytic shear-calibration
# rescale, a few roundings of a product of double-precision factors.
NSTEP = 3
SCRAMBLE_STEP = 4  # every sector at step 4, bias included
RESCALE_RTOL = 1.0e-12


def _sector_of(name):
    """Return the sector of one sampled-parameter name.

    Arguments:
      name = a sampled-parameter name, e.g. "LSST_DZ_S1".

    Returns:
      the name of the first SECTORS entry whose pattern matches
      (pat.search looks for the pattern anywhere in the name).
    """
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """Return {parameter: per-step delta} for this sector's sampled names.

    Arguments:
      sector = a SECTORS name.
      names  = the sampled-parameter names that belong to the sector.

    Returns:
      a dictionary with one entry per name that a DELTAS key of the
      sector matches; names no key matches are left out.
    """
    # DELTAS.get(sector, {}) returns an empty table for a sector
    # without offsets (bias moves only in the scramble, "other" never)
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            # the condition reads: when key is a string, compare it with
            # n; otherwise key is a compiled pattern, so search n for it
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


class TestCacheConsistency(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen fiducial."""

    @classmethod
    def setUpClass(cls):
        """Move to ROOTDIR and verify the frozen state, once per class."""
        u.require_cocoa_environment()
        u.verify_frozen()

    def _point_at(self, fid, steps):
        """Return the ladder point with each sector at its given step count.

        Arguments:
          fid   = the fiducial {parameter name: value} point.
          steps = {sector name: step count}; each parameter of the
                  sector moves to fiducial + step * delta.

        Returns:
          a new {parameter name: value} dictionary (fid is not changed).
        """
        point = dict(fid)
        for sector, step in steps.items():
            for n, d in self.sector_deltas[sector].items():
                point[n] = fid[n] + step * d
        return point

    def _mpairs(self, like, np_):
        """Return, per data-vector entry, the source bins of its (1+m) factors.

        Cosmic shear scales by both bins' factors (1+m_i)(1+m_j),
        gamma_t (and the ks cross where present) by the source bin's,
        clustering, gk and kk by nothing. Fourier data vectors use ncl
        entries per block, real ones ntheta; the xi_pm split doubles
        the shear block in real space. The Fourier and 6x2pt branches
        serve projects that share this harness; lsst_y1 takes the
        real-space 3x2pt path.

        Arguments:
          like = the likelihood object (read for its bin counts and,
                 when present, its ggl_exclude pairs).
          np_  = the numpy module, passed in because numpy is imported
                 inside the test methods.

        Returns:
          int array [n_data, 2]: row k = (i, j), the source bins whose
          factors multiply entry k; -1 marks "no factor" (for gamma_t
          the row is (-1, source bin); for clustering both are -1).
        """
        import cosmolike_lsst_y1_interface as ci
        real = hasattr(ci, "compute_data_vector_3x2pt_real_sizes")
        sizes = [int(x) for x in
                 (ci.compute_data_vector_3x2pt_real_sizes() if real else
                  ci.compute_data_vector_3x2pt_fourier_sizes())]
        nlen = int(like.ntheta) if real else int(like.ncl)
        nsrc = int(like.source_ntomo)
        # sspairs = every source pair (i, j) with i <= j, in data-vector
        # order (the two for clauses nest like two C loops, i outer)
        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        # excluded = the set of (lens, source) pairs the likelihood drops
        # from gamma_t; getattr returns None when the likelihood has no
        # ggl_exclude attribute (the case here), and `or []` turns that
        # into an empty list
        excluded = {(int(a), int(b)) for a, b in
                    (getattr(like, "ggl_exclude", None) or [])}
        # gglpairs = every (lens, source) pair kept in gamma_t, lens bin
        # outer, source bin inner
        gglpairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                    for zs in range(nsrc) if (zl, zs) not in excluded]
        # start with -1 ("no factor") everywhere; the loops below fill
        # the shear and gamma_t rows block by block
        fac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        k = 0
        ssrep = sspairs + sspairs if real else sspairs  # xi_plus + xi_minus
        for (i, j) in ssrep:
            for t in range(nlen):
                fac[k] = (i, j)
                k += 1
        for (zl, zs) in gglpairs:
            for t in range(nlen):
                fac[k] = (-1, zs)
                k += 1
        k = sizes[0] + sizes[1] + sizes[2]  # skip clustering
        if len(sizes) > 3:  # 6x2pt: gk (no factor), ks (source), kk (none)
            k += sizes[3]
            for zs in range(nsrc):
                for t in range(nlen):
                    fac[k] = (-1, zs)
                    k += 1
        return fac

    def _run_ladder(self, tatt):
        """Walk the forward and mirrored ladders and assert checks 1-5.

        Each order builds its own model in this process (the two
        models share the compiled library's C globals); check 5
        compares their final vectors bit for bit.

        Arguments:
          tatt = True runs the TATT variant, False the NLA one.

        Returns:
          nothing; a failed check raises AssertionError.
        """
        import numpy as np
        import cosmolike_lsst_y1_interface as ci

        name = u.EXAMPLES[EXAMPLE]["likelihood"]
        results = {}
        for order in ("forward", "mirrored"):
            info = u.load_frozen_info(EXAMPLE, tatt=tatt)
            model = u.make_model(info)
            fid = dict(u.build_point(model, EXAMPLE, tatt=tatt))
            # sector_deltas = {sector: {parameter: per-step delta}}; the
            # inner comprehension lists the fiducial's names that belong
            # to sector s
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
            for s in ("cosmo", "dz_source", "m"):
                self.assertTrue(self.sector_deltas[s],
                                f"no sampled parameters in sector {s}")
            # a sector nothing samples drops out of the ladder: lenses
            # that are the source sample carry no separate DZ_L shifts,
            # and roman_kl fixes every IA amplitude
            active = tuple(s for s in PHASES if self.sector_deltas[s])

            phases = active if order == "forward" else tuple(reversed(active))
            # every sector starts at step 0 (the fiducial point)
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            # compute_data_vector_masked returns the theory vector of the
            # state the last evaluation left in cosmolike
            prev = np.array(ci.compute_data_vector_masked())
            mfac = self._mpairs(model.likelihood[name], np)

            for sector in phases:
                for r in range(1, NSTEP + 1):
                    m_prev = {n: fid[n] + steps["m"] * d
                              for n, d in self.sector_deltas["m"].items()}
                    steps[sector] = r
                    point = self._point_at(fid, steps)
                    u.evaluate_chi2(model, point)
                    dv = np.array(ci.compute_data_vector_masked())
                    self.assertFalse(
                        np.array_equal(dv, prev),
                        f"{order}: {sector} step {r} left the data vector "
                        "unchanged (dead sector flag or stale cache)")
                    if sector == "m":
                        # check 2: the expected ratio dv/prev of every
                        # entry is the product of (1+m_new)/(1+m_old)
                        # over the source bins mfac names for it
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        mp = sorted(m_prev)  # M1..M5 in bin order
                        ratio = np.ones(dv.size)
                        for k in range(dv.size):
                            i, j = mfac[k]
                            if j >= 0:
                                ratio[k] *= ((1 + m_now[mp[j]]) /
                                             (1 + m_prev[mp[j]]))
                            if i >= 0:
                                ratio[k] *= ((1 + m_now[mp[i]]) /
                                             (1 + m_prev[mp[i]]))
                        # nz = boolean mask of the entries where prev is
                        # nonzero (masked entries are 0 and cannot carry
                        # a ratio); rel = relative deviation from the
                        # analytic rescale on those entries
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: M step {r} is not the analytic "
                            f"(1+m_i)(1+m_j) rescale (max {rel.max():.2e})")
                    prev = dv

            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_masked())

            # no-op probe: identical point again, bitwise
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(np.array(ci.compute_data_vector_masked()),
                               final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # scramble: every sector at once, bias included
            scr = {s: SCRAMBLE_STEP for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, scr))

            # return: the ladder's final point must reproduce bitwise
            back_chi2 = u.evaluate_chi2(model, final_point)
            back_dv = np.array(ci.compute_data_vector_masked())
            self.assertTrue(
                np.array_equal(back_dv, final_dv),
                f"{order}: returning after the scramble did not reproduce "
                "the data vector bit for bit (stale sector cache)")
            self.assertEqual(
                back_chi2, final_chi2,
                f"{order}: chi2 after the scramble return differs")
            results[order] = final_dv
            print(f"  {order} ladder ({'TATT' if tatt else 'NLA'}): "
                  f"final chi2 = {final_chi2:.6f}", flush=True)

        self.assertTrue(
            np.array_equal(results["forward"], results["mirrored"]),
            "the mirrored-order ladder landed on a different data vector: "
            "the answer depends on the invalidation history")

    def test_cache_consistency_nla(self):
        """The ladder checks 1-5 hold for the 3x2pt NLA configuration."""
        self._run_ladder(tatt=False)

    def test_cache_consistency_tatt(self):
        """The ladder checks 1-5 hold for the 3x2pt TATT configuration."""
        self._run_ladder(tatt=True)


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
