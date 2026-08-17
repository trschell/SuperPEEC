# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""CONSISTENCY: does the cross-section model converge to the exact
physics as the radial mesh is refined without limit?

This is not a design study -- 25 elements is already more than anyone
wants to pay. It is the check that the model is CONSISTENT: if the
error does not go to zero as rings -> infinity, there is a systematic
defect that no amount of resolution removes, and every accuracy figure
quoted at a practical element count is sitting on top of it.

TWO TRAPS THAT WOULD FAKE A FLOOR, both avoided here:
  * A FIXED CORE DISC never refines. round_wire's core is one element
    whatever nring is, so leaving core_frac fixed would leave a coarse
    element at the centre forever. Here core_frac = 1/(nring+1), so the
    core shrinks with the rings.
  * GEOMETRIC GRADING never refines the innermost ring: with grade=0.5
    the first ring keeps ~half the annulus width at every nring. Good
    for accuracy per unknown, useless for a convergence test. Here
    grade=1.0, i.e. uniform radial spacing.
An isolated wire is axisymmetric, so nsect=1 (full annuli) costs
nothing in accuracy and buys the element count needed to push the
refinement far.

The GMD quadrature is deliberately decoupled from the mutual
quadrature and run much finer: gmd converges slowly (log singularity)
and feeds every self term, so it is the prime suspect for any floor
that does appear. If the error stalls, re-run with GMDN higher -- if
the floor moves, it was gmd.

Run: PYTHONPATH=src python3 studies/wire_converge.py
Env: RINGS (max ring count, 64), GMDN (gmd angular order, 96),
     FREQS (a/delta targets)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import wirekernel as wk
from wire_bessel import bessel_zint

A = 200e-6
SIGMA = 3.5e7
LEN = 5e-3


def rac_ratio(nring, freq, nr=3, nth=None, gmd_nr=6, gmd_nth=96):
    """R_ac/R_dc from a uniform radial mesh of nring+1 elements."""
    # ANGULAR QUADRATURE MUST SCALE WITH THE MESH. Adjacent rings of
    # thickness t ~ a/nring at radius r ~ a make the mutual integrand a
    # thin-gap problem, peaked at coincident angles with width ~t/r, so
    # a fixed nth STARVES faster than the mesh refines. Measured at
    # a/delta 7.43: nring=16 gives -6.31% at nth=16 but -0.34% at
    # nth=128, and holding nth=16 made the error DIVERGE with
    # refinement (+43%, +140%, +80% at 4/8/16 rings, a/delta 23.5) --
    # which looks exactly like an inconsistent model and is not one.
    if nth is None:
        nth = max(24, 8*nring)
    cf = 1.0/(nring + 1)
    fils = wk.round_wire([0, 0, 0], [0, 0, 1], LEN, A, nsect=1,
                         core_frac=cf, nring=nring, grade=1.0,
                         nr=nr, nth=nth)
    n = len(fils)
    rc = cf*A
    edges = np.concatenate([[0.0, rc],
                            rc + (A - rc)*np.arange(1, nring + 1)/nring])
    g = []
    for m in range(n):
        off, ar = wk.sector_points(edges[m], edges[m + 1], 0.0, 2*np.pi,
                                   gmd_nr, gmd_nth)
        g.append(wk.gmd(off, ar))
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = (wk.mutual(fils[i], fils[i], self_gmd=g[i]) if i == j
                 else wk.mutual(fils[i], fils[j]))
            L[i, j] = L[j, i] = v
    rho = 1.0/SIGMA
    R = np.array([f.resistance(rho) for f in fils])
    Z = np.diag(R) + 1j*2*np.pi*freq*L
    one = np.ones(n)
    rdc = LEN/(SIGMA*np.pi*A*A)
    return (1.0/complex(one @ np.linalg.solve(Z, one))).real/rdc


def main():
    rmax = int(os.environ.get('RINGS', '64'))
    gmdn = int(os.environ.get('GMDN', '96'))
    freqs = [1e6, 1e7, 1e8]
    counts = [n for n in (1, 2, 4, 8, 16, 32, 64, 128) if n <= rmax]
    delta = [1.0/np.sqrt(np.pi*f*wk.MU0*SIGMA) for f in freqs]
    print("uniform radial refinement, isolated wire (nsect=1), "
          "a = %.0f um, gmd order %d" % (1e6*A, gmdn), flush=True)
    print("exact R_ac/R_dc: " + "  ".join(
        "a/d=%.1f: %.4f" % (A/d, bessel_zint(f, A, SIGMA).real*LEN
                            / (LEN/(SIGMA*np.pi*A*A)))
        for f, d in zip(freqs, delta)), flush=True)
    print("\n%6s %7s %s" % ("rings", "elems",
                            "".join("  err(a/d=%4.1f)" % (A/d)
                                    for d in delta)), flush=True)
    for nring in counts:
        t0 = time.perf_counter()
        row = []
        for f, d in zip(freqs, delta):
            ex = bessel_zint(f, A, SIGMA).real*LEN/(LEN/(SIGMA*np.pi*A*A))
            row.append(100*(rac_ratio(nring, f, gmd_nth=gmdn)/ex - 1))
        print("%6d %7d %s   (%.0f s)"
              % (nring, nring + 1,
                 "".join("%14.3f" % v for v in row),
                 time.perf_counter() - t0), flush=True)
    print("\nA TRUE floor would show as the error stalling at a fixed "
          "value; halving with each doubling is O(h), quartering is "
          "O(h^2).", flush=True)


if __name__ == '__main__':
    main()
