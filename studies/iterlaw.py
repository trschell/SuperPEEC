# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""ITERATION-COUNT GROWTH LAW for the LpR loop solve.

THE QUESTION THIS SETTLES. Extrapolating the per-matvec cost is easy --
the FMM is O(N) with a per-cell cost that FALLS as it amortises (3.44 ->
2.92 us/cell over a 16x size range), so a billion cells is ~48 min per
matvec. What nobody can extrapolate from an armchair is how many
matvecs a converged solve NEEDS at that size, and that is the exponent
while per-matvec cost is only a constant factor:

    ~300 iterations at 48 min/matvec  ->  days      (feasible)
    ~10000 iterations                 ->  months    (not)

No amount of p2p acceleration closes a 30x gap in the count, which is
why this measurement outranks every per-matvec optimisation on the
docket. It is also the cheapest of them: no new code, just the existing
solver over a size ladder.

WHAT IS HELD FIXED, because a growth law is only as good as its
controls (this session has already produced three studies ruined by
something else moving):
  * FREQUENCY fixed -- iteration counts move with wL/R, so a ladder
    that also changes frequency measures two things at once.
  * The SAME basis and preconditioner at every rung ('overcomplete' +
    AMG, with the macro-block fix of 2026-08-10 -- older recorded
    counts on perforated geometry are pessimistic by up to 35%).
  * The SAME geometry family, scaled: pdn_planes conductor-only with
    stitching vias, so the port always has a galvanic return and the
    via/antipad filigree scales with the board.
  * Tolerance fixed. Counts are meaningless if the target moves.

The board is PERFORATED and multiply connected, which is deliberate:
the homology grows with the board (hole cycles scale with the via
count), so this measures the honest case rather than a solid block.

Usage:
    PYTHONPATH=.:studies python3 studies/iterlaw.py
Env: SIZES (comma-separated nplane, default 80,160,320), FREQ (1e8),
     STITCH (8), RTOL (1e-12)
"""
import os
import sys
import time
import resource

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import equiterminal as eq
from pdn_planes import build_pdn


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6


def run(nplane, freq, stitch, rtol, maxiter, inner_m):
    m = build_pdn(nplane=nplane, eps_r=None, stitch=stitch)
    ncell = int((np.asarray(m.sigma) != 0).sum())
    leaf, levels = m.partition()
    t0 = time.perf_counter()
    M = m.build_tree(leaf, levels)
    m.prepare(M, freq)
    t_tree = time.perf_counter() - t0
    t0 = time.perf_counter()
    S = eq.EquiTerminalSolver(m, M, 0, basis='overcomplete')
    t_set = time.perf_counter() - t0
    t0 = time.perf_counter()
    z, i, info = S.solve(freq, rtol=rtol, maxiter=maxiter,
                         inner_m=inner_m)
    t_solve = time.perf_counter() - t0
    return dict(nplane=nplane, cells=ncell, loops=int(S.meshsize),
                nplaq=int(S.nplaq), nholes=int(S.nholes),
                mv=int(info['matvecs']), tree=t_tree, setup=t_set,
                solve=t_solve, rss=rss_gb(), z=z,
                flag=info['flag'], resid=float(info['residual']),
                budget=maxiter*inner_m,
                nleaf=list(int(v) for v in leaf), levels=int(levels))


def main():
    sizes = [int(v) for v in
             os.environ.get('SIZES', '80,160,320').split(',')]
    freq = float(os.environ.get('FREQ', '1e8'))
    stitch = int(os.environ.get('STITCH', '8'))
    rtol = float(os.environ.get('RTOL', '1e-12'))
    # BUDGET, and why the default is not the solver's. lgmres here is
    # maxiter OUTER cycles of inner_m, so the matvec count SATURATES at
    # ~maxiter*inner_m and a capped run looks exactly like a converged
    # one unless you read the flag. Measured 2026-08-10 on the 160^2
    # board: maxiter=10 gives 311 mv / flag 10 / resid 9.6e-11, while
    # maxiter=20 gives 402 mv / flag 0 / resid 5.4e-13. The first ladder
    # run compared two SATURATED rungs (311 vs 311) and reported an
    # exponent of exactly 0.000 -- a ceiling, not a scaling law. The
    # recorded 320^2 anchor of "311 matvecs, resid 6.4e-9" is the same
    # artefact. Answers were unaffected (Z identical to 6 digits).
    maxiter = int(os.environ.get('MAXITER', '40'))
    inner_m = int(os.environ.get('INNER', '30'))
    print("LpR iteration-growth ladder: pdn_planes conductor-only, "
          "stitch %d, %.3g Hz, rtol %.0e, basis overcomplete"
          % (stitch, freq, rtol), flush=True)
    print("%6s %9s %9s %7s %6s %5s %9s %7s %8s %7s %s"
          % ("nplane", "cells", "loops", "holes", "mv", "flag", "resid",
             "tree s", "solve s", "RSS GB", "nleaf/lv"), flush=True)
    rows = []
    for n in sizes:
        try:
            r = run(n, freq, stitch, rtol, maxiter, inner_m)
        except Exception as e:
            print("  nplane %d FAILED %s: %s"
                  % (n, type(e).__name__, str(e)[:120]), flush=True)
            continue
        rows.append(r)
        print("%6d %9d %9d %7d %6d %5d %9.2e %7.1f %8.1f %7.1f %s/%d%s"
              % (r['nplane'], r['cells'], r['loops'], r['nholes'],
                 r['mv'], r['flag'], r['resid'], r['tree'], r['solve'],
                 r['rss'], r['nleaf'], r['levels'],
                 "" if r['flag'] == 0 else "  <-- NOT CONVERGED"),
              flush=True)
    bad = [r for r in rows if r['flag'] != 0]
    if bad:
        print("\nNO GROWTH LAW: %d of %d rungs hit the iteration "
              "BUDGET (%d matvecs) rather than converging -- their "
              "counts are ceilings and comparing them measures the cap, "
              "not the solver. Raise MAXITER."
              % (len(bad), len(rows), rows[0]['budget']), flush=True)
    elif len(rows) >= 2:
        levs = {r['levels'] for r in rows}
        if len(levs) > 1:
            print("\nWARNING rungs differ in TREE DEPTH (%s): the FMM "
                  "operator changes with it, so this ladder measures "
                  "size AND topology at once."
                  % sorted(levs), flush=True)
        print("\nGROWTH LAW (matvecs vs cells, log-log):", flush=True)
        c = np.array([r['cells'] for r in rows], dtype=float)
        v = np.array([r['mv'] for r in rows], dtype=float)
        for i in range(1, len(rows)):
            p = np.log(v[i]/v[i-1])/np.log(c[i]/c[i-1])
            print("  %d -> %d cells: %d -> %d mv  (x%.2f cells, "
                  "x%.2f mv, local exponent %.3f)"
                  % (c[i-1], c[i], v[i-1], v[i], c[i]/c[i-1],
                     v[i]/v[i-1], p), flush=True)
        p = np.polyfit(np.log(c), np.log(v), 1)[0]
        print("  overall exponent p = %.3f  (mv ~ N^p)" % p, flush=True)
        for target in (1e8, 1e9):
            print("    extrapolated to %.0e cells: %.0f matvecs"
                  % (target, v[-1]*(target/c[-1])**p), flush=True)
        print("  READ THIS AS A BOUND, NOT A PREDICTION: three or four "
              "rungs over a 16x range cannot separate p = 0.1 from a "
              "logarithm, and the extrapolation spans another 1000x.",
              flush=True)


if __name__ == '__main__':
    main()
