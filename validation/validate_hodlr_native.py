# SPDX-License-Identifier: MIT
"""Correctness gate for hodlr_native.c: the native (C+BLAS) forward solve
must reproduce the Python recursive-Woodbury solve to rounding (~1e-13).
Checks 7^3 and 11^3, single- and multi-RHS, two frequencies (100 MHz-class
and 4*f_res-class), and reports the per-solve speedup that motivated the
native path (S_c sampling drives 17k-70k columns through this solve).
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import time
import numpy as np
from numpy.linalg import norm
pc = time.perf_counter

with open(_op.path.join(_op.path.dirname(_op.path.abspath(__file__)), 'main.py')) as fh:
    src = fh.read()
head = src[:src.index('\nconductivity = 5.81e7')]
ns = {}
exec(compile(head, '<main.py head>', 'exec'), ns)
mp = ns['mp']
import hodlr

assert hodlr._nat is not None, "libhodlrnat.so not loaded"

conductivity = 5.81e7
CELL = 1e-5
c0 = 299792458.0
EPS = 1e-3
fails = 0

for NT3 in [7, 11]:
    fs = np.ones((NT3,)*3, np.int8)
    M = mp.Tree(fs, np.array([4]*3), np.array([NT3+1]*3)*CELL, 2, 1e0, 4,
                capacitive=True)
    M.e.r = M.e.l[1]/(M.e.l[0]*M.e.l[2]*conductivity)
    M.f.r = M.f.l[0]/(M.f.l[1]*M.f.l[2]*conductivity)
    M.g.r = M.g.l[2]/(M.g.l[0]*M.g.l[1]*conductivity)
    S = ns['SystemMat'](M, 1j)
    H = hodlr.HodlrZ(M, eps=EPS, leaf=96)
    n = S.efgsize
    rng = np.random.default_rng(7)
    f_res = c0/(6*NT3*CELL)
    for tag, f in [("100MHz", 1e8), ("4fres", 4*f_res)]:
        jw = 2j*np.pi*f
        H.factor(jw, (M.e.r, M.f.r, M.g.r))
        assert all(p is not None or H.trees[o] is None
                   for o, p in enumerate(H._packed)), "pack missing"
        for nrhs in [1, 7]:
            b = (rng.standard_normal((n, nrhs)) +
                 1j*rng.standard_normal((n, nrhs)))
            if nrhs == 1:
                b = b[:, 0]
            xn = H.solve(b)
            xnt = H.solve_t(b)
            # force the python path for the reference
            packed = H._packed
            H._packed = []
            xp = H.solve(b)
            xpt = H.solve_t(b)
            H._packed = packed
            rel = norm(xn - xp)/norm(xp)
            relt = norm(xnt - xpt)/norm(xpt)
            ok = rel < 1e-12 and relt < 1e-12
            fails += 0 if ok else 1
            print("%2d^3 %-6s nrhs=%d  native-vs-python N rel=%.2e "
                  "T rel=%.2e  %s"
                  % (NT3, tag, nrhs, rel, relt, "OK" if ok else "FAIL"),
                  flush=True)
    # timing: the S_c sampling pattern (many narrow solves)
    b1 = rng.standard_normal(n) + 1j*rng.standard_normal(n)
    reps = 60
    t0 = pc()
    for _ in range(reps):
        H.solve(b1)
    tn = (pc() - t0)/reps
    packed = H._packed
    H._packed = []
    t0 = pc()
    for _ in range(reps):
        H.solve(b1)
    tp = (pc() - t0)/reps
    H._packed = packed
    print("%2d^3 per-column solve: python %.2f ms -> native %.2f ms "
          "(%.1fx)" % (NT3, 1e3*tp, 1e3*tn, tp/tn), flush=True)

print("RESULT:", "ALL OK" if fails == 0 else "%d FAILURES" % fails)
