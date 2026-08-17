# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validation for the windowed reluctance extractor (reluctance.py), setup1.

Checks:
  1. the scalable genL3D-kernel block-getter matches the ground-truth dense
     (FMM-probed Lp) block-getter;
  2. K~ is a sparse local stencil (few nonzeros per row);
  3. K~ preconditions the branch impedance Z = R + jw*Lp to a small, frequency-
     robust condition number, best at high frequency (the regime the topological
     loop preconditioner failed);
  4. preconditioned GMRES converges in few iterations, decreasing with
     frequency.

Run in the toolbox: python3 validate_reluctance.py   (nonzero exit on failure)
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
from numpy.linalg import cond, norm
from scipy.sparse.linalg import LinearOperator, gmres
import multipole as mp
import reluctance as rel

FAIL = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)

conductivity = 5.81e7
NT = np.array([7, 7, 7]); N = np.array([8, 8, 8]); LT = np.array([87.5e-6]*3)
M = mp.Tree(np.ones(NT, np.int8), N, LT, 1, 1e0, None, capacitive=False)
M.e.r = M.e.l[1]/(M.e.l[0]*M.e.l[2]*conductivity)
M.f.r = M.f.l[0]/(M.f.l[1]*M.f.l[2]*conductivity)
M.g.r = M.g.l[2]/(M.g.l[0]*M.g.l[1]*conductivity)
es = M.e.struc.size; fs = M.f.struc.size; gs = M.g.struc.size; efg = es+fs+gs
whole = np.zeros(efg, np.complex128)
M.e.data = whole[:es]; M.f.data = whole[es:es+fs]; M.g.data = whole[es+fs:efg]
Rdiag = np.concatenate([np.full(es, M.e.r), np.full(fs, M.f.r),
                        np.full(gs, M.g.r)]).astype(np.complex128)
M.jomega = 1j
Lp = np.zeros((efg, efg), np.complex128)
for k in range(efg):
    whole[:] = 0; whole[k] = 1; M.traverseRL(); Lp[:, k] = whole
Lp = np.real(0.5*((Lp - np.diag(Rdiag))/1j + ((Lp - np.diag(Rdiag))/1j).T))
Rc = np.diag(Rdiag)

Kd = rel.extract_reluctance(M, Lp_dense=Lp, window=50, retain=27)
Kk = rel.extract_reluctance(M, Lp_dense=None, window=50, retain=27, verbose=True)
print("setup1: %d filaments;  K~ = %.1f nnz/row (%.1f%% sparse)\n"
      % (efg, Kd.nnz/efg, 100*(1 - Kd.nnz/efg**2)))

rel_err = norm((Kk - Kd).toarray())/norm(Kd.toarray())
check("scalable kernel block-getter matches ground-truth dense",
      rel_err < 1e-2, "|Kk - Kd|/|Kd| = %.2e" % rel_err)
check("K~ is a sparse local stencil (<= 40 nnz/row)",
      Kd.nnz/efg <= 40, "%.1f nnz/row" % (Kd.nnz/efg))


def gmres_iters(A, Mp, b, tol=1e-8, maxit=400):
    h = []
    Aop = LinearOperator((efg, efg), matvec=lambda v: A @ v, dtype=np.complex128)
    Mop = LinearOperator((efg, efg), matvec=lambda v: Mp @ v, dtype=np.complex128)
    gmres(Aop, b, M=Mop, rtol=tol, restart=maxit, maxiter=1,
          callback=lambda rk: h.append(rk), callback_type='pr_norm')
    h = np.asarray(h)
    return int(np.argmin(h/h[0])) + 1 if h.size else 0


rng = np.random.default_rng(0)
b = (rng.standard_normal(efg) + 1j*rng.standard_normal(efg)).astype(np.complex128)
Kk_c = Kk.astype(np.complex128)
print("%-8s %-11s %-11s %-9s" % ("f[GHz]", "cond(Z)", "cond(K~Z)", "gmres(K~Z)"))
conds = {}; its = {}
for f in [1e8, 1e9, 1e10]:
    w = 2*np.pi*f
    Z = Rc + 1j*w*Lp
    c = cond(Kk_c @ Z); it = gmres_iters(Z, Kk_c, b)
    conds[f] = c; its[f] = it
    print("%-8.2f %-11.2e %-11.2e %-9d" % (f/1e9, cond(Z), c, it))

print()
check("cond(K~Z) < 10 across 100 MHz - 10 GHz",
      max(conds.values()) < 10, "max = %.2f" % max(conds.values()))
check("preconditioner is best at high frequency",
      conds[1e10] <= conds[1e8] and its[1e10] <= its[1e8],
      "cond 10GHz=%.1f <= 100MHz=%.1f" % (conds[1e10], conds[1e8]))
check("GMRES converges in <= 15 iters at 1-10 GHz",
      its[1e9] <= 15 and its[1e10] <= 15,
      "1GHz=%d 10GHz=%d" % (its[1e9], its[1e10]))

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL)); sys.exit(1)
print("all checks passed -- windowed reluctance extractor produces a sparse, "
      "frequency-robust inductive preconditioner")
