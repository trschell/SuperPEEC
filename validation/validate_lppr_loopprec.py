# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the scalable loop-reduction preconditioner for the mixed-row LpPR
solve (setup1, 100 MHz - 10 GHz).

Architecture (see docs/precond_whitepaper.pdf): rescaling the external node
rows of the mixed-row operator by P_ext^{-1} turns it into the standard PEEC
MNA K' = [[Z,A],[A^T,-jw C]] (C = P_ext^{-1}, tiny), which the inductive
saddle [[Z,A],[A^T,0]] preconditions; that saddle is solved by divergence-free
loop reduction (loop basis B from getmesh_fortran, geometric Cholesky
chol(B^T B)), an INNER lgmres. The outer solver is flexible (FGMRES) because
the preconditioner is itself iterative.

THIS FILE CALLS PRODUCTION. It used to REIMPLEMENT the loop basis, the
Cholesky, the saddle solve and the W rescale, and therefore validated its own
copy rather than ``SystemMat``. That is how the 1e-3 inner tolerance survived:
the copy and the original shared a bug, so the test agreed with the code and
both were wrong. Everything here now goes through
``SystemMat.loopprecinit`` / ``rescaleLpPR`` / ``precondFoldin``; only the
DENSE reference is built locally, which is the point of an oracle.

Checks:
  0. the dense reference operator IS production's matvecLpPR (else the
     oracle and the code under test are different problems);
  1. the saddle solve is correct (satisfies [[Z,A],[A^T,0]] to tolerance);
  2. FGMRES on the rescaled system produces currents matching a dense
     equilibrated direct solve of the original mixed-row operator;
  3. it is efficient at 100 MHz - 1 GHz (few outer iterations). Honest note:
     it degrades toward 10 GHz, where the capacitance the inductive saddle
     ignores grows -- folding capacitance in further is the documented next
     step.

Run inside the toolbox:  python3 validate_lppr_loopprec.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import fgmres
import multipole as mp
import stencils as st
from systemmat import SystemMat

conductivity = 5.81e7
FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


NT = np.array([7, 7, 7]); N = st.single_level_nleaf(NT); LT = np.array([87.5e-6]*3)
M = mp.Tree(np.ones(NT, np.int8), N, LT, 1, 1e0, None, capacitive=True)
M.e.r = M.e.l[1]/(M.e.l[0]*M.e.l[2]*conductivity)
M.f.r = M.f.l[0]/(M.f.l[1]*M.f.l[2]*conductivity)
M.g.r = M.g.l[2]/(M.g.l[0]*M.g.l[1]*conductivity)
M.lv[0].beta = 1.0; M.alpha = 0; M.RDFinit()

S = SystemMat(M, 1j)
efg, nn = S.efgsize, S.nodesize
es, fs = S.esize, S.fsize
whole = S.wholedata
Rd = np.concatenate([np.full(es, M.e.r), np.full(S.fsize, M.f.r),
                     np.full(S.gsize, M.g.r)]).astype(np.complex128)

# ---- dense reference operators (the ORACLE; built here on purpose) -------
M.jomega = 1j
Lp = np.zeros((efg, efg), np.complex128)
for k in range(efg):
    whole[:efg] = 0; whole[k] = 1; M.traverseRL(); Lp[:, k] = whole[:efg]
Lp = 0.5*((Lp-np.diag(Rd))/1j + ((Lp-np.diag(Rd))/1j).T)
Amat = np.zeros((efg, nn), np.complex128)
for k in range(nn):
    M.lv[0].data[:] = 0; M.lv[0].data[k] = 1
    a = M.connectA(); Amat[:, k] = np.concatenate(a)
P = np.asarray(M.n2n, np.complex128)
extmask, intmask, ext = S.extmask, S.intmask, M.external

S.loopprecinit()
print("setup1: efg=%d nn=%d ext=%d loops=%d  saddle_rtol=%.0e\n"
      % (efg, nn, ext.size, S._loopsize, S.saddle_rtol))


def dense_K(w):
    Z = np.diag(Rd) + 1j*w*Lp
    D = np.diag(np.where(extmask, -1j*w, 0.0).astype(np.complex128))
    Cop = (np.diag(extmask+0j) @ P + np.diag(intmask+0j)) @ Amat.T
    K = np.zeros((efg+nn,)*2, np.complex128)
    K[:efg, :efg] = Z; K[:efg, efg:] = Amat
    K[efg:, :efg] = Cop; K[efg:, efg:] = D
    return K


print("%-9s %-9s %-11s %-11s" % ("f[GHz]", "outer_it", "resid(K')", "vs_direct"))
itlow = []
for f in [1e8, 3.16e8, 1e9, 3.16e9, 1e10]:
    w = 2*np.pi*f
    M.jomega = S.jomega = 1j*w
    K = dense_K(w)
    rng = np.random.default_rng(1); sn = np.zeros(nn, np.complex128)
    pk = rng.choice(nn, 80, replace=False); sn[pk[:40]] = 1e-6; sn[pk[40:]] = -1e-6
    rhs = np.zeros(efg+nn, np.complex128)
    rhs[efg:][extmask] = (P @ sn)[extmask]; rhs[efg:][intmask] = sn[intmask]
    if f == 1e9:
        # 0. the oracle must BE the operator under test
        v = rng.standard_normal(efg+nn) + 1j*rng.standard_normal(efg+nn)
        e0 = norm(np.asarray(S.matvecLpPR(v)) - K @ v)/norm(K @ v)
        check("dense reference matches production matvecLpPR (1 GHz)",
              e0 < 1e-10, "rel=%.1e" % e0)
        # 1. production's saddle solve satisfies the inductive saddle
        i0 = (rng.standard_normal(efg)+1j*rng.standard_normal(efg))*1e-6
        V0 = rng.standard_normal(nn)+1j*rng.standard_normal(nn)
        Z = np.diag(Rd) + 1j*w*Lp
        ri = Z @ i0 + Amat @ V0
        xi, xV = S._saddle_solve(ri, Amat.T @ i0)
        e1 = norm(Z @ xi + Amat @ xV - ri)/norm(ri)
        e2 = norm(Amat.T @ xi - Amat.T @ i0)/norm(Amat.T @ i0)
        check("saddle solve satisfies [[Z,A],[A^T,0]] (1 GHz)",
              e1 < 1e-6 and e2 < 1e-6, "|eq1|=%.1e |eq2|=%.1e" % (e1, e2))
    # equilibrated direct reference
    rs = np.abs(K).max(1); rs[rs == 0] = 1
    K1 = K/rs[:, None]; cs = np.abs(K1).max(0); cs[cs == 0] = 1
    xref = np.linalg.solve(K1/cs[None, :], rhs/rs)/cs
    # production operator + production preconditioner
    rhsp = S.rescaleRHS(rhs)
    Kpop = LinearOperator((efg+nn,)*2, matvec=S.rescaleLpPR,
                          dtype=np.complex128)
    Msad = LinearOperator((efg+nn,)*2, matvec=S.precondFoldin,
                          dtype=np.complex128)
    cnt = [0]
    x, fl = fgmres(Kpop, rhsp, M=Msad, tol=1e-8, maxiter=8, restrt=40,
                   callback=lambda xx: cnt.__setitem__(0, cnt[0]+1))
    resid = norm(S.rescaleLpPR(x) - rhsp)/norm(rhsp)
    vsd = norm(x[:efg] - xref[:efg])/norm(xref[:efg])
    if f <= 1e9:
        itlow.append(cnt[0])
    print("%-9.2f %-9d %-11.2e %-11.2e" % (f/1e9, cnt[0], resid, vsd))
    check("currents correct at %.2g Hz (vs dense direct)" % f, vsd < 3e-3,
          "vs_direct=%.2e" % vsd)

check("efficient at 100 MHz - 1 GHz (<=10 outer iters)",
      max(itlow) <= 10, "max outer iters = %d" % max(itlow))

print("\n%d checks failed" % len(FAIL))
sys.exit(1 if FAIL else 0)
