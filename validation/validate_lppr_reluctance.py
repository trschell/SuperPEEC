# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the reluctance block-triangular preconditioner for the mixed-row
LpPR solve under STANDARD GMRES (setup1, 100 MHz - 10 GHz).

Motivation. The loop-reduction preconditioner (validate_lppr_loopprec.py) is an
ITERATIVE inner solve, so it forces flexible FGMRES and DEGRADES toward 10 GHz
(58-66 outer iters) as the geometric loop Cholesky weakens with frequency. The
reluctance K = Lp^{-1} is a localized, frequency-INDEPENDENT operator; a sparse
K~ (reluctance.py: extract ~50 / retain ~27, per orientation, symmetrized) gives
a FIXED linear preconditioner, so ordinary GMRES applies and the conditioning
IMPROVES with frequency -- the opposite trend, and the regime the loop path
failed.

Why not the naive (R,K). Left-multiplying only the branch rows by K~ leaves the
mixed-row system a saddle point (the internal nodes have q=0, a zero (2,2)
block); block-diagonal preconditioning clusters the (1,1) block to cond ~6 but
NOT the saddle's null space, so GMRES stalls (cond(MK) ~ 1e20, measured). The
fix is a saddle-aware, still-FIXED block-triangular preconditioner.

Architecture. First remove the P ~ 1e16 scale disparity by rescaling the
external node rows by W = P_ext^{-1} (as in the loop path), giving the standard
PEEC MNA K' = [[Z, A], [A^T, -jw C_cap]], C_cap = P_ext^{-1} (tiny). Then apply
the block-upper-triangular preconditioner

    x_v = S~^{-1} r_v ;   x_i = N_Z (r_i - A x_v),

with N_Z ~ Z^{-1} and S~ ~ the nodal Schur complement. Reluctance gives
K~ Z ~ jw I, i.e. K~ ~ jw Z^{-1}, so the consistent choices are

    N_Z = (1/jw) K~ ,   S~ = -jw C_cap - (1/jw) A^T K~ A

(the jw factors matter: dropping them -- as a naive S~ = -jw C_cap - A^T K~ A --
converges the preconditioned residual to the WRONG currents). Both N_Z and S~
are FIXED sparse operators per frequency (K~ and A^T K~ A are frequency
independent; only the scalar jw and the direct factorization of S~ change), so
the outer solver is ordinary GMRES. S~ is factored WITHOUT gauge pinning: the
-jw C_cap diagonal breaks the null space of the nodal reluctance Laplacian, so
S~ is nonsingular as-is -- and pinning a node while zeroing its residual in the
apply makes M singular along that node, converging the preconditioned residual
to wrong currents (found by validate_resonator_precond.py, where the pinned
node was external and carried capacitance).

THIS FILE CALLS PRODUCTION. It used to REIMPLEMENT the whole preconditioner
-- A^T K~ A, C_cap, the S~ assembly, its factorization and the block-triangular
apply -- and so validated its own copy of ``SystemMat.reluctanceprecinit`` /
``precondReluctance`` rather than the shipped one. That pattern is how the
loop path's rtol=1e-3 bug survived for so long: the copy and the original
shared it, so the test agreed with the code and both were wrong. Only the
DENSE reference is built locally now, which is the point of an oracle.

Checks:
  0. the dense reference operator IS production's rescaleLpPR (else the oracle
     and the code under test are different problems);
  1. the K~ production actually built matches a dense-extracted K~;
  2. standard GMRES on K', block-tri preconditioned, gives currents matching a
     dense equilibrated direct solve of the SAME K';
  3. iteration count is low at high frequency and does NOT grow toward 10 GHz
     (the reluctance regime), unlike the loop-reduction preconditioner. (Honest
     note: it is a high-frequency preconditioner -- iterations rise below ~300
     MHz, where the loop/LpR path is the better choice.)

Run inside the toolbox:  python3 validate_lppr_reluctance.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.linalg import lu_factor, lu_solve
import multipole as mp
import stencils as st
import reluctance as rel
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
whole = S.wholedata
Rd = np.concatenate([np.full(S.esize, M.e.r), np.full(S.fsize, M.f.r),
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
luPext = lu_factor(P[np.ix_(ext, ext)].real)

# 1. the K~ PRODUCTION built, against a dense-extracted ground truth
S.reluctanceprecinit()
Kd = rel.extract_reluctance(M, Lp_dense=np.real(Lp), window=50, retain=27)
kerr = norm((S._Ktilde - Kd).toarray())/norm(Kd.toarray())
print("setup1: efg=%d nn=%d ext=%d;  K~ %.1f nnz/row (%.1f%% sparse)\n"
      % (efg, nn, ext.size, S._Ktilde.nnz/efg,
         100*(1-S._Ktilde.nnz/efg**2)))
check("production K~ matches dense-extracted K~", kerr < 1e-2,
      "|Kk-Kd|/|Kd| = %.2e" % kerr)


def gmres_solve(Kop, Mop, b, tol=1e-8, maxit=300):
    h = []
    x, _ = gmres(Kop, b, M=Mop, rtol=tol, restart=maxit, maxiter=1,
                 callback=lambda rk: h.append(rk), callback_type='pr_norm')
    h = np.asarray(h)
    return x, (int(np.argmin(h/h[0]))+1 if h.size else 0)


print("%-9s %-9s %-11s %-11s" % ("f[GHz]", "gmres_it", "resid(K')", "vs_direct"))
its = {}
opmax = 0.0
for f in [1e8, 3.16e8, 1e9, 3.16e9, 1e10]:
    w = 2*np.pi*f; jw = 1j*w
    S.jomega = jw; S.M.jomega = jw
    S._factorReluctanceSchur()          # per-frequency S~ refactor
    # dense oracle K'
    Kmix = np.zeros((efg+nn,)*2, np.complex128)
    Kmix[:efg, :efg] = np.diag(Rd) + jw*Lp; Kmix[:efg, efg:] = Amat
    Kmix[efg:, :efg] = (np.diag(extmask+0j) @ P
                        + np.diag(intmask+0j)) @ Amat.T
    Kmix[efg:, efg:] = np.diag(np.where(extmask, -jw, 0.0).astype(np.complex128))
    Kp = Kmix.copy()
    rb = Kmix[efg:][ext]
    Kp[efg:][ext] = lu_solve(luPext, rb.real) + 1j*lu_solve(luPext, rb.imag)
    # 0. the oracle must BE the operator under test
    rng = np.random.default_rng(1)
    vt = rng.standard_normal(efg+nn) + 1j*rng.standard_normal(efg+nn)
    opmax = max(opmax, norm(S.rescaleLpPR(vt) - Kp @ vt)/norm(Kp @ vt))
    # physical distributed nodal source
    sn = np.zeros(nn, np.complex128)
    pk = rng.choice(nn, 80, replace=False); sn[pk[:40]] = 1e-6; sn[pk[40:]] = -1e-6
    rhs = np.zeros(efg+nn, np.complex128)
    rhs[efg:][extmask] = (P @ sn)[extmask]; rhs[efg:][intmask] = sn[intmask]
    rhsp = S.rescaleRHS(rhs)
    # equilibrated direct reference on the SAME K'
    rs = np.abs(Kp).max(1); rs[rs == 0] = 1
    K1 = Kp/rs[:, None]; cs = np.abs(K1).max(0); cs[cs == 0] = 1
    xref = np.linalg.solve(K1/cs[None, :], rhsp/rs)/cs
    # production operator + production preconditioner
    Kpop = LinearOperator((efg+nn,)*2, matvec=S.rescaleLpPR,
                          dtype=np.complex128)
    Mop = LinearOperator((efg+nn,)*2, matvec=S.precondReluctance,
                         dtype=np.complex128)
    x, it = gmres_solve(Kpop, Mop, rhsp)
    resid = norm(S.rescaleLpPR(x) - rhsp)/norm(rhsp)
    vsd = norm(x[:efg] - xref[:efg])/norm(xref[:efg])
    its[f] = it
    print("%-9.2f %-9d %-11.2e %-11.2e" % (f/1e9, it, resid, vsd))
    check("currents correct at %.2g Hz (vs dense direct)" % f, vsd < 3e-3,
          "vs_direct=%.2e" % vsd)

print()
check("dense reference matches production rescaleLpPR", opmax < 1e-10,
      "max rel = %.2e" % opmax)
check("standard GMRES efficient at 1-10 GHz (<= 40 iters)",
      max(its[1e9], its[3.16e9], its[1e10]) <= 40,
      "1GHz=%d 10GHz=%d" % (its[1e9], its[1e10]))
check("best at high frequency (iters do NOT grow toward 10 GHz)",
      its[1e10] <= its[1e8], "100MHz=%d 10GHz=%d" % (its[1e8], its[1e10]))

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL)); sys.exit(1)
print("all checks passed -- reluctance block-triangular preconditioner solves "
      "mixed-row LpPR under STANDARD GMRES, best at high frequency")
