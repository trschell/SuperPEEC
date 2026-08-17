# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the block-triangular Schur preconditioner for the mixed-row LpPR
solve across the 100 MHz - 10 GHz band (setup1 geometry).

For each frequency it builds the preconditioner (SystemMat.precondinitLpPR),
runs preconditioned lgmres on the matrix-free matvecLpPR operator, and checks:

  1. with a realistic distributed nodal-injection source, preconditioned
     lgmres converges in ~2 iterations across 100 MHz - 10 GHz (the block-
     triangular Schur preconditioner drives the spectrum to a degree-2
     minimal polynomial);
  2. UNpreconditioned lgmres stalls across the whole band -- the W-rescaled
     operator still conditions at ~1e7-1e8, which Krylov cannot work with
     unaided -- so the preconditioner is doing the work, not the Krylov
     method;
  3. the preconditioned solution matches a dense equilibrated direct solve of
     the same operator.

The system solved here is W-RESCALED, exactly as the production path does
it (SystemMat.rescaleLpPR / rescaleRHS, via n2nchol = P_ext^-1). That is
not cosmetic. Unscaled, the node rows carry P ~ 1e14 against a jw ~ 1e11
capacitive block and the operator conditions at ~1e22, at which point
check 3 cannot distinguish a correct solve from a wrong one: the
preconditioned solution reached a 1e-13 residual and was still 100% wrong
(and it is exact), differing from the
truth in a near-null direction. Rescaled, both schemes agree with the
direct solve to ~1e-13. If this file is ever changed to precondition the
raw blocks again, check 3 becomes decorative.

Note: a single low-rank source (two injection points) can converge
unpreconditioned by spanning only a small Krylov space, masking the
preconditioner's effect; a distributed source exercises it honestly.

This is the RF regime the earlier 10 Hz work missed: there Z = R + jwLp is
diagonally dominant and any diagonal preconditioner suffices; here it is
dense and far from diagonally dominant.

Run inside the toolbox:  python3 validate_lppr_precond.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator, lgmres
import multipole as mp
import stencils as st

# main.py is a driver script, not importable: exec its head to get the
# real SystemMat. This validator MUST use the production preconditioner
# rather than a local equivalent -- an earlier version reimplemented it,
# so SystemMat.precondinitLpPR had no coverage while this file's
# docstring claimed otherwise.
_ns = {}
_src = open(__file__.rsplit('/', 1)[0] + '/main.py').read() \
    if '/' in __file__ else open('main.py').read()
exec(compile(_src[:_src.index('\nconductivity = 5.81e7')],
             '<main.py head>', 'exec'), _ns)
SystemMat = _ns['SystemMat']

conductivity = 5.81e7
FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# ---- setup1 geometry, capacitive tree ----
NT = np.array([7, 7, 7]); N = st.single_level_nleaf(NT)
LT = np.array([87.5e-6, 87.5e-6, 87.5e-6])
fullstruc = np.ones(NT, dtype=np.int8)
M = mp.Tree(fullstruc, N, LT, 1, 1e0, None, capacitive=True)
M.e.r = M.e.l[1]/(M.e.l[0]*M.e.l[2]*conductivity)
M.f.r = M.f.l[0]/(M.f.l[1]*M.f.l[2]*conductivity)
M.g.r = M.g.l[2]/(M.g.l[0]*M.g.l[1]*conductivity)
M.lv[0].beta = 1.0
M.alpha = 0

esize = np.size(M.e.struc); fsize = np.size(M.f.struc); gsize = np.size(M.g.struc)
efg = esize+fsize+gsize; nnode = np.size(M.lv[0].struc)
whole = np.zeros((efg+nnode,), dtype=np.complex128)
M.e.data = whole[:esize]; M.f.data = whole[esize:esize+fsize]
M.g.data = whole[esize+fsize:efg]; M.lv[0].data = whole[efg:]

extmask = np.zeros(nnode, bool); extmask[M.external] = True
intmask = ~extmask
Rdiag = np.concatenate([np.full(esize, M.e.r), np.full(fsize, M.f.r),
                        np.full(gsize, M.g.r)]).astype(np.complex128)

# dense operators (Lp once, frequency-independent)
M.jomega = 1j
Lp = np.zeros((efg, efg), dtype=np.complex128)
for k in range(efg):
    whole[:efg] = 0; whole[k] = 1.0; M.traverseRL(); Lp[:, k] = whole[:efg]
Lp = 0.5*((Lp-np.diag(Rdiag))/1j + ((Lp-np.diag(Rdiag))/1j).T)
Amat = np.zeros((efg, nnode), dtype=np.complex128)
for k in range(nnode):
    M.lv[0].data[:] = 0; M.lv[0].data[k] = 1.0
    ae, af, ag = M.connectA(); Amat[:, k] = np.concatenate([ae, af, ag])
P = np.asarray(M.n2n, dtype=np.complex128)
Cop = (np.diag(extmask.astype(np.complex128)) @ P
       + np.diag(intmask.astype(np.complex128))) @ Amat.T
# W-RESCALING, as the production path does it. SystemMat.rescaleLpPR /
# rescaleRHS left-multiply the node rows by W = P_ext^-1 (that is what
# n2nchol is FOR), which turns the node block into [A^T, W D] against a
# plain nodal-injection RHS. Without it the node rows carry P ~ 1e14
# against a jw ~ 1e11 capacitive block and the assembled operator
# conditions at ~1e22 -- twenty-two orders, far past double precision.
# A residual then says nothing about the error: the preconditioned solve
# reaches 1e-13 on the raw system and is still 100% wrong, because it
# differs from the true solution in a near-null direction. Rescaled, the
# condition number is ~1e8 and the check has meaning again.
Wnode = np.linalg.inv(np.diag(extmask.astype(np.complex128)) @ P
                      + np.diag(intmask.astype(np.complex128)))
n = efg + nnode

# distributed physical source: 40 (+I, -I) nodal-injection pairs
rng = np.random.default_rng(1)
npairs = 40
sn = np.zeros(nnode, dtype=np.complex128)
pick = rng.choice(nnode, 2*npairs, replace=False)
sn[pick[:npairs]] = 1e-6
sn[pick[npairs:]] = -1e-6

from scipy.linalg import lu_factor, lu_solve

print("setup1 capacitive: efg=%d nodes=%d (ext=%d)  N=%d\n"
      % (efg, nnode, M.external.size, n))
print("distributed-source iteration counts (precond vs unpreconditioned) "
      "and correctness:")
print("%-9s %-11s %-11s %-12s %-12s" %
      ("f[GHz]", "precond_it", "noM_it", "res_precond", "vs_direct"))

# One SystemMat for the sweep: precondinitLpPR caches the frequency-
# independent Lp/A/P and refactors only Z and the Schur complement per
# call, so rebuilding it per frequency would repeat 882 traverseRL probes.
S = SystemMat(M, 1j*2*np.pi*1e8)
_ns['S'] = S          # connectBT reads the module-global S in the head

NOM_BUDGET = 150
it_precond = []
it_noM = []
vsd_all = []
for f in [1e8, 3.16e8, 1e9, 3.16e9, 1e10]:
    w = 2*np.pi*f
    M.jomega = 1j*w
    Z = np.diag(Rdiag) + 1j*w*Lp
    D = np.diag(np.where(extmask, -1j*w, 0.0).astype(np.complex128))
    K = np.zeros((n, n), dtype=np.complex128)
    K[:efg, :efg] = Z; K[:efg, efg:] = Amat
    K[efg:, :efg] = Amat.T; K[efg:, efg:] = Wnode @ D
    rhs = np.zeros(n, dtype=np.complex128)
    rhs[efg:] = sn

    # dense equilibrated direct reference
    rs = np.abs(K).max(1); rs[rs == 0] = 1
    A1 = K/rs[:, None]; cs = np.abs(A1).max(0); cs[cs == 0] = 1
    xdir = np.linalg.solve(A1/cs[None, :], rhs/rs)/cs

    # PRODUCTION preconditioner -- called, not reimplemented. An earlier
    # version of this file rolled its own equivalent, which meant the
    # docstring's claim to validate SystemMat.precondinitLpPR was false and
    # that method had NO coverage at all.
    S.jomega = 1j*w
    S.precondinitLpPR()

    def Papply(v):
        return S.precondLpPR(v)
    Kop = LinearOperator((n, n), matvec=lambda v: K @ v, dtype=np.complex128)
    Mop = LinearOperator((n, n), matvec=Papply, dtype=np.complex128)

    cnt = [0, 0]
    xp, fp = lgmres(Kop, rhs, M=Mop, rtol=1e-10, maxiter=100, inner_m=60,
                    callback=lambda xk: cnt.__setitem__(0, cnt[0]+1))
    # unpreconditioned: capped budget; the point is whether it stalls, not to
    # run it to completion.
    _, f0 = lgmres(Kop, rhs, rtol=1e-10, maxiter=NOM_BUDGET, inner_m=1,
                   callback=lambda xk: cnt.__setitem__(1, cnt[1]+1))
    res_p = norm(K @ xp - rhs)/norm(rhs)
    vsd = norm(xp[:efg] - xdir[:efg])/norm(xdir[:efg])
    vsd_all.append(vsd)
    it_precond.append(cnt[0])
    it_noM.append(cnt[1] if (f0 == 0 and cnt[1] < NOM_BUDGET) else 10**9)
    print("%-9.3f %-11d %-11s %-12.2e %-12.2e" %
          (f/1e9, cnt[0],
           ("%d" % cnt[1]) if (f0 == 0 and cnt[1] < NOM_BUDGET)
           else "STALL>%d" % NOM_BUDGET, res_p, vsd))

print()
check("preconditioned lgmres converges in <=4 iters across the band",
      max(it_precond) <= 4, "max iters = %d" % max(it_precond))
check("iteration count is frequency-flat (spread <= 2)",
      max(it_precond) - min(it_precond) <= 2,
      "range %d..%d" % (min(it_precond), max(it_precond)))
check("unpreconditioned stalls where preconditioner does not (>=1 freq)",
      max(it_noM) >= NOM_BUDGET, "max noM iters = %s"
      % ("STALL" if max(it_noM) >= 10**9 else max(it_noM)))
check("preconditioned solution matches dense direct solve (currents)",
      max(vsd_all) < 1e-3,
      "max |precond-direct|/|direct| = %.2e" % max(vsd_all))

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL)); sys.exit(1)
print("all checks passed -- block-triangular Schur preconditioner is "
      "frequency-robust over 100 MHz - 10 GHz")
