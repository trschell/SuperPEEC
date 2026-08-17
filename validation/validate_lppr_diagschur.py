# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the diagonal-admittance Schur (diagschur) preconditioner for the
mixed-row LpPR solve under STANDARD GMRES (setup1 geometry, 100 kHz - 1 GHz).

Motivation. The low-frequency (resistive, wL/R <~ 1) band was owned by the
loop-reduction preconditioner, which is ITERATIVE (an inner lsqr + lgmres
saddle solve per apply) and therefore forces flexible FGMRES. PyPEEC solves
the same physics with a purely SPARSE preconditioner: diagonalize the dense
inductance matrix (keep only self partial inductances) and the dense
potential matrix, after which the branch-impedance block of the MNA is
diagonal and the Schur complement onto the nodes is EXACT and sparse. This
file validates that mathematical framework transplanted onto SuperPEEC's
W-rescaled mixed-row system K' = [[Z, A], [A^T, -jw C_cap]]:

    D   = R + jw diag(Lp)                    (diagonal branch impedance)
    S_d = -jw C_cap - A^T D^{-1} A           (7-point nodal admittance Laplacian)
    x_v = S_d^{-1} (r_v - A^T D^{-1} r_i);  x_i = D^{-1} (r_i - A x_v)

-- the exact block LU of the sparsified MNA [[D, A], [A^T, -jw C_cap]], a
FIXED operator, so ordinary GMRES applies. Valid where R dominates the
neglected mutual coupling; complements the reluctance preconditioner
(high-f) from the opposite end of the band.

Checks (production code, exec'd straight out of main.py):
  1. the three-probe diag(Lp) (translation-invariant self partial
     inductances) matches the dense-probed diag(Lp) on every filament;
  2. the preconditioner is the EXACT inverse of the diagonalized MNA:
     the dense apply of [[D, A], [A^T, -jw C_cap]] to M^{-1}v recovers v to
     the CONDITIONING floor eps*cond(S_d) -- S_d is nearly singular toward
     DC (jw C_cap is ~1e-10 of the Laplacian weights at 100 kHz, cond
     ~3e13), so the reconstruction error legitimately grows ~cond*eps
     there (measured: err/cond(S_d) constant at ~3e-18 across the sweep; a
     sign/wiring bug would be O(1) and frequency-INDEPENDENT). The
     near-null direction is the potential gauge, which the currents never
     see -- check 4 is the proof the solve is unaffected;
  3. the production rescaleLpPR operator matches the dense-assembled K'
     (oracle wiring check);
  4. standard GMRES on K', diagschur-preconditioned, gives currents matching
     a dense equilibrated direct solve of the SAME K' at every frequency;
  5. iteration counts are low across 100 kHz - 100 MHz and do NOT grow
     toward DC (the diagschur regime);
  6. the sdamg backend (diagschurprecinit(sdsolve='amg'): fixed k-cycle
     smoothed aggregation on S_d instead of the sparse LU) engages -- S_d,
     unlike the reluctance A^T K~ A, IS an M-matrix-real-part Laplacian
     and SA contracts ~0.2/cycle -- and reproduces the splu-backed solve
     at comparable iteration count. (Honest note: it is a low-frequency
     preconditioner -- toward/above the wL/R ~ 1 crossover (~1 GHz at these
     12.5 um cells) the neglected mutual inductance dominates and the
     reluctance preconditioner takes over; the printed reluctance column
     shows the two bracketing the band.)

Run inside the toolbox:  python3 validate_lppr_diagschur.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import stencils as st
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator, gmres as spgmres
from scipy.linalg import lu_factor, lu_solve

FAIL = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)

# ---- production SystemMat straight from main.py (exec the pre-driver head)
with open(_op.path.join(_op.path.dirname(_op.path.abspath(__file__)), 'main.py')) as fh:
    src = fh.read()
head = src[:src.index('\nconductivity = 5.81e7')]
ns = {}
exec(compile(head, '<main.py head>', 'exec'), ns)
mp = ns['mp']

conductivity = 5.81e7
NT = np.array([7, 7, 7]); LT = np.array([87.5e-6]*3)
N = st.single_level_nleaf(NT)
M = mp.Tree(np.ones(NT, np.int8), N, LT, 1, 1e0, None, capacitive=True)
M.e.r = M.e.l[1]/(M.e.l[0]*M.e.l[2]*conductivity)
M.f.r = M.f.l[0]/(M.f.l[1]*M.f.l[2]*conductivity)
M.g.r = M.g.l[2]/(M.g.l[0]*M.g.l[1]*conductivity)
M.lv[0].beta = 1.0
S = ns['SystemMat'](M, 1j)
ns['S'] = S                          # connectBT reads the global S
S.M.lv[0].neumann = False
S.M.alpha = 0
M.RDFinit()
efg, nn = S.efgsize, S.nodesize
ext = M.external
extmask = np.zeros(nn, bool); extmask[ext] = True; intmask = ~extmask

# ---- dense reference operators (probed from the code, correctness only)
Rd = np.concatenate([np.full(S.esize, M.e.r), np.full(S.fsize, M.f.r),
                     np.full(S.gsize, M.g.r)]).astype(np.complex128)
M.jomega = 1j
Lp = np.zeros((efg, efg), np.complex128)
for k in range(efg):
    S.wholedata[:efg] = 0
    S.wholedata[k] = 1.0
    M.traverseRL()
    Lp[:, k] = S.wholedata[:efg]
Lp = (Lp - np.diag(Rd))/1j
Lp = 0.5*(Lp + Lp.T)
Amat = np.zeros((efg, nn), np.complex128)
for k in range(nn):
    M.lv[0].data[:] = 0
    M.lv[0].data[k] = 1.0
    a = M.connectA()
    Amat[:, k] = np.concatenate(a)
S.wholedata[:] = 0
P = np.asarray(M.n2n, np.complex128)
luPext = lu_factor(P[np.ix_(ext, ext)].real)
print("setup1-7^3: efg=%d nn=%d ext=%d\n" % (efg, nn, ext.size))

# ---- production diagschur assembly + reluctance for the comparison column
S.diagschurprecinit()
derr = np.abs(S._Lpdiag - np.real(np.diag(Lp))).max()/np.abs(np.diag(Lp)).max()
check("three-probe diag(Lp) matches dense diag(Lp) on every filament",
      derr < 1e-10, "max rel err = %.2e" % derr)
S.reluctanceprecinit()

Adense = S._Ainc.toarray().astype(np.complex128)
Ccap = np.asarray(S._Ccap)


def gmres_solve(Kop, Mop, b, tol=1e-10, maxit=300):
    h = []
    x, _ = spgmres(Kop, b, M=Mop, rtol=tol, restart=maxit, maxiter=1,
                   callback=lambda rk: h.append(rk), callback_type='pr_norm')
    h = np.asarray(h)
    return x, (int(np.argmin(h/h[0]))+1 if h.size else 0)


rng = np.random.default_rng(1)
sn = np.zeros(nn, np.complex128)
pk = rng.choice(nn, 80, replace=False)
sn[pk[:40]] = 1e-6; sn[pk[40:]] = -1e-6

print("%-9s %-11s %-11s %-11s %-13s" % ("f[MHz]", "diagschur", "resid(K')",
                                        "vs_direct", "(reluctance)"))
its = {}
opmax = exfloor = 0.0
exhi = None
eps = np.finfo(np.float64).eps
for f in [1e5, 1e6, 1e7, 1e8, 1e9]:
    w = 2*np.pi*f; jw = 1j*w
    S.jomega = jw; M.jomega = jw
    S._factorDiagSchur()
    S._factorReluctanceSchur()
    # dense K' oracle: mixed-row operator, then W-rescale the external rows
    Z = np.diag(Rd) + jw*Lp
    Kmix = np.zeros((efg+nn,)*2, np.complex128)
    Kmix[:efg, :efg] = Z; Kmix[:efg, efg:] = Amat
    Kmix[efg:, :efg] = (np.diag(extmask+0j) @ P
                        + np.diag(intmask+0j)) @ Amat.T
    Kmix[efg:, efg:] = np.diag(np.where(extmask, -jw, 0.0).astype(np.complex128))
    Kp = Kmix.copy()
    rb = Kmix[efg:][ext]
    Kp[efg:][ext] = lu_solve(luPext, rb.real) + 1j*lu_solve(luPext, rb.imag)
    # production operator vs the dense oracle (wiring check)
    vt = rng.standard_normal(efg+nn) + 1j*rng.standard_normal(efg+nn)
    opmax = max(opmax, norm(S.rescaleLpPR(vt) - Kp @ vt)/norm(Kp @ vt))
    # exactness on the diagonalized MNA: Mdiag @ (M^{-1} v) == v, judged
    # against the eps*cond(S_d) reconstruction floor (see the docstring)
    Dg = np.asarray(S._Rdiag + jw*S._Lpdiag)
    Mdiag = np.zeros((efg+nn,)*2, np.complex128)
    Mdiag[:efg, :efg] = np.diag(Dg); Mdiag[:efg, efg:] = Adense
    Mdiag[efg:, :efg] = Adense.T;    Mdiag[efg:, efg:] = np.diag(-jw*Ccap)
    exerr = norm(Mdiag @ S.precondDiagSchur(vt) - vt)/norm(vt)
    condSd = np.linalg.cond(-jw*np.diag(Ccap)
                            - Adense.T @ (Adense/Dg[:, None]))
    exfloor = max(exfloor, exerr/(eps*condSd))
    exhi = exerr                    # last sweep point = best conditioned
    # mixed-row RHS (external rows get P s_n) and its W rescale
    rhs = np.zeros(efg+nn, np.complex128)
    rhs[efg:][extmask] = (P @ sn)[extmask]; rhs[efg:][intmask] = sn[intmask]
    rhsp = S.rescaleRHS(rhs)
    # equilibrated dense direct reference on the SAME K'
    rs = np.abs(Kp).max(1); rs[rs == 0] = 1
    K1 = Kp/rs[:, None]; cs = np.abs(K1).max(0); cs[cs == 0] = 1
    xref = np.linalg.solve(K1/cs[None, :], rhsp/rs)/cs
    # production GMRES solves (diagschur asserted; reluctance informational)
    Kop = LinearOperator((efg+nn,)*2, matvec=S.rescaleLpPR, dtype=np.complex128)
    Mds = LinearOperator((efg+nn,)*2, matvec=S.precondDiagSchur,
                         dtype=np.complex128)
    Mrl = LinearOperator((efg+nn,)*2, matvec=S.precondReluctance,
                         dtype=np.complex128)
    x, it = gmres_solve(Kop, Mds, rhsp)
    _, itr = gmres_solve(Kop, Mrl, rhsp)
    resid = norm(Kp @ x - rhsp)/norm(rhsp)
    vsd = norm(x[:efg] - xref[:efg])/norm(xref[:efg])
    its[f] = it
    print("%-9.1f %-11d %-11.2e %-11.2e %-13d" % (f/1e6, it, resid, vsd, itr))
    check("currents correct at %.2g Hz (vs dense direct)" % f, vsd < 3e-3,
          "vs_direct=%.2e" % vsd)
    if f == 1e7:
        # sdamg backend: fixed k-cycle SA replaces the sparse LU of S_d;
        # must engage (contraction guard) and reproduce the splu solve
        S._sdsolve = 'amg'
        S._factorDiagSchur()
        amg_on = getattr(S._sd_solve, '__name__', '') == 'sd_amg'
        xa, ita = gmres_solve(Kop, Mds, rhsp)
        vsda = norm(xa[:efg] - xref[:efg])/norm(xref[:efg])
        check("sdamg engages at 10 MHz (SA contraction guard passed)", amg_on)
        check("sdamg-backed solve matches dense direct", vsda < 3e-3,
              "vs_direct=%.2e" % vsda)
        check("sdamg iteration cost close to splu", ita <= it + 8,
              "amg %d vs splu %d iters" % (ita, it))
        S._sdsolve = 'splu'
        S._factorDiagSchur()

print()
check("production rescaleLpPR matches dense-assembled K'", opmax < 1e-8,
      "max rel err = %.2e" % opmax)
check("exact inverse of the diagonalized MNA (block-LU apply, cond-aware)",
      exfloor < 100 and exhi < 1e-8,
      "max err/(eps*cond(S_d)) = %.1f; err at 1 GHz = %.2e" % (exfloor, exhi))
check("standard GMRES efficient at 0.1-100 MHz (<= 40 iters)",
      max(its[1e5], its[1e6], its[1e7], its[1e8]) <= 40,
      "0.1MHz=%d 100MHz=%d" % (its[1e5], its[1e8]))
check("best at low frequency (iters do NOT grow toward DC)",
      its[1e5] <= its[1e9], "0.1MHz=%d 1GHz=%d" % (its[1e5], its[1e9]))

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL)); sys.exit(1)
print("all checks passed -- diagschur (PyPEEC diagonal-admittance exact-Schur "
      "framework) solves mixed-row LpPR under STANDARD GMRES, best at low "
      "frequency")
