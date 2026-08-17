# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""End-to-end MULTILEVEL LpPR solve validation (production code paths).

The last integration step: every operator was validated individually --
mixed-row matvecLpPR, the near-field n2n, the traverseP3 multilevel far
field (validate_traverseP3_farfield.py, incl. the midinit m2m z-frame fix),
the W = P_ext^{-1} rescaling via the sparse cholmod n2nchol, and the
reluctance block-triangular preconditioner. This file solves the SAME
physical problem end to end on trees of different depth and demands the same
currents:

  oracle   : single-level LpPR solve (numlevels=1; n2n is the complete dense
             P; this configuration is validated against a dense direct solve
             by validate_lppr_reluctance.py);
  subjects : numlevels=2 / nleaf=4 (top-level FFT M2L far field, ~40% of P)
             numlevels=3 / nleaf=2 (mid-level M2M/M2L/L2L + top far field)

on a 15^3-cell copper cube with a distributed 40-pair nodal source, solved
by the PRODUCTION path exec'd straight out of main.py: SystemMat +
reluctanceprecinit (K~ extracted from the multilevel tree -- exercising the
group-offset filament_geometry -- and P_ext^{-1} through the cholmod
n2nchol at multilevel) under standard GMRES.

Checks:
  1. the reluctance K~ extracted from a multilevel tree is IDENTICAL to the
     single-level extraction (same physical filaments, same kernel tables);
  2. all three solves converge, with multilevel iteration counts comparable
     to the single-level count;
  3. the multilevel currents match the single-level currents to the FMM
     truncation level (the operators differ by ~2.5e-4 at nmax=4).

Run inside the toolbox:  python3 validate_lppr_multilevel.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import stencils as st

FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# ---- production SystemMat straight from main.py
with open(_op.path.join(_op.path.dirname(_op.path.abspath(__file__)), 'main.py')) as fh:
    src = fh.read()
head = src[:src.index('\nconductivity = 5.81e7')]
ns = {}
exec(compile(head, '<main.py head>', 'exec'), ns)
mp = ns['mp']
LinearOperator = ns['LinearOperator']
gmres = ns['gmres']
rel = ns['rel']

conductivity = 5.81e7
CELL = 1e-5
NT = np.array([15, 15, 15])
fullstruc = np.ones(NT, dtype=np.int8)
LTpad = np.array([16, 16, 16])*CELL


def coords(idx, dims):
    return np.stack([idx // (dims[1]*dims[2]),
                     (idx // dims[2]) % dims[1], idx % dims[2]], 1)


def leafglob(M, leaf):
    """Global grid coordinates of a leaf's elements (any numlevels)."""
    n = leaf.n.astype(int)
    c = coords(leaf.idx, n)
    if M.numlevels > 1:
        bn = M.lv[0].n.astype(int)
        for g in range(np.size(leaf.idx0) - 1):
            sl = np.s_[leaf.idx0[g]:leaf.idx0[g+1]]
            c[sl, 0] += M.lv[0].xidx[g]*bn[0]
            c[sl, 1] += M.lv[0].yidx[g]*bn[1]
            c[sl, 2] += M.lv[0].zidx[g]*bn[2]
    return c


def key3(c):
    return c[:, 0]*100000 + c[:, 1]*1000 + c[:, 2]


def build(nleaf, numlevels, nmax):
    """Tree + SystemMat via the production constructor conventions."""
    if numlevels == 1:
        M = mp.Tree(fullstruc, st.single_level_nleaf(NT), NT*CELL, 1,
                    1e0, nmax, capacitive=True)
    else:
        M = mp.Tree(fullstruc, np.array([nleaf]*3), LTpad, numlevels, 1e0,
                    nmax, capacitive=True)
    M.e.r = M.e.l[1]/(M.e.l[0]*M.e.l[2]*conductivity)
    M.f.r = M.f.l[0]/(M.f.l[1]*M.f.l[2]*conductivity)
    M.g.r = M.g.l[2]/(M.g.l[0]*M.g.l[1]*conductivity)
    S = ns['SystemMat'](M, 1j)
    S.M.lv[0].neumann = False
    S.M.alpha = 0
    M.RDFinit()
    return M, S


def maps_vs(Ms, Mm):
    """Node and filament permutations mapping Mm's ordering into Ms's."""
    nk = {k: i for i, k in enumerate(key3(coords(Ms.lv[0].idx,
                                                 Ms.lv[0].n.astype(int))))}
    cm = leafglob(Mm, Mm.lv[0]) if Mm.numlevels > 1 else \
        coords(Mm.lv[0].idx, Mm.lv[0].n.astype(int))
    nperm = np.array([nk[k] for k in key3(cm)])
    fperm = []
    for ls, lm, off in [(Ms.e, Mm.e, 0), (Ms.f, Mm.f, Ms.e.struc.size),
                        (Ms.g, Mm.g, Ms.e.struc.size + Ms.f.struc.size)]:
        fk = {k: i for i, k in enumerate(key3(leafglob(Ms, ls)))}
        fperm.append(off + np.array([fk[k] for k in key3(leafglob(Mm, lm))]))
    return nperm, np.concatenate(fperm)


def solve(S, Source, jw):
    # tol=1e-10, tighter than the production 1e-8: the block-triangular
    # preconditioner's S~^{-1} SHRINKS residual components along the stiff
    # nodal-Laplacian directions (scale |A^T K~ A|/w) in the preconditioned
    # norm, so a loose preconditioned tolerance under-resolves the pure-KCL
    # (internal node) rows -- worst at low frequency, where 1/w is stiffest
    # (measured at 1 GHz: true residual 2.7e-3 at tol 1e-8 vs 5.9e-5 at
    # 1e-10 for ~1.3x the matvecs). Truncation-level cross-comparisons need
    # the tighter setting. This tolerance also serves as a FIXEDNESS canary
    # for the S~ backend: a trial UMFPACK backend whose default adaptive
    # iterative refinement (a data-dependent stopping test inside the
    # apply) made the preconditioner a VARYING operator stagnated standard
    # GMRES near 1e-9 and was caught exactly here (2026-07-28; backend
    # since reverted -- single-threaded dead end). Any future S~ backend
    # must keep the apply strictly linear and fixed to pass this file.
    S.jomega = jw
    S.M.jomega = jw
    if not hasattr(S, '_Ktilde'):
        S.reluctanceprecinit()          # one-time build (K~, A, AtKA, Ccap)
    else:
        S._factorReluctanceSchur()      # per-frequency S~ refactor
    Kprime = LinearOperator((S.wholesize,)*2, matvec=S.rescaleLpPR,
                            dtype=np.complex128)
    PRel = LinearOperator((S.wholesize,)*2, matvec=S.precondReluctance,
                          dtype=np.complex128)
    Sourcep = S.rescaleRHS(Source)
    n0 = S.numiters
    x, flag = gmres(Kprime, Sourcep, M=PRel, tol=1e-10, maxiter=1,
                    restrt=300)
    return np.asarray(x).ravel(), flag, S.numiters - n0


def make_source(S, sn):
    """Mixed-row RHS exactly as the main.py driver builds it."""
    Source = np.zeros(S.wholesize, np.complex128)
    Source[S.efgsize:] = sn
    S.M.lv[0].data[:] = Source[S.efgsize:]
    S.M.traverseP3()
    Source[S.efgsize:][S.extmask] = S.M.lv[0].data[S.extmask]
    S.wholedata[:] = 0
    return Source


# ---- build the configurations (nmax=6 duplicate discriminates truncation
# from wiring: the current mismatch must SHRINK with expansion order)
# Deepest-tree leaf size. The
# 27-box near field at nleaf 2 (and 3) truncates the coefficient-of-
# potential operator so hard that n2n[ext] goes INDEFINITE -- measured
# lambda_min at 15^3: nleaf 2 -3.49e14, 3 -4.44e14, 4 +1.34e13, 5
# +2.83e14 -- and the preconditioner consumers now refuse to build from
# it (they used to fall back to a dense LU, so this case silently
# "passed" on an indefinite P_ext). 4 is the smallest definite leaf here.
NLEAF3 = 4
TAG3 = "nl=3 nleaf=%d" % NLEAF3

Ms, Ss = build(None, 1, None)
M2, S2 = build(4, 2, 4)
M3, S3 = build(NLEAF3, 3, 4)
M3b, S3b = build(NLEAF3, 3, 6)
np2, fp2 = maps_vs(Ms, M2)
np3, fp3 = maps_vs(Ms, M3)
print("single: efg=%d nn=%d ext=%d | nl=2: ext=%d | nl=3: ext=%d\n"
      % (Ss.efgsize, Ss.nodesize, Ms.external.size, M2.external.size,
         M3.external.size))

# ---- check 1: K~ extraction identical across tree depths
Ks = rel.extract_reluctance(Ms)
K2 = rel.extract_reluctance(M2, verbose=True)
K2p = K2[fp2.argsort()][:, fp2.argsort()]      # reorder into single ordering
# fp maps multilevel index -> single index; rows of K2 are multilevel
P2 = np.zeros_like(fp2)
P2[np.arange(fp2.size)] = fp2
from scipy.sparse import coo_matrix as _coo
Pm = _coo((np.ones(fp2.size), (fp2, np.arange(fp2.size))),
          shape=(fp2.size, fp2.size)).tocsr()
K2s = (Pm @ K2 @ Pm.T).tocsr()                 # multilevel K~ in single order
kdiff = abs(K2s - Ks).max() / abs(Ks).max()
check("K~ extraction identical on multilevel tree", kdiff < 1e-9,
      "max rel diff = %.2e" % kdiff)

# ---- distributed source (single-level node ordering)
rng = np.random.default_rng(1)
sn_s = np.zeros(Ss.nodesize, np.complex128)
pk = rng.choice(Ss.nodesize, 80, replace=False)
sn_s[pk[:40]] = 1e-6
sn_s[pk[40:]] = -1e-6

print("\n%-9s %-22s %-8s %-6s %-11s %-11s"
      % ("f[GHz]", "config", "matvecs", "flag", "vs_single", "xres_s"))
rows = []
for f in [1e9, 1e10]:
    jw = 1j*2*np.pi*f
    Src_s = make_source(Ss, sn_s)
    xs, fls, its = solve(Ss, Src_s, jw)
    Srcp_s = Ss.rescaleRHS(Src_s)
    # the single-level solve's own TRUE residual: left-preconditioned GMRES
    # converges ||M r|| to 1e-8; the true-residual floor is set by the
    # preconditioner conditioning and is the honest yardstick for how well
    # ANY solution can be expected to satisfy this system.
    xres_own = (np.linalg.norm(Ss.rescaleLpPR(xs) - Srcp_s)
                / np.linalg.norm(Srcp_s))
    print("%-9.1f %-22s %-8d %-6d %-11s %-11.2e"
          % (f/1e9, "single-level", its, fls, "--", xres_own))
    for name, S, M, nperm, fperm in [("nl=2 nleaf=4", S2, M2, np2, fp2),
                                     (TAG3, S3, M3, np3, fp3),
                                     (TAG3 + " nmax=6", S3b, M3b,
                                      np3, fp3)]:
        Src = make_source(S, sn_s[nperm])
        x, fl, it = solve(S, Src, jw)
        vsd = (np.linalg.norm(x[:S.efgsize] - xs[:Ss.efgsize][fperm])
               / np.linalg.norm(xs[:Ss.efgsize][fperm]))
        # cross-residual: how well the multilevel SOLUTION satisfies the
        # SINGLE-LEVEL system. If this sits at operator-truncation level,
        # the raw current difference above is soft-mode amplification (the
        # below-resonance near-null charge mode), not a wiring defect.
        xm_s = np.zeros_like(xs)
        xm_s[fperm] = x[:S.efgsize]
        xm_s[Ss.efgsize + nperm] = x[S.efgsize:]
        xres = (np.linalg.norm(Ss.rescaleLpPR(xm_s) - Srcp_s)
                / np.linalg.norm(Srcp_s))
        rows.append(dict(f=f, name=name, it=it, its=its, fl=fl, vsd=vsd,
                         xres=xres, xown=xres_own))
        print("%-9.1f %-22s %-8d %-6d %-11.2e %-11.2e"
              % (f/1e9, name, it, fl, vsd, xres))

print()
check("all solves converge (flag == 0)",
      all(r['fl'] == 0 for r in rows), "")
check("multilevel iteration counts comparable to single-level (<= 2x)",
      all(r['it'] <= 2*r['its'] for r in rows),
      "; ".join("%s@%gGHz %d vs %d" % (r['name'], r['f']/1e9, r['it'],
                                       r['its']) for r in rows))
worst = max(r['vsd'] for r in rows if 'nmax=6' not in r['name'])
check("multilevel currents match single-level (< 2e-2 at nmax=4)",
      worst < 2e-2, "worst = %.2e" % worst)
v4 = next(r['vsd'] for r in rows
          if r['f'] == 1e10 and r['name'] == TAG3)
v6 = next(r['vsd'] for r in rows
          if r['f'] == 1e10 and r['name'] == TAG3 + " nmax=6")
check("current mismatch shrinks with nmax at 10 GHz (truncation, not "
      "wiring)", v6 < 0.5*v4, "nmax 4 -> 6: %.2e -> %.2e" % (v4, v6))
# The cross-residual is bounded below by the operator truncation
# (delta_K x ~ 1e-4 at nmax=6), so the criterion is that it REACHES that
# level -- proving the multilevel solution satisfies the single-level
# system as accurately as the operators agree at all.
worstx = max(r['xres'] for r in rows if 'nmax=6' in r['name'])
check("multilevel solution satisfies the single-level system to operator-"
      "truncation level (cross-residual at nmax=6 < 1e-3)", worstx < 1e-3,
      "worst = %.2e" % worstx)

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL))
    sys.exit(1)
print("all checks passed -- end-to-end multilevel LpPR solve (production "
      "SystemMat + reluctance preconditioner) matches the single-level solve")
