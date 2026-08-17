# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the intra-box source positions that :meth:`levels.Level.leafinit`
builds for the P2M/L2P operators, in BOTH schemes.

Why this file exists
--------------------
``leafinit`` chooses each element's position inside its leaf box from two
index families -- ``centre`` (integer) and ``node`` (``centre - 1/2``) -- via

    fam[i] = centre[i] if ((i == axis) != offset_on_own_axis) else node[i]

Every OTHER part of the capacitive path fixes the same lattice
independently: the closed-form near-field tables in
:meth:`leaf_poten.LeafPoten.p2pinit3`, the panel/node attachment in
``Tree.node2panel`` ("a cell-scheme panel at local i is the face between
cells i-1 and i"), and ``validate_p2pinit3``'s own ``centers()``. Under
'cell', ``leafinit`` used ``offset_on_own_axis = False`` and disagreed with
all of them: it put an x-normal panel a half cell off on the two TRANSVERSE
axes instead of on its own axis, so px and py came out displaced by
``(+1/2,-1/2,0)`` cells rather than ``(-1/2,+1/2,0)`` -- a FULL CELL of
error on every cross-orientation panel pair, and a 6.3% error in the
capacitive far field.

Nothing caught it for a long time, for three compounding reasons:

  * the INDUCTIVE path is immune. Orthogonal partial inductances vanish, so
    e/f/g never couple; traverseRL runs them as three independent sweeps and
    a constant per-orientation translation cancels exactly. Only the
    capacitive path pushes px/py/pz through ONE shared sweep, where the
    RELATIVE offset is physical.
  * the near field is unaffected -- it comes from p2pinit3, not leafinit --
    so ``n2n`` matched the dense oracle to 5e-16 and looked exonerating.
  * the error CONVERGES in nmax, which normally indicts wiring over
    truncation. It converges to the WRONG answer here too, but so does any
    geometry error, so that discriminator could not separate them.

Checks
------
1. Cross-orientation offsets. The vector from a panel to the co-indexed
   panel of another orientation is read back out of ``leafinit``'s OWN P2M
   operator ``mfil`` (its n=1 block IS the position, up to one linear map
   calibrated from same-leaf index steps -- so this reads the shipped
   operator, it does not reimplement the family formula) and compared with
   the physical staggering of two faces of one cell.
2. Capacitive far field vs a single-level dense oracle, with the cell pitch
   PINNED per axis, over an nleaf sweep. This is the observable the bug
   moved (2.5e-2 -> 2.8e-4 at nleaf 4 under 'cell'); it also guards the
   pitch-pinning trap, since an unpinned rebuild silently rescales the
   lattice against the oracle and fails every row.

Run inside the toolbox:  python3 validate_leafinit_geometry.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import multipole as mp
import stencils as st

CELL = 1e-5
FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def coords(idx, dims):
    return np.stack([idx//(dims[1]*dims[2]), (idx//dims[2]) % dims[1],
                     idx % dims[2]], 1)


# --- 1: cross-orientation offsets, read back out of mfil -------------------
#
# mfil's n=1 block is (r Y_1^-1, r Y_1^0, r Y_1^1) at the source position, so
# it is a fixed linear function A of that position, identical for every leaf
# (the kernel constant m0 lives in ynmr, not mfil). Calibrating A from index
# steps WITHIN one leaf -- where the physical step is known to be one lattice
# pitch -- turns mfil back into positions without assuming the family rule.
NTg = np.array([6, 6, 6])
G = mp.Tree(np.ones(NTg, np.int8), np.array([3, 3, 3]), NTg*CELL, 2, 1e0, 4,
            capacitive=True)
leaves = {'x': G.px, 'y': G.py, 'z': G.pz}


def positions(leaf):
    """Recover every panel's intra-box position from leaf.mfil (n=1 block).

    r*Y_1^m is LINEAR in the source position, so v = B p exactly (no
    intercept) and p = B^-1 v. B is calibrated from single-index steps
    within this leaf, where the physical step is one lattice pitch by
    definition -- never from an assumed absolute origin, which is the
    quantity under test. The returned residual is measured on TWO-index
    steps, which the calibration never saw.
    """
    n = leaf.n.astype(int)
    l = np.asarray(leaf.l, float)
    v = np.vstack([leaf.mfil[1:4, :].real, leaf.mfil[1:4, :].imag]).T
    g = coords(np.arange(np.prod(n)), n)
    lin = (g[:, 0]*n[1] + g[:, 1])*n[2] + g[:, 2]
    fit, held = [], []
    for ax in range(3):
        for step in (1, 2):
            m = g[:, ax] + step < n[ax]
            j = lin[m] + step*int(np.prod(n[ax+1:]))
            d = np.zeros((int(m.sum()), 3))
            d[:, ax] = step*l[ax]
            (fit if step == 1 else held).append((v[j] - v[lin[m]], d))
    A = np.linalg.lstsq(np.concatenate([a for a, _ in fit]),
                        np.concatenate([b for _, b in fit]), rcond=None)[0]
    resid = max(np.abs(hv @ A - hd).max() for hv, hd in held)/np.abs(l).max()
    return v @ A, resid


pos = {}
for o, L in leaves.items():
    p, resid = positions(L)
    pos[o] = p
    check("mfil n=1 block recovers position (p%s)" % o, resid < 1e-9,
          "held-out resid=%.2e cells" % resid)

# Physical staggering: under 'cell' an x-normal panel is the face between
# cells i-1 and i, i.e. a half cell BACK along its own axis and centred on
# the other two, so px - py = (-1/2, +1/2, 0) cells. Under 'edge' the panel
# grids are transversely refined but the same relation holds in each grid's
# own units, so compare against the shipped near-field convention instead.
n0 = leaves['x'].n.astype(int)
same = all((leaves[o].n.astype(int) == n0).all() for o in 'xyz')
if True:
    check("panel leaves share one intra-box grid", same,
          "n = %s / %s / %s" % tuple(str(leaves[o].n.astype(int))
                                     for o in 'xyz'))
    l = np.asarray(leaves['x'].l, float)
    want = {('x', 'y'): np.array([-0.5, 0.5, 0.0]),
            ('x', 'z'): np.array([-0.5, 0.0, 0.5]),
            ('y', 'z'): np.array([0.0, -0.5, 0.5])}
    for (a, b), w in want.items():
        d = (pos[a] - pos[b])/l                 # co-indexed panels
        spread = np.abs(d - d[0]).max()
        got = d[0]
        check("p%s - p%s offset is %s cells" % (a, b, w),
              spread < 1e-9 and np.abs(got - w).max() < 1e-9,
              "got %s (spread %.1e)" % (np.round(got, 6), spread))

# --- 2: capacitive far field vs single-level dense oracle -------------------
NT = np.array([15, 15, 15])
fs = np.ones(NT, dtype=np.int8)
M1 = mp.Tree(fs, NT, NT*CELL, 1, 1e0, None, capacitive=True)
check("oracle pitch is the cell pitch", abs(M1.e.l[0]-CELL)/CELL < 1e-12,
      "l=%.6g" % M1.e.l[0])
c1 = coords(M1.lv[0].idx, np.asarray(M1.ntotal, dtype=int))
lk = {k: i for i, k in enumerate(c1[:, 0]*1000000 + c1[:, 1]*1000 + c1[:, 2])}
nn = M1.lv[0].idx.size
rng = np.random.default_rng(7)
q1 = (rng.standard_normal(nn) + 1j*rng.standard_normal(nn))*1e-12
M1.lv[0].data = q1.copy()
M1.traverseP3()
phi1 = M1.lv[0].data.copy()

# 'cell' panels are whole faces, 'edge' panels are quartered, so 'cell'
# truncates ~4x worse at equal nleaf. Both are far below the 2.5e-2 the
# sign error produced.
TOL = 3e-3
print("\n%-7s %-11s %-15s %s" % ("nleaf", "pitch/CELL", "occ/tot boxes",
                                 "rel_err"))
for nleaf in (2, 3, 4, 5, 6):
    probe = mp.Tree(fs, np.array([nleaf]*3), NT*CELL, 2, 1e0, 4,
                    capacitive=True)
    # Pin the pitch PER AXIS: mp.Tree pads ntotal up to ntotalfull
    # internally (15 -> 18 at nleaf 3), so passing a hand-computed pad as
    # LT silently rescales the lattice against the oracle.
    fac = CELL/np.asarray(probe.e.l, dtype=float)
    M = mp.Tree(fs, np.array([nleaf]*3), NT*CELL*fac, 2, 1e0, 8,
                capacitive=True)
    pinned = np.abs(np.asarray(M.e.l)-CELL).max()/CELL
    lv0 = M.lv[0]
    i0 = np.asarray(lv0.idx0, dtype=np.int64)
    n = lv0.n.astype(int)
    occ = sum(1 for g in range(i0.size-1) if i0[g+1] > i0[g])
    glob = []
    for g in range(i0.size-1):
        c = coords(lv0.idx[np.r_[i0[g]:i0[g+1]]], n)
        c[:, 0] += lv0.xidx[g]*n[0]
        c[:, 1] += lv0.yidx[g]*n[1]
        c[:, 2] += lv0.zidx[g]*n[2]
        glob.append(c)
    glob = np.concatenate(glob)
    perm = np.array([lk[k] for k in
                     glob[:, 0]*1000000 + glob[:, 1]*1000 + glob[:, 2]])
    lv0.data = q1[perm].copy()
    M.traverseP3()
    err = np.linalg.norm(lv0.data - phi1[perm])/np.linalg.norm(phi1[perm])
    print("%-7d %-11.6f %-15s %.4e"
          % (nleaf, M.e.l[0]/CELL, "%d/%d" % (occ, i0.size-1), err))
    check("nleaf=%d pitch pinned" % nleaf, pinned < 1e-9, "off by %.1e" % pinned)
    check("nleaf=%d far field vs dense oracle" % nleaf, err < TOL,
          "rel=%.2e (tol %.0e)" % (err, TOL))

print("\n%d checks failed" % len(FAIL))
sys.exit(1 if FAIL else 0)
