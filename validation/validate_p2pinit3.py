# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validation suite for the explicit capacitive near field (p2pinit3/n2n).

Three independent checks of the panel-panel coefficient assembly:

1. Block invariants (single level, asymmetric geometry): same-orientation
   blocks must be exactly symmetric, and cross-orientation blocks must be
   reciprocal (``P[s->t] == P[t->s].T``).
2. Physics: for well-separated panel pairs the coefficient must approach
   the point-charge law, ``P_ij * 4*pi*eps0*r -> 1``.
3. Cross-level oracle: a geometry small enough that every leaf box lies in
   every other box's 27-neighborhood is built both as a single-level tree
   (near field == everything) and as a 2-level tree; the two node-to-node
   matrices ``n2n`` must agree to machine precision after mapping the
   multilevel (group, local) node ordering to global coordinates.

Run inside the toolbox:  python3 validate_p2pinit3.py
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

FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def relerr(A, B):
    return np.abs(A - B).max() / np.abs(A).max()


def coords(idx, dims):
    return np.stack([idx // (dims[1]*dims[2]),
                     (idx // dims[2]) % dims[1], idx % dims[2]], 1)


# --- 1 & 2: single level, asymmetric brick with a notch ---------------------
NT = np.array([7, 6, 4])
CELL = 12.5e-6
fullstruc = np.ones(NT, dtype=np.int8)
fullstruc[4:, 3:, :] = 0
M = mp.Tree(fullstruc, st.single_level_nleaf(fullstruc.shape),
            NT*CELL, 1, 1e0, None, capacitive=True)

blocks = {}
for name, leaf in [('x', M.px), ('y', M.py), ('z', M.pz)]:
    bx, by, bz = leaf.p2pinit3(M.px, M.py, M.pz)
    blocks[name+'x'], blocks[name+'y'], blocks[name+'z'] = bx, by, bz

for s in 'xyz':
    e = relerr(blocks[s+s], blocks[s+s].T.tocsr())
    check(f"symmetry P{s}{s}", e < 1e-12, f"err={e:.2e}")
for s, t in [('x', 'y'), ('x', 'z'), ('y', 'z')]:
    e = relerr(blocks[s+t], blocks[t+s].T.tocsr())
    check(f"reciprocity P{s}{t}", e < 1e-9, f"err={e:.2e}")

mu0 = 4*np.pi*1e-7
eps0 = 1/(mu0 * 299792458**2)


def centers(leaf, orient):
    c = coords(leaf.idx, leaf.n.astype(int)).astype(float)
    o = {'x': 0, 'y': 1, 'z': 2}[orient]
    for ax in range(3):
        if ax != o:
            c[:, ax] += 0.5
        c[:, ax] *= leaf.l[ax]
    return c


pos = {'x': centers(M.px, 'x'), 'y': centers(M.py, 'y'),
       'z': centers(M.pz, 'z')}
for s, t in [('x', 'x'), ('x', 'y'), ('x', 'z'), ('y', 'z')]:
    A = blocks[s+t].tocsr()
    ii, jj = A.nonzero()
    r = np.linalg.norm(pos[s][ii] - pos[t][jj], axis=1)
    far = r > 8*CELL
    ratio = np.asarray(A[ii[far], jj[far]]).ravel() * 4*np.pi*eps0 * r[far]
    dev = np.abs(ratio - 1).max()
    check(f"far field P{s}{t}", dev < 0.02, f"max |ratio-1|={dev:.4f}")

next_ = np.asarray(M.n2n)[M.external][:, M.external]
w = np.linalg.eigvalsh(next_)
check("n2n[external] SPD (single level)", w.min() > 0,
      f"eig min={w.min():.3e}")

# --- 3: cross-level oracle --------------------------------------------------
CELL = 1e-5
fullstruc = np.ones((3, 3, 2), dtype=np.int8)
fullstruc[2, 2, :] = 0
fullstruc[0, 2, 1] = 0
NT = np.array(fullstruc.shape)
M1 = mp.Tree(fullstruc, st.single_level_nleaf(fullstruc.shape),
             NT*CELL, 1, 1e0, None, capacitive=True)
M3 = mp.Tree(fullstruc, np.array([2, 2, 2]), np.array([4, 4, 4])*CELL,
             2, 1e0, 2, capacitive=True)

n = M3.lv[0].n.astype(int)
glob = []
for g in range(np.size(M3.lv[0].idx0) - 1):
    fidx = np.r_[M3.lv[0].idx0[g]:M3.lv[0].idx0[g+1]]
    c = coords(M3.lv[0].idx[fidx], n)
    c[:, 0] += M3.lv[0].xidx[g]*n[0]
    c[:, 1] += M3.lv[0].yidx[g]*n[1]
    c[:, 2] += M3.lv[0].zidx[g]*n[2]
    glob.append(c)
glob = np.concatenate(glob)
# node-grid dims are the tree's own: NT under
# 'cell'. Ask the tree rather than assuming either.
c1 = coords(M1.lv[0].idx, M1.ntotal)
key3 = glob[:, 0]*10000 + glob[:, 1]*100 + glob[:, 2]
key1 = c1[:, 0]*10000 + c1[:, 1]*100 + c1[:, 2]
check("node sets match", np.array_equal(np.sort(key3), np.sort(key1)))
lookup = {k: i for i, k in enumerate(key1)}
perm = np.array([lookup[k] for k in key3])
err = relerr(M3.n2n.tocsr().toarray(), np.asarray(M1.n2n)[perm][:, perm])
check("n2n multilevel == single-level", err < 1e-12, f"rel err={err:.2e}")
check("external sets match",
      set(key3[M3.external]) == set(key1[M1.external]))
check("n2nchol factorized (multilevel)", M3.n2nchol is not None)

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL))
    sys.exit(1)
print("all checks passed")
