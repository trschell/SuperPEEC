# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validation of the capacitive near-field FFT path (leaf_poten.p2pinit /
leaf_poten.p2p) against the explicit p2pinit3 assembly.

Both paths read the same closed-form gen2_p_parz / gen_p_per tables, so
agreement must be at machine precision (~1e-15), not truncation level.
The check is reported per (source, target) orientation pair, because the
parallel blocks (t == s, gen2_p_parz, plain full-cell lattice) and the
cross blocks (t != s, gen_p_per, common half-unit lattice with one grid
upsampled) fail independently.

Convention (tree.py): p2pinit3 returns P[source, target], so the
potential on targets is  phi_t = P[s,t].T @ q_s , which is what
leaf.p2p(px, py, pz, tx, ty, tz) accumulates into tx/ty/tz.

Several geometries are exercised on purpose. The circulant and workspace
sizing depends on how a target's extent along an axis relates to the
source's, and a cubic domain with a cubic leaf satisfies several of those
relations by accident.

Run inside the toolbox:  python3 validate_leaf_poten_p2p.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import traceback
import numpy as np
import multipole as mp

FAIL = []
ORI = ['x', 'y', 'z']
CELL = 1e-5


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail
                                                   else ""), flush=True)
    if not ok:
        FAIL.append(name)


def case_cubic():
    f = np.ones((4, 4, 4), np.int8)
    f[3, 3, 3] = 0
    return "cubic 4^3, leaf 2", f, [2, 2, 2], [4, 4, 4]


def case_aniso():
    f = np.ones((6, 4, 4), np.int8)
    f[5, 3, 3] = 0
    f[0, 0, 3] = 0
    return "anisotropic 6x4x4, leaf 2", f, [2, 2, 2], [6, 4, 4]


def case_bigleaf():
    f = np.ones((8, 8, 4), np.int8)
    f[7, 7, 3] = 0
    return "8x8x4, leaf 4", f, [4, 4, 4], [8, 8, 4]


for maker in (case_cubic, case_aniso, case_bigleaf):
    tag, fullstruc, N, LT = maker()
    M = mp.Tree(fullstruc, np.array(N), np.array(LT)*CELL, 2, 1e0, 2,
                capacitive=True)
    leaves = {'x': M.px, 'y': M.py, 'z': M.pz}
    print("\n=== %s ===" % tag)
    print("  px.n=%s  groups=%d  groups/slab=%s"
          % (list(M.px.n.astype(int)), np.size(M.px.idx0) - 1,
             list(np.diff(np.asarray(M.px.slabidx0)))), flush=True)

    ref = {}
    for s in ORI:
        for t, B in zip(ORI, leaves[s].p2pinit3(M.px, M.py, M.pz)):
            ref[s+t] = B.tocsr()

    init_ok = {}
    for s in ORI:
        try:
            leaves[s].p2pinit(M.px, M.py, M.pz)
            init_ok[s] = True
        except Exception as exc:
            init_ok[s] = False
            check("%s: p2pinit(%s)" % (tag, s), False,
                  "%s: %s" % (type(exc).__name__, exc))
            traceback.print_exc()

    rng = np.random.default_rng(0)
    for s in ORI:
        if not init_ok.get(s):
            continue
        q = rng.standard_normal(np.size(leaves[s].idx)) \
            + 1j*rng.standard_normal(np.size(leaves[s].idx))
        saved = leaves[s].data
        leaves[s].data = q.copy()
        out = {t: np.zeros(np.size(leaves[t].idx), np.complex128)
               for t in ORI}
        try:
            leaves[s].p2p(M.px, M.py, M.pz, out['x'], out['y'], out['z'])
        except Exception as exc:
            check("%s: p2p(%s) sweep" % (tag, s), False,
                  "%s: %s" % (type(exc).__name__, exc))
            traceback.print_exc()
            leaves[s].data = saved
            continue
        leaves[s].data = saved
        for t in ORI:
            want = ref[s+t].T.dot(q)
            scale = np.abs(want).max()
            if scale == 0:
                check("%s: %s->%s" % (tag, s, t), False, "reference is 0")
                continue
            err = np.abs(out[t] - want).max()/scale
            kind = "parallel" if s == t else "cross   "
            check("%s: %s->%s (%s)" % (tag, s, t, kind), err < 1e-12,
                  "rel err=%.3e" % err)

print("\n" + ("ALL PASS" if not FAIL else "FAILURES: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
