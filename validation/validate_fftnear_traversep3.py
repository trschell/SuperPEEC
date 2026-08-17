# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Acceptance for the FFT near field wired into tree.traverseP3.

`Tree(..., fftnear=True)` applies the 27-neighbour capacitive near field
as a Toeplitz/FFT convolution on the panels (leaf_poten.p2p) instead of
the assembled sparse node-to-node matrix. The two must produce the SAME
operator: same tables, same neighbourhood, so agreement is expected at
machine precision rather than truncation level.

The assembled n2n is still built in both cases -- W = P_ext^-1 and C_cap
need matrix entries -- so this also checks that turning the flag on does
not disturb the preconditioner-side data.

Run inside the toolbox:  python3 validate_fftnear_traversep3.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import multipole as mp

FAIL = []
CELL = 1e-5


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail
                                                   else ""), flush=True)
    if not ok:
        FAIL.append(name)


def cases():
    f = np.ones((4, 4, 4), np.int8)
    f[3, 3, 3] = 0
    yield "cubic 4^3, leaf 2", f, [2, 2, 2], [4, 4, 4]
    f = np.ones((6, 4, 4), np.int8)
    f[5, 3, 3] = 0
    f[0, 0, 3] = 0
    yield "anisotropic 6x4x4, leaf 2", f, [2, 2, 2], [6, 4, 4]
    # NOTE: leaf boxes must be >= 3 per axis. traverseP3 returns NaN for
    # ng = 2 on any axis (e.g. 8x8x4 with leaf 4 -> ng = [3,3,2]) in BOTH
    # the sparse-n2n and FFT paths -- a pre-existing far-field defect,
    # unrelated to fftnear, presumably because a 27-neighbourhood leaves
    # no far field at all when an axis has only two boxes.
    f = np.ones((8, 8, 8), np.int8)
    f[7, 7, 7] = 0
    yield "8x8x8, leaf 4", f, [4, 4, 4], [8, 8, 8]


for tag, fullstruc, N, LT in cases():
    ref = mp.Tree(fullstruc, np.array(N), np.array(LT)*CELL, 2, 1e0, 2,
                  capacitive=True)
    fft = mp.Tree(fullstruc, np.array(N), np.array(LT)*CELL, 2, 1e0, 2,
                  capacitive=True, fftnear=True)
    check("%s: fftnear flag set" % tag, getattr(fft, 'fftnear', False))
    # the assembled n2n must be unchanged -- the preconditioner still uses it
    d = np.abs(np.asarray((ref.n2n - fft.n2n).todense())).max() \
        / np.abs(np.asarray(ref.n2n.todense())).max()
    check("%s: n2n still assembled and identical" % tag, d == 0.0,
          "max diff=%.1e" % d)

    rng = np.random.default_rng(4)
    q = rng.standard_normal(np.size(ref.lv[0].idx)) \
        + 1j*rng.standard_normal(np.size(ref.lv[0].idx))
    outs = []
    for M in (ref, fft):
        M.lv[0].data = q.copy()
        M.traverseP3()
        outs.append(np.asarray(M.lv[0].data).copy())
    err = np.abs(outs[1] - outs[0]).max()/np.abs(outs[0]).max()
    check("%s: traverseP3 fftnear == sparse n2n" % tag, err < 1e-12,
          "rel err=%.3e" % err)

print("\n" + ("ALL PASS" if not FAIL else "FAILURES: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
