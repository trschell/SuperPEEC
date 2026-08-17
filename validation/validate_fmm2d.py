# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the 2-D FMM: unsplit axes in the mid-level M2L.

A FLAT geometry wants leaf boxes that SPAN its thin axis, with the
levels above interacting as a 2-D plane (``nmidlev = 1`` on that axis).
The level construction was per-axis from the start, but
``MidLevel.midm2linit`` hard-coded the 2x2x2 child geometry -- 8
parities read as ``4x+2y+z``, the 189-entry list, and child slots at
half-parent spacing -- so the first time an unsplit axis reached a mid
level (circular_coil, 2026-08-06) the far field came out ~4% high, from
three compounding errors: wrong transfer distances (2x on the unsplit
axis), wrong near/far exclusion, and scrambled parity indexing (the
scan stores MIXED-RADIX ``(x*n1+y)*n2+z``). All fixed by the per-axis
generalisation of ``midm2linit`` + the ``MID_M2L`` kernel (dims now
hidden shape arguments).

THE CHECK: on a solid pancake plate, the operator applied to the
uniform current must agree between a 2-level tree (top-level FFT M2L,
correct for flat axes since before the fix and verified against four
independent configurations) and a 3-level tree whose mid level runs
with an UNSPLIT thin axis. Disagreement beyond FMM-truncation scatter
is exactly the defect class this file pins. A [2,2,2]-path config is
compared too, so a regression that breaks the STANDARD mid level would
also be caught here (cheaply -- the full suite covers it at depth).

Run inside the toolbox:  python3 validate_fmm2d.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]


import os
import sys

import numpy as np

import vhr
import stencils as st

SPD = os.path.dirname(os.path.abspath(__file__))

fails = []


def check(tag, cond, detail=''):
    print("    %-4s %s  %s" % ('ok' if cond else 'FAIL', tag, detail))
    if not cond:
        fails.append(tag)
    return cond


def invariants(m, nleaf, levels, f=2.5e9):
    M = m.build_tree(np.array(nleaf), levels)
    whole, (esz, efsz, efgsz, _) = m.prepare(M, f)
    whole[:efgsz] = 1.0 + 0.0j
    M.traverseRL()
    out = whole[:efgsz]
    return complex(out.sum()), float(np.abs(out).sum())


def main():
    print("2-D FMM: unsplit mid-level axis vs 2-level reference")
    # solid pancake: 48 x 48 x 4 cells, thin axis z
    struc = np.ones((48, 48, 4), dtype=np.int8)
    ports = [('p1', 'P', 0, j, k, '-x') for j in range(48)
             for k in range(4)]
    ports += [('p1', 'N', 47, j, k, '+x') for j in range(48)
              for k in range(4)]
    p = os.path.join(SPD, '..', 'studies', 'fmm2d_plate.vhr')
    vhr.write_vhr(p, struc, 1e-6, 5.8e7, (2.5e9,), ports)
    m = vhr.read_vhr(p)

    s_ref, a_ref = invariants(m, [6, 6, 5], 2)     # z spanned, no mid level
    # Canary configs are chosen COMPACT: a high-aspect leaf like
    # [3,3,1] degrades multipole convergence and shows ~2e-3 honest
    # truncation scatter that would drown the defect signature (~4e-2).
    # [3,3,2] used to NaN -- misattributed at the time to the ng==2
    # family, actually the MIXED-PARITY r=0 leafinit degeneracy (an
    # element exactly at the box expansion centre), fixed 2026-08-07
    # during the anisotropic-cell work; [3,3,2]lv3 now agrees to
    # 1.9e-7. Measured here: [6,6,5]lv3 1.5e-6, [4,4,2]lv3 2.4e-6
    # against the lv2 reference.
    cases = [
        ('unsplit-z mid level', [6, 6, 5], 3),     # nmidlev [2,2,1]
        ('standard 2x2x2 path', [4, 4, 2], 3),     # nmidlev [2,2,2]
    ]
    for tag, nleaf, lv in cases:
        s, a = invariants(m, nleaf, lv)
        dr = abs(s - s_ref)/abs(s_ref)
        da = abs(a - a_ref)/a_ref
        check(tag, dr < 1e-3 and da < 1e-3,
              "vs lv2: dsum %.2e, d|.| %.2e (nleaf %s lv %d)"
              % (dr, da, nleaf, lv))
    print("\n%d checks failed" % len(fails))
    if fails:
        print("  " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
