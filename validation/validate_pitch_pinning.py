# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Cell-pitch pinning must be PER AXIS, not a single scalar factor.

`Tree` does not use the caller's `LT/nt` as the cell pitch. It pads the
grid up to `ntotalfull`, a whole number of leaf boxes, and divides the
top box by that PADDED count -- so the realised pitch is
`LT/ntotalfull`. Rebuilding with `LT` rescaled to compensate is the
standard recipe here (main.build_tree, profile_fill_fraction.build,
vhr.VhrModel.build_tree).

The trap: `ntotalfull` pads EACH AXIS INDEPENDENTLY, so on a non-cubic
grid the padding ratio differs between axes. 60x20x20 at leaf 5 pads to
65x25x25 -- 1.083 on x but 1.25 on y and z. A single scalar factor
taken from axis 0 then pins x correctly and leaves y and z wrong, i.e.
ANISOTROPIC cells for a geometry that asked for cubic ones.

That failure is quiet. The cells are still uniform, the solve still
converges, and nothing asserts. It surfaces only as the x-directed
filament resistance drifting from the y- and z-directed ones, because
`f.r = l[0]/(l[1]*l[2]*sigma)` while `e.r = l[1]/(l[0]*l[2]*sigma)` --
equal only when the pitch is isotropic.

This checks the property directly against `mp.Tree`, independent of
which module implements the recipe: for a range of non-cubic grids and
leaf sizes it confirms (a) the per-axis rescale pins all three axes,
(b) the three filament resistances come out equal, and (c) the scalar
rescale would NOT have -- the last so that a revert fails loudly here
rather than silently in a solve.

Run inside the toolbox:  python3 validate_pitch_pinning.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import multipole as mp

CELL = 1e-6
SIGMA = 5.8e7

# (dims, leaf, numlevels). The first is the VoxHenry straight-conductor
# grid that exposed this; the rest vary which axis is long and by how
# much, including one already-cubic case as a control.
CASES = [
    ((60, 20, 20), 5, 2),
    ((50, 10, 10), 3, 2),
    ((24, 24, 8), 4, 2),
    ((30, 12, 18), 4, 2),
    ((17, 9, 33), 3, 2),
    ((16, 16, 16), 4, 2),       # control: cubic, scalar == per-axis
]

fails = []


def build(dims, leaf, numlevels, per_axis):
    """Build twice, pinning the pitch either per axis or by one scalar."""
    struc = np.ones(dims, dtype=np.int8)
    LT0 = np.asarray(dims, dtype=float)*CELL
    nleaf = np.array([leaf]*3)
    probe = mp.Tree(struc, nleaf, LT0, numlevels, 1e0, 4, capacitive=False)
    l0 = np.asarray(probe.e.l, dtype=float).copy()
    del probe
    fac = CELL/l0 if per_axis else np.full(3, CELL/l0[0])
    M = mp.Tree(struc, nleaf, LT0*fac, numlevels, 1e0, 4, capacitive=False)
    lf = np.asarray(M.e.l, dtype=float).copy()
    del M
    return lf


print("  %-14s %-5s | %-28s %-28s" % ('dims', 'leaf', 'per-axis pitch/CELL',
                                      'scalar pitch/CELL'))
for dims, leaf, numlevels in CASES:
    good = build(dims, leaf, numlevels, True)
    bad = build(dims, leaf, numlevels, False)
    gr = good/CELL
    br = bad/CELL
    cubic = len(set(dims)) == 1

    ok = np.all(np.abs(gr - 1.0) < 1e-12)
    if not ok:
        fails.append("%s leaf %d: per-axis pitch %s" % (dims, leaf, list(gr)))

    # isotropy => the three filament resistances agree
    def rs(l):
        return (l[1]/(l[0]*l[2]*SIGMA), l[0]/(l[1]*l[2]*SIGMA),
                l[2]/(l[0]*l[1]*SIGMA))
    re, rf, rg = rs(good)
    iso = abs(rf/re - 1.0) < 1e-12 and abs(rg/re - 1.0) < 1e-12
    if not iso:
        fails.append("%s leaf %d: anisotropic r  e %.6g f %.6g g %.6g"
                     % (dims, leaf, re, rf, rg))
        ok = False

    # and the scalar form must be wrong exactly when the grid is not cubic
    scalar_ok = bool(np.all(np.abs(br - 1.0) < 1e-12))
    if cubic and not scalar_ok:
        fails.append("%s leaf %d: scalar should agree on a cubic grid"
                     % (dims, leaf))
        ok = False
    if not cubic and scalar_ok:
        fails.append("%s leaf %d: scalar unexpectedly correct -- this case "
                     "no longer exercises the bug, pick another"
                     % (dims, leaf))
        ok = False

    print("  %-14s %-5d | %-28s %-28s %s%s"
          % ('x'.join(str(d) for d in dims), leaf,
             np.array2string(gr, precision=6),
             np.array2string(br, precision=6),
             'PASS' if ok else 'FAIL',
             '  (cubic control)' if cubic else ''))

print()
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("ALL CHECKS PASSED")
