# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the element position stagger used by levels.leafinit.

The FMM needs a physical position for every element in a leaf box. Those
positions used to be six hand-written cases picking components out of two
meshgrids; they are now one rule (see ``Level.leafinit``):

    use CENTRE on axis i  iff  (i is the element's own axis)
                                XOR (this is a panel)

where NODE and CENTRE are two lattices of ``n[i]`` points offset by
exactly half a cell.

PART A -- BEHAVIOUR PRESERVED. The rule is checked BITWISE against the
six legacy cases transcribed verbatim, over random ``n`` and ``l`` for
every orientation. This is what makes the refactor provably
behaviour-preserving. When the cell-centred rewrite lands these
reference cases change deliberately, in step with the rule.

PART B -- STRUCTURAL PROPERTIES, scheme-independent. Whatever the
scheme, positions must form a regular lattice of the right size and
spacing, be centred on the box, and be staggered against one another by
either nothing or exactly half a cell on each axis. These should survive
the rewrite untouched, so a failure here means something is wrong in a
way that is not merely a convention change.

PART C -- THE TARGET, pinned in advance. Under the pending cell-centred
scheme (``docs/cell_centred_scope.md``) the XOR disappears: nodes move to
cell centres, so an x-filament spans cell centres i..i+1 while an
x-normal panel is the face between those same two cells, and the two
become CO-LOCATED. That is the congruence the rewrite is being done for,
so it is asserted here as an executable statement of intent -- it
describes the rule that is NOT yet in force, and is what step 2 of the
rewrite must reproduce.

Run inside the toolbox:  python3 validate_stagger.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
from stencils import AXIS_OF, PANEL_ORIENTATIONS

MU0 = 4*np.pi*1e-7
EPS0 = 1/(MU0 * 299792458**2)
ORIENTATIONS = 'efgxyz'

fails = []


def check(tag, cond, detail=''):
    if cond:
        return True
    fails.append("%s: %s" % (tag, detail))
    print("    FAIL %s  %s" % (tag, detail))
    return False


def legacy(n, l, o):
    """The six position cases exactly as levels.leafinit had them."""
    x0 = np.arange(-(n[0]-1)/2-0.5, (n[0]-1)/2+0.5)
    x1 = np.arange(-(n[0]-1)/2, (n[0]-1)/2+1)
    y0 = np.arange(-(n[1]-1)/2-0.5, (n[1]-1)/2+0.5)
    y1 = np.arange(-(n[1]-1)/2, (n[1]-1)/2+1)
    z0 = np.arange(-(n[2]-1)/2-0.5, (n[2]-1)/2+0.5)
    z1 = np.arange(-(n[2]-1)/2, (n[2]-1)/2+1)
    ax, ay, az = np.meshgrid(x0, y0, z0, indexing='ij')
    bx, by, bz = np.meshgrid(x1, y1, z1, indexing='ij')
    ax, az = l[0]*ax.ravel(), l[2]*az.ravel()
    bx, bz = l[0]*bx.ravel(), l[2]*bz.ravel()
    ay, by = l[1]*ay.ravel(), l[1]*by.ravel()
    if o == 'e':
        return (ax, by, az), MU0/(4*np.pi)*l[1]**2
    if o == 'f':
        return (bx, ay, az), MU0/(4*np.pi)*l[0]**2
    if o == 'g':
        return (ax, ay, bz), MU0/(4*np.pi)*l[2]**2
    if o == 'x':
        return (ax, by, bz), 1/(4*np.pi*EPS0)
    if o == 'y':
        return (bx, ay, bz), 1/(4*np.pi*EPS0)
    return (bx, by, az), 1/(4*np.pi*EPS0)


def positions(n, l, o, cell_centred=False):
    """levels.leafinit's rule, and the cell-centred rule it becomes."""
    axis = AXIS_OF[o]
    panel = o in PANEL_ORIENTATIONS
    node = [np.arange(-(n[i]-1)/2-0.5, (n[i]-1)/2+0.5) for i in range(3)]
    centre = [np.arange(-(n[i]-1)/2, (n[i]-1)/2+1) for i in range(3)]
    if cell_centred:
        # the offset family on the element's own axis, centred elsewhere
        # -- the XOR with `panel` is gone
        use = [i == axis for i in range(3)]
    else:
        use = [(i == axis) != panel for i in range(3)]
    fam = [centre[i] if use[i] else node[i] for i in range(3)]
    g = np.meshgrid(fam[0], fam[1], fam[2], indexing='ij')
    p = tuple(l[i]*g[i].ravel() for i in range(3))
    m0 = 1/(4*np.pi*EPS0) if panel else MU0/(4*np.pi)*l[axis]**2
    return p, m0


rng = np.random.default_rng(20260731)
CASES = []
for _ in range(120):
    CASES.append((tuple(int(v) for v in rng.integers(1, 10, 3)),
                  tuple(float(v) for v in rng.uniform(1e-7, 1e-4, 3))))

# ---------------------------------------------------------------- A

print("=== PART A: the rule vs the six legacy cases (bitwise) ===")
nA = 0
for n, l in CASES:
    for o in ORIENTATIONS:
        nA += 1
        (pa, ma) = legacy(n, l, o)
        (pb, mb) = positions(n, l, o)
        check('legacy %s n=%s' % (o, n),
              ma == mb and all(np.array_equal(u, v)
                               for u, v in zip(pa, pb)),
              "prefactor %r vs %r" % (ma, mb))
print("  %d comparisons (%d shapes x 6 orientations)" % (nA, len(CASES)))

# ---------------------------------------------------------------- B

print("\n=== PART B: structural properties (scheme-independent) ===")
nB = 0
for n, l in CASES[:40]:
    got = {o: positions(n, l, o)[0] for o in ORIENTATIONS}
    for o, p in got.items():
        nB += 1
        check('%s count' % o, p[0].size == int(np.prod(n)),
              "%d vs %d" % (p[0].size, np.prod(n)))
        for i in range(3):
            u = np.unique(np.round(p[i]/l[i], 9))
            nB += 1
            check('%s axis %d lattice' % (o, i),
                  u.size == n[i]
                  and (n[i] == 1
                       or np.allclose(np.diff(u), 1.0, rtol=0, atol=1e-9)),
                  "%d distinct, spacing %s" % (u.size, np.diff(u)[:3]))
            nB += 1
            # box-centred: the mean sits within half a cell of the origin
            check('%s axis %d centred' % (o, i),
                  abs(p[i].mean()) <= 0.5*l[i] + 1e-18,
                  "mean %.3e vs l/2 %.3e" % (p[i].mean(), 0.5*l[i]))
    # any two orientations differ by 0 or exactly half a cell per axis
    for a in ORIENTATIONS:
        for b in ORIENTATIONS:
            for i in range(3):
                d = abs(got[a][i][0] - got[b][i][0])
                nB += 1
                check('stagger %s-%s axis %d' % (a, b, i),
                      min(abs(d), abs(d - 0.5*l[i])) < 1e-9*l[i],
                      "offset %.6g, l/2 %.6g" % (d, 0.5*l[i]))
print("  %d property checks over 40 shapes" % nB)

# ---------------------------------------------------------------- C

print("\n=== PART C: the cell-centred target (not yet in force) ===")
print("  filament of orientation d and panel normal to d must become")
print("  CO-LOCATED -- the congruence the rewrite is being done for")
nC = 0
PAIRS = [('f', 'x'), ('e', 'y'), ('g', 'z')]     # same axis: 0, 1, 2
for n, l in CASES[:40]:
    for fil, pan in PAIRS:
        pf = positions(n, l, fil, cell_centred=True)[0]
        pp = positions(n, l, pan, cell_centred=True)[0]
        nC += 1
        check('cell-centred %s/%s co-located' % (fil, pan),
              all(np.array_equal(u, v) for u, v in zip(pf, pp)),
              "axis offsets %s" % [float(u[0]-v[0]) for u, v in zip(pf, pp)])
        # and under the CURRENT scheme they must NOT be -- they are half a
        # cell apart on every axis, which is the defect being removed
        qf = positions(n, l, fil)[0]
        qp = positions(n, l, pan)[0]
        nC += 1
        check('current %s/%s staggered' % (fil, pan),
              all(abs(abs(float(u[0]-v[0])) - 0.5*l[i]) < 1e-9*l[i]
                  for i, (u, v) in enumerate(zip(qf, qp))),
              "offsets %s" % [float(u[0]-v[0]) for u, v in zip(qf, qp)])
print("  %d checks over 40 shapes" % nC)

print()
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    for f in fails[:20]:
        print("  " + f)
    sys.exit(1)
print("ALL CHECKS PASSED")
