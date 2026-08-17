# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate greens.p_per_rect, the general perpendicular-panel kernel.

``gen_p_per`` gives the coefficient of potential between perpendicular
panels only for EQUAL panels on a lattice whose stride equals the panel
size -- it uses its arguments as both. The cell-centred discretisation
needs whole ``dx`` faces on lattices staggered by ``dx/2``, where size
and stride differ, so :func:`greens.p_per_rect` generalises it to
independent sizes at an arbitrary offset. (The parallel counterpart,
``gen_p_parz``, was already general.)

PART A -- vs DIRECT QUADRATURE. ``panel_quad.py`` integrates the
defining double surface integral numerically and shares no code with
``greens``. This is the primary absolute check. It only applies where
the panels are separated: the integrand is singular when they touch, and
the quadrature error is reported alongside so the comparison is never
tighter than the reference deserves.

PART B -- vs gen_p_per. In the special case gen_p_per covers, the two
must agree, which pins the generalisation to the existing convention.

PART C -- RECIPROCITY. ``P(A,B) == P(B,A)`` for a coefficient of
potential, whatever the sizes and offset.

PART D -- SUBDIVISION. A rectangle is the union of its four quarters,
and with uniform density on each,

    P(A,B) = (1/16) sum_{a in A} sum_{b in B} P(a,b)

-- one factor of 1/4 from averaging the potential over A's quarters,
another from splitting B's charge. This is the check that MATTERS MOST,
because unlike quadrature it holds for TOUCHING panels, which is exactly
where the near field lives and where quadrature cannot follow.

PART E -- FAR FIELD. ``P -> 1/(4 pi eps0 d)`` as the separation grows.

PART F -- THE b, c >= 0 INVARIANT. The primitive J is valid only for
non-negative b and c; on a negative one it silently returns the b = 0 or
c = 0 limit, a plausible number indistinguishable from a correct one. So
_J_per raises instead, and a 400-placement stress sweep confirms
p_per_rect never triggers it -- i.e. that _split really does guarantee
non-negative arguments for every relative placement, overlapping ones
included.

Run inside the toolbox:  python3 validate_panel_kernel.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import panel_quad as pq
from greens import p_per_rect, gen_p_per, gen_p_parz

A0 = 1e-6
EPS0 = 1/((4*np.pi*1e-7) * 299792458.0**2)

fails = []


def check(tag, cond, detail=''):
    if cond:
        return True
    fails.append("%s: %s" % (tag, detail))
    print("    FAIL %s  %s" % (tag, detail))
    return False


def quarters(rc):
    ax, (u0, u1), (v0, v1), pl = rc
    um, vm = 0.5*(u0 + u1), 0.5*(v0 + v1)
    return [pq.rect(ax, (x0, x1), (y0, y1), pl)
            for x0, x1 in ((u0, um), (um, u1))
            for y0, y1 in ((v0, vm), (vm, v1))]


# ---------------------------------------------------------------- A

print("=== PART A: vs direct quadrature (no shared code) ===")
print("  %-40s %-14s %-14s %-9s %s"
      % ('configuration', 'p_per_rect', 'quadrature', 'rel err', 'quad err'))
a = A0
CASES = []
A = pq.rect(2, (0, a), (0, a), 0.0)
for p, r, q in ((2, 2, 0), (3, 0, 0), (0, 3, 0), (2, 3, 1), (4, 1, 2)):
    CASES.append(("equal faces, x+%d z+%d y=%d" % (p, r, q), A,
                  pq.rect(1, (p*a, p*a + a), (r*a, r*a + a), q*a)))
CASES.append(("unequal 1x1 vs 2x3", pq.rect(2, (0, a), (0, a), 0.0),
              pq.rect(1, (2*a, 4*a), (3*a, 6*a), 1*a)))
CASES.append(("unequal 3x2 vs 1x1", pq.rect(2, (0, 3*a), (0, 2*a), 0.0),
              pq.rect(1, (4*a, 5*a), (2*a, 3*a), 2*a)))
# the actual cell-scheme need: whole faces on half-cell staggered lattices
for (ca, axa, sa), (cb, axb, sb) in (
        (((0, 0, 0), 0, +1), ((2, 0, 0), 1, +1)),
        (((0, 0, 0), 2, +1), ((2, 1, 1), 0, +1)),
        (((0, 0, 0), 1, +1), ((0, 2, 2), 2, +1))):
    CASES.append(("staggered whole faces %s%s vs %s%s"
                  % (ca, 'xyz'[axa], cb, 'xyz'[axb]),
                  pq.face(ca, axa, sa, a), pq.face(cb, axb, sb, a)))
worst = 0.0
for tag, X, Y in CASES:
    got = p_per_rect(X, Y)
    ref, qerr = pq.converged(X, Y, npt=14)
    rel = abs(got - ref)/abs(ref)
    worst = max(worst, rel)
    check('quadrature %s' % tag, rel < 1e-10,
          "%.10e vs %.10e (rel %.2e)" % (got, ref, rel))
    print("  %-40s %-14.7e %-14.7e %-9.2e %.0e"
          % (tag, got, ref, rel, qerr))
print("  worst %.2e over %d configurations" % (worst, len(CASES)))

# ---------------------------------------------------------------- B

print("\n=== PART B: reduces to gen_p_per in its special case ===")
ref_tab = gen_p_per(a, a, a, 6, 6, 6)
worst = 0.0
n = 0
for p in range(4):
    for r in range(4):
        X = pq.rect(2, (0, a), (0, a), 0.0)
        Y = pq.rect(1, (p*a, p*a + a), (r*a, r*a + a), 0.0)
        got = p_per_rect(X, Y)
        # locate the matching table entry rather than assume the index
        # convention, which gen_p_per never documented
        best = min(((abs(got - ref_tab[i, j, k])/abs(ref_tab[i, j, k]),
                     i, j, k)
                    for i in range(5) for j in range(5) for k in range(5)))
        n += 1
        worst = max(worst, best[0])
        check('gen_p_per match p=%d r=%d' % (p, r), best[0] < 1e-10,
              "closest table entry differs by %.2e" % best[0])
print("  %d placements, worst rel err %.2e" % (n, worst))

# ---------------------------------------------------------------- C

print("\n=== PART C: reciprocity P(A,B) == P(B,A) ===")
rng = np.random.default_rng(11)
worst = 0.0
for _ in range(30):
    na, nb = rng.choice(3, 2, replace=False)
    X = pq.rect(na, (0, a*rng.integers(1, 4)), (0, a*rng.integers(1, 4)),
                a*rng.integers(-2, 3))
    lo = rng.integers(-3, 3)
    lo2 = rng.integers(-3, 3)
    Y = pq.rect(nb, (lo*a, (lo + rng.integers(1, 4))*a),
                (lo2*a, (lo2 + rng.integers(1, 4))*a),
                a*rng.integers(-2, 3))
    p1, p2 = p_per_rect(X, Y), p_per_rect(Y, X)
    if p1 == 0:
        continue
    worst = max(worst, abs(p1 - p2)/abs(p1))
check('reciprocity', worst < 1e-11, "worst asymmetry %.2e" % worst)
print("  worst asymmetry over 30 random pairs: %.2e" % worst)

# ---------------------------------------------------------------- D

print("\n=== PART D: subdivision -- HOLDS FOR TOUCHING PANELS ===")
print("  P(A,B) = (1/16) sum_ij P(a_i, b_j) over quarters")
SUBD = [
    ("separated", pq.rect(2, (0, a), (0, a), 0.0),
     pq.rect(1, (2*a, 3*a), (2*a, 3*a), 0.0)),
    ("TOUCHING: two faces of one cell", pq.face((0, 0, 0), 2, +1, a),
     pq.face((0, 0, 0), 1, +1, a)),
    ("TOUCHING: edge-adjacent cells", pq.face((0, 0, 0), 0, +1, a),
     pq.face((1, 0, 0), 1, +1, a)),
    ("TOUCHING: unequal sizes", pq.rect(2, (0, 2*a), (0, 2*a), 0.0),
     pq.rect(1, (0, a), (0, a), 0.0)),
]
worst = 0.0
for tag, X, Y in SUBD:
    whole = p_per_rect(X, Y)
    sub = sum(p_per_rect(qa, qb)
              for qa in quarters(X) for qb in quarters(Y))/16.0
    rel = abs(whole - sub)/abs(whole)
    worst = max(worst, rel)
    check('subdivision %s' % tag, rel < 1e-12,
          "whole %.10e vs quarters %.10e (rel %.2e)" % (whole, sub, rel))
    print("  %-34s whole %.10e  quarters %.10e  rel %.1e"
          % (tag, whole, sub, rel))
print("  worst %.2e" % worst)

# ---------------------------------------------------------------- E

print("\n=== PART E: far field  P -> 1/(4 pi eps0 d) ===")
print("  %-8s %-15s %-15s %s" % ('d/dx', 'P', '1/(4 pi eps0 d)', 'ratio'))
devs = []
for k in (10, 20, 40, 80):
    X = pq.rect(2, (0, a), (0, a), 0.0)
    Y = pq.rect(1, (k*a, k*a + a), (0, a), 0.0)
    got = p_per_rect(X, Y)
    d = (k + 0.5)*a - 0.5*a
    approx = 1/(4*np.pi*EPS0*(k*a))
    devs.append(abs(got/approx - 1.0))
    print("  %-8d %-15.7e %-15.7e %.6f" % (k, got, approx, got/approx))
check('far field approaches the point-charge law',
      devs[-1] < 2e-3 and all(devs[i] > devs[i+1]
                              for i in range(len(devs)-1)),
      "deviations %s" % ["%.2e" % v for v in devs])

# ---------------------------------------------------------------- F

print("\n=== PART F: the b,c >= 0 invariant is ENFORCED and HOLDS ===")
from greens import _J_per                                  # noqa: E402
for bad_b, bad_c in ((-0.7, 0.5), (0.5, -0.7), (-1e-18, 0.5)):
    raised = False
    try:
        _J_per(1.0, bad_b, bad_c)
    except ValueError:
        raised = True
    check('_J_per rejects b=%g c=%g' % (bad_b, bad_c), raised,
          "returned a value instead of raising -- without this it hands "
          "back the b=0 or c=0 limit, which is a plausible number and "
          "indistinguishable from a correct one")
print("  negative arguments raise rather than silently returning the "
      "degenerate limit")

# ...and a stress sweep confirming p_per_rect never triggers it, i.e.
# that _split really does guarantee non-negative arguments for every
# relative placement, including heavy overlap in projection
rng2 = np.random.default_rng(20260731)
ntry = 0
for _ in range(400):
    na2, nb2 = rng2.choice(3, 2, replace=False)
    u0, v0 = rng2.integers(-4, 5, 2)
    w0, z0 = rng2.integers(-4, 5, 2)
    X = pq.rect(na2, (u0*a, (u0 + rng2.integers(1, 4))*a),
                (v0*a, (v0 + rng2.integers(1, 4))*a),
                a*rng2.integers(-4, 5))
    Y = pq.rect(nb2, (w0*a, (w0 + rng2.integers(1, 4))*a),
                (z0*a, (z0 + rng2.integers(1, 4))*a),
                a*rng2.integers(-4, 5))
    try:
        p_per_rect(X, Y)
        ntry += 1
    except ValueError as exc:
        check('stress placement', False, "%s on %s vs %s" % (exc, X, Y))
print("  %d random placements evaluated, none violated it" % ntry)

print()
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    for f in fails[:20]:
        print("  " + f)
    sys.exit(1)
print("ALL CHECKS PASSED")
