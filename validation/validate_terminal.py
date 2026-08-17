# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate terminal.py: half-length port filaments and their couplings.

A terminal filament is half as long as an ordinary one, so their mutual
partial inductance couples bars of UNEQUAL axial extent -- which the
identical-bar superposition in ``greens.genL3D`` cannot produce
directly. terminal.py avoids the general unequal-bar formula by halving
the lattice along the filament axis, making every bar a contiguous run
of identical half-slots.

PART A -- SUPERPOSITION. Splitting a bar axially must not change the
answer, so the half-slot sum must reproduce the full-pitch ``genL3D``
kernel wherever both can be evaluated. Checked over a grid of axial and
transverse separations.

PART B -- INDEPENDENT QUADRATURE. Part A shares ``genL3D`` with the
thing it is testing, so it cannot catch an error in ``genL3D`` itself,
nor confirm the UNEQUAL-length case (which has no full-pitch
counterpart). Part B evaluates

    Mp = (mu0/4pi) / (Aa*Ab) * Int_A Int_B dV dV' / |r - r'|

directly by 6-D Gauss-Legendre for terminal-to-terminal,
terminal-to-ordinary and ordinary-to-ordinary pairs. This shares no code
with the kernel. Bars are kept separated so the integrand is smooth;
convergence is demonstrated by refining the rule.

PART C -- FAR-FIELD LIMIT. For separation much larger than the bar
dimensions the cross-section stops mattering and
``Mp -> mu0*la*lb/(4*pi*d)``. Checked as a ratio tending to 1.

PART D -- SERIES IDENTITY. An ordinary filament IS two terminal
filaments in series, so its self partial inductance must equal
``2*Lp(half) + 2*Mp(adjacent halves)``.

PART E -- PORT LAYOUT. The half-slots of a driven run must tile it
exactly: ``1 + 2(L-1) + 1 = 2L``, i.e. length ``L*dx``, which is the
whole point of the terminal filaments.

Run inside the toolbox:  python3 validate_terminal.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
from greens import genL3D
import terminal as tm

MU0 = 4*np.pi*1e-7
DX = 1e-6
L = (DX, DX, DX)

fails = []


def check(tag, cond, detail=''):
    if cond:
        return True
    fails.append("%s: %s" % (tag, detail))
    print("    FAIL %s  %s" % (tag, detail))
    return False


def quad_mutual(box_a, box_b, npt=8):
    """Mp between two parallel bars by direct 6-D Gauss-Legendre.

    Boxes are ``((x0,x1),(y0,y1),(z0,z1))``. Shares no code with
    greens.genL3D.
    """
    xs, ws = np.polynomial.legendre.leggauss(npt)
    pts, wts, vol = [], [], []
    for box in (box_a, box_b):
        axes, wgt = [], []
        for (a, b) in box:
            axes.append(0.5*(b - a)*xs + 0.5*(a + b))
            wgt.append(0.5*(b - a)*ws)
        gx, gy, gz = np.meshgrid(*axes, indexing='ij')
        wx, wy, wz = np.meshgrid(*wgt, indexing='ij')
        pts.append(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1))
        wts.append((wx*wy*wz).ravel())
        vol.append(np.prod([b - a for (a, b) in box]))
    # cross-sectional areas: volume / axial length, with x the axis here
    aa = vol[0]/(box_a[0][1] - box_a[0][0])
    ab = vol[1]/(box_b[0][1] - box_b[0][0])
    d = np.linalg.norm(pts[0][:, None, :] - pts[1][None, :, :], axis=2)
    integral = np.einsum('i,j,ij->', wts[0], wts[1], 1.0/d)
    return MU0/(4*np.pi)*integral/(aa*ab)


def slot_box(offset, length, transverse=(0, 0)):
    """The physical box of a half-slot run, x as the axial direction."""
    return ((offset*DX/2, (offset + length)*DX/2),
            (transverse[0]*DX - DX/2, transverse[0]*DX + DX/2),
            (transverse[1]*DX - DX/2, transverse[1]*DX + DX/2))


# ---------------------------------------------------------------- A

print("=== PART A: half-slot superposition vs the full-pitch kernel ===")
Kf = genL3D(DX, DX, DX, 5, 7, 5)
Kh = tm.axial_halfstep_kernel(L, 'f', (5, 7, 5))
worst = 0.0
nA = 0
for ax in range(5):
    for ay in range(5):
        for az in range(4):
            # 'f' is x-directed: axial index is 0, transverse are (y, z)
            full = Kf[ay, ax, az] if False else None
            # genL3D(dw,dl,dt) for 'f' in leaf_induct is called with
            # (l[1],l[0],l[2]) then transposed; replicate that here
            full = np.transpose(genL3D(DX, DX, DX, 5, 7, 5), (1, 0, 2))[
                ax, ay, az]
            half = tm.mutual_segments(Kh, 'f', (0, 2), (2*ax, 2),
                                      transverse=(ay, az))
            nA += 1
            rel = abs(half - full)/abs(full)
            worst = max(worst, rel)
            check('superposition (%d,%d,%d)' % (ax, ay, az), rel < 1e-11,
                  "%.10e vs %.10e (rel %.2e)" % (half, full, rel))
print("  %d separations, worst relative difference %.2e" % (nA, worst))

# ---------------------------------------------------------------- B

print("\n=== PART B: independent 6-D quadrature (no shared code) ===")
print("  %-26s %-15s %-15s %s"
      % ('pair', 'terminal.py', 'quadrature', 'rel err'))
CASES = [
    ("terminal-terminal, 3 cells", (0, 1), (6, 1), (0, 0)),
    ("terminal-ordinary, 3 cells", (0, 1), (6, 2), (0, 0)),
    ("ordinary-ordinary, 3 cells", (0, 2), (6, 2), (0, 0)),
    ("terminal-ordinary, offset",  (0, 1), (5, 2), (1, 0)),
    ("terminal-terminal, diag",    (0, 1), (4, 1), (1, 1)),
    ("ordinary-terminal, far",     (0, 2), (10, 1), (2, 1)),
]
for tag, sa, sb, tr in CASES:
    got = tm.mutual_segments(Kh, 'f', sa, sb, transverse=tr)
    ref8 = quad_mutual(slot_box(*sa), slot_box(*sb, transverse=tr), npt=8)
    ref10 = quad_mutual(slot_box(*sa), slot_box(*sb, transverse=tr), npt=10)
    conv = abs(ref10 - ref8)/abs(ref10)
    rel = abs(got - ref10)/abs(ref10)
    check('quadrature converged: %s' % tag, conv < 1e-9,
          "8 vs 10 point differ by %.2e" % conv)
    check('quadrature agrees: %s' % tag, rel < 1e-7,
          "%.10e vs %.10e (rel %.2e)" % (got, ref10, rel))
    print("  %-26s %-15.8e %-15.8e %.2e" % (tag, got, ref10, rel))

# ---------------------------------------------------------------- C

print("\n=== PART C: far-field limit  Mp -> mu0*la*lb/(4 pi d) ===")
print("  %-10s %-15s %-15s %s" % ('sep (cells)', 'Mp', 'mu0 la lb/4pi d',
                                  'ratio'))
# 'f' is x-directed, so a TRANSVERSE sweep runs along y: that is the
# axis the kernel must be tabulated far along.
Kfar = tm.axial_halfstep_kernel(L, 'f', (2, 64, 3))
devs = []
for k in (5, 10, 20, 40):
    got = tm.mutual_segments(Kfar, 'f', (0, 2), (0, 2), transverse=(k, 0))
    approx = MU0*DX*DX/(4*np.pi*k*DX)
    devs.append(abs(got/approx - 1.0))
    print("  %-10d %-15.8e %-15.8e %.6f" % (k, got, approx, got/approx))
check('far-field deviation shrinks monotonically',
      all(devs[i] > devs[i+1] for i in range(len(devs)-1)),
      "deviations %s" % ["%.2e" % d for d in devs])
check('far-field ratio approaches 1', devs[-1] < 1e-6,
      "deviation at 40 cells %.2e" % devs[-1])
# For a CUBIC bar the three O(1/d^2) corrections cancel exactly. With
# separation d along y and the bar of side D in each direction, second
# moments give: finite length along x (perpendicular to d)
# -D^2/12d^2; finite width along y (parallel to d) +D^2/6d^2; finite
# thickness along z (perpendicular) -D^2/12d^2. The sum is zero, so the
# leading error is O(1/d^4) and the ratio should fall by 4^4 = 256 when
# the separation quadruples.
check('correction scales as 1/d^4 (cubic-bar cancellation)',
      abs(devs[1]/devs[3] - 256.0) < 40.0,
      "ratio dev(10)/dev(40) = %.2f, expected ~256" % (devs[1]/devs[3]))

# ---------------------------------------------------------------- D

print("\n=== PART D: an ordinary filament is two terminals in series ===")
lp_full = tm.mutual_segments(Kh, 'f', (0, 2), (0, 2))
lp_half = tm.mutual_segments(Kh, 'f', (0, 1), (0, 1))
m_adj = tm.mutual_segments(Kh, 'f', (0, 1), (1, 1))
series = 2*lp_half + 2*m_adj
rel = abs(series - lp_full)/abs(lp_full)
check('series identity', rel < 1e-12,
      "%.12e vs %.12e (rel %.2e)" % (series, lp_full, rel))
print("  Lp(ordinary)        %.12e" % lp_full)
print("  2 Lp(half) + 2 Mp   %.12e   (rel diff %.2e)" % (series, rel))

# ---------------------------------------------------------------- E

print("\n=== PART E: port layout tiles the conductor exactly ===")
print("  %-8s %-10s %-12s %-12s %s"
      % ('cells', 'filaments', 'half-slots', 'length', 'exact?'))
for ncells in (1, 2, 3, 5, 12, 60):
    spans = tm.port_run_slots(ncells)
    slots = tm.run_length_slots(ncells)
    nterm = sum(1 for s in spans if s[2] == 'terminal')
    nint = sum(1 for s in spans if s[2] == 'interior')
    ok = check('layout L=%d slot count' % ncells, slots == 2*ncells,
               "%d vs %d" % (slots, 2*ncells))
    ok &= check('layout L=%d contiguous' % ncells,
                all(spans[i][0] + spans[i][1] == spans[i+1][0]
                    for i in range(len(spans)-1)) and spans[0][0] == 0,
                "spans %s" % spans)
    ok &= check('layout L=%d composition' % ncells,
                nterm == 2 and nint == ncells - 1,
                "%d terminal, %d interior" % (nterm, nint))
    print("  %-8d %-10s %-12d %-12s %s"
          % (ncells, "2+%d" % nint, slots, "%d dx" % ncells,
             'yes' if ok else 'NO'))
# without terminals the run is one cell short, which is the point
short = tm.run_length_slots(5, terminals=(False, False))
check('no-terminal run is one cell short', short == 2*5 - 2,
      "%d half-slots vs %d" % (short, 2*5 - 2))
print("  L=5 with terminals suppressed: %d half-slots = %.1f dx "
      "(one cell short, as expected)" % (short, short/2))

print()
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    for f in fails[:20]:
        print("  " + f)
    sys.exit(1)
print("ALL CHECKS PASSED")
