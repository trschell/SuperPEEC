# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Conduction-palette ablation: can richer corner shapes close the
engine's straight-run delivery ceiling (the corner program's "rank 2")?

MEASURED PROBLEM (Z-trace/straight ladder, 2026-08-19): the shipped
palette (symmetric face sum, two antisymmetric face pairs, ONE fully
symmetric corner sum) delivers ~93% of the crowding correction at
dx/delta = 1.9 and ~83% at 6 on a 2x2-cell cross-section. The span
analysis says why: four corner anchors admit four parity combinations
(symmetric, x-odd, y-odd, xy-odd) and the palette carries only the
symmetric one -- a corner cell cannot crowd toward its ONE exposed
corner without equally crowding toward its three interior corners.

VARIANTS (monkey-patched over equiterminal.conduction_weights; both
call sites -- __init__ and the set_frequency retune -- resolve the
module global, so src/ is untouched and the suite unaffected):
  P0  shipped palette (unpatched -- must reproduce the ladder rows);
  P1  4 individual face exponentials + 4 INDIVIDUAL corner
      exponentials (same face span; corner parity space completed);
  P2  P1 + per-corner intermediate decay angles phi = 22.5/67.5 deg
      ((alpha, beta) = (1+j)/delta (cos phi, sin phi) -- denser
      sampling of the Helmholtz constraint's direction continuum;
      phi = 45 deg IS the corner exponential).
All variants get the shipped post-processing verbatim: re/im split,
mean removal (net-zero), norm filter, pivoted-QR prune.

HARNESS: the ladder's straight control bar (80 um, 8x8 um, copper,
NO corners), conduction engine k = 7 boundary-only, rungs at 2/3/4
cells across, 1e9 and 1e10 Hz. Delivered fraction uses the recorded
engine-off PWC values and the h->0 ladder extrapolates:
  truth_S:  1e9 = 0.02904, 1e10 = 0.0799  (+-~1 in the last digit)
  R_pwc(2/3/4 across): 1e9 = 0.0215517/0.0256162/0.0271256
                       1e10 = 0.0215517/0.0446211/0.049023
Rows with flag != 0 and resid > 1e-7 are VOID (standing ladder rule;
the coarse engine's missing mode_precond is a known docketed issue).

Usage: PYTHONPATH=src python3 studies/palette_ablation.py
Env: LADDER="0.25,0.375,0.5", FREQS="1e9,1e10", MAXITER=60.

MEASURED VERDICT (2026-08-20, all rows converged by residual):
  * P1 (+4 individual corners) is a CONSISTENT +6-point delivery gain
    at deep skin: 83.1->88.7 / 75.4->81.5 / 69.3->75.3 % at 2/3/4
    cells across, 1e10 Hz -- the missing corner-parity space is real.
    At the 1e9 transition it is neutral (-0.6/0/0 points): the corner
    exponentials barely differ from face products at dx/delta ~ 2.
  * P2 (intermediate angles) is an EXACT NULL everywhere (R changes
    < 1e-4 relative) at 2-4x the mode count: direction sampling of
    the Helmholtz continuum is NOT where the residual lives.
  * The post-P1 residual PLATEAUS at ~8-9.5% of R across dx/delta
    3-6: non-separable profile structure, the analytic family's
    ceiling. Same conclusion as the corner referee: tabulated
    per-class profiles (C.2 merged architecture) are the road past
    ~90%, not more palette.
  * Cost of P1 is small WITH the new engine mode_precond (km 8->16,
    matvecs 11->12 / 15->17); P2 additionally stresses conditioning
    (one 387-matvec row, converged by residual only).
This experiment also forced shipping Redistribution.mode_precond
(follow-up (a)): without it, P1/P2 at 1e9 stalled at 621 matvecs
unconverged; with it the SHIPPED palette's engine solves dropped
95->11 and 63->15 matvecs, answers bit-identical.

K-FLOOR ADDENDUM (2026-08-20, KVAL env; see xsection_tabulated.py for
the 2-D evidence): the palette and k gains are ADDITIVE, matvecs
unchanged (km drives cost, k is solve-free):
    2-across 1e10:  P0 k7 83.1 -> P1 k7 88.7 -> P0 k12 87.0
                    -> P1 k12 92.7 % delivered (err -12.3 -> -5.3%)
    4-across 1e10:  69.3 -> 77.3 % (k-gain shrinks to +1.7: the
                    remaining term is PER-CELL FACE ANCHORING, whose
                    fix is the C.2 surface-anchored architecture).
2-D evidence (xsection_tabulated.py): the smooth conduction family is
COMPLETE (-0.0% everywhere, transfers across aspect ratio); the
sub-bar piecewise-constant floor binds (tab1(same-freq) reproduces
cond k=8 digit-for-digit); TABULATED section profiles are a DEAD END
(same floor, and -16% when a square-class table is reused on a 2:1
rectangle). The evidenced rank-2 package is P1 + higher k (+ the
shipped mode_precond); palette/k defaults remain a user decision.
"""
import os
import sys

sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import qr

import vhr
import equiterminal as eq

SIG = 5.8e7
W_UM, ARM_UM = 8.0, 32.0
LADDER = [float(s) for s in
          os.environ.get('LADDER', '0.25,0.375,0.5').split(',')]
FREQS = tuple(float(s) for s in
              os.environ.get('FREQS', '1e9,1e10').split(','))
MAXITER = int(os.environ.get('MAXITER', 60))
KVAL = int(os.environ.get('KVAL', 7))
OUT = os.environ.get('TMPDIR', '/tmp')

TRUTH = {1e9: 0.02904, 1e10: 0.0799}
PWC = {(0.25, 1e9): 0.0215517, (0.375, 1e9): 0.0256162,
       (0.5, 1e9): 0.0271256, (0.25, 1e10): 0.0215517,
       (0.375, 1e10): 0.0446211, (0.5, 1e10): 0.049023}

_ship = eq.conduction_weights          # keep the shipped function


def _finish(shapes):
    """The shipped post-processing, verbatim."""
    cols = []
    for c in shapes:
        for part in (c.real, c.imag):
            w = part - part.mean()
            n = np.linalg.norm(w)
            if n > 1e-12*max(np.linalg.norm(part), 1.0):
                cols.append(w/n)
    W = np.stack(cols, axis=1)
    _, R, piv = qr(W, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    W = W[:, np.sort(piv[d > 1e-7*d[0]])]
    return W - W.mean(axis=0, keepdims=True)


def make_palette(variant):
    def weights(kk, dx, delta):
        k0, k1 = kk
        u = (np.arange(k0) + 0.5)/k0
        v = (np.arange(k1) + 0.5)/k1
        U, V = np.meshgrid(u, v, indexing='ij')
        x = U.ravel()*dx
        y = V.ravel()*dx
        p = (1.0 + 1.0j)/delta
        xx = {0: x, 1: dx - x}
        yy = {0: y, 1: dx - y}
        shapes = [np.exp(-p*xx[0]), np.exp(-p*xx[1]),
                  np.exp(-p*yy[0]), np.exp(-p*yy[1])]
        pc = p/np.sqrt(2.0)
        for sx in (0, 1):
            for sy in (0, 1):
                shapes.append(np.exp(-pc*(xx[sx] + yy[sy])))
        if variant == 'P2':
            for phi in (np.pi/8, 3*np.pi/8):
                a, b = np.cos(phi), np.sin(phi)
                for sx in (0, 1):
                    for sy in (0, 1):
                        shapes.append(np.exp(-p*(a*xx[sx] + b*yy[sy])))
        return _finish(shapes)
    return weights


def build(nper):
    dx = 1e-6/nper
    W = int(round(W_UM*nper))
    nx = int(round((3*ARM_UM - 2*W_UM)*nper))   # the ladder's 80um control
    struc = np.ones((nx, W, W), dtype=np.int8)
    ports = [(('p1', 'P', 0, j, k, '-x') if s == 0 else
              ('p1', 'N', nx - 1, j, k, '+x'))
             for s in (0, 1) for j in range(W) for k in range(W)]
    path = os.path.join(OUT, 'pa_%g.vhr' % nper)
    vhr.write_vhr(path, struc, dx, SIG, FREQS, ports)
    m = vhr.read_vhr(path)
    M = m.build_tree()
    m.prepare(M, FREQS[0])
    return m, M


print("palette ablation: straight 80um bar, 8x8um Cu, engine k=7 "
      "boundary-only\nrungs %s (cells across = 8*nper), freqs %s\n"
      % (LADDER, list(FREQS)))
print("%-4s %-5s %-4s | %-6s %-11s %-9s %-9s | %s"
      % ("pal", "n/um", "km", "freq", "R[Ohm]", "err_tru", "delivered",
         "mv/flag/resid"))
for nper in LADDER:
    m, M = build(nper)
    for pal in ("P0", "P1"):
        eq.conduction_weights = _ship if pal == 'P0' else \
            make_palette(pal)
        try:
            S = eq.EquiTerminalSolver(
                m, M, 0, subdivide=KVAL, mode_basis='conduction',
                boundary_only=True, skin_freq=max(FREQS))
            km = S.redist.km
            for f in FREQS:
                Z, _, info = S.solve(f, rtol=1e-8, maxiter=MAXITER)
                void = info['flag'] != 0 and info['residual'] > 1e-7
                dlv = ((Z.real - PWC[(nper, f)])
                       / (TRUTH[f] - PWC[(nper, f)]))
                print("%-4s %-5g %-4d | %-6.0e %-11.6g %-+8.2f%% "
                      "%-8.1f%% | %d/%s/%.0e%s"
                      % (pal, nper, km, f, Z.real,
                         100*(Z.real/TRUTH[f] - 1), 100*dlv,
                         info['matvecs'], info['flag'],
                         info['residual'], "  VOID" if void else ""),
                      flush=True)
            del S
        except Exception as e:
            print("%-4s %-5g FAILED %s: %s"
                  % (pal, nper, type(e).__name__, str(e)[:70]),
                  flush=True)
        finally:
            eq.conduction_weights = _ship
print("\ndelivered = share of the PWC->truth crowding correction; "
      "100 = truth.")
