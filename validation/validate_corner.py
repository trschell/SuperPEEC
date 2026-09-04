# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Corner circulation modes (cornermode.py): the 38th validator.

Gates, on the coarse Z-trace (two 90-degree corners, dx/W = 1/2 and
1/3 -- the resolution regime where the ladder measured the corner
error at 3.2-19.1% of R):

 1. detection: exactly two corners, opposite handedness, W and z
    extents right;
 2-4. (net-zero, Zuu symmetry and augmented reciprocity are generic
    family invariants and live in validate_enrich J);
 5. DC decoupling: the resistive mode<->aggregate cross term is
    exactly zero, so the mode-induced dR/R vanishes as O(omega^2) --
    measured 1.47e-9 at 1e5 Hz falling to 1.46e-13 at 1e3 Hz (exactly
    100^2). Gated on BOTH the magnitude and the scaling;
 6. AC improvement bands vs the ladder's h->0 extrapolates
    (R_L(1e9) = 0.0279045, R_L(1e10) = 0.0770577): corner modes must
    cut the coarse error at 1e9 to 0.50..0.80 of itself and at 1e10
    to 0.60..0.88 (measured 0.66 / 0.74 at dx/W = 1/2) -- direction
    AND magnitude, no overshoot past truth;
 7. the dx/W = 1/3 rung improves too and every solve is converged
    (flag 0, residual <= 1e-7; ladder rule: unflagged rows are void).

Run:  PYTHONPATH=../src python3 validate_corner.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import vhr
import equiterminal as eq
from cornermode import find_corners

W_UM, T_UM, ARM_UM, SIG = 8.0, 8.0, 32.0, 5.8e7
REF = {1e9: 0.0279045, 1e10: 0.0770577}    # ladder h->0 extrapolates
OUT = os.environ.get('TMPDIR', '/tmp')
nfail = 0


def check(name, ok, detail=""):
    global nfail
    print("%-52s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        nfail += 1


def build(nper):
    dx = 1e-6/nper
    W, T = int(round(W_UM*nper)), int(round(T_UM*nper))
    A = int(round(ARM_UM*nper))
    nx = int(round((2*ARM_UM - W_UM)*nper))
    struc = np.zeros((nx, A, T), dtype=np.int8)
    struc[:A, :W, :] = 1
    struc[A - W:A, :, :] = 1
    struc[A - W:, A - W:, :] = 1
    ports = ([('p1', 'P', 0, j, k, '-x') for j in range(W)
              for k in range(T)]
             + [('p1', 'N', nx - 1, j, k, '+x')
                for j in range(A - W, A) for k in range(T)])
    path = os.path.join(OUT, 'vc_%g.vhr' % nper)
    vhr.write_vhr(path, struc, dx, SIG, (1e9,), ports)
    m = vhr.read_vhr(path)
    M = m.build_tree()
    m.prepare(M, 1e9)
    return m, M


# ---------------------------------------------------------- dx/W = 1/2
m, M = build(0.25)
cs = find_corners(m.struc())
check("detection: two corners", len(cs) == 2, str([c[:5] for c in cs]))
if len(cs) == 2:
    (I1, J1, sx1, sy1, W1, z1), (I2, J2, sx2, sy2, W2, z2) = cs
    check("detection: widths and z", W1 == W2 == 2
          and len(z1) == len(z2) == 2)
    check("detection: opposite handedness",
          (sx1, sy1) == (-sx2, -sy2))

S0 = eq.EquiTerminalSolver(m, M, 0)                       # modes OFF
S1 = eq.EquiTerminalSolver(m, M, 0, enrich=dict(families=['corner']))
r = S1.redist
check("modes built", S1.nu == 6, "nu=%d" % S1.nu)

res = {}
for S, tag in ((S0, 'off'), (S1, 'on')):
    for f in (1e3, 1e5, 1e9, 1e10):
        Z, _, info = S.solve(f, rtol=1e-10 if f <= 1e5 else 1e-8)
        ok = info['flag'] == 0 and info['residual'] <= 1e-7
        check("converged %s f=%.0e" % (tag, f), ok,
              "flag %s resid %.1e" % (info['flag'], info['residual']))
        res[(tag, f)] = Z

d5 = abs(res[('on', 1e5)].real/res[('off', 1e5)].real - 1)
d3 = abs(res[('on', 1e3)].real/res[('off', 1e3)].real - 1)
check("DC decoupling magnitude", d5 < 1e-8 and d3 < 1e-11,
      "dR/R %.2e @1e5, %.2e @1e3" % (d5, d3))
check("DC decoupling is O(omega^2)", d3 < 1e-3*d5,
      "ratio %.2e (expect ~1e-4)" % (d3/max(d5, 1e-300)))
for f, band in ((1e9, (0.50, 0.80)), (1e10, (0.60, 0.88))):
    e0 = res[('off', f)].real - REF[f]
    e1 = res[('on', f)].real - REF[f]
    ratio = e1/e0
    check("AC improvement f=%.0e in [%.2f, %.2f]" % (f, *band),
          band[0] <= ratio <= band[1] and e1*e0 > 0,
          "err %.4g -> %.4g (x%.2f)" % (e0, e1, ratio))

# ---------------------------------------------------------- dx/W = 1/3
m3, M3 = build(0.375)
S3o = eq.EquiTerminalSolver(m3, M3, 0)
S3 = eq.EquiTerminalSolver(m3, M3, 0, enrich=dict(families=['corner']))
check("dx/W=1/3 modes built", S3.nu == 6, "nu=%d" % S3.nu)
for f in (1e9, 1e10):
    Z0, _, i0 = S3o.solve(f, rtol=1e-8)
    Z1, _, i1 = S3.solve(f, rtol=1e-8)
    ok = (i0['flag'] == 0 and i1['flag'] == 0
          and i0['residual'] <= 1e-7 and i1['residual'] <= 1e-7)
    check("dx/W=1/3 converged f=%.0e" % f, ok)
    # improvement: closer to the (finer-rung) truth direction
    better = abs(Z1.real - REF[f]) < abs(Z0.real - REF[f])
    check("dx/W=1/3 improves f=%.0e" % f, better,
          "R %.5g -> %.5g (ref %.5g)" % (Z0.real, Z1.real, REF[f]))

# ------------------------------------------- phase 2: engine composition
# coarse conduction engine (k=7, boundary-only) + corner modes via
# ModeStack, with engine-baseline tables (single-axis engine analog in
# the tabulation). Improvement bands measured 2026-08-19: the corner
# modes cut the engine's REMAINING error to ~0.53x at both 1e9 and
# 1e10 (dx/W = 1/2).
Se = eq.EquiTerminalSolver(m, M, 0, enrich=dict(k=7, f_ref=1e10))
Sc = eq.EquiTerminalSolver(m, M, 0, enrich=dict(
    families=['section', 'corner'], k=7, f_ref=1e10))
check("composed stack built", Sc.nu == Se.nu + 6,
      "nu %d -> %d" % (Se.nu, Sc.nu))
st = Sc.redist
check("engine<->corner cross block present",
      st.cross[0, 1].shape == (Se.nu, 6) and st.cross[0, 1].nnz > 0)

cres = {}
for f in (1e5, 1e9, 1e10):
    Z, _, info = Sc.solve(f, rtol=1e-8)
    # gate on the TRUE residual, not the lgmres flag (the C.2 v1
    # lesson, doctrine rule 14): with the 2026-08-20 individual-corner
    # palette the quasi-DC mode system carries near-degenerate columns
    # and lgmres can stall at the fp32-precond floor AFTER the answer
    # converged (measured: flag 10 at resid 5.2e-8, f = 1e5).
    check("composed converged f=%.0e" % f,
          info['residual'] <= 1e-7,
          "flag %s resid %.1e mv %d" % (info['flag'],
                                        info['residual'],
                                        info['matvecs']))
    cres[f] = Z
check("composed DC decoupling",
      abs(cres[1e5].real - res[('off', 1e5)].real)
      < 1e-6*res[('off', 1e5)].real)
for f, band in ((1e9, (0.40, 0.70)), (1e10, (0.40, 0.70))):
    Zeng, _, ie = Se.solve(f, rtol=1e-8)
    ok_e = ie['flag'] == 0 and ie['residual'] <= 1e-7
    check("engine-only converged f=%.0e" % f, ok_e,
          "flag %s resid %.1e" % (ie['flag'], ie['residual']))
    e0, e1 = Zeng.real - REF[f], cres[f].real - REF[f]
    ratio = e1/e0
    check("composed improvement f=%.0e in [%.2f, %.2f]" % (f, *band),
          band[0] <= ratio <= band[1] and e1*e0 > 0,
          "err %.4g -> %.4g (x%.2f)" % (e0, e1, ratio))

print("\n%s" % ("ALL OK" if nfail == 0 else "%d FAILURES" % nfail))
sys.exit(1 if nfail else 0)
