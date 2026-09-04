# SPDX-License-Identifier: MIT
"""L-TRACE REFINEMENT LADDER: size the corner error before any corner-mode work.

THE DE-RISK QUESTION (corner program, docket 2026-08-14): the proposed
bend/corner modes only earn their design cost if the corner-attributable
discretisation error is material at practical resolutions. Measure it
first.

METHOD -- differential ladder. Two geometries through the SAME halving
refinement ladder, same cross-section (W x T), same total centreline
length, same solver route (equiterminal, engine off):
  * Z-trace: three arms joined at TWO 90-degree corners (the
    equiterminal port model requires both terminal faces on one axis,
    so a single-corner L is not portable -- two corners double the
    signal and keep both ports on x faces);
  * straight control bar of the identical centreline length.
The control's convergence carries the cross-section / port / staircase
error common to both; the DRIFT of the difference dZ(h) = Z_L - Z_str
across the ladder is the corner-attributable error, to first order.
Reported per frequency: R, L per geometry, dR, dL, observed order p and
Richardson extrapolate for each, and the corner error at each rung.

ANCHOR (low f): conformal mapping gives ~0.56 effective squares for a
right-angle bend in an equal-width strip (Hall). With the centreline
convention (each arm measured to the corner CENTRE) the straight
equivalent counts each corner as exactly 1 square, so with 2 corners
    dR(DC) ~ 2 * (0.56 - 1) * rho/T   (~ -1.9 mOhm here).
The extrapolated low-f dR should land near this; the ladder's job is
the HIGH-f (skin regime) number, where no analytic anchor exists and
the corner modes would actually act.

STANDING RULE (C.2 ladder lesson): ladder measurements without
convergence flags are void -- every row prints matvecs/flag/residual and
rows with flag != 0 AND residual above RESID_VOID are marked VOID.

Usage:
    PYTHONPATH=src python3 studies/ltrace_ladder.py
Env: LADDER="1,2,4" (cells per um; halving triples give clean
     Richardson), FLOW=1e5, FHIGH=1e10, RTOL=1e-8, W_UM=8, T_UM=8,
     ARM_UM=32, OUTDIR (default scratchpad-ish ./ltrace_out).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import vhr
import equiterminal as eq
from enrich import skin_depth

SIG = 5.8e7
RHO = 1.0/SIG
W_UM = float(os.environ.get('W_UM', 8))
T_UM = float(os.environ.get('T_UM', 8))
ARM_UM = float(os.environ.get('ARM_UM', 32))
RTOL = float(os.environ.get('RTOL', 1e-8))
RESID_VOID = 1e-6
LADDER = [float(s) for s in os.environ.get('LADDER', '1,2,4').split(',')]
OUTDIR = os.environ.get('OUTDIR', os.path.join(os.getcwd(), 'ltrace_out'))
os.makedirs(OUTDIR, exist_ok=True)
FREQS = tuple(float(s)
              for s in os.environ.get('FREQS', '1e5,1e10').split(','))


def build_L(nper):
    """Z-trace: arm A along x (low y), riser along y, arm C along x
    (high y) extending to the +x face. Both ports on x faces."""
    dx = 1e-6/nper
    W, T = int(round(W_UM*nper)), int(round(T_UM*nper))
    A = int(round(ARM_UM*nper))          # arm-A x extent = riser y extent
    nx = int(round((2*ARM_UM - W_UM)*nper))
    struc = np.zeros((nx, A, T), dtype=np.int8)
    struc[:A, :W, :] = 1                 # arm A
    struc[A - W:A, :, :] = 1             # riser
    struc[A - W:, A - W:, :] = 1         # arm C
    ports = ([('p1', 'P', 0, j, k, '-x') for j in range(W)
              for k in range(T)]
             + [('p1', 'N', nx - 1, j, k, '+x')
                for j in range(A - W, A) for k in range(T)])
    path = os.path.join(OUTDIR, 'lt_Z_%g.vhr' % nper)
    vhr.write_vhr(path, struc, dx, SIG, FREQS, ports)
    return path, dx, int(struc.sum())


def build_straight(nper):
    """Control bar: same centreline length, same W x T."""
    dx = 1e-6/nper
    W, T = int(round(W_UM*nper)), int(round(T_UM*nper))
    nx = int(round((3*ARM_UM - 2*W_UM)*nper))
    struc = np.ones((nx, W, T), dtype=np.int8)
    ports = [(('p1', 'P', 0, j, k, '-x') if s == 0 else
              ('p1', 'N', nx - 1, j, k, '+x'))
             for s in (0, 1) for j in range(W) for k in range(T)]
    path = os.path.join(OUTDIR, 'lt_S_%g.vhr' % nper)
    vhr.write_vhr(path, struc, dx, SIG, FREQS, ports)
    return path, dx, int(struc.sum())


CORNER = os.environ.get('CORNER_MODES', '0') == '1'
ENGINE = os.environ.get('ENGINE', '0') == '1'


def run(path):
    """Solve both frequencies; return {freq: (Z, matvecs, flag, resid)}."""
    m = vhr.read_vhr(path)
    M = m.build_tree()
    m.prepare(M, FREQS[0])
    # CORNER_MODES=1: corner modes on the Z-trace (the straight control
    # has no corners, so the flag is harmless there).
    # ENGINE=1: the shipped conduction engine (k=7, boundary-only) on
    # BOTH geometries -- with CORNER_MODES too this is the phase-2
    # composed acceptance, and the dZ differential is meaningful again.
    fam = ['corner'] if CORNER else []
    if ENGINE:
        fam.append('section')
    kw = dict(enrich=dict(families=fam, k=7, f_ref=max(FREQS))) if fam else {}
    S = eq.EquiTerminalSolver(m, M, 0, **kw)
    out = {}
    for f in FREQS:
        Z, _, info = S.solve(f, rtol=RTOL)
        out[f] = (Z, info['matvecs'], info['flag'], info['residual'])
    del S, M
    return out, m.dims


def order_extrap(hs, qs):
    """Observed order + Richardson from the three finest rungs.

    Halving ladders get the closed form; other ratios solve for p by
    bisection. Returns (p, q0) or (None, qs[-1]) when the differences
    are non-monotone (noise floor)."""
    if len(hs) < 3:
        return None, (qs[-1] if qs else float('nan'))
    (h1, h2, h3), (q1, q2, q3) = hs[-3:], qs[-3:]
    d12, d23 = q1 - q2, q2 - q3
    if d23 == 0 or d12/d23 <= 1.0:
        return None, qs[-1]

    def g(p):
        return (h1**p - h2**p)/(h2**p - h3**p) - d12/d23

    lo, hi = 0.05, 8.0
    if g(lo)*g(hi) > 0:
        return None, qs[-1]
    for _ in range(200):
        mid = 0.5*(lo + hi)
        if g(lo)*g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    p = 0.5*(lo + hi)
    q0 = q3 - d23/((h2/h3)**p - 1.0)
    return p, q0


print("Z-trace corner ladder (2 corners): W=%g T=%g um, arms %g um, "
      "centreline %g um, copper"
      % (W_UM, T_UM, ARM_UM, 3*ARM_UM - 2*W_UM))
for f in FREQS:
    d = skin_depth(SIG, f)
    print("  f=%.3g Hz: delta=%.4g um, W/delta=%.2f"
          % (f, d*1e6, W_UM*1e-6/d))
print("ladder (cells/um): %s   rtol %g\n" % (LADDER, RTOL))
print("%-4s %-5s %-11s %-8s | %-10s %-12s %-11s | %s"
      % ("geo", "n/um", "grid", "cells", "freq", "R[Ohm]", "L[H]",
         "mv/flag/resid"))

res = {}          # (geo, nper, freq) -> (R, L, void)
for nper in LADDER:
    for geo, builder in (('L', build_L), ('S', build_straight)):
        path, dx, ncell = builder(nper)
        t0 = time.perf_counter()
        try:
            out, dims = run(path)
        except Exception as e:
            print("%-4s %-5d FAILED %s: %s"
                  % (geo, nper, type(e).__name__, str(e)[:70]), flush=True)
            continue
        for f in FREQS:
            Z, mv, flag, resid = out[f]
            void = bool(flag) and resid > RESID_VOID
            R, L = Z.real, Z.imag/(2*np.pi*f)
            res[(geo, nper, f)] = (R, L, void)
            print("%-4s %-5g %-11s %-8d | %-10.3g %-12.6g %-11.5g | "
                  "%d/%s/%.1e%s  (%.1f s)"
                  % (geo, nper, "x".join(map(str, dims)), ncell, f, R, L,
                     mv, flag, resid, "  VOID" if void else "",
                     time.perf_counter() - t0), flush=True)

sheet = RHO/(T_UM*1e-6)
print("\n== differential dZ = Z_L - Z_straight (corner increment) ==")
print("low-f anchor: dR ~ 2*(0.56-1)*rho/T = %.4g Ohm" % (-0.88*sheet))
for f in FREQS:
    print("\nfreq %.3g Hz" % f)
    print("%-5s %-13s %-13s %-10s" % ("n/um", "dR[Ohm]", "dL[H]", "sq_eff"))
    hs, dRs, dLs, ns = [], [], [], []
    for nper in LADDER:
        a = res.get(('L', nper, f))
        b = res.get(('S', nper, f))
        if not a or not b or a[2] or b[2]:
            continue
        dR, dL = a[0] - b[0], a[1] - b[1]
        hs.append(1.0/nper)
        ns.append(nper)
        dRs.append(dR)
        dLs.append(dL)
        print("%-5g %-13.5g %-13.5g %-10.4f"
              % (nper, dR, dL, 1.0 + dR/(2*sheet)))
    if not hs:
        print("  (no valid rung pairs)")
        continue
    for name, qs, scale in (("dR", dRs, "Ohm"), ("dL", dLs, "H")):
        p, q0 = order_extrap(hs, qs)
        if p is None:
            print("  %s: no clean order (noise floor or <3 rungs); "
                  "finest = %.5g %s" % (name, q0, scale))
            continue
        print("  %s: order p=%.2f, extrapolate %.5g %s" % (name, p, q0,
                                                           scale))
        for h, q, nper in zip(hs, qs, ns):
            aL = res.get(('L', nper, f))
            base = abs(aL[0] if name == "dR" else aL[1])
            print("    h=%-6.3f corner err %.3g %s  (%.2f%% of Z_L, "
                  "%.1f%% of the corner increment)"
                  % (h, q - q0, scale, 100*abs(q - q0)/base,
                     100*abs((q - q0)/q0) if q0 else float('nan')))

print("\n== per-geometry convergence (context) ==")
for f in FREQS:
    for geo in ('L', 'S'):
        sel = [n for n in LADDER if (geo, n, f) in res
               and not res[(geo, n, f)][2]]
        hs = [1.0/n for n in sel]
        if not hs:
            continue
        for name, idx in (("R", 0), ("L", 1)):
            qs = [res[(geo, n, f)][idx] for n in sel]
            p, q0 = order_extrap(hs, qs)
            tag = ("p=%.2f -> %.6g" % (p, q0)) if p is not None else \
                ("no clean order, finest %.6g" % q0)
            print("  %s %s @ %.3g Hz: %s" % (geo, name, f, tag))
print("\ndone.", flush=True)
