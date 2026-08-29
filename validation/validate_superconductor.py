# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate the superconductor (two-fluid London) term.

SuperPEEC models superconductors exactly as VoxHenry does (its executer,
lines 237-254): each cell's series impedance density is the two-fluid
parallel combination of the normal channel ``sigma`` and the London
channel ``1/(j w mu lambda^2)``,

    z(w) = (sigma*(w mu lam^2)^2 + j w mu lam^2)/((sigma w mu lam^2)^2 + 1)

folded into the DIAGONAL R exactly like per-cell conductivity -- the
Toeplitz/FMM structure carries geometry only and is untouched. The
imaginary part of z is the KINETIC INDUCTANCE.

PART A -- EXACT KINETIC INDUCTANCE, in closed form. On a wire whose
cross-section is fully symmetric (2x2 cells), the current split is
pinned by symmetry no matter what lambda is, so the kinetic term is
EXACTLY mu*lam^2*l_eff/A and the geometric inductance cancels in a
lambda difference: L(lam2) - L(lam1) == mu*(lam2^2 - lam1^2)*l_eff/A,
with l_eff = (nx-1)*dx + 2*t_l as PART E of validate_equiterminal
pins. Lossless (sigma = 0), so R must also vanish to solver noise.

PART B -- NORMAL-METAL LIMIT. lambda -> inf gives z -> 1/sigma with
Im(z)/Re(z) = 1/(sigma w mu lam^2); at lambda = 1 m that is ~1e-12, so
a superconductor-flagged copper wire must match the plain solve to
~1e-9.

PART C -- THE TWO-FLUID CORPUS FILE (sigma = 5.8e8, lam = 50 nm). In
the kinetic-dominated regime R ~ sigma*(w mu lam^2)^2/den, so
R(1e10)/R(2.5e9) = 16*(den(2.5e9)/den(1e10)) = 15.8; the London
screening profile is frequency independent, so L must be nearly flat.

PART D -- LONDON SCREENING, measured MID-WIRE where the profile is
fully developed. TWO CALIBRATION TRAPS, both walked into and kept as
warnings: the PORT-face split is flattened by the equipotential
injection (1.36x where the bulk gives 1.53x), and the 1-D cosh slab
model over-predicts the ring/interior contrast (~2.1x) because in 2-D
the area weighting pulls the interior mean UP -- the cylinder analog
(J ~ I0(r/lambda), a = 2*lambda) predicts ~1.5-1.6x, which is what
the solver delivers. So the check is the part that is unambiguous:
the contrast must GROW as lambda shrinks (half-width 2*lambda ->
4*lambda), and sit in the cylinder-model window at each. J decays on
the lambda scale at ANY frequency -- what distinguishes a
superconductor from a normal metal at low f -- and it falls out of
the diagonal z(w) with no extra machinery.

PART E -- GUARDS. freq = 0 must raise (the superfluid shorts DC, the
operator would be singular); skin subdivision on a superconductor must
raise, INCLUDING subdivide=True (the `1 == True` trap).

Run inside the toolbox:  python3 validate_superconductor.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]


import os as _os
if not _os.path.isdir(_os.path.join(_os.path.dirname(
        _os.path.abspath(__file__)), 'VoxHenry')):
    print('SKIP: VoxHenry corpus not present -- this validator '
          'compares against VoxHenry shipped inputs/reference values. '
          'Place a VoxHenry checkout at validation/VoxHenry to enable it.')
    raise SystemExit(0)


import os
import sys

import numpy as np

import vhr
import equiterminal as eq
import stencils as st

SPD = os.path.dirname(os.path.abspath(__file__))
TWOFLUID = ('VoxHenry/Input_files/'
            'straight_cond1_len0.6u_wid0.2u-supercond_two_fluid.vhr')
MU0 = 4e-7*np.pi

fails = []


def check(tag, cond, detail=''):
    print("    %-4s %s  %s" % ('ok' if cond else 'FAIL', tag, detail))
    if not cond:
        fails.append(tag)
    return cond


def wire(tag, sigma, lambdaL, freq, nx=20, nt=2, dx=1e-8):
    """Synthetic nx x nt x nt wire; returns a solved-ready (m, M)."""
    struc = np.ones((nx, nt, nt), dtype=np.int8)
    ports = []
    for j in range(nt):
        for k in range(nt):
            ports.append(('p1', 'P', 0, j, k, '-x'))
            ports.append(('p1', 'N', nx - 1, j, k, '+x'))
    p = os.path.join(SPD, 'studies', 'sc_%s.vhr' % tag)
    vhr.write_vhr(p, struc, dx, sigma, (freq,), ports, lambdaL=lambdaL)
    m = vhr.read_vhr(p)
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, freq)
    return m, M


def part_a():
    print("\nPART A -- exact kinetic inductance (lossless, sigma = 0)")
    f = 2.5e9
    w = 2*np.pi*f
    nx, nt, dx = 20, 2, 1e-8
    out = {}
    for lam in (2e-8, 5e-8):
        m, M = wire('lossless_%g' % lam, 0.0, lam, f, nx, nt, dx)
        S = eq.EquiTerminalSolver(m, M, 0)
        Z, _, _ = S.solve(f)
        out[lam] = Z
        check("R ~ 0 at lam=%g" % lam, abs(Z.real) < 1e-8*abs(Z),
              "|R|/|Z| = %.2e" % (abs(Z.real)/abs(Z)))
    l_eff = (nx - 1)*dx + 2*(0.5*dx)
    area = (nt*dx)**2
    dL = (out[5e-8].imag - out[2e-8].imag)/w
    want = MU0*(5e-8**2 - 2e-8**2)*l_eff/area
    rel = abs(dL/want - 1.0)
    check("dL == mu*d(lam^2)*l/A", rel < 1e-8,
          "%.10g vs %.10g, rel %.2e" % (dL, want, rel))


def part_b():
    print("\nPART B -- normal-metal limit (lambda = 1 m)")
    f = 2.5e9
    mn, Mn = wire('normal', 5.8e7, None, f)
    Zn, _, _ = eq.EquiTerminalSolver(mn, Mn, 0).solve(f)
    ms, Ms = wire('biglam', 5.8e7, 1.0, f)
    Zs, _, _ = eq.EquiTerminalSolver(ms, Ms, 0).solve(f)
    rel = abs(Zs - Zn)/abs(Zn)
    check("Z(sc, lam=1) == Z(normal)", rel < 1e-9, "rel %.2e" % rel)


def part_c():
    print("\nPART C -- two-fluid corpus file (%s)" % TWOFLUID.split('/')[-1])
    m = vhr.read_vhr(TWOFLUID)
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, 2.5e9)
    S = eq.EquiTerminalSolver(m, M, 0)
    Z1, _, _ = S.solve(2.5e9)
    Z2, _, _ = S.solve(1e10)
    check("R > 0 both freqs", Z1.real > 0 and Z2.real > 0,
          "R = %.4g / %.4g" % (Z1.real, Z2.real))
    sig, lam = 5.8e8, 5e-8
    den = lambda f: (sig*2*np.pi*f*MU0*lam*lam)**2 + 1.0
    want = 16.0*den(2.5e9)/den(1e10)
    ratio = Z2.real/Z1.real
    check("R scales ~ w^2", abs(ratio/want - 1.0) < 0.15,
          "R(1e10)/R(2.5e9) = %.2f, two-fluid predicts %.2f"
          % (ratio, want))
    L1, L2 = Z1.imag/(2*np.pi*2.5e9), Z2.imag/(2*np.pi*1e10)
    check("L nearly flat", abs(L2/L1 - 1.0) < 0.02,
          "L ratio %.5f" % (L2/L1))


def _ring_ratio(S, i):
    """Boundary-ring / interior mean |J| over mid-wire x-filaments."""
    fa, fc = S.fil_axis, S.fil_cell
    mid = int(fc[fa == 0][:, 0].max())//2
    sel = np.flatnonzero((fa == 0) & (fc[:, 0] == mid))
    cur = np.abs(i[sel])
    y, z = fc[sel, 1], fc[sel, 2]
    edge = (y == y.min()) | (y == y.max()) \
        | (z == z.min()) | (z == z.max())
    return cur[edge].mean()/cur[~edge].mean()


def part_d():
    print("\nPART D -- London screening vs lambda, mid-wire")
    f = 1e10
    ratios = {}
    for lam, window in ((5e-8, (1.3, 2.2)), (2.5e-8, (1.8, 4.0))):
        m, M = wire('rod_%g' % lam, 5.8e8, lam, f, nx=24, nt=20)
        S = eq.EquiTerminalSolver(m, M, 0)
        _, i, _ = S.solve(f)
        r = ratios[lam] = _ring_ratio(S, i)
        check("ratio in window lam=%g" % lam,
              window[0] < r < window[1],
              "ring/interior |J| = %.2f (window %.1f-%.1f)"
              % (r, window[0], window[1]))
    check("contrast grows as lambda shrinks",
          ratios[2.5e-8] > 1.15*ratios[5e-8],
          "%.2f -> %.2f" % (ratios[5e-8], ratios[2.5e-8]))


def part_e():
    print("\nPART E -- guards")
    f = 2.5e9
    m, M = wire('guard', 0.0, 5e-8, f)
    try:
        m.prepare(M, 0.0)
        check("DC raises", False, "prepare(freq=0) went through")
    except ValueError as e:
        check("DC raises", 'freq > 0' in str(e), str(e)[:50])
    # The generic net-zero bases carry no London content, so they are
    # still refused on a superconductor; the CONDUCTION palette is not,
    # since 2026-08-29 its shapes take the Helmholtz rate 1/lambda
    # directly (studies/london_crowding.py measures what that buys).
    for sub in (3, True):
        try:
            eq.EquiTerminalSolver(m, M, 0, subdivide=sub)
            check("subdivide=%r raises on the generic basis" % sub,
                  False, "went through")
        except NotImplementedError:
            check("subdivide=%r raises on the generic basis" % sub,
                  True, "")
    try:
        S = eq.EquiTerminalSolver(m, M, 0, subdivide=3,
                                  mode_basis='conduction', skin_freq=f)
        p_used = S.redist._london_p if hasattr(S, 'redist') else None
        check("conduction basis is ALLOWED on uniform lambda",
              True, "rate 1/lambda = %.4g" % (p_used or 0.0))
        check("the London rate, not a skin depth, sets the shapes",
              p_used is not None and abs(p_used - 1.0/5e-8) < 1e-6*p_used,
              "%.6g vs %.6g" % (p_used or 0.0, 1.0/5e-8))
    except NotImplementedError as e:
        check("conduction basis is ALLOWED on uniform lambda", False,
              str(e)[:60])


def main():
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    print("\n%d checks failed" % len(fails))
    if fails:
        print("  " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
