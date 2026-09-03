# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate equiterminal.py: equipotential port terminals as unknowns.

PART A -- AGAINST THE DENSE ORACLE. ``dense_oracle`` solves the same
physics directly and densely, with no FMM, no mesh basis and no Krylov
tolerance, so it certifies the production path's operator and topology.
Agreement is expected at FMM-truncation level, not machine precision.
Also compares the SOLVED terminal current split, which is the quantity
the old prescribed model had to guess.

PART B -- AGAINST VOXHENRY on the straight-conductor control, whose
reference values ship with VoxHenry itself. This is the regression that
matters: the control is the geometry the prescribed model was tuned
against, and the equipotential model must not lose ground on it. (It
does not -- it gains: R goes 0.967 -> 1.000.)

PART C -- KCL AT THE PORT NODE. The solved split must sum to exactly
+1 over the P faces and -1 over the N faces. This is the invariant the
prescribed model imposed by construction and this one has to earn.

The module's own build-time guards (incidence rows are +-1 pairs,
filament orientation is uniform, cycle-space dimension, and every basis
column divergence-free) fire as exceptions and are exercised by simply
constructing the solver here.

Run inside the toolbox:  python3 validate_equiterminal.py
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


import sys

import numpy as np

import vhr
import equiterminal as eq
import stencils as st

SETUP3 = 'validation/setups_vhr/setup3_k1.vhr'
STRAIGHT = ('VoxHenry/Input_files/'
            'straight_cond1_len30.0u_wid10.0u_dist20.0u-two_freq.vhr')
# VoxHenry's own shipped reference values for the straight conductor
VH = {2.5e9: (0.011236, 9.90236e-12), 1e10: (0.0200206, 9.61477e-12)}

fails = []


def check(tag, cond, detail=''):
    print("    %-4s %s  %s" % ('ok' if cond else 'FAIL', tag, detail))
    if not cond:
        fails.append(tag)
    return cond


def build(path, freq):
    m = vhr.read_vhr(path)
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, freq)
    return m, M


def part_a():
    print("\nPART A -- vs the dense oracle (%s)" % SETUP3)
    import dense_oracle as do
    m, M = build(SETUP3, 1e9)
    solver = eq.EquiTerminalSolver(m, M, 0)
    geom, port, _ = do.from_vhr(SETUP3)
    ref = do.System(geom, port, 'exact', 'equipotential')
    npos = len(port.pos)
    for f in (1e8, 1e9, 1e10):
        w = 2*np.pi*f
        Z, i, info = solver.solve(f)
        Zo, io, _ = ref.solve(f)
        u = np.abs(solver.terminal_split(i))
        uo = np.abs(ref.terminal_currents(io))
        dr = abs(Z.real/Zo.real - 1.0)
        dl = abs((Z.imag/w)/(Zo.imag/w) - 1.0)
        du = np.max(np.abs(u - uo))/uo.max()
        check("R  @%.0e" % f, dr < 5e-3, "rel %.2e" % dr)
        check("L  @%.0e" % f, dl < 5e-3, "rel %.2e" % dl)
        check("split @%.0e" % f, du < 5e-2,
              "max rel %.2e, crowding %.1f:1"
              % (du, uo[npos:].max()/uo[npos:].min()))


def part_b():
    print("\nPART B -- vs VoxHenry, straight-conductor control")
    m, M = build(STRAIGHT, 2.5e9)
    solver = eq.EquiTerminalSolver(m, M, 0)
    for f in sorted(VH):
        Rv, Lv = VH[f]
        w = 2*np.pi*f
        Z, i, info = solver.solve(f)
        dr = abs(Z.real/Rv - 1.0)
        dl = abs((Z.imag/w)/Lv - 1.0)
        check("R @%.1e" % f, dr < 5e-3, "ratio %.4f" % (Z.real/Rv))
        check("L @%.1e" % f, dl < 5e-3, "ratio %.4f" % ((Z.imag/w)/Lv))


def part_c():
    print("\nPART C -- KCL at the port node")
    m, M = build(SETUP3, 1e9)
    solver = eq.EquiTerminalSolver(m, M, 0)
    _, i, _ = solver.solve(1e9)
    u = solver.terminal_split(i)
    pol = solver.term.pol
    sp_ = complex(u[pol > 0].sum())
    sn = complex(u[pol < 0].sum())
    check("P faces sum to +1", abs(sp_ - 1.0) < 1e-6, "%.12g" % abs(sp_ - 1))
    check("N faces sum to -1", abs(sn + 1.0) < 1e-6, "%.12g" % abs(sn + 1))


def part_d():
    """PART D -- the FMM coupling must reproduce the dense one.

    ``fmm=True`` replaces the dense terminal<->interior block by the
    tree's own near/far split: direct inside the 27-box neighbourhood,
    and the terminal as an off-lattice point source of weight
    ``i_t*t_l/l_axis`` through the M2L ladder outside it. Agreement is
    expected at FMM-truncation level, and the OPERATOR is compared
    directly (not just the solved Z) because a near/far double count or
    gap shows up there first and can hide in an integrated quantity.
    """
    print("\nPART D -- FMM coupling vs the dense block")
    for path, freq in ((SETUP3, 1e9), (STRAIGHT, 2.5e9)):
        m, M = build(path, freq)
        a = eq.EquiTerminalSolver(m, M, 0, fmm=False)
        b = eq.EquiTerminalSolver(m, M, 0, fmm=True)
        tag = path.split('/')[-1][:22]
        rng = np.random.default_rng(0)
        n = a.efg + a.term.n
        x = rng.standard_normal(n) + 1j*rng.standard_normal(n)
        ya, yb = a.apply_Z(x), b.apply_Z(x)
        ef = (np.linalg.norm(yb[:a.efg] - ya[:a.efg])
              / np.linalg.norm(ya[:a.efg]))
        et = (np.linalg.norm(yb[a.efg:] - ya[a.efg:])
              / np.linalg.norm(ya[a.efg:]))
        check("op filament rows %s" % tag, ef < 1e-3, "rel %.2e" % ef)
        check("op terminal rows %s" % tag, et < 1e-2, "rel %.2e" % et)
        za, _, _ = a.solve(freq)
        zb, _, _ = b.solve(freq)
        dz = abs(zb - za)/abs(za)
        check("Z %s" % tag, dz < 1e-4,
              "rel %.2e, near %.1f%% of the dense block"
              % (dz, 100*b.coupler.near_frac))


def part_e():
    """PART E -- ARBITRARY TERMINAL LENGTH, against DC in closed form.

    A terminal runs from the port reference plane to the node it feeds,
    so with both ends closed the modelled conductor is
    ``(L-1)*dx + 2*t_l`` long -- ``t_l`` IS the reference-plane position.
    On a uniform bar the DC resistance of that is exactly
    ``length/(sigma*A)``, which pins BOTH the generalised resistance and
    the geometry (a terminal that started at the cell face instead of
    ending at the node would give the same answer only at t_l = dx/2,
    which is precisely how that bug hid).
    """
    print("\nPART E -- arbitrary t_l vs the DC closed form")
    m, M = build(STRAIGHT, 1.0)
    dx = m.dx
    ncell = int(m.dims[0])
    area = (int(m.dims[1])*dx)*(int(m.dims[2])*dx)
    sigma = m.uniform_sigma()
    for tau in (0.25, 0.5, 0.75):
        t_l = tau*dx
        solver = eq.EquiTerminalSolver(m, M, 0, t_l=t_l)
        Z, _, _ = solver.solve(0.0)
        want = ((ncell - 1)*dx + 2*t_l)/(sigma*area)
        r = Z.real/want
        check("R_dc t_l=%.2f dx" % tau, abs(r - 1) < 1e-9,
              "%.12g vs %.12g, ratio %.10f" % (Z.real, want, r))


def part_h():
    """PART H -- galvanically isolated multi-conductor models.

    The spanning FOREST (one tree per component) replaced the single
    spanning tree 2026-08-05; the cycle-space count uses the augmented
    graph's true component count, and the undriven conductor's eddy
    currents are spanned by its own plaquettes. Checks: DC against the
    closed form (the floating conductor must contribute NOTHING at DC),
    port symmetry on the identical-conductor pair, the eddy-coupling
    signature against the single-conductor control (a floating
    neighbour ADDS loss and SHEDS inductance at AC), and the
    no-return-path guard (P and N on different conductors must be
    rejected, not least-squares'd into a non-answer).
    """
    print("\nPART H -- multi-conductor (spanning forest)")
    two = ('VoxHenry/Input_files/'
           'straight_cond2_len30.0u_wid10.0u_dist20.0u.vhr')
    f = 2.5e9
    m, M = build(two, f)
    zs = {}
    for port in (0, 1):
        S = eq.EquiTerminalSolver(m, M, port)
        check("ncomp == 2 (port %d)" % port, S.ncomp == 2,
              "%d components" % S.ncomp)
        zs[port], _, _ = S.solve(f)
    dz = abs(zs[0] - zs[1])/abs(zs[0])
    check("port symmetry", dz < 1e-3, "rel %.2e" % dz)
    S = eq.EquiTerminalSolver(m, M, 0)
    Zdc, _, _ = S.solve(0.0)
    A = (float(np.asarray(m.struc()).sum())/m.dims[0]/2)*m.dx**2
    want = ((m.dims[0] - 1)*m.dx + m.dx)/(m.uniform_sigma()*A)
    check("DC closed form (floating neighbour inert)",
          abs(Zdc.real/want - 1.0) < 1e-9,
          "%.10g vs %.10g" % (Zdc.real, want))
    m1, M1 = build(STRAIGHT, f)
    Z1, _, _ = eq.EquiTerminalSolver(m1, M1, 0).solve(f)
    w = 2*np.pi*f
    check("eddy neighbour: R up, L down",
          zs[0].real > Z1.real and zs[0].imag/w < Z1.imag/w,
          "R %.6g vs %.6g, L %.6g vs %.6g pH"
          % (zs[0].real, Z1.real, zs[0].imag/w*1e12, Z1.imag/w*1e12))
    # mixed materials through the equiterminal path: Cu_Al has one
    # conductor (and port) per metal, so each port's DC resistance is
    # its own closed form -- Terminals must take the PORT's sigma
    # (port_sigma), not a global one that does not exist here.
    cual = ('VoxHenry/Input_files/'
            'straight_cond2_len30.0u_wid10.0u_dist20.0u_Cu_Al.vhr')
    mc, Mc2 = build(cual, f)
    A2 = (float(np.asarray(mc.struc()).sum())/mc.dims[0]/2)*mc.dx**2
    for port in (0, 1):
        Sd = eq.EquiTerminalSolver(mc, Mc2, port)
        Zd, _, _ = Sd.solve(0.0)
        wantd = ((mc.dims[0] - 1)*mc.dx + mc.dx)/(mc.port_sigma(port)*A2)
        check("Cu_Al DC port %d" % port,
              abs(Zd.real/wantd - 1.0) < 1e-9,
              "%.10g vs %.10g (sigma %.3g)"
              % (Zd.real, wantd, mc.port_sigma(port)))

    # no return path: P on one bar, N on the other, nothing in between
    import os
    struc = np.zeros((8, 6, 2), dtype=np.int8)
    struc[:, 0:2, :] = 1
    struc[:, 4:6, :] = 1
    ports = [('p1', 'P', 0, j, k, '-x') for j in (0, 1) for k in (0, 1)]
    ports += [('p1', 'N', 7, j, k, '+x') for j in (4, 5) for k in (0, 1)]
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'studies', 'noreturn.vhr')
    vhr.write_vhr(p, struc, 1e-6, 5.8e7, (f,), ports)
    mg, Mg = build(p, f)
    try:
        eq.EquiTerminalSolver(mg, Mg, 0)
        check("no-return-path raises", False, "went through")
    except ValueError as e:
        check("no-return-path raises", 'return path' in str(e),
              str(e)[:60])


def main():
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_h()
    print("\n%d checks failed" % len(fails))
    if fails:
        print("  " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
