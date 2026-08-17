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


def part_f():
    """PART F -- the skin-effect engine's mode bases (subdivide > 1).

    Constructing ``Redistribution`` already runs its two build-time
    guards -- the aggregate identity ``W_agg^T Z_sub W_agg == Z_full``
    and net-zero mode weights -- as exceptions; they are re-asserted
    here explicitly so a loosened guard cannot pass silently, and for
    ALL THREE bases, since each fills ``W`` by a different rule.

    The conduction basis gets a SPAN assertion on top: the shapes it was
    built from (the Daniel/Sangiovanni-Vincentelli/White face and corner
    exponentials at the build skin depth) are re-derived from the solver's
    OWN sub-bar geometry (``_sub_boxes`` centroids), not from the index
    arithmetic ``conduction_weights`` uses, and must sit in span(W) to
    the pruning tolerance. A wrong face-distance mapping, a swapped
    split axis, or a dropped independent column shows up as a residual.
    """
    print("\nPART F -- skin-effect mode bases (subdivide > 1)")
    freq = 1e10
    m, M = build(STRAIGHT, freq)
    delta = eq.skin_depth(m.uniform_sigma(), freq)
    cond = None
    for basis, k in (('diff', 3), ('linear', 4), ('conduction', 6)):
        s = eq.EquiTerminalSolver(m, M, 0, subdivide=k, mode_basis=basis,
                                  skin_freq=freq)
        r = s.redist
        nz = np.abs(r.W.sum(axis=0)).max()
        check("net-zero  %s" % basis, nz < 1e-12, "max colsum %.2e" % nz)
        check("aggregate %s" % basis, r.aggregate_err < 1e-10,
              "rel %.2e" % r.aggregate_err)
        if basis == 'conduction':
            cond = s
    r = cond.redist
    lo, hi = r.lo[:r.k], r.hi[:r.k]
    c = 0.5*(lo + hi)
    x = c[:, r.split[0]] - lo[:, r.split[0]].min()
    y = c[:, r.split[1]] - lo[:, r.split[1]].min()
    dx, p = m.dx, (1 + 1j)/delta
    pc = (1 + 1j)/(delta*np.sqrt(2.0))
    shapes = np.stack(
        [np.exp(-p*x) + np.exp(-p*(dx - x))
         + np.exp(-p*y) + np.exp(-p*(dx - y)),
         np.exp(-p*x) - np.exp(-p*(dx - x)),
         np.exp(-p*y) - np.exp(-p*(dx - y)),
         np.exp(-pc*(x + y)) + np.exp(-pc*(x + dx - y))
         + np.exp(-pc*(dx - x + y)) + np.exp(-pc*(2*dx - x - y))],
        axis=1)
    P = np.concatenate([shapes.real, shapes.imag], axis=1)
    P -= P.mean(axis=0, keepdims=True)            # the net-zero parts
    keep = np.linalg.norm(P, axis=0) > 1e-10
    P = P[:, keep]/np.linalg.norm(P[:, keep], axis=0)
    res = P - r.W @ np.linalg.lstsq(r.W, P, rcond=None)[0]
    rr = np.linalg.norm(res, axis=0).max()
    check("span conduction", rr < 1e-6,
          "max residual %.2e over %d shapes, km=%d" % (rr, P.shape[1], r.km))


def part_g():
    """PART G -- frequency-dependent conduction modes (set_frequency).

    The conduction W depends on the skin depth, so ``solve`` retunes it
    to the solve frequency on top of the cached geometric tables. The
    check with teeth: a solver built at the WRONG frequency and retuned
    must match a solver built fresh at the right one -- same W, same
    folded blocks, same Z. Retuning from a LOW frequency also changes
    km (the delta-dependent pruning), so the augmented-system rebuild
    path is exercised, not just the re-fold. Both apply paths are
    compared: FFT via the solved Z, CSR by the assembled blocks
    themselves (bitwise -- same inputs, same arithmetic).
    """
    import time
    print("\nPART G -- conduction set_frequency vs fresh build")
    f_lo, f_hi = 1e7, 1e10
    m, M = build(STRAIGHT, f_hi)
    a = eq.EquiTerminalSolver(m, M, 0, subdivide=4, mode_basis='conduction',
                              skin_freq=f_lo)
    b = eq.EquiTerminalSolver(m, M, 0, subdivide=4, mode_basis='conduction',
                              skin_freq=f_hi)
    km_lo, nu_lo = a.redist.km, a.nu
    Za, _, _ = a.solve(f_hi)             # retunes W from f_lo to f_hi
    Zb, _, _ = b.solve(f_hi)
    check("km retuned", a.redist.km == b.redist.km and a.nu == b.nu,
          "km %d -> %d, nu %d -> %d" % (km_lo, a.redist.km, nu_lo, a.nu))
    dz = abs(Za - Zb)/abs(Zb)
    check("Z retuned == fresh (fft)", dz < 1e-10, "rel %.2e" % dz)
    t0 = time.perf_counter()
    a.redist.set_frequency(2.5e9)
    t_re = time.perf_counter() - t0
    print("         (retune %.3f s vs %.1f s first build)"
          % (t_re, b.t_setup))
    d = eq.EquiTerminalSolver(m, M, 0, subdivide=3, mode_basis='diff')
    check("diff is frequency-independent",
          not d.redist.set_frequency(2.5e9), "set_frequency no-op")

    # km CHANGE across the retune (the pruning is delta-dependent), so
    # nu shrinks and the augmented system -- mesh basis, Cholesky --
    # must rebuild. setup3's k=3 grid gives km=5 at 1e2 Hz, km=4 at
    # 1e9: cheap solves, and the path the STRAIGHT pair above misses.
    m3, M3 = build(SETUP3, 1e9)
    g1 = eq.EquiTerminalSolver(m3, M3, 0, subdivide=3,
                               mode_basis='conduction', skin_freq=1e2)
    nu0 = g1.nu
    Zg1, _, _ = g1.solve(1e9)
    g2 = eq.EquiTerminalSolver(m3, M3, 0, subdivide=3,
                               mode_basis='conduction', skin_freq=1e9)
    Zg2, _, _ = g2.solve(1e9)
    dz = abs(Zg1 - Zg2)/abs(Zg2)
    check("km-change rebuild", g1.nu == g2.nu and nu0 != g1.nu and dz < 1e-10,
          "nu %d -> %d, rel %.2e" % (nu0, g1.nu, dz))
    kw = dict(subdivide=3, mode_basis='conduction', use_fft=False,
              rc_uu=1, rc_cross=2)
    c1 = eq.EquiTerminalSolver(m3, M3, 0, skin_freq=f_lo, **kw)
    c1.redist.set_frequency(f_hi)
    c2 = eq.EquiTerminalSolver(m3, M3, 0, skin_freq=f_hi, **kw)
    du = np.abs(c1.redist.Zuu - c2.redist.Zuu).max()
    dc = np.abs(c1.redist.Zcross - c2.redist.Zcross).max()
    dt = np.abs(c1.redist.Zt - c2.redist.Zt).max() \
        if c2.redist.Zt is not None else 0.0
    dr = np.abs(c1.redist.Ru - c2.redist.Ru).max()
    check("csr blocks retuned == fresh", max(du, dc, dt, dr) == 0.0,
          "max |diff| Zuu %.1e Zcross %.1e Zt %.1e Ru %.1e"
          % (du, dc, dt, dr))


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
    part_f()
    part_g()
    part_h()
    print("\n%d checks failed" % len(fails))
    if fails:
        print("  " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
