# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The enrichment framework's own invariants (docs/enrichment_plan.md).

Phase 1 -- GEOMETRY AND TABLES, checked once for every family that
rides them (the skin engine, the film palette, subpixel modes, corner
modes and the partial-cell inductance correction all read the same
``Split``/``PairTables`` objects):

  A  Split.boxes: the whole-cell split IS the filament box, and any
     sub-prism split tiles it exactly (volumes sum, extents match), for
     cubic and anisotropic pitches and every orientation;
  B  PairTables against terminal.box_mutual_matrix on explicit boxes
     -- same kernel, different bookkeeping -- for two different splits
     at random separations (absorbs the old zuudiag/zcross studies:
     the mode-mode and mode-aggregate tables are exact);
  C  reciprocity of the tables: T_ab(D)[p, q] == T_ba(-D)[q, p];
  D  THE AGGREGATE IDENTITY: u' T u over a k x k split reproduces the
     undivided filament's mutual at every separation -- the volume
     integral over a cross-section is the sum over its pieces. This is
     what lets the net-zero modes ride on the existing Toeplitz/FMM
     near field untouched;
  E  the terminal bar split reproduces the equipotential terminal
     kernel's geometry (length t_l off the face, both signs);
  F  partial_dL: None on a whole-cell model, exactly symmetric, zero on
     whole-whole pairs, and one slab pair against first principles.

Phase 2 -- THE FAMILY (``Enrichment``), generic over its palettes:

  H  frequency policy: a solver built at the WRONG frequency and
     retuned matches one built fresh (same km, same Z on the FFT path,
     BITWISE blocks on the CSR path, including a km change that
     rebuilds the augmented system); a London rate does not move, so
     set_frequency returns False and only Ru scales, exactly with w;
  I  the London bar (studies/london_crowding.py): at two cells across
     a lossless 360 nm niobium bar the plain mesh returns EXACTLY the
     bulk kinetic inductance (symmetry pins the split) and the modes
     lift kinetic/bulk to ~1.42, toward the equal-area cylinder's 1.53.

Phase 3 -- COMPOSITION (``ModeStack``), on the corner Z-trace:

  J  per-entry net-zero of a prolonged (patch) palette, Zuu
     unconjugated symmetry, augmented-operator reciprocity y'(Zx) ==
     x'(Zy) with one family and with a two-family stack (exercises
     every Zcross / Zcross' and cross-block wiring), and the stack's
     mode count is the sum of its families'.

Run: PYTHONPATH=src python3 validation/validate_enrich.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np                                   # noqa: E402
import sppeec_threads as _spthreads                  # noqa: E402
_spthreads.enforce_blas()
from enrich import (Split, PairTables, unique_separations,  # noqa: E402
                    neighbour_pairs, partial_dL, slab_weights)
from terminal import box_mutual_matrix              # noqa: E402

FAIL = []


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


def rel(a, b):
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def main():
    rng = np.random.default_rng(3)
    d3 = np.array([1.0e-6, 2.0e-6, 0.5e-6])

    print('\nA -- Split.boxes tiles the filament box')
    for axis in range(3):
        whole = Split(axis, (1, 1, 1), d3)
        c = np.array([[3, 1, 2], [0, 4, 0]])
        lo, hi = whole.boxes(c)
        exp_lo = c*d3
        exp_hi = (c + 1)*d3
        exp_lo[:, axis] = (c[:, axis] + 0.5)*d3[axis]
        exp_hi[:, axis] = (c[:, axis] + 1.5)*d3[axis]
        check('whole-cell split is the filament box (axis %d)' % axis,
              np.array_equal(lo, exp_lo) and np.array_equal(hi, exp_hi))
        n = [int(v) for v in rng.integers(1, 5, size=3)]
        sub = Split(axis, n, d3)
        lo, hi = sub.boxes(c)
        vol = np.prod(hi - lo, axis=1).reshape(2, -1).sum(axis=1)
        vfull = np.prod(exp_hi - exp_lo, axis=1)
        ext_lo = lo.reshape(2, -1, 3).min(axis=1)
        ext_hi = hi.reshape(2, -1, 3).max(axis=1)
        check('%s split tiles it (axis %d)' % (tuple(n), axis),
              rel(vol, vfull) < 1e-14 and rel(ext_lo, exp_lo) < 1e-14
              and rel(ext_hi, exp_hi) < 1e-14 and sub.nsub == np.prod(n),
              'vol rel %.1e' % rel(vol, vfull))

    print('\nB -- PairTables vs box_mutual_matrix on explicit boxes')
    tables = PairTables()
    for axis, ka, kb in ((0, (3, 2), (1, 1)), (1, (2, 2), (2, 2)),
                         (2, (1, 3), (2, 1))):
        sa = Split.transverse(axis, ka, d3)
        sb = Split.transverse(axis, kb, d3)
        D = rng.integers(-3, 4, size=(6, 3))
        T = tables(sa, sb, D)
        worst = 0.0
        for i, dd in enumerate(D):
            lo_a, hi_a = sa.boxes(np.zeros((1, 3)))
            lo_b, hi_b = sb.boxes(dd[None, :])
            full = box_mutual_matrix(np.vstack([lo_a, lo_b]),
                                     np.vstack([hi_a, hi_b]), axis)
            worst = max(worst, rel(T[i], full[:sa.nsub, sa.nsub:]))
        check('tables == explicit boxes (axis %d, %s x %s)'
              % (axis, ka, kb), worst < 1e-12, 'rel %.1e' % worst)
    # caching: same request returns the same object
    sa = Split.transverse(0, (3, 2), d3)
    D = np.array([[1, 0, 0], [0, 2, -1]])
    check('separation-set cache hit', tables(sa, sa, D) is tables(sa, sa, D))
    try:
        tables(Split(0, (1, 1, 1), d3), Split(1, (1, 1, 1), d3), D)
        check('perpendicular splits refused', False)
    except ValueError:
        check('perpendicular splits refused', True)

    print('\nC -- reciprocity T_ab(D)[p, q] == T_ba(-D)[q, p]')
    sa = Split.transverse(1, (3, 1), d3)
    sb = Split.transverse(1, (2, 2), d3)
    D = rng.integers(-3, 4, size=(8, 3))
    Tab = tables(sa, sb, D)
    Tba = tables(sb, sa, -D)
    check('reciprocity', rel(Tab, np.transpose(Tba, (0, 2, 1))) < 1e-12,
          'rel %.1e' % rel(Tab, np.transpose(Tba, (0, 2, 1))))

    print('\nD -- the aggregate identity u\'Tu == whole-filament mutual')
    for axis, kk in ((0, (3, 3)), (2, (4, 2))):
        sub = Split.transverse(axis, kk, d3)
        whole = Split(axis, (1, 1, 1), d3)
        D = np.vstack([np.zeros((1, 3), dtype=int),
                       rng.integers(-4, 5, size=(7, 3))])
        u = np.full(sub.nsub, 1.0/sub.nsub)
        agg = np.einsum('p,dpq,q->d', u, tables(sub, sub, D), u)
        full = tables(whole, whole, D)[:, 0, 0]
        check('aggregate identity (axis %d, %s), self term included'
              % (axis, kk), rel(agg, full) < 1e-10, 'rel %.1e' % rel(agg, full))

    print('\nE -- terminal bar split')
    t_l = 0.3e-6
    for sign in (-1, +1):
        term = Split(0, (1, 1, 1), d3).terminal(t_l, sign)
        lo, hi = term.boxes(np.array([[2, 1, 1]]))
        mid = 2.5*d3[0]
        exp = (mid, mid + t_l) if sign > 0 else (mid - t_l, mid)
        check('terminal bar sign %+d: [%.2e, %.2e]' % (sign, *exp),
              lo[0, 0] == exp[0] and hi[0, 0] == exp[1])

    print('\nF -- partial_dL')
    import sppeec_input
    body = """
[grid]
dims  = [%d, %d, %d]
pitch = %g
%s

[[block]]
from_m = [0.0, 0.0, 0.0]
to_m   = [6.0e-7, 1.8e-7, 6.0e-8]
sigma  = 5.8e7

[[block]]
from_m = [0.0, 0.0, 1.2e-7]
to_m   = [6.0e-7, 1.8e-7, 1.95e-7]
sigma  = 5.8e7

[port]
p_faces = [%s]
n_faces = [%s]
equipotential = true

[solve]
freq = [1e10]
"""
    nx, ny, nz, zlo, zhi = 20, 6, 7, 4, 6
    pf = ", ".join('[0, %d, %d, "-x"]' % (j, k)
                   for j in range(ny) for k in range(zlo, zhi))
    nf = ", ".join('[%d, %d, %d, "+x"]' % (nx - 1, j, k)
                   for j in range(ny) for k in range(zlo, zhi))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pw = sppeec_input.loads(body % (nx, ny, nz, 3e-8, '', pf, nf))
        mw = pw.model()
    check('whole-cell model -> None', partial_dL(mw, None) is None)
    ps = sppeec_input.loads(body % (nx, ny, nz, 3e-8, 'subpixel = true',
                                    pf, nf))
    ms = ps.model()
    Ms = ps.tree(ms)
    dL = partial_dL(ms, Ms)
    check('slab model -> sparse dL', dL is not None and dL.nnz > 0,
          '' if dL is None else 'nnz %d' % dL.nnz)
    check('exactly symmetric', abs(dL - dL.T).max() == 0.0)
    # one pair from first principles: an x-filament in the partial top
    # layer against the x-filament directly below it
    from equiterminal import filament_cells
    fa, fc = filament_cells(Ms)
    fill = ms.slab_fill['fill']
    cz = int(ms.slab_fill['axis'])
    xs = np.flatnonzero(fa == 0)
    cells = fc[xs]
    part = [n for n, c in enumerate(cells)
            if 1e-12 < fill[c[0], c[1], c[2]] < 1.0 - 1e-12]
    n0 = part[0]
    c0 = cells[n0]
    below = c0.copy()
    below[cz] -= 1
    n1 = next(n for n, c in enumerate(cells) if np.array_equal(c, below))
    k = 8
    sub = Split(0, (1, 1, k), np.asarray(ms.d))
    w0 = slab_weights(fill[c0[0], c0[1], c0[2]], k)
    w1 = slab_weights(fill[below[0], below[1], below[2]], k)
    w1 = np.full(k, 1.0/k) if w1 is None else w1
    u = np.full(k, 1.0/k)
    lo0, hi0 = sub.boxes(c0[None, :])
    lo1, hi1 = sub.boxes(below[None, :])
    B = box_mutual_matrix(np.vstack([lo0, lo1]), np.vstack([hi0, hi1]),
                          0)[:k, k:]
    oracle = float(w0 @ B @ w1) - float(u @ B @ u)
    got = dL[int(xs[n0]), int(xs[n1])]
    check('slab pair vs first principles',
          abs(got - oracle) <= 1e-10*abs(oracle),
          '%.6e vs %.6e' % (got, oracle))
    whole = [n for n, c in enumerate(cells)
             if fill[c[0], c[1], c[2]] >= 1.0]
    ww = xs[whole]
    sub_ww = dL[ww][:, ww]
    check('whole-whole pairs carry no correction', sub_ww.nnz == 0
          or abs(sub_ww).max() == 0.0)

    print('\nG -- neighbour pairs and separations')
    cells = rng.integers(0, 6, size=(40, 3))
    cells = np.unique(cells, axis=0)
    fa_, fb_ = neighbour_pairs(cells, 2)
    dist = np.abs(cells[fa_] - cells[fb_]).max(axis=1)
    brute = sum(int(np.abs(cells - c).max(axis=1).__le__(2).sum())
                for c in cells)
    check('neighbour pairs: within radius, both orders, self included',
          dist.max() <= 2 and fa_.size == brute)
    D = cells[fb_] - cells[fa_]
    U, inv = unique_separations(D)
    check('unique separations invert', np.array_equal(U[inv], D)
          and U.shape[0] == len({tuple(r) for r in D}))

    print('\nH -- frequency policy (retune == fresh)')
    import equiterminal as eq
    faces = lambda i, sgn: ", ".join(                     # noqa: E731
        '[%d, %d, %d, "%s"]' % (i, j, k, sgn) for j in range(4)
        for k in range(4))
    bar = '\n'.join([
        '[grid]', 'dims = [24, 4, 4]', 'pitch = 10e-6', '[[block]]',
        'from = [0, 0, 0]', 'to = [24, 4, 4]', 'sigma = 5.8e7', '[port]',
        'equipotential = true', 'p_faces = [%s]' % faces(0, '-x'),
        'n_faces = [%s]' % faces(23, '+x'), '[solve]', 'freq = [1e10]'])
    pb = sppeec_input.loads(bar)
    mb = pb.model()
    Mb = pb.tree(mb)
    f_lo, f_hi = 1e2, 1e10
    mb.prepare(Mb, f_hi)
    a = eq.EquiTerminalSolver(mb, Mb, 0, subdivide=4, skin_freq=f_lo)
    b = eq.EquiTerminalSolver(mb, Mb, 0, subdivide=4, skin_freq=f_hi)
    km_lo, nu_lo = a.redist.km, a.nu
    Za, _, _ = a.solve(f_hi)             # retunes W from f_lo to f_hi
    Zb, _, _ = b.solve(f_hi)
    check('km retuned, augmented system rebuilt',
          a.redist.km == b.redist.km and a.nu == b.nu and nu_lo != a.nu,
          'km %d -> %d, nu %d -> %d' % (km_lo, a.redist.km, nu_lo, a.nu))
    check('Z retuned == fresh (FFT path)', abs(Za - Zb)/abs(Zb) < 1e-10,
          'rel %.2e' % (abs(Za - Zb)/abs(Zb)))
    kw = dict(subdivide=3, use_fft=False, rc_uu=1, rc_cross=2)
    c1 = eq.EquiTerminalSolver(mb, Mb, 0, skin_freq=f_lo, **kw)
    c1.redist.set_frequency(f_hi)
    c2 = eq.EquiTerminalSolver(mb, Mb, 0, skin_freq=f_hi, **kw)
    diffs = [abs(getattr(c1.redist, nm) - getattr(c2.redist, nm)).max()
             for nm in ('Zuu', 'Zcross', 'Zt', 'Ru')]
    check('CSR blocks retuned == fresh, bitwise', max(diffs) == 0.0,
          'max |diff| %s' % ' '.join('%.1e' % v for v in diffs))
    lb = bar.replace('sigma = 5.8e7', 'lambda_l = 90e-9').replace(
        'pitch = 10e-6', 'pitch = 200e-9')
    pl = sppeec_input.loads(lb)
    ml = pl.model()
    Ml = pl.tree(ml)
    ml.prepare(Ml, 1e9)
    L1 = eq.EquiTerminalSolver(ml, Ml, 0, subdivide=3, skin_freq=1e9)
    Ru1 = L1.redist.Ru.copy()
    moved = L1.redist.set_frequency(2e9)
    check('London rate does not move: set_frequency False, Ru ~ w exactly',
          moved is False and abs(L1.redist.Ru - 2*Ru1).max() == 0.0
          and np.imag(L1.redist._p) == 0.0)

    print('\nI -- the London bar: modes lift kinetic/bulk at two cells')
    MU0 = 4e-7*np.pi
    side, ln, lam, fr = 3.6e-7, 3.6e-6, 9e-8, 2.5e9
    nt = 2
    dxb = side/nt
    nx = int(round(ln/dxb))
    fb = lambda i, sgn: ", ".join(                        # noqa: E731
        '[%d, %d, %d, "%s"]' % (i, j, k, sgn) for j in range(nt)
        for k in range(nt))

    def imz(lam_v, modes):
        doc = '\n'.join([
            '[grid]', 'dims = [%d, %d, %d]' % (nx, nt, nt),
            'pitch = %g' % dxb, '[[block]]', 'from = [0, 0, 0]',
            'to = [%d, %d, %d]' % (nx, nt, nt), 'lambda_l = %g' % lam_v,
            '[port]', 'equipotential = true',
            'p_faces = [%s]' % fb(0, '-x'), 'n_faces = [%s]' % fb(nx - 1, '+x'),
            '[solve]', 'freq = [%g]' % fr])
        pz = sppeec_input.loads(doc)
        mz = pz.model()
        Mz = pz.tree(mz)
        mz.prepare(Mz, fr)
        kwz = dict(subdivide=3, skin_freq=fr) if modes else {}
        Z, _, _ = eq.EquiTerminalSolver(mz, Mz, 0, **kwz).solve(fr)
        return Z.imag
    # lambda difference against a SMALL lambda (1e-8: at 1e-9 the rate
    # 1/lambda prunes every mode column on a 180 nm cell), normalised
    # by the bulk difference so the plain, symmetry-pinned mesh reads
    # exactly 1
    w = 2*np.pi*fr
    lam2 = 1e-8
    dbulk = MU0*(lam**2 - lam2**2)*ln/(side*side)
    ratio = {md: (imz(lam, md) - imz(lam2, md))/w/dbulk for md in (0, 1)}
    check('plain mesh returns exactly bulk (symmetry-pinned split)',
          abs(ratio[0] - 1.0) < 2e-3, 'ratio %.4f' % ratio[0])
    check('modes lift kinetic/bulk toward the cylinder analog 1.53 '
          '(band 1.30..1.55)', 1.30 < ratio[1] < 1.55,
          'ratio %.4f' % ratio[1])

    print('\nJ -- composition on the corner Z-trace')
    import vhr
    W_UM, T_UM, ARM_UM, SIG = 8.0, 8.0, 32.0, 5.8e7
    nper = 0.25
    dxc = 1e-6/nper
    Wc, Tc = int(round(W_UM*nper)), int(round(T_UM*nper))
    Ac = int(round(ARM_UM*nper))
    nxc = int(round((2*ARM_UM - W_UM)*nper))
    struc = np.zeros((nxc, Ac, Tc), dtype=np.int8)
    struc[:Ac, :Wc, :] = 1
    struc[Ac - Wc:Ac, :, :] = 1
    struc[Ac - Wc:, Ac - Wc:, :] = 1
    ports = ([('p1', 'P', 0, j, k, '-x') for j in range(Wc)
              for k in range(Tc)]
             + [('p1', 'N', nxc - 1, j, k, '+x')
                for j in range(Ac - Wc, Ac) for k in range(Tc)])
    path = _op.path.join(_op.environ.get('TMPDIR', '/tmp'), 've_corner.vhr')
    vhr.write_vhr(path, struc, dxc, SIG, (1e9,), ports)
    mc = vhr.read_vhr(path)
    Mc = mc.build_tree()
    mc.prepare(Mc, 1e9)
    S1 = eq.EquiTerminalSolver(mc, Mc, 0, corner_modes=True)
    r = S1.redist
    check('patch palette: per-entry net-zero',
          np.abs(r.Wf.sum(axis=1)).max() < 1e-12)
    check('patch palette: 3 modes per corner',
          S1.nu == 3*len(r.palette.corners))
    check('Zuu unconjugated symmetry',
          abs(r.Zuu - r.Zuu.T).max() < 1e-10*abs(r.Zuu).max())
    Sc = eq.EquiTerminalSolver(mc, Mc, 0, subdivide=7, skin_freq=1e10,
                               corner_modes=True)
    Se = eq.EquiTerminalSolver(mc, Mc, 0, subdivide=7, skin_freq=1e10)
    check('stack mode count = engine + corners',
          Sc.nu == Se.nu + S1.nu and Sc.redist.cross[0, 1].nnz > 0)
    for S, tag in ((S1, 'corners only'), (Sc, 'engine + corners')):
        n = S.efg + S.term.n + S.nu
        x = rng.standard_normal(n) + 1j*rng.standard_normal(n)
        y = rng.standard_normal(n) + 1j*rng.standard_normal(n)
        lhs, rhs = y @ S.apply_Z(x), x @ S.apply_Z(y)
        check('augmented reciprocity, %s' % tag,
              abs(lhs - rhs) < 1e-9*abs(lhs),
              'rel %.1e' % (abs(lhs - rhs)/abs(lhs)))

    print('\n%d checks failed' % len(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    _sp.exit(main())
