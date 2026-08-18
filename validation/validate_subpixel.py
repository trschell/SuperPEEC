# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for subpixel stage A: the [[cylinder]] fill-fraction primitive.

Covers the geometry (fill areas match pi*R^2 across radii), the
schema contract (loud rejections), and the physics headline: on the
round-wire example's cross-section the center-in staircase reads DC
resistance -11.6% low (the recorded 24um-wire error class) while the
fill-corrected model lands within ~1.5% of L/(sigma*pi*R^2) -- with
zero solver changes (sigma_eff = sigma*fill rides the per-cell-
conductivity machinery). Lp remains full-cell until stage B.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import math

import numpy as np

FAIL = []


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


def wire_doc(R, dims_t=8, nx=96, cyl_extra=''):
    dx = 1e-6
    c = dims_t*dx/2.0
    fp, fn = [], []
    for i in range(dims_t):
        for j in range(dims_t):
            if ((i+0.5)*dx - c)**2 + ((j+0.5)*dx - c)**2 < R*R:
                fp.append('[0, %d, %d, "-x"]' % (i, j))
                fn.append('[%d, %d, %d, "+x"]' % (nx-1, i, j))
    return '\n'.join([
        '[grid]', 'dims = [%d, %d, %d]' % (nx, dims_t, dims_t),
        'pitch = 1e-6',
        '[[cylinder]]', 'axis = "x"',
        'center = [%g, %g]' % (c, c), 'radius = %g' % R,
        'sigma = 5.8e7', cyl_extra,
        '[port]', 'p_faces = [%s]' % ', '.join(fp),
        'n_faces = [%s]' % ', '.join(fn),
        '[solve]', 'freq = [1e5]'])


def expect_error(name, text, needle):
    import sppeec_input
    try:
        sppeec_input.loads(text).model()
        check(name, False, 'no exception')
    except ValueError as exc:
        check(name, needle in str(exc), str(exc)[:70])


def main():
    import sppeec_input

    # -- geometry: fill areas across radii ----------------------------
    for Rc in (2.5, 3.0, 5.5):
        R = Rc*1e-6
        dt = 14 if Rc > 4 else 8
        prob = sppeec_input.loads(wire_doc(R, dims_t=dt, nx=4))
        m = prob.model()
        area = float(m.fill_frac[0].sum())*1e-12
        rel = abs(area - math.pi*R*R)/(math.pi*R*R)
        check('fill area == pi*R^2 (R = %.1f cells)' % Rc,
              rel < 5e-4, 'rel %.1e' % rel)
    check('.fill() percent method survives the fill_frac attribute',
          0.0 < m.fill() < 100.0, '%.1f%%' % m.fill())
    check('fill_frac values in (0, 1]',
          float(m.fill_frac.max()) <= 1.0
          and float(m.fill_frac[m.fill_frac > 0].min()) >= 1e-3)

    # -- schema contract ---------------------------------------------
    expect_error('cylinder missing radius rejected',
                 wire_doc(3e-6).replace('radius = 3e-06\n', ''),
                 "missing 'radius'")
    expect_error('bad axis rejected',
                 wire_doc(3e-6).replace('axis = "x"', 'axis = "w"'),
                 "axis must be")
    expect_error('bad center length rejected',
                 wire_doc(3e-6).replace('center = [4e-06, 4e-06]',
                                        'center = [4e-06]'),
                 'TWO transverse')
    expect_error('overlap with block rejected',
                 wire_doc(3e-6) + '\n[[block]]\nfrom = [0, 0, 0]\n'
                 'to = [96, 8, 8]\nsigma = 1e7', 'overlaps')

    # -- physics: DC R vs analytic, staircase vs fill -----------------
    R, DX, NX = 3e-6, 1e-6, 96
    prob = sppeec_input.loads(wire_doc(R))
    m = prob.model()
    sw = prob.sweeper(m, prob.tree(m))
    Z, info = sw.solve(1e5)
    r_true = (NX*DX)/(5.8e7*math.pi*R*R)
    n_st = 32                      # center-in cells on this grid
    r_st = (NX*DX)/(5.8e7*n_st*DX*DX)
    e_fill = abs(Z.real - r_true)/r_true
    e_st = abs(r_st - r_true)/r_true
    check('fill-corrected DC R within 1.5%% of L/(sigma*pi*R^2)',
          e_fill < 0.015, 'R %.6g vs %.6g (%.2f%%)'
          % (Z.real, r_true, 100*e_fill))
    check('fill beats the staircase by >= 5x',
          e_fill < e_st/5.0,
          'fill %.2f%% vs staircase %.1f%%'
          % (100*e_fill, 100*e_st))
    check('solve clean (LpPR true residual)',
          info['true_residual'] < 1e-8,
          '%.1e' % info['true_residual'])

    # -- stage B: sparse partial-cell inductance ----------------------
    import subpixel
    from terminal import box_mutual_matrix
    M = prob.tree(m)          # the fill wire from the R gate above
    dL = subpixel.build_dL(m, M)
    check('dL is exactly symmetric', abs(dL - dL.T).max() == 0.0)
    mf = sppeec_input.loads(
        '\n'.join(['[grid]', 'dims = [8, 4, 4]', 'pitch = 1e-6',
                   '[[block]]', 'from = [0, 0, 0]', 'to = [8, 4, 4]',
                   'sigma = 5.8e7', '[port]',
                   'p_faces = [[0, 1, 1, "-x"]]',
                   'n_faces = [[7, 1, 1, "+x"]]',
                   '[solve]', 'freq = [1e5]'])).model()
    check('full-cell model -> no dL',
          subpixel.build_dL(mf, None) is None)
    # oracle: one near pair from first principles
    spx = m.subpixel
    k, axis = spx['k'], spx['axis']
    t1, t2 = [c for c in range(3) if c != axis]
    from equiterminal import filament_cells
    fa, fc = filament_cells(M)
    slx = np.nonzero(fa == axis)[0]
    cl = fc[slx]
    tgt = next(n for n, c in enumerate(cl)
               if (int(c[t1]), int(c[t2])) in spx['cells']
               and 0.05 < spx['cells'][(int(c[t1]),
                                        int(c[t2]))].mean() < 0.95)
    c = cl[tgt]
    w = spx['cells'][(int(c[t1]), int(c[t2]))].ravel().astype(float)
    w = w/w.sum()
    u = np.full(k*k, 1.0/(k*k))
    dv = np.asarray(m.d, float)
    g1, g2 = np.meshgrid(range(k), range(k), indexing='ij')
    lo = np.zeros((2*k*k, 3))
    lo[:k*k, t1] = g1.ravel()*dv[t1]/k
    lo[:k*k, t2] = g2.ravel()*dv[t2]/k
    lo[k*k:] = lo[:k*k]
    lo[k*k:, axis] += dv[axis]
    ext = np.zeros(3)
    ext[axis], ext[t1], ext[t2] = dv[axis], dv[t1]/k, dv[t2]/k
    B = box_mutual_matrix(lo, lo + ext, axis)[:k*k, k*k:]
    oracle = float(w @ B @ w) - float(u @ B @ u)
    cc = c.copy()
    cc[axis] += 1
    j = next(int(slx[n2]) for n2, c2 in enumerate(cl)
             if tuple(c2) == tuple(cc))
    check('dL near-pair matches first principles',
          abs(dL[int(slx[tgt]), j] - oracle) <= 1e-18,
          '%.4g vs %.4g' % (dL[int(slx[tgt]), j], oracle))

    # L referee: staircase -> A -> A+B strictly approach the 2x ref
    def wire2(nx, dt, dx, stair=False):
        c0 = dt*dx/2.0
        fp, fn, blocks = [], [], []
        for i in range(dt):
            row = [j2 for j2 in range(dt)
                   if ((i+0.5)*dx - c0)**2
                   + ((j2+0.5)*dx - c0)**2 < R*R]
            if row and stair:
                blocks += ['[[block]]',
                           'from = [0, %d, %d]' % (i, row[0]),
                           'to = [%d, %d, %d]' % (nx, i+1, row[-1]+1),
                           'sigma = 5.8e7']
            for j2 in row:
                fp.append('[0, %d, %d, "-x"]' % (i, j2))
                fn.append('[%d, %d, %d, "+x"]' % (nx-1, i, j2))
        geo = blocks if stair else [
            '[[cylinder]]', 'axis = "x"',
            'center = [%g, %g]' % (c0, c0), 'radius = %g' % R,
            'sigma = 5.8e7']
        return '\n'.join(
            ['[grid]', 'dims = [%d, %d, %d]' % (nx, dt, dt),
             'pitch = %g' % dx] + geo +
            ['[port]', 'p_faces = [%s]' % ', '.join(fp),
             'n_faces = [%s]' % ', '.join(fn),
             '[solve]', 'freq = [1e5]'])

    def runL(doc, strip_b=False):
        pr = sppeec_input.loads(doc)
        mm = pr.model()
        if strip_b:
            mm.subpixel = None
        sw2 = pr.sweeper(mm, pr.tree(mm))
        Z2, _ = sw2.solve(1e5)
        return Z2.real, Z2.imag/(2*math.pi*1e5)

    r_st, l_st = runL(wire2(96, 8, 1e-6, stair=True))
    r_a, l_a = runL(wire2(96, 8, 1e-6), strip_b=True)
    r_ab, l_ab = runL(wire2(96, 8, 1e-6))
    _, l_ref = runL(wire2(192, 16, 0.5e-6))
    e = [abs(x - l_ref)/l_ref for x in (l_st, l_a, l_ab)]
    check('L errors strictly improve: staircase > A > A+B',
          e[0] > e[1] > e[2],
          'st %.2f%%, A %.2f%%, A+B %.2f%% vs 2x ref'
          % (100*e[0], 100*e[1], 100*e[2]))
    check('A+B within 1.2%% of the 2x reference', e[2] < 0.012,
          '%.2f%%' % (100*e[2]))
    check('dL leaves R untouched', abs(r_ab - r_a)/r_a < 1e-9,
          '%.3g vs %.3g' % (r_ab, r_a))

    # -- skin effect on the subpixel wire (the Kelvin razor) ----------
    # The A+B stack at 8 cells across the section delivers the exact
    # round-wire R_AC/R_DC to a few percent through dx/delta = 2 --
    # measured 2026-08-18, and the imposed-profile alternative
    # (subpixel.build_dZ) measured WORSE (see its docstring), which
    # is the recorded justification for solved-amplitude C.2 modes.
    from scipy.special import jv
    MU0 = 4e-7*math.pi
    SIG = 5.8e7

    def z_int(freq):
        delta = math.sqrt(2.0/(2*math.pi*freq*MU0*SIG))
        kb = (1.0 - 1.0j)/delta
        return (kb/(SIG*2*math.pi*R))*jv(0, kb*R)/jv(1, kb*R)

    pw = sppeec_input.loads(wire2(96, 8, 1e-6))
    mw = pw.model()
    sww = pw.sweeper(mw, pw.tree(mw))
    zlo2, _ = sww.solve(1e7)
    for f, band in ((4.37e9, 0.03), (1.747e10, 0.04)):
        zf, _ = sww.solve(f)
        rs = zf.real/zlo2.real
        ra = z_int(f).real/z_int(1e7).real
        check('skin R-ratio vs Kelvin at dx/delta %.0f (band %d%%)'
              % (round(1e-6/math.sqrt(2.0/(2*math.pi*f*MU0*SIG))),
                 100*band),
              abs(rs - ra)/ra < band,
              '%.4f vs %.4f (rel %.1e)' % (rs, ra, abs(rs - ra)/ra))

    print('\n%d checks failed' % len(FAIL))
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
