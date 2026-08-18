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

    print('\n%d checks failed' % len(FAIL))
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
