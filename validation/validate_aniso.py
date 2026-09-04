# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate ANISOTROPIC (non-cubic) voxel support.

The FMM core was per-axis all along; what changed 2026-08-07 is the
plumbing: ``VoxelModel.d`` (per-axis pitch, ``dx`` raises on
anisotropic models), per-axis pitch pinning in ``build_tree``,
aspect-compensating leaves in ``partition()`` (multipole truncation
grows over stretched BOXES, so leaf counts compensate cell aspect),
and per-axis ``Terminals``. The skin engine stays cubic-only behind a
loud guard.

PART A -- THE EXACT ORACLE. traverseRL on a 2:1-cell bar against dense
``tm.box_mutual_matrix`` on the very same filament boxes, per
orientation. This is ground truth, not a reference solver: the box
integrals are closed-form. Measured levels: cubic 3-4e-4, 2:1 with
compensating leaves ~1e-3; the gate is 5e-3 (comfortably above
truncation, far below the 25x degradation the uncompensated leaf
showed).

PART B -- DC CLOSED FORM through the FULL PORT PATH (equiterminal:
per-axis terminals, kernels, incidence): R_dc = l_eff/(sigma*A) with
the anisotropic cross-section, exact to 1e-9.

PART C -- SAME PHYSICAL BAR, cubic vs anisotropic mesh: identical
R_dc; inductance at mild frequency within discretisation distance.

PART D -- GUARDS AND REGRESSION: ``dx`` raises on anisotropic models;
skin subdivision ENGAGES per axis (since 296bcdb); cubic corpus
partition choices are UNCHANGED.

Run inside the toolbox:  python3 validate_aniso.py
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
import voxmodel
import stencils as st
import terminal as tm

fails = []


def check(tag, cond, detail=''):
    print("    %-4s %s  %s" % ('ok' if cond else 'FAIL', tag, detail))
    if not cond:
        fails.append(tag)
    return cond


def bar_model(dims, d, sigma=5.8e7, freq=(1e8,)):
    m = voxmodel.VoxelModel('aniso_bar')
    m.dims = tuple(int(v) for v in dims)
    m.d = np.asarray(d, dtype=float)
    m.sigma = np.full(m.dims, float(sigma))
    m.freq = np.asarray(freq, dtype=float)
    p = voxmodel.Port('p1')
    for j in range(dims[1]):
        for k in range(dims[2]):
            p._add('P', (0, j, k, 0, -1))
            p._add('N', (dims[0] - 1, j, k, 0, 1))
    p._freeze()
    m.ports = [p]
    return m


def part_a():
    print("\nPART A -- traverseRL vs exact dense oracle (2:1 cells)")
    m = bar_model((16, 4, 4), (1e-6, 1e-6, 2e-6))
    leaf, levels = m.partition()
    print("    (partition chose nleaf %s, %d levels)"
          % (list(int(v) for v in leaf), levels))
    M = m.build_tree(leaf, levels)
    whole, (esz, efsz, efgsz, _) = voxmodel.allocate(M)
    M.jomega = 1j
    for lf in (M.e, M.f, M.g):
        lf.r = 0.0
    rng = np.random.default_rng(0)
    l_t = m.d
    for lf, axis, lo_, hi_ in ((M.e, 1, 0, esz), (M.f, 0, esz, efsz),
                               (M.g, 2, efsz, efgsz)):
        cells = voxmodel.filament_cells(M, lf)
        lo = cells*l_t[None, :]
        hi = (cells + 1)*l_t[None, :]
        lo[:, axis] = (cells[:, axis] + 0.5)*l_t[axis]
        hi[:, axis] = (cells[:, axis] + 1.5)*l_t[axis]
        Z = tm.box_mutual_matrix(lo, hi, axis)
        x = rng.standard_normal(len(cells)) \
            + 1j*rng.standard_normal(len(cells))
        whole[:efgsz] = 0
        whole[lo_:hi_] = x
        M.traverseRL()
        rel = np.linalg.norm(whole[lo_:hi_] - 1j*(Z @ x)) \
            / np.linalg.norm(1j*(Z @ x))
        check("axis %d vs oracle" % axis, rel < 5e-3, "rel %.2e" % rel)


def part_b():
    print("\nPART B -- DC closed form through per-axis terminals")
    import equiterminal as eq
    m = bar_model((16, 4, 4), (1e-6, 1e-6, 2e-6), freq=(1.0,))
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, 1.0)
    S = eq.EquiTerminalSolver(m, M, 0)
    Z, _, _ = S.solve(0.0)
    A = m.dims[1]*m.d[1]*m.dims[2]*m.d[2]
    want = ((m.dims[0] - 1)*m.d[0] + m.d[0])/(5.8e7*A)
    check("R_dc closed form", abs(Z.real/want - 1.0) < 1e-9,
          "%.10g vs %.10g" % (Z.real, want))


def part_c():
    print("\nPART C -- same physical bar, cubic vs anisotropic mesh")
    import equiterminal as eq
    out = {}
    for tag, dims, d in (('aniso', (16, 4, 4), (1e-6, 1e-6, 2e-6)),
                         ('cubic', (16, 4, 8), (1e-6, 1e-6, 1e-6))):
        m = bar_model(dims, d, freq=(1e8,))
        leaf, levels = m.partition()
        M = m.build_tree(leaf, levels)
        m.prepare(M, 1e8)
        S = eq.EquiTerminalSolver(m, M, 0)
        Zdc, _, _ = S.solve(0.0)
        Zac, _, _ = S.solve(1e8)
        out[tag] = (Zdc.real, Zac.imag/(2*np.pi*1e8))
    drc = abs(out['aniso'][0]/out['cubic'][0] - 1.0)
    dl = abs(out['aniso'][1]/out['cubic'][1] - 1.0)
    check("R_dc identical", drc < 1e-9, "rel %.2e" % drc)
    check("L within discretisation", dl < 2e-2,
          "L %.6g vs %.6g H (rel %.2e)"
          % (out['aniso'][1], out['cubic'][1], dl))


def part_d():
    print("\nPART D -- guards and cubic regression")
    m = bar_model((16, 4, 4), (1e-6, 1e-6, 2e-6))
    try:
        m.dx
        check("dx raises on anisotropic", False, "returned a value")
    except ValueError as e:
        check("dx raises on anisotropic", 'anisotropic' in str(e), '')
    import equiterminal as eq
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, 1e8)
    # The skin engine ACCEPTS anisotropic cells since 296bcdb
    # (2026-09-01): boxes, sub-bar areas and shapes are per-axis. This
    # check used to expect NotImplementedError and went stale with that
    # commit (caught by the enrichment-plan phase-0 baseline).
    S = eq.EquiTerminalSolver(m, M, 0, enrich=dict(k=3))
    r = S.redist
    check("skin engine engages on anisotropic cells",
          r is not None and r.nmode > 0
          and tuple(r.dt) == (1e-6, 2e-6),
          "nmode %d dt %s" % (0 if r is None else r.nmode,
                              None if r is None else tuple(r.dt)))
    mc = vhr.read_vhr('VoxHenry/Input_files/'
                      'straight_cond1_len30.0u_wid10.0u_dist20.0u'
                      '-two_freq.vhr')
    nl, lv = mc.partition()
    check("cubic partition unchanged",
          list(int(v) for v in nl) == [5, 5, 5] and lv == 2,
          "%s lv %d" % (list(int(v) for v in nl), lv))


def main():
    part_a()
    part_b()
    part_c()
    part_d()
    print("\n%d checks failed" % len(fails))
    if fails:
        print("  " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
