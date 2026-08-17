# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate homogeneous background dielectric support (phase 1).

A uniform relative permittivity ``eps_bg`` divides every coefficient
of potential while the kernels stay free-space, so the implementation
divides each eps-carrying object ONCE at tree build (panel ``ynmr``,
the assembled ``n2n``, circulant spectra, fftnear tables) before any
factorisation. The inductive path is untouched -- dielectrics do not
exist quasi-magnetostatically.

PART A -- EXACT OPERATOR SCALING. ``n2n(eps)`` must equal
``n2n(1)/eps`` bit-for-bit at eps = 4 (division by a power of two is
exact in IEEE) and to 1e-15 at eps = 3.7.

PART B -- PHYSICAL CAPACITANCE. Parallel plates solved through the
coefficient-of-potential system: C(eps)/C(1) == eps to solver
precision (fringing cancels exactly in the ratio), and the absolute
C(1) is sanity-checked against eps0*A/d within the generous fringing
allowance a 3-cell gap deserves.

PART C -- CIRCULANT PATH. The single-level circulant build scales its
near matrix and FFT spectra identically.

Run inside the toolbox:  python3 validate_dielectric.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]


import sys

import numpy as np

import voxmodel
import stencils as st

fails = []


def check(tag, cond, detail=''):
    print("    %-4s %s  %s" % ('ok' if cond else 'FAIL', tag, detail))
    if not cond:
        fails.append(tag)
    return cond


def plates(eps_bg, circulant=False):
    m = voxmodel.VoxelModel('plates')
    m.dims = (8, 8, 4)
    m.d = 1e-6
    m.sigma = np.zeros(m.dims)
    m.sigma[:, :, 0] = 5.8e7
    m.sigma[:, :, 3] = 5.8e7
    m.freq = np.array([1e9])
    m.eps_bg = eps_bg
    kw = dict(capacitive=True)
    if circulant:
        kw['circulant'] = True
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels, **kw)
    return m, M


def cap(m, M):
    """Plate capacitance from the node coefficient-of-potential."""
    P = M.n2n.toarray() if hasattr(M.n2n, 'toarray') else np.asarray(M.n2n)
    cells = voxmodel.filament_cells(M, M.lv[0])
    v = np.where(cells[:, 2] == 0, +0.5, -0.5)
    q = np.linalg.solve(P, v)
    return float(q[cells[:, 2] == 0].sum())


def part_a():
    print("\nPART A -- exact operator scaling")
    _, M1 = plates(1.0)
    n1 = np.asarray(M1.n2n.toarray() if hasattr(M1.n2n, 'toarray')
                    else M1.n2n)
    for eps, tol in ((4.0, 0.0), (3.7, 1e-15)):
        _, Me = plates(eps)
        ne = np.asarray(Me.n2n.toarray() if hasattr(Me.n2n, 'toarray')
                        else Me.n2n)
        err = np.abs(ne*eps - n1).max()/np.abs(n1).max()
        check("n2n scaling eps=%g" % eps, err <= tol,
              "rel %.2e (tol %g)" % (err, tol))


def part_b():
    print("\nPART B -- physical capacitance")
    m1, M1 = plates(1.0)
    c1 = cap(m1, M1)
    m4, M4 = plates(4.0)
    c4 = cap(m4, M4)
    check("C(4)/C(1) == 4", abs(c4/c1 - 4.0) < 1e-12,
          "ratio %.14g" % (c4/c1))
    eps0 = 1/(voxmodel.MU0*299792458.0**2)
    # FACE-to-face gap: plates occupy z-cells {0} and {3}, so the
    # facing surfaces sit 2 cells apart, not the 3-cell centre
    # spacing.
    ideal = eps0*(8e-6*8e-6)/(2e-6)
    check("C(1) vs ideal within fringing", 1.0 < c1/ideal < 2.5,
          "C %.4g F vs plate formula %.4g F (x%.2f, fringing on "
          "W/d = 4 plates)" % (c1, ideal, c1/ideal))


def part_c():
    print("\nPART C -- circulant path scaling")
    _, M1 = plates(1.0, circulant=True)
    _, M4 = plates(4.0, circulant=True)
    n1 = M1.n2n.toarray()
    n4 = M4.n2n.toarray()
    err = np.abs(n4*4.0 - n1).max()/np.abs(n1).max()
    check("circulant n2n scaling", err == 0.0, "rel %.2e" % err)
    k1 = next(iter(M1.circpoten._blocks.values()))['FK']
    k4 = next(iter(M4.circpoten._blocks.values()))['FK']
    err = np.abs(k4*4.0 - k1).max()/np.abs(k1).max()
    check("circulant spectrum scaling", err == 0.0, "rel %.2e" % err)


def _plate_model(dims, eps_layers):
    """Plates at z = 0 and z = dims[2]-1; eps_layers maps z -> eps_r."""
    m = voxmodel.VoxelModel('d2')
    m.dims = dims
    m.d = 1e-6
    m.sigma = np.zeros(dims)
    m.sigma[:, :, 0] = 5.8e7
    m.sigma[:, :, dims[2] - 1] = 5.8e7
    if eps_layers:
        m.epsilon = np.ones(dims)
        for z, er in eps_layers.items():
            m.epsilon[:, :, z] = er
    m.freq = np.array([1e8])
    return m


def _capacitance(m, freq):
    """C from a DENSE oracle-grade LpPR solve: exact box-integral Lp,
    the tree's n2n for P, and the model's (complex) branch impedances
    -- the same MNA shape systemmat solves, assembled from ground-truth
    pieces so the only thing under test is the dielectric physics."""
    import terminal as tm
    import equiterminal as eq
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels, capacitive=True)
    m.prepare(M, freq)
    B, ncell = eq.sparse_incidence(M, M._vhr_whole,
                                   int(np.size(M.e.struc))
                                   + int(np.size(M.f.struc))
                                   + int(np.size(M.g.struc)),
                                   int(np.size(M.lv[0].struc)))
    fa, fc = eq.filament_cells(M)
    nfil, nn = B.shape
    l_t = m.d
    lo = fc*l_t[None, :]
    hi = (fc + 1)*l_t[None, :]
    Lp = np.zeros((nfil, nfil))
    for ax in range(3):
        s = np.flatnonzero(fa == ax)
        if s.size:
            alo, ahi = lo[s].copy(), hi[s].copy()
            alo[:, ax] = (fc[s, ax] + 0.5)*l_t[ax]
            ahi[:, ax] = (fc[s, ax] + 1.5)*l_t[ax]
            Lp[np.ix_(s, s)] = tm.box_mutual_matrix(alo, ahi, ax)
    # r in e|f|g whole order, matching sparse_incidence rows
    r = np.concatenate([np.broadcast_to(np.atleast_1d(M.e.r),
                                        (int(np.size(M.e.struc)),)),
                        np.broadcast_to(np.atleast_1d(M.f.r),
                                        (int(np.size(M.f.struc)),)),
                        np.broadcast_to(np.atleast_1d(M.g.r),
                                        (int(np.size(M.g.struc)),))])
    w = 2*np.pi*freq
    Z = np.diag(r.astype(np.complex128)) + 1j*w*Lp
    P = M.n2n.toarray() if hasattr(M.n2n, 'toarray') else np.asarray(M.n2n)
    Cnode = np.zeros((nn, nn), dtype=np.complex128)
    ext = M.external
    Cnode[np.ix_(ext, ext)] = np.linalg.inv(P[np.ix_(ext, ext)])
    # equiterminal's stated convention: Z i = B phi, B^T i = s, and
    # with node charge q = C phi the KCL gains +jw C phi:
    #     [[Z, -B], [B^T, +jw C]] (i, phi) = (0, s)
    A = B.toarray().astype(np.complex128)
    K = np.block([[Z, -A], [A.T, 1j*w*Cnode]])
    # drive: +-I at one node of each plate, read the voltage
    cells = voxmodel.filament_cells(M, M.lv[0])
    ntop = np.flatnonzero(cells[:, 2] == 0)
    nbot = np.flatnonzero(cells[:, 2] == m.dims[2] - 1)
    rhs = np.zeros(nfil + nn, dtype=np.complex128)
    rhs[nfil + ntop[0]] = 1.0
    rhs[nfil + nbot[0]] = -1.0
    # ROW-EQUILIBRATE before the least-squares solve: dielectric branch
    # impedances (~1e8) and node capacitance rows (~1e-7) span fifteen
    # orders, and lstsq's rank cutoff would discard the small-scale
    # PHYSICS as noise (measured: residual 8.8e-2 on a unit rhs, C off
    # by 1e5). Row scaling preserves the solution set exactly; the only
    # near-null direction left is the per-component potential gauge,
    # which the min-norm solution fixes and the voltage DIFFERENCE
    # never sees. The production solvers do the same job with
    # systemmat's W-rescaling.
    rs = 1.0/np.maximum(np.abs(K).max(axis=1), 1e-300)
    x = np.linalg.lstsq(K*rs[:, None], rhs*rs, rcond=None)[0]
    v = x[nfil + ntop[0]] - x[nfil + nbot[0]]
    y = 1.0/v
    return float(np.imag(y)/w)


def part_d():
    print("\nPART D -- per-cell dielectrics: physical capacitance")
    dims = (16, 16, 4)
    c_vac = _capacitance(_plate_model(dims, {}), 1e8)
    c_fill = _capacitance(_plate_model(dims, {1: 4.0, 2: 4.0}), 1e8)
    c_half = _capacitance(_plate_model(dims, {1: 4.0}), 1e8)
    rf = c_fill/c_vac
    rh = c_half/c_vac
    want = 2*4.0/(1 + 4.0)      # series: 2 eps/(1 + eps)
    # RESOLVED 2026-08-08: the half-fill deficit (1.06 measured for
    # months where the truth is ~1.59) was the material-ID value 2
    # leaking into the SINGLE-LEVEL struc -- and from there into the
    # node2panel projection weights: n2n blocks gained exactly 2x per
    # dielectric index, crushing the free-surface bound-charge dipole.
    # struc_from_cells now normalises to {0,1} like struc_from_block
    # always did (the multilevel path never had the bug). Against the
    # BEM reference (studies/bem_dielectric.py, validated on the
    # coated-sphere closed form): half 1.500 vs BEM 1.589 at this
    # 16-cell size (series limit 1.600), fill 3.092 vs BEM 3.099.
    # The MESH-CONVERGED truths at this shape are half 1.44 and fill
    # 3.025 (4-point Richardson, see the oracle's docstring): both
    # methods carry few-% finite-h error at this resolution, SuperPEEC
    # within +-5% of the limit. ENFORCED since the fix; the windows
    # accommodate the finite-plate fringing physics and that error.
    for tag, cond, det in (
            ("filled gap ~ eps_r", 3.0 < rf <= 4.05,
             "C ratio %.3f (ideal 4, fringe stays vacuum; BEM 3.099)"
             % rf),
            ("half-filled series formula", abs(rh/want - 1.0) < 0.12,
             "C ratio %.3f vs series %.3f (BEM 1.589)" % (rh, want))):
        check(tag, cond, det)
    c_lo = _capacitance(_plate_model(dims, {1: 4.0, 2: 4.0}), 1e7)
    check("C frequency-flat (pure reactance)",
          abs(c_lo/c_fill - 1.0) < 1e-3,
          "C(1e7)/C(1e8) = %.6f" % (c_lo/c_fill))


def part_e():
    """MULTILEVEL capacitive operator with dielectrics: the panel FMM
    (near n2n + traverseP3 far field) against the dense single-level
    n2n on the same geometry. Every other check in this file runs
    single-level trees; production filled boards partition MULTILEVEL,
    and the two struc paths are built by different code (the phase-2
    root-cause bug lived in exactly that asymmetry). Verified
    2026-08-08: vac 8.6e-4 / half 1.06e-3 / fill 1.12e-3 -- the
    dielectric configs sit at the VACUUM baseline (the known FMM
    near/far-seam floor, partially nmax-reducible, worst at leaf-box
    boundaries), so the tolerance is generous over that floor and a
    dielectric-specific break (which would read ~1e-1) trips loudly."""
    print("\nPART E -- multilevel capacitive operator (dielectrics)")
    dims = (16, 16, 12)
    m1 = _plate_model(dims, {})
    m1.sigma[:, :, :3] = 5.8e7          # thicken plates: 3 cells
    m1.sigma[:, :, 9:] = 5.8e7
    mh = _plate_model(dims, {})
    mh.sigma = m1.sigma.copy()
    mh.epsilon = np.ones(dims)
    mh.epsilon[:, :, 3:6] = 4.0

    def key_of(M):
        c = voxmodel.filament_cells(M, M.lv[0])
        return c[:, 0]*1000000 + c[:, 1]*1000 + c[:, 2]

    for tag, m in (('vac', m1), ('half-fill', mh)):
        M1 = m.build_tree(st.single_level_nleaf(dims), 1,
                          capacitive=True)
        P = M1.n2n.toarray() if hasattr(M1.n2n, 'toarray') \
            else np.asarray(M1.n2n)
        k1 = key_of(M1)
        o1 = np.argsort(k1)
        Mm = m.build_tree([4, 4, 4], 2, capacitive=True)
        km = key_of(Mm)
        om = np.argsort(km)
        check("node sets match (%s)" % tag,
              bool(np.array_equal(k1[o1], km[om])),
              "%d nodes" % k1.size)
        rng = np.random.default_rng(42)
        qs = rng.standard_normal(k1.size)
        q1 = np.empty(k1.size)
        q1[o1] = qs
        ref = P.dot(q1)[o1]
        qm = np.empty(km.size, np.complex128)
        qm[om] = qs
        Mm.lv[0].data = qm.copy()
        Mm.traverseP3()
        got = np.asarray(Mm.lv[0].data)[om]
        err = float((np.abs(got - ref)/np.abs(ref).max()).max())
        check("multilevel apply matches dense (%s)" % tag, err < 5e-3,
              "max rel %.2e (FMM seam floor ~1e-3)" % err)


def part_f():
    """PRODUCTION LpPR path with dielectrics: port_impedance.LpPRSolver
    (SystemMat rescaleLpPR + diagschur + right-preconditioned fgmres)
    against the dense part-D oracle on the same geometry and drive.
    Plates are 2 cells thick: 1-cell plates with an empty gap have no
    g-filaments and the fortran kernels reject the empty orientation
    (LpPRSolver raises the documented NotImplementedError there)."""
    print("\nPART F -- production LpPR solve (dielectrics)")
    from port_impedance import LpPRSolver
    dims = (16, 16, 6)
    freq = 1e8
    w = 2*np.pi*freq

    def build(eps_layers):
        m = _plate_model(dims, {})
        m.sigma[:, :, :] = 0.0
        m.sigma[:, :, :2] = 5.8e7
        m.sigma[:, :, 4:] = 5.8e7
        if eps_layers:
            m.epsilon = np.ones(dims)
            for z, er in eps_layers.items():
                m.epsilon[:, :, z] = er
        else:
            m.epsilon = None
        p = voxmodel.Port('cap')
        p._add('P', (0, 0, 5, 2, 1))
        p._add('N', (0, 0, 0, 2, -1))
        p._freeze()
        m.ports = [p]
        return m

    for tag, eps_layers in (('vac', {}), ('half', {2: 4.0}),
                            ('fill', {2: 4.0, 3: 4.0})):
        m = build(eps_layers)
        leaf, levels = m.partition()
        M = m.build_tree(leaf, levels, capacitive=True)
        m.prepare(M, freq)
        S = LpPRSolver(m, M)
        z, x, info = S.solve(freq)
        c_prod = float(np.imag(1.0/z)/w)
        c_ref = _capacitance(build(eps_layers), freq)
        check("production C == oracle (%s)" % tag,
              abs(c_prod/c_ref - 1.0) < 1e-3,
              "prod %.5g ref %.5g (%d matvecs, resid %.1e)"
              % (c_prod, c_ref, info['matvecs'], info['residual']))


def part_g():
    """MULTILEVEL production LpPR with dielectrics: LpPRSolver on a
    multilevel tree (banded W rescale -- the near-field Cholesky is
    indefinite here, which is exactly the case the band exists for)
    against the single-level exact-W answer. Verified against a dense
    probe of the full multilevel operator (C to 6 digits); the
    remaining tolerance is the FMM operator seam. Also the permanent
    regression for the rhs convention: the old shortcut (injections
    straight into the rescaled rhs, exact-W-only) read 49% high here
    while converging its rescaled residual -- the true-residual field
    in info is the guard."""
    print("\nPART G -- multilevel production LpPR (dielectrics)")
    from port_impedance import LpPRSolver
    dims = (16, 16, 12)
    freq = 1e8
    w = 2*np.pi*freq

    def build():
        m = _plate_model(dims, {})
        m.sigma[:, :, :] = 0.0
        m.sigma[:, :, :3] = 5.8e7
        m.sigma[:, :, 9:] = 5.8e7
        m.epsilon = np.ones(dims)
        m.epsilon[:, :, 3:6] = 4.0
        p = voxmodel.Port('cap')
        p._add('P', (0, 0, 11, 2, 1))
        p._add('N', (0, 0, 0, 2, -1))
        p._freeze()
        m.ports = [p]
        return m

    res = {}
    for tag, leaf, lv in (('single', None, 1), ('multi', [4, 4, 4], 2)):
        m = build()
        if lv == 1:
            M = m.build_tree(st.single_level_nleaf(dims), 1,
                             capacitive=True)
        else:
            M = m.build_tree(leaf, lv, capacitive=True)
        m.prepare(M, freq)
        S = LpPRSolver(m, M)
        z, x, info = S.solve(freq)
        res[tag] = (float(np.imag(1.0/z)/w), info, S.wsolve)
    c1, i1, w1 = res['single']
    cm, im_, wm = res['multi']
    check("W routes as designed", w1 == 'exact' and wm == 'band',
          "single %s, multi %s" % (w1, wm))
    check("multi C == single C (seam tol)", abs(cm/c1 - 1.0) < 5e-3,
          "multi %.6g single %.6g ratio %.5f" % (cm, c1, cm/c1))
    check("true residual small (multi)",
          im_['true_residual'] < 1e-6,
          "true %.2e rescaled %.2e" % (im_['true_residual'],
                                       im_['residual']))


def part_h():
    """LEAN capacitive tree (keep_n2n=False + fftnear): the multilevel
    tree without the stored near-field n2n -- its operator comes from
    the Toeplitz p2p tables and its band W / C_cap from KERNEL-DIRECT
    window blocks (reluctance.kernel_ccap_block_getter). Shipped
    2026-08-08: the stored n2n was the capacitive memory wall
    (~27*leaf^3 entries/ext node; 320^2 board: tree 33.1 -> 0.2 GB,
    solve peak 33.1 -> 13.1 GB, total wall 1204 -> 661 s, C identical
    to 6 digits, slightly FEWER matvecs). Gate: lean vs stored-n2n
    solve on the multilevel half-fill model."""
    print("\nPART H -- lean capacitive tree (keep_n2n=False)")
    from port_impedance import LpPRSolver
    dims = (16, 16, 12)
    freq = 1e8
    w = 2*np.pi*freq

    def build():
        m = _plate_model(dims, {})
        m.sigma[:, :, :] = 0.0
        m.sigma[:, :, :3] = 5.8e7
        m.sigma[:, :, 9:] = 5.8e7
        m.epsilon = np.ones(dims)
        m.epsilon[:, :, 3:6] = 4.0
        p = voxmodel.Port('cap')
        p._add('P', (0, 0, 11, 2, 1))
        p._add('N', (0, 0, 0, 2, -1))
        p._freeze()
        m.ports = [p]
        return m

    res = {}
    for tag, kw in (('stored', dict()),
                    ('lean', dict(fftnear=True, keep_n2n=False))):
        m = build()
        M = m.build_tree([4, 4, 4], 2, capacitive=True, **kw)
        m.prepare(M, freq)
        S = LpPRSolver(m, M)
        z, x, info = S.solve(freq)
        res[tag] = (float(np.imag(1.0/z)/w), info, S.wsolve)
    cs, is_, ws = res['stored']
    cl, il, wl = res['lean']
    check("lean tree solves (band W)", wl == 'band', "wsolve %s" % wl)
    check("lean C == stored C", abs(cl/cs - 1.0) < 1e-3,
          "lean %.6g stored %.6g ratio %.6f" % (cl, cs, cl/cs))
    check("lean true residual small", il['true_residual'] < 1e-6,
          "true %.2e (%d mv vs stored %d)"
          % (il['true_residual'], il['matvecs'], is_['matvecs']))


def main():
    print("dielectric phases 1+2")
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
