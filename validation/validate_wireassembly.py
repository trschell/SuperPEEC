# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for wireassembly.py -- the coupled wire + voxel-eddy solve.

THE CHECK THAT MATTERS IS THE DENSE ORACLE (part B): the identical
physics -- exact bar Lp for the lattice (terminal.box_mutual_matrix,
the repo's oracle kernel), exact batched wire<->voxel and wire<->wire
blocks, the same mesh basis, the same drive -- assembled dense and
solved directly. Any disagreement beyond the FMM's far-field truncation
class is an assembly bug, not physics.

Geometry: a 16x16x4 copper slab with three wires flying 1.6 cells
above it, A and B in PARALLEL against return C (the module question's
shape), wire B deliberately TILTED a few degrees so the skew kernel
path is exercised inside the assembly, not just in unit tests.

Run: PYTHONPATH=src python3 validate_wireassembly.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np

import terminal as tm
import voxmodel
import wirekernel as wk
from wireassembly import Wire, WireEddySolver

FAIL = []
FREQ = 1e9
D = 1e-6
SIG_AL = 3.77e7


def check(name, ok, detail=""):
    print("  %-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def build_model(sigma=5.8e7):
    m = voxmodel.VoxelModel('wireassembly_gate')
    m.dims = (16, 16, 4)
    m.d = D
    m.sigma = np.full(m.dims, sigma)
    m.freq = np.array([FREQ])
    return m, m.build_tree(nleaf=[4, 4, 5], numlevels=2)


def build_wires(delta):
    a = 0.4e-6
    A = Wire([[4*D, 6.5*D, 5.6*D], [12*D, 6.5*D, 5.6*D]], a, SIG_AL,
             delta=delta, max_seglen=2*D)
    B = Wire([[4*D, 9.2*D, 5.6*D], [12*D, 9.8*D, 5.6*D]], a, SIG_AL,
             delta=delta, max_seglen=2.1*D)     # tilted: skew path
    C = Wire([[4*D, 12.5*D, 5.6*D], [12*D, 12.5*D, 5.6*D]], a, SIG_AL,
             delta=delta, max_seglen=2*D)
    return [A, B, C]


def dense_reference(sol, freq):
    """The whole coupled system, dense and exact, on the same basis."""
    m, M, wc = sol.model, sol.M, sol.wc
    m.prepare(M, freq)
    jw = M.jomega
    efg, nwel = sol.efg, sol.nwel
    l = wc.l
    # voxel Lp: exact bars, block-diagonal over orientations
    Z = np.zeros((efg + nwel, efg + nwel), dtype=np.complex128)
    r_f = np.concatenate([np.asarray(leaf.r, dtype=float).ravel()
                          * np.ones(np.size(leaf.idx))
                          for leaf, _, _ in wc.leaves])
    for leaf, axis, off in wc.leaves:
        size = np.size(leaf.idx)
        cells = wc.fil_cell[off:off + size]
        lo = cells*l[None, :]
        lo[:, axis] += 0.5*l[axis]
        hi = lo + l[None, :]
        Lp = tm.box_mutual_matrix(lo, hi, axis)
        Z[off:off + size, off:off + size] += jw*Lp
    Z[:efg, :efg] += np.diag(r_f)
    # wire <-> voxel, exact batched kernels, ALL pairs
    Cd = np.zeros((nwel, efg))
    for s, fs in enumerate(wc.segments):
        for leaf, axis, off in wc.leaves:
            if abs(float(fs[0].u[axis])) < 1e-14:
                continue
            size = np.size(leaf.idx)
            Cd[wc.seg0[s]:wc.seg0[s + 1], off:off + size] = \
                wk.mutual_voxels(fs, wc.fil_cell[off:off + size], l, axis)
    Z[efg:, :efg] += jw*Cd
    Z[:efg, efg:] += jw*Cd.T
    # wire <-> wire, exact blocks
    Lw = np.zeros((nwel, nwel))
    for s1 in range(len(wc.segments)):
        a1, b1 = wc.seg0[s1], wc.seg0[s1 + 1]
        Lw[a1:b1, a1:b1] = wk.segment_self_block(wc.segments[s1])
        for s2 in range(s1 + 1, len(wc.segments)):
            a2, b2 = wc.seg0[s2], wc.seg0[s2 + 1]
            B = wk.mutual_block(wc.segments[s1], wc.segments[s2])
            Lw[a1:b1, a2:b2] = B
            Lw[a2:b2, a1:b1] = B.T
    Z[efg:, efg:] += jw*Lw + np.diag(sol.r_w)
    # same basis, same drive, direct solve
    import scipy.sparse as sp
    Bfull = sp.bmat([[sol.Y, None, None],
                     [None, sol.S, sol.T]], format='csc')
    ihat = np.concatenate([np.zeros(efg), sol.ihat])
    A = (Bfull.T @ (Z @ Bfull.toarray()))
    rhs = -(Bfull.T @ (Z @ ihat))
    x = np.linalg.solve(A, rhs)
    i = ihat + Bfull @ x
    v = Z @ i
    V = complex(sol.ihat @ v[efg:])
    share = np.zeros(len(sol.wires), dtype=np.complex128)
    for j in range(len(sol.wires)):
        segs = np.where(sol.wire_of_seg == j)[0]
        tots = [i[efg + wc.seg0[s]:efg + wc.seg0[s + 1]].sum()
                for s in segs]
        share[j] = np.mean(tots)
    return V, share


def analytic_dc(wires, loop, groups):
    """Loop R at DC from rho*l/A per wire, exact for the uniform
    cross-section (the elements partition the disc exactly)."""
    R = []
    for w in wires:
        ltot = sum(seg[0].length for seg in w.segments)
        R.append(w.rho*ltot/(np.pi*w.radius**2))
    Rg = []
    for g in groups:
        Rg.append(1.0/sum(1.0/R[j] for j in g))
    return sum(Rg), R


if __name__ == '__main__':
    delta = 0.0851/np.sqrt(FREQ)      # Al skin depth
    m, M = build_model()
    m.prepare(M, FREQ)
    wires = build_wires(delta)
    loop = [+1, +1, -1]
    groups = [[0, 1], [2]]
    sol = WireEddySolver(m, M, wires, loop, groups=groups, verbose=True)

    print("\nPART A -- the coupled solve runs and converges")
    Z, info = sol.solve(FREQ, verbose=True)
    check("lgmres converged", info['flag'] == 0 and
          info['residual'] < 1e-8,
          "flag %s resid %.2e" % (info['flag'], info['residual']))
    check("chain totals consistent along each wire",
          info['share_spread'] < 1e-8, "%.2e" % info['share_spread'])
    sh = np.real(info['share'])
    check("KCL: group shares sum to the loop pattern",
          abs(sh[0] + sh[1] - 1) < 1e-8 and abs(sh[2] + 1) < 1e-8,
          "A+B %.6f C %.6f" % (sh[0] + sh[1], sh[2]))
    check("passivity: Re Z > 0", Z.real > 0, "Z = %s" % Z)

    print("\nPART B -- dense oracle of the ENTIRE coupled system")
    Vd, shd = dense_reference(sol, FREQ)
    relZ = abs(Z - Vd)/abs(Vd)
    relsh = np.abs(info['share'] - shd).max()
    print("    solver Z = %.6e%+.6ej" % (Z.real, Z.imag))
    print("    oracle Z = %.6e%+.6ej" % (Vd.real, Vd.imag))
    check("Z_loop matches dense oracle to 1%", relZ < 1e-2,
          "rel %.2e" % relZ)
    check("shares match dense oracle to 1e-3", relsh < 1e-3,
          "max %.2e" % relsh)

    print("\nPART C -- image screening physics")
    m1, M1 = build_model(sigma=1.0)   # the slab, electrically absent
    m1.prepare(M1, FREQ)
    sol1 = WireEddySolver(m1, M1, wires, loop, groups=groups)
    Z1, info1 = sol1.solve(FREQ)
    w = 2*np.pi*FREQ
    L, L1 = Z.imag/w, Z1.imag/w
    # jomega sign convention: take magnitudes for L
    L, L1 = abs(L), abs(L1)
    print("    with slab: R %.4e  L %.4e ; slab absent: R %.4e  L %.4e"
          % (Z.real, L, Z1.real, L1))
    check("slab REDUCES loop L (image screening)", L < L1,
          "%.4g -> %.4g nH" % (1e9*L1, 1e9*L))
    check("slab ADDS eddy loss to R", Z.real > Z1.real,
          "%.4g -> %.4g mOhm" % (1e3*Z1.real, 1e3*Z.real))

    print("\nPART D -- wires-only limit against the dense oracle")
    Vd1, shd1 = dense_reference(sol1, FREQ)
    rel = abs(Z1 - Vd1)/abs(Vd1)
    check("sigma=1 slab: solver == dense oracle to 1%", rel < 1e-2,
          "rel %.2e" % rel)

    print("\nPART E -- DC limit against closed-form resistance")
    wires_dc = build_wires(None)      # uniform shapes at DC
    m2, M2 = build_model()
    m2.prepare(M2, 10.0)
    sol2 = WireEddySolver(m2, M2, wires_dc, loop, groups=groups)
    Z2, info2 = sol2.solve(10.0)
    Rdc, Rw = analytic_dc(wires_dc, loop, groups)
    rel = abs(Z2.real - Rdc)/Rdc
    check("Re Z(10 Hz) == analytic loop R_dc", rel < 1e-6,
          "%.8e vs %.8e (rel %.2e)" % (Z2.real, Rdc, rel))
    shdc = np.real(info2['share'])
    gA = (1/Rw[0])/(1/Rw[0] + 1/Rw[1])
    check("DC sharing follows conductances", abs(shdc[0] - gA) < 1e-6,
          "%.6f vs %.6f" % (shdc[0], gA))

    print("\n%d checks failed" % len(FAIL))
    raise SystemExit(1 if FAIL else 0)
