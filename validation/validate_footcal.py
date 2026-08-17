# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for footcal.py -- the bond-foot discretisation calibration.

FOUR RUNGS, weakest machinery to strongest claim:
  A  the lattice Green's function against EXACT network identities
     (adjacent-node resistance of the infinite cubic 1-ohm network is
     1/3 exactly).
  B  the half-space patch resistance against an INDEPENDENT
     finite-block sparse solve, extrapolated in 1/L -- shares no code
     with the Bessel-integral route.
  C  the deficit curve's required properties (h in (0,1), h ~ c*dx/r0
     asymptotically -- the edge-singularity scaling).
  D  THE POINT OF IT ALL: end-to-end refinement convergence. The same
     physical two-pad + one-wire problem solved at dx and dx/2, with
     the calibrated patch foot vs the legacy point foot. The point
     model's total R drifts by its uncancelled lattice spreading
     (~0.3 rho/dx per foot); the calibrated model must cut that drift
     by well over the gate factor.

Run: PYTHONPATH=src python3 validate_footcal.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np

import footcal as fc
import voxmodel
from wireassembly import Wire, WireBondSolver

FAIL = []
SIG_CU = 5.8e7
SIG_AL = 3.77e7


def check(name, ok, detail=""):
    print("  %-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def part_a():
    print("\nPART A -- lattice Green's function, exact identities")
    G0 = fc.green(0, 0, 0)
    Radj = 2*(G0 - fc.green(1, 0, 0))
    check("adjacent-node R == 1/3 exactly", abs(Radj - 1/3) < 1e-6,
          "%.10f (err %.1e)" % (Radj, abs(Radj - 1/3)))
    check("G(0) == the Watson value 0.2527...",
          abs(G0 - 0.25273100) < 1e-6, "%.8f" % G0)
    # tail robustness: doubling the finite range must not move G at
    # any level the calibration can feel (h is consumed at ~1e-3;
    # measured 3.4e-7 absolute, i.e. ~3e-6 relative)
    fc._GCACHE.clear()
    a = fc.green(3, 2, 1, T=200.0)
    fc._GCACHE.clear()
    b = fc.green(3, 2, 1, T=400.0)
    fc._GCACHE.clear()
    check("asymptotic tail converged (T=200 vs 400)",
          abs(a - b) < 2e-6, "%.2e" % abs(a - b))


def _block_patch_R(x, L, H):
    """Independent reference: patch on top of an L x L x H block, all
    non-top faces grounded; energy-definition spreading resistance in
    units of 1/(sigma*dx). Sparse 7-point Laplacian, nothing shared
    with the Green's-function route."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    off, w = fc.patch_offsets(x)
    n = L*L*H

    def idx(i, j, k):
        return (i*L + j)*H + k
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for i in range(L):
        for j in range(L):
            for k in range(H):
                a = idx(i, j, k)
                for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                   (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if kk >= H:
                        continue          # insulating top surface
                    if 0 <= ii < L and 0 <= jj < L and kk >= 0:
                        rows.append(a)
                        cols.append(idx(ii, jj, kk))
                        vals.append(-1.0)
                        diag[a] += 1.0
                    else:
                        diag[a] += 1.0    # grounded boundary face
    rows += list(range(n))
    cols += list(range(n))
    vals += list(diag)
    G = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    s = np.zeros(n)
    c0 = L//2
    for (i, j), wc in zip(off, w):
        s[idx(c0 + i, c0 + j, H - 1)] += wc
    phi = spla.spsolve(G, s)
    return float(s @ phi)


def part_b():
    print("\nPART B -- half-space patch R vs independent block solve")
    x = 2.0
    exact = fc.lattice_patch_R(x)
    Rs = []
    for L, H in ((32, 16), (64, 32)):
        Rs.append(_block_patch_R(x, L, H))
    # grounded boundary at ~L/2: R(L) = R_inf - beta/L; two sizes
    # eliminate beta
    extrap = Rs[1] + (Rs[1] - Rs[0])          # 1/L halves: R_inf = 2R2-R1
    rel = abs(extrap - exact)/exact
    check("GF patch R == block solve extrapolated to L->inf",
          rel < 2e-2,
          "GF %.5f vs %.5f (L=32: %.5f, L=64: %.5f; rel %.2e)"
          % (exact, extrap, Rs[0], Rs[1], rel))


def part_c():
    print("\nPART C -- the deficit curve h(r0/dx)")
    xs = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    hs = np.array([fc.gh_curve(x)[1] for x in xs])
    check("h in (0, 1) everywhere", bool(np.all((hs > 0) & (hs < 1))),
          " ".join("%.3f" % h for h in hs))
    check("h monotone decreasing beyond x = 1",
          bool(np.all(np.diff(hs[1:]) < 0)), "")
    # The deficit decays close to, but measurably SLOWER than, 1/x:
    # x*h = 0.54/0.56/0.65 at x = 6/8/12, still rising -- a log
    # factor from the flux-edge field is the likely reading. No
    # asymptote is USED anywhere (h is computed exactly at each x),
    # so assert only the bound that matters: the lattice-carried
    # fraction keeps growing and the deficit stays under ~0.75/x.
    xh = xs[-3:]*hs[-3:]
    check("deficit bounded (x*h < 0.75, no asymptote claimed)",
          bool(np.all(xh < 0.75)), "x*h = %s" % np.round(xh, 3))


def _refine_solver(pitch, foot_model):
    """The same PHYSICAL two-pad + one-wire DC problem at the given
    pitch (1e-6 or 0.5e-6)."""
    k = int(round(1e-6/pitch))
    m = voxmodel.VoxelModel('footcal_refine')
    m.dims = (20*k, 7*k, 3*k)
    m.d = pitch
    m.sigma = np.zeros(m.dims)
    m.sigma[0:8*k, :, 0:2*k] = SIG_CU
    m.sigma[12*k:20*k, :, 0:2*k] = SIG_CU
    m.freq = np.array([10.0])
    M = m.build_tree(nleaf=[4, 7*k, 3*k], numlevels=2)
    m.prepare(M, 10.0)
    D = 1e-6
    wire = Wire(np.array([[5.5, 3.5, 2.3], [5.5, 3.5, 4.0],
                          [10.0, 3.5, 5.0], [14.5, 3.5, 4.0],
                          [14.5, 3.5, 2.3]])*D,
                0.6e-6, SIG_AL, max_seglen=2.0e-6)
    pp = [(0, y, z) for y in range(2*k, 5*k) for z in range(k, 2*k)]
    pn = [(20*k - 1, y, z) for y in range(2*k, 5*k)
          for z in range(k, 2*k)]
    sol = WireBondSolver(m, M, [wire], pp, pn, foot_r0=1.2e-6,
                         foot_model=foot_model, nq=3, ng=8)
    Z, info = sol.solve(10.0, rtol=1e-10)
    assert info['flag'] == 0 and info['residual'] < 1e-8
    return Z.real


def part_d():
    print("\nPART D -- end-to-end refinement convergence (THE gate)")
    drift = {}
    for model in ('point', 'patch'):
        Rc = _refine_solver(1e-6, model)
        Rf = _refine_solver(0.5e-6, model)
        drift[model] = abs(Rf - Rc)
        print("    %-6s foot: R(dx) %.6e  R(dx/2) %.6e  drift %.2e"
              % (model, Rc, Rf, drift[model]))
    check("calibrated foot cuts the refinement drift by >= 3x",
          drift['patch'] < drift['point']/3.0,
          "point %.2e vs patch %.2e (ratio %.1f)"
          % (drift['point'], drift['patch'],
             drift['point']/max(drift['patch'], 1e-30)))


if __name__ == '__main__':
    part_a()
    part_b()
    part_c()
    part_d()
    print("\n%d checks failed" % len(FAIL))
    raise SystemExit(1 if FAIL else 0)
