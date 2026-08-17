# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""GEOMETRIC multigrid on the loop (plaquette) basis -- a matrix-free
alternative to the AMG hierarchy that is the memory wall at scale.

THE PROBLEM IT ATTACKS -- AND A CORRECTION MEASURED HERE FIRST. This
file was written to escape a supposed AMG memory wall: 1.39 GB at 186k
loops reads as ~7.5 kB/loop, which would extrapolate to ~7 TB at a
billion loops. THAT EXTRAPOLATION IS WRONG BY ~30x. Measured retained
hierarchy (data + indices + indptr of every level operator and
prolongator), over a 17x size range:

    plaquettes   3750    11960    27466    63812
    AMG  B/loop  220.6    232.5    235.8    240.1
    Geo  B/loop  183.8    188.2    192.1    193.6

Both are FLAT per loop -- textbook multigrid complexity -- so at 1e9
loops the stored hierarchy is ~240 GB (AMG) or ~190 GB (geometric),
NOT terabytes. The 1.39 GB figure was a process PEAK RSS during setup,
which includes Y, the Gram and the sparse-product temporaries; reading
it as the retained hierarchy conflated a transient with a residency.
The stored hierarchy is therefore NOT the barrier to extreme scale.

What remains genuinely worth attacking: (a) the SETUP TRANSIENT, which
is what that 1.39 GB actually measured, and where a piecewise-constant
P should be much lighter than smoothed aggregation's densified one --
not yet measured; (b) the AMG APPLY at ~49% of solve time.

WHY GEOMETRY SHOULD WIN HERE. The loop space is not an arbitrary
algebraic space: it is the cycle space ``ker(A^T) = im(d_2)``, so the
plaquette basis IS the discrete curl of face variables, on a UNIFORM
VOXEL LATTICE. The coarsening AMG has to discover by inspecting matrix
entries is already known -- it is the lattice, halved. And the SuperPEEC
carries per-level occupancy (``Level.struc``) so the hierarchy exists
before we start.

PLAQUETTE GEOMETRY IS RECOVERABLE, which is the enabling trick here:
``getmesh_full`` returns quads in Fortran emission order with no
geometric labelling, but every column has exactly 4 filament rows, and
``filament_cells`` gives each filament an axis and a cell. A face's 4
edges span exactly TWO axes, so the missing third axis is its NORMAL,
and the componentwise min of the 4 cells is its position. That gives
(normal, i, j, k) per column, which is all a geometric hierarchy needs.

WHAT THIS FILE MEASURES (the honest A/B, same matrix, same rhs):
setup time, HIERARCHY MEMORY, and convergence, against pyamg on the
identical Gram. Memory is the number that decides the billion-cell
question; iterations decide whether it is usable at all.

Usage:
    PYTHONPATH=.:studies python3 studies/geomg.py
Env: NX/NY/NZ (block size in cells, default 30/30/8), NU (smoother
     sweeps, 2), CYCLES (V-cycles per apply, 2), MAXC (coarsest size,
     400), OMEGA (Jacobi damping, 0.667)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def plaquette_geometry(Y, fil_axis, fil_cell, nplaq):
    """(normal, cell) per plaquette column of ``Y``.

    A lattice face has 4 edges spanning exactly two axes; the third is
    its normal. Position is the componentwise min of the 4 edge cells.
    Vectorised over all columns at once -- every column has exactly 4
    nonzeros, so the index array reshapes to (nplaq, 4).
    """
    Yc = Y.tocsc()
    rows = Yc.indices[:4*nplaq].reshape(nplaq, 4)
    ax = np.asarray(fil_axis)[rows]                 # (nplaq, 4)
    cells = np.asarray(fil_cell)[rows]              # (nplaq, 4, 3)
    # normal = the axis not present among the 4 edges: 0+1+2 minus the
    # two distinct axes present
    amin = ax.min(axis=1)
    amax = ax.max(axis=1)
    normal = 3 - amin - amax
    base = cells.min(axis=1)
    ok = amin != amax
    if not np.all(ok):
        raise RuntimeError("%d plaquette(s) span a single axis -- not a "
                           "lattice face" % int((~ok).sum()))
    return normal.astype(np.int64), base.astype(np.int64)


class GeoMG:
    """Matrix-free-hierarchy geometric multigrid on plaquette DOFs.

    Aggregates 2x2x2 blocks of same-normal faces per level (SEMI-
    COARSENING: an axis whose extent is already small is not coarsened,
    which is the right behaviour for the pancake geometries SuperPEEC
    partitions into anyway -- a 4-cell-thick board must not be coarsened
    in z more than once). Coarse operators are Galerkin ``P^T A P`` with
    a PIECEWISE-CONSTANT P: unlike smoothed aggregation, P is not
    smoothed, so the coarse operators do not densify and the hierarchy
    stays cheap -- which is the entire point.

    Smoother: damped Jacobi. Deliberately not Gauss-Seidel: Jacobi is
    what parallelises on the many-core machine this exists to reach, and
    the GPU AMG experiment already had to swap GS out for the same
    reason.

    NOTE ON THE KERNEL. ``Y^T Y`` is SINGULAR here -- its kernel is the
    cube boundaries (a current differing by a cube boundary is the same
    current). Unlike the classic H(curl) case there is no mass term, so
    the kernel is EXACT rather than near-kernel: A annihilates it, a
    consistent rhs never excites it, and no Hiptmair-style auxiliary
    space correction is needed to damp it. The coarsest level is solved
    with a pseudo-inverse for the same reason.
    """

    def __init__(self, A, normal, base, nu=2, omega=2.0/3.0,
                 max_coarse=400, max_levels=12):
        self.nu = int(nu)
        self.omega = float(omega)
        self.levels = []
        self.Ps = []
        A = A.tocsr()
        nrm, bs = normal.copy(), base.copy()
        for _ in range(max_levels):
            self.levels.append(A)
            if A.shape[0] <= max_coarse:
                break
            P, nrm, bs = self._aggregate(nrm, bs)
            if P.shape[1] >= A.shape[0]:
                break                      # coarsening stalled
            self.Ps.append(P.tocsr())
            A = (P.T @ A @ P).tocsr()
        self.dinv = [1.0/np.where(np.abs(L.diagonal()) > 0,
                                  L.diagonal(), 1.0)
                     for L in self.levels]
        Ac = self.levels[-1]
        self.coarse_pinv = np.linalg.pinv(Ac.toarray()) \
            if Ac.shape[0] <= 2000 else None
        self.nnz = sum(L.nnz for L in self.levels)
        self.nnz_ratio = self.nnz/max(self.levels[0].nnz, 1)
        self.sizes = [L.shape[0] for L in self.levels]

    def _aggregate(self, normal, base):
        """2x2x2 geometric agglomeration, per face orientation."""
        span = base.max(axis=0) - base.min(axis=0) + 1
        div = np.where(span > 2, 2, 1)            # semi-coarsening
        cb = base//div[None, :]
        key = np.stack([normal, cb[:, 0], cb[:, 1], cb[:, 2]], axis=1)
        # np.unique returns (unique, index, inverse) IN THAT ORDER when
        # both flags are set -- binding them the other way round gives a
        # P with the coarse count as its ROW dimension and a dimension
        # mismatch two levels later.
        _, first, inv = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
        inv = np.asarray(inv).ravel()
        nc = int(inv.max()) + 1
        P = sp.coo_matrix((np.ones(inv.size), (np.arange(inv.size), inv)),
                          shape=(inv.size, nc))
        return P.tocsc(), normal[first], cb[first]

    def _smooth(self, lv, x, b):
        A, di = self.levels[lv], self.dinv[lv]
        for _ in range(self.nu):
            x = x + self.omega*di*(b - A @ x)
        return x

    def _vcycle(self, lv, b, x):
        if lv == len(self.levels) - 1:
            if self.coarse_pinv is not None:
                return self.coarse_pinv @ b
            return spla.lsqr(self.levels[lv], b, atol=1e-10, btol=1e-10)[0]
        x = self._smooth(lv, x, b)
        r = b - self.levels[lv] @ x
        P = self.Ps[lv]
        xc = self._vcycle(lv + 1, P.T @ r, np.zeros(P.shape[1]))
        x = x + P @ xc
        return self._smooth(lv, x, b)

    def __call__(self, b, cycles=1):
        x = np.zeros(b.shape[0])
        for _ in range(cycles):
            x = self._vcycle(0, np.asarray(b, dtype=np.float64), x)
        return x


def build_gram(nx, ny, nz):
    """A solid conductor block -> (Gram of the plaquette basis, geometry)."""
    import voxmodel
    import equiterminal as eq
    import meshgraph as mg
    m = voxmodel.VoxelModel('block')
    m.dims = (nx, ny, nz)
    m.d = 1e-4
    m.sigma = np.full((nx, ny, nz), 5.8e7)
    m.freq = np.array([1e8])
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, 1e8)
    efg = (np.size(M.e.struc) + np.size(M.f.struc) + np.size(M.g.struc))
    nn = np.size(M.lv[0].struc)
    Y = mg.getmesh_full(M.adjmats(), np.size(M.e.struc),
                        np.size(M.e.struc) + np.size(M.f.struc), efg, nn)
    Y.data = np.float64(Y.data)
    fil_axis, fil_cell = eq.filament_cells(M)
    normal, base = plaquette_geometry(Y, fil_axis, fil_cell, Y.shape[1])
    G = (Y.T @ Y).tocsr()
    return G, normal, base, Y


def hierarchy_bytes(mats):
    """Retained bytes of a sparse hierarchy (data + indices + indptr)."""
    tot = 0
    for A in mats:
        A = A.tocsr()
        tot += A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
    return tot


def scan():
    """Memory scaling law: bytes of RETAINED hierarchy per loop, both
    methods, as the block grows. This is the number my billion-cell
    estimate turned on, so measure it rather than extrapolate it."""
    import pyamg
    print("%9s %9s %10s %10s %10s %10s %8s"
          % ("plaquettes", "Gram nnz", "geo B/loop", "amg B/loop",
             "geo ratio", "amg ratio", "geo/amg"), flush=True)
    for (nx, ny, nz) in ((16, 16, 6), (24, 24, 8), (32, 32, 10),
                         (44, 44, 12)):
        G, normal, base, _ = build_gram(nx, ny, nz)
        n = G.shape[0]
        gm = GeoMG(G, normal, base, max_coarse=400)
        gb = hierarchy_bytes(gm.levels) + hierarchy_bytes(gm.Ps)
        ml = pyamg.smoothed_aggregation_solver(G.astype(np.float64),
                                               max_coarse=400)
        ab = hierarchy_bytes([l.A for l in ml.levels]) \
            + hierarchy_bytes([l.P for l in ml.levels if hasattr(l, 'P')])
        print("%9d %9d %10.1f %10.1f %10.2f %10.2f %8.3f"
              % (n, G.nnz, gb/n, ab/n, gb/(G.data.nbytes + G.indices.nbytes),
                 ab/(G.data.nbytes + G.indices.nbytes), gb/max(ab, 1)),
              flush=True)


def main():
    if os.environ.get('SCAN', '0') == '1':
        scan()
        return
    nx = int(os.environ.get('NX', '30'))
    ny = int(os.environ.get('NY', '30'))
    nz = int(os.environ.get('NZ', '8'))
    nu = int(os.environ.get('NU', '2'))
    cycles = int(os.environ.get('CYCLES', '2'))
    maxc = int(os.environ.get('MAXC', '400'))
    omega = float(os.environ.get('OMEGA', '0.667'))

    t0 = time.perf_counter()
    G, normal, base, Y = build_gram(nx, ny, nz)
    print("block %dx%dx%d: %d plaquettes, Gram nnz %d  (%.1f s)"
          % (nx, ny, nz, G.shape[0], G.nnz, time.perf_counter() - t0),
          flush=True)
    print("  face orientations: %s"
          % np.bincount(normal, minlength=3).tolist(), flush=True)

    t0 = time.perf_counter()
    gm = GeoMG(G, normal, base, nu=nu, omega=omega, max_coarse=maxc)
    t_geo = time.perf_counter() - t0
    print("  GeoMG   setup %6.2f s   levels %s   hierarchy nnz %9d "
          "(%.2fx fine)"
          % (t_geo, gm.sizes, gm.nnz, gm.nnz_ratio), flush=True)

    import pyamg
    t0 = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(G.astype(np.float64),
                                           max_coarse=maxc)
    t_amg = time.perf_counter() - t0
    amg_nnz = sum(l.A.nnz for l in ml.levels)
    print("  pyamg   setup %6.2f s   levels %s   hierarchy nnz %9d "
          "(%.2fx fine)"
          % (t_amg, [l.A.shape[0] for l in ml.levels], amg_nnz,
             amg_nnz/G.nnz), flush=True)
    print("  MEMORY RATIO geo/amg = %.3f   (hierarchy nnz; the number "
          "that decides the billion-cell budget)"
          % (gm.nnz/max(amg_nnz, 1)), flush=True)

    # CONSISTENT rhs: G is singular (cube boundaries), so drive it with
    # something in its range -- G v for a random v -- exactly as the
    # Krylov solve does.
    rng = np.random.default_rng(0)
    v = rng.standard_normal(G.shape[0])
    b = G @ v
    nb = np.linalg.norm(b)

    for name, apply_ in (("GeoMG", lambda r: gm(r, cycles=cycles)),
                         ("pyamg", lambda r: _amg_apply(ml, r, cycles))):
        x = np.zeros_like(b)
        t0 = time.perf_counter()
        hist = []
        for _ in range(30):
            x = x + apply_(b - G @ x)
            hist.append(np.linalg.norm(b - G @ x)/nb)
            if hist[-1] < 1e-8:
                break
        rho = (hist[-1]/hist[0])**(1.0/max(len(hist) - 1, 1))
        print("  %s stationary: %2d iters to %.2e, mean reduction "
              "%.3f/iter, %.2f s" % (name, len(hist), hist[-1], rho,
                                     time.perf_counter() - t0), flush=True)


def _amg_apply(ml, r, cycles):
    x = np.zeros(r.shape[0])
    for _ in range(cycles):
        x = ml.solve(r, x0=x, tol=1e-14, maxiter=1, cycle='V')
    return x


if __name__ == '__main__':
    main()
