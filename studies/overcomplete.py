# SPDX-License-Identifier: MIT
"""Does the OVER-COMPLETE plaquette Gram still beat the selected cycle
basis on ELONGATED and IRREGULAR geometries?

The win was measured only on small CHUNKY FILLED boxes. Real targets are
elongated bars, staircase-boundary wires and bent traces. This builds
from an arbitrary occupancy mask and compares, for each geometry:
    selected basis (dual spanning tree, axis-greedy = what getmesh does)
    over-complete (ALL plaquettes; singular, kernel = cube boundaries)
on the metric that matters: AMG + PCG iterations, and memory.
Singular but CONSISTENT -> take b in the range.
"""
import sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import pyamg


def geometry(name):
    if name.startswith('elong'):                  # elong<L>x<W>
        if 'x' in name:
            L, W = (int(v) for v in name[5:].split('x'))
        else:
            L, W = 40, 6
        occ = np.ones((L, W, W), bool)
    elif name == 'chunky':                        # control, as before
        occ = np.ones((12, 6, 6), bool)
    elif name == 'wire':                          # staircase circular x-sec
        occ = np.zeros((30, 9, 9), bool)
        jj, kk = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
        disc = ((jj-4.0)**2 + (kk-4.0)**2) < 16.0
        occ[:, :, :] = disc[None, :, :]
    elif name == 'Lbend':                         # non-convex
        occ = np.zeros((24, 24, 5), bool)
        occ[:6, :, :] = True
        occ[:, :6, :] = True
    elif name == 'hollow':                        # cavity -> extra topology
        occ = np.ones((16, 8, 8), bool)
        occ[4:12, 3:5, 3:5] = False
    else:
        raise SystemExit('unknown geometry')
    return occ


def build(occ):
    NX, NY, NZ = occ.shape
    nid = -np.ones(occ.shape, int)
    nid[occ] = np.arange(int(occ.sum()))
    ex = -np.ones(occ.shape, int); ey = ex.copy(); ez = ex.copy()
    c = 0
    for a, arr, d in ((0, ex, (1, 0, 0)), (1, ey, (0, 1, 0)),
                      (2, ez, (0, 0, 1))):
        for i in range(NX-d[0]):
            for j in range(NY-d[1]):
                for k in range(NZ-d[2]):
                    if occ[i, j, k] and occ[i+d[0], j+d[1], k+d[2]]:
                        arr[i, j, k] = c; c += 1
    E = c
    rows, cols, vals, kind, dual = [], [], [], [], []
    cid = -np.ones((NX-1, NY-1, NZ-1), int); nc = 0
    for i in range(NX-1):
        for j in range(NY-1):
            for k in range(NZ-1):
                if occ[i:i+2, j:j+2, k:k+2].all():
                    cid[i, j, k] = nc; nc += 1
    OUT = nc

    def cub(i, j, k):
        if 0 <= i < NX-1 and 0 <= j < NY-1 and 0 <= k < NZ-1 and cid[i, j, k] >= 0:
            return cid[i, j, k]
        return OUT
    p = 0
    def emit(edges, orient, ca, cb):
        nonlocal p
        if any(e < 0 for e, _ in edges):
            return
        for e, s in edges:
            rows.append(e); cols.append(p); vals.append(s)
        kind.append(orient); dual.append((ca, cb)); p += 1
    for i in range(NX-1):
        for j in range(NY-1):
            for k in range(NZ):
                emit([(ex[i, j, k], 1), (ey[i+1, j, k], 1),
                      (ex[i, j+1, k], -1), (ey[i, j, k], -1)], 0,
                     cub(i, j, k-1), cub(i, j, k))
    for i in range(NX-1):
        for j in range(NY):
            for k in range(NZ-1):
                emit([(ex[i, j, k], 1), (ez[i+1, j, k], 1),
                      (ex[i, j, k+1], -1), (ez[i, j, k], -1)], 1,
                     cub(i, j-1, k), cub(i, j, k))
    for i in range(NX):
        for j in range(NY-1):
            for k in range(NZ-1):
                emit([(ey[i, j, k], 1), (ez[i, j+1, k], 1),
                      (ey[i, j, k+1], -1), (ez[i, j, k], -1)], 2,
                     cub(i-1, j, k), cub(i, j, k))
    Y = sp.csc_matrix((vals, (rows, cols)), shape=(E, p))
    return Y, np.array(kind), np.array(dual), int(occ.sum()), E, nc


def sel_axis_greedy(Y, kind, dual, ncubes):
    order = np.concatenate([np.flatnonzero(kind == 1),
                            np.flatnonzero(kind == 0),
                            np.flatnonzero(kind == 2)])
    par = list(range(ncubes+1))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    drop = []
    for pi in order:
        a, b = find(dual[pi, 0]), find(dual[pi, 1])
        if a != b:
            par[a] = b; drop.append(pi)
    return np.setdiff1d(np.arange(Y.shape[1]), np.array(drop))


def amg_iters(G):
    rng = np.random.default_rng(1)
    b = G @ rng.standard_normal(G.shape[0])       # in range
    ml = pyamg.smoothed_aggregation_solver(G, max_coarse=400)
    nz = sum(l.A.nnz for l in ml.levels)
    it = [0]
    x, _ = sla.cg(G, b, M=ml.aspreconditioner(cycle='V'), rtol=1e-10,
                  maxiter=600, callback=lambda xk: it.__setitem__(0, it[0]+1))
    r = np.linalg.norm(G@x-b)/max(np.linalg.norm(b), 1e-300)
    return it[0], nz/G.nnz, r


print("%-10s %-8s %-7s %-7s  %-22s  %-22s"
      % ("geometry", "nodes", "plaq", "rank", "SELECTED basis (AMG)",
         "OVER-COMPLETE (AMG)"))
for name in sys.argv[1:]:
    occ = geometry(name)
    Y, kind, dual, V, E, nc = build(occ)
    P = Y.shape[1]
    keep = sel_axis_greedy(Y, kind, dual, nc)
    Gs = (Y[:, keep].T @ Y[:, keep]).tocsr().astype(np.float64)
    Gf = (Y.T @ Y).tocsr().astype(np.float64)
    its, mems, rs = amg_iters(Gs)
    itf, memf, rf = amg_iters(Gf)
    print("%-10s %-8d %-7d %-7d  %4d it %5.1fx %7.0e   %4d it %5.1fx %7.0e"
          % (name, V, P, keep.size, its, mems, rs, itf, memf, rf), flush=True)
