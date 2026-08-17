# SPDX-License-Identifier: MIT
"""Is the ill-conditioning caused by the ASYMMETRIC plaquette selection?

Plaquettes over-span the cycle space; the relations are exactly the cube
boundaries. An independent subset = drop a SPANNING TREE OF THE DUAL
GRAPH (cubes + one 'outside' node; each plaquette is a dual edge joining
the two cubes it separates). A tree on Nc+1 nodes has Nc edges = exactly
the number that must be dropped.

HOW the tree grows sets the orientation balance. Compare:
  axis-greedy  -- tree grown preferring one dual direction, mimicking
                  what getmesh_fortran evidently does (it keeps 98.9%
                  xy, 98.2% yz but only 7.8% xz)
  balanced     -- round-robin over the three dual directions
  random       -- randomised spanning tree
Self-contained: builds the lattice from scratch, no SuperPEEC indexing.
"""
import numpy as np, scipy.sparse as sp

import sys
NX, NY, NZ = (tuple(int(v) for v in sys.argv[1].split("x"))
              if len(sys.argv) > 1 else (8, 5, 5))   # NODE grid
cx, cy, cz = NX-1, NY-1, NZ-1  # cube grid


def eidx():
    ex = -np.ones((NX, NY, NZ), int); ey = ex.copy(); ez = ex.copy()
    c = 0
    for i in range(NX-1):
        for j in range(NY):
            for k in range(NZ):
                ex[i, j, k] = c; c += 1
    for i in range(NX):
        for j in range(NY-1):
            for k in range(NZ):
                ey[i, j, k] = c; c += 1
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ-1):
                ez[i, j, k] = c; c += 1
    return ex, ey, ez, c


ex, ey, ez, E = eidx()
V = NX*NY*NZ
rows, cols, vals, kind, dual = [], [], [], [], []


def add(p, edges, orient, ca, cb):
    for e, s in edges:
        rows.append(e); cols.append(p); vals.append(s)
    kind.append(orient); dual.append((ca, cb))


def cube(i, j, k):
    return (i*cy + j)*cz + k if (0 <= i < cx and 0 <= j < cy
                                 and 0 <= k < cz) else cx*cy*cz  # outside


p = 0
for i in range(NX-1):
    for j in range(NY-1):
        for k in range(NZ):
            add(p, [(ex[i, j, k], 1), (ey[i+1, j, k], 1),
                    (ex[i, j+1, k], -1), (ey[i, j, k], -1)], 0,
                cube(i, j, k-1), cube(i, j, k)); p += 1
for i in range(NX-1):
    for j in range(NY):
        for k in range(NZ-1):
            add(p, [(ex[i, j, k], 1), (ez[i+1, j, k], 1),
                    (ex[i, j, k+1], -1), (ez[i, j, k], -1)], 1,
                cube(i, j-1, k), cube(i, j, k)); p += 1
for i in range(NX):
    for j in range(NY-1):
        for k in range(NZ-1):
            add(p, [(ey[i, j, k], 1), (ez[i, j+1, k], 1),
                    (ey[i, j, k+1], -1), (ez[i, j, k], -1)], 2,
                cube(i-1, j, k), cube(i, j, k)); p += 1
P = p
kind = np.array(kind); dual = np.array(dual)
Yfull = sp.csc_matrix((vals, (rows, cols)), shape=(E, P))
rank = E - V + 1
print("nodes=%d edges=%d plaquettes=%d cubes=%d  cycle rank=%d  drop=%d"
      % (V, E, P, cx*cy*cz, rank, P-rank))
print("available by orientation: xy=%d xz=%d yz=%d"
      % ((kind == 0).sum(), (kind == 1).sum(), (kind == 2).sum()))


def spanning_drop(order):
    """Kruskal over dual edges in the given priority order -> the tree
    edges are the plaquettes to DROP."""
    nd = cx*cy*cz + 1
    par = list(range(nd))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    drop = []
    for pi in order:
        a, b = find(dual[pi, 0]), find(dual[pi, 1])
        if a != b:
            par[a] = b; drop.append(pi)
    return np.array(drop)


rng = np.random.default_rng(0)
schemes = {}
schemes['axis-greedy(xz last)'] = np.concatenate(
    [np.flatnonzero(kind == 1), np.flatnonzero(kind == 0),
     np.flatnonzero(kind == 2)])
rr = []
byk = [list(np.flatnonzero(kind == k)) for k in range(3)]
while any(byk):
    for k in range(3):
        if byk[k]:
            rr.append(byk[k].pop(0))
schemes['balanced round-robin'] = np.array(rr)
schemes['random'] = rng.permutation(P)

print("\n%-22s %-22s %-11s %s"
      % ("scheme", "kept xy/xz/yz", "cond(G)", "modes < lmax/1000"))
for name, order in schemes.items():
    drop = spanning_drop(order)
    keep = np.setdiff1d(np.arange(P), drop)
    if keep.size != rank:
        print("%-22s INVALID: kept %d != rank %d" % (name, keep.size, rank))
        continue
    Y = Yfull[:, keep]
    G = (Y.T@Y).toarray()
    ev = np.linalg.eigvalsh(G)
    kk = kind[keep]
    print("%-22s %-22s %-11.4g %d"
          % (name, "%d/%d/%d" % ((kk == 0).sum(), (kk == 1).sum(),
                                 (kk == 2).sum()),
             ev[-1]/ev[0], int((ev < ev[-1]/1000).sum())))


# ---- OVER-COMPLETE full plaquette Gram -----------------------------------
# Use ALL plaquettes instead of selecting rank-many. G_full is SINGULAR:
# ker = the cube boundaries, which are PHYSICALLY MEANINGLESS (currents
# differing by a cube boundary are the same current), so a consistent
# singular system is fine for Krylov. The prize is that the full set is
# TRANSLATION-INVARIANT on a filled region -> block-Toeplitz ->
# FFT-diagonalisable with zero fill. Question: is it better conditioned
# ON ITS RANGE than the selected basis?
Gf = (Yfull.T @ Yfull).toarray()
evf = np.linalg.eigvalsh(Gf)
ncubes = cx*cy*cz
tol = evf[-1]*1e-10
nz = int((evf <= tol).sum())
pos = evf[evf > tol]
drop = spanning_drop(schemes['axis-greedy(xz last)'])
keep = np.setdiff1d(np.arange(P), drop)
Gs = (Yfull[:, keep].T @ Yfull[:, keep]).toarray()
evs = np.linalg.eigvalsh(Gs)
print()
print("OVER-COMPLETE  P=%d  null dim=%d (cubes=%d, match=%s)  rank=%d"
      % (P, nz, ncubes, nz == ncubes, P-nz))
print("  full   : lmax=%.4g  lmin(nonzero)=%.4g  cond_on_range=%.4g"
      % (pos[-1], pos[0], pos[-1]/pos[0]))
print("  selected: lmax=%.4g  lmin=%.4g  cond=%.4g"
      % (evs[-1], evs[0], evs[-1]/evs[0]))
print("  RATIO cond_selected / cond_full_on_range = %.3f"
      % ((evs[-1]/evs[0])/(pos[-1]/pos[0])))


# ---- does AMG work on the OVER-COMPLETE Gram? ----------------------------
# It failed badly on the selected basis (265-369 PCG iters). G_full is
# ~10x better conditioned and scales like a Laplacian, so retry. Singular
# but CONSISTENT: take b in the range.
import scipy.sparse.linalg as _sla, pyamg as _pyamg
Gfs = (Yfull.T @ Yfull).tocsr().astype(np.float64)
rng2 = np.random.default_rng(1)
b = Gfs @ rng2.standard_normal(P)          # guaranteed in range
try:
    mlf = _pyamg.smoothed_aggregation_solver(Gfs, max_coarse=400)
    nzf = sum(l.A.nnz for l in mlf.levels)
    it = [0]
    x, info = _sla.cg(Gfs, b, M=mlf.aspreconditioner(cycle='V'),
                      rtol=1e-10, maxiter=600,
                      callback=lambda xk: it.__setitem__(0, it[0]+1))
    r = np.linalg.norm(Gfs@x - b)/np.linalg.norm(b)
    print("  AMG on FULL Gram : %3d PCG iters, nnz %.1fx, resid %.1e"
          % (it[0], nzf/Gfs.nnz, r))
except Exception as e:
    print("  AMG on FULL Gram FAILED:", type(e).__name__, str(e)[:60])
it = [0]
x, info = _sla.cg(Gfs, b, rtol=1e-10, maxiter=600,
                  callback=lambda xk: it.__setitem__(0, it[0]+1))
print("  plain CG, no precond: %3d iters (cond_on_range predicts ~%d)"
      % (it[0], int(np.sqrt(pos[-1]/pos[0]))*2))
