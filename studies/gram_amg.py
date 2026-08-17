# SPDX-License-Identifier: MIT
"""Can AMG replace the exact Cholesky of the loop Gram Y^T Y?

The preconditioner's job is to fix the BASIS conditioning of Y (it is
topological and frequency-independent), so we need SPECTRAL EQUIVALENCE,
not an exact inverse -- which is exactly the regime AMG is built for.
Measures: conditioning of G, AMG setup cost/memory, and PCG iterations
to solve G x = b, against chol(G).
"""
import time, numpy as np, scipy.sparse as sp, vhr, meshgraph as mg
import scipy.sparse.linalg as sla
from sksparse import cholmod
import pyamg

m = vhr.read_vhr('VoxHenry/Input_files/'
                 'straight_cond1_len30.0u_wid10.0u_dist20.0u.vhr')
M = m.build_tree(); m.prepare(M, 1e10)
es, fs_, gs = np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc)
efg, nn = es+fs_+gs, np.size(M.lv[0].struc)
Y = sp.csc_matrix(mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn))
Y.data = np.float64(Y.data)
G = (Y.T @ Y).tocsr().astype(np.float64)
n = G.shape[0]
print("G: %d x %d, nnz=%d (%.2f/row)" % (n, n, G.nnz, G.nnz/n))

lmax = sla.eigsh(G, k=1, which='LA', return_eigenvectors=False,
                 tol=1e-3)[0]
lmin = sla.eigsh(G, k=1, sigma=0.0, which='LM',
                 return_eigenvectors=False, tol=1e-3)[0]
print("  lambda_min=%.4g  lambda_max=%.4g  cond=%.4g" % (lmin, lmax, lmax/lmin))

rng = np.random.default_rng(0); b = rng.standard_normal(n)

t0 = time.perf_counter()
F = cholmod.cholesky(G.tocsc(), ordering_method='metis')
tc = time.perf_counter()-t0
nzL = F.L().nnz
print("\nchol(metis): setup %.2f s, nnz(L)=%d (%.1fx nnz(G)), "
      "~%.0f MB" % (tc, nzL, nzL/G.nnz, nzL*12/1e6))

t0 = time.perf_counter()
ml = pyamg.smoothed_aggregation_solver(G, max_coarse=500)
ta = time.perf_counter()-t0
ops = ml.operator_complexity(); grd = ml.grid_complexity()
amg_nnz = sum(l.A.nnz for l in ml.levels)
print("AMG (SA):    setup %.2f s, levels=%d, op_cx=%.2f, grid_cx=%.2f, "
      "nnz(all levels)=%d (%.1fx), ~%.0f MB"
      % (ta, len(ml.levels), ops, grd, amg_nnz, amg_nnz/G.nnz,
         amg_nnz*12/1e6))

for name, Mop in (("AMG", ml.aspreconditioner(cycle='V')),
                  ("chol", sla.LinearOperator((n, n), matvec=F))):
    it = [0]
    t0 = time.perf_counter()
    x, info = sla.cg(G, b, M=Mop, rtol=1e-10, maxiter=500,
                     callback=lambda xk: it.__setitem__(0, it[0]+1))
    r = np.linalg.norm(G@x - b)/np.linalg.norm(b)
    print("  PCG with %-5s: %3d iters, %.2f s, resid %.1e"
          % (name, it[0], time.perf_counter()-t0, r))

# ---- the near-null space is NOT constants -------------------------------
# G = d2^T d2 is curl-curl-like: its small eigenvalues belong to nearly
# CLOSED surfaces, not to the constant vector SA assumes by default.
# Either discover the candidates (adaptive SA) or use a stronger
# coarsening (root-node / energy-minimising interpolation).
print()
import pyamg
from pyamg.aggregation import adaptive_sa_solver, rootnode_solver
trials = []
t0 = time.perf_counter()
try:
    ml2 = rootnode_solver(G, max_coarse=500)
    trials.append(("rootnode", ml2, time.perf_counter()-t0))
except Exception as e:
    print("  rootnode FAILED:", type(e).__name__, str(e)[:60])
t0 = time.perf_counter()
try:
    ml3 = adaptive_sa_solver(G, num_candidates=4, max_coarse=500)[0]
    trials.append(("adaptiveSA", ml3, time.perf_counter()-t0))
except Exception as e:
    print("  adaptive_sa FAILED:", type(e).__name__, str(e)[:60])
for name, mlx, ts in trials:
    nz = sum(l.A.nnz for l in mlx.levels)
    it = [0]
    t1 = time.perf_counter()
    x, info = sla.cg(G, b, M=mlx.aspreconditioner(cycle='V'), rtol=1e-10,
                     maxiter=500, callback=lambda xk: it.__setitem__(0, it[0]+1))
    r = np.linalg.norm(G@x - b)/np.linalg.norm(b)
    print("%-11s setup %5.2f s, nnz %.1fx (~%.0f MB) -> PCG %3d iters, "
          "%.2f s, resid %.1e"
          % (name, ts, nz/G.nnz, nz*12/1e6, it[0],
             time.perf_counter()-t1, r))
