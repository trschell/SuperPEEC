# SPDX-License-Identifier: MIT
"""HX step 1: WHAT is the near-null space of G = Y^T Y, and is supplying
it to AMG sufficient?

HX works by handling the near-null space explicitly, so before building
any d3-like operator we must know what that space IS. Compute the lowest
eigenvectors numerically, characterise them, then feed them to SA as
candidates. If SA-with-true-candidates converges fast, the HX route is
sound and we know precisely what cheap generator to build. If it does
NOT, the near-null space is not the obstacle and HX would be wasted.
"""
import time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import vhr, meshgraph as mg, pyamg
from sksparse import cholmod

m = vhr.read_vhr('VoxHenry/Input_files/'
                 'straight_cond1_len30.0u_wid10.0u_dist20.0u.vhr')
M = m.build_tree(); m.prepare(M, 1e10)
es, fs_, gs = np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc)
efg, nn = es+fs_+gs, np.size(M.lv[0].struc)
Y = sp.csc_matrix(mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn))
Y.data = np.float64(Y.data)
G = (Y.T@Y).tocsr().astype(np.float64)
n = G.shape[0]
F = cholmod.cholesky(G.tocsc(), ordering_method='metis')
OP = sla.LinearOperator((n, n), matvec=lambda v: F(v))

NC = 24
t0 = time.perf_counter()
w, V = sla.eigsh(G, k=NC, sigma=0.0, which='LM', OPinv=OP, tol=1e-8)
print("lowest %d eigenvalues in [%.3e, %.3e]  (lambda_max=11.95), %.1f s"
      % (NC, w.min(), w.max(), time.perf_counter()-t0))

# characterise: are they LARGE SURFACES?  a surface-like vector has most
# of its mass spread over many plaquettes with coherent sign.
for j in (0, NC//2, NC-1):
    v = V[:, j]
    a = np.abs(v)
    part = (a.sum()**2)/(n*(a**2).sum())      # participation ratio in [0,1]
    print("  vec %2d: lambda=%.3e  participation=%.3f  "
          "|Yv|/|v|=%.3e  sign split %.0f%%/%.0f%%"
          % (j, w[j], part, np.linalg.norm(Y@v)/np.linalg.norm(v),
             100*(v > 0).mean(), 100*(v < 0).mean()))

rng = np.random.default_rng(0); b = rng.standard_normal(n)
print()
for label, B in (("SA default (ones)", None),
                 ("SA + %d true near-null vecs" % NC, V)):
    t0 = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(
        G, B=(np.ones((n, 1)) if B is None else B), max_coarse=500)
    ts = time.perf_counter()-t0
    nz = sum(l.A.nnz for l in ml.levels)
    it = [0]
    x, info = sla.cg(G, b, M=ml.aspreconditioner(cycle='V'), rtol=1e-10,
                     maxiter=500, callback=lambda xk: it.__setitem__(0, it[0]+1))
    print("%-32s setup %5.2f s, nnz %5.1fx -> PCG %3d iters"
          % (label, ts, nz/G.nnz, it[0]))
