# SPDX-License-Identifier: MIT
"""Does deflating the 361-dim isolated cluster fix AMG on G = Y^T Y?

Inertia count (studies/inertia.py): 361 eigenvalues in [4.18e-4, 8e-4],
then a GAP -- only one more below 3e-3. Classic deflation territory.

Additive two-level preconditioner with EXACT eigenvectors W (so
W^T G W = diag(lambda)):
        M^-1 r  =  M_AMG^-1 r  +  W diag(1/lambda) W^T r

CAVEAT, stated up front: computing W by shift-invert needs the Cholesky
factor we are trying to REPLACE. This measures whether deflation WORKS,
not whether it is yet practical. If it works, the question becomes
whether the space can be CONSTRUCTED from geometry (361 = 19^2 and the
bar is 60x20x20, whose 20x20 cross-section has 19x19 interior sites).
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
print("G: n=%d  nnz=%d" % (n, G.nnz), flush=True)

F = cholmod.cholesky(G.tocsc(), ordering_method='metis')
OP = sla.LinearOperator((n, n), matvec=lambda v: F(v))
NEV = 420
t0 = time.perf_counter()
w, V = sla.eigsh(G, k=NEV, sigma=0.0, which='LM', OPinv=OP, tol=1e-8)
idx = np.argsort(w); w, V = w[idx], V[:, idx]
print("computed %d eigenpairs in %.1f s: lambda in [%.3e, %.3e]"
      % (NEV, time.perf_counter()-t0, w[0], w[-1]), flush=True)
print("  #below 8e-4 = %d   (cluster);  #below 3e-3 = %d"
      % (int((w < 8e-4).sum()), int((w < 3e-3).sum())), flush=True)
print("  deflation space memory at m=361: %.0f MB (chol was 131 MB)"
      % (361*n*8/1e6), flush=True)

ml = pyamg.smoothed_aggregation_solver(G, max_coarse=500)
Pamg = ml.aspreconditioner(cycle='V')
rng = np.random.default_rng(0); b = rng.standard_normal(n)


def run(mdef, use_amg=True):
    if mdef:
        W, lam = V[:, :mdef], w[:mdef]
        def mv(r):
            x = W @ ((W.T @ r)/lam)
            return x + Pamg.matvec(r) if use_amg else x
    else:
        mv = Pamg.matvec
    Mop = sla.LinearOperator((n, n), matvec=mv)
    it = [0]
    t0 = time.perf_counter()
    x, info = sla.cg(G, b, M=Mop, rtol=1e-10, maxiter=1000,
                     callback=lambda xk: it.__setitem__(0, it[0]+1))
    r = np.linalg.norm(G@x-b)/np.linalg.norm(b)
    return it[0], time.perf_counter()-t0, r


print("\n%-28s %-7s %-8s %s" % ("preconditioner", "iters", "sec", "resid"))
for mdef, amg, label in [(0, True, "AMG alone"),
                         (50, True, "AMG + deflate 50"),
                         (150, True, "AMG + deflate 150"),
                         (361, True, "AMG + deflate 361 (cluster)"),
                         (420, True, "AMG + deflate 420"),
                         (361, False, "deflate 361 ALONE (no AMG)")]:
    it, ts, r = run(mdef, amg)
    print("%-28s %-7d %-8.2f %.1e" % (label, it, ts, r), flush=True)
