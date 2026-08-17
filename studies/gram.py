# SPDX-License-Identifier: MIT
"""What IS the loop Gram matrix Y^T Y, and why does it fill in?

Y maps LOOP space -> EDGE space: it is the discrete boundary operator
d_2 (2-chains -> 1-chains), i.e. a curl. So Y^T Y = d_2^T d_2 is the
"down Laplacian" on 2-chains -- a Laplacian-LIKE SPD operator on a
structured grid, NOT an arbitrary sparse matrix. That identification is
what makes AMG/multigrid/FFT plausible here where they failed on A^T K A.
"""
import numpy as np, scipy.sparse as sp, vhr, meshgraph as mg
from sksparse import cholmod

m = vhr.read_vhr('VoxHenry/Input_files/'
                 'straight_cond1_len30.0u_wid10.0u_dist20.0u.vhr')
M = m.build_tree()
m.prepare(M, 1e10)
es, fs_, gs = np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc)
efg, nn = es+fs_+gs, np.size(M.lv[0].struc)
Y = mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn)
Y.data = np.float64(Y.data)
Y = sp.csc_matrix(Y)
nl = Y.shape[1]
print("edges(efg)=%d  nodes=%d  loops=%d   (cycle rank E-V+1 = %d)"
      % (efg, nn, nl, efg-nn+1))
cpc = np.diff(Y.indptr)
print("nnz(Y)=%d   per LOOP column: min=%d max=%d mean=%.2f"
      % (Y.nnz, cpc.min(), cpc.max(), cpc.mean()))
vals, cnt = np.unique(Y.data, return_counts=True)
print("  Y entries:", dict(zip(vals.astype(int), cnt)))

G = (Y.T @ Y).tocsr()
rpr = np.diff(G.indptr)
print("\nY^T Y: %dx%d  nnz=%d  per ROW: min=%d max=%d mean=%.2f"
      % (nl, nl, G.nnz, rpr.min(), rpr.max(), rpr.mean()))
d = G.diagonal()
off = G - sp.diags(d)
ov, oc = np.unique(off.data, return_counts=True)
print("  diagonal: min=%g max=%g   off-diagonal values: %s"
      % (d.min(), d.max(), dict(zip(ov.astype(int), oc))))
print("  M-matrix? (all off-diagonals <= 0):", bool((off.data <= 0).all()))

for om in ('amd', 'metis'):
    F = cholmod.cholesky(G.tocsc(), ordering_method=om)
    nzL = F.L().nnz
    print("  chol(%-6s) nnz(L)=%10d   fill ratio vs nnz(G) = %5.1fx"
          % (om, nzL, nzL/G.nnz))
    del F
