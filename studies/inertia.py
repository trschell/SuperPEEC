# SPDX-License-Identifier: MIT
"""How MANY eigenvalues of G = Y^T Y sit in the near-null cluster?

Sylvester's law of inertia: #{eig(G) < sigma} = #negative pivots of the
LDL^T factorisation of (G - sigma I).  CHOLMOD refuses indefinite input,
so use SuperLU in symmetric mode with diagonal pivoting forced
(diag_pivot_thresh=0), where diag(U) IS the D of LDL^T.

Self-check: we already know the 120 smallest eigenvalues all lie in
[4.181e-4, 7.912e-4], so a shift below that must count 0 and a shift
above it must count >= 120. If those two brackets don't come out right,
the method is wrong and the numbers mean nothing.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import vhr, meshgraph as mg

m = vhr.read_vhr('VoxHenry/Input_files/'
                 'straight_cond1_len30.0u_wid10.0u_dist20.0u.vhr')
M = m.build_tree(); m.prepare(M, 1e10)
es, fs_, gs = np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc)
efg, nn = es+fs_+gs, np.size(M.lv[0].struc)
Y = sp.csc_matrix(mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn))
Y.data = np.float64(Y.data)
G = (Y.T@Y).tocsc().astype(np.float64)
n = G.shape[0]
I = sp.identity(n, format='csc')
print("G: n=%d  voxels=%d  edges=%d   lambda_max=11.95" % (n, nn, efg))


def count_below(sigma):
    A = (G - sigma*I).tocsc()
    lu = sla.splu(A, diag_pivot_thresh=0.0, permc_spec='MMD_AT_PLUS_A',
                  options=dict(SymmetricMode=True))
    return int((lu.U.diagonal() < 0).sum())


print("\n%-12s %-10s %s" % ("sigma", "# < sigma", "% of n"))
for sig in (1e-4, 4.0e-4, 8.0e-4, 1e-3, 3e-3, 1e-2):
    try:
        c = count_below(sig)
        print("%-12.3g %-10d %.1f%%" % (sig, c, 100.0*c/n), flush=True)
    except Exception as e:
        print("%-12.3g FAILED %s" % (sig, str(e)[:40]), flush=True)
print("\nself-check: sigma=4e-4 must be 0 (below lambda_min=4.181e-4),")
print("            sigma=8e-4 must be >=120 (lambda_120=7.912e-4)")
