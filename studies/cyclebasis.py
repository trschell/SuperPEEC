# SPDX-License-Identifier: MIT
"""What STRUCTURE in the cycle basis creates the 361-mode near-null
cluster?  Y's columns all have exactly 4 nonzeros, so this is already a
MINIMUM-LENGTH cycle basis -- the conditioning must come from WHICH
plaquettes are selected, not from cycle length.

Classify each loop by plaquette ORIENTATION from its edge membership
(edges are ordered e|f|g = x|y|z-directed):
    xy-plaquette : 2 e-edges + 2 f-edges
    xz-plaquette : 2 e + 2 g
    yz-plaquette : 2 f + 2 g
then ask where the near-null eigenvectors put their mass.
361 = 19^2 and the bar is 60x20x20, whose yz-plaquettes number 19x19
per x-slice -- so a cross-sectional family is the obvious suspect.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import vhr, meshgraph as mg
from sksparse import cholmod

m = vhr.read_vhr('VoxHenry/Input_files/'
                 'straight_cond1_len30.0u_wid10.0u_dist20.0u.vhr')
M = m.build_tree(); m.prepare(M, 1e10)
es, fs_, gs = np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc)
efg, nn = es+fs_+gs, np.size(M.lv[0].struc)
Y = sp.csc_matrix(mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn))
Y.data = np.float64(Y.data)
n = Y.shape[1]
print("edges: e=%d f=%d g=%d (efg=%d)   loops=%d" % (es, fs_, gs, efg, n))

# classify each loop column by how its 4 edges split across e|f|g
ind, ptr = Y.indices, Y.indptr
ne = np.zeros(n, int); nf = np.zeros(n, int); ng = np.zeros(n, int)
for j in range(n):
    r = ind[ptr[j]:ptr[j+1]]
    ne[j] = int((r < es).sum())
    nf[j] = int(((r >= es) & (r < es+fs_)).sum())
    ng[j] = int((r >= es+fs_).sum())
kind = np.where((ne == 2) & (nf == 2), 0,
                np.where((ne == 2) & (ng == 2), 1,
                         np.where((nf == 2) & (ng == 2), 2, 3)))
names = ['xy (2e+2f)', 'xz (2e+2g)', 'yz (2f+2g)', 'other']
for k in range(4):
    c = int((kind == k).sum())
    if c:
        print("  %-12s %6d loops (%.1f%%)" % (names[k], c, 100.0*c/n))

G = (Y.T@Y).tocsr().astype(np.float64)
F = cholmod.cholesky(G.tocsc(), ordering_method='metis')
OP = sla.LinearOperator((n, n), matvec=lambda v: F(v))
w, V = sla.eigsh(G, k=400, sigma=0.0, which='LM', OPinv=OP, tol=1e-8)
o = np.argsort(w); w, V = w[o], V[:, o]
ncl = int((w < 8e-4).sum())
print("\ncluster = %d modes below 8e-4; next eigenvalue %.3e" % (ncl, w[ncl]))

print("\nwhere the eigenvector MASS sits, by plaquette orientation:")
print("%-22s %-9s %-9s %-9s" % ("mode set", names[0], names[1], names[2]))
for lo, hi, lab in ((0, ncl, "cluster (0..%d)" % ncl),
                    (ncl, 400, "above gap (%d..400)" % ncl)):
    Vs = V[:, lo:hi]
    mass = (Vs**2)
    frac = [mass[kind == k].sum()/mass.sum() for k in range(3)]
    print("%-22s %-9.3f %-9.3f %-9.3f" % (lab, frac[0], frac[1], frac[2]))
share = [float((kind == k).sum())/n for k in range(3)]
print("%-22s %-9.3f %-9.3f %-9.3f  <- share of loops (baseline)"
      % ("(uniform would be)", share[0], share[1], share[2]))
