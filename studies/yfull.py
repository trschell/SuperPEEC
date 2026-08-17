# SPDX-License-Identifier: MIT
"""Build the OVER-COMPLETE plaquette basis Y_full in SuperPEEC's own edge
index space, and wire it into a real LpR solve.

meshgraph_aux.getmesh emits only ~meshsize quads (already an independent
spanning set), so the full plaquette set must be enumerated here. Every
4-cycle of the node graph IS a plaquette on a cubic lattice.

adjmats() gives a ONE-SIDED node x node CSR whose data are 1-BASED
filament indices; (uf, vf) recovers each filament's endpoint nodes, and
the sign of a traversal u->v is +1 iff (uf,vf) == (u,v).
"""
import numpy as np, scipy.sparse as sp


def build_yfull(M):
    es, fs_, gs = (np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc))
    efg = es + fs_ + gs
    A = M.adjmats()
    adjs = (A + A.T).tocsr()
    coo = A.tocoo()
    fid = coo.data.astype(np.int64) - 1
    uf = np.full(efg, -1, np.int64); vf = np.full(efg, -1, np.int64)
    uf[fid] = coo.row; vf[fid] = coo.col
    # filament index for an ordered node pair
    fmap = {}
    for f, (a, b) in enumerate(zip(uf, vf)):
        if a >= 0:
            fmap[(a, b)] = f; fmap[(b, a)] = f
    ind, ptr = adjs.indices, adjs.indptr
    seen = set(); cols = []
    for u in range(adjs.shape[0]):
        nb = ind[ptr[u]:ptr[u+1]]
        nb = nb[nb > u]                       # canonicalise: u smallest
        for ai in range(len(nb)):
            v = nb[ai]
            nv = set(ind[ptr[v]:ptr[v+1]])
            for bi in range(ai+1, len(nb)):
                w = nb[bi]
                for x in (nv & set(ind[ptr[w]:ptr[w+1]])):
                    if x <= u:
                        continue
                    loop = [(u, v), (v, x), (x, w), (w, u)]
                    fs = tuple(sorted(fmap[e] for e in loop))
                    if len(set(fs)) != 4 or fs in seen:
                        continue
                    seen.add(fs)
                    cols.append([(fmap[e], 1.0 if (uf[fmap[e]], vf[fmap[e]])
                                  == e else -1.0) for e in loop])
    P = len(cols)
    rows = np.empty(4*P, np.int64); cs = np.empty(4*P, np.int64)
    vals = np.empty(4*P)
    for j, c in enumerate(cols):
        for t, (f, s) in enumerate(c):
            rows[4*j+t] = f; cs[4*j+t] = j; vals[4*j+t] = s
    return sp.csc_matrix((vals, (rows, cs)), shape=(efg, P))


if __name__ == '__main__':
    import sys, vhr, meshgraph as mg
    name = sys.argv[1] if len(sys.argv) > 1 else 'wire_len50.0u_dia10.0u.vhr'
    m = vhr.read_vhr('VoxHenry/Input_files/' + name)
    M = m.build_tree(); m.prepare(M, 1e10)
    es, fs_, gs = (np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc))
    efg, nn = es+fs_+gs, np.size(M.lv[0].struc)
    Ysel = sp.csc_matrix(mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn))
    Yf = build_yfull(M)
    print("%s: efg=%d nodes=%d" % (name, efg, nn))
    print("  selected basis : %d loops" % Ysel.shape[1])
    print("  OVER-COMPLETE  : %d plaquettes  (%.2fx)"
          % (Yf.shape[1], Yf.shape[1]/Ysel.shape[1]))
    Gf = (Yf.T@Yf).tocsr()
    print("  G_full: %dx%d nnz=%d  diag min/max=%g/%g"
          % (Gf.shape[0], Gf.shape[1], Gf.nnz,
             Gf.diagonal().min(), Gf.diagonal().max()))
    # sanity: every column is a genuine cycle -> B^T Y = 0 via incidence
    inc = sp.lil_matrix((nn, efg))
    A = M.adjmats().tocoo()
    for f, r, c in zip(A.data.astype(int)-1, A.row, A.col):
        inc[r, f] = -1.0; inc[c, f] = +1.0
    div = np.abs((inc.tocsr() @ Yf).toarray()).max()
    print("  divergence-free check  max|B Y_full| = %.2e" % div)
