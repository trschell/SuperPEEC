# SPDX-License-Identifier: MIT
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import meshgraph_aux
# from scipy.sparse.csgraph import breadth_first_tree


class CycleTree:
    def __init__(self, initnode, initsign):
        self.nodes = np.array([initnode], dtype=np.int64)
        self.frontierstart = 0
        self.frontierstop = 1
        self.pred = np.array([-1], dtype=np.int64)

    def append(self, newnode, newpred):
        self.nodes = np.append(self.nodes, newnode)
        self.pred = np.append(self.pred, newpred)

    def explore(self, mst):
        newnodecount = 0
        for frontierptr in range(self.frontierstart, self.frontierstop):
            frontiernode = self.nodes[frontierptr]
            prevnode = self.nodes[self.pred[frontierptr]]
            newfrontierlist = mst.indices[mst.indptr[frontiernode]:
                                          mst.indptr[frontiernode+1]]
            for newfrontiernode in newfrontierlist:
                if newfrontiernode != prevnode:
                    self.append(newfrontiernode, frontierptr)
                    newnodecount += 1
        self.frontierstart = self.frontierstop
        self.frontierstop += newnodecount

    def getpath(self, ptr):
        path = np.array([], dtype=np.int64)
        nextptr = ptr
        while True:
            path = np.append(path, self.nodes[nextptr])
            nextptr = self.pred[nextptr]
            if nextptr == -1:
                return path


def compare_nodes(tree1, tree2):
    for frontier1ptr in range(tree1.frontierstart, tree1.frontierstop):
        for frontier2ptr in range(tree2.frontierstart, tree2.frontierstop):
            if tree1.nodes[frontier1ptr] == tree2.nodes[frontier2ptr]:
                return (frontier1ptr, frontier2ptr)
    return (-1, -1)


def findcycle(mst, fromnode, tonode):
    tree1 = CycleTree(fromnode, 1)
    tree2 = CycleTree(tonode, -1)
    tree1.explore(mst)
    tree2.explore(mst)
    while True:
        tree1.explore(mst)
        (ptr1, ptr2) = compare_nodes(tree1, tree2)
        if ptr1 >= 0:
            break
        tree2.explore(mst)
        (ptr1, ptr2) = compare_nodes(tree1, tree2)
        if ptr1 >= 0:
            break
    path1 = tree1.getpath(ptr1)
    path2 = tree2.getpath(ptr2)
    return np.concatenate([path1[::-1], path2[1:]])


def get_mesh_incidence(adjmat, efgsize):
    nodesize = np.shape(adjmat)[0]
    adjmat1 = csr_matrix((np.ones_like(adjmat.data), adjmat.indices,
                          adjmat.indptr), shape=(nodesize, nodesize))
    mst = minimum_spanning_tree(np.abs(adjmat1))
    # mst = breadth_first_tree(np.abs(adjmat1), 0, directed=False)
    cyclelist = adjmat1 - mst
    mst -= mst.T
    ZZ1 = lil_matrix((efgsize, cyclelist.nnz), dtype=np.int8)
    cycle = 0
    for node in range(nodesize):
        for cycleind in range(cyclelist.indptr[node],
                              cyclelist.indptr[node+1]):
            tonode = cyclelist.indices[cycleind]
            nodes1 = findcycle(mst, node, tonode)
            nodes2 = nodes1.take(np.r_[1:nodes1.size, 0], 0)
            negind = np.argwhere(nodes1 > nodes2).T[0]
            nodes1[negind], nodes2[negind] = nodes2[negind], nodes1[negind]
            filpath = np.asarray(adjmat[nodes1, nodes2])[0] - 1
            vals = np.ones_like(filpath)
            vals[negind] = -1
            cyclevec = cycle*np.ones_like(filpath)
            ZZ1[filpath, cyclevec] = vals
            cycle += 1
    return ZZ1.tocsr()


def getmesh(adjmat, esize, efsize, efgsize, nodesize):
    white = 0
    grey = 1
    black = -1
    adjmat += adjmat.T
    adjind = adjmat.indices
    adjindptr = adjmat.indptr
    adjdat = adjmat.data
    meshsize = efgsize - nodesize + 1
    # Z = csc_matrix((efgsize, meshsize), dtype=np.int8)
    Zind = np.zeros((4*meshsize,), dtype=np.int64)
    Zindptr = np.r_[:4*meshsize:4, 4*meshsize]
    Zdat = np.zeros((4*meshsize,), dtype=np.int8)
    fil = np.zeros((4,), dtype=adjind.dtype)

    def signbranch(f1, f2):
        if f1 < esize and esize <= f2 < efsize:
            return -1
        if esize <= f1 < efsize and efsize <= f2 < efgsize:
            return -1
        if efsize <= f1 < efgsize and f2 < esize:
            return -1
        return 1

    def neighbors(node):
        return adjind[adjindptr[node]:adjindptr[node+1]]

    def store_loop(mesh, n1, n2, n3, n4):
        for i in range(adjindptr[n1], adjindptr[n1+1]):
            if adjind[i] == n2:
                fil[0] = adjdat[i] - 1
        for i in range(adjindptr[n2], adjindptr[n2+1]):
            if adjind[i] == n3:
                fil[1] = adjdat[i] - 1
        for i in range(adjindptr[n3], adjindptr[n3+1]):
            if adjind[i] == n4:
                fil[2] = adjdat[i] - 1
        for i in range(adjindptr[n4], adjindptr[n4+1]):
            if adjind[i] == n1:
                fil[3] = adjdat[i] - 1
        sign13 = np.sign(fil[2] - fil[0])
        sign24 = np.sign(fil[3] - fil[1])
        signf1 = sign13*signbranch(fil[0], fil[1])
        signf2 = sign24*signbranch(fil[1], fil[0])
        signs = np.array([signf1, signf2, -signf1, -signf2])
        sortorder = np.argsort(fil)
        Zind[4*mesh:4*(mesh+1)] = fil[sortorder]
        Zdat[4*mesh:4*(mesh+1)] = signs[sortorder]
        # Z[f1, mesh] = signf1
        # Z[f2, mesh] = signf2
        # Z[f3, mesh] = -signf1
        # Z[f4, mesh] = -signf2

    Q = -1*np.ones((nodesize+1,), dtype=np.int64)
    Q[0] = 0
    mesh = 0
    startq = 0
    stopq = 1
    color = np.zeros((nodesize), dtype=np.int8)
    parent = np.empty((nodesize), dtype=np.int64)
    while(startq != stopq):
        u = Q[startq]
        startq += 1
        for v in neighbors(u):
            if color[v] == white:
                color[v] = grey
                parent[v] = u
                Q[stopq] = v
                stopq += 1
            elif color[v] == grey:
                if parent[u] in neighbors(parent[v]):
                    store_loop(mesh, u, v, parent[v], parent[u])
                    mesh += 1
                else:
                    for vp in neighbors(v):
                        # print("vp =", vp)
                        if parent[u] in neighbors(vp) and vp != u:
                            store_loop(mesh, u, v, vp, parent[u])
                            mesh += 1
        color[u] = black
    return csc_matrix((Zdat, Zind, Zindptr), shape=(efgsize, meshsize))


def getmesh_fortran(adjmat, esize, efsize, efgsize, nodesize):
    """Divergence-free loop (mesh) basis, shape (efgsize, meshsize).

    Fast path: the Fortran plaquette enumerator (meshgraph_aux.getmesh),
    which spans the cycle space with elementary 4-filament square loops --
    exact on solid grids, where the returned basis is identical to the
    original build. The plaquette assumption FAILS on geometries whose
    cycle space needs macro loops no square spans (e.g. the
    plane+trace+via PCB structure: 849 valid quads for a cycle dimension
    of 857, plus zero/garbage columns whose duplicate indices crashed the
    downstream cholmod analyze). The output is therefore validated column
    by column -- exactly four distinct in-range filaments with +-1 signs
    closing a quad (every endpoint node appearing exactly twice) -- and
    if the valid count falls short of the cycle dimension
    efg - nn + numtrees, we fall back to the general MST
    fundamental-cycle basis (get_mesh_incidence): correct on arbitrary
    structures, slower to build (python BFS per non-tree edge).
    """
    adjs = (adjmat + adjmat.T).tocsr()
    numtrees = meshgraph_aux.counttrees(adjs.indices, adjs.indptr)
    meshsize = efgsize - nodesize + numtrees
    # numtrees sizes the Fortran output: the cycle rank is E-V+C and
    # the old single-component sizing overflowed the heap on
    # multi-component models (8 components on the DBC flagship --
    # floating signal traces + bottom metal; found 2026-08-11)
    Zdat, Zind = meshgraph_aux.getmesh(adjs.indices, adjs.indptr,
                                       adjs.data, esize, efsize,
                                       efgsize, numtrees)
    Zdat = np.asarray(Zdat)
    Zind = np.asarray(Zind)
    cols = Zind.reshape(-1, 4)
    dats = Zdat.reshape(-1, 4)
    # filament -> (node, node) endpoint map from the one-sided adjacency
    coo = adjmat.tocoo()
    ufil = np.full(efgsize, -1, dtype=np.int64)
    vfil = np.full(efgsize, -1, dtype=np.int64)
    ufil[coo.data.astype(np.int64) - 1] = coo.row
    vfil[coo.data.astype(np.int64) - 1] = coo.col
    ok = (np.abs(dats) == 1).all(axis=1)
    ok &= ((cols >= 0) & (cols < efgsize)).all(axis=1)
    sc = np.sort(cols, axis=1)
    ok &= (sc[:, 1:] != sc[:, :-1]).all(axis=1)
    ends = np.empty((cols.shape[0], 8), dtype=np.int64)
    ends[:, 0::2] = ufil[cols.clip(0, efgsize-1)]
    ends[:, 1::2] = vfil[cols.clip(0, efgsize-1)]
    ok &= (ends >= 0).all(axis=1)
    se = np.sort(ends, axis=1)
    ok &= (se[:, 0::2] == se[:, 1::2]).all(axis=1)
    if int(ok.sum()) >= meshsize:
        keep = np.nonzero(ok)[0][:meshsize]
        Zindptr = np.r_[:4*(meshsize+1):4]
        return csc_matrix((dats[keep].ravel(), cols[keep].ravel(), Zindptr),
                          shape=(efgsize, meshsize))
    print("getmesh_fortran: plaquette basis deficient (%d valid quads for "
          "cycle dimension %d) -- falling back to the MST fundamental-cycle "
          "basis" % (int(ok.sum()), meshsize))
    return get_mesh_incidence(adjmat, efgsize).tocsc()


def getmesh_full(adjmat, esize, efsize, efgsize, nodesize, maxq=None):
    """OVER-COMPLETE plaquette basis: EVERY 4-cycle, shape (efgsize, nq).

    Unlike :func:`getmesh_fortran`, which returns an INDEPENDENT spanning
    set (the BFS fundamental-cycle construction emits exactly one quad
    per non-tree edge), this keeps them all. The resulting Gram Y^T Y is
    SINGULAR -- its kernel is the cube boundaries, which are physically
    meaningless because currents differing by a cube boundary are the
    same current -- but singular-and-CONSISTENT is fine for Krylov, and
    it is far better conditioned on its range.

    Measured on the 24k-voxel bar (studies/oc_solve.py): identical R and
    L to 7 significant figures, the same matvec count at 2.5-10 GHz, and
    12 MB of AMG hierarchy against 121 MB of Cholesky fill. See
    studies/overcomplete.py for the geometry sweep.

    `maxq` bounds the output; on overflow the Fortran returns nq < 0 and
    this retries with double the bound.
    """
    adjs = (adjmat + adjmat.T).tocsr()
    if maxq is None:
        maxq = 2*efgsize
    while True:
        nq, Zdat, Zind = meshgraph_aux.getmeshfull(
            adjs.indices, adjs.indptr, adjs.data, esize, efsize, efgsize,
            maxq)
        if nq >= 0:
            break
        maxq *= 2
    cols = np.asarray(Zind)[:4*nq].reshape(-1, 4)
    dats = np.asarray(Zdat)[:4*nq].reshape(-1, 4)
    indptr = np.r_[:4*(nq+1):4]
    return csc_matrix((dats.ravel().astype(np.float64), cols.ravel(),
                       indptr), shape=(efgsize, nq))
