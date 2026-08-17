# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Windowed reluctance extraction for the LpPR inductive preconditioner.

The reluctance K = Lp^{-1} is localized (shielding), so a sparse K~ is an
excellent, frequency-independent, FIXED preconditioner for the branch impedance
Z = R + jw*Lp: preconditioning by K~ turns the dense jw*Lp into ~jw*I, giving a
diagonally dominant system solvable by standard GMRES. Dense prototyping settled
the recipe (setup1):

  * per orientation (e/f/g are block-diagonal in Lp -- perpendicular filaments
    do not couple);
  * EXTRACT from a ~50-neighbour window (invert the local Lp block for accurate
    reluctance values);
  * RETAIN the ~27 strongest entries per row (the 3x3x3 near-neighbour stencil;
    a hard floor -- fewer collapses);
  * SYMMETRIZE.

The local Lp blocks come from the closed-form genL3D kernel (translation-
invariant, so a per-orientation offset table), never the full dense Lp -- this
is what makes the extraction scalable (O(N * window^3)). A dense block-getter is
provided for validation against the FMM-probed Lp.
"""
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from greens import genL3D


def filament_geometry(M):
    """Grid positions (cell units) and orientation (0=e,1=f,2=g) of every
    filament, in the e|f|g concatenated ordering the solver uses.

    At a single level the leaf ``idx`` are global grid indices. At
    ``numlevels > 1`` they are PER-BOX indices grouped by ``idx0``, so the
    global position is the local decode plus the leaf box offset
    (``lv[0].xidx/yidx/zidx`` times the box size) -- same-orientation
    grid-coordinate distances stay integers on the cell lattice either way,
    which is all the windowing and the genL3D offset tables need."""
    es = M.e.struc.size; fs = M.f.struc.size; gs = M.g.struc.size
    efg = es + fs + gs
    pos = np.zeros((efg, 3), dtype=np.int64)
    orient = np.zeros(efg, dtype=np.int64)
    off = 0
    for oi, (leaf, cnt) in enumerate([(M.e, es), (M.f, fs), (M.g, gs)]):
        n = leaf.n.astype(int); idx = leaf.idx
        pos[off:off+cnt, 0] = idx // (n[1]*n[2])
        pos[off:off+cnt, 1] = (idx // n[2]) % n[1]
        pos[off:off+cnt, 2] = idx % n[2]
        if M.numlevels > 1:
            bn = M.lv[0].n.astype(int)
            for g in range(np.size(leaf.idx0) - 1):
                sl = np.s_[off+leaf.idx0[g]:off+leaf.idx0[g+1]]
                pos[sl, 0] += M.lv[0].xidx[g]*bn[0]
                pos[sl, 1] += M.lv[0].yidx[g]*bn[1]
                pos[sl, 2] += M.lv[0].zidx[g]*bn[2]
        orient[off:off+cnt] = oi
        off += cnt
    return pos, orient


def _windows(pos, orient, window):
    """For each filament, the `window` nearest SAME-orientation filaments plus
    itself. Ties in distance are broken by the RELATIVE offset (lexicographic),
    which depends only on geometry, not absolute index -- so filaments in the
    same neighbourhood configuration select the same window *shape*, enabling
    translation-invariant factor caching. Returns an object array of index
    arrays.

    Implementation: a per-orientation cKDTree. A bare k-nearest query would
    break distance ties ARBITRARILY (by tree topology), destroying the
    translation invariance the caching depends on -- so instead the k-th
    neighbour distance sets a cutoff radius, a ball query gathers the ENTIRE
    tied shell (radius + eps; lattice distances are sqrt-integers in cell
    units, so distinct shells differ by >> 1e-6), and the same
    (distance, relative-offset) lexsort as the original O(N^2) scan picks the
    window from that small candidate set. The selection is exactly identical
    to the original; the cost falls from O(N * N_same) to
    O(N log N + N * shell * log shell) -- measured 77.6 s -> seconds for the
    23^3 cube extraction, where the scan had become the dominant setup cost
    (profile_fmm_crossover.py)."""
    W = np.empty(pos.shape[0], dtype=object)
    for o in range(3):
        s = np.nonzero(orient == o)[0]
        if s.size == 0:
            continue
        P = pos[s].astype(np.float64)
        k = min(window + 1, s.size)
        tree = cKDTree(P)
        dk, _ = tree.query(P, k=k, workers=-1)
        dk = dk.reshape(s.size, k)
        cand = tree.query_ball_point(P, dk[:, -1] + 1e-6, workers=-1)
        for ii in range(s.size):
            c = np.asarray(cand[ii])
            off = pos[s[c]] - pos[s[ii]]
            d = np.einsum('ij,ij->i', off, off)       # squared distance
            order = np.lexsort((off[:, 2], off[:, 1], off[:, 0], d))
            W[s[ii]] = s[c[order[:k]]]
    return W


def _center_rows_batched(blocks, ilocs, use_gpu=None, chunk=4096):
    """Centre row of ``inv(block)`` for a LIST of small dense blocks,
    solved in BATCHES instead of a Python loop of per-window
    factorizations (the loop overhead, not the O(k^3) flops, dominated
    the extraction: ~100 us of Python per window against ~30 us of
    LAPACK at k ~ 31, across up to hundreds of thousands of windows).

    Ragged windows are padded to the batch's max size with an identity
    tail -- block-diagonal padding, so the centre row of the padded
    inverse is exactly the original centre row plus zeros. Solved by
    batched LU (numpy gesv): the blocks are SPD in exact arithmetic
    (principal submatrices of Gram matrices) but LU needs no such
    promise, which also absorbs the old per-window non-PD fallback.

    Backend: numpy by default; CuPy under SPPEEC_GPU=1 (opt-in, same
    convention as the rest of the GPU track, permanent CPU fallback) --
    batched small dense solves are the textbook GPU workload, and the
    windows of FILIGREE geometry (vias/antipads breaking the
    translation-invariant shape cache) are exactly where the batch is
    large. The anchors are unaffected structurally: main.py's anchor
    path is LpR and never calls the extractors.
    """
    import os as _os
    if use_gpu is None:
        use_gpu = _os.environ.get('SPPEEC_GPU', '0') == '1'
    xp = np
    if use_gpu:
        try:
            import cupy as _cp
            _cp.zeros(1)          # fail early if no device
            xp = _cp
        except Exception:
            xp = np               # silent-ish fallback: numpy batch
    n = len(blocks)
    out = [None]*n
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        ks = [b.shape[0] for b in blocks[lo:hi]]
        kmax = max(ks)
        A = np.zeros((hi - lo, kmax, kmax))
        A[:, np.arange(kmax), np.arange(kmax)] = 1.0
        rhs = np.zeros((hi - lo, kmax, 1))
        for j, b in enumerate(blocks[lo:hi]):
            k = ks[j]
            A[j, :k, :k] = b
            rhs[j, ilocs[lo + j], 0] = 1.0
        Ax = xp.asarray(A)
        bx = xp.asarray(rhs)
        sol = xp.linalg.solve(Ax, bx)[..., 0]
        sol = sol.get() if xp is not np else sol
        for j in range(hi - lo):
            out[lo + j] = np.ascontiguousarray(sol[j, :ks[j]])
    return out


def dense_block_getter(Lp):
    """Ground-truth block-getter: slice a full dense (FMM-probed) Lp."""
    def get(W):
        return np.real(Lp[np.ix_(W, W)])
    return get


def kernel_block_getter(M, pos, orient, maxsep):
    """Scalable block-getter: local Lp blocks from the closed-form genL3D
    offset tables (one per orientation), matching leaf_induct.p2pinit's
    conventions (orientation-specific args + transpose, /lscale)."""
    l = M.e.l
    Ns = int(maxsep) + 2
    ie = genL3D(l[0], l[1], l[2], Ns, Ns, Ns) / getattr(M.e, 'lscale', 1.0)
    ff = np.transpose(genL3D(l[1], l[0], l[2], Ns, Ns, Ns), (1, 0, 2)) \
        / getattr(M.f, 'lscale', 1.0)
    gg = np.transpose(genL3D(l[0], l[2], l[1], Ns, Ns, Ns), (0, 2, 1)) \
        / getattr(M.g, 'lscale', 1.0)
    tables = [ie, ff, gg]

    def get(W):
        table = tables[orient[W[0]]]
        P = pos[W]
        d = np.abs(P[:, None, :] - P[None, :, :])
        return table[d[..., 0], d[..., 1], d[..., 2]]
    return get


def extract_reluctance(M, Lp_dense=None, window=50, retain=27, verbose=False):
    """Build the sparse, symmetric reluctance preconditioner K~ ~ Lp^{-1}.

    Uses translation-invariant factor caching: on a regular grid the local block
    Lp[W,W] depends only on the window's SHAPE (its set of relative offsets), so
    filaments sharing a shape share one Cholesky factorization and one reluctance
    stencil. Interior filaments all share a single shape, so the O(N)
    factorizations collapse to roughly O(boundary). The result is identical to
    factoring every window independently.

    Parameters
    ----------
    M : Tree
        The PEEC tree (any number of levels; positions are decoded to the
        global cell lattice by :func:`filament_geometry`).
    Lp_dense : ndarray or None
        If given, local blocks are sliced from this dense Lp (validation /
        ground truth). If None, blocks come from the genL3D kernel (scalable).
    window : int
        Extraction window: neighbours whose local Lp block is inverted.
    retain : int
        Entries kept per row (the significant near-neighbour stencil).
    verbose : bool
        Report the number of distinct window shapes (factorizations).

    Returns
    -------
    scipy.sparse.csr_matrix
        Real, symmetric K~ of shape (efg, efg).
    """
    pos, orient = filament_geometry(M)
    efg = pos.shape[0]
    W = _windows(pos, orient, window)
    # the local block Lp[W,W] needs mutual inductances between ALL pairs in a
    # window, so the table must cover each window's full bounding-box extent
    # (pairwise separation), not just the centre-to-edge distance.
    maxsep = max(int((pos[Wi].max(0) - pos[Wi].min(0)).max())
                 for Wi in W)
    get_block = (dense_block_getter(Lp_dense) if Lp_dense is not None
                 else kernel_block_getter(M, pos, orient, maxsep))
    # position -> global filament index, for scattering a shape's stencil onto
    # every filament that shares it
    posmap = {(int(orient[j]), int(pos[j, 0]), int(pos[j, 1]), int(pos[j, 2])): j
              for j in range(efg)}
    offs = [pos[W[i]] - pos[i] for i in range(efg)]      # relative offsets
    # group filaments by (orientation, canonical relative-offset set)
    groups = defaultdict(list)
    for i in range(efg):
        key = (int(orient[i]), tuple(sorted(map(tuple, offs[i].tolist()))))
        groups[key].append(i)

    # batched: assemble one representative block per shape group, solve
    # all centre rows in one batched pass (see _center_rows_batched),
    # then scatter -- identical math, no per-group Python factorization
    glist = list(groups.values())
    blocks = []
    ilocs = []
    for members in glist:
        r = members[0]                                   # representative
        Wr = W[r]
        blocks.append(np.asarray(get_block(Wr), dtype=np.float64))
        ilocs.append(int(np.nonzero(Wr == r)[0][0]))
    rowsol = _center_rows_batched(blocks, ilocs)
    rows = []; cols = []; data = []
    for members, row in zip(glist, rowsol):
        r = members[0]
        keep = np.argsort(-np.abs(row))[:retain+1]       # top entries (+ diag)
        kept_off = [tuple(int(c) for c in offs[r][k]) for k in keep]
        kept_val = row[keep]
        for i in members:                                # scatter to the group
            oi = int(orient[i]); p = pos[i]
            for off, val in zip(kept_off, kept_val):
                gj = posmap[(oi, int(p[0]+off[0]), int(p[1]+off[1]),
                             int(p[2]+off[2]))]
                rows.append(i); cols.append(gj); data.append(val)
    if verbose:
        print("reluctance: %d filaments -> %d distinct window shapes "
              "(factorizations, %.1fx fewer)"
              % (efg, len(groups), efg/len(groups)))
    K = coo_matrix((data, (rows, cols)), shape=(efg, efg)).tocsr()
    return (0.5*(K + K.T)).tocsr()                     # symmetrize


def node_geometry(M):
    """Global cell-lattice positions of every lv[0] node, mirroring the
    validated decode of SystemMat._mnaprecassembly (single level: global
    node-grid indices over M.ntotal; multilevel: per-box indices plus the
    leaf-box offset)."""
    lv0 = M.lv[0]
    nn = lv0.struc.size
    pos = np.zeros((nn, 3), dtype=np.int64)
    idx = lv0.idx
    if M.numlevels == 1:
        dims = M.ntotal.astype(int)
        pos[:, 0] = idx // (dims[1]*dims[2])
        pos[:, 1] = (idx // dims[2]) % dims[1]
        pos[:, 2] = idx % dims[2]
    else:
        n0 = lv0.n.astype(int)
        pos[:, 0] = idx // (n0[1]*n0[2])
        pos[:, 1] = (idx // n0[2]) % n0[1]
        pos[:, 2] = idx % n0[2]
        for g in range(np.size(lv0.idx0) - 1):
            sl = np.s_[lv0.idx0[g]:lv0.idx0[g+1]]
            pos[sl, 0] += lv0.xidx[g]*n0[0]
            pos[sl, 1] += lv0.yidx[g]*n0[1]
            pos[sl, 2] += lv0.zidx[g]*n0[2]
    return pos


def _rect_V(ctr, ax, obs, half_u, half_v, taxes):
    """Exact potential of a uniformly charged axis-aligned RECTANGLE
    (total charge 1) at points ``obs``. Ported from the BEM oracle
    (studies/bem_dielectric.py; principal-arctan branch -- atan2 puts
    below-plane observations on the wrong branch). Generalized to
    rectangular panels (anisotropic pitch): half-sides ``half_u``,
    ``half_v`` along tangent axes ``taxes``."""
    u0 = obs[:, taxes[0]] - ctr[taxes[0]]
    v0 = obs[:, taxes[1]] - ctr[taxes[1]]
    w = obs[:, ax] - ctr[ax]
    V = np.zeros(len(obs))
    for su in (+1.0, -1.0):
        du = u0 + su*half_u
        for sv in (+1.0, -1.0):
            dv = v0 + sv*half_v
            r = np.sqrt(du*du + dv*dv + w*w)
            s = su*sv
            with np.errstate(divide='ignore', invalid='ignore'):
                lv = np.log(dv + r)
                lu = np.log(du + r)
                at = np.nan_to_num(np.arctan(du*dv/(w*r)))
            V += s*(du*lv + dv*lu - w*at)
    eps0 = 8.8541878128e-12
    return V/(4*np.pi*eps0*(4.0*half_u*half_v))


def _leaf_cells(M, leaf):
    """Global lattice index of every element of ``leaf``, in leaf
    order (the voxmodel.filament_cells decode, local to avoid an
    import cycle)."""
    n = np.asarray(leaf.n, dtype=int)
    lv0 = M.lv[0]
    out = np.zeros((int(np.size(leaf.idx)), 3), dtype=int)
    for g in range(np.size(leaf.idx0) - 1):
        if leaf.idx0[g+1] <= leaf.idx0[g]:
            continue
        sl = np.s_[leaf.idx0[g]:leaf.idx0[g+1]]
        idx = np.asarray(leaf.idx[sl])
        c = np.stack([idx//(n[1]*n[2]), (idx//n[2]) % n[1],
                      idx % n[2]], 1)
        c[:, 0] += lv0.xidx[g]*lv0.n[0]
        c[:, 1] += lv0.yidx[g]*lv0.n[1]
        c[:, 2] += lv0.zidx[g]*lv0.n[2]
        out[sl] = c
    return out


def node_panel_geometry(M):
    """Per-node panel lists for the KERNEL-DIRECT ccap blocks: for
    every node, the centres (physical), normal axes and projection
    weights of its panels, from the node2px/py/pz maps -- the same
    condensation convention the n2n assembly uses (equal split,
    count-normalized). Returns dict node -> (centres (k,3), axes (k,),
    weight scalar 1/k)."""
    l = np.asarray(M.e.l, dtype=float)
    pans = {}
    for leaf, ax, name in ((M.px, 0, 'px'), (M.py, 1, 'py'),
                           (M.pz, 2, 'pz')):
        if int(np.size(leaf.idx)) == 0:
            continue
        cells = _leaf_cells(M, leaf)
        ctr = (cells + 0.5)*l[None, :]
        ctr[:, ax] = cells[:, ax]*l[ax]
        n2p = getattr(M, 'node2' + name).tocsr()
        for p in range(n2p.shape[0]):
            lo, hi = n2p.indptr[p], n2p.indptr[p+1]
            if hi == lo:
                continue
            n = int(n2p.indices[lo])
            pans.setdefault(n, []).append((ctr[p], ax))
    out = {}
    for n, pl in pans.items():
        out[n] = (np.array([c for c, _ in pl]),
                  np.array([a for _, a in pl], dtype=int),
                  1.0/len(pl))
    return out


def kernel_ccap_block_getter(M):
    """Block-getter for :func:`extract_ccap` computing exact
    node-condensed potential blocks DIRECTLY from panel kernels -- no
    stored n2n needed (the point: the multilevel near-field n2n costs
    ~27*leaf^3 entries per node, 33 GB at a 320^2 board, and with
    ``fftnear`` the operator never reads it either). Blocks are exact
    Gram condensations, PD by construction. Because kernel blocks
    depend ONLY on the window's relative panel geometry (unlike
    n2n-sliced blocks, whose entries embed each node's global panel
    environment -- see the no-sharing NOTE in extract_ccap), a shape
    cache IS valid here and plate interiors collapse to few shapes.

    Values differ from the n2n-sliced blocks at the collocation-vs-
    Galerkin level (~1e-2 relative on near entries); W is an
    approximate inverse either way and the true-residual postcheck
    guards the solve, but expect retain tie-flips and slightly
    different iteration counts, not different answers.
    """
    l = np.asarray(M.e.l, dtype=float)
    halves = {0: (0.5*l[1], 0.5*l[2], (1, 2)),
              1: (0.5*l[0], 0.5*l[2], (0, 2)),
              2: (0.5*l[0], 0.5*l[1], (0, 1))}
    pans = node_panel_geometry(M)
    ext = np.asarray(M.external)
    cache = {}

    def get(Wi, key=None):
        if key is not None and key in cache:
            return cache[key]
        gl = [pans[int(ext[w])] for w in Wi]
        flat_c = np.concatenate([c for c, a, wt in gl])
        flat_a = np.concatenate([a for c, a, wt in gl])
        owner = np.concatenate([np.full(len(a), j)
                                for j, (c, a, wt) in enumerate(gl)])
        wts = np.concatenate([np.full(len(a), wt)
                              for c, a, wt in gl])
        npn = flat_c.shape[0]
        Pp = np.empty((npn, npn))
        for jsrc in range(npn):
            hu, hv, ta = halves[int(flat_a[jsrc])]
            Pp[:, jsrc] = _rect_V(flat_c[jsrc], int(flat_a[jsrc]),
                                  flat_c, hu, hv, ta)
        nn = len(gl)
        Wc = np.zeros((npn, nn))
        Wc[np.arange(npn), owner] = wts
        B = Wc.T.dot(Pp).dot(Wc)
        if key is not None:
            cache[key] = B
        return B
    return get


def extract_ccap(M, Pext_dense=None, window=50, retain=13, verbose=False):
    """Sparse banded C_cap ~ P_ext^{-1} on the EXTERNAL nodes -- the
    electrostatic twin of :func:`extract_reluctance`: charge screening
    localizes the Maxwell capacitance matrix just as magnetic shielding
    localizes the reluctance, so a windowed inversion captures the
    off-diagonal capacitive coupling that the diagonal C_cap discards (and
    that the nodal Schur must balance near resonance) at O(n_ext) storage.

    Same recipe as the reluctance: for each external node, gather the
    `window` nearest external nodes (cKDTree tied-shell + relative-offset
    lexsort -- deterministic, and nodes sharing a window SHAPE share one
    Cholesky; plate interiors collapse to a few shapes), invert the local
    P_ext block, keep the `retain` strongest centre-row entries (+ the
    diagonal), symmetrize. Because the Maxwell matrix has positive diagonal
    and non-positive off-diagonals, truncation only strengthens diagonal
    dominance: the banded C_cap stays positive definite, so -jw*C_cap still
    breaks the nodal-Laplacian null space and the no-pinning argument for
    S~ / S_d survives.

    Local blocks come from `Pext_dense` ((n_ext, n_ext), real ground truth)
    when given, else from ``M.n2n`` restricted to the external nodes: the
    full dense block at single level, the sparse NEAR-FIELD n2n at
    multilevel/circulant -- which is consistent by construction, since the
    W rescale (``n2nchol``) inverts that same near-field operator.

    Returns real symmetric csr of shape (n_ext, n_ext), in M.external order.
    """
    from scipy.sparse import issparse
    ext = np.asarray(M.external)
    ne = ext.size
    pe = node_geometry(M)[ext]
    # windows: nearest external nodes, tied-shell gathered, offset-lexsorted
    W = np.empty(ne, dtype=object)
    P = pe.astype(np.float64)
    k = min(window + 1, ne)
    tree = cKDTree(P)
    dk, _ = tree.query(P, k=k, workers=-1)
    dk = dk.reshape(ne, k)
    cand = tree.query_ball_point(P, dk[:, -1] + 1e-6, workers=-1)
    for ii in range(ne):
        c = np.asarray(cand[ii])
        off = pe[c] - pe[ii]
        d = np.einsum('ij,ij->i', off, off)
        order = np.lexsort((off[:, 2], off[:, 1], off[:, 0], d))
        W[ii] = c[order[:k]]
    kernel_blocks = Pext_dense is None and getattr(M, 'n2n', None) is None
    if Pext_dense is not None:
        def get_block(Wi, key=None):
            return np.real(np.asarray(Pext_dense)[np.ix_(Wi, Wi)])
        keyfn = None
    elif kernel_blocks:
        # KERNEL-DIRECT blocks (keep_n2n=False trees): exact panel
        # condensation, no stored n2n. Shape caching is VALID here
        # (kernel entries depend only on relative panel geometry) --
        # the key is the window's relative node offsets plus each
        # node's local panel signature.
        get_block = kernel_ccap_block_getter(M)
        l = np.asarray(M.e.l, dtype=float)
        pans = node_panel_geometry(M)
        sig = {}
        for jj in range(ne):
            n = int(ext[jj])
            c, a, wt = pans[n]
            nc = (pe[jj] + 0.5)*l
            rel = np.round((c - nc[None, :])/(0.5*l[None, :])).astype(int)
            sig[jj] = tuple(sorted((int(ax),) + tuple(int(v) for v in r)
                                   for ax, r in zip(a, rel)))

        def keyfn(ii, Wi):
            rel = pe[Wi] - pe[ii]
            return (tuple(map(tuple, rel.tolist())),
                    tuple(sig[int(w)] for w in Wi))
    else:
        A = M.n2n
        gext = ext

        def get_block(Wi, key=None):
            g = gext[Wi]
            if issparse(A):
                return np.real(A[np.ix_(g, g)].toarray())
            return np.real(np.asarray(A)[np.ix_(g, g)])
        keyfn = None
    # NOTE: with n2n-sliced blocks, windows are factored PER NODE, with no
    # shape-based sharing. The reluctance's caching rests on a theorem --
    # Lp entries depend only on the filament offset -- that does NOT hold
    # for n2n: its entries depend on each node's PANEL environment, so
    # identical window shapes need not give identical blocks (false-sharing
    # risk on edges/corners). KERNEL blocks restore the theorem (see
    # kernel_ccap_block_getter) and use the shape cache via `keyfn`.
    # STREAMED (2026-08-08): assemble, batch-solve and sparsify in
    # CHUNKS instead of collecting every window's dense block first --
    # the all-at-once list held ~2.3 GB of transient blocks at a
    # 320^2 board (303k windows) that were discarded minutes later
    # but inflated the setup peak forever. The kernel-path shape
    # cache lives in the getter closure and persists across chunks,
    # so plate-interior dedup is unaffected.
    rows = []; cols = []; data = []
    CH = 4096
    for lo in range(0, ne, CH):
        hi = min(lo + CH, ne)
        blocks = []
        ilocs = []
        for i in range(lo, hi):
            Wi = W[i]
            key = keyfn(i, Wi) if keyfn is not None else None
            blocks.append(np.asarray(get_block(Wi, key), dtype=np.float64))
            ilocs.append(int(np.nonzero(Wi == i)[0][0]))
        rowsol = _center_rows_batched(blocks, ilocs)
        for j, i in enumerate(range(lo, hi)):
            Wi = W[i]
            row = rowsol[j]
            keep = np.argsort(-np.abs(row))[:retain+1]
            rows.extend([i]*keep.size)
            cols.extend(Wi[keep].tolist())
            data.extend(row[keep].tolist())
        del blocks, rowsol
    if verbose:
        C0 = coo_matrix((data, (rows, cols)), shape=(ne, ne)).tocsr()
        print("ccap: %d external nodes, window %d retain %d -> %.1f nnz/row"
              % (ne, window, retain, C0.nnz/max(ne, 1)))
    C = coo_matrix((data, (rows, cols)), shape=(ne, ne)).tocsr()
    return (0.5*(C + C.T)).tocsr()
