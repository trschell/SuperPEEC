# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Tree: the multilevel FMM tree assembling all level classes.

The :class:`Tree` owns the whole PEEC/FMM state: the level hierarchy
(``lv[0]`` leaf .. ``lv[-1]`` top, from :mod:`levels`), the three inductive
filament leaves (``e``/``f``/``g``, :class:`leaf_induct.LeafInduct`), the
three capacitive panel leaves (``px``/``py``/``pz``,
:class:`leaf_poten.LeafPoten`), and the node set (``lv[0]`` doubles as the
circuit-node leaf). Its methods fall into four groups:

* **construction** -- ``__init__`` scans the voxel occupancy `fullstruc` and
  derives the compressed per-leaf index sets; :meth:`node2panel` builds the
  node<->panel projections (capacitive path only).
* **matvec traversals** -- :meth:`traverseRL` (inductive ``R + jwLp``, the
  working LpR path) and :meth:`traverseP`/
  :meth:`traverseP3` (capacitive ``P``, unfinished LpPR path).
* **incidence operators** -- :meth:`connectA`/:meth:`connectAT` and the
  :meth:`node2e`-family, the discrete divergence/gradient (KCL) maps between
  node and filament quantities, scaled by ``beta``.
* **preconditioning (dormant)** -- the tree-level SPAI/RDF machinery
  (:meth:`spaiinit`, :meth:`spaiapply`/:meth:`spaiapply2`,
  :meth:`RDFinit`/:meth:`RDFapply*`), implemented but unused by the default
  solver (main.py preconditions with a sparse Cholesky instead).

Split out of multipole.py.
"""
from multipole_common import *  # noqa: F401,F403  shared imports/guards
from special import *  # noqa: F401,F403
from greens import *  # noqa: F401,F403
from levels import Level, LeafLevel, MidLevel, TopLevel
from leaf_induct import LeafInduct
from leaf_poten import LeafPoten
import stencils as st
import warnings
from scipy.sparse import coo_matrix


class Tree:
    """The multilevel FMM/PEEC tree: levels, leaves, and traversals.

    See the module docstring for the method groups. Key attributes:

    Attributes
    ----------
    lv : ndarray of object
        The level hierarchy: ``lv[0]`` is the node-set :class:`LeafLevel`,
        ``lv[1..numlevels-2]`` are :class:`MidLevel`, ``lv[numlevels-1]`` is
        the :class:`TopLevel` (multi-level only).
    e, f, g : LeafInduct
        Inductive filament leaves (y-, x-, z-directed currents).
    px, py, pz : LeafPoten
        Capacitive panel leaves (x-, y-, z-normal panels); populated but
        inert when ``capacitive=False``.
    node2px, node2py, node2pz : sparse matrix
        Node-to-panel charge projections (capacitive path,
        :meth:`node2panel`).
    n2n : sparse/dense matrix
        Assembled node-to-node coefficient-of-potential operator
        (capacitive path).
    jomega, alpha : complex or None
        Angular-frequency factor ``j*omega`` and continuity penalty weight;
        set by the driver / :meth:`RDFinit` before traversals.
    """
    def __init__(self, fullstruc, nleaf, ltop, numlevels, lscale, nmax,
                 capacitive=True, circulant=False, fftnear=False,
                 eps_r=1.0, keep_n2n=True):
        """Build the tree and all leaf index sets from the voxel occupancy.

        Derives seven compressed element sets from `fullstruc` (nonzero =
        conductor voxel): the three filament orientations (``e``/``f``/``g``),
        the circuit nodes (stored on ``lv[0]``), and the three panel
        orientations (``px``/``py``/``pz``, on half-step staggered grids;
        panels exist only on the conductor *surface*, hence the
        occupancy-difference stencils). For ``numlevels > 1`` a single
        depth-first scan assigns every occupied leaf box its place in the
        level hierarchy (filling each level's ``idx``/``idx0``/``x/y/zidx``),
        after which the neighbour lists and mid-level M2L data are built; for
        ``numlevels == 1`` the whole domain is one group and the seven sets
        are extracted by direct max/difference stencils over the full grid.

        Parameters
        ----------
        fullstruc : ndarray
            3-D voxel occupancy grid of the conductor geometry.
        nleaf : ndarray of int
            Filaments per leaf box along x, y, z (multi-level only).
        ltop : ndarray of float
            Physical size of the top-level box (metres).
        numlevels : int
            Number of tree levels (1 = single-level: leaf P2P + top FFT M2L).
        lscale : float
            Inductance scale divisor passed to :class:`LeafInduct`.
        nmax : int or None
            Multipole expansion order (forced to ``None`` for single-level).
        capacitive : bool, optional
            When ``False`` (the LpR case), construction stops after the
            inductive leaves and level hierarchy are complete, skipping
            :meth:`node2panel`, the ``p2pinit3`` panel couplings and the
            ``n2n``/``n2nchol`` assembly -- those are consumed only by the
            LpPR ``traverseP*`` paths. The capacitive assembly works at any
            ``numlevels`` (validated by ``validate_p2pinit3.py``). Default
            ``True`` preserves the original behaviour.
        """
        [ntx, nty, ntz] = np.shape(fullstruc)
        self.numlevels = numlevels
        self.jomega = None
        nt = np.array((ntx, nty, ntz))
        # `nt` is rebound to a scalar later in the multilevel branch, so
        # keep an immutable copy of the CELL dimensions for the window
        # clipping in the depth-first scan.
        ntcells = np.array((ntx, nty, ntz))
        # The node grid: cell centres, one node per cell. M.ntotal is
        # consumed downstream as the node-grid dimensions.
        ntotal = np.array((ntx, nty, ntz))
        self.ntotal = ntotal
        lv = np.ndarray((numlevels,), dtype=object)
        # Box-partition basis. Always nt+1, independent of where the
        # nodes sit: it must exceed the cell count so the padded lattice
        # has at least one empty cell beyond the geometry, which is what
        # lets the topmost panel face be represented (a panel at index i
        # is the face between cells i-1 and i, so the face above the last
        # conductor cell needs a cell above it to exist).
        npad = nt + 1
        if numlevels > 1:
            ntop = npad/(nleaf*(2**(numlevels - 2)))
            ntop = np.ceil(ntop)
            lv[numlevels-1] = TopLevel(ntop, ltop/ntop, nmax)
            nmid = np.ones((3,), dtype=int)
            if numlevels > 2:
                for level in range(numlevels - 2):
                    for dim in range(3):
                        if nleaf[dim]*nmid[dim] < npad[dim]:
                            nmid[dim] *= 2
            ntotalfull = ntop * nmid * nleaf
            ngroups = ntotalfull/nleaf
            ngroups = ngroups.astype(int)
            # Sized from stencils.py, the same rules the depth-first
            # scan below fills the arrays with, so counter and filler
            # cannot drift apart. (mp_fortran.get_node_size counts the
            # same totals but pins each element's own-axis cell to the
            # opposite neighbour -- a whole-lattice shift, so its totals
            # agree by translation invariance while its per-box
            # distribution does not. Only the totals and the node-box
            # count were ever read.)
            nnzsize = st.count_elements(fullstruc, nleaf, ngroups)
            print("nnzsize =", nnzsize)
            print("nleaf =", nleaf, "  ntop =", ntop, "  ntotal =", ntotal,
                  "  ntotalfull =", ntotalfull)
            nmidlev = np.ones((3,))
            for level in range(numlevels-2, 0, -1):
                for dim in range(3):
                    if nmid[dim]/2**level >= 1:
                        nmidlev[dim] = 2
                lv[level] = MidLevel(nmidlev, lv[level+1].l/nmidlev, nmax)
                print("level =", level, "nmidlev =", nmidlev, "nmid =", nmid)
            lf = lv[1].l/nleaf
            nefg = ntotalfull/nleaf
            npx = ntotalfull/nleaf
            npy = ntotalfull/nleaf
            npz = ntotalfull/nleaf
            nleafpx = nleaf.copy()
            nleafpy = nleaf.copy()
            nleafpz = nleaf.copy()
            # a panel is a whole face on the cell lattice, so the panel
            # pitch is the cell pitch
            lfpx = lf.copy()
            lfpy = lf.copy()
            lfpz = lf.copy()
        else:
            lf = ltop/nt
            ntotalfull = ntotal
            nmax = None
            nefg = nt
            eshapes = st.element_shapes(nt)
            npx = np.array(eshapes[4])
            npy = np.array(eshapes[5])
            npz = np.array(eshapes[6])
            nleafpx = npx.copy()
            nleafpy = npy.copy()
            nleafpz = npz.copy()
            lfpx = lf.copy()
            lfpy = lf.copy()
            lfpz = lf.copy()
        leaf_e = LeafInduct(nleaf, nefg, lf, lscale, nmax, 'e')
        leaf_f = LeafInduct(nleaf, nefg, lf, lscale, nmax, 'f')
        leaf_g = LeafInduct(nleaf, nefg, lf, lscale, nmax, 'g')
        lv[0] = LeafLevel(nleaf, ntotalfull/nleaf, lf, nmax, None)
        leaf_px = LeafPoten(nleafpx, npx, lfpx, nmax, 'x')
        leaf_py = LeafPoten(nleafpy, npy, lfpy, nmax, 'y')
        leaf_pz = LeafPoten(nleafpz, npz, lfpz, nmax, 'z')
        ii = 0
        for leaf in [leaf_e, leaf_f, leaf_g, lv[0], leaf_px, leaf_py, leaf_pz]:
            if numlevels > 1:
                leaf.struc = np.zeros((nnzsize[ii, 0],),
                                      dtype=fullstruc.dtype)
                leaf.idx = np.zeros((nnzsize[ii, 0],), dtype=int)
                leaf.idx0 = np.zeros((nnzsize[3, 1] + 1,), dtype=int)
                leaf.above = lv[1]
                ii += 1
        if numlevels > 1:
            nt = np.prod(ngroups)
            for level in range(1, numlevels):
                lv[level].idx = np.zeros((nt,), dtype=int)
                if level < numlevels-1:
                    lv[level].idx0 = np.zeros((int(nt/np.prod(lv[level].n)+1),),
                                              dtype=int)
                elif level == numlevels-1:
                    lv[level].idx0 = np.zeros((2,), dtype=int)
                    lv[level].idx0[1] = nt
                lv[level-1].xidx = np.zeros((nt,), dtype=int)
                lv[level-1].yidx = np.zeros((nt,), dtype=int)
                lv[level-1].zidx = np.zeros((nt,), dtype=int)
                lv[level-1].above = lv[level]
                lv[level].below = lv[level-1]
                nt = np.size(lv[level].idx0) - 1
            iterator = np.zeros((numlevels,), dtype=int)
            treennz = np.zeros((numlevels,), dtype=int)
            oldtreennz = np.zeros((numlevels,), dtype=int)
            leafnnz = np.zeros((7,), dtype=int)
            # 7 leaves as follows:
            #     [0]: filaments, y-directed (holds If_e vector)
            #     [1]: filaments, x-directed (holds If_f vector)
            #     [2]: filaments, z-directed (holds If_g vector)
            #     [3]: nodes (holds Vn vector)
            #     [4]: panels, x-directed normal vector
            #     [5]: panels, y-directed normal vector
            #     [6]: panels, z-directed normal vector
            level = 1
            x = np.zeros((numlevels,), dtype=int)
            y = np.zeros((numlevels,), dtype=int)
            z = np.zeros((numlevels,), dtype=int)
            while True:
                if iterator[level] < np.prod(lv[level].n):
                    level = 1
                    x[:] = 0
                    y[:] = 0
                    z[:] = 0
                    for l in range(numlevels-1, 0, -1):
                        x[l] += np.floor_divide(iterator[l], lv[l].n[1] * lv[l].n[2])
                        y[l] += np.mod(np.floor_divide(iterator[l], lv[l].n[2]),
                                       lv[l].n[1])
                        z[l] += np.mod(iterator[l], lv[l].n[2])
                        x[l-1] = x[l] * lv[l-1].n[0]
                        y[l-1] = y[l] * lv[l-1].n[1]
                        z[l-1] = z[l] * lv[l-1].n[2]
                    # Cell window for this box, with ONE OVERHANG LAYER
                    # ON EACH SIDE: [x0-1, x0+nleaf] inclusive. Both are
                    # needed -- a cell-centred filament at local i joins
                    # cells i and i+1 (reaches high) while a panel at
                    # local i is the face between cells i-1 and i
                    # (reaches low). Clipped to the CELL count nt, not
                    # the node count: fullstruc has nt cells on each axis
                    # regardless of where the nodes sit.
                    xstart = np.max([x[0]-1, 0])
                    ystart = np.max([y[0]-1, 0])
                    zstart = np.max([z[0]-1, 0])
                    xstart = np.min([xstart, ntcells[0]])
                    ystart = np.min([ystart, ntcells[1]])
                    zstart = np.min([zstart, ntcells[2]])
                    xstop = np.min([x[0] + lv[0].n[0] + 1, ntcells[0]])
                    ystop = np.min([y[0] + lv[0].n[1] + 1, ntcells[1]])
                    zstop = np.min([z[0] + lv[0].n[2] + 1, ntcells[2]])
                    xsstart = 1 - x[0] + xstart
                    ysstart = 1 - y[0] + ystart
                    zsstart = 1 - z[0] + zstart
                    xsstop = xsstart + xstop - xstart
                    ysstop = ysstart + ystop - ystart
                    zsstop = zsstart + zstop - zstart
                    xsstart = np.max([xsstart, 0])
                    ysstart = np.max([ysstart, 0])
                    zsstart = np.max([zsstart, 0])
                    xsstop = np.max([xsstop, 0])
                    ysstop = np.max([ysstop, 0])
                    zsstop = np.max([zsstop, 0])
                    singlegroup = np.zeros((lv[0].n[0]+2, lv[0].n[1]+2,
                                            lv[0].n[2]+2),
                                           dtype=lv[0].struc.dtype)
                    singlegroup[xsstart:xsstop, ysstart:ysstop, zsstart:zsstop] = \
                        fullstruc[xstart:xstop, ystart:ystop, zstart:zstop]
                    # Element existence comes from stencils.py -- the
                    # single definition, shared with the single-level
                    # branch below. Every set is a flat nleaf**3 block
                    # here; each element's overhang lives in the
                    # neighbouring box and is gathered at apply time.
                    blk = (int(lv[0].n[0]), int(lv[0].n[1]),
                           int(lv[0].n[2]))
                    single = np.ndarray((7,), dtype=object)
                    for ii, s in enumerate(
                            st.struc_from_block(singlegroup, blk)):
                        single[ii] = s.ravel()
                    gnnznode = np.size(np.nonzero(single[3])[0])
                    # A box is stored if it holds ANY element, not only
                    # nodes. Under the cell scheme a surface panel sits
                    # on the face BELOW its conductor cell, so the
                    # panels capping the far side of a conductor land in
                    # the next box along -- which is all empty cells and
                    # therefore has no nodes. Gating on nodes alone
                    # silently dropped them (measured: 144 of 288 px
                    # panels on a solid 12^3). Nodes still gate nothing
                    # else: an empty node group is handled downstream
                    # (getslabidx already tests idx0[g+1] > idx0[g]).
                    keep = gnnznode > 0
                    if not keep:
                        keep = any(np.size(np.nonzero(single[ii])[0]) > 0
                                   for ii in range(7))
                    if keep:
                        lv[0].xidx[treennz[1]] = x[1]
                        lv[0].yidx[treennz[1]] = y[1]
                        lv[0].zidx[treennz[1]] = z[1]
                        lv[1].idx[treennz[1]] = iterator[1]
                        for leaf, ii in zip([leaf_e, leaf_f, leaf_g, lv[0],
                                             leaf_px, leaf_py, leaf_pz],
                                            list(range(7))):
                            single_idx = np.nonzero(single[ii])[0]
                            gnnz = np.size(single_idx)
                            leaf.struc[leafnnz[ii]:leafnnz[ii]+gnnz] = \
                                single[ii][single_idx]
                            leaf.idx[leafnnz[ii]:leafnnz[ii]+gnnz] = single_idx
                            leaf.idx0[treennz[1] + 1] = leafnnz[ii] + gnnz
                            leafnnz[ii] += gnnz
                        treennz[0] += gnnznode
                        treennz[1] += 1
                    iterator[1] += 1
                else:
                    iterator[level] = 0
                    level += 1
                    if level == numlevels:
                        break
                    if treennz[level-1] > oldtreennz[level-1]:
                        oldtreennz[level-1] = treennz[level-1]
                        lv[level-1].xidx[treennz[level]] = x[level]
                        lv[level-1].yidx[treennz[level]] = y[level]
                        lv[level-1].zidx[treennz[level]] = z[level]
                        lv[level].idx[treennz[level]] = iterator[level]
                        treennz[level] += 1
                        lv[level-1].idx0[treennz[level]] = treennz[level-1]
                    iterator[level] += 1
            # leaf_px.idx0 *= 4
            # leaf_py.idx0 *= 4
            # leaf_pz.idx0 *= 4
            for level in range(1, numlevels):
                lv[level].idx = lv[level].idx[:treennz[level]]
                lv[level-1].xidx = lv[level-1].xidx[:treennz[level]]
                lv[level-1].yidx = lv[level-1].yidx[:treennz[level]]
                lv[level-1].zidx = lv[level-1].zidx[:treennz[level]]
                lv[level-1].idx0 = lv[level-1].idx0[:treennz[level] + 1]
                lv[level].data = np.zeros((np.size(lv[level].idx),
                                           lv[level].nnmax),
                                          dtype=np.complex128)

            lv[numlevels-1].getneighbors()
            for level in range(1, numlevels-1):
                lv[level].midm2linit()
        else:
            lv[0].xidx = np.zeros((1,), dtype=int)
            lv[0].yidx = np.zeros((1,), dtype=int)
            lv[0].zidx = np.zeros((1,), dtype=int)
            lv[0].neighbors = np.ndarray((27, 1), dtype=int)
            lv[0].neighbors[...] = -1
            lv[0].neighbors[13, 0] = 0
            # leaf.slabidx0 = leaf.idx0.copy()
            # Element existence comes from stencils.py -- the single
            # definition, shared with the multilevel scan above. The
            # whole domain is one box here, so each element lattice
            # carries its own +1 face overhang (unlike the multilevel
            # case, where every set is a flat nleaf**3 block and the
            # overhang lives in the neighbouring box).
            strucs = st.struc_from_cells(fullstruc)
            leaf_e.n = np.array(eshapes[0])
            leaf_f.n = np.array(eshapes[1])
            leaf_g.n = np.array(eshapes[2])
            for leaf, s in zip([leaf_e, leaf_f, leaf_g, lv[0], leaf_px,
                                leaf_py, leaf_pz], strucs):
                s = s.flatten()
                leaf.idx = np.nonzero(s)[0]
                leaf.struc = s[leaf.idx]
            for leaf in [leaf_e, leaf_f, leaf_g, lv[0], leaf_px, leaf_py,
                         leaf_pz]:
                leaf.idx0 = np.zeros((2,), dtype=int)
                leaf.idx0[1] = np.size(leaf.idx)
        for leaf in [leaf_e, leaf_f, leaf_g, leaf_px, leaf_py, leaf_pz]:
            leaf.xidx = lv[0].xidx
            leaf.yidx = lv[0].yidx
            leaf.zidx = lv[0].zidx
            leaf.getslabidx()
            leaf.neighbors = lv[0].neighbors
        self.lv = lv
        self.e = leaf_e
        self.f = leaf_f
        self.g = leaf_g
        self.px = leaf_px
        self.py = leaf_py
        self.pz = leaf_pz
        if not capacitive:
            # LpR-only tree: skip the potential/capacitive machinery
            # (node2panel, the LeafPoten.p2pinit3 coupling, and the node-to-node
            # n2n matrix). These are consumed only by the LpPR traverseP* paths,
            # never by the LpR solve (traverseRL / connectA / connectAT). The
            # inductive leaves e/f/g and the level hierarchy are already fully
            # built above.
            return
        self.node2panel()
        # Homogeneous background dielectric (phase 1, 2026-08-07): a
        # uniform eps_r divides EVERY coefficient of potential, and the
        # kernels stay free-space -- so divide each eps-carrying object
        # ONCE, at build, before anything is factored: the panel
        # leaves' L2P harmonics (ynmr carries the single 1/(4 pi eps0)
        # prefactor of the far field), the assembled near-field n2n,
        # the circulant spectra, and the fftnear transfer tables. The
        # inductive path is untouched (dielectrics do not exist
        # quasi-magnetostatically).
        self.eps_r = float(eps_r)
        if self.eps_r != 1.0:
            for pleaf in (self.px, self.py, self.pz):
                # single-level trees have no far field, hence no L2P
                # harmonics on the panel leaves
                if hasattr(pleaf, 'ynmr'):
                    pleaf.ynmr = pleaf.ynmr/self.eps_r
        if circulant and numlevels == 1:
            # Circulant single level (the PyPEEC lesson): the whole-domain
            # panel potential operator is applied by FFT from the SAME gen
            # kernel tables the dense assembly below would look up -- exact,
            # O(N_grid log N_grid) per apply, O(N_grid) spectra storage --
            # while self.n2n holds only the sparse NEAR-FIELD node coupling
            # (offset band), consumed by the W = P_ext^{-1} rescaling and
            # C_cap exactly as at multilevel. traverseP3 dispatches on
            # self.circpoten. Validated bit-against-the-dense-oracle by
            # validate_circulant_poten.py.
            import circulant_poten
            self.circpoten = circulant_poten.CirculantPoten(self)
            if self.eps_r != 1.0:
                for blk in self.circpoten._blocks.values():
                    blk['FK'] = blk['FK']/self.eps_r
                self.circpoten.n2n_near = \
                    self.circpoten.n2n_near/self.eps_r
            self.n2n = self.circpoten.n2n_near.tocsr()
            n2nexternal = self.n2n[self.external][:, self.external]
            self.n2nchol = self._factor_n2n(n2nexternal,
                                            truncation='circulant')
            return
        if not keep_n2n:
            # NO stored near-field n2n (2026-08-08). The assembled n2n
            # is the multilevel capacitive path's memory wall --
            # ~27*leaf^3 entries per external node, 33 GB at a 320^2
            # board against 0.1 GB for the inductive tree -- and with
            # fftnear the OPERATOR never reads it (traverseP3's
            # Toeplitz near field, validated ~1e-15 by
            # validate_leaf_poten_p2p). Its only production consumers
            # were the exact-W Cholesky (superseded by the band W at
            # scale) and extract_ccap's window blocks (now computable
            # kernel-direct: reluctance.kernel_ccap_block_getter). So
            # skip the p2pinit3 assembly entirely; solvers must use
            # wsolve='band' (LpPRSolver 'auto' does, since n2nchol is
            # absent).
            if not (bool(fftnear) and numlevels > 1):
                raise ValueError(
                    "keep_n2n=False requires fftnear=True and "
                    "numlevels > 1: without the stored n2n the near "
                    "field must come from the Toeplitz p2p tables")
            self.fftnear = True
            for leaf in (self.px, self.py, self.pz):
                leaf.p2pinit(leaf_px, leaf_py, leaf_pz)
            if self.eps_r != 1.0:
                for leaf in (self.px, self.py, self.pz):
                    leaf.p2p_trans_x = leaf.p2p_trans_x/self.eps_r
                    leaf.p2p_trans_y = leaf.p2p_trans_y/self.eps_r
                    leaf.p2p_trans_z = leaf.p2p_trans_z/self.eps_r
            self.n2n = None
            self.n2nchol = None
            self.n2nchol_error = (
                "keep_n2n=False: the near-field n2n was never "
                "assembled (kernel-direct band-W route)")
            return
        # p2pinit3 returns P[source, target] (potential on targets is
        # P.T @ q_source), so the node-to-node map is
        #     n2n = sum_t B_t.T @ ( sum_s P[s->t].T @ B_s )
        # with B_t = node2p* the node-to-panel charge projections. The
        # original multiplied the untransposed blocks against the wrong
        # projections (shape mismatch for cross blocks whenever the panel
        # counts differ -- the "1600 vs 1856" abort on setup2).
        (PX2PX, PX2PY, PX2PZ) = self.px.p2pinit3(leaf_px, leaf_py, leaf_pz)
        (PY2PX, PY2PY, PY2PZ) = self.py.p2pinit3(leaf_px, leaf_py, leaf_pz)
        (PZ2PX, PZ2PY, PZ2PZ) = self.pz.p2pinit3(leaf_px, leaf_py, leaf_pz)
        PHIX = PX2PX.T.dot(self.node2px) + PY2PX.T.dot(self.node2py) + \
            PZ2PX.T.dot(self.node2pz)
        PHIY = PX2PY.T.dot(self.node2px) + PY2PY.T.dot(self.node2py) + \
            PZ2PY.T.dot(self.node2pz)
        PHIZ = PX2PZ.T.dot(self.node2px) + PY2PZ.T.dot(self.node2py) + \
            PZ2PZ.T.dot(self.node2pz)
        self.n2n = self.node2px.T.dot(PHIX) + self.node2py.T.dot(PHIY) + \
            self.node2pz.T.dot(PHIZ)
        # Optional FFT near field for the capacitive matvec (the
        # leaf_poten.p2pinit/p2p path, validated to ~1e-15 against the
        # p2pinit3 assembly above by validate_leaf_poten_p2p.py). The
        # assembled n2n is still built and kept: the W = P_ext^-1
        # rescaling and C_cap need matrix ENTRIES, not an operator --
        # the same split circulant_poten uses. Only meaningful at
        # numlevels > 1, where the FMM supplies everything outside the
        # 27-neighbourhood that p2p covers.
        self.fftnear = bool(fftnear) and numlevels > 1
        if self.fftnear:
            for leaf in (self.px, self.py, self.pz):
                leaf.p2pinit(leaf_px, leaf_py, leaf_pz)
        if self.eps_r != 1.0:
            self.n2n = self.n2n/self.eps_r
            if self.fftnear:
                for leaf in (self.px, self.py, self.pz):
                    leaf.p2p_trans_x = leaf.p2p_trans_x/self.eps_r
                    leaf.p2p_trans_y = leaf.p2p_trans_y/self.eps_r
                    leaf.p2p_trans_z = leaf.p2p_trans_z/self.eps_r
        n2nexternal = self.n2n.tocsr()[self.external][:, self.external]
        if numlevels == 1:
            # toarray (ndarray), not todense (np.matrix): traverseP3 dots
            # this against 1-D vectors and np.matrix would reshape them.
            self.n2n = self.n2n.toarray()
            self.n2nchol = self._factor_n2n(n2nexternal, dense=True,
                                            truncation='none')
        else:
            self.n2nchol = self._factor_n2n(n2nexternal,
                                            truncation='leaf')

    def _factor_n2n(self, n2nexternal, dense=False, truncation='leaf'):
        """Cholesky-factor the external node-to-node block, loudly.

        ``truncation`` names WHAT limited the near field, because that is
        what determines the remedy and the three cases have nothing in
        common operationally:

        * ``'circulant'`` -- distance band; widen
          ``circulant_poten.CirculantPoten.DEFAULT_CUTOFF``.
        * ``'leaf'`` -- the 27-box leaf neighbourhood at numlevels > 1.
          There is NO cutoff parameter on this path; only nleaf widens it.
        * ``'none'`` -- single level, one box spanning the grid, so the
          operator is complete and indefiniteness is not a truncation
          question at all.

        Cholesky is not just the fast choice here -- it is an ASSERTION.
        The untruncated coefficient-of-potential matrix is a Gram matrix
        and therefore positive definite, so a failure means the operator
        being factored is NOT that matrix. In practice it means the
        near-field truncation has dropped too much: measured under the
        cell scheme at cutoff 2, the smallest eigenvalue declines
        monotonically with mesh size and changes sign near NT = 16.
        An LU would factor that happily and hand back a preconditioner
        for an operator the surrounding algorithm assumes is SPD, with
        nothing to show for it but poor convergence at large N.

        Failure is therefore recorded on the tree and warned about
        rather than discarded. It is not raised, because several solver
        paths never consume ``n2nchol`` and should still build; the
        consumers that DO need it can report ``n2nchol_error``.
        """
        self.n2nchol_error = None
        # Distinguishes a MATH failure (the operator is not SPD, so any
        # consumer that assumes it is must refuse to proceed) from an
        # environment/other failure (cholmod missing or similar, where the
        # matrix is fine and a dense LU fallback is merely slower). Only
        # the former is a reason to abort.
        self.n2nchol_indefinite = False
        try:
            if dense:
                return np.linalg.cholesky(n2nexternal.todense())
            return cholmod.cholesky(n2nexternal.tocsc(),
                                    ordering_method='amd')
        except Exception as exc:
            npd = getattr(cholmod, 'CholmodNotPositiveDefiniteError', ())
            indefinite = isinstance(exc, np.linalg.LinAlgError) or \
                (npd and isinstance(exc, npd))
            self.n2nchol_indefinite = bool(indefinite)
            if indefinite:
                # The remedy depends ENTIRELY on which path truncated the
                # operator, and naming the wrong knob sends the reader off
                # to a parameter that has no effect here.
                if truncation == 'circulant':
                    where = ("This is the CIRCULANT path, whose near field "
                             "is a distance band, so it is the band "
                             "dropping too much: widen "
                             "circulant_poten.CirculantPoten.DEFAULT_CUTOFF "
                             "for this scheme and re-check the smallest "
                             "eigenvalue.")
                elif truncation == 'leaf':
                    where = ("This is the MULTILEVEL path, where the near "
                             "field is the 27-box leaf neighbourhood and "
                             "there is NO cutoff parameter -- "
                             "circulant_poten.DEFAULT_CUTOFF does not apply "
                             "and changing it will do nothing. nleaf is the "
                             "only knob: the near field is set by the leaf "
                             "size ALONE, not by numlevels (measured 15^3 "
                             "cell, nleaf 2: lambda_min = -3.49e14 at both "
                             "2 and 3 levels; nleaf 3 -4.44e14, nleaf 4 "
                             "+1.34e13, nleaf 5 +2.83e14 -- so >= 4 is "
                             "definite there). Dropping to numlevels=1 also "
                             "removes the truncation outright, at the cost "
                             "of the FMM.")
                else:
                    where = ("This is the SINGLE-LEVEL path, where one box "
                             "spans the whole grid, so the near field "
                             "covers every panel pair and n2n is NOT "
                             "truncated at all. Indefiniteness here is "
                             "therefore not a cutoff question -- suspect "
                             "the panel/node assembly (node2panel, "
                             "p2pinit3) or the scheme conventions instead.")
                self.n2nchol_error = (
                    "the near-field n2n is NOT positive definite (%dx%d "
                    "over the external nodes). The untruncated operator is "
                    "a Gram matrix and always is. %s Do NOT substitute an "
                    "LU: it would succeed on an indefinite matrix and hide "
                    "this."
                    % (n2nexternal.shape[0], n2nexternal.shape[1], where))
            else:
                self.n2nchol_error = ("%s: %s" % (type(exc).__name__, exc))
            warnings.warn("n2nchol unavailable -- " + self.n2nchol_error,
                          RuntimeWarning, stacklevel=2)
            return None

    def node2panel(self):
        """Build the sparse node-to-panel charge projection matrices.

        For every surface panel, finds the circuit node it belongs to (each
        panel maps to exactly one node; the search covers the 8 forward
        neighbour groups since panel grids straddle group boundaries) and
        assembles the maps as one-entry-per-row CSR matrices
        ``self.node2px/py/pz`` (panel x node). Rows are then normalized by
        the total number of panels attached to each node, so applying
        ``node2px`` distributes a node charge equally over its panels and the
        transpose gathers panel potentials back to nodes. Also records
        ``self.external`` -- the nodes that own at least one surface panel
        (the only ones with capacitive coupling). Capacitive (LpPR) path
        only.
        """
        pxindptr = np.r_[0, 1:self.px.idx0[-1]+1]
        pyindptr = np.r_[0, 1:self.py.idx0[-1]+1]
        pzindptr = np.r_[0, 1:self.pz.idx0[-1]+1]
        pxindices = np.zeros((self.px.idx0[-1],), dtype=np.int_)
        pyindices = np.zeros((self.py.idx0[-1],), dtype=np.int_)
        pzindices = np.zeros((self.pz.idx0[-1],), dtype=np.int_)
        pxdata = np.zeros_like(pxindices, dtype=self.px.struc.dtype)
        pydata = np.zeros_like(pyindices, dtype=self.py.struc.dtype)
        pzdata = np.zeros_like(pzindices, dtype=self.pz.struc.dtype)
        nextneigh = np.array([13, 14, 16, 17, 22, 23, 25, 26], dtype=np.int_)
        # Backward octant, for the cell scheme. Neighbour index is
        # 9*(dx+1) + 3*(dy+1) + (dz+1), so these are the offsets with
        # each component in {0, -1}, ordered to match the same
        # (nnx, nny, nnz) bit decomposition as nextneigh. A cell-scheme
        # panel at local i is the face between cells i-1 and i, so it
        # reaches BACKWARD -- the opposite of the corner-node case.
        prevneigh = np.array([13, 12, 10, 9, 4, 3, 1, 0], dtype=np.int_)
        # Node-grid workspace. At a single level the nodes live on the whole
        # domain's (NT+1) grid (self.ntotal), independent of the leaf box
        # size nleaf; the multilevel branch uses one box + a forward halo
        # (lv[0].n + 1). Decoupling this from nleaf lets nleaf be sized for
        # the inductive p2p (which needs the box to cover the staggered
        # filament grids) without corrupting the node<->panel attachment.
        if self.numlevels == 1:
            ndims = (int(self.ntotal[0]), int(self.ntotal[1]),
                     int(self.ntotal[2]))
        else:
            ndims = (self.lv[0].n[0]+1, self.lv[0].n[1]+1, self.lv[0].n[2]+1)
        singlegroup = np.zeros(ndims, dtype=self.lv[0].struc.dtype)
        singleind = np.zeros_like(singlegroup, dtype=np.int_)
        for group in range(np.size(self.lv[0].idx0) - 1):
            singlegroup[...] = 0
            singleind[...] = 0
            if self.numlevels == 1:
                # Single level: nodes live directly on the (n+1) grid of the
                # whole domain (one group, no neighbour halo), so scatter the
                # compressed node set straight into the workspace. The
                # per-box decoding below assumes the multilevel convention
                # (n nodes per box, far side supplied by forward neighbours)
                # and produces garbage attachments at a single level.
                dims = singlegroup.shape
                ii = self.lv[0].idx
                xx = ii // (dims[1]*dims[2])
                yy = (ii // dims[2]) % dims[1]
                zz = ii % dims[2]
                singlegroup[xx, yy, zz] = self.lv[0].struc
                singleind[xx, yy, zz] = np.arange(np.size(ii))
            else:
              # box's own nodes land at local 1..n; local 0 on each axis
              # holds the previous box's last node
              for nn in range(8):
                neigh = prevneigh[nn]
                neighgroup = self.lv[0].neighbors[neigh, group]
                if neighgroup >= 0:
                    nnx = nn//4
                    nny = nn//2 % 2
                    nnz = nn % 2
                    n0 = self.lv[0].n
                    ii = self.lv[0].idx0[neighgroup]
                    endngp = self.lv[0].idx0[neighgroup+1]
                    if endngp <= ii:
                        # panel-only box: no nodes to gather from
                        continue
                    for x in range(1 if nnx else int(n0[0])):
                        sx = int(n0[0])-1 if nnx else x
                        for y in range(1 if nny else int(n0[1])):
                            sy = int(n0[1])-1 if nny else y
                            for z in range(1 if nnz else int(n0[2])):
                                sz = int(n0[2])-1 if nnz else z
                                pos = n0[1:].prod()*sx + n0[2]*sy + sz
                                idxii = self.lv[0].idx[ii]
                                while idxii < pos and ii < endngp-1:
                                    ii += 1
                                    idxii = self.lv[0].idx[ii]
                                if idxii == pos:
                                    xx = 0 if nnx else x+1
                                    yy = 0 if nny else y+1
                                    zz = 0 if nnz else z+1
                                    singlegroup[xx, yy, zz] = \
                                        self.lv[0].struc[ii]
                                    singleind[xx, yy, zz] = ii
            # A panel is a WHOLE FACE lying between cells i-1 and i
            # along its normal, and it exists only where occupancy
            # changes across that face -- so exactly one of the two
            # is conductor, and that cell's node owns it. No
            # tie-breaking is needed or possible.
            nd = singlegroup.shape
            # multilevel holds a backward halo at index 0 on each
            # axis, so a box-local cell index c sits at c+1
            shift = 0 if self.numlevels == 1 else 1
            for leaf, axis, ind, dat in (
                    (self.px, 0, pxindices, pxdata),
                    (self.py, 1, pyindices, pydata),
                    (self.pz, 2, pzindices, pzdata)):
                for i in range(leaf.idx0[group], leaf.idx0[group+1]):
                    ii = leaf.idx[i]
                    c = [ii//leaf.n[1:].prod(),
                         ii//leaf.n[2] % leaf.n[1],
                         ii % leaf.n[2]]
                    hi = [v + shift for v in c]
                    lo = list(hi)
                    lo[axis] -= 1
                    pick = None
                    for cand in (lo, hi):
                        if all(0 <= cand[k] < nd[k] for k in range(3)) \
                                and singlegroup[cand[0], cand[1],
                                                cand[2]]:
                            pick = cand
                            break
                    if pick is None:
                        continue
                    dat[i] = singlegroup[pick[0], pick[1], pick[2]]
                    ind[i] = singleind[pick[0], pick[1], pick[2]]
            continue
            for i in range(self.px.idx0[group], self.px.idx0[group+1]):
                ii = self.px.idx[i]
                x = ii//self.px.n[1:].prod()
                y = ((ii//self.px.n[2] % self.px.n[1]) + 1) // 2
                z = ((ii % self.px.n[2]) + 1) // 2
                pxdata[i] = singlegroup[x, y, z]
                pxindices[i] = singleind[x, y, z]
            for i in range(self.py.idx0[group], self.py.idx0[group+1]):
                ii = self.py.idx[i]
                x = (ii//self.py.n[1:].prod() + 1) // 2
                y = ii//self.py.n[2] % self.py.n[1]
                z = ((ii % self.py.n[2]) + 1) // 2
                pydata[i] = singlegroup[x, y, z]
                pyindices[i] = singleind[x, y, z]
            for i in range(self.pz.idx0[group], self.pz.idx0[group+1]):
                ii = self.pz.idx[i]
                x = (ii//self.pz.n[1:].prod() + 1) // 2
                y = ((ii//self.pz.n[2] % self.pz.n[1]) + 1) // 2
                z = ii % self.pz.n[2]
                pzdata[i] = singlegroup[x, y, z]
                pzindices[i] = singleind[x, y, z]
        external_alloc = np.zeros((self.lv[0].idx0[-1],), dtype=np.int8)
        for i in pxindices:
            external_alloc[i] = 1
        for i in pyindices:
            external_alloc[i] = 1
        for i in pzindices:
            external_alloc[i] = 1
        self.external = np.nonzero(external_alloc)[0]
        px2node = csr_matrix((np.float32(pxdata), pxindices, pxindptr),
                             shape=(pxindptr[-1], self.lv[0].idx0[-1]))
        py2node = csr_matrix((np.float32(pydata), pyindices, pyindptr),
                             shape=(pyindptr[-1], self.lv[0].idx0[-1]))
        pz2node = csr_matrix((np.float32(pzdata), pzindices, pzindptr),
                             shape=(pzindptr[-1], self.lv[0].idx0[-1]))
        self.node2px = px2node.tocsc()
        self.node2py = py2node.tocsc()
        self.node2pz = pz2node.tocsc()
        pxcount = self.node2px.indptr[1:] - self.node2px.indptr[:-1]
        pycount = self.node2py.indptr[1:] - self.node2py.indptr[:-1]
        pzcount = self.node2pz.indptr[1:] - self.node2pz.indptr[:-1]
        totalcount = pxcount + pycount + pzcount
        for i in range(np.size(self.node2px.indptr)-1):
            pxind = np.s_[self.node2px.indptr[i]:self.node2px.indptr[i+1]]
            self.node2px.data[pxind] /= totalcount[i]
        for i in range(np.size(self.node2py.indptr)-1):
            pyind = np.s_[self.node2py.indptr[i]:self.node2py.indptr[i+1]]
            self.node2py.data[pyind] /= totalcount[i]
        for i in range(np.size(self.node2pz.indptr)-1):
            pzind = np.s_[self.node2pz.indptr[i]:self.node2pz.indptr[i+1]]
            self.node2pz.data[pzind] /= totalcount[i]

    def spaiinit(self):
        """Build the coupled three-orientation SPAI preconditioner (dormant).

        Tree-level generalization of :meth:`leaf_induct.LeafInduct.spaiinit`:
        instead of preconditioning each filament orientation independently
        with a 7-point stencil, it builds a **15-point** inverse stencil per
        filament -- 7 same-orientation neighbours plus 4+4 cross-orientation
        neighbours from the other two filament directions, coupled through
        the KCL continuity terms (the ``1/alpha`` entries and the
        ``Asign = [1,-1,-1,1]`` incidence pattern in the ``A*`` blocks).
        For each leaf group it assembles the local block operator

        ``[[Ze, Afe, Age], [Aef, Zf, Agf], [Aeg, Afg, Zg]]``

        (impedance ``jw*Lp + R`` on the diagonal blocks, scaled incidence on
        the off-diagonal ones) and solves one :func:`lsqr` least-squares
        problem per filament, storing the 15 weights as a column of
        ``self.Zeinv``/``Zfinv``/``Zginv``. Consumed by :meth:`spaiapply` /
        :meth:`spaiapply2`.

        Dormant and unfinished: never called by the default driver (the
        ``RDFinit`` calls are commented out), requires ``self.jomega`` and
        ``self.alpha`` to be preset, and still contains a debug
        ``print(innere)`` in the inner loop.
        """
        self.Zeinv = None
        self.Zfinv = None
        self.Zginv = None
        m = self.lv[0].m
        coeffe = np.zeros((4*m[0]+1, 4*m[1]+1, 4*m[2]+1),
                          dtype=self.e.induct.dtype)
        coefff = np.zeros((4*m[0]+1, 4*m[1]+1, 4*m[2]+1),
                          dtype=self.f.induct.dtype)
        coeffg = np.zeros((4*m[0]+1, 4*m[1]+1, 4*m[2]+1),
                          dtype=self.g.induct.dtype)
        for induct, coeff in zip([self.e.induct, self.f.induct, self.g.induct],
                                 [coeffe, coefff, coeffg]):
            ncx = 2*m[0]
            ncy = 2*m[1]
            ncz = 2*m[2]
            coeff[ncx:, ncy:, ncz:] = induct
            coeff[ncx:, ncy:, ncz::-1] = induct
            coeff[ncx:, ncy::-1, ncz:] = induct
            coeff[ncx:, ncy::-1, ncz::-1] = induct
            coeff[ncx::-1, ncy:, ncz:] = induct
            coeff[ncx::-1, ncy:, ncz::-1] = induct
            coeff[ncx::-1, ncy::-1, ncz:] = induct
            coeff[ncx::-1, ncy::-1, ncz::-1] = induct
        struce = np.zeros((3*m[0], 3*m[1], 3*m[2]), dtype=self.e.induct.dtype)
        strucf = np.zeros((3*m[0], 3*m[1], 3*m[2]), dtype=self.f.induct.dtype)
        strucg = np.zeros((3*m[0], 3*m[1], 3*m[2]), dtype=self.g.induct.dtype)
        singlegroupe = np.zeros((np.prod(self.e.n),),
                                dtype=self.e.induct.dtype)
        singlegroupf = np.zeros((np.prod(self.f.n),),
                                dtype=self.f.induct.dtype)
        singlegroupg = np.zeros((np.prod(self.g.n),),
                                dtype=self.g.induct.dtype)
        Zeinv = np.zeros((15, self.e.idx0[-1]), dtype=self.e.data.dtype)
        Zfinv = np.zeros((15, self.f.idx0[-1]), dtype=self.f.data.dtype)
        Zginv = np.zeros((15, self.g.idx0[-1]), dtype=self.g.data.dtype)
        Asign = np.array([1, -1, -1, 1], dtype=int)
        for group in range(np.size(self.lv[0].idx0)-1):
            struce[...] = 0
            strucf[...] = 0
            strucg[...] = 0
            for neigh in range(27):
                neighgroup = self.lv[0].neighbors[neigh, group]
                if neighgroup >= 0:
                    neighx = neigh//9
                    neighy = (neigh//3) % 3
                    neighz = neigh % 3
                    startx = neighx*m[0]
                    starty = neighy*m[1]
                    startz = neighz*m[2]
                    singlegroupe[:] = 0
                    singlegroupf[:] = 0
                    singlegroupg[:] = 0
                    neighidx0e = self.e.idx0[neighgroup]
                    neighidx1e = self.e.idx0[neighgroup+1]
                    iidxe = self.e.idx[neighidx0e:neighidx1e]
                    singlegroupe[iidxe] = self.e.struc[neighidx0e:neighidx1e]
                    stopxe = startx + self.e.n[0]
                    stopye = starty + self.e.n[1]
                    stopze = startz + self.e.n[2]
                    struce[startx:stopxe, starty:stopye, startz:stopze] = \
                        np.reshape(singlegroupe, (self.e.n[0], self.e.n[1],
                                                  self.e.n[2]))
                    neighidx0f = self.f.idx0[neighgroup]
                    neighidx1f = self.f.idx0[neighgroup+1]
                    iidxf = self.f.idx[neighidx0f:neighidx1f]
                    singlegroupf[iidxf] = self.f.struc[neighidx0f:neighidx1f]
                    stopxf = startx + self.f.n[0]
                    stopyf = starty + self.f.n[1]
                    stopzf = startz + self.f.n[2]
                    strucf[startx:stopxf, starty:stopyf, startz:stopzf] = \
                        np.reshape(singlegroupf, (self.f.n[0], self.f.n[1],
                                                  self.f.n[2]))
                    neighidx0g = self.g.idx0[neighgroup]
                    neighidx1g = self.g.idx0[neighgroup+1]
                    iidxg = self.g.idx[neighidx0g:neighidx1g]
                    singlegroupg[iidxg] = self.g.struc[neighidx0g:neighidx1g]
                    stopxg = startx + self.g.n[0]
                    stopyg = starty + self.g.n[1]
                    stopzg = startz + self.g.n[2]
                    strucg[startx:stopxg, starty:stopyg, startz:stopzg] = \
                        np.reshape(singlegroupg, (self.g.n[0], self.g.n[1],
                                                  self.g.n[2]))
            totalsizee = np.count_nonzero(struce)
            totalsizef = np.count_nonzero(strucf)
            totalsizeg = np.count_nonzero(strucg)
            innerx = np.s_[m[0]-1:2*m[0]+1]
            innery = np.s_[m[1]-1:2*m[1]+1]
            innerz = np.s_[m[2]-1:2*m[2]+1]
            innerstruce = struce[innerx, innery, innerz]
            innerstrucf = strucf[innerx, innery, innerz]
            innerstrucg = strucg[innerx, innery, innerz]
            innersizee = np.count_nonzero(innerstruce)
            innersizef = np.count_nonzero(innerstrucf)
            innersizeg = np.count_nonzero(innerstrucg)
            Ze = np.zeros((innersizee, totalsizee), dtype=self.e.data.dtype)
            Zf = np.zeros((innersizef, totalsizef), dtype=self.f.data.dtype)
            Zg = np.zeros((innersizeg, totalsizeg), dtype=self.g.data.dtype)
            totalnnze = np.flatnonzero(struce)
            totalnnzf = np.flatnonzero(strucf)
            totalnnzg = np.flatnonzero(strucg)
            strucidxe = np.empty_like(struce, dtype=int)
            strucidxf = np.empty_like(strucf, dtype=int)
            strucidxg = np.empty_like(strucg, dtype=int)
            strucidxe[...] = -1
            strucidxf[...] = -1
            strucidxg[...] = -1
            strucidxe[np.nonzero(struce)] = np.r_[:totalsizee]
            strucidxf[np.nonzero(strucf)] = np.r_[:totalsizef]
            strucidxg[np.nonzero(strucg)] = np.r_[:totalsizeg]
            innerstrucidxe = np.empty_like(innerstruce, dtype=int)
            innerstrucidxf = np.empty_like(innerstrucf, dtype=int)
            innerstrucidxg = np.empty_like(innerstrucg, dtype=int)
            innerstrucidxe[...] = -1
            innerstrucidxf[...] = -1
            innerstrucidxg[...] = -1
            innerstrucidxe[np.nonzero(innerstruce)] = np.r_[:innersizee]
            innerstrucidxf[np.nonzero(innerstrucf)] = np.r_[:innersizef]
            innerstrucidxg[np.nonzero(innerstrucg)] = np.r_[:innersizeg]
            innere = 0
            innerf = 0
            innerg = 0
            Afe = np.zeros((totalsizef, innersizee), dtype=int)
            Age = np.zeros((totalsizeg, innersizee), dtype=int)
            Aef = np.zeros((totalsizee, innersizef), dtype=int)
            Agf = np.zeros((totalsizeg, innersizef), dtype=int)
            Aeg = np.zeros((totalsizee, innersizeg), dtype=int)
            Afg = np.zeros((totalsizef, innersizeg), dtype=int)
            fe = np.zeros((4,), dtype=int)
            ge = np.zeros((4,), dtype=int)
            ef = np.zeros((4,), dtype=int)
            gf = np.zeros((4,), dtype=int)
            eg = np.zeros((4,), dtype=int)
            fg = np.zeros((4,), dtype=int)
            #     for fily in range(-1, m[1]+1):
            #         for filz in range(-1, m[2]+1):
            for filx in range(m[0]):
                for fily in range(m[1]):
                    for filz in range(m[2]):
                        file = struce[m[0]+filx, m[1]+fily, m[2]+filz]
                        filf = strucf[m[0]+filx, m[1]+fily, m[2]+filz]
                        filg = strucg[m[0]+filx, m[1]+fily, m[2]+filz]
                        if file > 0 or filf > 0 or filg > 0:
                            xstart = m[0] - filx
                            ystart = m[1] - fily
                            zstart = m[2] - filz
                            xstop = xstart + 3*m[0]
                            ystop = ystart + 3*m[1]
                            zstop = zstart + 3*m[2]
                            xmid = 2*m[0] - xstart
                            ymid = 2*m[1] - ystart
                            zmid = 2*m[2] - zstart
                        if file > 0:
                            Ztempe = struce*self.jomega * coeffe[xstart:xstop,
                                                                 ystart:ystop,
                                                                 zstart:zstop]
                            Ztempe[xmid, ymid, zmid] += self.e.r
                            Ztempe[xmid, ymid-1, zmid] += 1/self.alpha
                            Ztempe[xmid, ymid+1, zmid] += 1/self.alpha
                            Ztempe[xmid, ymid, zmid] -= 2/self.alpha
                            Ze[innere, :] = np.ravel(Ztempe)[totalnnze]
                            # fe[0] = strucidxf[filx+1, fily+1, filz+1]
                            # fe[1] = strucidxf[filx+1, fily, filz+1]
                            # fe[2] = strucidxf[filx+2, fily+1, filz+1]
                            # fe[3] = strucidxf[filx+2, fily, filz+1]
                            fe[0] = innerstrucidxf[filx, fily+1, filz+1]
                            fe[1] = innerstrucidxf[filx, fily+2, filz+1]
                            fe[2] = innerstrucidxf[filx+1, fily+1, filz+1]
                            fe[3] = innerstrucidxf[filx+1, fily+2, filz+1]
                            fennz = np.nonzero(fe + 1)[0]
                            Afe[fe[fennz], innere] = 1/self.alpha*Asign[fennz]
                            ge[0] = innerstrucidxg[filx+1, fily+1, filz]
                            ge[1] = innerstrucidxg[filx+1, fily+1, filz+1]
                            ge[2] = innerstrucidxg[filx+1, fily+2, filz]
                            ge[3] = innerstrucidxg[filx+1, fily+2, filz+1]
                            gennz = np.nonzero(ge + 1)[0]
                            Age[ge[gennz], innere] = 1/self.alpha*Asign[gennz]
                            innere += 1
                        if filf > 0:
                            Ztempf = strucf*self.jomega * coefff[xstart:xstop,
                                                                 ystart:ystop,
                                                                 zstart:zstop]
                            Ztempf[xmid, ymid, zmid] += self.f.r
                            Ztempf[xmid-1, ymid, zmid] += 1/self.alpha
                            Ztempf[xmid+1, ymid, zmid] += 1/self.alpha
                            Ztempf[xmid, ymid, zmid] -= 2/self.alpha
                            Zf[innerf, :] = np.ravel(Ztempf)[totalnnzf]
                            ef[0] = innerstrucidxe[filx+1, fily, filz+1]
                            ef[1] = innerstrucidxe[filx+1, fily+1, filz+1]
                            ef[2] = innerstrucidxe[filx+2, fily, filz+1]
                            ef[3] = innerstrucidxe[filx+2, fily+1, filz+1]
                            efnnz = np.nonzero(ef + 1)[0]
                            Aef[ef[efnnz], innerf] = 1/self.alpha*Asign[efnnz]
                            gf[0] = innerstrucidxg[filx+1, fily+1, filz]
                            gf[1] = innerstrucidxg[filx+1, fily+1, filz+1]
                            gf[2] = innerstrucidxg[filx+2, fily+1, filz]
                            gf[3] = innerstrucidxg[filx+2, fily+1, filz+1]
                            gfnnz = np.nonzero(gf + 1)[0]
                            Agf[gf[gfnnz], innerf] = 1/self.alpha*Asign[gfnnz]
                            innerf += 1
                        if filg > 0:
                            Ztempg = strucg*self.jomega * coeffg[xstart:xstop,
                                                                 ystart:ystop,
                                                                 zstart:zstop]
                            Ztempg[xmid, ymid, zmid] += self.g.r
                            Ztempg[xmid, ymid, zmid] -= 2/self.alpha
                            Ztempg[xmid, ymid, zmid-1] += 1/self.alpha
                            Ztempg[xmid, ymid, zmid+1] += 1/self.alpha
                            Zg[innerg, :] = np.ravel(Ztempg)[totalnnzg]
                            if filx == -1:
                                raise IndexError('Error: filx = -1')
                            if fily == -1:
                                raise IndexError('Error: fily = -1')
                            if filz == -1:
                                raise IndexError('Error: filz = -1')
                            eg[0] = innerstrucidxe[filx+1, fily, filz+1]
                            eg[1] = innerstrucidxe[filx+1, fily, filz+2]
                            eg[2] = innerstrucidxe[filx+1, fily+1, filz+1]
                            eg[3] = innerstrucidxe[filx+1, fily+1, filz+2]
                            egnnz = np.nonzero(eg + 1)[0]
                            Aeg[eg[egnnz], innerg] = 1/self.alpha*Asign[egnnz]
                            fg[0] = innerstrucidxf[filx, fily+1, filz+1]
                            fg[1] = innerstrucidxf[filx, fily+1, filz+2]
                            fg[2] = innerstrucidxf[filx+1, fily+1, filz+1]
                            fg[3] = innerstrucidxf[filx+1, fily+1, filz+2]
                            fgnnz = np.nonzero(fg + 1)[0]
                            Afg[fg[fgnnz], innerg] = 1/self.alpha*Asign[fgnnz]
                            innerg += 1
            gidxe = self.e.idx0[group]
            gidxf = self.f.idx0[group]
            gidxg = self.g.idx0[group]
            totalsize = totalsizee + totalsizef + totalsizeg
            innersize = innersizee + innersizef + innersizeg
            e = np.zeros((totalsize,), dtype=np.complex128)
            innere = 0
            innerf = 0
            innerg = 0
            Ze = Ze.T
            Zf = Zf.T
            Zg = Zg.T
            for filx in range(m[0]):
                fx = filx + 1
                fox = filx + m[0]
                for fily in range(m[1]):
                    fy = fily + 1
                    foy = fily + m[1]
                    for filz in range(m[2]):
                        fz = filz + 1
                        foz = filz + m[2]
                        if innerstruce[fx, fy, fz] > 0:
                            i7 = np.zeros((7,), dtype=int)
                            i4f = np.zeros((4,), dtype=int)
                            i4g = np.zeros((4,), dtype=int)
                            i7[0] = innerstrucidxe[fx-1, fy, fz]
                            i7[1] = innerstrucidxe[fx, fy-1, fz]
                            i7[2] = innerstrucidxe[fx, fy, fz-1]
                            i7[3] = innerstrucidxe[fx, fy, fz]
                            i7[4] = innerstrucidxe[fx, fy, fz+1]
                            i7[5] = innerstrucidxe[fx, fy+1, fz]
                            i7[6] = innerstrucidxe[fx+1, fy, fz]
                            i4f[0] = innerstrucidxf[fx-1, fy, fz]
                            i4f[1] = innerstrucidxf[fx-1, fy+1, fz]
                            i4f[2] = innerstrucidxf[fx, fy, fz]
                            i4f[3] = innerstrucidxf[fx, fy+1, fz]
                            i4g[0] = innerstrucidxg[fx, fy, fz-1]
                            i4g[1] = innerstrucidxg[fx, fy, fz]
                            i4g[2] = innerstrucidxg[fx, fy+1, fz-1]
                            i4g[3] = innerstrucidxg[fx, fy+1, fz]
                            o3 = strucidxe[fox, foy, foz]
                            i7nnz = np.nonzero(i7+1)[0]
                            i4fnnz = np.nonzero(i4f+1)[0]
                            i4gnnz = np.nonzero(i4g+1)[0]
                            i7 = i7[i7nnz]
                            i4f = i4f[i4fnnz]
                            i4g = i4g[i4gnnz]
                            # Zeview = np.hstack((Ze, Afe, Age))[i7, :]
                            # Zfview = np.hstack((Aef, Zf, Agf))[i4f, :]
                            # Zgview = np.hstack((Aeg, Afg, Zg))[i4g, :]
                            # Zview = np.vstack((Zeview, Zfview, Zgview))
                            #       'Aeg: ', np.shape(Aeg))
                            #       'Afg: ', np.shape(Afg))
                            #       'Zg: ', np.shape(Zg))
                            #       np.shape(np.hstack((Ze, Aef, Aeg))))
                            Zeview = np.vstack((Ze, Afe, Age))[:, i7]
                            Zfview = np.vstack((Aef, Zf, Agf))[:, i4f]
                            Zgview = np.vstack((Aeg, Afg, Zg))[:, i4g]
                            Zview = np.hstack((Zeview, Zfview, Zgview))
                            # Zeview = np.hstack((Ze, Aef, Aeg))[i7, :]
                            # Zfview = np.hstack((Afe, Zf, Afg))[i4f, :]
                            # Zgview = np.hstack((Age, Agf, Zg))[i4g, :]
                            # Zview = np.vstack((Zeview, Zfview, Zgview))
                            e[:] = 0
                            e[o3] = 1
                            #       'Zfview: ', np.shape(Zfview),
                            #       'Zgview: ', np.shape(Zgview))
                            ZinvCol = lsqr(Zview, e, iter_lim=3)
                            i15nnz = np.r_[i7nnz, i4fnnz+7, i4gnnz+11]
                            if innere == 0:
                                self.Zeview0 = np.copy(Zeview)
                                self.Zfview0 = np.copy(Zfview)
                                self.Zgview0 = np.copy(Zgview)
                            Zeinv[i15nnz, gidxe + innere] = ZinvCol[0]
                            # Zeinv[i15nnz, gidxe + innere] = Zview[:, innere]
                            # Zeinv[i15nnz, gidxe + innere] = \
                            #     np.r_[:np.size(i15nnz)] + 1
                            innere += 1
                            print(innere)
                        if innerstrucf[fx, fy, fz] > 0:
                            i7 = np.zeros((7,), dtype=int)
                            i4e = np.zeros((4,), dtype=int)
                            i4g = np.zeros((4,), dtype=int)
                            i7[0] = innerstrucidxf[fx-1, fy, fz]
                            i7[1] = innerstrucidxf[fx, fy-1, fz]
                            i7[2] = innerstrucidxf[fx, fy, fz-1]
                            i7[3] = innerstrucidxf[fx, fy, fz]
                            i7[4] = innerstrucidxf[fx, fy, fz+1]
                            i7[5] = innerstrucidxf[fx, fy+1, fz]
                            i7[6] = innerstrucidxf[fx+1, fy, fz]
                            i4e[0] = innerstrucidxe[fx, fy-1, fz]
                            i4e[1] = innerstrucidxe[fx, fy, fz]
                            i4e[2] = innerstrucidxe[fx+1, fy-1, fz]
                            i4e[3] = innerstrucidxe[fx+1, fy, fz]
                            i4g[0] = innerstrucidxg[fx, fy, fz-1]
                            i4g[1] = innerstrucidxg[fx, fy, fz]
                            i4g[2] = innerstrucidxg[fx+1, fy, fz-1]
                            i4g[3] = innerstrucidxg[fx+1, fy, fz]
                            o3 = strucidxf[fox, foy, foz]
                            i4ennz = np.nonzero(i4e+1)[0]
                            i7nnz = np.nonzero(i7+1)[0]
                            i4gnnz = np.nonzero(i4g+1)[0]
                            i7 = i7[i7nnz]
                            i4e = i4e[i4ennz]
                            i4g = i4g[i4gnnz]
                            # Zeview = np.hstack((Ze, Afe, Age))[i4e, :]
                            # Zfview = np.hstack((Aef, Zf, Agf))[i7, :]
                            # Zgview = np.hstack((Aeg, Afg, Zg))[i4g, :]
                            # Zview = np.vstack((Zeview, Zfview, Zgview))
                            Zeview = np.vstack((Ze, Afe, Age))[:, i4e]
                            Zfview = np.vstack((Aef, Zf, Agf))[:, i7]
                            Zgview = np.vstack((Aeg, Afg, Zg))[:, i4g]
                            Zview = np.hstack((Zeview, Zfview, Zgview))
                            # Zeview = np.hstack((Ze, Aef, Aeg))[i4e, :]
                            # Zfview = np.hstack((Afe, Zf, Afg))[i7, :]
                            # Zgview = np.hstack((Age, Agf, Zg))[i4g, :]
                            # Zview = np.vstack((Zeview, Zfview, Zgview))
                            e[:] = 0
                            e[o3+totalsizee] = 1
                            ZinvCol = lsqr(Zview, e, iter_lim=3)
                            i15nnz = np.r_[i4ennz, i7nnz+4, i4gnnz+11]
                            # Zfinv[i15nnz, gidxf + innerf] = Zview[:, innerf+48]
                            Zfinv[i15nnz, gidxf + innerf] = ZinvCol[0]
                            # Zfinv[i15nnz, gidxf + innerf] = \
                            #     np.r_[:np.size(i15nnz)] + 1
                            innerf += 1
                        if innerstrucg[fx, fy, fz] > 0:
                            i7 = np.zeros((7,), dtype=int)
                            i4e = np.zeros((4,), dtype=int)
                            i4f = np.zeros((4,), dtype=int)
                            i7[0] = innerstrucidxg[fx-1, fy, fz]
                            i7[1] = innerstrucidxg[fx, fy-1, fz]
                            i7[2] = innerstrucidxg[fx, fy, fz-1]
                            i7[3] = innerstrucidxg[fx, fy, fz]
                            i7[4] = innerstrucidxg[fx, fy, fz+1]
                            i7[5] = innerstrucidxg[fx, fy+1, fz]
                            i7[6] = innerstrucidxg[fx+1, fy, fz]
                            i4e[0] = innerstrucidxe[fx, fy-1, fz]
                            i4e[1] = innerstrucidxe[fx, fy-1, fz+1]
                            i4e[2] = innerstrucidxe[fx, fy, fz]
                            i4e[3] = innerstrucidxe[fx, fy, fz+1]
                            i4f[0] = innerstrucidxf[fx-1, fy, fz]
                            i4f[1] = innerstrucidxf[fx-1, fy, fz+1]
                            i4f[2] = innerstrucidxf[fx, fy, fz]
                            i4f[3] = innerstrucidxf[fx, fy, fz+1]
                            o3 = strucidxg[fox, foy, foz]
                            i4ennz = np.nonzero(i4e+1)[0]
                            i4fnnz = np.nonzero(i4f+1)[0]
                            i7nnz = np.nonzero(i7+1)[0]
                            i4e = i4e[i4ennz]
                            i4f = i4f[i4fnnz]
                            i7 = i7[i7nnz]
                            # Zeview = np.hstack((Ze, Afe, Age))[i4e, :]
                            # Zfview = np.hstack((Aef, Zf, Agf))[i4f, :]
                            # Zgview = np.hstack((Aeg, Afg, Zg))[i7, :]
                            # Zview = np.vstack((Zeview, Zfview, Zgview))
                            Zeview = np.vstack((Ze, Afe, Age))[:, i4e]
                            Zfview = np.vstack((Aef, Zf, Agf))[:, i4f]
                            Zgview = np.vstack((Aeg, Afg, Zg))[:, i7]
                            Zview = np.hstack((Zeview, Zfview, Zgview))
                            # Zeview = np.hstack((Ze, Aef, Aeg))[i4e, :]
                            # Zfview = np.hstack((Afe, Zf, Afg))[i4f, :]
                            # Zgview = np.hstack((Age, Agf, Zg))[i7, :]
                            # Zview = np.vstack((Zeview, Zfview, Zgview))
                            e[:] = 0
                            e[o3+totalsizee+totalsizef] = 1
                            ZinvCol = lsqr(Zview, e, iter_lim=3)
                            i15nnz = np.r_[i4ennz, i4fnnz+4, i7nnz+8]
                            # Zginv[i15nnz, gidxg + innerg] = Zview[:, innerg+96]
                            #     self.Zview0 = np.copy(Zview)
                            #     self.Zgview0 = np.copy(Zgview)
                            #     print(i15nnz)
                            Zginv[i15nnz, gidxg + innerg] = ZinvCol[0]
                            # Zginv[i15nnz, gidxg + innerg] = \
                            #     np.r_[:np.size(i15nnz)] + 1
                            innerg += 1
        self.Zeinv = Zeinv
        self.Zfinv = Zfinv
        self.Zginv = Zginv
        self.Afe = Afe

    def spaiapply(self, vec):
        """Apply the coupled 15-point SPAI stencil (gather variant, dormant).

        Multiplies the stacked filament vector ``[e; f; g]`` by the sparse
        approximate inverse built in :meth:`spaiinit`. For each leaf group it
        scatters the group and its 27 neighbours into padded 3n workspaces,
        then for every filament gathers the 15 stencil contributions: the 7
        same-orientation taps (rows ``offset..offset+6`` of the orientation's
        ``Zinv``) followed by the cross-orientation taps applied via shifted
        window products. Returns the preconditioned vector, concatenated back
        as ``[e; f; g]``. Part of the dormant RDF path.
        """
        ne = self.e.n
        nf = self.f.n
        ng = self.g.n
        sizee = self.e.idx0[-1]
        sizef = self.f.idx0[-1]
        vece = vec[:sizee]
        vecf = vec[sizee:sizee+sizef]
        vecg = vec[sizee+sizef:]
        singlegroupe = np.zeros((3*ne[0], 3*ne[1], 3*ne[2]), dtype=vec.dtype)
        singlegroupf = np.zeros((3*nf[0], 3*nf[1], 3*nf[2]), dtype=vec.dtype)
        singlegroupg = np.zeros((3*ng[0], 3*ng[1], 3*ng[2]), dtype=vec.dtype)
        singleZinve = np.zeros((15, ne[0], ne[1], ne[2]), dtype=vec.dtype)
        singleZinvf = np.zeros((15, nf[0], nf[1], nf[2]), dtype=vec.dtype)
        singleZinvg = np.zeros((15, ng[0], ng[1], ng[2]), dtype=vec.dtype)
        outmate = np.zeros_like(vece)
        outmatf = np.zeros_like(vecf)
        outmatg = np.zeros_like(vecg)
        outfulle = np.zeros((ne[0], ne[1], ne[2]), dtype=vec.dtype)
        outfullf = np.zeros((nf[0], nf[1], nf[2]), dtype=vec.dtype)
        outfullg = np.zeros((ng[0], ng[1], ng[2]), dtype=vec.dtype)
        for group in range(np.size(self.lv[0].idx0) - 1):
            singlegroupe[...] = 0
            singlegroupf[...] = 0
            singlegroupg[...] = 0
            for neigh in range(27):
                neighx = neigh // 9
                neighy = (neigh//3) % 3
                neighz = neigh % 3
                neighgroup = self.lv[0].neighbors[neigh, group]
                if neighgroup >= 0:
                    for orient, singlegroup, vec in \
                        zip([self.e, self.f, self.g],
                            [singlegroupe, singlegroupf, singlegroupg],
                            [vece, vecf, vecg]):
                        nn = orient.n
                        nidx = np.s_[orient.idx0[neighgroup]:
                                     orient.idx0[neighgroup+1]]
                        xidx = neighx*nn[0] + orient.idx[nidx]//(nn[1]*nn[2])
                        yidx = neighy*nn[1] + (orient.idx[nidx]//nn[2]) % nn[1]
                        zidx = neighz*nn[2] + orient.idx[nidx] % nn[2]
                        singlegroup[xidx, yidx, zidx] = vec[nidx]
                        #     print(nidx, neighx, nn[0], nn[1]*nn[2],
                        #           max(orient.idx[nidx]), max(xidx))
            for orient, singlegroup, singleZinv, Zinv, outmat in \
                zip([self.e, self.f, self.g],
                    [singlegroupe, singlegroupf, singlegroupg],
                    [singleZinve, singleZinvf, singleZinvg],
                    [self.Zeinv, self.Zfinv, self.Zginv],
                    [outmate, outmatf, outmatg]):
                nn = orient.n
                gidx = np.s_[orient.idx0[group]:orient.idx0[group+1]]
                fidx = orient.idx[gidx]
                xidx = fidx // (nn[1]*nn[2])
                yidx = (fidx // nn[2]) % nn[1]
                zidx = fidx % nn[2]
                singleZinv[...] = 0
                singleZinv[:, xidx, yidx, zidx] = Zinv[:, gidx]
                if orient.orientation == 'e':
                    offset = 0
                elif orient.orientation == 'f':
                    offset = 4
                elif orient.orientation == 'g':
                    offset = 8
                outfull = singlegroup[nn[0]+1:2*nn[0]+1,
                                      nn[1]:2*nn[1],
                                      nn[2]:2*nn[2]] * \
                    singleZinv[offset, ...]
                # outmat[gidx] = np.ravel(outfull)[fidx]
                outfull += singlegroup[nn[0]:2*nn[0],
                                       nn[1]+1:2*nn[1]+1,
                                       nn[2]:2*nn[2]] * \
                    singleZinv[offset+1, ...]
                # outmat[gidx] += np.ravel(outfull)[fidx]
                outfull += singlegroup[nn[0]:2*nn[0],
                                       nn[1]:2*nn[1],
                                       nn[2]+1:2*nn[2]+1] * \
                    singleZinv[offset+2, ...]
                # outmat[gidx] += np.ravel(outfull)[fidx]
                outfull += singlegroup[nn[0]:2*nn[0],
                                       nn[1]:2*nn[1],
                                       nn[2]:2*nn[2]] * \
                    singleZinv[offset+3, ...]
                # outmat[gidx] += np.ravel(outfull)[fidx]
                outfull += singlegroup[nn[0]:2*nn[0],
                                       nn[1]:2*nn[1],
                                       nn[2]-1:2*nn[2]-1] * \
                    singleZinv[offset+4, ...]
                # outmat[gidx] += np.ravel(outfull)[fidx]
                outfull += singlegroup[nn[0]:2*nn[0],
                                       nn[1]-1:2*nn[1]-1,
                                       nn[2]:2*nn[2]] * \
                    singleZinv[offset+5, ...]
                # outmat[gidx] += np.ravel(outfull)[fidx]
                outfull += singlegroup[nn[0]-1:2*nn[0]-1,
                                       nn[1]:2*nn[1],
                                       nn[2]:2*nn[2]] * \
                    singleZinv[offset+6, ...]
                outmat[gidx] += np.ravel(outfull)[fidx]
            nn = self.lv[0].n
            gidxe = np.s_[self.e.idx0[group]:self.e.idx0[group+1]]
            gidxf = np.s_[self.f.idx0[group]:self.f.idx0[group+1]]
            gidxg = np.s_[self.g.idx0[group]:self.g.idx0[group+1]]
            fidxe = self.e.idx[gidxe]
            fidxf = self.f.idx[gidxf]
            fidxg = self.g.idx[gidxg]
            outfullf[...] = 0
            outfull = singlegroupe[ne[0]+1:2*ne[0]+1,
                                   ne[1]:2*ne[1],
                                   ne[2]:2*ne[2]] * singleZinve[7, ...]
            outfullf[:, :ne[1], :] += outfull[:nf[0], :, :]
            outfull = singlegroupe[ne[0]+1:2*ne[0]+1,
                                   ne[1]:2*ne[1],
                                   ne[2]:2*ne[2]] * singleZinve[8, ...]
            outfullf[:, nn[1]-ne[1]:, :] += outfull[:nf[0], :, :]
            outfull = singlegroupe[ne[0]:2*ne[0],
                                   ne[1]:2*ne[1],
                                   ne[2]:2*ne[2]] * singleZinve[9, ...]
            outfullf[:, :ne[1], :] += outfull[:nf[0], :, :]
            outfull = singlegroupe[ne[0]:2*ne[0],
                                   ne[1]:2*ne[1],
                                   ne[2]:2*ne[2]] * singleZinve[10, ...]
            outfullf[:, nn[1]-ne[1]:, :] += outfull[:nf[0], :, :]
            outmatf[gidxf] += np.ravel(outfullf)[fidxf]
            outfullg[...] = 0
            outfull = singlegroupe[ne[0]:2*ne[0],
                                   ne[1]:2*ne[1],
                                   ne[2]+1:2*ne[2]+1] * singleZinve[11, ...]
            outfullg[:, :ne[1], :] += outfull[:, :, :ng[2]]
            outfull = singlegroupe[ne[0]:2*ne[0],
                                   ne[1]:2*ne[1],
                                   ne[2]:2*ne[2]] * singleZinve[12, ...]
            outfullg[:, :ne[1], :] += outfull[:, :, :ng[2]]
            outfull = singlegroupe[ne[0]:2*ne[0],
                                   ne[1]:2*ne[1],
                                   ne[2]+1:2*ne[2]+1] * singleZinve[13, ...]
            outfullg[:, nn[1]-ne[1]:, :] += outfull[:, :, :ng[2]]
            outfull = singlegroupe[ne[0]:2*ne[0],
                                   ne[1]:2*ne[1],
                                   ne[2]:2*ne[2]] * singleZinve[14, ...]
            outfullg[:, nn[1]-ne[1]:, :] += outfull[:, :, :ng[2]]
            outmatg[gidxg] += np.ravel(outfullg)[fidxg]
            outfulle[...] = 0
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]+1:2*nf[1]+1,
                                   nf[2]:2*nf[2]] * singleZinvf[0, ...]
            outfulle[:nf[0], :, :] += outfull[:, :ne[1], :]
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]:2*nf[1],
                                   nf[2]:2*nf[2]] * singleZinvf[1, ...]
            outfulle[:nf[0], :, :] += outfull[:, :ne[1], :]
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]+1:2*nf[1]+1,
                                   nf[2]:2*nf[2]] * singleZinvf[2, ...]
            outfulle[nn[0]-nf[0]:, :, :] += outfull[:, :ne[1], :]
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]:2*nf[1],
                                   nf[2]:2*nf[2]] * singleZinvf[3, ...]
            outfulle[nn[0]-nf[0]:, :, :] += outfull[:, :ne[1], :]
            outmate[gidxe] += np.ravel(outfulle)[fidxe]
            outfullg[...] = 0
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]:2*nf[1],
                                   nf[2]+1:2*nf[2]+1] * singleZinvf[11, ...]
            outfullg[:nf[0], :, :] += outfull[:, :, :ng[2]]
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]:2*nf[1],
                                   nf[2]:2*nf[2]] * singleZinvf[12, ...]
            outfullg[:nf[0], :, :] += outfull[:, :, :ng[2]]
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]:2*nf[1],
                                   nf[2]+1:2*nf[2]+1] * singleZinvf[13, ...]
            outfullg[nn[0]-nf[0]:, :, :] += outfull[:, :, :ng[2]]
            outfull = singlegroupf[nf[0]:2*nf[0],
                                   nf[1]:2*nf[1],
                                   nf[2]:2*nf[2]] * singleZinvf[14, ...]
            outfullg[nn[0]-nf[0]:, :, :] += outfull[:, :, :ng[2]]
            outmatg[gidxg] += np.ravel(outfullg)[fidxg]
            outfulle[...] = 0
            outfull = singlegroupg[ng[0]:2*ng[0],
                                   ng[1]+1:2*ng[1]+1,
                                   ng[2]:2*ng[2]] * singleZinvg[0, ...]
            outfulle[:, :, :ng[2]] += outfull[:, :ne[1], :]
            outfull = singlegroupg[ng[0]:2*ng[0],
                                   ng[1]:2*ng[1],
                                   ng[2]:2*ng[2]] * singleZinvg[1, ...]
            outfulle[:, :, :ng[2]] += outfull[:, :ne[1], :]
            outfull = singlegroupg[ng[0]:2*ng[0],
                                   ng[1]+1:2*ng[1]+1,
                                   ng[2]:2*ng[2]] * singleZinvg[2, ...]
            outfulle[:, :, nn[2]-ng[2]:] += outfull[:, :ne[1], :]
            outfull = singlegroupg[ng[0]:2*ng[0],
                                   ng[1]:2*ng[1],
                                   ng[2]:2*ng[2]] * singleZinvg[3, ...]
            outfulle[:, :, nn[2]-ng[2]:] += outfull[:, :ne[1], :]
            outmate[gidxe] += np.ravel(outfulle)[fidxe]
            outfullf[...] = 0
            outfull = singlegroupg[ng[0]+1:2*ng[0]+1,
                                   ng[1]:2*ng[1],
                                   ng[2]:2*ng[2]] * singleZinvg[4, ...]
            outfullf[:, :, :ng[2]] += outfull[:nf[0], :, :]
            outfull = singlegroupg[ng[0]+1:2*ng[0]+1,
                                   ng[1]:2*ng[1],
                                   ng[2]:2*ng[2]] * singleZinvg[5, ...]
            outfullf[:, :, nn[2]-ng[2]:] += outfull[:nf[0], :, :]
            outfull = singlegroupg[ng[0]:2*ng[0],
                                   ng[1]:2*ng[1],
                                   ng[2]:2*ng[2]] * singleZinvg[6, ...]
            outfullf[:, :, :ng[2]] += outfull[:nf[0], :, :]
            outfull = singlegroupg[ng[0]:2*ng[0],
                                   ng[1]:2*ng[1],
                                   ng[2]:2*ng[2]] * singleZinvg[7, ...]
            outfullf[:, :, nn[2]-ng[2]:] += outfull[:nf[0], :, :]
            outmatf[gidxf] += np.ravel(outfullf)[fidxf]
        return np.hstack((outmate, outmatf, outmatg))

    def spaiapply2(self, vec):
        """Apply the coupled 15-point SPAI stencil (scatter variant, dormant).

        Same operator as :meth:`spaiapply` but organized as a scatter: each
        source filament's value, weighted by its 15 ``Zinv`` rows, is added
        into padded 3n output volumes at the stencil offsets given by the
        ``offx*/offy*/offz*`` tables (rows split per target orientation by
        ``ranges``), and the 27-neighbour loop then gathers each group's
        window back into the compressed output. Returns the preconditioned
        ``[e; f; g]`` vector. Part of the dormant RDF path.
        """
        ne = self.e.n
        nf = self.f.n
        ng = self.g.n
        sizee = self.e.idx0[-1]
        sizef = self.f.idx0[-1]
        vece = vec[:sizee]
        vecf = vec[sizee:sizee+sizef]
        vecg = vec[sizee+sizef:]
        singlegroupe = np.zeros((ne[0], ne[1], ne[2]), dtype=vec.dtype)
        singlegroupf = np.zeros((nf[0], nf[1], nf[2]), dtype=vec.dtype)
        singlegroupg = np.zeros((ng[0], ng[1], ng[2]), dtype=vec.dtype)
        outmate = np.zeros_like(vece)
        outmatf = np.zeros_like(vecf)
        outmatg = np.zeros_like(vecg)
        outfulle = np.zeros((3*ne[0], 3*ne[1], 3*ne[2]), dtype=vec.dtype)
        outfullf = np.zeros((3*nf[0], 3*nf[1], 3*nf[2]), dtype=vec.dtype)
        outfullg = np.zeros((3*ng[0], 3*ng[1], 3*ng[2]), dtype=vec.dtype)
        offxe = np.array([-1, 0, 0, 0, 0, 0, 1, -2, -2, -1, -1, 0, 0, 0, 0])
        offye = np.array([0, -1, 0, 0, 0, 1, 0, 1, 2, 1, 2, 1, 1, 2, 2])
        offze = np.array([0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, -2, -1, -2, -1])
        offxf = np.array([1, 1, 2, 2, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        offyf = np.array([-2, -1, -2, -1, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        offzf = np.array([0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, -2, -1, -2, -1])
        offxg = np.array([0, 0, 0, 0, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 1])
        offyg = np.array([-2, -2, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0])
        offzg = np.array([1, 2, 1, 2, 1, 2, 1, 2, 0, 0, -1, 0, 1, 0, 0])
        for group in range(np.size(self.lv[0].idx0) - 1):
            singlegroupe[...] = 0
            singlegroupf[...] = 0
            singlegroupg[...] = 0
            for orient, singlegroup, vec, Zinv in \
                    zip([self.e, self.f, self.g],
                        [singlegroupe, singlegroupf, singlegroupg],
                        [vece, vecf, vecg],
                        [self.Zeinv, self.Zfinv, self.Zginv]):
                gidx = np.s_[orient.idx0[group]:orient.idx0[group+1]]
                fidx = orient.idx[gidx]
                nn = orient.n
                xidx = fidx//(nn[1]*nn[2])
                yidx = (fidx//nn[2]) % nn[1]
                zidx = fidx % nn[2]
                # singlegroup[xidx, yidx, zidx] = vec[fidx]
                singlegroup[xidx, yidx, zidx] = vec[gidx]
                singleZinv = np.zeros((15, nn[0], nn[1], nn[2]),
                                      dtype=Zinv.dtype)
                singleZinv[:, xidx, yidx, zidx] = Zinv[:, gidx]
                if orient.orientation == 'e':
                    offx = offxe
                    offy = offye
                    offz = offze
                    ranges = np.array([7, 11])
                elif orient.orientation == 'f':
                    offx = offxf
                    offy = offyf
                    offz = offzf
                    ranges = np.array([4, 11])
                elif orient.orientation == 'g':
                    offx = offxg
                    offy = offyg
                    offz = offzg
                    ranges = np.array([4, 8])
                for ii in range(ranges[0]):
                    outfulle[nn[0]+offx[ii]:2*nn[0]+offx[ii],
                             nn[1]+offy[ii]:2*nn[1]+offy[ii],
                             nn[2]+offz[ii]:2*nn[2]+offz[ii]] \
                        += singleZinv[ii, ...] * singlegroup
                for ii in range(ranges[0], ranges[1]):
                    outfullf[nn[0]+offx[ii]:2*nn[0]+offx[ii],
                             nn[1]+offy[ii]:2*nn[1]+offy[ii],
                             nn[2]+offz[ii]:2*nn[2]+offz[ii]] \
                        += singleZinv[ii, ...] * singlegroup
                for ii in range(ranges[1], 15):
                    outfullg[nn[0]+offx[ii]:2*nn[0]+offx[ii],
                             nn[1]+offy[ii]:2*nn[1]+offy[ii],
                             nn[2]+offz[ii]:2*nn[2]+offz[ii]] \
                        += singleZinv[ii, ...] * singlegroup
            for neigh in range(27):
                neighgroup = self.lv[0].neighbors[neigh, group]
                if neighgroup >= 0:
                    neighx = neigh // 9
                    neighy = (neigh//3) % 3
                    neighz = neigh % 3
                    for orient, outfull, outmat in \
                        zip([self.e, self.f, self.g],
                            [outfulle, outfullf, outfullg],
                            [outmate, outmatf, outmatg]):
                        gidx = np.s_[orient.idx0[neighgroup]:
                                     orient.idx0[neighgroup+1]]
                        fidx = orient.idx[gidx]
                        nn = orient.n
                        xidx = np.s_[neighx*nn[0]:(neighx+1)*nn[0]]
                        yidx = np.s_[neighy*nn[1]:(neighy+1)*nn[1]]
                        zidx = np.s_[neighz*nn[2]:(neighz+1)*nn[2]]
                        outflat = np.ravel(outfull[xidx, yidx, zidx])
                        outmat[gidx, ...] += outflat[fidx]
        return np.hstack((outmate, outmatf, outmatg))

    def RDFinit(self):
        """Distribute the RDF scalar parameters to the leaves.

        Copies ``jomega`` and ``alpha`` from the tree to the e/f/g leaves and
        the node level, and sets ``beta = 1.0`` everywhere (the continuity
        row/column scaling used by the ``connect*``/``node2*`` incidence
        operators). The per-leaf ``spaiinit()`` calls that would actually
        build the SPAI preconditioner are commented out, so in the current
        code this only sets scalars -- it is called by main.py, but the RDF
        preconditioner itself stays dormant.
        """
        self.e.jomega = self.jomega
        self.f.jomega = self.jomega
        self.g.jomega = self.jomega
        self.e.alpha = self.alpha
        self.f.alpha = self.alpha
        self.g.alpha = self.alpha
        self.lv[0].alpha = self.alpha
        self.e.beta = 1e-0
        self.f.beta = 1e-0
        self.g.beta = 1e-0
        self.lv[0].beta = 1e-0

    def RDFapply(self, vec):
        """Apply the RDF preconditioner to a stacked ``[e; f; g; node]``
        vector (dormant).

        Block-triangular application of the per-leaf 7-point SPAI inverses
        (:meth:`leaf_induct.LeafInduct.spaiapply`) to the saddle-point vector
        of the divergence-free-constrained LpR system: the three filament
        blocks are preconditioned in sequence with cross-couplings routed
        through the node block via the ``node2*``/``*2node`` incidence maps
        and the ``-1/alpha`` continuity weights, and the node output is the
        accumulated continuity residual. Requires the leaves' ``Zinv`` (i.e.
        their ``spaiinit``) -- which is never built in the current code, so
        this path would fail if invoked; dormant.
        """
        esize = np.size(self.e.data)
        efsize = esize + np.size(self.f.data)
        efgsize = efsize + np.size(self.g.data)
        in1 = vec[:esize]
        in2 = vec[esize:efsize]
        in3 = vec[efsize:efgsize]
        in4 = vec[efgsize:]
        # eout
        in1 = self.e.spaiapply(in1)
        in4copy = np.copy(in4)
        in4 = self.node2e(in4)
        in4 = -1/self.e.alpha*self.e.spaiapply(in4)
        eout = in1 + in4
        # fout
        in1 = self.e2node(in1)
        in1copy = np.copy(in1)
        in1 = self.f.spaiapply(self.node2f(in1))
        in2 = self.f.spaiapply(in2)
        in4 = in4copy + self.e2node(in4)
        in4copy = np.copy(in4)
        in4 = -1/self.f.alpha*self.f.spaiapply(self.node2f(in4))
        fout = in1 + in2 + in4
        # gout
        in1 = in1copy + self.f2node(in1)
        in1copy = np.copy(in1)
        in1 = -1/self.g.alpha*self.g.spaiapply(self.node2g(in1))
        in2 = self.f2node(in2)
        in2copy = np.copy(in2)
        in2 = -1/self.g.alpha*self.g.spaiapply(self.node2g(in2))
        in3 = self.g.spaiapply(in3)
        in4 = in4copy + self.f2node(in4)
        in4copy = np.copy(in4)
        in4 = -1/self.g.alpha*self.g.spaiapply(self.node2g(in4))
        gout = in1 + in2 + in3 + in4
        # nodeout
        in1 = in1copy + self.g2node(in1)
        in2 = in2copy + self.g2node(in2)
        in3 = self.g2node(in3)
        in4 = in4copy + self.g2node(in4)
        nodeout = 1/self.lv[0].alpha*(in1 + in2 + in3 + in4)
        return np.concatenate([eout, fout, gout, nodeout])

    def RDFapply2(self, vec):
        """Apply the RDF preconditioner using the coupled tree-level SPAI
        (dormant).

        Variant of :meth:`RDFapply` built on the 15-point coupled inverse
        (:meth:`spaiapply2`): the node block is first folded into the
        filament blocks through the incidence maps and ``-1/alpha``, the
        combined ``[e; f; g]`` vector is preconditioned in one shot, and the
        node output is recovered from the incidence transpose. Requires
        :meth:`spaiinit` to have populated ``Zeinv/Zfinv/Zginv``; dormant.
        """
        esize = np.size(self.e.data)
        efsize = esize + np.size(self.f.data)
        efgsize = efsize + np.size(self.g.data)
        in1 = vec[:esize]
        in2 = vec[esize:efsize]
        in3 = vec[efsize:efgsize]
        in4 = vec[efgsize:]
        in4copy = np.copy(in4)
        in4 *= -1/self.lv[0].alpha
        in1 += self.node2e(in4)
        in2 += self.node2f(in4)
        in3 += self.node2g(in4)
        efgout = self.spaiapply2(np.concatenate([in1, in2, in3]))
        eout = efgout[:esize]
        fout = efgout[esize:efsize]
        gout = efgout[efsize:efgsize]
        nodeout = self.e2node(eout) + self.f2node(fout) + self.g2node(gout)
        nodeout -= in4copy
        nodeout *= -1/self.lv[0].alpha
        return np.concatenate([eout, fout, gout, nodeout])

    def RDFapply3(self, vec):
        """Apply an exact block (Schur-complement) preconditioner
        (non-functional).

        Textbook 2x2 block inverse of the saddle-point system using dense
        precomputed blocks: ``self.SchurDinv`` (inverse Schur complement of
        the filament block), ``self.Dinv``, ``self.AB`` and ``self.AC``.
        None of those attributes is ever constructed anywhere in the project
        (the assembly code in main.py is commented out), so calling this
        raises :class:`AttributeError`; kept as a record of the intended
        formulation.
        """
        esize = np.size(self.e.data)
        efsize = esize + np.size(self.f.data)
        efgsize = efsize + np.size(self.g.data)
        inefg = vec[:efgsize]
        innode = vec[efgsize:]
        invA = np.dot(self.SchurDinv, inefg)
        invC = -np.dot(np.dot(self.Dinv, self.AC), invA)
        invB = -np.dot(np.dot(np.dot(self.SchurDinv, self.AB), self.Dinv), innode)
        invD = np.dot(self.Dinv, innode - np.dot(self.AC, invB))
        return np.concatenate([invA + invB, invC + invD])

    def diaginverse(self):
        """Add the Jacobi (diagonal-inverse) scaled data to each leaf.

        In-place update ``data += data / (r + jw*L_self)`` for the e/f/g
        leaves -- the diagonal-inverse weighting used between iterations of
        the Neumann-series preconditioner (:func:`main.precondneumann`, which
        itself is currently unused by the driver).
        """
        for leaf in [self.e, self.f, self.g]:
            leaf.data += 1/(leaf.r + self.jomega*leaf.selfind)*leaf.data

    def traverseRL(self, neumann=False, extra=None):
        """Apply the inductive PEEC operator ``R + jw*Lp`` (the LpR matvec).

        The workhorse of the working solver: for each filament orientation
        (e, f, g) runs one full FMM sweep over ``leaf.data`` -- upward P2M /
        M2M, mid-level M2L, top-level FFT M2L (:meth:`TopLevel.m2lfortran`),
        downward L2L / L2P, plus the near-field Toeplitz P2P -- multiplies
        the result by ``jomega`` and adds the resistive drop ``r * I``. The
        result overwrites each leaf's ``data`` in place.

        With ``neumann=True`` the operator is modified for the Neumann-series
        preconditioner: the self-inductance diagonal contribution is
        subtracted after P2P and the resistive term is omitted, leaving only
        the off-diagonal part of the impedance.

        ``extra`` couples sources that are NOT on the filament lattice into
        the same sweep -- today the port terminal half-filaments of
        ``equiterminal``, which sit a quarter cell off the lattice and half
        a cell short. It is an object exposing ``p2m(leaf)``, ``p2p(leaf)``
        and ``l2p(leaf)``, called at the three points where the sweep can
        accept them:

          after ``leaf.p2m()``  add the extra sources' multipole moments
                                (AFTER, because ``p2m(accumulate=False)``
                                REALLOCATES ``above.data``)
          after ``leaf.p2p()``  their near field, both directions, direct
          after the downward pass
                                read the local expansion back at their
                                positions

        so the M2M/M2L/L2L ladder carries them with no changes at all: a
        multipole moment does not care whether its source sits on the
        lattice. That is the property a whole-domain FFT convolution does
        not have, and it is why a sub-cell terminal is affordable here.
        ``extra=None`` (the default) is bit-for-bit the original sweep.
        """
        # inductances:
        for leaf in [self.e, self.f, self.g]:
            if neumann is False:
                rvolt = leaf.r*leaf.data
            else:
                selfcontrib = leaf.selfind*leaf.data
            for l in range(1, self.numlevels):
                self.lv[l].data[:, :] = 0
            if self.numlevels > 1:
                leaf.p2m()
                if extra is not None:
                    extra.p2m(leaf)
            leaf.p2p()
            if extra is not None:
                extra.p2p(leaf)
            if neumann is True:
                leaf.data -= selfcontrib
            if self.numlevels > 1:
                for level in range(1, self.numlevels - 1):
                    self.lv[level].m2m()
                    self.lv[level].m2l()
                self.lv[self.numlevels - 1].m2lfortran()
                for level in range(self.numlevels - 2, 0, -1):
                    self.lv[level].l2l()
                # pre_l2p fires while ``leaf.above.data`` still holds the
                # LOCAL expansions and before the tree distributes them to
                # filaments -- the injection point for off-lattice FAR
                # sources (wirecoupler's P2L): coefficients added here ride
                # ``leaf.l2p()`` to every filament with the tree's own m0
                # scaling. The l2p hook below fires AFTER distribution, so
                # it can only read. hasattr-guarded: extras without the
                # hook (equiterminal's TerminalCoupler) are bit-for-bit
                # unaffected.
                if extra is not None and hasattr(extra, 'pre_l2p'):
                    extra.pre_l2p(leaf)
                leaf.l2p()
                if extra is not None:
                    extra.l2p(leaf)
            leaf.data[:] *= self.jomega
            if neumann is False:
                leaf.data[:] += rvolt

    def traverseP(self):
        """Apply the capacitive (coefficient-of-potential) operator
        (unfinished LpPR path).

        Node-charge to node-potential matvec: projects the node vector onto
        the three panel sets (Fortran ``node2panel``), runs the FMM sweep
        over the panel leaves (P2M, the rectangular three-orientation
        :meth:`leaf_poten.LeafPoten.p2p`, M2M/M2L/L2L/L2P for
        ``numlevels > 1``), and gathers the panel potentials back to nodes
        (Fortran ``panel2node``). Requires the capacitive machinery from
        ``__init__`` (``capacitive=True``); part of the unfinished LpPR path.
        """
        # coefficients of potential:
        if self.lv[0].data is None:
            self.lv[0].data = self.lv[0].struc.astype(np.complex128)
        for l in range(1, self.numlevels):
            sizelv1 = np.size(self.lv[l].idx)
            self.lv[l].data[:, :] = np.zeros((sizelv1, self.lv[l].nnmax),
                                             dtype=np.complex128)
        # The Fortran gather/scatter hard-codes the edge scheme's
        # corner-node <-> quarter-panel pairing. Under 'cell' a panel
        # is a whole face owned by the single cell behind it, which
        # node2panel already expresses as the sparse projections, so
        # route through those. They carry the same normalisation --
        # circulant_poten.py uses them and validate_circulant_poten.py
        # checks this sweep against it.
        self.px.data = self.node2px.dot(self.lv[0].data)
        self.py.data = self.node2py.dot(self.lv[0].data)
        self.pz.data = self.node2pz.dot(self.lv[0].data)
        if self.numlevels > 1:
            self.lv[1].data[...] = 0
            for leaf in [self.px, self.py, self.pz]:
                leaf.p2m()
        targetdatax = np.zeros_like(self.px.data)
        targetdatay = np.zeros_like(self.py.data)
        targetdataz = np.zeros_like(self.pz.data)
        for leaf in [self.px, self.py, self.pz]:
            leaf.p2p(self.px, self.py, self.pz,
                     targetdatax, targetdatay, targetdataz)
        #             targetdataz)
        self.px.data[...] = targetdatax
        self.py.data[...] = targetdatay
        self.pz.data[...] = targetdataz
        targetdatax = None
        targetdatay = None
        targetdataz = None
        if self.numlevels > 1:
            for level in range(1, self.numlevels - 1):
                self.lv[level].m2m()
                self.lv[level].m2l()
            self.lv[self.numlevels - 1].m2l()
            for level in range(self.numlevels - 2, 0, -1):
                self.lv[level].l2l()
            for leaf in [self.px, self.py, self.pz]:
                leaf.l2p()
        # transpose of the scatter above -- gather panel potentials
        # back to the node that owns each panel
        self.lv[0].data[:] = (self.node2px.T.dot(self.px.data)
                              + self.node2py.T.dot(self.py.data)
                              + self.node2pz.T.dot(self.pz.data))
            # leaf.data = None

    def traverseP3(self):
        """Capacitive matvec: near field from the assembled ``n2n`` matrix,
        far field (``numlevels > 1``) from one combined panel FMM sweep.

        Node-charge to node-potential map ``phi = P q``. The near field --
        panel pairs within the 27-neighbour leaf groups -- is applied
        directly as the precomputed sparse node-to-node matrix ``self.n2n``
        (built in ``__init__`` from the :meth:`leaf_poten.LeafPoten.p2pinit3`
        products, validated by ``validate_p2pinit3.py``; at ``numlevels == 1``
        it is the complete dense operator and no far field exists).

        For ``numlevels > 1`` the remaining (non-neighbour) interactions are
        the standard FMM far field over the panel leaves: scatter the node
        charges onto the panels (``node2px/py/pz``), one combined upward pass
        -- all three panel orientations share the single Coulomb kernel, so
        px runs a fresh :meth:`~levels.LeafLevel.p2m` and py/pz accumulate
        into the same parent moments (unlike the inductive e/f/g sweeps,
        which are independent) -- then the same level traversal as
        :meth:`traverseRL` (ascending ``m2m``/``m2l``, top-level FFT
        :meth:`~levels.TopLevel.m2lfortran`, descending ``l2l``), and
        :meth:`~levels.LeafLevel.l2p` back onto the panels, whose data is
        zeroed after P2M so it receives the far-field potential alone
        (interaction lists exclude the 27-neighbourhood, so near and far are
        complementary by construction). The panel potentials are gathered
        back to nodes with the transposed (equal-split, count-normalized)
        projections, matching the ``n2n`` assembly convention. The electric
        kernel constant ``1/(4*pi*eps0)`` enters through the panel leaves'
        L2P operator (``ynmr``, built by ``leafinit`` for orientations
        x/y/z). Validated against a single-level dense oracle by
        ``validate_traverseP3_farfield.py``.
        """
        if self.lv[0].data is None:
            self.lv[0].data = self.lv[0].struc.astype(np.complex128)
        if getattr(self, 'circpoten', None) is not None:
            # circulant single level: the whole-domain panel FFT applies the
            # COMPLETE operator (self.n2n holds only the near field, for the
            # preconditioning machinery -- not consumed here)
            self.lv[0].data[:] = self.circpoten.apply_nodes(self.lv[0].data)
            return
        near = None
        if self.numlevels > 1:
            # scatter node charges to panel charges (fresh, not accumulated)
            self.px.data = self.node2px.dot(self.lv[0].data)
            self.py.data = self.node2py.dot(self.lv[0].data)
            self.pz.data = self.node2pz.dot(self.lv[0].data)
            if getattr(self, 'fftnear', False):
                # 27-neighbour near field by Toeplitz/FFT convolution on
                # the panels, BEFORE p2m/l2p overwrite px/py/pz.data with
                # the far field. Replaces the sparse n2n node matvec below.
                near = [np.zeros_like(self.px.data),
                        np.zeros_like(self.py.data),
                        np.zeros_like(self.pz.data)]
                for leaf in (self.px, self.py, self.pz):
                    leaf.p2p(self.px, self.py, self.pz,
                             near[0], near[1], near[2])
            for l in range(1, self.numlevels):
                self.lv[l].data[:, :] = 0
            # combined upward pass: one Coulomb kernel, one set of moments
            self.px.p2m()
            self.py.p2m(accumulate=True)
            self.pz.p2m(accumulate=True)
            # panels now receive the far field only
            self.px.data[:] = 0
            self.py.data[:] = 0
            self.pz.data[:] = 0
            for level in range(1, self.numlevels - 1):
                self.lv[level].m2m()
                self.lv[level].m2l()
            self.lv[self.numlevels - 1].m2lfortran()
            for level in range(self.numlevels - 2, 0, -1):
                self.lv[level].l2l()
            self.px.l2p()
            self.py.l2p()
            self.pz.l2p()
        # In-place so views into a caller's backing array (SystemMat wires
        # lv[0].data as a slice of wholedata) stay bound. n2n consumes the
        # original charges -- the panel sweep above never touches lv[0].data.
        if near is None:
            self.lv[0].data[:] = self.n2n.dot(self.lv[0].data)
        else:
            self.lv[0].data[:] = 0
            self.px.data += near[0]
            self.py.data += near[1]
            self.pz.data += near[2]
        if self.numlevels > 1:
            self.lv[0].data += self.node2px.T.dot(self.px.data)
            self.lv[0].data += self.node2py.T.dot(self.py.data)
            self.lv[0].data += self.node2pz.T.dot(self.pz.data)

    def connectA(self):
        """Apply the node-to-filament incidence operator (discrete gradient).

        Maps the node potential vector ``lv[0].data`` to per-orientation
        filament voltage drops via the Fortran ``node2filament`` kernel --
        each filament receives the (signed) difference of its two end nodes.
        Returns the ``(e, f, g)`` triple scaled by ``beta``. Together with
        :meth:`connectAT` this is the KCL constraint block (the ``B``/``C``
        matrices of the saddle-point formulation) used by the LpR solve in
        main.py.
        """
        beta = self.lv[0].beta
        (e, f, g) = mp_fortran.node2filament(self.lv[0].data, self.lv[0].idx,
            self.lv[0].idx0, self.e.idx, self.e.idx0, self.f.idx, self.f.idx0,
            self.g.idx, self.g.idx0, self.lv[0].neighbors.T, self.e.n,
            self.f.n, self.g.n, self.lv[0].n)
        return (beta*e, beta*f, beta*g)

    def incidence(self):
        """Explicit sparse filament-to-node incidence ``A`` (efg x nn).

        The operator form is :meth:`connectA`; this materialises it, which
        is what any graph algorithm over the filament mesh needs (the loop
        basis, a spanning-tree particular solution, a nodal Laplacian).

        Built by the TWO-PROBE PARITY TRICK, not by ``nn`` one-hot probes.
        The node lattice is bipartite by coordinate parity
        (``x+y+z mod 2``) and every filament joins two ADJACENT nodes --
        exactly one of each parity. Probing once with the node keys
        (``index+1``) on the even-parity nodes and once on the odd ones
        therefore returns, per filament, ``+-beta*key(endpoint)``: both
        endpoints and both incidence signs decode exactly (integer keys are
        exact in float64, and ``sign(r)*beta`` reproduces the ``+-beta``
        entries bit-for-bit). O(efg) -- two Fortran sweeps plus a vector
        decode -- against O(nn*efg) for one-hot probing, which took 5.3 s
        at 23^3. Convention-proof: it measures the operator itself, so the
        beta scaling and sign conventions of ``node2filament`` carry over
        automatically. Verified against the one-hot build on five
        geometries; if any consistency check fails (it never should), the
        one-hot loop is the fallback.

        ``lv[0].data`` is restored on return. Requires ``beta``, i.e.
        ``RDFinit`` must have run.

        Raises
        ------
        RuntimeError
            If the decoded incidence does not have exactly one ``+beta``
            and one ``-beta`` per filament row -- which happens when
            ``connectA`` itself is invalid, e.g. a single-level EDGE tree
            built with ``N = NT``.
        """
        lv0 = self.lv[0]
        nn = np.size(lv0.idx)
        efg = np.size(self.e.idx) + np.size(self.f.idx) + np.size(self.g.idx)
        if self.numlevels == 1:
            dims = self.ntotal.astype(int)
            cx = (lv0.idx // (dims[1]*dims[2])).astype(np.int64)
            cy = ((lv0.idx // dims[2]) % dims[1]).astype(np.int64)
            cz = (lv0.idx % dims[2]).astype(np.int64)
        else:
            n0 = lv0.n.astype(int)
            cx = (lv0.idx // (n0[1]*n0[2])).astype(np.int64)
            cy = ((lv0.idx // n0[2]) % n0[1]).astype(np.int64)
            cz = (lv0.idx % n0[2]).astype(np.int64)
            for g in range(np.size(lv0.idx0) - 1):
                sl = np.s_[lv0.idx0[g]:lv0.idx0[g+1]]
                cx[sl] += lv0.xidx[g]*n0[0]
                cy[sl] += lv0.yidx[g]*n0[1]
                cz[sl] += lv0.zidx[g]*n0[2]
        par = (cx + cy + cz) % 2
        keys = np.arange(1, nn + 1, dtype=np.float64)
        saved = np.array(lv0.data)
        resp = []
        for p in (0, 1):
            lv0.data[:] = 0.0
            lv0.data[par == p] = keys[par == p]
            (ae, af, ag) = self.connectA()
            resp.append(np.real(np.concatenate([ae, af, ag])))
        lv0.data[:] = saved
        beta = float(np.real(lv0.beta))
        r_ev, r_od = resp
        n_ev = np.rint(np.abs(r_ev)/beta).astype(np.int64) - 1
        n_od = np.rint(np.abs(r_od)/beta).astype(np.int64) - 1
        ok = ((np.abs(np.abs(r_ev)/beta - (n_ev + 1)) < 1e-6).all()
              and (np.abs(np.abs(r_od)/beta - (n_od + 1)) < 1e-6).all()
              and (n_ev >= 0).all() and (n_ev < nn).all()
              and (n_od >= 0).all() and (n_od < nn).all()
              and (np.sign(r_ev)*np.sign(r_od) < 0).all())
        if ok:
            arows = np.repeat(np.arange(efg), 2)
            acols = np.column_stack([n_ev, n_od]).ravel()
            avals = np.column_stack([np.sign(r_ev)*beta,
                                     np.sign(r_od)*beta]).ravel()
            Ainc = coo_matrix((avals, (arows, acols)),
                              shape=(efg, nn)).tocsr()
        else:
            # fallback: the original one-hot probing (slow but
            # assumption-free)
            rows = []; cols = []; vals = []
            for k in range(nn):
                lv0.data[:] = 0.0
                lv0.data[k] = 1.0
                (ae, af, ag) = self.connectA()
                col = np.real(np.concatenate([ae, af, ag]))
                nz = np.nonzero(col)[0]
                rows.extend(nz.tolist())
                cols.extend([k]*nz.size)
                vals.extend(col[nz].tolist())
            lv0.data[:] = saved
            Ainc = coo_matrix((vals, (rows, cols)), shape=(efg, nn)).tocsr()
        # Incidence sanity: every filament row must hold exactly one +beta
        # and one -beta (sum zero). This FAILS when connectA itself is
        # invalid -- a single-level tree built with N = NT violates the
        # node2filament grid convention (like the known nleaf >= NT+1 rule
        # for the inductive p2p; measured: 217/583 malformed rows on a
        # 7x6x4 brick with N = NT, all perfect with N = NT+1) -- and a
        # garbage A would otherwise surface only as an inscrutable
        # singular S~.
        per_row = np.diff(Ainc.indptr)
        rowsum = np.asarray(Ainc.sum(axis=1)).ravel()
        if (per_row != 2).any() or np.abs(rowsum).max() > 1e-9*abs(beta):
            raise RuntimeError(
                "connectA produced an invalid incidence (%d rows without "
                "exactly one +beta/-beta pair). Under the EDGE scheme a "
                "single-level tree needs N >= NT+1 (the node2filament grid "
                "convention, like the nleaf >= NT+1 rule for the inductive "
                "p2p). Under the CELL scheme N = NT is correct and this "
                "check passes -- use stencils.single_level_nleaf() rather "
                "than hard-coding either."
                % int((per_row != 2).sum()))
        return Ainc

    def connectAT(self):
        """Apply the filament-to-node incidence transpose (discrete
        divergence).

        Adjoint of :meth:`connectA`: accumulates the three filament data
        vectors into net node currents (KCL sums) via the Fortran
        ``filament2node`` kernel, scaled by ``beta``. A zero result on every
        node means the filament currents are divergence-free.
        """
        beta = self.lv[0].beta
        return beta*mp_fortran.filament2node(self.e.data, self.f.data,
            self.g.data, self.lv[0].idx, self.lv[0].idx0, self.e.idx,
            self.e.idx0, self.f.idx, self.f.idx0, self.g.idx, self.g.idx0,
            self.lv[0].neighbors.T, self.e.n, self.f.n, self.g.n, self.lv[0].n)

    def adjmats(self):
        """Build the node adjacency matrix of the filament mesh graph.

        Uses the ``meshgraph_aux`` Fortran helpers to enumerate node
        adjacency and the filament-to-node incidence, and returns the sparse
        CSR node-by-node adjacency matrix (also caching its ``data``/
        ``indices`` on the tree). Used by the driver to construct the
        mesh/loop basis (the null space of the incidence operator) for the
        divergence-free reduction.
        """
        adjnode = meshgraph_aux.adjnode(self.lv[0].idx, self.lv[0].idx0,
                                        self.lv[0].neighbors.T, self.lv[0].n)
        node2fil = meshgraph_aux.fil2nodesparse(self.lv[0].idx,
            self.lv[0].idx0, self.e.idx, self.e.idx0, self.f.idx, self.f.idx0,
            self.g.idx, self.g.idx0, self.lv[0].neighbors.T,
            self.e.n, self.f.n, self.g.n, self.lv[0].n)
        adjnnz = np.count_nonzero(adjnode)
        nn = np.size(adjnode, axis=1)
        (adjdat, adjind, adjindptr) = meshgraph_aux.adjmat(adjnode, node2fil,
                                                           adjnnz)
        self.adjdat = adjdat.copy()
        self.adjind = adjind.copy()
        return csr_matrix((adjdat, adjind, adjindptr), shape=(nn, nn))

    def node2e(self, vec):
        """Node-to-filament incidence for the e (y-directed) leaf only.

        Single-orientation slice of :meth:`connectA`: maps the node vector
        `vec` to e-filament end-node differences, scaled by ``beta``.
        """
        beta = self.lv[0].beta
        return beta*mp_fortran.node2file(vec, self.lv[0].idx, self.lv[0].idx0,
                                         self.e.idx, self.e.idx0,
                                         self.lv[0].neighbors.T, self.e.n,
                                         self.lv[0].n)

    def node2f(self, vec):
        """Node-to-filament incidence for the f (x-directed) leaf; see
        :meth:`node2e`."""
        beta = self.lv[0].beta
        return beta*mp_fortran.node2filf(vec, self.lv[0].idx, self.lv[0].idx0,
                                         self.f.idx, self.f.idx0,
                                         self.lv[0].neighbors.T, self.f.n,
                                         self.lv[0].n)

    def node2g(self, vec):
        """Node-to-filament incidence for the g (z-directed) leaf; see
        :meth:`node2e`."""
        beta = self.lv[0].beta
        return beta*mp_fortran.node2filg(vec, self.lv[0].idx, self.lv[0].idx0,
                                         self.g.idx, self.g.idx0,
                                         self.lv[0].neighbors.T, self.g.n,
                                         self.lv[0].n)

    def e2node(self, vec):
        """Filament-to-node incidence transpose for the e leaf only.

        Single-orientation slice of :meth:`connectAT`: accumulates e-filament
        values into their end nodes (signed), scaled by ``beta``.
        """
        beta = self.lv[0].beta
        return beta*mp_fortran.file2node(vec, self.lv[0].idx, self.lv[0].idx0,
                                         self.e.idx, self.e.idx0,
                                         self.lv[0].neighbors.T, self.e.n,
                                         self.lv[0].n)

    def f2node(self, vec):
        """Filament-to-node incidence transpose for the f leaf; see
        :meth:`e2node`."""
        beta = self.lv[0].beta
        return beta*mp_fortran.filf2node(vec, self.lv[0].idx, self.lv[0].idx0,
                                         self.f.idx, self.f.idx0,
                                         self.lv[0].neighbors.T, self.f.n,
                                         self.lv[0].n)

    def g2node(self, vec):
        """Filament-to-node incidence transpose for the g leaf; see
        :meth:`e2node`."""
        beta = self.lv[0].beta
        return beta*mp_fortran.filg2node(vec, self.lv[0].idx, self.lv[0].idx0,
                                         self.g.idx, self.g.idx0,
                                         self.lv[0].neighbors.T, self.g.n,
                                         self.lv[0].n)

    def parsesource(self, sourcex, sourcey, sourcez, value, orientation):
        """Build a source (excitation) vector from global grid coordinates.

        Translates each global ``(x, y, z)`` element coordinate into its
        position in the compressed storage by descending the tree level by
        level (dividing out each level's box size and following the
        ``idx``/``idx0`` maps), then writes ``value[i]`` at that position in
        a fresh vector shaped like the chosen element set.

        Parameters
        ----------
        sourcex, sourcey, sourcez : array_like of int
            Global element coordinates (equal lengths).
        value : array_like
            Excitation values; node-orientation values are scaled by
            ``beta`` in place.
        orientation : {'e', 'f', 'g'} or other
            Which element set the coordinates address; anything other than
            the three filament orientations selects the node set.

        Returns
        -------
        ndarray
            Zero vector of the selected set's size with the excitations
            placed at the resolved indices.

        Raises
        ------
        IndexError
            If input lengths differ, a coordinate exceeds the grid, or the
            location is not part of the discretized structure.
        """
        if np.size(sourcex) != np.size(sourcey) or \
                np.size(sourcey) != np.size(sourcez) or \
                np.size(sourcez) != np.size(value):
            message = 'Input arguments must be vectors of equal length!'
            raise IndexError(message)
        if orientation == 'e':
            source = np.zeros_like(self.e.data)
        elif orientation == 'f':
            source = np.zeros_like(self.f.data)
        elif orientation == 'g':
            source = np.zeros_like(self.g.data)
        else:
            source = np.zeros_like(self.lv[0].data)
        for i in range(np.size(sourcex)):
            nfilx = 1
            nfily = 1
            nfilz = 1
            for l in range(self.numlevels):
                nfilx *= self.lv[l].n[0]
                nfily *= self.lv[l].n[1]
                nfilz *= self.lv[l].n[2]
            x = sourcex[i]
            y = sourcey[i]
            z = sourcez[i]
            startidx = 0
            pp = 0
            if x > nfilx or y > nfily or z > nfilz:
                print('x:', x, '   y:', y, '   z:', z)
                print('nfilx:', nfilx, '   nfily:', nfily, '   nfilz:', nfilz)
                raise IndexError('Index out of range!')
            for l in range(self.numlevels-1, 0, -1):
                nfilx /= self.lv[l].n[0]
                nfily /= self.lv[l].n[1]
                nfilz /= self.lv[l].n[2]
                xx = x // nfilx
                yy = y // nfily
                zz = z // nfilz
                stopidx = self.lv[l].idx0[startidx+pp+1]
                startidx = self.lv[l].idx0[startidx+pp]
                ny = self.lv[l].n[1]
                nz = self.lv[l].n[2]
                p = xx*ny*nz + yy*nz + zz
                ip = self.lv[l].idx[startidx:stopidx]
                if p not in ip:
                    raise IndexError('Location is not within structure!')
                pp = np.argwhere(ip == p)[0][0]
                x = x % nfilx
                y = y % nfily
                z = z % nfilz
            if orientation == 'e':
                stopidx = self.e.idx0[startidx+pp+1]
                startidx = self.e.idx0[startidx+pp]
                ip = self.e.idx[startidx:stopidx]
            elif orientation == 'f':
                stopidx = self.f.idx0[startidx+pp+1]
                startidx = self.f.idx0[startidx+pp]
                ip = self.f.idx[startidx:stopidx]
            elif orientation == 'g':
                stopidx = self.g.idx0[startidx+pp+1]
                startidx = self.g.idx0[startidx+pp]
                ip = self.g.idx[startidx:stopidx]
            else:
                stopidx = self.lv[0].idx0[startidx+pp+1]
                startidx = self.lv[0].idx0[startidx+pp]
                ip = self.lv[0].idx[startidx:stopidx]
                value[i] *= self.lv[0].beta
            ny = self.lv[0].n[1]
            nz = self.lv[0].n[2]
            p = x*ny*nz + y*nz + z
            if p not in ip:
                raise IndexError('Location is not within structure!')
            pp = np.argwhere(ip == p)[0][0]
            source[startidx+pp] = value[i]
        return source


