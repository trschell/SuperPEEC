# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Whole-domain circulant application of the panel coefficient-of-potential
operator (single level) -- the PyPEEC lesson imported.

The single-level capacitive operator n2n was the one dense object left in the
solver: p2pinit3 materializes the panel-to-panel Toeplitz blocks entry by
entry (O(N_p^2) storage/build; the 1 GB + transients that OOM a 19^3 cube on
a 12 GB machine) and traverseP3 applies it at O(N^2). But the kernel TABLES
behind that assembly (greens.gen2_p_parz / gen_p_per) are translation-
invariant offset tables -- the circulant kernel already exists. This module
keeps only the tables' FFT spectra and applies P by convolution:
O(N_grid log N_grid) time, O(N_grid) memory, and EXACT -- the same table
values the dense path looks up, so agreement with the dense oracle is at
machine precision, not truncation level.

Geometry of the blocks. The three panel families live on different staggered
lattices (normal axis: integer planes with full-cell spacing; transverse
axes: half-cell centers). For a (source s, target t) block, each axis falls
in one of three cases, mirrored EXACTLY from the validated p2pinit3 index
arithmetic (see its docstring for the (k <-> k+1/2) table semantics):

  * registry axis (s == t, or the shared transverse axis of a cross block):
    both grids share one lattice; kernel index |m|, m = T - S;
  * c == s (cross block): the source is the plane grid (2 half-units per
    index); on the common half-unit lattice m = T - 2S and the physical
    separation is (m + 1/2) half-units: index m for m >= 0, -m-1 otherwise;
  * c == t: mirrored, m = 2T - S, index m-1 for m >= 1, -m otherwise.

Per block, the source grid is upsampled (stride 2 on its plane axis) onto
the common lattice, the signed-offset kernel K[m] is gathered from the gen
tables with those mappings, and the product is a single zero-padded FFT
convolution; the target samples are read off at its own strides. Nine
blocks; spectra total tens of MB where the dense matrix was GB.

The same offset tables also yield the sparse NEAR-FIELD n2n (a bounded
offset band around each panel, projected through node2panel) that the LpPR
machinery needs for the W = P_ext^{-1} rescaling and the C_cap diagonal --
mirroring what the multilevel path has always done (near-field-only n2n +
fast far field).

Validated by validate_circulant_poten.py against the dense n2n oracle.
"""
import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.sparse import coo_matrix
from greens import panel_tables
import stencils as st


def _source_tables(leaf, targets):
    """The kernel tables for this source orientation, in (x, y, z) axis
    order.

    Delegates to :func:`greens.panel_tables`, the single construction
    shared with ``leaf_poten.LeafPoten.p2pinit3``. This used to be a
    verbatim copy of that code; the cell scheme would have made it a
    three-way duplicate, so it is now one definition and every value
    this module uses is by construction the one the dense oracle uses.
    """
    return panel_tables(leaf.l, leaf.n, [t.n for t in targets],
                        leaf.orientation)


def _axis_map(si, ti, c, m, off):
    """Signed common-lattice separation -> kernel-table index.

    Parallel blocks index by |sep|; cross blocks index a SIGNED table
    (a perpendicular coupling is not symmetric under reflecting one
    axis) shifted by ``off``.
    """
    return np.abs(m) if si == ti else m + off[c]


class CirculantPoten:
    """Circulant panel-potential operator + near-field sparse n2n for a
    single-level capacitive Tree."""

    # Near-field truncation radius, in panel-lattice steps. Truncating
    # a Coulomb operator does not preserve positive definiteness, and
    # the smallest eigenvalue of the truncated n2n DEGRADES with mesh
    # size and changes sign at cutoff 2: measured min/max over
    # NT = 7..19 runs 2.4e-2, 1.6e-2, 1.0e-2, 5.7e-3, 2.0e-3,
    # -6.8e-4, -2.7e-3, crossing zero near NT = 16 -- which is exactly
    # where cholmod.cholesky starts failing. Cutoff 3 restores it
    # (+6.7e-3 at 17^3) for about 1.7x the nonzeros; 4 gains nothing
    # further. This is a property of the truncation, not a defect: the
    # untruncated P is a Gram matrix and always positive definite.
    DEFAULT_CUTOFF = 3

    def __init__(self, M, cutoff=None):
        if cutoff is None:
            cutoff = self.DEFAULT_CUTOFF
        self.M = M
        leaves = (M.px, M.py, M.pz)
        self._leaves = leaves
        # full-grid <-> compressed-panel index maps per family
        self._grid2comp = []
        for p in leaves:
            g2c = np.full(int(np.prod(p.n)), -1, dtype=np.int64)
            g2c[p.idx] = np.arange(p.idx.size)
            self._grid2comp.append(g2c)
        # per-block convolution data
        self._blocks = {}
        self._celloff = {}
        tables = {}
        for si, s in enumerate(leaves):
            so, tabs, offs = _source_tables(s, leaves)
            assert so == si
            tables[si] = tabs
            self._celloff[si] = offs
            for ti, t in enumerate(leaves):
                self._blocks[(si, ti)] = self._build_block(
                    s, t, si, ti, tabs[ti])
        self.n2n_near = self._build_near(tables, cutoff)

    # ---------------------------------------------------------------- build
    def _strides(self, si, ti):
        """Common-lattice strides for a (source, target) block.

        Edge scheme: panels are quarter faces, so the transverse
        lattices run at half-cell pitch while a panel's own-normal axis
        is a plane grid at full-cell pitch. A cross block therefore has
        to work on a common HALF-cell lattice, on which the two
        plane-grid axes advance 2 units per index.

        Cell scheme: a panel is a whole face and every lattice is at
        full-cell pitch, so there is no finer common lattice and every
        stride is 1. The staggering between differently oriented panels
        is half a cell of OFFSET, not of pitch, and that lives in the
        kernel table (greens.p_per_table) rather than in the indexing.
        """
        return np.ones(3, dtype=int), np.ones(3, dtype=int)

    def _build_block(self, s, t, si, ti, table):
        ns = s.n.astype(int)
        nt = t.n.astype(int)
        us, ut = self._strides(si, ti)
        Ls = us*(ns - 1) + 1
        Lt = ut*(nt - 1) + 1
        Lpad = tuple(next_fast_len(int(Ls[c] + Lt[c] - 1)) for c in range(3))
        # signed-offset kernel K[m], m_c in [-(Ls_c-1), Lt_c-1], stored with
        # index p_c = m_c + (Ls_c - 1)
        allo = self._celloff.get(si)
        off = None if allo is None else allo[ti]
        ax = [_axis_map(si, ti, c,
                        np.arange(-(Ls[c] - 1), Lt[c]), off)
              for c in range(3)]
        K = table[np.ix_(ax[0], ax[1], ax[2])]
        Kpad = np.zeros(Lpad, dtype=np.float64)
        Kpad[:K.shape[0], :K.shape[1], :K.shape[2]] = K
        return dict(ns=ns, nt=nt, us=us, ut=ut, Ls=Ls, Lt=Lt, Lpad=Lpad,
                    FK=fftn(Kpad))

    # ---------------------------------------------------------------- apply
    def apply(self, q):
        """Panel charges (qx, qy, qz) -> panel potentials (phix, phiy, phiz),
        i.e. phi_t = sum_s P[s->t]^T q_s in p2pinit3's convention."""
        leaves = self._leaves
        qgrid = []
        for p, qs in zip(leaves, q):
            g = np.zeros(int(np.prod(p.n)), dtype=np.complex128)
            g[p.idx] = qs
            qgrid.append(g.reshape(tuple(p.n.astype(int))))
        phi = [np.zeros(tuple(p.n.astype(int)), dtype=np.complex128)
               for p in leaves]
        for (si, ti), blk in self._blocks.items():
            qpad = np.zeros(blk['Lpad'], dtype=np.complex128)
            sl = tuple(slice(0, blk['Ls'][c], blk['us'][c]) for c in range(3))
            qpad[sl] = qgrid[si]
            out = ifftn(fftn(qpad)*blk['FK'])
            # phi_t[T] = out[(Ls-1) + ut*T]
            ex = tuple(slice(blk['Ls'][c] - 1,
                             blk['Ls'][c] - 1 + blk['ut'][c]*blk['nt'][c],
                             blk['ut'][c]) for c in range(3))
            phi[ti] += out[ex]
        return tuple(phi[ti].ravel()[leaves[ti].idx] for ti in range(3))

    def apply_nodes(self, qn):
        """Node charges -> node potentials: node2panel scatter, circulant
        panel apply, transpose gather. The single-level traverseP3 body."""
        M = self.M
        q = (M.node2px @ qn, M.node2py @ qn, M.node2pz @ qn)
        (fx, fy, fz) = self.apply(q)
        return (M.node2px.T @ fx + M.node2py.T @ fy + M.node2pz.T @ fz)

    # ----------------------------------------------------------- near field
    def _build_near(self, tables, cutoff):
        """Sparse near-field n2n (offset band |physical| <~ cutoff cells),
        the analog of the multilevel 27-neighbour n2n: consumed by the
        W = P_ext^{-1} rescaling and the C_cap diagonal, never by the
        operator itself."""
        M = self.M
        leaves = self._leaves
        n2p = (M.node2px, M.node2py, M.node2pz)
        nn = M.lv[0].struc.size
        phinear = [None, None, None]         # PHI_t = sum_s P_st^T node2p_s
        for ti, t in enumerate(leaves):
            acc = None
            for si, s in enumerate(leaves):
                P = self._near_block(s, t, si, ti, tables[si][ti], cutoff)
                term = (P.T @ n2p[si]).tocsr()
                acc = term if acc is None else acc + term
            phinear[ti] = acc
        n2n = (n2p[0].T @ phinear[0] + n2p[1].T @ phinear[1]
               + n2p[2].T @ phinear[2]).tocsr()
        return n2n

    def _near_block(self, s, t, si, ti, table, cutoff):
        ns = s.n.astype(int)
        nt = t.n.astype(int)
        us, ut = self._strides(si, ti)
        # physical spacing of one common-lattice unit per axis
        h = np.asarray(s.l, dtype=np.float64)/us
        cell = np.asarray(self.M.e.l, dtype=np.float64)  # full cell size
        lim = np.ceil(cutoff*cell/h).astype(int) + 1
        g2c_s = self._grid2comp[si]
        g2c_t = self._grid2comp[ti]
        # source grid coordinates (per axis) of every source lattice point
        Sx, Sy, Sz = np.meshgrid(np.arange(ns[0]), np.arange(ns[1]),
                                 np.arange(ns[2]), indexing='ij')
        Sflat = (Sx.ravel(), Sy.ravel(), Sz.ravel())
        rows = []
        cols = []
        vals = []
        allo = self._celloff.get(si)
        off = None if allo is None else allo[ti]
        for mx in range(-lim[0], lim[0] + 1):
            tx = _axis_map(si, ti, 0, np.array([mx]), off)[0]
            if not 0 <= tx < table.shape[0]:
                continue
            for my in range(-lim[1], lim[1] + 1):
                ty = _axis_map(si, ti, 1, np.array([my]), off)[0]
                if not 0 <= ty < table.shape[1]:
                    continue
                for mz in range(-lim[2], lim[2] + 1):
                    tz = _axis_map(si, ti, 2, np.array([mz]), off)[0]
                    if not 0 <= tz < table.shape[2]:
                        continue
                    m = (mx, my, mz)
                    # target index per axis: ut*T = m + us*S
                    T = []
                    ok = np.ones(Sflat[0].size, dtype=bool)
                    for c in range(3):
                        o = m[c] + us[c]*Sflat[c]
                        if ut[c] == 2:
                            ok &= (o % 2) == 0
                            o = o // 2
                        ok &= (o >= 0) & (o < nt[c])
                        T.append(o)
                    if not ok.any():
                        continue
                    sidx = (Sflat[0][ok]*ns[1] + Sflat[1][ok])*ns[2] \
                        + Sflat[2][ok]
                    tidx = (T[0][ok]*nt[1] + T[1][ok])*nt[2] + T[2][ok]
                    sc = g2c_s[sidx]
                    tc = g2c_t[tidx]
                    keep = (sc >= 0) & (tc >= 0)
                    if not keep.any():
                        continue
                    rows.append(sc[keep])
                    cols.append(tc[keep])
                    vals.append(np.full(int(keep.sum()),
                                        table[tx, ty, tz]))
        if rows:
            P = coo_matrix((np.concatenate(vals),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(s.idx.size, t.idx.size))
        else:
            P = coo_matrix((s.idx.size, t.idx.size))
        return P.tocsr()

    def spectra_bytes(self):
        """Total memory held by the kernel spectra (the 'dense n2n
        replacement' footprint)."""
        return sum(b['FK'].nbytes for b in self._blocks.values())
