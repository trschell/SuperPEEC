# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The plaquette basis with no stored indices (2026-09-28).

The loop basis's plaquette block is four entries per column: the
edges of a lattice face, plus or minus one in one fixed pattern per
face normal. With the plaquettes in tile order (``loopmg.tile_
permutation``) their numbering is a formula over the stencil's
occupancy mask, and the filament numbering is a formula over the
tree's leaf groups (x, y, z order of groups, x, y, z order inside a
group, each with an occupancy mask). So both products can be gathers
computed from coordinates, in :mod:`krylov_kernels`, and the three
per-entry arrays -- indices, palette codes, column pointer, 1.14 GB at
R5 and ~5.7 GB at R6 -- need not exist. What remains is the tail: the
distribution and chord columns, a small explicit block.

The products add in the order the stored form added (columns
ascending for the forward scatter, ascending filament index within a
column for the transpose), so they are bit-identical to the CSC/CSR
and palette forms; ``scratch/lattice_ab.py`` checks that.

Presents the palette protocol :func:`spmv.spmv_c` dispatches on:
``format == 'palette'``, ``matvec_pair``, ``matvec``, ``T``.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import _sparsetools as _ST

try:
    from krylov_kernels import lattice_b as _kb, lattice_bt as _kbt
except ImportError:                    # module not rebuilt: not available
    _kb = _kbt = None

# the stored entry pattern of a plaquette column, per normal, in the
# stored (ascending filament index) order: (axis, ox, oy, oz, sign)
PATTERN = np.array([
    [[1, 0, 0, 0, 1], [1, 0, 0, 1, -1], [2, 0, 0, 0, -1], [2, 0, 1, 0, 1]],
    [[0, 0, 0, 0, -1], [0, 0, 0, 1, 1], [2, 0, 0, 0, 1], [2, 1, 0, 0, -1]],
    [[1, 0, 0, 0, -1], [1, 1, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, -1]],
], dtype=np.int32)


def available():
    return _kb is not None


class LatticeBasis(object):
    """``Bmat`` (filaments + wire rows) x (plaquettes | tail columns)
    with the plaquette block implicit."""

    format = 'palette'

    def __init__(self, Bmat, M, efg, nplaq, fil_axis, fil_cell, TL):
        if _kb is None:
            raise ImportError("krylov_kernels has no lattice kernels")
        import loopmg
        from spmv import csc_prefix
        B = Bmat.tocsc()
        self.shape = tuple(int(v) for v in B.shape)
        self.nnz = int(B.nnz)
        self.dtype = np.dtype(np.float64)
        self.efg, self.nplaq = int(efg), int(nplaq)
        self.TL = int(TL)
        fil_axis = np.asarray(fil_axis)
        fil_cell = np.asarray(fil_cell)
        if fil_axis.dtype != np.int8 or fil_cell.dtype != np.int16:
            raise ValueError("lattice basis wants int8 axes and int16 cells")
        # ---- plaquettes: tile order over the mask. Built CHUNKED and in
        # int32 (2026-09-28): the first form made int64 copies of every
        # per-plaquette array -- ~3 GB of transient at R5 at the first
        # product, above the run's peak -- for tables of a few MB.
        Y = csc_prefix(B, self.efg, self.nplaq)
        if not np.all(np.diff(Y.indptr) == 4):
            raise ValueError("a plaquette column without four entries")
        nrm, bse = loopmg.plaquette_geometry(Y, fil_axis, fil_cell,
                                             self.nplaq)
        del Y
        nrm = np.asarray(nrm)
        bse = np.asarray(bse)
        n = self.nplaq
        cube = TL**3
        nwt = (3*cube + 63)//64
        CH = 1 << 22
        # pass 1: the tiles, from where the tile key changes
        starts, coords = [], []
        prev = None
        for a0 in range(0, n, CH):
            a1 = min(n, a0 + CH)
            tc = (bse[a0:a1]//TL).astype(np.int32)
            key = ((tc[:, 0].astype(np.int64) << 42)
                   | (tc[:, 1].astype(np.int64) << 21)
                   | tc[:, 2].astype(np.int64))
            if np.any(key[1:] < key[:-1]) or (prev is not None
                                              and key[0] < prev):
                raise ValueError("plaquettes are not in tile order")
            chg = np.flatnonzero(key[1:] != key[:-1]) + 1
            if prev is None or key[0] != prev:
                chg = np.concatenate([[0], chg])
            starts.append(a0 + chg.astype(np.int64))
            coords.append(tc[chg])
            prev = int(key[-1])
            del tc, key, chg
        tpre = np.concatenate(starts)
        tcoord = np.concatenate(coords).astype(np.int32)     # (nt, 3)
        del starts, coords
        nt = int(tpre.size)
        self.tpre = tpre
        self.tcoord = np.asfortranarray(tcoord.T)          # (3, nt)
        dims = tcoord.max(axis=0) + 1
        tgrid = np.full(tuple(int(d) for d in dims), -1, np.int32)
        tgrid[tcoord[:, 0], tcoord[:, 1], tcoord[:, 2]] = \
            np.arange(nt, dtype=np.int32)
        self.tgrid = np.asfortranarray(tgrid)
        del tgrid
        # pass 2: the masks, a chunk of plaquettes at a time
        tmask = np.zeros(nt*nwt, np.uint64)
        prevk = -1
        for a0 in range(0, n, CH):
            a1 = min(n, a0 + CH)
            tc = (bse[a0:a1]//TL).astype(np.int32)
            loc = bse[a0:a1].astype(np.int32) - tc*TL
            slot = (((nrm[a0:a1].astype(np.int32)*TL + loc[:, 2])*TL
                     + loc[:, 1])*TL + loc[:, 0])
            tile = (np.searchsorted(tpre, np.arange(a0, a1), side='right')
                    - 1)
            full = tile*(3*cube) + slot
            if np.any(full[1:] <= full[:-1]) or full[0] <= prevk:
                raise ValueError("plaquettes are not in slot order")
            prevk = int(full[-1])
            wkey = tile*nwt + (slot >> 6)
            bit = np.left_shift(np.uint64(1), (slot & 63).astype(np.uint64))
            st = np.flatnonzero(np.r_[True, wkey[1:] != wkey[:-1]])
            tmask[wkey[st]] |= np.bitwise_or.reduceat(bit, st)
            del tc, loc, slot, tile, full, wkey, bit, st
        self.tmask = np.ascontiguousarray(
            tmask.view(np.int64).reshape(nt, nwt).T)      # (nwt, nt) F
        del tmask, nrm, bse
        # ---- filaments: leaf groups over the box grid, masks within
        from equiterminal import _leaf_order
        nleaf = None
        gb, gm, grids = [], [], []
        ntot = 0
        for leaf, axis, off in _leaf_order(M):
            ln = np.asarray(leaf.n, np.int64)
            if nleaf is None:
                nleaf = ln
            elif not np.array_equal(ln, nleaf):
                raise ValueError("leaf boxes differ between orientations")
            idx = np.asarray(leaf.idx)
            if getattr(M, 'numlevels', 1) > 1:
                i0 = np.asarray(leaf.idx0, np.int64)
                bx = np.asarray(leaf.xidx, np.int64)
                by = np.asarray(leaf.yidx, np.int64)
                bz = np.asarray(leaf.zidx, np.int64)
            else:
                i0 = np.array([0, idx.size], np.int64)
                bx = by = bz = np.zeros(1, np.int64)
            ng = int(i0.size - 1)
            ncell = int(np.prod(nleaf))
            nwg = (ncell + 63)//64
            mask = np.zeros(ng*nwg, np.uint64)
            for a0 in range(0, idx.size, CH):
                a1 = min(idx.size, a0 + CH)
                ix = idx[a0:a1].astype(np.int32)
                gid = (np.searchsorted(i0, np.arange(a0, a1), side='right')
                       - 1).astype(np.int32)
                wk = gid*nwg + (ix >> 6)
                if np.any(np.diff(wk) < 0):
                    raise ValueError("leaf idx not sorted within groups")
                bitg = np.left_shift(np.uint64(1), (ix & 63).astype(np.uint64))
                st = np.flatnonzero(np.r_[True, wk[1:] != wk[:-1]])
                mask[wk[st]] |= np.bitwise_or.reduceat(bitg, st)
                del ix, gid, wk, bitg, st
            gm.append(mask.view(np.int64).reshape(ng, nwg))
            gb.append(off + i0[:-1])
            grids.append((axis, bx, by, bz, ntot))
            ntot += ng
        self.n = nleaf
        self.gbase = np.concatenate(gb).astype(np.int64)
        gmask = np.concatenate(gm, axis=0)
        self.gmask = np.asfortranarray(gmask.T)          # (nwg, ng)
        gd = [1 + max(int(g[k].max()) for g in grids) for k in (1, 2, 3)]
        ggrid = np.full((gd[0], gd[1], gd[2], 3), -1, np.int32)
        for axis, bx, by, bz, base in grids:
            ggrid[bx, by, bz, axis] = base + np.arange(bx.size, dtype=np.int32)
        self.ggrid = np.asfortranarray(ggrid)
        # ---- the sign of edge (axis a, offset d along the third axis)
        # in the face of normal `on`: sgn[d, a, on]
        sgn = np.zeros((2, 3, 3), np.int32)
        for on in range(3):
            for k in range(4):
                a, ox, oy, oz, s = PATTERN[on, k]
                sgn[int(ox + oy + oz), a, on] = s
        self.sgn = np.asfortranarray(sgn)
        self.pat = np.asfortranarray(PATTERN.transpose(2, 1, 0))  # (5, 4, 3)
        self.fil_axis = np.ascontiguousarray(fil_axis)
        self.fil_cellT = np.asfortranarray(fil_cell.T)   # (3, nf) view
        # ---- the tail: distribution + chord columns, explicit
        tail = B[:, self.nplaq:].tocsc()
        self.tail = sp.csc_matrix((tail.data.astype(np.float64),
                                   tail.indices, tail.indptr), shape=tail.shape)
        self.tailT = self.tail.T.tocsr()
        self._T = _LatticeT(self)

    # ------------------------------------------------------------ size
    def nbytes(self):
        return int(sum(a.nbytes for a in (self.tmask, self.tpre, self.tcoord,
                                          self.tgrid, self.gbase, self.gmask,
                                          self.ggrid))
                   + self.tail.data.nbytes + self.tail.indices.nbytes
                   + self.tail.indptr.nbytes + self.tailT.data.nbytes
                   + self.tailT.indices.nbytes + self.tailT.indptr.nbytes)

    @property
    def T(self):
        return self._T

    # -------------------------------------------------------- products
    def matvec2(self, x1, x2, out1, out2):
        """``out = B x`` for two real vectors."""
        nrow = self.shape[0]
        n0, n1, n2 = (int(v) for v in self.n)
        yr, yi = _kb(self.tmask, self.tpre, self.tgrid, self.TL, self.sgn,
                     self.fil_axis, self.fil_cellT,
                     x1[:self.nplaq], x2[:self.nplaq])
        out1[:self.efg] = yr
        out2[:self.efg] = yi
        out1[self.efg:] = 0.0
        out2[self.efg:] = 0.0
        t = self.tail
        nc = int(t.shape[1])
        if t.nnz:
            _ST.csc_matvec(nrow, nc, t.indptr, t.indices, t.data,
                           np.ascontiguousarray(x1[self.nplaq:]), out1)
            _ST.csc_matvec(nrow, nc, t.indptr, t.indices, t.data,
                           np.ascontiguousarray(x2[self.nplaq:]), out2)
        return out1, out2

    def rmatvec2(self, y1, y2, out1, out2):
        """``out = B^T y`` for two real vectors."""
        nrow = self.shape[0]
        n0, n1, n2 = (int(v) for v in self.n)
        yr, yi = _kbt(self.tmask, self.tpre, self.tcoord, self.TL, n0, n1, n2,
                      self.ggrid, self.gbase, self.gmask, self.pat,
                      y1, y2, self.nplaq)
        out1[:self.nplaq] = yr
        out2[:self.nplaq] = yi
        o1, o2 = out1[self.nplaq:], out2[self.nplaq:]
        o1.fill(0.0)
        o2.fill(0.0)
        t = self.tailT
        if t.nnz:
            _ST.csr_matvec(int(t.shape[0]), nrow, t.indptr, t.indices,
                           t.data, y1, o1)
            _ST.csr_matvec(int(t.shape[0]), nrow, t.indptr, t.indices,
                           t.data, y2, o2)
        return out1, out2

    def matvec_pair(self, x1, x2):
        n = self.shape[0]
        return self.matvec2(np.ascontiguousarray(x1, np.float64),
                            np.ascontiguousarray(x2, np.float64),
                            np.empty(n, np.float64), np.empty(n, np.float64))

    def matvec(self, x, out=None):
        x = np.ascontiguousarray(x, np.float64)
        n = self.shape[0]
        o1 = out if out is not None else np.empty(n, np.float64)
        self.matvec2(x, x, o1, np.empty(n, np.float64))
        return o1

    def rmatvec_pair(self, y1, y2):
        n = self.shape[1]
        return self.rmatvec2(np.ascontiguousarray(y1, np.float64),
                             np.ascontiguousarray(y2, np.float64),
                             np.empty(n, np.float64), np.empty(n, np.float64))

    def rmatvec(self, y, out=None):
        y = np.ascontiguousarray(y, np.float64)
        n = self.shape[1]
        o1 = out if out is not None else np.empty(n, np.float64)
        self.rmatvec2(y, y, o1, np.empty(n, np.float64))
        return o1


class _LatticeT(object):
    format = 'palette'

    def __init__(self, parent):
        self._p = parent
        self.shape = (parent.shape[1], parent.shape[0])
        self.dtype = parent.dtype
        self.nnz = parent.nnz

    @property
    def T(self):
        return self._p

    def matvec(self, x, out=None):
        return self._p.rmatvec(x, out)

    def matvec_pair(self, x1, x2):
        return self._p.rmatvec_pair(x1, x2)

    def nbytes(self):
        return 0
