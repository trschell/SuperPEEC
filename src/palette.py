# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""A sparse matrix whose values come from a small palette.

The stacked loop basis holds exactly FOUR distinct values across its
nonzeros -- ``+-1`` and ``+-0.04`` measured on the DBC ladder -- and
stores them as float64: 385 MB at R4, about 1.5 GB at R5, the largest
single object on the host.

It cannot simply narrow. ``float32`` data meeting a float64 vector
makes scipy promote the WHOLE data array back to float64 for the
product (measured: a 32 MB float32 array costs 96 MB of peak against
32 MB for the float64 one), so narrowing the storage would cost more
per product than it saves in residency -- twice per matvec, forever.

So the values are stored once in a table and each nonzero keeps an
8-bit code. The product expands ONE COLUMN BLOCK at a time into a
reusable buffer, so the float64 data never exists in full.

Exactness, which is the point
-----------------------------
The table holds the original float64 values, so every expanded entry
is bit-for-bit the value it replaced. CSC accumulates ``y[i] +=
a[k]*x[j]`` column by column, so chunking by column preserves the
summation order exactly; the transposed product sums each output entry
over its own row and does not care how the rows are grouped. Both are
therefore bit-identical to the dense-data products they replace, which
is what lets this be used on the operator rather than only inside a
preconditioner.
"""
import os

import numpy as np

try:
    from scipy.sparse import _sparsetools as _ST
except Exception:                                    # pragma: no cover
    _ST = None

# Nonzeros per expansion block. This buffer is the ONLY memory the
# palette form adds, so it is kept small: at 64 MB it cost more than
# the data array saved on R3 (peak +148 MB), where at 8 MB the same
# model breaks even and the big rungs keep essentially all of the win.
# More blocks is only more Python loop trips -- 48 of them at R4.
BLOCK_NNZ = int(float(os.environ.get('SPPEEC_PALETTE_BLOCK_MB', '8'))*2**20//8)

MAX_VALUES = int(os.environ.get('SPPEEC_PALETTE_MAX', '256'))

# Below this the form cannot pay for itself at all. The saving is 7/8
# of the data array; the price is an expansion pass per product.
# Measured, same build, peak / wall: R3 (18 M nonzeros) -0.7% / +4.3%,
# R4 (48 M) -3.0% / +3.3%, R5 (79 M) -0.5% / +3.0%.
#
# R5 is the one to read. The saving there is the LARGEST of the three
# in bytes -- 553 MB of resident -- and the smallest in peak, because
# freeing resident state only moves the peak when that state is what
# sets it, and at R5 the peak belongs to the lgmres Krylov basis
# (~16.6 GiB, 21 vectors). Hence opt-in, and most useful beside
# `method = "gmres_stream"`.
#
# Note also that this matrix does NOT scale with cells: 48 M nonzeros
# at R4 against 79 M at R5, 1.64x for a 3.9x cell step.
MIN_NNZ = int(float(os.environ.get('SPPEEC_PALETTE_MIN_MB', '256'))*2**20//8)


class PaletteCSC(object):
    """CSC storage with an 8-bit code per nonzero and a value table."""

    def __init__(self, indptr, indices, code, table, shape):
        self.indptr = indptr
        self.indices = indices
        self.code = code
        self.table = np.ascontiguousarray(table, dtype=np.float64)
        self.shape = (int(shape[0]), int(shape[1]))
        self.nnz = int(code.size)
        self.dtype = np.dtype(np.float64)
        self.format = 'palette'
        self._buf = None
        self._cols = self._block_bounds()
        self._T = _PaletteT(self)

    # ---------------------------------------------------------- build

    @classmethod
    def maybe(cls, M, max_values=None):
        """A palette form of ``M``, or None when it would not pay.

        Refuses anything that is not CSC, anything with more distinct
        values than a byte can index, and anything whose data array is
        too small for the bookkeeping to be worth it.
        """
        if _ST is None or getattr(M, 'format', None) != 'csc':
            return None
        data = getattr(M, 'data', None)
        if data is None or data.dtype != np.float64 or data.size < MIN_NNZ:
            return None
        mx = int(max_values if max_values is not None else MAX_VALUES)
        table = np.unique(data)
        if table.size > mx:
            return None
        code = np.searchsorted(table, data).astype(np.uint8)
        # the round trip must be exact, not close: this replaces an
        # operator, and a palette that merely approximates is a silent
        # change to every answer
        if not np.array_equal(table[code], data):
            return None
        return cls(M.indptr, M.indices, code, table, M.shape)

    def _block_bounds(self):
        """Column indices at which an expansion block starts."""
        p = self.indptr
        bounds, c = [0], 0
        n = self.shape[1]
        while c < n:
            nxt = int(np.searchsorted(p, p[c] + BLOCK_NNZ, side='right')) - 1
            nxt = max(nxt, c + 1)
            c = min(nxt, n)
            bounds.append(c)
        return np.asarray(bounds, dtype=np.int64)

    # --------------------------------------------------------- report

    def nbytes(self):
        return int(self.code.nbytes + self.indices.nbytes
                   + self.indptr.nbytes + self.table.nbytes)

    def dense_nbytes(self):
        """What the same matrix costs with a float64 data array."""
        return int(8*self.nnz + self.indices.nbytes + self.indptr.nbytes)

    @property
    def T(self):
        """The transpose, as a thin view: the same arrays read the
        other way, exactly as scipy's CSC-to-CSR transpose is."""
        return self._T

    # NO kept output buffers. Holding two per direction scaled with the
    # problem -- 1.6 GB at R5 -- and only duplicated what the dense
    # path allocates per product anyway, so the palette would have been
    # trading a data array for working set. The products allocate their
    # own output exactly as `A @ x` does.

    # -------------------------------------------------------- products

    def _expand(self, k0, k1):
        need = k1 - k0
        if self._buf is None or self._buf.size < need:
            self._buf = np.empty(max(need, BLOCK_NNZ), dtype=np.float64)
        out = self._buf[:need]
        np.take(self.table, self.code[k0:k1], out=out)
        return out

    def matvec2(self, x1, x2, out1, out2):
        """Two forward products against the same matrix.

        ``spmv_c`` always wants a real and an imaginary product of one
        matrix, and each block's values must be expanded to feed them.
        As two separate calls that expansion happens twice, which is
        the whole of this form's overhead: an extra pass writing eight
        bytes per nonzero, against a product that reads twelve.
        Sharing it halves that -- 12% of the R3 wall became 6%.
        """
        nrow, ncol = self.shape
        out1.fill(0.0)
        out2.fill(0.0)
        p, b = self.indptr, self._cols
        for i in range(b.size - 1):
            c0, c1 = int(b[i]), int(b[i + 1])
            k0, k1 = int(p[c0]), int(p[c1])
            if k1 == k0:
                continue
            d = self._expand(k0, k1)
            ip, ix = p[c0:c1 + 1] - p[c0], self.indices[k0:k1]
            _ST.csc_matvec(nrow, c1 - c0, ip, ix, d, x1[c0:c1], out1)
            _ST.csc_matvec(nrow, c1 - c0, ip, ix, d, x2[c0:c1], out2)
        return out1, out2

    def rmatvec2(self, y1, y2, out1, out2):
        """Two transposed products, sharing the expansion."""
        nrow, ncol = self.shape
        out1.fill(0.0)
        out2.fill(0.0)
        p, b = self.indptr, self._cols
        for i in range(b.size - 1):
            c0, c1 = int(b[i]), int(b[i + 1])
            k0, k1 = int(p[c0]), int(p[c1])
            if k1 == k0:
                continue
            d = self._expand(k0, k1)
            ip, ix = p[c0:c1 + 1] - p[c0], self.indices[k0:k1]
            _ST.csr_matvec(c1 - c0, nrow, ip, ix, d, y1, out1[c0:c1])
            _ST.csr_matvec(c1 - c0, nrow, ip, ix, d, y2, out2[c0:c1])
        return out1, out2

    def matvec_pair(self, x1, x2):
        """Both forward products, into fresh output arrays."""
        n = self.shape[0]
        return self.matvec2(x1, x2, np.empty(n, np.float64),
                            np.empty(n, np.float64))

    def matvec(self, x, out=None):
        """``A @ x`` for a real contiguous ``x``."""
        nrow, ncol = self.shape
        x = np.ascontiguousarray(x, dtype=np.float64)
        if out is None:
            out = np.zeros(nrow, dtype=np.float64)
        else:
            out.fill(0.0)
        p, b = self.indptr, self._cols
        for i in range(b.size - 1):
            c0, c1 = int(b[i]), int(b[i + 1])
            k0, k1 = int(p[c0]), int(p[c1])
            if k1 == k0:
                continue
            _ST.csc_matvec(nrow, c1 - c0, p[c0:c1 + 1] - p[c0],
                           self.indices[k0:k1], self._expand(k0, k1),
                           x[c0:c1], out)
        return out

    def rmatvec(self, y, out=None):
        """``A.T @ y``: the same arrays read as CSR of the transpose."""
        nrow, ncol = self.shape
        y = np.ascontiguousarray(y, dtype=np.float64)
        if out is None:
            out = np.zeros(ncol, dtype=np.float64)
        else:
            out.fill(0.0)
        p, b = self.indptr, self._cols
        for i in range(b.size - 1):
            c0, c1 = int(b[i]), int(b[i + 1])
            k0, k1 = int(p[c0]), int(p[c1])
            if k1 == k0:
                continue
            _ST.csr_matvec(c1 - c0, nrow, p[c0:c1 + 1] - p[c0],
                           self.indices[k0:k1], self._expand(k0, k1),
                           y, out[c0:c1])
        return out


class _PaletteT(object):
    """``A.T`` for a :class:`PaletteCSC`, without moving any data."""

    def __init__(self, parent):
        self._p = parent
        self.shape = (parent.shape[1], parent.shape[0])
        self.dtype = parent.dtype
        self.nnz = parent.nnz
        self.format = 'palette'

    @property
    def T(self):
        return self._p

    def matvec(self, x, out=None):
        return self._p.rmatvec(x, out)

    def matvec_pair(self, x1, x2):
        n = self.shape[0]
        return self._p.rmatvec2(x1, x2, np.empty(n, np.float64),
                                np.empty(n, np.float64))

    def nbytes(self):
        return 0                      # shares the parent's arrays
