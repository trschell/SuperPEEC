# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The conductivity grid as one byte per voxel and a value table.

``VoxelModel.sigma`` is a dense float32 grid over the whole bounding
box: 800 MB at the R5 flagship, holding exactly TWO distinct values
(empty, copper) over a box that is 9% occupied. Every consumer reads it
in one of a few ways -- index it, compare it with a scalar, or ask for
its distinct values -- and all of those come straight off a byte grid
and a table. So that is what this is. Exact: the table holds the
original float32 values, and every read reproduces the dense grid bit
for bit. 200 MB at R5.

It is not an ndarray subclass on purpose. A subclass would be a dense
array again the moment numpy touched it. This supports the operations
the model actually uses and materialises only for ``np.asarray``,
which two callers still do (``impedance_density``, ``sigma_along``);
those are the next thing to make table-aware.
"""
import numpy as np

MAX_VALUES = 255                 # uint8 codes; 255 keeps a spare


class PaletteGrid(object):
    """A grid whose values come from a small table."""

    def __init__(self, code, table, dtype):
        self.code = np.ascontiguousarray(code, dtype=np.uint8)
        self.table = np.ascontiguousarray(table, dtype=dtype)
        self.dtype = np.dtype(dtype)
        self.shape = self.code.shape
        self.ndim = self.code.ndim
        self.size = self.code.size

    # ------------------------------------------------------------ build

    @classmethod
    def compact(cls, arr):
        """The palette form of a dense grid, or None when it cannot be
        represented (more distinct values than a byte can index)."""
        arr = np.asarray(arr)
        table, inv = np.unique(arr, return_inverse=True)
        if table.size > MAX_VALUES:
            return None
        g = cls(inv.reshape(arr.shape), table, arr.dtype)
        # the round trip is the whole claim
        assert np.array_equal(g.table[g.code], arr)
        return g

    @property
    def nbytes(self):
        return int(self.code.nbytes + self.table.nbytes)

    def todense(self, dtype=None):
        out = self.table[self.code]
        return out if dtype is None else out.astype(dtype, copy=False)

    def __array__(self, dtype=None, copy=None):
        return self.todense(dtype)

    def astype(self, dtype, copy=True):
        return self.todense(dtype)

    def values(self):
        """The distinct values present, sorted."""
        return self.table.copy()

    # ------------------------------------------------------------ reads

    def __getitem__(self, idx):
        c = self.code[idx]
        return self.table[c]

    def _code_of(self, value):
        """The code for ``value``, or -1 when it is not in the table."""
        v = np.asarray(value, dtype=self.dtype)
        if v.ndim:
            raise TypeError("scalar expected")
        k = int(np.searchsorted(self.table, v))
        return k if k < self.table.size and self.table[k] == v else -1

    def __eq__(self, other):
        if isinstance(other, PaletteGrid):
            return self.todense() == other.todense()
        if np.ndim(other):
            return self.todense() == other
        k = self._code_of(other)
        return (self.code == k) if k >= 0 else np.zeros(self.shape, bool)

    def __ne__(self, other):
        if isinstance(other, PaletteGrid):
            return self.todense() != other.todense()
        if np.ndim(other):
            return self.todense() != other
        k = self._code_of(other)
        return (self.code != k) if k >= 0 else np.ones(self.shape, bool)

    __hash__ = None

    def __lt__(self, other):
        return self._cmp(other, np.less)

    def __le__(self, other):
        return self._cmp(other, np.less_equal)

    def __gt__(self, other):
        return self._cmp(other, np.greater)

    def __ge__(self, other):
        return self._cmp(other, np.greater_equal)

    def _cmp(self, other, op):
        # a scalar comparison is a comparison on the TABLE, indexed
        # back out by the codes: exact, and never touches a dense grid
        if np.ndim(other) or isinstance(other, PaletteGrid):
            return op(self.todense(), np.asarray(other))
        return op(self.table, np.asarray(other, dtype=self.dtype))[self.code]

    # ---------------------------------------------- ndarray-style reductions

    def min(self):
        return self.table.min()          # the table holds only present values

    def max(self):
        return self.table.max()

    def sum(self, dtype=None):
        """Exact sum: each distinct value times its count, in the
        table's dtype widened to float64 (or ``dtype``)."""
        counts = np.bincount(self.code.ravel(), minlength=self.table.size)
        acc = np.dtype(dtype) if dtype is not None else np.result_type(self.dtype, np.float64)
        return (counts.astype(acc)*self.table.astype(acc)).sum(dtype=acc)

    def mean(self):
        return self.sum()/self.size

    def any(self):
        return bool((self.table != 0).any()) and bool(self.code.size)

    def all(self):
        return bool((self.table != 0).all())

    def nonzero(self):
        k = self._code_of(0)
        return np.nonzero(self.code != k) if k >= 0 else np.nonzero(np.ones(self.shape, bool))

    def copy(self):
        return PaletteGrid(self.code.copy(), self.table.copy(), self.dtype)

    def ravel(self):
        return self.todense().ravel()

    def flatten(self):
        return self.todense().ravel()

    def reshape(self, *shape):
        return self.todense().reshape(*shape)

    # ----------------------------------------------------------- writes

    def __setitem__(self, idx, value):
        vals = np.asarray(value, dtype=self.dtype)
        new = np.setdiff1d(np.unique(vals), self.table)
        if new.size:
            table = np.union1d(self.table, new)
            if table.size > MAX_VALUES:
                raise ValueError("palette grid would need %d distinct "
                                 "values; a byte indexes %d"
                                 % (table.size, MAX_VALUES))
            # recode the whole grid against the enlarged table
            remap = np.searchsorted(table, self.table).astype(np.uint8)
            self.code = remap[self.code]
            self.table = table
        codes = np.searchsorted(self.table, vals).astype(np.uint8)
        self.code[idx] = codes

    def __repr__(self):
        return ('PaletteGrid(shape=%s, %d values, %.1f MB against %.1f dense)'
                % (self.shape, self.table.size, self.nbytes/1e6,
                   self.size*self.dtype.itemsize/1e6))
