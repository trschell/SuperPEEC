# -*- coding: utf-8 -*-
"""Sparse-times-complex without the data upcast.

scipy's CSR/CSC matvec kernels take ONE data type, so a real matrix
times a complex128 vector first converts the whole data array to
complex128: measured at R4 size (48M nonzeros) a +0.89 GiB transient
per product against +0.36 for the same product as two real ones, at
the same speed (2026-09-16). The loop-basis operators do two such
products per matvec; this helper is used at every one of them.
"""
import numpy as np


def spmv_c(A, v, out=None):
    """``A @ v`` for a real sparse ``A`` and a complex ``v``, as two
    real products; anything else falls through to ``A @ v``.

    Writes the two real products straight into the halves of the
    complex result. The earlier form built the result by widening the
    real product to complex and then adding ``1j`` times the imaginary
    one, which costs two extra complex temporaries the size of the
    output -- 414 MB of the 517 MB peak that one R4 call was measured
    at, and four times that at R5, allocated and dropped twice per
    matvec where glibc does not hand it back.

    ``out``, when given, is filled instead of a fresh array. Use it
    only where the value does not escape: the result returned to a
    Krylov solver is stored by it and must be its own array.

    Bit-identical to the old form: the same two products in the same
    order, assembled rather than accumulated.
    """
    v = np.asarray(v)
    # a palette matrix (see :mod:`palette`) keeps its values in a table
    # and expands one block at a time, so it multiplies through its own
    # method rather than the `@` operator; the result is bit-identical
    pal = getattr(A, 'format', None) == 'palette'
    if np.iscomplexobj(v) and not np.iscomplexobj(A):
        if out is None:
            out = np.empty(A.shape[0], dtype=np.complex128)
        if pal:
            # one expansion pass feeds both halves
            y1, y2 = A.matvec_pair(np.ascontiguousarray(v.real),
                                   np.ascontiguousarray(v.imag))
            out.real = y1
            out.imag = y2
        else:
            out.real = A @ np.ascontiguousarray(v.real)
            out.imag = A @ np.ascontiguousarray(v.imag)
        return out
    return A.matvec(v) if pal else A @ v


import scipy.sparse as sp


def csc_prefix(M, nrows, ncols):
    """The leading ``ncols`` columns of a CSC matrix as a VIEW over its
    arrays (a column prefix is contiguous in CSC), with ``nrows``
    rows: the ``M[:nrows, :ncols].tocsc()`` it replaces copied 0.4 GB
    on R4 (2026-09-17). Requires the prefix to carry no row >= nrows."""
    M = M.tocsc()
    e = int(M.indptr[ncols])
    assert e == 0 or M.indices[:e].max() < nrows
    return sp.csc_matrix((M.data[:e], M.indices[:e], M.indptr[:ncols + 1]),
                         shape=(nrows, ncols))
