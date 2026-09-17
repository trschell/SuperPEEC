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


def spmv_c(A, v):
    """``A @ v`` for a real sparse ``A`` and a complex ``v``, as two
    real products; anything else falls through to ``A @ v``."""
    v = np.asarray(v)
    if np.iscomplexobj(v) and not np.iscomplexobj(A):
        out = A @ np.ascontiguousarray(v.real)
        out = out.astype(np.complex128, copy=False)
        out += 1j*(A @ np.ascontiguousarray(v.imag))
        return out
    return A @ v
