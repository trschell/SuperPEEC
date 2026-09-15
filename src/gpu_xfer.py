# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Chunked host<->device transfers.

WHY (2026-09-15, the memory survey's "CUDA driver mappings" segment):
a pageable host->device copy goes through a driver staging area mapped
from /dev/zero, sized to the LARGEST single transfer and kept for the
life of the process -- it shows up in the resident set as file-backed
pages the OOM killer counts (R4: 2.42 GiB; a 0.81 GiB upload leaves
1.0 GiB behind, the same upload in 64 MB slices 0.13). Every large
transfer therefore goes through these helpers, which copy in
``CHUNK``-byte slices so the staging area never grows past that.
"""
import numpy as np

CHUNK = 64 << 20          # bytes per slice


def to_device(a, cp, dtype=None):
    """Device copy of host array ``a`` (any dtype/shape), in slices."""
    a = np.ascontiguousarray(a)
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype)
    if a.nbytes <= CHUNK:
        return cp.asarray(a)
    d = cp.empty(a.shape, dtype=a.dtype)
    fh, fd = a.reshape(-1), d.reshape(-1)
    step = max(1, CHUNK//a.dtype.itemsize)
    for i in range(0, fh.size, step):
        fd[i:i+step] = cp.asarray(fh[i:i+step])
    return d


def to_host(d, cp, out=None):
    """Host copy of device array ``d``, in slices (into ``out`` if given)."""
    if d.nbytes <= CHUNK and out is None:
        return cp.asnumpy(d)
    if out is None:
        out = np.empty(d.shape, dtype=d.dtype)
    fo, fd = out.reshape(-1), d.reshape(-1)
    step = max(1, CHUNK//d.dtype.itemsize)
    for i in range(0, fd.size, step):
        fo[i:i+step] = cp.asnumpy(fd[i:i+step])
    return out


def csr_to_device(M, cp, csp, dtype=None):
    """cupyx csr_matrix from a scipy csr/csc, uploaded in slices."""
    M = M.tocsr()
    data = to_device(M.data, cp, dtype)
    indices = to_device(M.indices, cp)
    indptr = to_device(M.indptr, cp)
    return csp.csr_matrix((data, indices, indptr), shape=M.shape)
