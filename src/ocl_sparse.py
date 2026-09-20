# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Deterministic sparse and vector kernels for the device preconditioner.

There is no maintained OpenCL sparse library worth depending on, but
the preconditioner does not need one. Per apply it needs a CSR
matrix-vector product, a fused Jacobi update, a small dense product for
the coarse level, and gathers and scatters by index array. The sparse
matrix-matrix products it needs are build-phase, once per solve, and
stay on the host.

Writing the product by hand is not a concession here, it is the point.
The preconditioner must be the same map on every call: a map that
drifts between applies breaks the Arnoldi relation of a long GMRES
cycle, which is how the streamed basis stalled on R4 at 1.9e-4 where it
had converged in 133 steps. cuSPARSE reduces long rows with atomics and
cannot promise that, which is why the CUDA path carries two workarounds
(a materialised transpose and a fixed-order row loop in Python). Here
every row is reduced by a fixed number of work items in a fixed tree,
so the order is a property of the code rather than of the scheduler,
and the row loop can run on the device at full width.

The preconditioner's vectors are real (float32 by default): the complex
system is preconditioned through its real part, and the macro Schur
solve stays on the host. So these kernels are real-valued, with the
scalar type chosen at build time.
"""
import numpy as np

import ocl_core

# The scalar type comes from ocl_core's REAL define, which it derives
# from a complex dtype; these kernels use real_t only.
_DT = {np.dtype(np.float32): np.complex64,
       np.dtype(np.float64): np.complex128}

SOURCE = """
/* One work group per row, WG work items striding its nonzeros, then a
   fixed binary tree in local memory. Same operations in the same order
   on every call, whatever the scheduler does. */
inline real_t row_reduce(__local real_t *s, unsigned int lid)
{
    /* 'half' is an OpenCL type name, so the stride is 'span' */
    for (unsigned int span = WG >> 1; span > 0; span >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < span) s[lid] += s[lid + span];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return s[0];
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void csr_spmv(__global const real_t *data,
              __global const int *indices,
              __global const int *indptr,
              __global const real_t *x,
              __global real_t *y,
              const unsigned int nrow)
{
    __local real_t part[WG];
    const unsigned int row = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    if (row >= nrow) return;
    const int a = indptr[row], b = indptr[row + 1];
    real_t acc = (real_t)0;
    for (int k = a + lid; k < b; k += WG)
        acc += data[k]*x[indices[k]];
    part[lid] = acc;
    const real_t tot = row_reduce(part, lid);
    if (lid == 0) y[row] = tot;
}

/* One damped Jacobi sweep, fused: xout = x + omega*dinv*(b - A x).
   The sweep reads only the old x, so no temporary is needed. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void csr_jacobi(__global const real_t *data,
                __global const int *indices,
                __global const int *indptr,
                __global const real_t *x,
                __global const real_t *b,
                __global const real_t *dinv,
                __global real_t *xout,
                const real_t omega,
                const unsigned int nrow)
{
    __local real_t part[WG];
    const unsigned int row = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    if (row >= nrow) return;
    const int a = indptr[row], e = indptr[row + 1];
    real_t acc = (real_t)0;
    for (int k = a + lid; k < e; k += WG)
        acc += data[k]*x[indices[k]];
    part[lid] = acc;
    const real_t ax = row_reduce(part, lid);
    if (lid == 0) xout[row] = x[row] + omega*dinv[row]*(b[row] - ax);
}

/* r = b - A x, same reduction */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void csr_residual(__global const real_t *data,
                  __global const int *indices,
                  __global const int *indptr,
                  __global const real_t *x,
                  __global const real_t *b,
                  __global real_t *r,
                  const unsigned int nrow)
{
    __local real_t part[WG];
    const unsigned int row = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    if (row >= nrow) return;
    const int a = indptr[row], e = indptr[row + 1];
    real_t acc = (real_t)0;
    for (int k = a + lid; k < e; k += WG)
        acc += data[k]*x[indices[k]];
    part[lid] = acc;
    const real_t ax = row_reduce(part, lid);
    if (lid == 0) r[row] = b[row] - ax;
}

/* y += A x, for the prolongation update */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void csr_spmv_add(__global const real_t *data,
                  __global const int *indices,
                  __global const int *indptr,
                  __global const real_t *x,
                  __global real_t *y,
                  const unsigned int nrow)
{
    __local real_t part[WG];
    const unsigned int row = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    if (row >= nrow) return;
    const int a = indptr[row], e = indptr[row + 1];
    real_t acc = (real_t)0;
    for (int k = a + lid; k < e; k += WG)
        acc += data[k]*x[indices[k]];
    part[lid] = acc;
    const real_t tot = row_reduce(part, lid);
    if (lid == 0) y[row] += tot;
}

/* dense row-major (m, n) times x, for the coarse level's pseudo-inverse */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void dense_gemv(__global const real_t *A,
                __global const real_t *x,
                __global real_t *y,
                const unsigned int m, const unsigned int n)
{
    __local real_t part[WG];
    const unsigned int row = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    if (row >= m) return;
    real_t acc = (real_t)0;
    for (unsigned int k = lid; k < n; k += WG)
        acc += A[(size_t)row*n + k]*x[k];
    part[lid] = acc;
    const real_t tot = row_reduce(part, lid);
    if (lid == 0) y[row] = tot;
}

/* Vector reductions. PyOpenCL's own need the Mako templating engine,
   which is a dependency this tree does not carry, and these are fixed
   in shape anyway: a fixed number of work groups, each reducing
   through the same tree, then a fixed-order sum of the partials on the
   host. Same answer every call. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void red_dot(__global const real_t *a, __global const real_t *b,
             __global real_t *part, const unsigned int n)
{
    __local real_t s[WG];
    const unsigned int lid = get_local_id(0);
    const unsigned int nb = get_num_groups(0);
    real_t acc = (real_t)0;
    for (unsigned int i = get_group_id(0)*WG + lid; i < n; i += WG*nb)
        acc += a[i]*b[i];
    s[lid] = acc;
    const real_t tot = row_reduce(s, lid);
    if (lid == 0) part[get_group_id(0)] = tot;
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void red_maxabs(__global const real_t *a, __global real_t *part,
                const unsigned int n)
{
    __local real_t s[WG];
    const unsigned int lid = get_local_id(0);
    const unsigned int nb = get_num_groups(0);
    real_t acc = (real_t)0;
    for (unsigned int i = get_group_id(0)*WG + lid; i < n; i += WG*nb)
        acc = fmax(acc, fabs(a[i]));
    s[lid] = acc;
    for (unsigned int span = WG >> 1; span > 0; span >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < span) s[lid] = fmax(s[lid], s[lid + span]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) part[get_group_id(0)] = s[0];
}

/* A 0/1 matrix with exactly one entry per row -- which is what an
   aggregation prolongator is, since every fine row belongs to exactly
   one aggregate -- carries no information in its values or its row
   pointers. The values are all 1 and the pointer is the row index, so
   the whole matrix is one column index per row and the product is a
   gather. Each work item owns one output, so nothing is scattered. */
__kernel void ones_gather_add(__global const int *col,
                              __global const real_t *x,
                              __global real_t *y,
                              const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) y[i] += x[col[i]];
}

/* Its transpose: sum the fine entries of each aggregate. Rows are an
   aggregate's size, at most eight, so one work item per row sums them
   in index order. */
__kernel void ones_rowsum(__global const int *ptr,
                          __global const int *ind,
                          __global const real_t *x,
                          __global real_t *y,
                          const unsigned int nrow)
{
    const unsigned int r = get_global_id(0);
    if (r >= nrow) return;
    real_t acc = (real_t)0;
    for (int k = ptr[r]; k < ptr[r + 1]; ++k)
        acc += x[ind[k]];
    y[r] = acc;
}

__kernel void gather_idx(__global const real_t *src,
                         __global const int *idx,
                         __global real_t *dst,
                         const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) dst[i] = src[idx[i]];
}

__kernel void scatter_idx(__global const real_t *src,
                          __global const int *idx,
                          __global real_t *dst,
                          const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) dst[idx[i]] = src[i];
}

/* dst[idx[i]] = src[i] - sub[i] */
__kernel void scatter_sub(__global const real_t *src,
                          __global const real_t *sub,
                          __global const int *idx,
                          __global real_t *dst,
                          const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) dst[idx[i]] = src[i] - sub[i];
}
"""


# ---------------------------------------------------------------------
# Sparse times sparse, and transpose.
#
# Needed for the two build-phase products the CUDA backend does on the
# card: the Gram Y Y^T when no certified stencil is available, and the
# Galerkin P^T A P for each coarse level. Both are formed once per
# solve, not per apply.
#
# The matrices here have short rows. A plaquette touches four
# filaments, so a Gram row is the 36-slot stencil, and the aggregation
# is 0/1 with a handful of entries per column. That makes a per-row
# sorted insert the right shape: one work item owns one output row,
# keeps its column set in a small private array, and writes it out in
# ascending column order. Deterministic by construction, and no hash
# table, no segmented sort, no scratch proportional to the expansion.
#
# A row that exceeds the private bound raises the overflow flag and the
# caller retries wider or falls back to the host, rather than writing
# something wrong.

SPGEMM_SOURCE = """
/* insert (c, v) into the ascending private set; 0 on overflow */
inline int ins(int *cols, real_t *vals, int *n, int c, real_t v)
{
    int lo = 0, hi = *n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (cols[mid] < c) lo = mid + 1; else hi = mid;
    }
    if (lo < *n && cols[lo] == c) { vals[lo] += v; return 1; }
    if (*n >= MAXC) return 0;
    for (int k = *n; k > lo; --k) { cols[k] = cols[k-1]; vals[k] = vals[k-1]; }
    cols[lo] = c; vals[lo] = v; *n = *n + 1;
    return 1;
}

__kernel void spgemm_count(__global const int *aptr,
                           __global const int *aind,
                           __global const int *bptr,
                           __global const int *bind,
                           __global int *cnnz,
                           __global int *over,
                           const unsigned int nrow)
{
    const unsigned int i = get_global_id(0);
    if (i >= nrow) return;
    int cols[MAXC];
    real_t vals[MAXC];
    int n = 0;
    for (int p = aptr[i]; p < aptr[i+1]; ++p) {
        const int k = aind[p];
        for (int q = bptr[k]; q < bptr[k+1]; ++q)
            if (!ins(cols, vals, &n, bind[q], (real_t)0)) { over[0] = 1; return; }
    }
    cnnz[i] = n;
}

__kernel void spgemm_fill(__global const int *aptr,
                          __global const int *aind,
                          __global const real_t *adat,
                          __global const int *bptr,
                          __global const int *bind,
                          __global const real_t *bdat,
                          __global const int *cptr,
                          __global int *cind,
                          __global real_t *cdat,
                          __global int *over,
                          const unsigned int nrow)
{
    const unsigned int i = get_global_id(0);
    if (i >= nrow) return;
    int cols[MAXC];
    real_t vals[MAXC];
    int n = 0;
    for (int p = aptr[i]; p < aptr[i+1]; ++p) {
        const int k = aind[p];
        const real_t av = adat[p];
        for (int q = bptr[k]; q < bptr[k+1]; ++q)
            if (!ins(cols, vals, &n, bind[q], av*bdat[q])) { over[0] = 1; return; }
    }
    const int base = cptr[i];
    for (int j = 0; j < n; ++j) { cind[base+j] = cols[j]; cdat[base+j] = vals[j]; }
}

/* transpose: count, then scatter with an atomic cursor, then sort each
   output row so the result does not depend on the scatter order */
__kernel void trans_count(__global const int *aind,
                          __global int *cnt,
                          const unsigned int nnz)
{
    const unsigned int p = get_global_id(0);
    if (p < nnz) atomic_inc(&cnt[aind[p]]);
}

__kernel void trans_scatter(__global const int *aptr,
                            __global const int *aind,
                            __global const real_t *adat,
                            __global int *cursor,
                            __global int *tind,
                            __global real_t *tdat,
                            const unsigned int nrow)
{
    const unsigned int i = get_global_id(0);
    if (i >= nrow) return;
    for (int p = aptr[i]; p < aptr[i+1]; ++p) {
        const int slot = atomic_inc(&cursor[aind[p]]);
        tind[slot] = (int)i;
        tdat[slot] = adat[p];
    }
}

__kernel void trans_sort(__global const int *tptr,
                         __global int *tind,
                         __global real_t *tdat,
                         const unsigned int nrow)
{
    const unsigned int i = get_global_id(0);
    if (i >= nrow) return;
    const int a = tptr[i], b = tptr[i+1];
    for (int j = a + 1; j < b; ++j) {           /* insertion sort */
        const int c = tind[j];
        const real_t v = tdat[j];
        int k = j - 1;
        while (k >= a && tind[k] > c) { tind[k+1] = tind[k]; tdat[k+1] = tdat[k]; --k; }
        tind[k+1] = c; tdat[k+1] = v;
    }
}
"""


class SpGEMMOverflow(RuntimeError):
    """A row needed more distinct columns than the build allows."""


def _sp_program(dtype, maxc):
    return ocl_core.program(SPGEMM_SOURCE, _DT[np.dtype(dtype)],
                            {'MAXC': int(maxc)},
                            key='ocl_spgemm/%d' % int(maxc))


class DeviceCSR(object):
    """A CSR that stays on the device: the three arrays and a shape."""

    def __init__(self, indptr, indices, data, shape, nnz):
        self.indptr, self.indices, self.data = indptr, indices, data
        self.shape = tuple(int(v) for v in shape)
        self.nnz = int(nnz)

    def to_host(self):
        import scipy.sparse as sp
        q = ocl_core.queue()
        M = sp.csr_matrix((self.data.get(queue=q)[:self.nnz],
                           self.indices.get(queue=q)[:self.nnz],
                           self.indptr.get(queue=q)), shape=self.shape)
        M.eliminate_zeros()
        return M


def transpose_device(A, dtype=np.float32):
    """``A.T`` as a :class:`DeviceCSR`, formed on the device."""
    dt = np.dtype(dtype)
    q = ocl_core.queue()
    prg = _sp_program(dt, 64)
    nrow, ncol = (int(v) for v in A.shape)
    aptr = ocl_core.to_device(A.indptr.astype(np.int32))
    aind = ocl_core.to_device(A.indices.astype(np.int32))
    adat = ocl_core.to_device(A.data.astype(dt))
    cnt = ocl_core.zeros((ncol,), np.int32)
    ocl_core.kernel(prg, 'trans_count')(
        q, (max(1, int(A.nnz)),), None, aind.data, cnt.data,
        np.uint32(A.nnz))
    tptr = np.zeros(ncol + 1, np.int32)
    np.cumsum(cnt.get(queue=q), out=tptr[1:])
    tptr_d = ocl_core.to_device(tptr)
    cursor = ocl_core.to_device(tptr[:-1].copy())
    tind = ocl_core.zeros((max(1, int(A.nnz)),), np.int32)
    tdat = ocl_core.zeros((max(1, int(A.nnz)),), dt)
    ocl_core.kernel(prg, 'trans_scatter')(
        q, (max(1, nrow),), None, aptr.data, aind.data, adat.data,
        cursor.data, tind.data, tdat.data, np.uint32(nrow))
    ocl_core.kernel(prg, 'trans_sort')(
        q, (max(1, ncol),), None, tptr_d.data, tind.data, tdat.data,
        np.uint32(ncol))
    return DeviceCSR(tptr_d, tind, tdat, (ncol, nrow), int(A.nnz))


def spgemm_device(A, B, dtype=np.float32, maxc=64):
    """``A @ B`` as a :class:`DeviceCSR`.

    ``A`` and ``B`` may be scipy matrices or :class:`DeviceCSR`, so a
    chain such as the Gram stays on the card from end to end.
    """
    dt = np.dtype(dtype)
    q = ocl_core.queue()
    prg = _sp_program(dt, maxc)

    def parts(M):
        if isinstance(M, DeviceCSR):
            return M.indptr, M.indices, M.data, M.shape
        return (ocl_core.to_device(M.indptr.astype(np.int32)),
                ocl_core.to_device(M.indices.astype(np.int32)),
                ocl_core.to_device(M.data.astype(dt)),
                tuple(int(v) for v in M.shape))

    aptr, aind, adat, ashape = parts(A)
    bptr, bind, bdat, bshape = parts(B)
    nrow = int(ashape[0])
    over = ocl_core.zeros((1,), np.int32)
    cnnz = ocl_core.zeros((nrow,), np.int32)
    ocl_core.kernel(prg, 'spgemm_count')(
        q, (max(1, nrow),), None, aptr.data, aind.data, bptr.data,
        bind.data, cnnz.data, over.data, np.uint32(nrow))
    if int(over.get(queue=q)[0]):
        raise SpGEMMOverflow(
            "an output row needs more than %d distinct columns" % maxc)
    cptr = np.zeros(nrow + 1, np.int32)
    np.cumsum(cnnz.get(queue=q), out=cptr[1:])
    nnz = int(cptr[-1])
    cptr_d = ocl_core.to_device(cptr)
    cind = ocl_core.zeros((max(1, nnz),), np.int32)
    cdat = ocl_core.zeros((max(1, nnz),), dt)
    ocl_core.kernel(prg, 'spgemm_fill')(
        q, (max(1, nrow),), None, aptr.data, aind.data, adat.data,
        bptr.data, bind.data, bdat.data, cptr_d.data, cind.data,
        cdat.data, over.data, np.uint32(nrow))
    if int(over.get(queue=q)[0]):
        raise SpGEMMOverflow(
            "an output row needs more than %d distinct columns" % maxc)
    return DeviceCSR(cptr_d, cind, cdat, (nrow, int(bshape[1])), nnz)


def gram_device(basis, dtype=np.float32, maxc=64):
    """``Y @ Y.T`` formed and kept on the device.

    This is the level-0 operator when no certified stencil is
    available. Forming it on the card rather than with scipy keeps a
    Gram-sized matrix off the host entirely, which is the whole point:
    on R4 it would be gigabytes.
    """
    return spgemm_device(basis, transpose_device(basis, dtype), dtype,
                         maxc)


def transpose(A, dtype=np.float32):
    """``A.T`` as a scipy CSR, formed on the device."""
    import scipy.sparse as sp
    dt = np.dtype(dtype)
    q = ocl_core.queue()
    prg = _sp_program(dt, 64)
    nrow, ncol = (int(v) for v in A.shape)
    aptr = ocl_core.to_device(A.indptr.astype(np.int32))
    aind = ocl_core.to_device(A.indices.astype(np.int32))
    adat = ocl_core.to_device(A.data.astype(dt))
    cnt = ocl_core.zeros((ncol,), np.int32)
    ocl_core.kernel(prg, 'trans_count')(
        q, (max(1, int(A.nnz)),), None, aind.data, cnt.data,
        np.uint32(A.nnz))
    counts = cnt.get(queue=q)
    tptr = np.zeros(ncol + 1, np.int32)
    np.cumsum(counts, out=tptr[1:])
    tptr_d = ocl_core.to_device(tptr)
    cursor = ocl_core.to_device(tptr[:-1].copy())
    tind = ocl_core.zeros((max(1, int(A.nnz)),), np.int32)
    tdat = ocl_core.zeros((max(1, int(A.nnz)),), dt)
    ocl_core.kernel(prg, 'trans_scatter')(
        q, (max(1, nrow),), None, aptr.data, aind.data, adat.data,
        cursor.data, tind.data, tdat.data, np.uint32(nrow))
    ocl_core.kernel(prg, 'trans_sort')(
        q, (max(1, ncol),), None, tptr_d.data, tind.data, tdat.data,
        np.uint32(ncol))
    return sp.csr_matrix((tdat.get(queue=q), tind.get(queue=q), tptr),
                         shape=(ncol, nrow))


def spgemm(A, B, dtype=np.float32, maxc=64):
    """``A @ B`` as a scipy CSR, formed on the device.

    ``maxc`` bounds the distinct columns in any output row. A row that
    exceeds it raises :class:`SpGEMMOverflow`; the caller widens or
    falls back rather than getting a wrong answer.
    """
    import scipy.sparse as sp
    dt = np.dtype(dtype)
    q = ocl_core.queue()
    prg = _sp_program(dt, maxc)
    nrow = int(A.shape[0])
    aptr = ocl_core.to_device(A.indptr.astype(np.int32))
    aind = ocl_core.to_device(A.indices.astype(np.int32))
    adat = ocl_core.to_device(A.data.astype(dt))
    bptr = ocl_core.to_device(B.indptr.astype(np.int32))
    bind = ocl_core.to_device(B.indices.astype(np.int32))
    bdat = ocl_core.to_device(B.data.astype(dt))
    over = ocl_core.zeros((1,), np.int32)
    cnnz = ocl_core.zeros((nrow,), np.int32)
    ocl_core.kernel(prg, 'spgemm_count')(
        q, (max(1, nrow),), None, aptr.data, aind.data, bptr.data,
        bind.data, cnnz.data, over.data, np.uint32(nrow))
    if int(over.get(queue=q)[0]):
        raise SpGEMMOverflow(
            "an output row needs more than %d distinct columns" % maxc)
    counts = cnnz.get(queue=q)
    cptr = np.zeros(nrow + 1, np.int32)
    np.cumsum(counts, out=cptr[1:])
    nnz = int(cptr[-1])
    cptr_d = ocl_core.to_device(cptr)
    cind = ocl_core.zeros((max(1, nnz),), np.int32)
    cdat = ocl_core.zeros((max(1, nnz),), dt)
    ocl_core.kernel(prg, 'spgemm_fill')(
        q, (max(1, nrow),), None, aptr.data, aind.data, adat.data,
        bptr.data, bind.data, bdat.data, cptr_d.data, cind.data,
        cdat.data, over.data, np.uint32(nrow))
    if int(over.get(queue=q)[0]):
        raise SpGEMMOverflow(
            "an output row needs more than %d distinct columns" % maxc)
    C = sp.csr_matrix((cdat.get(queue=q)[:nnz],
                       cind.get(queue=q)[:nnz], cptr),
                      shape=(nrow, int(B.shape[1])))
    # scipy's product drops entries that cancel to exactly zero, and
    # the hierarchy should not depend on which backend assembled it,
    # so the structure is canonicalised the same way
    C.eliminate_zeros()
    return C


class CSR(object):
    """A CSR matrix resident on the device, with deterministic products."""

    WG = 64

    def __init__(self, M, dtype=np.float32):
        import scipy.sparse as sp
        self.dtype = np.dtype(dtype)
        if isinstance(M, DeviceCSR):
            # already on the card (a Gram formed there); adopt it
            self.shape, self.nnz = M.shape, M.nnz
            self.data, self.indices, self.indptr = (M.data, M.indices,
                                                    M.indptr)
            self.src_dtype, self.ones_only, self.int8_ok = '?', False, False
            self.prg = program(self.dtype, self.WG)
            self._k = {n: ocl_core.kernel(self.prg, n)
                       for n in ('csr_spmv', 'csr_jacobi', 'csr_residual',
                                 'csr_spmv_add')}
            return
        M = sp.csr_matrix(M)
        self.shape = tuple(int(v) for v in M.shape)
        self.nnz = int(M.nnz)
        # what the host held, and whether every stored value is 1: an
        # aggregation prolongator is 0/1 with one entry per row, so it
        # needs no data array at all
        self.src_dtype = str(M.data.dtype)
        self.ones_only = bool(M.nnz and np.all(M.data == 1))
        self.int8_ok = bool(M.nnz and np.all(M.data == np.rint(M.data))
                            and np.abs(M.data).max() <= 127)
        self.data = ocl_core.to_device(M.data.astype(self.dtype))
        self.indices = ocl_core.to_device(M.indices.astype(np.int32))
        self.indptr = ocl_core.to_device(M.indptr.astype(np.int32))
        self.prg = program(self.dtype, self.WG)
        self._k = {n: ocl_core.kernel(self.prg, n)
                   for n in ('csr_spmv', 'csr_jacobi', 'csr_residual',
                             'csr_spmv_add')}

    def device_bytes(self):
        return int(self.data.nbytes + self.indices.nbytes
                   + self.indptr.nbytes)

    def _grid(self):
        return (self.shape[0]*self.WG,), (self.WG,)

    def spmv(self, x, y):
        """``y = A x``."""
        g, l = self._grid()
        self._k['csr_spmv'](ocl_core.queue(), g, l, self.data.data,
                            self.indices.data, self.indptr.data, x.data,
                            y.data, np.uint32(self.shape[0]))
        return y

    def spmv_add(self, x, y):
        """``y += A x``."""
        g, l = self._grid()
        self._k['csr_spmv_add'](ocl_core.queue(), g, l, self.data.data,
                                self.indices.data, self.indptr.data, x.data,
                                y.data, np.uint32(self.shape[0]))
        return y

    def residual(self, x, b, r):
        """``r = b - A x``."""
        g, l = self._grid()
        self._k['csr_residual'](ocl_core.queue(), g, l, self.data.data,
                                self.indices.data, self.indptr.data, x.data,
                                b.data, r.data, np.uint32(self.shape[0]))
        return r

    def jacobi(self, x, b, dinv, xout, omega):
        """``xout = x + omega*dinv*(b - A x)``, one fused sweep."""
        g, l = self._grid()
        self._k['csr_jacobi'](ocl_core.queue(), g, l, self.data.data,
                              self.indices.data, self.indptr.data, x.data,
                              b.data, dinv.data, xout.data,
                              self.dtype.type(omega),
                              np.uint32(self.shape[0]))
        return xout


class OnesProlong(object):
    """An aggregation prolongator: one column index per row, no more.

    Dropping the all-ones data array and the row pointer takes level
    zero's prolongation from 595 MB to 198 at R5. Raises if the matrix
    is not one entry per row, so the caller keeps the general form.
    """

    def __init__(self, P, dtype=np.float32):
        import scipy.sparse as sp
        M = sp.csr_matrix(P)
        counts = np.diff(M.indptr)
        if M.nnz and (counts.max() != 1 or counts.min() != 1
                      or not np.all(M.data == 1)):
            raise ValueError("not a one-per-row 0/1 prolongator")
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(v) for v in M.shape)
        self.nnz = int(M.nnz)
        self.src_dtype, self.ones_only, self.int8_ok = \
            str(M.data.dtype), True, True
        self.col = ocl_core.to_device(M.indices.astype(np.int32))
        self._k = ocl_core.kernel(program(self.dtype, CSR.WG),
                                  'ones_gather_add')

    def device_bytes(self):
        return int(self.col.nbytes)

    def spmv_add(self, x, y):
        """``y += P x``."""
        n = int(self.shape[0])
        self._k(ocl_core.queue(), (n,), None, self.col.data, x.data,
                y.data, np.uint32(n))
        return y


class OnesRestrict(object):
    """The transpose of an aggregation prolongator: a segmented sum."""

    def __init__(self, P, dtype=np.float32):
        import scipy.sparse as sp
        M = sp.csr_matrix(sp.csr_matrix(P).T)
        if M.nnz and not np.all(M.data == 1):
            raise ValueError("not a 0/1 restriction")
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(v) for v in M.shape)
        self.nnz = int(M.nnz)
        self.src_dtype, self.ones_only, self.int8_ok = \
            str(M.data.dtype), True, True
        self.indptr = ocl_core.to_device(M.indptr.astype(np.int32))
        self.indices = ocl_core.to_device(M.indices.astype(np.int32))
        self._k = ocl_core.kernel(program(self.dtype, CSR.WG),
                                  'ones_rowsum')

    def device_bytes(self):
        return int(self.indptr.nbytes + self.indices.nbytes)

    def spmv(self, x, y):
        """``y = P^T x``."""
        n = int(self.shape[0])
        self._k(ocl_core.queue(), (n,), None, self.indptr.data,
                self.indices.data, x.data, y.data, np.uint32(n))
        return y


def program(dtype=np.float32, wg=64):
    """The sparse program for a scalar type and work-group width."""
    return ocl_core.program(SOURCE, _DT[np.dtype(dtype)], {'WG': int(wg)},
                            key='ocl_sparse')


def gather(src, idx, dst, dtype=np.float32, wg=64):
    """``dst[i] = src[idx[i]]``."""
    n = int(idx.size)
    k = ocl_core.kernel(program(dtype, wg), 'gather_idx')
    k(ocl_core.queue(), (n,), None, src.data, idx.data, dst.data,
      np.uint32(n))
    return dst


def scatter(src, idx, dst, dtype=np.float32, wg=64):
    """``dst[idx[i]] = src[i]``."""
    n = int(idx.size)
    k = ocl_core.kernel(program(dtype, wg), 'scatter_idx')
    k(ocl_core.queue(), (n,), None, src.data, idx.data, dst.data,
      np.uint32(n))
    return dst


NBLOCK = 256              # work groups per reduction; fixed, so the
                          # partial-sum order is fixed too


def dot(a, b, dtype=np.float32, wg=64):
    """``a . b`` as a Python float, by a fixed-shape reduction."""
    n = int(a.size)
    part = ocl_core.zeros((NBLOCK,), dtype)
    ocl_core.kernel(program(dtype, wg), 'red_dot')(
        ocl_core.queue(), (NBLOCK*wg,), (wg,), a.data, b.data, part.data,
        np.uint32(n))
    return float(np.sum(part.get(queue=ocl_core.queue()), dtype=np.float64))


def maxabs(a, dtype=np.float32, wg=64):
    """``max |a|`` as a Python float."""
    n = int(a.size)
    part = ocl_core.zeros((NBLOCK,), dtype)
    ocl_core.kernel(program(dtype, wg), 'red_maxabs')(
        ocl_core.queue(), (NBLOCK*wg,), (wg,), a.data, part.data,
        np.uint32(n))
    return float(np.max(part.get(queue=ocl_core.queue())))


def dense_gemv(A, x, y, m, n, dtype=np.float32, wg=64):
    """``y = A x`` for a row-major dense A."""
    k = ocl_core.kernel(program(dtype, wg), 'dense_gemv')
    k(ocl_core.queue(), (int(m)*int(wg),), (int(wg),), A.data, x.data,
      y.data, np.uint32(m), np.uint32(n))
    return y
