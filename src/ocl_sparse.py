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


class CSR(object):
    """A CSR matrix resident on the device, with deterministic products."""

    WG = 64

    def __init__(self, M, dtype=np.float32):
        import scipy.sparse as sp
        self.dtype = np.dtype(dtype)
        M = sp.csr_matrix(M)
        self.shape = tuple(int(v) for v in M.shape)
        self.nnz = int(M.nnz)
        self.data = ocl_core.to_device(M.data.astype(self.dtype))
        self.indices = ocl_core.to_device(M.indices.astype(np.int32))
        self.indptr = ocl_core.to_device(M.indptr.astype(np.int32))
        self.prg = program(self.dtype, self.WG)
        self._k = {n: ocl_core.kernel(self.prg, n)
                   for n in ('csr_spmv', 'csr_jacobi', 'csr_residual',
                             'csr_spmv_add')}

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


def dense_gemv(A, x, y, m, n, dtype=np.float32, wg=64):
    """``y = A x`` for a row-major dense A."""
    k = ocl_core.kernel(program(dtype, wg), 'dense_gemv')
    k(ocl_core.queue(), (int(m)*int(wg),), (int(wg),), A.data, x.data,
      y.data, np.uint32(m), np.uint32(n))
    return y
