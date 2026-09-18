# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The wire-bond particular current on OpenCL.

Solves ``(B^T B) phi = rhs`` for the node potentials on the wire graph
and returns the filament current ``ihat = B phi``, with the tree roots
held at zero. Everything stays on the card: the incidence transpose and
the graph Laplacian are formed there, so the host never holds either.

The Laplacian is assembled from the device sparse product, and the
Dirichlet condition is applied in place rather than through two more
sparse products. Zeroing a root's row and column and putting one on its
diagonal is a scaling of each entry by ``d[i] d[j]`` plus a correction
on the diagonal, which is one pass over the entries instead of
``D L D`` assembled as matrix products.

The solve is a Jacobi-preconditioned conjugate gradient, the same
algorithm and the same convergence test as the CUDA path, with the
vector updates fused so a long run does not allocate a temporary per
step.
"""
import numpy as np

import ocl_core
import ocl_sparse

SOURCE = """
/* Dirichlet in place: scale each entry by d[i]*d[j], then a root's
   diagonal becomes one. A root has d = 0, so its row and column
   vanish and the added (1 - d) leaves a clean unit pivot. */
__kernel void dirichlet(__global const int *ptr,
                        __global const int *ind,
                        __global real_t *dat,
                        __global const real_t *d,
                        const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const real_t di = d[i];
    for (int p = ptr[i]; p < ptr[i + 1]; ++p) {
        const int j = ind[p];
        real_t v = di*d[j]*dat[p];
        if ((unsigned int)j == i) v += (real_t)1 - di;
        dat[p] = v;
    }
}

__kernel void diag_inv(__global const int *ptr,
                       __global const int *ind,
                       __global const real_t *dat,
                       __global real_t *out,
                       const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    real_t v = (real_t)0;
    for (int p = ptr[i]; p < ptr[i + 1]; ++p)
        if ((unsigned int)ind[p] == i) v = dat[p];
    out[i] = (v != (real_t)0) ? (real_t)1/v : (real_t)1;
}

__kernel void axpy(__global real_t *y, const real_t a,
                   __global const real_t *x, const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) y[i] += a*x[i];
}

__kernel void xpay(__global real_t *p, __global const real_t *z,
                   const real_t beta, const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) p[i] = z[i] + beta*p[i];
}

__kernel void vmul(__global real_t *out, __global const real_t *a,
                   __global const real_t *b, const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) out[i] = a[i]*b[i];
}
"""


def _program(dt):
    return ocl_core.program(SOURCE, ocl_sparse._DT[np.dtype(dt)],
                            key='ocl_wire')


def laplacian_current(B, parent, rhs, tol=1e-12, maxiter=50000):
    """``ihat = B phi`` with ``(B^T B) phi = rhs``, solved on the device.

    Returns ``(ihat on the host, max |B^T ihat - rhs|)``, the same
    contract as the CUDA path.
    """
    dt = np.dtype(np.float64)
    q = ocl_core.queue()
    prg = _program(dt)
    k = {n: ocl_core.kernel(prg, n)
         for n in ('dirichlet', 'diag_inv', 'axpy', 'xpay', 'vmul')}

    Bc = B.tocsr()
    Bd = ocl_sparse.CSR(Bc, dt)
    BTd_raw = ocl_sparse.transpose_device(Bc, dt)
    BTd = ocl_sparse.CSR(BTd_raw, dt)
    # a Laplacian row holds the node's neighbours plus itself, so the
    # bound is small; widen on overflow rather than sizing the private
    # array for the worst node in the graph
    maxc = 64
    while True:
        try:
            Ld = ocl_sparse.spgemm_device(BTd_raw, Bc, dt, maxc=maxc)
            break
        except ocl_sparse.SpGEMMOverflow:
            if maxc >= 256:
                raise
            maxc *= 2
    nn = int(Ld.shape[0])
    d = ocl_core.to_device(np.asarray(parent >= 0, dt))   # zero at roots
    k['dirichlet'](q, (nn,), None, Ld.indptr.data, Ld.indices.data,
                   Ld.data.data, d.data, np.uint32(nn))
    Lg = ocl_sparse.CSR(Ld, dt)
    dinv = ocl_core.zeros((nn,), dt)
    k['diag_inv'](q, (nn,), None, Ld.indptr.data, Ld.indices.data,
                  Ld.data.data, dinv.data, np.uint32(nn))

    b0 = ocl_core.to_device(np.asarray(rhs, dt))
    b = ocl_core.zeros((nn,), dt)
    k['vmul'](q, (nn,), None, b.data, b0.data, d.data, np.uint32(nn))

    x = ocl_core.zeros((nn,), dt)
    r = ocl_core.zeros((nn,), dt)
    r.set(b.get(queue=q), queue=q)
    z = ocl_core.zeros((nn,), dt)
    p = ocl_core.zeros((nn,), dt)
    Ap = ocl_core.zeros((nn,), dt)
    k['vmul'](q, (nn,), None, z.data, dinv.data, r.data, np.uint32(nn))
    p.set(z.get(queue=q), queue=q)
    rz = ocl_sparse.dot(r, z, dt)
    bn = float(np.sqrt(ocl_sparse.dot(b, b, dt)))

    it = 0
    while it < maxiter and bn > 0.0:
        Lg.spmv(p, Ap)
        pap = ocl_sparse.dot(p, Ap, dt)
        if pap == 0.0:
            break
        alpha = rz/pap
        k['axpy'](q, (nn,), None, x.data, dt.type(alpha), p.data,
                  np.uint32(nn))
        k['axpy'](q, (nn,), None, r.data, dt.type(-alpha), Ap.data,
                  np.uint32(nn))
        it += 1
        if it % 50 == 0:
            if float(np.sqrt(ocl_sparse.dot(r, r, dt))) <= tol*bn:
                break
        k['vmul'](q, (nn,), None, z.data, dinv.data, r.data, np.uint32(nn))
        rz_new = ocl_sparse.dot(r, z, dt)
        k['xpay'](q, (nn,), None, p.data, z.data, dt.type(rz_new/rz),
                  np.uint32(nn))
        rz = rz_new

    ihat = ocl_core.zeros((int(Bd.shape[0]),), dt)
    Bd.spmv(x, ihat)
    chk = ocl_core.zeros((nn,), dt)
    BTd.spmv(ihat, chk)
    k['axpy'](q, (nn,), None, chk.data, dt.type(-1.0), b0.data,
              np.uint32(nn))
    resid = ocl_sparse.maxabs(chk, dt)
    return ihat.get(queue=q), resid
