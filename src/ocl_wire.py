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

One matrix at a time
--------------------
The incidence matrix, its transpose and the Laplacian are each wanted
in a different part of this routine, and holding all three at once put
1.14 GiB on the card at R4 and 4.55 GiB at R5 -- which, once the solve
phase was narrowed to single precision, became the whole run's
high-water mark. So each is built where it is needed and dropped where
it is not: the transpose goes as soon as the Laplacian is formed, the
Laplacian as soon as the iteration ends, and the incidence matrix is
not uploaded until there is a solution to multiply. The transpose is
rebuilt for the final KCL check, which costs one pass against a solve
of hundreds of iterations.
"""
import numpy as np

import ocl_core
import ocl_sparse


def _copy(dst, src):
    """Device-to-device copy of a whole array."""
    import pyopencl as cl
    cl.enqueue_copy(ocl_core.queue(), dst.data, src.data,
                    byte_count=int(src.nbytes))

SOURCE = """
/* The Laplacian's values are node degrees and minus ones, and the
   incidence matrix's are plus and minus ones: they travel as single
   bytes (data_t) while every vector stays double (2026-09-25). The
   arithmetic is unchanged -- a byte widened to double is the double
   that was stored before. */
typedef DATA data_t;

/* Dirichlet in place: scale each entry by d[i]*d[j], then a root's
   diagonal becomes one. A root has d = 0, so its row and column
   vanish and the added (1 - d) leaves a clean unit pivot. */
__kernel void dirichlet(__global const int *ptr,
                        __global const int *ind,
                        __global data_t *dat,
                        __global const real_t *d,
                        const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const real_t di = d[i];
    for (int p = ptr[i]; p < ptr[i + 1]; ++p) {
        const int j = ind[p];
        real_t v = di*d[j]*(real_t)dat[p];
        if ((unsigned int)j == i) v += (real_t)1 - di;
        dat[p] = (data_t)v;
    }
}

__kernel void diag_inv(__global const int *ptr,
                       __global const int *ind,
                       __global const data_t *dat,
                       __global real_t *out,
                       const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    real_t v = (real_t)0;
    for (int p = ptr[i]; p < ptr[i + 1]; ++p)
        if ((unsigned int)ind[p] == i) v = (real_t)dat[p];
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


def _program(dt, data8=True):
    real = 'float' if np.dtype(dt) == np.float32 else 'double'
    data = 'char' if data8 else real
    return ocl_core.program(SOURCE, ocl_sparse._DT[np.dtype(dt)],
                            {'DATA': data}, key='ocl_wire/' + data)


def _laplacian_host(B, chunk=1 << 24):
    """``B^T B`` for a filament incidence with exactly two entries
    (+1, -1) per row, as a CSR with int8 values, int32 indices, sorted
    columns: the node degrees on the diagonal, -1 per filament off
    it. Built by a counting sort over the 2E off-diagonal entries plus
    the diagonal, in chunks, with no sparse product and no int64
    temporaries the size of the incidence."""
    import scipy.sparse as sp
    Bc = B.tocsr()
    if not np.all(np.diff(Bc.indptr) == 2):
        raise ValueError("incidence rows must have exactly two entries")
    ne, nn = (int(v) for v in Bc.shape)
    pair = Bc.indices.reshape(ne, 2)
    lo = pair[:, 0]
    hi = pair[:, 1]
    del pair
    deg = np.bincount(lo, minlength=nn) + np.bincount(hi, minlength=nn)
    counts = deg.astype(np.int64) + 1                 # neighbours + self
    indptr = np.zeros(nn + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    if nnz >= (1 << 31):
        raise OverflowError("Laplacian past int32 indexing")
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.int8)
    fill = indptr[:-1].copy()
    # the diagonal first, then each filament's two off-diagonals, in
    # chunks: a counting sort by row, columns sorted afterwards
    indices[fill] = np.arange(nn, dtype=np.int32)
    data[fill] = np.minimum(deg, 127).astype(np.int8)
    if int(deg.max()) > 127:
        raise OverflowError("node degree past a byte")
    fill += 1
    for a0 in range(0, ne, chunk):
        a1 = min(ne, a0 + chunk)
        for r, c in ((lo[a0:a1], hi[a0:a1]), (hi[a0:a1], lo[a0:a1])):
            # stable placement: rows repeat inside a chunk, so the slot
            # of each entry is fill[r] plus its rank among equal rows
            srt = np.argsort(r, kind='stable')
            rs = r[srt]
            st = np.flatnonzero(np.r_[True, rs[1:] != rs[:-1]])
            ln = np.diff(np.r_[st, rs.size])
            within = np.arange(rs.size, dtype=np.int64) - np.repeat(st, ln)
            pos = np.repeat(fill[rs[st]], ln) + within
            indices[pos] = c[srt]
            data[pos] = -1
            fill[rs[st]] += ln
            del srt, rs, st, ln, within, pos
    L = sp.csr_matrix((data, indices, indptr.astype(np.int32)),
                      shape=(nn, nn))
    L.sort_indices()
    return L


def _laplacian_device(B, dt):
    """The host-assembled byte Laplacian, uploaded as a DeviceCSR."""
    L = _laplacian_host(B)
    return ocl_sparse.DeviceCSR(ocl_core.to_device(L.indptr),
                                ocl_core.to_device(L.indices),
                                ocl_core.to_device(L.data),
                                L.shape, L.nnz)


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

    # The Laplacian assembled on the host from the incidence's two
    # entries per row and uploaded alone (2026-09-29): B^T B was formed
    # ON THE DEVICE from an uploaded B and B^T -- 6.8 GB of the pair at
    # R6 beside the 3.5 GB result and the tree's resident state, which
    # is the allocation the card refused on the first R6 run (falling
    # back to a host CG that ran for hours). L = D - A needs no product:
    # a filament (lo, hi) is -1 at (lo, hi) and (hi, lo) and adds one
    # to both degrees. Bytes for the values as before, columns sorted
    # within each row, so the device sums each row in the same order
    # the product's rows had.
    Bc = B.tocsr()
    Ld = _laplacian_device(Bc, dt)
    nn = int(Ld.shape[0])
    d = ocl_core.to_device(np.asarray(parent >= 0, dt))   # zero at roots
    k['dirichlet'](q, (nn,), None, Ld.indptr.data, Ld.indices.data,
                   Ld.data.data, d.data, np.uint32(nn))
    Lg = ocl_sparse.CSR(Ld, dt)
    dinv = ocl_core.zeros((nn,), dt)
    k['diag_inv'](q, (nn,), None, Ld.indptr.data, Ld.indices.data,
                  Ld.data.data, dinv.data, np.uint32(nn))

    b0 = ocl_core.to_device(np.asarray(rhs, dt))
    # the masked right-hand side IS the initial residual at x = 0, so
    # one vector serves both and its norm is the convergence scale
    r = ocl_core.zeros((nn,), dt)
    k['vmul'](q, (nn,), None, r.data, b0.data, d.data, np.uint32(nn))
    bn = float(np.sqrt(ocl_sparse.dot(r, r, dt)))

    # the mask and the right-hand side have done their work (the check
    # runs on the host now): two node vectors fewer through the
    # iteration, 1.4 GB at R6 on a card that is full there
    del d, b0
    x = ocl_core.zeros((nn,), dt)
    z = ocl_core.zeros((nn,), dt)
    p = ocl_core.zeros((nn,), dt)
    Ap = ocl_core.zeros((nn,), dt)
    k['vmul'](q, (nn,), None, z.data, dinv.data, r.data, np.uint32(nn))
    _copy(p, z)              # device to device, not out through the host
    rz = ocl_sparse.dot(r, z, dt)

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

    del Lg, Ld               # the iteration is over; the Laplacian is dead
    # The potential is done; the current and the KCL check run on the
    # HOST (2026-09-29): they used to upload B and then B^T again --
    # 10 GB at R6 beside the solve's vectors, the second allocation
    # the card refuses there. A filament's current is the two-term
    # difference of its end potentials, computed as the device kernel
    # did (v0*x0 + v1*x1, exact plus or minus ones), so the bits are
    # the same; the check is a threshold on a maximum.
    xh = x.get(queue=q)
    del x, r, z, p, Ap, dinv
    ne = int(Bc.shape[0])
    pair = Bc.indices.reshape(ne, 2)
    vals = Bc.data.reshape(ne, 2)
    ihat = np.empty(ne, dt)
    chk = -np.asarray(rhs, dt)              # B^T ihat - rhs, accumulated
    CH = 1 << 24
    for a0 in range(0, ne, CH):
        a1 = min(ne, a0 + CH)
        c0, c1 = pair[a0:a1, 0], pair[a0:a1, 1]
        v0 = vals[a0:a1, 0].astype(dt)
        v1 = vals[a0:a1, 1].astype(dt)
        blk = v0*xh[c0] + v1*xh[c1]
        ihat[a0:a1] = blk
        chk += np.bincount(c0, weights=v0*blk, minlength=nn)
        chk += np.bincount(c1, weights=v1*blk, minlength=nn)
        del c0, c1, v0, v1, blk
    resid = float(np.abs(chk).max())
    return ihat, resid
