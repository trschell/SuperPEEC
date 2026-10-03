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
import os

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


def _scale(v, a):
    """v *= a on the device."""
    v *= v.dtype.type(a)


def _axpy(y, a, x):
    """y += a*x on the device."""
    y += x if a == 1.0 else x*y.dtype.type(a)


class LaplaceMG(object):
    """Geometric multigrid V-cycle for the masked node Laplacian, as the
    preconditioner of :func:`laplacian_current`'s CG (2026-10-02).

    WHY. Jacobi-preconditioned CG needs iterations proportional to the
    lattice's linear size: 3.3k at R3, 6.7k at R4, ~26k at R6 -- an hour
    of the R6 build. With this V-cycle: 26 / 33 iterations at R3 / R4,
    the same current to the KCL tolerance.

    THE HIERARCHY. Occupied cells are aggregated 2 x 2 x 2 per level,
    NEVER across connected components (a block straddling the gap
    between two conductors couples them in the coarse operator: 621
    iterations at R4 against 77 when split), with piecewise-constant
    prolongation P and Galerkin coarse operators P^T A P. Those are
    7-POINT operators again -- aggregates of a nearest-neighbour graph
    touch only face-adjacent aggregates -- so each level is built from
    the previous level's EDGE WEIGHTS in chunks: an aggregate's
    diagonal is its members' diagonals minus twice its internal edge
    weight, and a face weight is the sum of the edges crossing it. The
    fine Laplacian is never formed in floating point on the host and no
    sparse product is taken.

    THE CYCLE. Damped Jacobi (omega 0.6, nu sweeps) before and after,
    the coarse correction scaled by alpha (1.8: plain aggregation's
    coarse correction is too weak, 77 -> 33 iterations at R4), dense
    inverse on the coarsest level; symmetric, so CG stays valid. Level 0
    is the CG's own device Laplacian and diagonal, its input and output
    the CG's r and z, its Jacobi scratch the caller's spare vector: the
    only level-0 additions are the aggregation member lists and one
    residual vector. Deterministic kernels throughout.
    """

    def __init__(self, B, parent, cells, comp, dt, A0, dinv0, nu=2,
                 omega=0.6, alpha=None, coarse_n=1000):
        import scipy.sparse as sp
        self.dt = np.dtype(dt)
        self.nu = int(os.environ.get('SPPEEC_IHAT_MG_NU', nu))
        self.omega = float(os.environ.get('SPPEEC_IHAT_MG_OMEGA', omega))
        self.alpha = float(alpha if alpha is not None else
                           os.environ.get('SPPEEC_IHAT_MG_ALPHA', '1.8'))
        B = sp.csr_matrix(B)
        nn = int(B.shape[1])
        d = (np.asarray(parent) >= 0).astype(np.float64)
        pair = B.indices.reshape(-1, 2)
        CH = 1 << 24
        # level 0 as (diagonal, edge chunks): D_i = d_i deg_i + (1 - d_i),
        # an edge (lo, hi) of weight d_lo d_hi
        deg = np.zeros(nn)
        for a0 in range(0, pair.shape[0], CH):
            p = pair[a0:a0 + CH]
            deg += np.bincount(p[:, 0], minlength=nn)
            deg += np.bincount(p[:, 1], minlength=nn)
        D = d*deg + (1.0 - d)
        del deg

        def edges0():
            for a0 in range(0, pair.shape[0], CH):
                p = pair[a0:a0 + CH].astype(np.int64)
                w = d[p[:, 0]]*d[p[:, 1]]
                k = w != 0
                yield p[k, 0], p[k, 1], w[k]

        c = np.asarray(cells, np.int64)
        g = (np.asarray(comp, np.int64) if comp is not None
             and os.environ.get('SPPEEC_IHAT_MG_COMP', '1') != '0'
             else np.zeros(nn, np.int64))
        g = g - g.min()
        self.levels = []
        edges = edges0
        Afine, dinvfine = A0, dinv0
        n = nn
        while n > coarse_n and len(self.levels) < 20:
            cc = c//2
            m = cc.max(axis=0) + 1
            span = int(m[0])*int(m[1])*int(m[2])
            key = g*span + (cc[:, 0]*m[1] + cc[:, 1])*m[2] + cc[:, 2]
            uk, agg = np.unique(key, return_inverse=True)
            nc = int(uk.size)
            if nc >= n:
                break
            rem = uk % span
            ccell = np.stack([rem//(m[1]*m[2]), (rem//m[2]) % m[1],
                              rem % m[2]], axis=1)
            # coarse diagonal and face weights from the edges
            Dc = np.bincount(agg, weights=D, minlength=nc)
            Wf = np.zeros((nc, 3))            # weight to the +axis face
            for i, j, w in edges():
                ai, aj = agg[i], agg[j]
                same = ai == aj
                Dc -= 2.0*np.bincount(ai[same], weights=w[same],
                                      minlength=nc)
                ai, aj, w = ai[~same], aj[~same], w[~same]
                dd = ccell[aj] - ccell[ai]
                ax = np.argmax(np.abs(dd), axis=1)
                sg = dd[np.arange(dd.shape[0]), ax]
                if not (np.abs(dd).sum(axis=1) == 1).all():
                    raise RuntimeError("aggregate edge is not a face step")
                lo = np.where(sg > 0, ai, aj)
                np.add.at(Wf, (lo, ax), w)
            # the coarse edges: aggregate -> its +axis neighbour (same
            # component, so the same key block)
            ea, eb, ew = [], [], []
            for ax in range(3):
                a = np.flatnonzero(Wf[:, ax] != 0)
                if a.size == 0:
                    continue
                nb = ccell[a].copy()
                nb[:, ax] += 1
                kb = (uk[a] - rem[a]) + (nb[:, 0]*m[1] + nb[:, 1])*m[2] \
                    + nb[:, 2]
                b = np.searchsorted(uk, kb)
                if not (uk[np.minimum(b, nc - 1)] == kb).all():
                    raise RuntimeError("missing face neighbour aggregate")
                ea.append(a)
                eb.append(b)
                ew.append(Wf[a, ax])
            ea = np.concatenate(ea) if ea else np.zeros(0, np.int64)
            eb = np.concatenate(eb) if eb else np.zeros(0, np.int64)
            ew = np.concatenate(ew) if ew else np.zeros(0)
            del Wf
            # level record: the fine operator (device), its aggregation
            P = sp.csr_matrix((np.ones(n), (np.arange(n), agg)),
                              shape=(n, nc))
            lv = dict(n=n, A=Afine, dinv=dinvfine,
                      R=ocl_sparse.OnesRestrict(P, self.dt))
            del P
            self.levels.append(lv)
            # the coarse operator for the next level
            Ac = sp.csr_matrix(
                (np.concatenate([Dc, -ew, -ew]),
                 (np.concatenate([np.arange(nc), ea, eb]),
                  np.concatenate([np.arange(nc), eb, ea]))),
                shape=(nc, nc))
            n = nc
            D = Dc
            c, g = ccell, uk//span
            edges = (lambda ea=ea, eb=eb, ew=ew: iter([(ea, eb, ew)]))
            if n > coarse_n:
                Afine = ocl_sparse.CSR(Ac, self.dt)
                dinvfine = ocl_core.to_device(1.0/np.where(Dc != 0, Dc, 1.0),
                                              self.dt)
            self._Ac = Ac
        self.nc = n
        Ainv = np.linalg.inv(self._Ac.toarray()) if self.levels else None
        del self._Ac
        self.Ainv = ocl_core.to_device(Ainv, self.dt)
        # level 0 keeps no residual: it is restricted as it is formed
        for i, lv in enumerate(self.levels):
            for k in (() if i == 0 else ('x', 'y', 'b', 'r')):
                lv[k] = ocl_core.zeros((lv['n'],), self.dt)
        self.cb = ocl_core.zeros((self.nc,), self.dt)
        self.cx = ocl_core.zeros((self.nc,), self.dt)
        self.nlev = len(self.levels) + 1

    def device_bytes(self):
        n = self.Ainv.nbytes + self.cb.nbytes + self.cx.nbytes
        for i, lv in enumerate(self.levels):
            n += lv['R'].device_bytes() + sum(lv[k].nbytes for k in
                                              ('x', 'y', 'b', 'r') if k in lv)
            if i > 0:
                n += lv['A'].device_bytes() + lv['dinv'].nbytes
        return int(n)

    def _cycle(self, l, b, x, y):
        q = ocl_core.queue()
        if l == len(self.levels):
            ocl_sparse.dense_gemv(self.Ainv, b, x, self.nc, self.nc,
                                  self.dt)
            return x
        lv = self.levels[l]
        A, dinv = lv['A'], lv['dinv']
        x.fill(self.dt.type(0), queue=q)
        for _ in range(self.nu):
            A.jacobi(x, b, dinv, y, self.omega)
            _copy(x, y)
        nxt = self.levels[l + 1] if l + 1 < len(self.levels) else None
        bc = nxt['b'] if nxt else self.cb
        xc = nxt['x'] if nxt else self.cx
        if 'r' in lv:
            A.residual(x, b, lv['r'])
            lv['R'].spmv(lv['r'], bc)
        else:
            A.residual_restrict(x, b, lv['R'], bc)
        self._cycle(l + 1, bc, xc, nxt['y'] if nxt else None)
        if self.alpha != 1.0:
            _scale(xc, self.alpha)
        lv['R'].prolong_add_tiled(xc, x)
        for _ in range(self.nu):
            A.jacobi(x, b, dinv, y, self.omega)
            _copy(x, y)
        return x

    def apply(self, r, z, scratch):
        """z = M^-1 r; ``scratch`` is a free level-0 vector."""
        return self._cycle(0, r, z, scratch)


def laplacian_current(B, parent, rhs, tol=1e-12, maxiter=50000,
                      cells=None, comp=None):
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
    mg = None
    if cells is not None and os.environ.get('SPPEEC_IHAT_MG', '1') != '0':
        import time as _time
        _t = _time.perf_counter()
        mg = LaplaceMG(Bc, parent, cells, comp, dt, A0=Lg, dinv0=dinv)
        if os.environ.get('SPPEEC_STREAM_VERBOSE') == '1':
            print("    ihat MG: %d levels, coarsest %d, %.1f MB card, "
                  "built in %.1f s" % (mg.nlev, mg.nc,
                                       mg.device_bytes()/1e6,
                                       _time.perf_counter() - _t),
                  flush=True)

    def precond(rr, zz):
        if mg is not None:
            # Ap is free here: the update that used it is done and the
            # next spmv overwrites it
            return mg.apply(rr, zz, Ap)
        k['vmul'](q, (nn,), None, zz.data, dinv.data, rr.data,
                  np.uint32(nn))
        return zz

    precond(r, z)
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
        if it % (1 if mg is not None else 50) == 0:
            if float(np.sqrt(ocl_sparse.dot(r, r, dt))) <= tol*bn:
                break
        precond(r, z)
        rz_new = ocl_sparse.dot(r, z, dt)
        k['xpay'](q, (nn,), None, p.data, z.data, dt.type(rz_new/rz),
                  np.uint32(nn))
        rz = rz_new

    if os.environ.get('SPPEEC_STREAM_VERBOSE') == '1':
        print("    ihat CG: %d iterations (%s)" % (it, 'MG' if mg is not None
                                                  else 'Jacobi'), flush=True)
    del Lg, Ld, mg           # the iteration is over; the Laplacian is dead
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
