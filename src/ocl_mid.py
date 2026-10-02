# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Mid-level multipole-to-local on OpenCL.

The host kernel (``mp_fortran.mid_m2l``) sums, for every box, the M2L
translation of its interaction-list boxes' multipole expansions:

    out[b, nm] = sum_k sum_jk  T[pos(b), k, nm, jk] * data[nb(b, k), jk]

with ``pos`` the box's position in its parent and ``nb`` its k-th far
neighbour. At nmax 4 that is 189 x 25 x 25 complex products per box --
~20 GFLOP per orientation at R4, the largest single line of the matvec
(2026-10-02 survey: 607 ms of 2.82 s). It is compute-bound on the host
and parallel per box, which is what the card is for.

CARD MEMORY. Nothing per filament, nothing per (box, neighbour): the
interaction list is NOT uploaded. Each work item finds its neighbours
through a box-id grid at leaf-box resolution and the fixed offset table
of its child position -- 2.4 MB of grid at R6 where the list would be
~150 MB. The grid and offsets are checked on the host to reproduce
``farneighbors`` exactly before first use. Resident: the transfer table
(15 MB complex128 at nmax 4, half in complex64), the grid, the box
coordinates and positions, and one in/out pair of level-data buffers.

PRECISION. Storage and accumulation follow
:func:`ocl_core.operator_dtype`. In complex64 the table is the
normalised one ``levels`` already designed for this port (the raw
entries carry 1/r**(j+n+1) and overflow float32 on fine pitches): with
u[jk] = r0**(j+1), v[nm] = r0**n the scaled table T*u*v is O(1), the
input is divided by u and the output by v, in fp64 on the host. The
scaled table is checked to fit float32 or the fp64 kernel is used.

Each work item owns one (box, nm) output and accumulates its neighbours
in the host kernel's order, so a call is reproducible call to call.
"""
import numpy as np

import ocl_core

SOURCE = """
/* M2M: parent[g, nm] = sum_children sum_jk M[nm, pos*NN + jk] * d[c, jk] */
__kernel void mid_m2m(__global const cplx_t *Mt,     /* (NN, NPAR*NN)  */
                      __global const cplx_t *d,      /* (nchild, NN)   */
                      __global const int *pos,       /* (nchild)       */
                      __global const int *i0,        /* (ngroup+1)     */
                      __global cplx_t *par,          /* (ngroup, NN)   */
                      const int ngroup)
{
    const int w = get_global_id(0);
    if (w >= ngroup*NN)
        return;
    const int g = w / NN;
    const int nm = w - g*NN;
    cplx_t acc = (cplx_t)(0, 0);
    for (int c = i0[g]; c < i0[g + 1]; c++) {
        __global const cplx_t *m = Mt + (size_t)nm*(NPAR*NN) + pos[c]*NN;
        __global const cplx_t *x = d + (size_t)c*NN;
        for (int jk = 0; jk < NN; jk++)
            acc += cmul(m[jk], x[jk]);
    }
    par[w] = acc;
}

/* L2L: d[c, nm] += sum_jk L[pos(c)*NN + nm, jk] * par[g(c), jk] */
__kernel void mid_l2l(__global const cplx_t *Lt,     /* (NPAR*NN, NN)  */
                      __global const cplx_t *par,    /* (ngroup, NN)   */
                      __global const int *pos,       /* (nchild)       */
                      __global const int *grp,       /* (nchild)       */
                      __global cplx_t *d,            /* (nchild, NN)   */
                      const int nchild)
{
    const int w = get_global_id(0);
    if (w >= nchild*NN)
        return;
    const int c = w / NN;
    const int nm = w - c*NN;
    __global const cplx_t *l = Lt + ((size_t)pos[c]*NN + nm)*NN;
    __global const cplx_t *x = par + (size_t)grp[c]*NN;
    cplx_t acc = (cplx_t)(0, 0);
    for (int jk = 0; jk < NN; jk++)
        acc += cmul(l[jk], x[jk]);
    d[w] += acc;
}

__kernel void mid_m2l(__global const cplx_t *T,      /* (NPAR,NFAR,NN,NN) */
                      __global const cplx_t *md,     /* (nbox, NN)        */
                      __global const int *boxc,      /* (nbox, 3)         */
                      __global const int *pos,       /* (nbox)            */
                      __global const int *offs,      /* (NPAR, NFAR, 3)   */
                      __global const int *grid,      /* (GX, GY, GZ)      */
                      __global cplx_t *out,          /* (nbox, NN)        */
                      const int nbox, const int GX, const int GY,
                      const int GZ)
{
    const int gid = get_global_id(0);
    if (gid >= nbox*NN)
        return;
    const int b = gid / NN;
    const int nm = gid - b*NN;
    const int p = pos[b];
    const int bx = boxc[3*b], by = boxc[3*b + 1], bz = boxc[3*b + 2];
    cplx_t acc = (cplx_t)(0, 0);
    for (int k = 0; k < NFAR; k++) {
        const int o = 3*(p*NFAR + k);
        const int x = bx + offs[o], y = by + offs[o + 1],
                  z = bz + offs[o + 2];
        if (x < 0 || y < 0 || z < 0 || x >= GX || y >= GY || z >= GZ)
            continue;
        const int nb = grid[(x*GY + y)*GZ + z];
        if (nb < 0)
            continue;
        /* numpy layout of levels' table: [pos][k][out][in] -- the
           host hands transfer.T to Fortran's TRANSFER(IN,OUT,K,POS) */
        __global const cplx_t *t = T + ((size_t)(p*NFAR + k)*NN + nm)*NN;
        __global const cplx_t *m = md + (size_t)nb*NN;
        for (int jk = 0; jk < NN; jk++)
            acc += cmul(t[jk], m[jk]);
    }
    out[gid] = acc;
}
"""


class MidM2L(object):
    """Device tables and apply for one mid level's M2L.

    ``level`` is the :class:`levels.MidLevel`; its data rows are the
    boxes of ``level.below`` (whose ``xidx/yidx/zidx`` give their
    lattice coordinates).
    """

    def __init__(self, level, dtype=None):
        below = level.below
        T = np.asarray(level.transfer)          # (npar, nfar, nn, nn)
        if getattr(level, '_mid_u', None) is not None:
            # SPPEEC_MID_FP32 stored the host table pre-scaled
            T = (T.astype(np.complex128)
                 / level._mid_u[None, None, :, None]
                 / level._mid_v[None, None, None, :])
        self.npar, self.nfar, self.nn = (int(T.shape[0]), int(T.shape[1]),
                                         int(T.shape[2]))
        fn = np.asarray(level.farneighbors)    # (nbox, nfar)
        self.nbox = int(fn.shape[0])
        pos = np.asarray(level.idx, dtype=np.int64)
        c = np.stack([np.asarray(below.xidx), np.asarray(below.yidx),
                      np.asarray(below.zidx)], axis=1).astype(np.int64)
        if c.shape[0] != self.nbox:
            raise RuntimeError("mid level rows (%d) and box coordinates "
                               "(%d) disagree" % (self.nbox, c.shape[0]))
        offs, grid = self._lookup(c, pos, fn)
        self.G = grid.shape
        self.dtype = np.dtype(dtype if dtype is not None
                              else ocl_core.operator_dtype())
        self.u = self.v = None
        if self.dtype == np.complex64:
            # the normalised table (levels.MidLevel.midm2linit's scheme);
            # r0 = the geometric mean neighbour distance, from the table
            # itself: |T[.., 0, 0]| ~ |c00|/r
            deg = np.floor(np.sqrt(np.arange(self.nn))).astype(int)
            t00 = np.abs(T[:, :, 0, 0])
            r = 1.0/t00[t00 > 0]
            r0 = float(np.exp(np.mean(np.log(r))))
            self.u = r0**(deg + 1.0)
            self.v = r0**deg.astype(float)
            # out[a] = sum_b T[a, b] d[b] with |T[a, b]| ~
            # r**-(deg a + deg b + 1): the output axis (2) takes
            # r0**deg, the input axis (3) r0**(deg+1); the input is
            # divided by u and the output by v
            Ts = T*self.v[None, None, :, None]*self.u[None, None, None, :]
            if not ocl_core.fits_float32(Ts):
                self.dtype = np.dtype(np.complex128)
                self.u = self.v = None
                Ts = T
        else:
            Ts = T
        self._T = ocl_core.to_device(Ts, self.dtype)
        self._boxc = ocl_core.to_device(c.astype(np.int32))
        self._pos = ocl_core.to_device(pos.astype(np.int32))
        self._offs = ocl_core.to_device(offs.astype(np.int32))
        self._grid = ocl_core.to_device(grid.astype(np.int32))
        self._in = ocl_core.empty((self.nbox, self.nn), self.dtype)
        self._out = ocl_core.empty((self.nbox, self.nn), self.dtype)
        self._host = np.empty((self.nbox, self.nn), self.dtype)
        self.prg = ocl_core.program(
            SOURCE, self.dtype, defines=dict(NN=self.nn, NFAR=self.nfar,
                                             NPAR=self.npar), key='ocl_mid')
        self.k = ocl_core.kernel(self.prg, 'mid_m2l')

    @staticmethod
    def _lookup(c, pos, fn):
        """Offset table per (position, neighbour slot) and the box-id
        grid, checked to reproduce ``farneighbors`` exactly."""
        npar = int(pos.max()) + 1
        nfar = fn.shape[1]
        offs = np.zeros((npar, nfar, 3), np.int64)
        seen = np.zeros((npar, nfar), bool)
        for p in range(npar):
            rows = np.flatnonzero(pos == p)
            for k in range(nfar):
                r = rows[fn[rows, k] >= 0]
                if r.size == 0:
                    continue
                d = c[fn[r, k]] - c[r]
                if not (d == d[0]).all():
                    raise RuntimeError("interaction list is not a fixed "
                                       "offset pattern (pos %d slot %d)"
                                       % (p, k))
                offs[p, k] = d[0]
                seen[p, k] = True
        lo = c.min(axis=0)
        if (lo < 0).any():
            raise RuntimeError("negative box coordinates")
        G = c.max(axis=0) + 1
        grid = np.full(tuple(int(v) for v in G), -1, np.int64)
        grid[c[:, 0], c[:, 1], c[:, 2]] = np.arange(c.shape[0])
        # reproduce farneighbors from (grid, offsets)
        for p in range(npar):
            rows = np.flatnonzero(pos == p)
            for k in range(nfar):
                q = c[rows] + offs[p, k]
                ok = ((q >= 0) & (q < G)).all(axis=1) & seen[p, k]
                nb = np.full(rows.size, -1, np.int64)
                nb[ok] = grid[q[ok, 0], q[ok, 1], q[ok, 2]]
                if not np.array_equal(nb, fn[rows, k].astype(np.int64)):
                    raise RuntimeError("box grid does not reproduce the "
                                       "interaction list (pos %d slot %d)"
                                       % (p, k))
        # slots never seen are empty for every box; point them outside
        offs[~seen] = -(G.max() + 1)
        return offs, grid

    def device_bytes(self):
        return int(sum(a.nbytes for a in (self._T, self._boxc, self._pos,
                                           self._offs, self._grid,
                                           self._in, self._out)))

    def apply(self, data):
        """M2L of host level data ``(nbox, nn)`` complex; returns a new
        complex128 array (the host kernel's output convention)."""
        q = ocl_core.queue()
        src = np.asarray(data)
        if self.u is not None:
            self._host[...] = src/self.u[None, :]
        else:
            self._host[...] = src
        self._in.set(self._host, queue=q)
        GX, GY, GZ = self.G
        n = self.nbox*self.nn
        self.k(q, (int(-(-n//64)*64),), (64,), self._T.data,
               self._in.data, self._boxc.data, self._pos.data,
               self._offs.data, self._grid.data, self._out.data,
               np.int32(self.nbox), np.int32(GX), np.int32(GY),
               np.int32(GZ))
        self._out.get(queue=q, ary=self._host)
        out = self._host.astype(np.complex128)
        if self.v is not None:
            out /= self.v[None, :]
        return out


class MidTranslate(object):
    """M2M (children -> parent) and L2L (parent -> children) of one mid
    level on the card. Tables are (NN, NPAR*NN) and (NPAR*NN, NN):
    5 kB each at nmax 4. Per child: its position in the parent and its
    parent's index (int32). Arithmetic in complex128 regardless of the
    operator precision: the tables are tiny and the work is a few
    hundred kFLOP, so narrowing buys nothing."""

    def __init__(self, level):
        self.dtype = np.dtype(np.complex128)
        Mt = np.asarray(level.m2mtrans)            # (nn, npar*nn)
        Lt = np.asarray(level.l2ltrans)            # (npar*nn, nn)
        self.nn = int(Mt.shape[0])
        self.npar = int(Mt.shape[1])//self.nn
        i0 = np.asarray(level.idx0, np.int64)
        self.ngroup = int(i0.size - 1)
        self.nchild = int(i0[-1])
        pos = np.asarray(level.idx, np.int64)
        grp = np.repeat(np.arange(self.ngroup), np.diff(i0))
        self._M = ocl_core.to_device(Mt, self.dtype)
        self._L = ocl_core.to_device(Lt, self.dtype)
        self._pos = ocl_core.to_device(pos.astype(np.int32))
        self._i0 = ocl_core.to_device(i0.astype(np.int32))
        self._grp = ocl_core.to_device(grp.astype(np.int32))
        self._d = ocl_core.empty((self.nchild, self.nn), self.dtype)
        self._p = ocl_core.empty((self.ngroup, self.nn), self.dtype)
        self.prg = ocl_core.program(
            SOURCE, self.dtype, defines=dict(NN=self.nn, NFAR=1,
                                             NPAR=self.npar),
            key='ocl_mid_tr')
        self.k_m2m = ocl_core.kernel(self.prg, 'mid_m2m')
        self.k_l2l = ocl_core.kernel(self.prg, 'mid_l2l')

    def m2m(self, data):
        q = ocl_core.queue()
        self._d.set(np.ascontiguousarray(data, self.dtype), queue=q)
        n = self.ngroup*self.nn
        self.k_m2m(q, (int(-(-n//64)*64),), (64,), self._M.data,
                   self._d.data, self._pos.data, self._i0.data,
                   self._p.data, np.int32(self.ngroup))
        return self._p.get(queue=q)

    def l2l(self, data, above):
        """``data += L2L(above)`` in place (host arrays)."""
        q = ocl_core.queue()
        self._d.set(np.ascontiguousarray(data, self.dtype), queue=q)
        self._p.set(np.ascontiguousarray(above, self.dtype), queue=q)
        n = self.nchild*self.nn
        self.k_l2l(q, (int(-(-n//64)*64),), (64,), self._L.data,
                   self._p.data, self._pos.data, self._grp.data,
                   self._d.data, np.int32(self.nchild))
        self._d.get(queue=q, ary=data)
