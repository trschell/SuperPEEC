# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The level-0 Gram as a tiled stencil, on OpenCL.

The plaquette Gram is translation invariant: a constant coefficient per
(source normal, offset), with 36 slots, coefficients of plus or minus
one, and a diagonal of four. The host exploits that by never forming
the matrix at all. It packs the plaquette vector into per-normal dense
16-cubed tiles over the occupied lattice and applies the stencil with a
one-cell halo gathered from the 27 neighbouring tiles, streaming only
vectors. That is the memory win the compression campaign bought, and it
is why ``mg.levels[0]`` is ``None`` on this path.

The CUDA backend sidesteps the question by forming the Gram on the card
through a sparse-sparse product. This backend has no such product, so
before this module it formed the Gram on the host and uploaded it: a
build transient worth about 0.7 GB on R3 and the reason the OpenCL
peak sat above the CUDA one. Applying the stencil directly removes it.

The halo is read through the neighbour table rather than staged in
local memory. A padded tile carries (TL+2)^3 cells for each of three
normals, which is 70 kB at TL 16 and does not fit in the 48 kB a work
group gets. Instead each work item owns one output cell and resolves
each slot's offset itself: an out-of-range coordinate names one of the
27 neighbours and wraps into it. Neighbouring work items read
overlapping cells, so the traffic is absorbed by cache.

An absent neighbour tile is zero, and absent plaquettes hold zero
inside their tile, so nothing needs masking.

Summation runs over the slots in the build-time verified order, which
is the order the Fortran kernel uses, so results track the host path.
"""
import numpy as np

import ocl_core

SOURCE = """
/* Tiles are (NT, 3, TL, TL, TL) with the last axis contiguous. In the
   Fortran kernel's names that last axis is X, the middle Y and the
   first Z, and a slot's offsets are given in that order. */
#define TOT(nt) ((size_t)(nt)*3*TL*TL*TL)

inline int wrap(int u, int *h)
{
    if (u < 0)   { *h = -1; return u + TL; }
    if (u >= TL) { *h =  1; return u - TL; }
    *h = 0; return u;
}

/* the stencil sum at one output cell */
inline real_t sten_acc(__global const real_t *xt,
                       __global const int *nbt,
                       __global const int *nsrc,
                       __global const int *of,
                       __global const char *cf,
                       __global const int *sptr,
                       unsigned int t, unsigned int on,
                       int cz, int cy, int cx)
{
    real_t acc = (real_t)0;
    const int s0 = sptr[on], s1 = sptr[on + 1];
    for (int s = s0; s < s1; ++s) {
        int hx, hy, hz;
        const int sx = wrap(cx + of[3*s + 0], &hx);
        const int sy = wrap(cy + of[3*s + 1], &hy);
        const int sz = wrap(cz + of[3*s + 2], &hz);
        const int nb = nbt[(size_t)t*27 + 9*(hx + 1) + 3*(hy + 1)
                           + (hz + 1)];
        if (nb >= 1) {
            const int ns = nsrc[s] - 1;
            const size_t i = ((((size_t)(nb - 1)*3 + ns)*TL + sz)*TL
                              + sy)*TL + sx;
            acc += (real_t)cf[s]*xt[i];
        }
    }
    return acc;
}

#define DECOMPOSE(gid, t, on, a1, a2, a3)                   \\
    size_t rem_ = (gid);                                    \\
    const unsigned int a3 = rem_ % TL; rem_ /= TL;          \\
    const unsigned int a2 = rem_ % TL; rem_ /= TL;          \\
    const unsigned int a1 = rem_ % TL; rem_ /= TL;          \\
    const unsigned int on = rem_ % 3;  rem_ /= 3;           \\
    const unsigned int t  = (unsigned int)rem_

__kernel void sten_mv(__global const real_t *xt,
                      __global const int *nbt,
                      __global const int *nsrc,
                      __global const int *of,
                      __global const char *cf,
                      __global const int *sptr,
                      __global real_t *yt,
                      const unsigned int nt)
{
    const size_t gid = get_global_id(0);
    if (gid >= TOT(nt)) return;
    DECOMPOSE(gid, t, on, a1, a2, a3);
    yt[gid] = sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                       (int)a1, (int)a2, (int)a3);
}

/* r = b - A x, on tiles */
__kernel void sten_res(__global const real_t *xt,
                       __global const real_t *bt,
                       __global const int *nbt,
                       __global const int *nsrc,
                       __global const int *of,
                       __global const char *cf,
                       __global const int *sptr,
                       __global real_t *rt,
                       const unsigned int nt)
{
    const size_t gid = get_global_id(0);
    if (gid >= TOT(nt)) return;
    DECOMPOSE(gid, t, on, a1, a2, a3);
    rt[gid] = bt[gid] - sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                                 (int)a1, (int)a2, (int)a3);
}

/* one fused damped-Jacobi sweep, y = x + wdi*(b - A x); wdi already
   carries the damping factor and the inverse diagonal */
__kernel void sten_jac(__global const real_t *xt,
                       __global const real_t *bt,
                       __global const real_t *wt,
                       __global const int *nbt,
                       __global const int *nsrc,
                       __global const int *of,
                       __global const char *cf,
                       __global const int *sptr,
                       __global real_t *yt,
                       const unsigned int nt)
{
    const size_t gid = get_global_id(0);
    if (gid >= TOT(nt)) return;
    DECOMPOSE(gid, t, on, a1, a2, a3);
    const real_t ax = sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                               (int)a1, (int)a2, (int)a3);
    yt[gid] = xt[gid] + wt[gid]*(bt[gid] - ax);
}

/* flat plaquette vector -> zeroed tiles, and back */
__kernel void sten_pack(__global const real_t *v,
                        __global const long *flat,
                        __global real_t *t,
                        const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) t[flat[i]] = v[i];
}

__kernel void sten_unpack(__global const real_t *t,
                          __global const long *flat,
                          __global real_t *v,
                          const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) v[i] = t[flat[i]];
}
"""

_DT = {np.dtype(np.float32): np.complex64,
       np.dtype(np.float64): np.complex128}


class Stencil0(object):
    """Level 0 applied from tiles, with the interface GeoCore expects.

    Exposes ``jacobi``, ``sweeps`` and ``residual`` on flat plaquette
    vectors, so the V-cycle does not care whether level 0 is a matrix
    or a stencil.
    """

    def __init__(self, sten, wdi_t, dtype=None):
        self.dtype = np.dtype(dtype if dtype is not None else sten.dtype)
        self.TL = int(sten.TL)
        self.shape_t = tuple(int(v) for v in sten.shape)
        self.nt = int(self.shape_t[0])
        self.n = int(sten.n)
        self.shape = (self.n, self.n)
        self.ntot = int(np.prod(self.shape_t))
        dt = self.dtype
        self._flat = ocl_core.to_device(np.asarray(sten.flat, np.int64))
        # the host holds these Fortran-shaped -- nbt is (27, tiles)
        # and of is (3, slots) -- so they are transposed on the way in
        # and the kernel indexes tile-major and slot-major
        self._nbt = ocl_core.to_device(
            np.ascontiguousarray(np.asarray(sten.nbt).T).astype(np.int32))
        self._nsrc = ocl_core.to_device(
            np.ascontiguousarray(sten.nsrc).astype(np.int32))
        self._of = ocl_core.to_device(
            np.ascontiguousarray(np.asarray(sten.of).T).astype(np.int32))
        self._cf = ocl_core.to_device(
            np.ascontiguousarray(sten.cf).astype(np.int8))
        self._sptr = ocl_core.to_device(
            np.ascontiguousarray(sten.sptr).astype(np.int32))
        self._wt = ocl_core.to_device(
            np.ascontiguousarray(wdi_t).astype(dt))
        self.prg = ocl_core.program(SOURCE, _DT[dt], {'TL': self.TL},
                                    key='ocl_stencil')
        self._k = {n: ocl_core.kernel(self.prg, n)
                   for n in ('sten_mv', 'sten_res', 'sten_jac',
                             'sten_pack', 'sten_unpack')}
        self._xt = ocl_core.zeros((self.ntot,), dt)
        self._bt = ocl_core.zeros((self.ntot,), dt)
        self._yt = ocl_core.zeros((self.ntot,), dt)

    def device_bytes(self):
        """Resident device bytes: the tables, the weights, the tiles."""
        return int(self._flat.nbytes + self._nbt.nbytes
                   + self._nsrc.nbytes + self._of.nbytes + self._cf.nbytes
                   + self._sptr.nbytes + self._wt.nbytes
                   + self._xt.nbytes + self._bt.nbytes + self._yt.nbytes)

    # ------------------------------------------------------- packing

    def _pack(self, v, t):
        q = ocl_core.queue()
        t.fill(self.dtype.type(0), queue=q)
        self._k['sten_pack'](q, (self.n,), None, v.data, self._flat.data,
                             t.data, np.uint32(self.n))
        return t

    def _unpack(self, t, v):
        q = ocl_core.queue()
        self._k['sten_unpack'](q, (self.n,), None, t.data,
                               self._flat.data, v.data, np.uint32(self.n))
        return v

    # ------------------------------------------- the operator itself

    def _tiles(self):
        return (self._nbt.data, self._nsrc.data, self._of.data,
                self._cf.data, self._sptr.data)

    def spmv(self, x, y):
        """``y = A x`` on flat vectors."""
        q = ocl_core.queue()
        self._pack(x, self._xt)
        self._k['sten_mv'](q, (self.ntot,), None, self._xt.data,
                           *self._tiles(), self._yt.data,
                           np.uint32(self.nt))
        return self._unpack(self._yt, y)

    def residual(self, x, b, r):
        """``r = b - A x`` on flat vectors."""
        q = ocl_core.queue()
        self._pack(x, self._xt)
        self._pack(b, self._bt)
        self._k['sten_res'](q, (self.ntot,), None, self._xt.data,
                            self._bt.data, *self._tiles(), self._yt.data,
                            np.uint32(self.nt))
        return self._unpack(self._yt, r)

    def sweeps(self, x, b, nu):
        """``nu`` damped-Jacobi sweeps, in place on the flat ``x``."""
        q = ocl_core.queue()
        self._pack(x, self._xt)
        self._pack(b, self._bt)
        cur, alt = self._xt, self._yt
        for _ in range(int(nu)):
            self._k['sten_jac'](q, (self.ntot,), None, cur.data,
                                self._bt.data, self._wt.data,
                                *self._tiles(), alt.data,
                                np.uint32(self.nt))
            cur, alt = alt, cur
        return self._unpack(cur, x)

    def jacobi(self, x, b, dinv, xout, omega):
        """One sweep, for the generic smoother loop.

        ``dinv`` and ``omega`` are ignored: the damping factor and the
        inverse diagonal are already folded into the tiled weights the
        host certified.
        """
        q = ocl_core.queue()
        self._pack(x, self._xt)
        self._pack(b, self._bt)
        self._k['sten_jac'](q, (self.ntot,), None, self._xt.data,
                            self._bt.data, self._wt.data, *self._tiles(),
                            self._yt.data, np.uint32(self.nt))
        return self._unpack(self._yt, xout)
