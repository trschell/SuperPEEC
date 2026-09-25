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
/* The damped inverse diagonal is one number. The plaquette Gram's
   diagonal is 4 everywhere by construction, so the weight is the same
   on every occupied cell and zero elsewhere -- verified at build, with
   the full array kept as a fallback if a geometry ever disagrees. One
   bit per slot replaces four bytes: 212 MB at R5. */
__kernel void sten_jac(__global const real_t *xt,
                       __global const real_t *bt,
                       __global const uint *mask,
                       __global const int *nbt,
                       __global const int *nsrc,
                       __global const int *of,
                       __global const char *cf,
                       __global const int *sptr,
                       __global real_t *yt,
                       const real_t wdi,
                       const unsigned int nt)
{
    const size_t gid = get_global_id(0);
    if (gid >= TOT(nt)) return;
    DECOMPOSE(gid, t, on, a1, a2, a3);
    const real_t ax = sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                               (int)a1, (int)a2, (int)a3);
    const real_t w = (mask[gid >> 5] & (1u << (gid & 31))) ? wdi
                                                          : (real_t)0;
    yt[gid] = xt[gid] + w*(bt[gid] - ax);
}

/* the same sweep where the weight really does vary */
__kernel void sten_jac_w(__global const real_t *xt,
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

/* ---- the same operator with one work item per PLAQUETTE (2026-09-25).
   The output cell is the plaquette's own slot, decoded from the flat
   map; the stencil sum is sten_acc exactly as above, so the bits are
   the same. What changes is who is indexed how: the right-hand side
   is read FLAT, by plaquette, straight from the caller's vector (no
   packed b grid, no occupancy mask -- an empty slot never has a work
   item), and the result goes either to the slot (for the V-cycle,
   which keeps level 0 in tiles) or to a flat vector. */
__kernel void sten_mv_p(__global const real_t *xt,
                        __global const int *flat,
                        __global const int *nbt,
                        __global const int *nsrc,
                        __global const int *of,
                        __global const char *cf,
                        __global const int *sptr,
                        __global real_t *y,
                        const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSE(gid, t, on, a1, a2, a3);
    y[i] = sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                    (int)a1, (int)a2, (int)a3);
}

/* r = b - A x: into a tile grid (rt) or a flat vector (r) */
__kernel void sten_res_t(__global const real_t *xt,
                         __global const real_t *b,
                         __global const int *flat,
                         __global const int *nbt,
                         __global const int *nsrc,
                         __global const int *of,
                         __global const char *cf,
                         __global const int *sptr,
                         __global real_t *rt,
                         const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSE(gid, t, on, a1, a2, a3);
    rt[gid] = b[i] - sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                              (int)a1, (int)a2, (int)a3);
}

__kernel void sten_res_p(__global const real_t *xt,
                         __global const real_t *b,
                         __global const int *flat,
                         __global const int *nbt,
                         __global const int *nsrc,
                         __global const int *of,
                         __global const char *cf,
                         __global const int *sptr,
                         __global real_t *r,
                         const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSE(gid, t, on, a1, a2, a3);
    r[i] = b[i] - sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                           (int)a1, (int)a2, (int)a3);
}

/* one damped-Jacobi sweep, tiles to tiles, b flat; uniform weight */
__kernel void sten_jac_p(__global const real_t *xt,
                         __global const real_t *b,
                         __global const int *flat,
                         __global const int *nbt,
                         __global const int *nsrc,
                         __global const int *of,
                         __global const char *cf,
                         __global const int *sptr,
                         __global real_t *yt,
                         const real_t wdi,
                         const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSE(gid, t, on, a1, a2, a3);
    const real_t ax = sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                               (int)a1, (int)a2, (int)a3);
    yt[gid] = xt[gid] + wdi*(b[i] - ax);
}

/* the same where the weight varies, one per plaquette */
__kernel void sten_jac_pw(__global const real_t *xt,
                          __global const real_t *b,
                          __global const int *flat,
                          __global const real_t *wp,
                          __global const int *nbt,
                          __global const int *nsrc,
                          __global const int *of,
                          __global const char *cf,
                          __global const int *sptr,
                          __global real_t *yt,
                          const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSE(gid, t, on, a1, a2, a3);
    const real_t ax = sten_acc(xt, nbt, nsrc, of, cf, sptr, t, on,
                               (int)a1, (int)a2, (int)a3);
    yt[gid] = xt[gid] + wp[i]*(b[i] - ax);
}

/* prolongation straight into the tiles: xt[flat[i]] += x1[col[i]] */
__kernel void sten_prolong_add(__global const int *col,
                               __global const real_t *x1,
                               __global const int *flat,
                               __global real_t *xt,
                               const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) xt[flat[i]] += x1[col[i]];
}

/* out[i] = yp[i] - t[flat[i]]: the local block's final combination,
   read from the tiled solution without a flat copy of it */
__kernel void sten_unpack_sub(__global const real_t *t,
                              __global const int *flat,
                              __global const real_t *yp,
                              __global real_t *out,
                              const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) out[i] = yp[i] - t[flat[i]];
}

/* flat plaquette vector -> zeroed tiles, and back */
__kernel void sten_pack(__global const real_t *v,
                        __global const int *flat,
                        __global real_t *t,
                        const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) t[flat[i]] = v[i];
}

__kernel void sten_unpack(__global const real_t *t,
                          __global const int *flat,
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
        flat = np.asarray(sten.flat)
        if int(flat.max()) >= 2**31:
            raise OverflowError("stencil tile array has %d slots, past "
                                "the 32-bit index" % int(flat.max()))
        self._flat = ocl_core.to_device(flat.astype(np.int32))
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
        # one weight, or the array if this geometry disagrees
        # The damped inverse diagonal is one number on this geometry
        # (the plaquette Gram's diagonal is 4 by construction), and
        # since every kernel now runs one work item per PLAQUETTE the
        # occupancy mask is not needed either: an empty slot has no
        # work item. Where the weight varies it is kept per plaquette.
        w = np.ascontiguousarray(wdi_t).astype(dt).ravel()
        wp = w[flat]
        nz = np.unique(wp)
        self.uniform_w = bool(nz.size == 1)
        if self.uniform_w:
            self.wdi = dt.type(nz[0])
            self._wp = None
        else:
            self.wdi = dt.type(0)
            self._wp = ocl_core.to_device(wp)
        del w, wp
        self.prg = ocl_core.program(SOURCE, _DT[dt], {'TL': self.TL},
                                    key='ocl_stencil')
        self._k = {n: ocl_core.kernel(self.prg, n)
                   for n in ('sten_mv_p', 'sten_res_t', 'sten_res_p',
                             'sten_jac_p', 'sten_jac_pw',
                             'sten_prolong_add', 'sten_unpack_sub',
                             'sten_pack', 'sten_unpack')}
        # Two tile grids, and only two: the V-cycle keeps level 0 in
        # them (x and its Jacobi partner; the residual takes the free
        # one), the right-hand side stays flat in the caller's vector.
        # Empty slots are zero at creation and no kernel ever writes
        # one, so a neighbour read across an empty cell reads zero.
        self._xt = ocl_core.zeros((self.ntot,), dt)
        self._yt = ocl_core.zeros((self.ntot,), dt)
        self._cur = self._xt     # the grid holding the current x

    def parts(self):
        """Device bytes by part, for sizing arguments."""
        tables = (self._nbt.nbytes + self._nsrc.nbytes + self._of.nbytes
                  + self._cf.nbytes + self._sptr.nbytes)
        return dict(flat_index=int(self._flat.nbytes),
                    weights=int(0 if self._wp is None else self._wp.nbytes),
                    work_grids=int(self._xt.nbytes + self._yt.nbytes),
                    tables=int(tables), slots=int(self.ntot),
                    plaquettes=int(self.n))

    def device_bytes(self):
        """Resident device bytes: the tables, the weights, the tiles."""
        w = 0 if self._wp is None else self._wp.nbytes
        return int(self._flat.nbytes + self._nbt.nbytes
                   + self._nsrc.nbytes + self._of.nbytes + self._cf.nbytes
                   + self._sptr.nbytes + w
                   + self._xt.nbytes + self._yt.nbytes)

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

    def _jac(self, cur, b, alt):
        """One sweep from tiles ``cur`` with flat ``b`` into tiles ``alt``."""
        q = ocl_core.queue()
        if self.uniform_w:
            self._k['sten_jac_p'](q, (self.n,), None, cur.data, b.data,
                                  self._flat.data, *self._tiles(),
                                  alt.data, self.wdi, np.uint32(self.n))
        else:
            self._k['sten_jac_pw'](q, (self.n,), None, cur.data, b.data,
                                   self._flat.data, self._wp.data,
                                   *self._tiles(), alt.data,
                                   np.uint32(self.n))

    # ---- the flat API (validators, the generic smoother loop)
    def spmv(self, x, y):
        """``y = A x`` on flat vectors."""
        q = ocl_core.queue()
        self._pack(x, self._xt)
        self._k['sten_mv_p'](q, (self.n,), None, self._xt.data,
                             self._flat.data, *self._tiles(), y.data,
                             np.uint32(self.n))
        return y

    def residual(self, x, b, r):
        """``r = b - A x`` on flat vectors."""
        q = ocl_core.queue()
        self._pack(x, self._xt)
        self._k['sten_res_p'](q, (self.n,), None, self._xt.data, b.data,
                              self._flat.data, *self._tiles(), r.data,
                              np.uint32(self.n))
        return r

    def sweeps(self, x, b, nu):
        """``nu`` damped-Jacobi sweeps, in place on the flat ``x``."""
        self._pack(x, self._xt)
        cur, alt = self._xt, self._yt
        for _ in range(int(nu)):
            self._jac(cur, b, alt)
            cur, alt = alt, cur
        return self._unpack(cur, x)

    def jacobi(self, x, b, dinv, xout, omega):
        """One sweep, for the generic smoother loop.

        ``dinv`` and ``omega`` are ignored: the damping factor and the
        inverse diagonal are already folded into the tiled weights the
        host certified.
        """
        self._pack(x, self._xt)
        self._jac(self._xt, b, self._yt)
        return self._unpack(self._yt, xout)

    # ---- the tiled-native API: level 0 stays in the tiles across a
    # whole V-cycle; only the right-hand side and the final answer
    # are flat vectors, and those are the caller's
    def t_zero(self):
        """x = 0 in the tiles (both grids, so every empty slot is 0)."""
        q = ocl_core.queue()
        self._xt.fill(self.dtype.type(0), queue=q)
        self._yt.fill(self.dtype.type(0), queue=q)
        self._cur = self._xt

    def t_free(self):
        return self._yt if self._cur is self._xt else self._xt

    def t_sweeps(self, b, nu):
        """``nu`` sweeps on the tiled x, flat ``b``."""
        cur, alt = self._cur, self.t_free()
        for _ in range(int(nu)):
            self._jac(cur, b, alt)
            cur, alt = alt, cur
        self._cur = cur

    def t_residual(self, b):
        """``b - A x`` into the free grid; returns that grid."""
        q = ocl_core.queue()
        rt = self.t_free()
        self._k['sten_res_t'](q, (self.n,), None, self._cur.data, b.data,
                              self._flat.data, *self._tiles(), rt.data,
                              np.uint32(self.n))
        return rt

    def t_prolong_add(self, col, x1):
        """x += P x1 on the tiles, P one entry per row (``col``)."""
        q = ocl_core.queue()
        self._k['sten_prolong_add'](q, (self.n,), None, col.data, x1.data,
                                    self._flat.data, self._cur.data,
                                    np.uint32(self.n))

    def t_unpack(self, out):
        """The tiled x as a flat vector."""
        return self._unpack(self._cur, out)

    def t_unpack_sub(self, yp, out):
        """``out = yp - x`` straight from the tiles."""
        q = ocl_core.queue()
        self._k['sten_unpack_sub'](q, (self.n,), None, self._cur.data,
                                   self._flat.data, yp.data, out.data,
                                   np.uint32(self.n))
        return out
