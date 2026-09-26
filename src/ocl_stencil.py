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

/* ---- level-0 aggregation from tile geometry (2026-09-25). An
   aggregate is the set of same-normal plaquettes sharing base>>1 per
   coarsened axis, and the tiles are base>>2, so every aggregate lies
   inside one tile and one normal: a (TL>>shz, TL>>shy, TL>>shx) block
   of slots. The prolongation reads the coarse index from a per-(tile,
   normal, block) table instead of a per-plaquette column array; the
   restriction sums the block's slots in slot order, where the host
   has verified that this is the CSR's own order (ascending fine
   index), so the bits are the same -- an empty slot adds an exact
   zero. */
#define NBLK(shz, shy, shx) ((TL >> (shz))*(TL >> (shy))*(TL >> (shx)))

__kernel void sten_prolong_impl(__global const int *tab,
                                __global const real_t *x1,
                                __global const int *flat,
                                __global real_t *xt,
                                const unsigned int n,
                                const int shz, const int shy,
                                const int shx)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSE(gid, t, on, a1, a2, a3);
    const int ny = TL >> shy, nx = TL >> shx;
    const int blk = (((int)a1 >> shz)*ny + ((int)a2 >> shy))*nx
                    + ((int)a3 >> shx);
    const int nb = NBLK(shz, shy, shx);
    xt[gid] += x1[tab[((size_t)t*3 + on)*nb + blk]];
}

/* The members of an aggregate are summed in the CSR's own order --
   ascending fine index -- which is NOT a fixed loop nesting (the
   plaquette numbering's axis priority varies by region), so each
   coarse row carries its members' offsets inside the block, 3 bits
   each in that order, and the count, in one uint. */
__kernel void sten_restrict_impl(__global const int *inv_t,
                                 __global const int *inv_b,
                                 __global const uint *order,
                                 __global const real_t *rt,
                                 __global real_t *y,
                                 const unsigned int nrow,
                                 const int shz, const int shy,
                                 const int shx)
{
    const unsigned int c = get_global_id(0);
    if (c >= nrow) return;
    const int ny = TL >> shy, nx = TL >> shx;
    const int nb = NBLK(shz, shy, shx);
    const int t = inv_t[c];
    const int ob = inv_b[c];
    const int on = ob / nb;
    int blk = ob - on*nb;
    const int bx = blk % nx; blk /= nx;
    const int by = blk % ny; blk /= ny;
    const int bz = blk;
    const size_t base = ((size_t)t*3 + on)*TL*TL*TL
                        + (((size_t)(bz << shz))*TL + (by << shy))*TL
                        + (bx << shx);
    const uint code = order[c];
    const int cnt = (int)(code >> 24);
    real_t acc = (real_t)0;
    for (int k = 0; k < cnt; ++k) {
        const uint off = (code >> (3*k)) & 7u;
        acc += rt[base + ((size_t)(off >> 2)*TL + ((off >> 1) & 1u))*TL
                  + (off & 1u)];
    }
    y[c] = acc;
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

#ifdef TLC
/* ---- level 1 as a stencil on the coarse lattice (2026-09-25). The
   Galerkin coarse operator of the plaquette stencil under 2x2x2
   aggregation is fixed by the two aggregates' occupancy patterns (a
   bit per block position), their normals and the coarse offset:
   verified on the host at construction, entry by entry. So a coarse
   cell carries a pattern byte, and a coefficient comes from a
   (slot, pattern_c, pattern_d) table. The tiles are the fine tiles at
   half the edge (TLC = TL/2), with the same neighbour table. The sum
   runs over the slots in order -- NOT the CSR kernel's lane tree, so
   the bits differ at the ulp; the operator is the same, certified
   to tolerance at construction like level 0's own stencil. */
#define CTOT(nt) ((size_t)(nt)*3*TLC*TLC*TLC)
inline int wrapc(int u, int *h)
{
    if (u < 0)    { *h = -1; return u + TLC; }
    if (u >= TLC) { *h =  1; return u - TLC; }
    *h = 0; return u;
}

inline real_t sten1_acc(__global const real_t *xt,
                        __global const uchar *patg,
                        __global const int *nbt,
                        __global const int *nsrc,
                        __global const int *of,
                        __global const char *ctab,
                        __global const int *sptr,
                        unsigned int t, unsigned int on,
                        int cz, int cy, int cx, uint pc)
{
    real_t acc = (real_t)0;
    const int s0 = sptr[on], s1 = sptr[on + 1];
    for (int s = s0; s < s1; ++s) {
        int hx, hy, hz;
        const int sx = wrapc(cx + of[3*s + 0], &hx);
        const int sy = wrapc(cy + of[3*s + 1], &hy);
        const int sz = wrapc(cz + of[3*s + 2], &hz);
        const int nb = nbt[(size_t)t*27 + 9*(hx + 1) + 3*(hy + 1)
                           + (hz + 1)];
        if (nb >= 1) {
            const int ns = nsrc[s] - 1;
            const size_t i = ((((size_t)(nb - 1)*3 + ns)*TLC + sz)*TLC
                              + sy)*TLC + sx;
            const uint pd = patg[i];
            acc += (real_t)ctab[((size_t)s*256 + pc)*256 + pd]*xt[i];
        }
    }
    return acc;
}

#define DECOMPOSEC(gid, t, on, a1, a2, a3)                  \
    size_t rem_ = (gid);                                    \
    const unsigned int a3 = rem_ % TLC; rem_ /= TLC;        \
    const unsigned int a2 = rem_ % TLC; rem_ /= TLC;        \
    const unsigned int a1 = rem_ % TLC; rem_ /= TLC;        \
    const unsigned int on = rem_ % 3;   rem_ /= 3;          \
    const unsigned int t  = (unsigned int)rem_

__kernel void sten1_mv_p(__global const real_t *xt,
                         __global const int *flat,
                         __global const uchar *patg,
                         __global const int *nbt,
                         __global const int *nsrc,
                         __global const int *of,
                         __global const char *ctab,
                         __global const int *sptr,
                         __global real_t *y,
                         const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSEC(gid, t, on, a1, a2, a3);
    y[i] = sten1_acc(xt, patg, nbt, nsrc, of, ctab, sptr, t, on,
                     (int)a1, (int)a2, (int)a3, patg[gid]);
}

__kernel void sten1_res_t(__global const real_t *xt,
                          __global const real_t *b,
                          __global const int *flat,
                          __global const uchar *patg,
                          __global const int *nbt,
                          __global const int *nsrc,
                          __global const int *of,
                          __global const char *ctab,
                          __global const int *sptr,
                          __global real_t *rt,
                          const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSEC(gid, t, on, a1, a2, a3);
    rt[gid] = b[i] - sten1_acc(xt, patg, nbt, nsrc, of, ctab, sptr, t, on,
                               (int)a1, (int)a2, (int)a3, patg[gid]);
}

__kernel void sten1_jac_p(__global const real_t *xt,
                          __global const real_t *b,
                          __global const int *flat,
                          __global const uchar *patg,
                          __global const real_t *wtab,
                          __global const int *nbt,
                          __global const int *nsrc,
                          __global const int *of,
                          __global const char *ctab,
                          __global const int *sptr,
                          __global real_t *yt,
                          const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const size_t gid = (size_t)flat[i];
    DECOMPOSEC(gid, t, on, a1, a2, a3);
    const uint pc = patg[gid];
    const real_t ax = sten1_acc(xt, patg, nbt, nsrc, of, ctab, sptr, t, on,
                                (int)a1, (int)a2, (int)a3, pc);
    yt[gid] = xt[gid] + wtab[on*256 + pc]*(b[i] - ax);
}
#endif

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
                             'sten_prolong_add', 'sten_prolong_impl',
                             'sten_restrict_impl', 'sten_unpack_sub',
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

    def grid(self):
        return self._cur

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

    def t_prolong_add(self, P, x1):
        """x += P x1 on the tiles; ``P`` is a one-per-row prolongator
        (``.col``) or the implicit one from tile geometry (``.tab``)."""
        q = ocl_core.queue()
        tab = getattr(P, 'tab', None)
        if tab is not None:
            shz, shy, shx = P.shifts
            self._k['sten_prolong_impl'](q, (self.n,), None, tab.data,
                                         x1.data, self._flat.data,
                                         self._cur.data, np.uint32(self.n),
                                         np.int32(shz), np.int32(shy),
                                         np.int32(shx))
            return
        self._k['sten_prolong_add'](q, (self.n,), None, P.col.data,
                                    x1.data, self._flat.data,
                                    self._cur.data, np.uint32(self.n))

    def t_restrict(self, R, rt, y):
        """y = R r from the tile grid ``rt``; ``R`` is a remapped
        OnesRestrict (``.spmv``) or the implicit one (``.inv_t``)."""
        inv_t = getattr(R, 'inv_t', None)
        if inv_t is None:
            return R.spmv(rt, y)
        q = ocl_core.queue()
        shz, shy, shx = R.shifts
        self._k['sten_restrict_impl'](q, (R.nrow,), None, inv_t.data,
                                      R.inv_b.data, R.order.data, rt.data,
                                      y.data, np.uint32(R.nrow),
                                      np.int32(shz), np.int32(shy),
                                      np.int32(shx))
        return y

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



class ImplicitAggregation(object):
    """Level-0 prolongation and restriction from tile geometry.

    Built from the host's P0, the stencil's flat map and the level's
    per-axis coarsening. Verifies, on the host, that every aggregate
    is one whole (tile, normal, block) and one whole column -- raises
    ValueError otherwise, so the caller keeps the explicit arrays --
    and records, per coarse row, its members' offsets inside the block
    in the CSR's own order (ascending fine index), so the device sums
    exactly what the arrays would, in the same order. Tables: one int
    per (tile, normal, block) for the prolongation; per coarse row a
    tile, a packed (normal, block) and the packed order."""

    def __init__(self, P0, flat, div, TL, nt):
        # Everything here is O(n0) on the host and runs inside the
        # preconditioner build, which sets the build's peak: int32
        # throughout, checks by scatter and round trip rather than
        # np.unique, intermediates released as they go. The first cut
        # (nine int64 arrays and a transposed copy of P0) cost 1.4 GiB
        # of host peak at R5 and put the build back above the solve.
        import scipy.sparse as sp
        M = sp.csr_matrix(P0)
        counts = np.diff(M.indptr)
        if M.nnz and (counts.max() != 1 or counts.min() != 1
                      or not np.all(M.data == 1)):
            raise ValueError("not a one-per-row 0/1 prolongator")
        del counts
        n0, nc = (int(v) for v in M.shape)
        col = np.ascontiguousarray(M.indices, dtype=np.int32)
        sh = [1 if int(d) == 2 else 0 for d in np.asarray(div)]
        shz, shy, shx = sh[2], sh[1], sh[0]   # base axes (x, y, z) -> slot axes (z, y, x)
        nz, ny, nx = TL >> shz, TL >> shy, TL >> shx
        nb = nz*ny*nx
        cell = TL*TL*TL
        f32 = np.asarray(flat)
        if f32.size and int(f32.max()) >= 2**31:
            raise OverflowError("flat map past the 32-bit index")
        f32 = f32.astype(np.int32)
        loc = f32 % cell
        z = (loc // (TL*TL)).astype(np.uint8)
        y = ((loc // TL) % TL).astype(np.uint8)
        x = (loc % TL).astype(np.uint8)
        del loc
        blk = (((z >> shz).astype(np.int32)*ny + (y >> shy))*nx
               + (x >> shx)).astype(np.int32)
        off = (((z & ((1 << shz) - 1)) << 2) | ((y & ((1 << shy) - 1)) << 1)
               | (x & ((1 << shx) - 1))).astype(np.uint8)
        del z, y, x
        key = (f32 // cell)*np.int32(nb) + blk       # (tile*3 + normal)*nb + block
        del f32, blk
        # bijection between blocks and coarse columns, by scatter and
        # round trip: every plaquette of a block names the same column,
        # every column names one block, and every column has one
        tab = np.full(int(nt)*3*nb, -1, np.int32)
        tab[key] = col
        if not np.array_equal(tab[key], col):
            raise ValueError("aggregates are not whole tile blocks")
        inv_key = np.full(nc, -1, np.int32)
        inv_key[col] = key
        if not (np.array_equal(inv_key[col], key)
                and not np.any(inv_key < 0)
                and np.array_equal(tab[inv_key], np.arange(nc, dtype=np.int32))):
            raise ValueError("aggregates and coarse columns are not one to one")
        del key
        # the restriction's order: each column's members in ascending
        # fine index (the CSR's canonical order); their block offsets
        # packed 3 bits each in that order, the count in the top byte
        cnt = np.bincount(col, minlength=nc)
        if cnt.size and int(cnt.max()) > 8:
            raise ValueError("an aggregate has more than 8 members")
        ptr = np.zeros(nc + 1, np.int64)
        np.cumsum(cnt, out=ptr[1:])
        srt = np.argsort(col, kind='stable')          # members grouped by column, ascending index within
        cols = col[srt]
        k = (np.arange(n0, dtype=np.int64) - ptr[cols]).astype(np.uint32)
        del cols
        shifted = off[srt].astype(np.uint32) << (3*k)
        del k
        order = np.bitwise_or.reduceat(shifted, ptr[:-1]).astype(np.uint32)
        del shifted
        order |= (cnt.astype(np.uint32) << 24)
        # the occupancy pattern of each aggregate (a bit per block
        # position): level 1's coefficients are a function of the two
        # patterns, the normals and the coarse offset
        occ = (np.uint8(1) << off[srt]).astype(np.uint8)
        del srt, off
        self.pattern = np.bitwise_or.reduceat(occ, ptr[:-1]).astype(np.uint8)
        del occ
        inv_t = (inv_key // (3*nb)).astype(np.int32)
        inv_b = (inv_key % (3*nb)).astype(np.int32)
        del inv_key
        # per coarse column: its tile, normal and block, kept on the
        # host for a coarse-level stencil to build on
        self.h_tile = inv_t
        self.h_normal = (inv_b // nb).astype(np.int32)
        self.h_block = (inv_b % nb).astype(np.int32)
        self.nb = int(nb)
        self.h_tab = tab
        self.shape = (int(n0), int(nc))
        self.nrow = int(nc)
        self.nnz = int(M.nnz)
        self.shifts = (int(shz), int(shy), int(shx))
        self.tab = ocl_core.to_device(tab)
        self.inv_t = ocl_core.to_device(inv_t)
        self.inv_b = ocl_core.to_device(inv_b)
        self.order = ocl_core.to_device(order)
        self.src_dtype, self.ones_only, self.int8_ok = 'implicit', True, True

    def retarget(self, slot_of_col):
        """Point the prolongation table at coarse tile SLOTS instead of
        coarse indices, for a level 1 that lives in tiles."""
        tab = self.h_tab.copy()
        ok = tab >= 0
        tab[ok] = np.asarray(slot_of_col, np.int32)[tab[ok]]
        self.tab = ocl_core.to_device(tab)

    def device_bytes(self):
        return int(self.tab.nbytes + self.inv_t.nbytes + self.inv_b.nbytes
                   + self.order.nbytes)


class Stencil1(object):
    """Level 1 as a table-driven stencil on the coarse lattice.

    Built from the host's level-1 CSR, the level-0 implicit
    aggregation (which knows each coarse cell's tile, normal, block
    and occupancy pattern) and the host stencil's neighbour table.
    Verifies on the host that every entry's coefficient is a function
    of (normals, coarse offset, the two patterns) and that the damped
    inverse diagonal is a function of the pattern; then certifies the
    device apply against the CSR on a random vector. Any failure
    raises ValueError and the caller keeps the CSR.

    Keeps level 1 in two coarse tile grids exactly as Stencil0 keeps
    level 0: the same t_* interface, the right-hand side flat.
    """

    def __init__(self, A1, agg, sten, wdi1, dtype, csr_dev=None):
        import scipy.sparse as sp
        dt = np.dtype(dtype)
        self.dtype = dt
        if agg.shifts != (1, 1, 1):
            raise ValueError("level-1 stencil needs full 2x2x2 coarsening")
        TL = int(sten.TL)
        if TL % 2:
            raise ValueError("odd tile edge")
        TLC = TL // 2
        self.TL, self.TLC = TL, TLC
        self.nt = int(sten.shape[0])
        nt = self.nt
        ccell = TLC*TLC*TLC
        self.ntot = int(nt*3*ccell)
        A1 = sp.csr_matrix(A1)
        A1.sort_indices()
        nc = int(A1.shape[0])
        self.n = nc
        self.shape = (nc, nc)
        tile, on, blk = agg.h_tile, agg.h_normal, agg.h_block
        pat = agg.pattern
        if pat.size != nc or tile.size != nc:
            raise ValueError("aggregation and level 1 disagree in size")
        # coarse local coordinates from the block index (z, y, x)
        cz, cy, cx = blk // (TLC*TLC), (blk // TLC) % TLC, blk % TLC
        slot1 = (((tile.astype(np.int64)*3 + on)*TLC + cz)*TLC + cy)*TLC + cx
        if slot1.size and int(slot1.max()) >= 2**31:
            raise OverflowError("coarse tile array past the 32-bit index")
        slot1 = slot1.astype(np.int32)
        # neighbour table on the host: (27, nt) with tile id + 1 or 0
        nbt = np.ascontiguousarray(np.asarray(sten.nbt).T)          # (nt, 27)
        # Per entry (c, d): the coarse offset -- from local coordinates,
        # and across tiles from the neighbour slot that names d's tile
        # -- then the (normals, offset, pattern_c, pattern_d) key and
        # its coefficient. In chunks, int32, with the constancy check
        # kept in three small min/max/seen tables: the entry list is
        # 77 M long at R5 and this runs inside the build's peak.
        NS = 3*3*27                        # (normal_c, normal_d, offset) keys
        NK = NS*256*256
        kmin = np.full(NK, 127, np.int8)
        kmax = np.full(NK, -128, np.int8)
        seen = np.zeros(NK, np.bool_)
        nbt = np.ascontiguousarray(np.asarray(sten.nbt).T)          # (nt, 27), tile id + 1 or 0
        indptr = A1.indptr
        nnz = int(A1.nnz)
        CH = 1 << 20
        for a0 in range(0, nnz, CH):
            a1 = min(nnz, a0 + CH)
            cols = A1.indices[a0:a1].astype(np.int32)
            vals = A1.data[a0:a1].astype(np.int8)
            rows = (np.searchsorted(indptr, np.arange(a0, a1), side='right') - 1).astype(np.int32)
            tc, td = tile[rows], tile[cols]
            dx = (cx[cols] - cx[rows]).astype(np.int8)
            dy = (cy[cols] - cy[rows]).astype(np.int8)
            dz = (cz[cols] - cz[rows]).astype(np.int8)
            diff = np.flatnonzero(tc != td)
            if diff.size:
                hit = nbt[tc[diff]] == (td[diff] + 1)[:, None]      # (m, 27)
                if not np.all(hit.any(axis=1)):
                    raise ValueError("a level-1 entry couples tiles that are not neighbours")
                k = np.argmax(hit, axis=1)
                dx[diff] += ((k // 9 - 1)*TLC).astype(np.int8)
                dy[diff] += (((k // 3) % 3 - 1)*TLC).astype(np.int8)
                dz[diff] += ((k % 3 - 1)*TLC).astype(np.int8)
                del hit, k
            if (int(np.abs(dx).max()) > 1 or int(np.abs(dy).max()) > 1
                    or int(np.abs(dz).max()) > 1):
                raise ValueError("level-1 reach exceeds one coarse cell")
            skey = ((((on[rows].astype(np.int32)*3 + on[cols])*3 + (dz + 1))*3
                     + (dy + 1))*3 + (dx + 1))
            tkey = (skey*256 + pat[rows].astype(np.int32))*256 + pat[cols]
            np.minimum.at(kmin, tkey, vals)
            np.maximum.at(kmax, tkey, vals)
            seen[tkey] = True
            del cols, vals, rows, tc, td, dx, dy, dz, diff, skey, tkey
        if not np.array_equal(kmin[seen], kmax[seen]):
            raise ValueError("a level-1 coefficient is not fixed by the patterns")
        # the slots actually present, ordered by (normal_c, normal_d, offset)
        sk_seen = np.flatnonzero(seen.reshape(NS, 256*256).any(axis=1))
        uk = sk_seen.astype(np.int64)
        nslot = int(uk.size)
        s_on = (uk // 81) % 3
        s_ns = (uk // 27) % 3
        s_dz = (uk // 9) % 3 - 1
        s_dy = (uk // 3) % 3 - 1
        s_dx = uk % 3 - 1
        sptr = np.searchsorted(s_on, np.arange(4)).astype(np.int32)
        ctab = np.zeros(nslot*256*256, np.int8)
        for i, sk in enumerate(sk_seen):
            blk_ = slice(int(sk)*65536, int(sk + 1)*65536)
            v = kmin[blk_]
            ctab[i*65536:(i + 1)*65536] = np.where(seen[blk_], v, 0)
        del kmin, kmax, seen
        # damped inverse diagonal: a function of the pattern AND the
        # normal (a 2x2 slab of plaquettes couples to itself differently
        # by orientation: diagonal 8 or 12 for one pattern at R4)
        w = np.asarray(wdi1, dt).ravel()
        if w.size != nc:
            raise ValueError("level-1 weights and size disagree")
        widx = on.astype(np.int64)*256 + pat
        wtab = np.zeros(3*256, dt)
        wtab[widx] = w
        if not np.array_equal(wtab[widx], w):
            raise ValueError("the level-1 weight is not fixed by the "
                             "normal and pattern")
        del widx
        # pattern grid over coarse slots (0 = absent)
        patg = np.zeros(self.ntot, np.uint8)
        patg[slot1] = pat
        # ---- device state
        self._flat = ocl_core.to_device(slot1)
        self._patg = ocl_core.to_device(patg)
        self._nbt = ocl_core.to_device(np.ascontiguousarray(nbt).astype(np.int32))
        self._nsrc = ocl_core.to_device((s_ns + 1).astype(np.int32))
        self._of = ocl_core.to_device(np.ascontiguousarray(
            np.stack([s_dx, s_dy, s_dz], axis=1)).astype(np.int32))
        self._ctab = ocl_core.to_device(ctab)
        self._sptr = ocl_core.to_device(sptr)
        self._wtab = ocl_core.to_device(wtab)
        self.nslot = nslot
        self.prg = ocl_core.program(SOURCE, _DT[dt], {'TL': TL, 'TLC': TLC},
                                    key='ocl_stencil1')
        self._k = {n: ocl_core.kernel(self.prg, n)
                   for n in ('sten1_mv_p', 'sten1_res_t', 'sten1_jac_p',
                             'sten_prolong_add', 'sten_pack', 'sten_unpack')}
        self._xt = ocl_core.zeros((self.ntot,), dt)
        self._yt = ocl_core.zeros((self.ntot,), dt)
        self._cur = self._xt
        self.slot_of_col = slot1        # host: coarse index -> coarse slot
        # ---- certification against the CSR, on the device
        if csr_dev is not None:
            rng = np.random.default_rng(23)
            xh = rng.standard_normal(nc).astype(dt)
            xd = ocl_core.to_device(xh)
            y_ref = ocl_core.zeros((nc,), dt)
            y_got = ocl_core.zeros((nc,), dt)
            csr_dev.spmv(xd, y_ref)
            self.spmv(xd, y_got)
            ref = y_ref.get(); got = y_got.get()
            nr = float(np.linalg.norm(ref))
            err = float(np.linalg.norm(got - ref))/nr if nr else 0.0
            tol = 1e-5 if dt.itemsize == 4 else 1e-11
            self.cert_err = err
            if err > tol:
                raise ValueError("level-1 stencil disagrees with the CSR "
                                 "(rel %.2e)" % err)
            del xd, y_ref, y_got

    def _tiles(self):
        return (self._patg.data, self._nbt.data, self._nsrc.data,
                self._of.data, self._ctab.data, self._sptr.data)

    def device_bytes(self):
        return int(self._flat.nbytes + self._patg.nbytes + self._nbt.nbytes
                   + self._nsrc.nbytes + self._of.nbytes + self._ctab.nbytes
                   + self._sptr.nbytes + self._wtab.nbytes
                   + self._xt.nbytes + self._yt.nbytes)

    def parts(self):
        return dict(flat_index=int(self._flat.nbytes),
                    pattern_grid=int(self._patg.nbytes),
                    tables=int(self._ctab.nbytes + self._nbt.nbytes
                               + self._of.nbytes + self._wtab.nbytes),
                    work_grids=int(self._xt.nbytes + self._yt.nbytes),
                    slots=int(self.ntot), cells=int(self.n))

    # ---- flat API (certification)
    def spmv(self, x, y):
        q = ocl_core.queue()
        self._xt.fill(self.dtype.type(0), queue=q)
        self._k['sten_pack'](q, (self.n,), None, x.data, self._flat.data,
                             self._xt.data, np.uint32(self.n))
        self._k['sten1_mv_p'](q, (self.n,), None, self._xt.data,
                              self._flat.data, *self._tiles(), y.data,
                              np.uint32(self.n))
        return y

    # ---- tiled-native API, as Stencil0
    def t_zero(self):
        q = ocl_core.queue()
        self._xt.fill(self.dtype.type(0), queue=q)
        self._yt.fill(self.dtype.type(0), queue=q)
        self._cur = self._xt

    def t_free(self):
        return self._yt if self._cur is self._xt else self._xt

    def grid(self):
        return self._cur

    def t_sweeps(self, b, nu):
        q = ocl_core.queue()
        cur, alt = self._cur, self.t_free()
        for _ in range(int(nu)):
            self._k['sten1_jac_p'](q, (self.n,), None, cur.data, b.data,
                                   self._flat.data, self._patg.data,
                                   self._wtab.data, self._nbt.data,
                                   self._nsrc.data, self._of.data,
                                   self._ctab.data, self._sptr.data,
                                   alt.data, np.uint32(self.n))
            cur, alt = alt, cur
        self._cur = cur

    def t_residual(self, b):
        q = ocl_core.queue()
        rt = self.t_free()
        self._k['sten1_res_t'](q, (self.n,), None, self._cur.data, b.data,
                               self._flat.data, *self._tiles(), rt.data,
                               np.uint32(self.n))
        return rt

    def t_prolong_add(self, P, x2):
        q = ocl_core.queue()
        self._k['sten_prolong_add'](q, (self.n,), None, P.col.data, x2.data,
                                    self._flat.data, self._cur.data,
                                    np.uint32(self.n))

    def t_restrict(self, R, rt, y):
        return R.spmv(rt, y)
