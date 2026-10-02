# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Leaf P2M and L2P on OpenCL.

The host forms each leaf box's multipole moments as a dense box image
times a table (``levels.LeafLevel._p2m_gemm``):

    above[g, nm] += sum_{i in box g} data[i] * conj(ynmr[pos_i, nm]) / m0

and evaluates the local expansion back at the filaments (``_l2p_gemm``):

    data[i] += sum_nm above[g(i), nm] * ynmr[pos_i, nm]

~110 ms each per orientation at R4, single-threaded BLAS.

CARD MEMORY. Nothing new per filament. Each filament's in-box position
and its box's run start are exactly what the near field already keeps
on the card per slab (:class:`ocl_p2p.NearField` packs: ``flatpos``,
``rbase``), so the kernels walk those packs slab by slab. Added: the
``ynmr`` table (positions x harmonics), per-box start offsets and box
ids per slab, and one (nbox, nn) moments buffer.

Order of summation differs from the host GEMM, so results agree to
rounding, not bit for bit; they are reproducible call to call (each
work item owns one output and walks a fixed order).
"""
import numpy as np

import ocl_core

SOURCE = """
/* moments: one work item per (box in slab, harmonic) */
__kernel void leaf_p2m(__global const cplx_t *data,    /* filaments     */
                       __global const int *flatpos,    /* slab entries  */
                       __global const int *estart,     /* (nb+1) entry  */
                       __global const int *rbase,      /* (nb)          */
                       __global const int *gid,        /* (nb) box id   */
                       __global const cplx_t *T,       /* (NPOS, NN)    */
                       __global cplx_t *mom,           /* (nbox, NN)    */
                       const int nb)
{
    const int w = get_global_id(0);
    if (w >= nb*NN)
        return;
    const int cg = w / NN;
    const int nm = w - cg*NN;
    cplx_t acc = (cplx_t)(0, 0);
    for (int q = estart[cg]; q < estart[cg + 1]; q++) {
        const int p = flatpos[q] - cg*NFLAT;
        acc += cmul(data[rbase[cg] + q], T[p*NN + nm]);
    }
    mom[(size_t)gid[cg]*NN + nm] += acc;
}

/* evaluation: one work item per slab entry */
__kernel void leaf_l2p(__global cplx_t *data,
                       __global const int *flatpos,
                       __global const int *rbase,
                       __global const int *gid,
                       __global const cplx_t *Y,       /* (NPOS, NN)    */
                       __global const cplx_t *loc,     /* (nbox, NN)    */
                       const int nent)
{
    const int q = get_global_id(0);
    if (q >= nent)
        return;
    const int f = flatpos[q];
    const int cg = f / NFLAT;
    const int p = f - cg*NFLAT;
    __global const cplx_t *L = loc + (size_t)gid[cg]*NN;
    __global const cplx_t *y = Y + (size_t)p*NN;
    cplx_t acc = (cplx_t)(0, 0);
    for (int nm = 0; nm < NN; nm++)
        acc += cmul(L[nm], y[nm]);
    data[rbase[cg] + q] += acc;
}
"""


class LeafExpansions(object):
    """P2M / L2P for one leaf, on the near field's resident packs."""

    def __init__(self, leaf, nf):
        self.nf = nf                      # ocl_p2p.NearField of this leaf
        self.dtype = nf.dtype
        self.nflat = int(nf.nflat)
        Y = np.asarray(leaf.ynmr)         # (npos, nn)
        self.npos, self.nn = int(Y.shape[0]), int(Y.shape[1])
        if self.npos != self.nflat:
            raise RuntimeError("ynmr rows %d != box positions %d"
                               % (self.npos, self.nflat))
        self.nbox = int(np.size(leaf.idx0) - 1)
        self._T = ocl_core.to_device(np.conj(Y)/leaf._m0, self.dtype)
        self._Y = ocl_core.to_device(Y, self.dtype)
        # per slab: the entry offset of each box's run and its box id
        self.slab = []
        i0 = np.asarray(leaf.idx0, dtype=np.int64)
        for cx, pk in enumerate(nf.packs):
            gs = np.asarray(leaf.slabidx[leaf.slabidx0[cx]:
                                         leaf.slabidx0[cx + 1]], np.int64)
            cnt = i0[gs + 1] - i0[gs]
            est = np.zeros(len(gs) + 1, np.int64)
            est[1:] = np.cumsum(cnt)
            self.slab.append(dict(
                nb=len(gs), nent=int(est[-1]),
                estart=ocl_core.to_device(est.astype(np.int32)),
                gid=ocl_core.to_device(gs.astype(np.int32))))
        self._mom = ocl_core.zeros((self.nbox, self.nn), self.dtype)
        self._host = np.empty((self.nbox, self.nn), self.dtype)
        self.prg = ocl_core.program(
            SOURCE, self.dtype, defines=dict(NN=self.nn, NFLAT=self.nflat),
            key='ocl_leaf')
        self.k_p2m = ocl_core.kernel(self.prg, 'leaf_p2m')
        self.k_l2p = ocl_core.kernel(self.prg, 'leaf_l2p')

    def _upload(self, data):
        """The leaf data into the near field's shared device buffer."""
        import ocl_p2p
        q = ocl_core.queue()
        src = np.asarray(data)
        if src.dtype != self.dtype or not src.flags.c_contiguous:
            stage = ocl_p2p._shared('host', self.dtype, int(src.size))
            stage[...] = src.ravel()
            src = stage
        dev = ocl_p2p._shared('dev', self.dtype, int(src.size))
        dev.set(src.reshape(-1), queue=q)
        return dev

    def p2m(self, data, above):
        """``above += moments of data`` (host arrays)."""
        q = ocl_core.queue()
        dev = self._upload(data)
        self._mom.fill(self.dtype.type(0), queue=q)
        for pk, sl in zip(self.nf.packs, self.slab):
            if sl['nb'] == 0 or sl['nent'] == 0:
                continue
            n = sl['nb']*self.nn
            self.k_p2m(q, (int(-(-n//64)*64),), (64,), dev.data,
                       pk['flatpos'].data, sl['estart'].data,
                       pk['rbase'].data, sl['gid'].data, self._T.data,
                       self._mom.data, np.int32(sl['nb']))
        self._mom.get(queue=q, ary=self._host)
        above += self._host

    def l2p(self, data, loc):
        """``data += local expansions of loc evaluated at the filaments``
        (host arrays; ``data`` updated in place)."""
        import ocl_p2p
        q = ocl_core.queue()
        self._host[...] = loc
        self._mom.set(self._host, queue=q)
        src = np.asarray(data)
        n = int(src.size)
        dev = ocl_p2p._shared('dev', self.dtype, n)
        dev.fill(self.dtype.type(0), queue=q)
        for pk, sl in zip(self.nf.packs, self.slab):
            if sl['nent'] == 0:
                continue
            self.k_l2p(q, (int(-(-sl['nent']//64)*64),), (64,), dev.data,
                       pk['flatpos'].data, pk['rbase'].data,
                       sl['gid'].data, self._Y.data, self._mom.data,
                       np.int32(sl['nent']))
        stage = ocl_p2p._shared('host', self.dtype, n)
        dev.get(queue=q, ary=stage)
        data += stage.reshape(src.shape)
