# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Near-field partial inductance on OpenCL: the 27-neighbour convolution.

The stage sweeps the domain one x-slab at a time, keeping a rolling
window of three forward-transformed source slabs, and for every target
box accumulates the contribution of its 27 neighbours through the
transfer table:

    tgt[cg, g] = sum over the box's neighbours  trans[tr, g] * src[p, g]

Two differences from the CuPy path, both structural rather than
cosmetic.

*No materialised pair array.* CuPy forms ``transfer[tr] * slab[p]`` for
every neighbour pair at once, an array of (pairs, grid) complex, and
only then reduces it. On the R3 flagship that temporary is over 100 MB
per slab per offset and it is written and read back for nothing. Here
the accumulation happens in a register: one work item owns one (target
box, grid point) and walks its own neighbour list.

*No atomics.* Because the loop is over output boxes rather than over
neighbour directions, nothing is scattered and ``scatter_add`` is not
needed. That also makes the result reproducible call to call, which the
CuPy form is not: it reduces with CUDA atomics in whatever order the
hardware delivers. The neighbour list is walked in the order the host
built it, which is the order the Fortran kernel uses, so the result
tracks the host path to fp64 rounding.

The neighbour lists are stored once per slab in compressed form: an
offset array indexed by target box, and parallel entry arrays holding
the source position, the transfer channel and which of the three
rolling slabs the source lives in.

The scatter and gather maps are compressed the same way. A target box
owns a *contiguous* range of the filament array, so the source index of
entry ``q`` is the box's first filament plus how far into the box's own
run ``q`` sits. Storing one base per box instead of one index per entry
removes the whole source-index array: at R5 that is 221 MB of card
across the three orientations, for a table of a few kilobytes. The box
itself is already recovered from the position map the kernel reads
anyway, so nothing extra is loaded per work item.
"""
import numpy as np

import ocl_core

SOURCE = """
/* tgt[cg, g] = sum over the box's neighbours trans[tr, g]*src[dx][p, g] */
__kernel void p2p_mac(__global const cplx_t *trans,   /* (27, GS)      */
                      __global const cplx_t *sm,      /* slab at dx=-1 */
                      __global const cplx_t *s0,      /* slab at dx= 0 */
                      __global const cplx_t *sp,      /* slab at dx=+1 */
                      __global const int *off,        /* (nbox+1)      */
                      __global const int *ent_p,
                      __global const int *ent_tr,
                      __global const int *ent_dx,
                      __global cplx_t *tgt,           /* (nbox, GS)    */
                      const unsigned long GS,
                      const unsigned int nbox)
{
    const unsigned long q = get_global_id(0);
    if (q >= (unsigned long)nbox*GS) return;
    const unsigned int cg = (unsigned int)(q/GS);
    const unsigned long g = q - (unsigned long)cg*GS;

    cplx_t acc = (cplx_t)(0, 0);
    const int e0 = off[cg], e1 = off[cg + 1];
    for (int e = e0; e < e1; ++e) {
        const int dx = ent_dx[e];
        __global const cplx_t *src = (dx < 0) ? sm : ((dx == 0) ? s0 : sp);
        acc += cmul(trans[(unsigned long)ent_tr[e]*GS + g],
                    src[(unsigned long)ent_p[e]*GS + g]);
    }
    tgt[q] = acc;
}

/* filament data -> zeroed padded slab grid, one work item per entry */
__kernel void scatter_slab(__global const cplx_t *data,
                           __global const int *rbase,   /* (nbox,) */
                           __global const int *flatpos,
                           __global cplx_t *pad,
                           const unsigned int nent,
                           const unsigned int nflat,
                           const unsigned long GS,
                           const unsigned int n1, const unsigned int n2,
                           const unsigned int S1, const unsigned int S2)
{
    const unsigned int q = get_global_id(0);
    if (q >= nent) return;
    const int fp = flatpos[q];
    const unsigned int box = (unsigned int)(fp/(int)nflat);
    const unsigned int f = (unsigned int)(fp - (int)box*(int)nflat);
    const unsigned int a = f/(n1*n2), b = (f/n2) % n1, d = f % n2;
    pad[(unsigned long)box*GS + ((unsigned long)a*S1 + b)*S2 + d]
        = data[rbase[box] + (int)q];
}

/* padded slab grid -> filament data */
__kernel void gather_slab(__global const cplx_t *pad,
                          __global const int *rbase,   /* (nbox,) */
                          __global const int *flatpos,
                          __global cplx_t *out,
                          const unsigned int nent,
                          const unsigned int nflat,
                          const unsigned long GS,
                          const unsigned int n1, const unsigned int n2,
                          const unsigned int S1, const unsigned int S2)
{
    const unsigned int q = get_global_id(0);
    if (q >= nent) return;
    const int fp = flatpos[q];
    const unsigned int box = (unsigned int)(fp/(int)nflat);
    const unsigned int f = (unsigned int)(fp - (int)box*(int)nflat);
    const unsigned int a = f/(n1*n2), b = (f/n2) % n1, d = f % n2;
    out[rbase[box] + (int)q] = pad[(unsigned long)box*GS
                                   + ((unsigned long)a*S1 + b)*S2 + d];
}
"""


class NearField(object):
    """Device state and apply for one leaf's near field."""

    def __init__(self, leaf, dtype=np.complex128):
        self.dtype = np.dtype(dtype)
        self.n = tuple(int(v) for v in leaf.n)
        self.nflat = int(np.prod(self.n))
        self.S = tuple(int(v) for v in leaf.p2p_transfer.shape[1:])
        if any(self.S[i] < 2*self.n[i] - 1 for i in range(3)):
            raise RuntimeError("p2p transfer grid %s too small for leaf "
                               "lattice %s" % (self.S, self.n))
        self.GS = int(np.prod(self.S))
        self.nslab = int(leaf.ng[0])
        self._build_packs(leaf)
        self._trans = ocl_core.to_device(leaf.p2p_transfer.reshape(27, -1),
                                         self.dtype)
        self.prg = ocl_core.program(SOURCE, self.dtype, key='ocl_p2p')
        # shaped (maxslab,) + S so a leading slice is exactly the
        # 4-D batch a transform plan expects; the kernels address the
        # same memory flat, with GS as the per-box stride
        mx = self.maxsize
        self._slab = [ocl_core.zeros((mx,) + self.S, self.dtype)
                      for _ in range(3)]
        self._tgt = ocl_core.zeros((mx,) + self.S, self.dtype)
        self._data = None        # input and output, in place
        self._fft = {}
        self.k_scatter = ocl_core.kernel(self.prg, 'scatter_slab')
        self.k_mac = ocl_core.kernel(self.prg, 'p2p_mac')
        self.k_gather = ocl_core.kernel(self.prg, 'gather_slab')

    def parts(self):
        """Device bytes by part, for sizing arguments."""
        idx64 = sum(pk[k].nbytes for pk in self.packs
                    for k in ('flatpos', 'rbase') if pk.get(k) is not None)
        other = sum(pk[k].nbytes for pk in self.packs
                    for k in ('off', 'ep', 'etr', 'edx')
                    if pk.get(k) is not None)
        return dict(transfer=int(self._trans.nbytes),
                    slabs=int(self._tgt.nbytes
                              + sum(b.nbytes for b in self._slab)),
                    packs_int64=int(idx64), packs_other=int(other))

    def device_bytes(self):
        """Resident device bytes: the transfer table, the rolling slab
        buffers, and the per-slab index packs."""
        n = self._trans.nbytes + self._tgt.nbytes \
            + sum(b.nbytes for b in self._slab)
        for pk in self.packs:
            for k in ('flatpos', 'rbase', 'off', 'ep', 'etr', 'edx'):
                a = pk.get(k)
                if a is not None:
                    n += a.nbytes
        return int(n)

    def _build_packs(self, leaf):
        """Per-slab scatter maps and compressed neighbour lists."""
        packs = []
        mx = 0
        for cx in range(self.nslab):
            gs = leaf.slabidx[leaf.slabidx0[cx]:leaf.slabidx0[cx+1]]
            rows, cols = [], []
            # a box owns a contiguous run of filaments, so its entries'
            # source indices are the run start offset by how far into
            # the run each entry sits: one base per box replaces the
            # per-entry array entirely
            rbase = np.zeros(len(gs), dtype=np.int64)
            nent = 0
            for cg, group in enumerate(gs):
                a, b = int(leaf.idx0[group]), int(leaf.idx0[group+1])
                cols.append(leaf.idx[a:b].astype(np.int64))
                rows.append(np.full(b - a, cg, dtype=np.int64))
                rbase[cg] = a - nent
                nent += b - a
            flatpos = (np.concatenate(rows)*self.nflat + np.concatenate(cols)
                       if rows else np.zeros(0, np.int64))
            # the position map indexes a slab's padded grid, which is
            # small, so it is a 32-bit quantity that was being stored
            # in 64; the base is a difference of two filament indices
            # and is signed, but bounded by the same array
            if flatpos.size and int(flatpos.max()) >= 2**31:
                raise OverflowError(
                    "near-field index %d exceeds the 32-bit pack"
                    % int(flatpos.max()))
            if rbase.size and int(np.abs(rbase).max()) >= 2**31:
                raise OverflowError(
                    "near-field source base %d exceeds the 32-bit pack"
                    % int(np.abs(rbase).max()))
            flatpos = flatpos.astype(np.int32)
            rbase = rbase.astype(np.int32)
            # neighbour lists, grouped by target box so the kernel can
            # accumulate in a register and stay reproducible
            off = np.zeros(len(gs) + 1, dtype=np.int32)
            ep, etr, edx = [], [], []
            for cg, group in enumerate(gs):
                for cn in range(27):
                    ngr = int(leaf.neighbors[cn, group])
                    if ngr < 0:
                        continue
                    dx = int(leaf.xidx[ngr]) - cx
                    if dx not in (-1, 0, 1):
                        continue
                    ep.append(int(leaf.revslabidx[ngr]))
                    etr.append(cn)
                    edx.append(dx)
                off[cg + 1] = len(ep)
            mx = max(mx, len(gs))
            packs.append(dict(
                size=len(gs), nent=int(nent),
                flatpos=ocl_core.to_device(flatpos) if nent else None,
                rbase=ocl_core.to_device(rbase) if nent else None,
                off=ocl_core.to_device(off.astype(np.int32)),
                ep=ocl_core.to_device(np.asarray(ep, dtype=np.int32)),
                etr=ocl_core.to_device(np.asarray(etr, dtype=np.int32)),
                edx=ocl_core.to_device(np.asarray(edx, dtype=np.int32))))
        self.packs = packs
        self.maxsize = max(1, mx)

    def _plan(self, size):
        app = self._fft.get(size)
        if app is None:
            app = ocl_core.fft_app((size,) + self.S, self.dtype, ndim=3)
            self._fft[size] = app
        return app

    def _forward(self, cx, slot, data_d):
        """Scatter slab ``cx`` into rolling buffer ``slot`` and transform."""
        q = ocl_core.queue()
        pk = self.packs[cx] if 0 <= cx < self.nslab else None
        buf = self._slab[slot]
        buf.fill(self.dtype.type(0), queue=q)
        if pk is None or pk['size'] == 0 or pk['nent'] == 0:
            return 0
        _n0, n1, n2 = self.n
        _S0, S1, S2 = self.S
        self.k_scatter(
            q, (pk['nent'],), None, data_d.data, pk['rbase'].data,
            pk['flatpos'].data, buf.data, np.uint32(pk['nent']),
            np.uint32(self.nflat), np.uint64(self.GS), np.uint32(n1),
            np.uint32(n2), np.uint32(S1), np.uint32(S2))
        self._plan(pk['size']).fft(buf[:pk['size']])
        return pk['size']

    def apply(self, data, out_=None):
        """Near-field contribution for filament data ``data``.

        Written into ``out_`` when given, which is how the traversal
        calls it: the leaf's data array is held elsewhere and must be
        updated in place rather than rebound.
        """
        q = ocl_core.queue()
        _n0, n1, n2 = self.n
        _S0, S1, S2 = self.S
        host = np.ascontiguousarray(data, dtype=self.dtype)
        if self._data is None or self._data.shape != host.shape:
            self._data = ocl_core.empty(host.shape, self.dtype)
        dev = self._data
        dev.set(host, queue=q)
        # the traversal passes the leaf's own buffer as both input and
        # output, so the upload must be complete before anything writes
        # back into it
        q.finish()
        # IN PLACE. The sweep writes each slab's result into the array
        # it read its input from, which is what the host path does too.
        # It is safe because of the ordering: at step cx the loop
        # forward-transforms slab cx+1 and then gathers slab cx, so a
        # slab's input is last read one step BEFORE its output is
        # written, and the per-slab index sets partition the filaments
        # so no two steps touch the same entry. That is one complex
        # filament array per orientation, 800 MB across the three at
        # R5, for an operator that is bandwidth-bound anyway.
        out = dev
        # a dtype mismatch here should raise rather than silently
        # return a fresh array the caller will discard
        out_host = out_
        # rolling window: order maps dx = -1, 0, +1 to slab buffers.
        # An absent or empty slab leaves its buffer zeroed, so the
        # kernel can read it unconditionally.
        self._forward(0, 2, dev)
        order = [0, 1, 2]
        for cx in range(self.nslab):
            order = [order[1], order[2], order[0]]
            self._forward(cx + 1, order[2], dev)
            pk = self.packs[cx]
            if pk['size'] == 0 or pk['nent'] == 0:
                continue
            nb = pk['size']
            self.k_mac(
                q, (int(nb*self.GS),), None,
                self._trans.data, self._slab[order[0]].data,
                self._slab[order[1]].data, self._slab[order[2]].data,
                pk['off'].data, pk['ep'].data, pk['etr'].data,
                pk['edx'].data, self._tgt.data, np.uint64(self.GS),
                np.uint32(nb))
            self._plan(nb).ifft(self._tgt[:nb])
            self.k_gather(
                q, (pk['nent'],), None, self._tgt.data, pk['rbase'].data,
                pk['flatpos'].data, out.data, np.uint32(pk['nent']),
                np.uint32(self.nflat), np.uint64(self.GS), np.uint32(n1),
                np.uint32(n2), np.uint32(S1), np.uint32(S2))
        return out.get(queue=q, ary=out_host)
