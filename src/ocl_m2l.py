# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Top-level all-to-all M2L on OpenCL, as one fused contraction.

The stage convolves every top-box multipole against every other through
the translation operator. On the padded spectral grid it is

    lnm[i, g] = sum_j  C[i, j] * ft[T[i, j], g] * fmg[j, g]

over the ``nn = (nmax+1)^2`` harmonics, where ``T`` maps each output
pair to one of ``nt = (2 nmax + 1)^2`` translation channels and ``C``
carries its weight. Every pair has exactly one channel, so the whole
operator is 625 scalar products per grid point at nmax 4.

Why this is a fused kernel rather than array expressions
-------------------------------------------------------
The CuPy path walks the output harmonics, and for each one gathers that
row's channel spectra into a temporary, multiplies, then contracts. The
arithmetic is minimal but the traffic is not: each of the nn passes
reads and writes an (nn, G) temporary, about 14.7 GB per call on the R3
flagship, and the stage measures 54.4 ms there while the card sustains
426 GB/s. Held in local memory instead, a grid point needs its nt
channel values and its nn moments once: 131 complex reads and nn
writes, about 0.6 GB per call for the same 1.47 GFLOP. That moves the
stage from far below the bandwidth roof to just above the fp64 one.

Each work group takes ``TG`` grid points, cooperatively stages their
channel spectra and moments in local memory, and then computes one
output harmonic per work item. Local footprint is
``TG * (nt + nn) * 16`` bytes, 27 kB at nmax 4 with TG 16.

Summation runs over ``j`` ascending, the order the Fortran kernel uses,
so the result tracks the host path to fp64 rounding rather than
drifting from a different reduction tree.
"""
import os

import numpy as np

import ocl_core

SOURCE = """
/* lnm[i, g] = sum_j C[i, j] * ft[T[i, j], g] * fmg[j, g] */
__kernel void m2l_contract(__global const cplx_t *ft,    /* (NT, G) */
                           __global const cplx_t *fmg,   /* (NN, G) */
                           __global const int    *Tm,    /* (NN, NN) */
                           __global const cplx_t *Cm,    /* (NN, NN) */
                           __global cplx_t *lnm,         /* (NN, G) */
                           const unsigned long G)
{
    __local cplx_t lft[TG][NT];
    __local cplx_t lfm[TG][NN];
    const unsigned int lid = get_local_id(0);
    const unsigned long g0 = (unsigned long)get_group_id(0)*TG;

    for (unsigned int s = lid; s < TG*NT; s += TG*NN) {
        const unsigned int gg = s/NT, t = s - gg*NT;
        const unsigned long g = g0 + gg;
        lft[gg][t] = (g < G) ? ft[(unsigned long)t*G + g]
                             : (cplx_t)(0, 0);
    }
    for (unsigned int s = lid; s < TG*NN; s += TG*NN) {
        const unsigned int gg = s/NN, j = s - gg*NN;
        const unsigned long g = g0 + gg;
        lfm[gg][j] = (g < G) ? fmg[(unsigned long)j*G + g]
                             : (cplx_t)(0, 0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int gg = lid/NN, i = lid - gg*NN;
    const unsigned long g = g0 + gg;
    if (g < G) {
        cplx_t acc = (cplx_t)(0, 0);
        for (unsigned int j = 0; j < NN; ++j) {
            const cplx_t c = Cm[i*NN + j];
            if (c.x != (real_t)0 || c.y != (real_t)0)
                acc += cmul(c, cmul(lft[gg][Tm[i*NN + j]], lfm[gg][j]));
        }
        lnm[(unsigned long)i*G + g] = acc;
    }
}

/* One translation channel at a time, for the streamed operator.
   Only the (i, j) pairs that map to this channel contribute, about
   NN*NN/NT of them, so this costs the same arithmetic as the resident
   contraction and needs only the channel's own grid on the card.
   Each work item owns one grid point and walks the pair list in a
   fixed order, so nothing is scattered and the result is reproducible;
   it differs from the resident form only in summation order. */
__kernel void m2l_channel(__global const cplx_t *ftt,   /* (G,)     */
                          __global const cplx_t *fmg,   /* (NN, G)  */
                          __global const int *pi,
                          __global const int *pj,
                          __global const cplx_t *pc,
                          __global cplx_t *lnm,         /* (NN, G)  */
                          const unsigned int npair,
                          const unsigned long G)
{
    const unsigned long g = get_global_id(0);
    if (g >= G) return;
    const cplx_t f = ftt[g];
    for (unsigned int p = 0; p < npair; ++p) {
        const unsigned long o = (unsigned long)pi[p]*G + g;
        lnm[o] += cmul(cmul(pc[p], f), fmg[(unsigned long)pj[p]*G + g]);
    }
}

__kernel void zero_grid(__global cplx_t *a, const unsigned long n)
{
    const unsigned long i = get_global_id(0);
    if (i < n) a[i] = (cplx_t)(0, 0);
}

/* moments (ncell, NN) -> zeroed padded spectral grid (NN, G) */
__kernel void scatter_pad(__global const cplx_t *data,
                          __global const int *idx,
                          __global cplx_t *padg,
                          const unsigned int ncell, const unsigned long G,
                          const unsigned int n1, const unsigned int n2,
                          const unsigned int S1, const unsigned int S2)
{
    const unsigned long q = get_global_id(0);
    if (q >= (unsigned long)ncell*NN) return;
    const unsigned int c = (unsigned int)(q/NN), i = (unsigned int)(q - (unsigned long)c*NN);
    const unsigned int f = (unsigned int)idx[c];
    const unsigned int a = f/(n1*n2), b = (f/n2) % n1, d = f % n2;
    padg[(unsigned long)i*G + ((unsigned long)a*S1 + b)*S2 + d] = data[q];
}

/* padded grid (NN, G) -> moments (ncell, NN) */
__kernel void gather_pad(__global const cplx_t *padg,
                         __global const int *idx,
                         __global cplx_t *data,
                         const unsigned int ncell, const unsigned long G,
                         const unsigned int n1, const unsigned int n2,
                         const unsigned int S1, const unsigned int S2)
{
    const unsigned long q = get_global_id(0);
    if (q >= (unsigned long)ncell*NN) return;
    const unsigned int c = (unsigned int)(q/NN), i = (unsigned int)(q - (unsigned long)c*NN);
    const unsigned int f = (unsigned int)idx[c];
    const unsigned int a = f/(n1*n2), b = (f/n2) % n1, d = f % n2;
    data[q] = padg[(unsigned long)i*G + ((unsigned long)a*S1 + b)*S2 + d];
}
"""


def smooth5(k):
    """The smallest 2/3/5-smooth integer at least ``k``."""
    def ok(x):
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        return x == 1
    while not ok(k):
        k += 1
    return k


def operator_tables(level):
    """Host-side M2L tables, shared by every backend.

    Returns ``(T, C, Ct, S, ft2, pairs)``: the per-pair channel map
    and weight, the per-channel masked weights, the 2/3/5-smooth padded
    grid, the channel spectra re-embedded on it, and the same operator
    as a per-channel list of (output, moment, weight).

    The padded size matters: the natural ``2n`` padding can carry large
    prime factors (R4's top grid pads to 2x107 and 4x67, both prime,
    which puts the transform on a Bluestein path about ten times off
    radix). A circular convolution of the ``(2n-1)``-offset kernel with
    ``n``-support moments is exact at any size at least ``2n-1``, so the
    kernel is recovered from the stored ``2n`` spectra and re-embedded
    on the next smooth grid.
    """
    nmax = int(level.nmax)
    nn = int(level.nnmax)
    nt = (2*nmax + 1)**2
    T = np.zeros((nn, nn), dtype=np.int64)
    C = np.zeros((nn, nn), dtype=np.complex128)
    for n in range(nmax + 1):
        for m in range(-n, n + 1):
            idxnm = n*n + n + m
            for j in range(nmax + 1):
                for k in range(-j, j + 1):
                    idxjk = j*j + j + k
                    T[idxnm, idxjk] = (j + n)**2 + (j + n) + (k - m)
                    C[idxnm, idxjk] = level.c[nn*idxnm + idxjk]
    Ct = np.zeros((nt, nn, nn), dtype=np.complex128)
    for t in range(nt):
        Ct[t][T == t] = C[T == t]
    # the same operator as a per-channel pair list, which is what the
    # streamed form walks: channel t contributes C[i, j] to output i
    # from moment j, for the pairs that map to it
    pairs = []
    for t in range(nt):
        ii, jj = np.nonzero((T == t) & (C != 0))
        pairs.append((ii.astype(np.int32), jj.astype(np.int32),
                      C[ii, jj].copy()))

    n0, n1, n2 = (int(v) for v in level.n)
    S = tuple(smooth5(2*v - 1) for v in (n0, n1, n2))
    ft2 = np.empty((nt, int(np.prod(S))), dtype=np.complex128)
    srcs, dsts = [], []
    for ni, Si in zip((n0, n1, n2), S):
        srcs.append(np.r_[0:ni, ni + 1:2*ni])
        dsts.append(np.r_[0:ni, Si - (ni - 1):Si])
    scale = level._ftrans_scale
    for t in range(nt):
        k2n = np.fft.ifftn((level.ftrans[t].astype(np.complex128)*scale[t])
                           .reshape(2*n0, 2*n1, 2*n2))
        emb = np.zeros(S, dtype=np.complex128)
        emb[np.ix_(dsts[0], dsts[1], dsts[2])] = \
            k2n[np.ix_(srcs[0], srcs[1], srcs[2])]
        ft2[t] = np.fft.fftn(emb).ravel()
    return T, C, Ct, S, ft2, pairs


class TopM2L(object):
    """Device state and apply for one top level."""

    TG = 16                      # grid points per work group

    def __init__(self, level, dtype=np.complex128):
        self.dtype = np.dtype(dtype)
        self.nn = int(level.nnmax)
        self.nt = (2*int(level.nmax) + 1)**2
        self.n = tuple(int(v) for v in level.n)
        T, C, _Ct, S, ft2, pairs = operator_tables(level)
        self.S = S
        self.G = int(np.prod(S))
        self.ncell = int(np.size(level.idx))
        itm = self.dtype.itemsize
        # Resident when the channel spectra fit, streamed when they do
        # not. The whole operator is nt grids; streaming holds one at a
        # time and pays a host transfer per channel per matvec instead,
        # which is far slower but is the difference between running a
        # model and not. The card is the binding resource here (the
        # XNOR sits at 70% of it while host memory is at 20%), so the
        # budget is checked against the card and the host keeps the
        # operator.
        import backend
        mode = os.environ.get('SPPEEC_OCL_M2L', 'auto').strip().lower()
        total = backend.device_memory_total()
        budget = 0.6*total if total else None
        resident_bytes = (self.nt + 2*self.nn)*self.G*itm
        stream_bytes = (3*self.nn + 1)*self.G*itm
        if mode == 'resident':
            self.streamed = False
        elif mode == 'stream':
            self.streamed = True
        else:
            self.streamed = budget is not None and resident_bytes > budget
        if self.streamed and budget is not None and stream_bytes > budget:
            raise MemoryError(
                "OpenCL top-level M2L needs %.1f GB even streamed (3 work "
                "grids of %d points) against a %.1f GB budget"
                % (stream_bytes/1e9, self.G, budget/1e9))
        if self.streamed:
            self._ft = None
            self._ft_host = ft2                 # stays on the host
            self._ftt = ocl_core.zeros((self.G,), self.dtype)
            self._acc = ocl_core.zeros((self.nn,) + S, self.dtype)
            self._used = [t for t in range(self.nt) if pairs[t][0].size]
            self._pairs = [
                (ocl_core.to_device(pairs[t][0]),
                 ocl_core.to_device(pairs[t][1]),
                 ocl_core.to_device(pairs[t][2], self.dtype))
                for t in self._used]
        else:
            self._ft = ocl_core.to_device(ft2, self.dtype)
            self._ft_host = None
        self._T = ocl_core.to_device(T.astype(np.int32))
        self._C = ocl_core.to_device(C, self.dtype)
        self._idx = ocl_core.to_device(np.asarray(level.idx, dtype=np.int32))
        self._pad = ocl_core.zeros((self.nn,) + S, self.dtype)
        self._lnm = ocl_core.empty((self.nn,) + S, self.dtype)
        self._data = ocl_core.empty((self.ncell, self.nn), self.dtype)
        self.prg = ocl_core.program(
            SOURCE, self.dtype,
            {'NN': self.nn, 'NT': self.nt, 'TG': self.TG},
            key='ocl_m2l')
        self.fft = ocl_core.fft_app((self.nn,) + S, self.dtype, ndim=3)
        self.k_scatter = ocl_core.kernel(self.prg, 'scatter_pad')
        self.k_contract = ocl_core.kernel(self.prg, 'm2l_contract')
        self.k_gather = ocl_core.kernel(self.prg, 'gather_pad')
        self.k_channel = ocl_core.kernel(self.prg, 'm2l_channel')
        self.k_zero = ocl_core.kernel(self.prg, 'zero_grid')

    def device_bytes(self):
        """Resident device bytes for this operator and its work grids."""
        itm = self.dtype.itemsize
        n = (3*self.nn + 1) if self.streamed else (self.nt + 2*self.nn)
        return int(n*self.G*itm)

    def apply(self, data):
        """Local expansions for moments ``data`` of shape (ncell, nn)."""
        q = ocl_core.queue()
        nn, G = self.nn, self.G
        _n0, n1, n2 = self.n
        _S0, S1, S2 = self.S
        shape = (np.uint32(self.ncell), np.uint64(G), np.uint32(n1),
                 np.uint32(n2), np.uint32(S1), np.uint32(S2))
        nq = (self.ncell*nn,)
        self._data.set(np.ascontiguousarray(data, dtype=self.dtype), queue=q)
        self._pad.fill(self.dtype.type(0), queue=q)
        self.k_scatter(q, nq, None, self._data.data, self._idx.data,
                       self._pad.data, *shape)
        self.fft.fft(self._pad)
        if self.streamed:
            out = self._acc
            self.k_zero(q, (int(nn*G),), None, out.data,
                        np.uint64(nn*G))
            for t, (pi, pj, pc) in zip(self._used, self._pairs):
                self._ftt.set(self._ft_host[t], queue=q)
                self.k_channel(q, (G,), None, self._ftt.data,
                               self._pad.data, pi.data, pj.data, pc.data,
                               out.data, np.uint32(pi.size), np.uint64(G))
        else:
            out = self._lnm
            groups = (G + self.TG - 1)//self.TG
            self.k_contract(q, (groups*self.TG*nn,), (self.TG*nn,),
                            self._ft.data, self._pad.data, self._T.data,
                            self._C.data, out.data, np.uint64(G))
        self.fft.ifft(out)
        self.k_gather(q, nq, None, out.data, self._idx.data,
                      self._data.data, *shape)
        return self._data.get(queue=q)
