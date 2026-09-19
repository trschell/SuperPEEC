# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Mode-block apply on OpenCL: the km-by-km convolution, one grid at a time.

The enrichment blocks are translation invariant, so applying them is a
convolution on a padded grid. With ``km`` modes per cell the apply is

    accu[m] = conj(Fc[m]) * F  +  sum_n  (Fu[iu[m, n]] or its conjugate)
                                          * U[n]
    accf    = sum_m Fc[m] * U[m]

where ``U[n]`` is the transform of mode ``n``'s coefficients, ``F`` that
of the filament current, and ``Fu`` holds the upper triangle of the
block spectra (reciprocity makes block (n, m) the conjugate of (m, n),
so ``iu`` maps each pair into ``km(km+1)/2`` stored grids and the sign
of ``n - m`` says which way to read it).

Sized for the card, not for convenience
---------------------------------------
These padded grids are the largest device allocation in the corpus: the
memory survey measured the km input slabs plus three more at 3.1 GiB
per matvec on the RSFQ XNOR, the model closest to the card's limit. The
first cut of this module ignored that, kept every grid in double
precision and held a second stack for the outputs, and cost 0.5 GiB of
card on exactly that model. It now follows the host's two economies.

*The input slabs carry the stored spectra's precision*, single by
default at engineering tolerances, while every product and sum
accumulates in double. The transforms run in double and the result is
rounded on the way into the slab, as the host does, so only the inputs
are rounded and they were already stored that way.

*There is one output grid, not km of them.* Each output harmonic is
contracted into a single accumulator, transformed back and gathered
before the next begins. Transforming the whole stack at once would be
faster and would cost another km padded grids; with the card at 80% on
the XNOR and the host at 20%, that is the wrong way to spend it.

The device state is keyed to the spectra generation, because the
spectra are rebuilt per frequency and a cached upload would otherwise
be applied to the next one.
"""
import numpy as np

import ocl_core

SOURCE = """
typedef STO sto_t;

inline cplx_t s2a(sto_t v) { return (cplx_t)((real_t)v.x, (real_t)v.y); }

/* one output harmonic, into a single accumulator */
__kernel void mode_one(__global const sto_t *Fu,    /* (NU, GP) */
                       __global const sto_t *Fc,    /* (KM, GP) */
                       __global const sto_t *U,     /* (KM, GP) */
                       __global const sto_t *Fv,    /* (GP,)    */
                       __global const int *iu,      /* (KM, KM) */
                       __global cplx_t *acc,        /* (GP,)    */
                       const unsigned long GP,
                       const unsigned int m)
{
    const unsigned long g = get_global_id(0);
    if (g >= GP) return;
    cplx_t a = cmul(cconj(s2a(Fc[(unsigned long)m*GP + g])), s2a(Fv[g]));
    for (unsigned int n = 0; n < KM; ++n) {
        const cplx_t fmn = s2a(Fu[(unsigned long)iu[m*KM + n]*GP + g]);
        a += cmul((n >= m) ? cconj(fmn) : fmn,
                  s2a(U[(unsigned long)n*GP + g]));
    }
    acc[g] = a;
}

/* the filament output: sum_m Fc[m] * U[m] */
__kernel void mode_fil(__global const sto_t *Fc,
                       __global const sto_t *U,
                       __global cplx_t *acc,
                       const unsigned long GP)
{
    const unsigned long g = get_global_id(0);
    if (g >= GP) return;
    cplx_t a = cmul(s2a(Fc[g]), s2a(U[g]));
    for (unsigned int m = 1; m < KM; ++m)
        a += cmul(s2a(Fc[(unsigned long)m*GP + g]),
                  s2a(U[(unsigned long)m*GP + g]));
    acc[g] = a;
}

/* round a transformed grid down into one slab of the input stack */
__kernel void to_slab(__global const cplx_t *src, __global sto_t *dst,
                      const unsigned long GP, const unsigned long off)
{
    const unsigned long g = get_global_id(0);
    if (g < GP) dst[off + g] = (sto_t)((real_t)src[g].x,
                                       (real_t)src[g].y);
}

__kernel void scatter_cells(__global const cplx_t *src,
                            __global const long *gflat,
                            __global cplx_t *pad,
                            const unsigned int ncell)
{
    const unsigned int c = get_global_id(0);
    if (c >= ncell) return;
    pad[gflat[c]] = src[c];
}

__kernel void gather_cells(__global const cplx_t *pad,
                           __global const long *gflat,
                           __global cplx_t *dst,
                           const unsigned int ncell)
{
    const unsigned int c = get_global_id(0);
    if (c >= ncell) return;
    dst[c] = pad[gflat[c]];
}
"""


class ModeApply(object):
    """Device state and apply for one enrichment's mode blocks."""

    def __init__(self, enr):
        self.km = int(enr.km)
        self.pad = tuple(int(v) for v in enr.pad)
        self.GP = int(np.prod(self.pad))
        self.acc = np.dtype(np.complex128)
        # the slabs take the stored spectra's precision: the kernel
        # reads the spectra and the slabs through one type, and the
        # enrichment's own lean rule is what chose that storage
        self.sdt = np.dtype(enr.Fu.dtype)
        self.nu = int(enr.Fu.shape[0])
        g3 = enr._g3
        gflat = ((g3[0].astype(np.int64)*self.pad[1] + g3[1])*self.pad[2]
                 + g3[2])
        self.ncell = int(gflat.size)
        self._gflat = ocl_core.to_device(gflat)
        # a palette may expose only a subset of the modes; the apply
        # runs on the full set and the caller sees the masked one, as
        # the host path does
        self.nmode = int(enr.nmode)
        self.nmode_full = int(enr.nmode_full)
        self.mask = (np.asarray(enr.mode_mask)
                     if self.nmode != self.nmode_full else None)
        self._Fu = ocl_core.to_device(
            np.ascontiguousarray(enr.Fu).reshape(self.nu, -1), self.sdt)
        self._Fc = ocl_core.to_device(
            np.ascontiguousarray(enr.Fc).reshape(self.km, -1), self.sdt)
        self._iu = ocl_core.to_device(
            np.ascontiguousarray(enr._iu).astype(np.int32))
        sto = 'float2' if self.sdt == np.complex64 else 'double2'
        self.prg = ocl_core.program(SOURCE, self.acc,
                                    {'KM': self.km, 'NU': self.nu,
                                     'STO': sto}, key='ocl_modes/' + sto)
        self._k = {n: ocl_core.kernel(self.prg, n)
                   for n in ('mode_one', 'mode_fil', 'to_slab',
                             'scatter_cells', 'gather_cells')}
        # one stack of input slabs, one double-precision work grid and
        # one accumulator; no second stack
        self._U = ocl_core.zeros((self.km, self.GP), self.sdt)
        self._F = ocl_core.zeros((self.GP,), self.sdt)
        self._tmp = ocl_core.zeros((self.GP,), self.acc)
        self._acc = ocl_core.zeros((self.GP,), self.acc)
        self._cell = ocl_core.empty((self.ncell,), self.acc)
        self.fft1 = ocl_core.fft_app(self.pad, self.acc, ndim=3)

    def device_bytes(self):
        """Resident device bytes, for sizing checks and the survey."""
        return int(self._Fu.nbytes + self._Fc.nbytes + self._U.nbytes
                   + self._F.nbytes + self._tmp.nbytes + self._acc.nbytes)

    # ------------------------------------------------------------ steps

    def _forward(self, host_vals, dst, off):
        """Scatter, transform in double, round into the slab stack."""
        q = ocl_core.queue()
        self._cell.set(np.ascontiguousarray(host_vals, dtype=self.acc),
                       queue=q)
        self._tmp.fill(self.acc.type(0), queue=q)
        self._k['scatter_cells'](q, (self.ncell,), None, self._cell.data,
                                 self._gflat.data, self._tmp.data,
                                 np.uint32(self.ncell))
        self.fft1.fft(self._tmp)
        self._k['to_slab'](q, (self.GP,), None, self._tmp.data, dst.data,
                           np.uint64(self.GP), np.uint64(off))

    def _back(self, out, sl):
        """Inverse-transform the accumulator and gather into ``out``."""
        q = ocl_core.queue()
        self.fft1.ifft(self._acc)
        self._k['gather_cells'](q, (self.ncell,), None, self._acc.data,
                                self._gflat.data, self._cell.data,
                                np.uint32(self.ncell))
        out[sl] = self._cell.get(queue=q)

    def apply(self, u, i_f):
        """Return ``(out_u, out_f)`` for mode coefficients ``u``.

        Same contract as the host ``apply_fft``: ``u`` is the masked
        coefficient vector where the palette exposes a subset, and the
        returned modes are masked to match.
        """
        q = ocl_core.queue()
        km = self.km
        if self.mask is not None:
            uf = np.zeros(self.nmode_full, dtype=self.acc)
            uf[self.mask] = u
            u = uf
        for m in range(km):
            self._forward(u[m::km], self._U, m*self.GP)
        self._forward(i_f, self._F, 0)

        out_u = np.empty(km*self.ncell, dtype=self.acc)
        for m in range(km):
            self._k['mode_one'](q, (self.GP,), None, self._Fu.data,
                                self._Fc.data, self._U.data, self._F.data,
                                self._iu.data, self._acc.data,
                                np.uint64(self.GP), np.uint32(m))
            self._back(out_u, slice(m, None, km))
        self._k['mode_fil'](q, (self.GP,), None, self._Fc.data,
                            self._U.data, self._acc.data,
                            np.uint64(self.GP))
        out_f = np.empty(self.ncell, dtype=self.acc)
        self._back(out_f, slice(None))
        if self.mask is not None:
            out_u = out_u[self.mask]
        return out_u, out_f
