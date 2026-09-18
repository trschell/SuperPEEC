# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Mode-block apply on OpenCL: the km-by-km convolution, fused.

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

Why fused: the CuPy path walks the (m, n) pairs and writes a full
padded grid for each one before adding it in, so a km of 4 moves 16
grid-sized temporaries through memory per apply for arithmetic that
needs one pass. Here each work item owns one grid point, reads its km
moments and km cross spectra once, and accumulates all km + 1 outputs
in registers.

Precision follows the CuPy path where it matters: the spectra are read
in whatever dtype they were built in (complex64 by default, which is
the whole point of the triangular storage) and every product and sum is
accumulated in complex128. The transforms run in complex128 here rather
than in the lean slab dtype, which costs device memory and gives up
nothing in accuracy.
"""
import numpy as np

import ocl_core

SOURCE = """
typedef STO sto_t;

inline cplx_t s2a(sto_t v) { return (cplx_t)((real_t)v.x, (real_t)v.y); }

__kernel void mode_contract(__global const sto_t *Fu,   /* (NU, GP) */
                            __global const sto_t *Fc,   /* (KM, GP) */
                            __global const cplx_t *U,   /* (KM, GP) */
                            __global const cplx_t *Fv,  /* (GP,)    */
                            __global const int *iu,     /* (KM, KM) */
                            __global cplx_t *accu,      /* (KM, GP) */
                            __global cplx_t *accf,      /* (GP,)    */
                            const unsigned long GP)
{
    const unsigned long g = get_global_id(0);
    if (g >= GP) return;

    cplx_t u[KM], fc[KM];
    for (unsigned int m = 0; m < KM; ++m) {
        u[m] = U[(unsigned long)m*GP + g];
        fc[m] = s2a(Fc[(unsigned long)m*GP + g]);
    }
    const cplx_t fv = Fv[g];

    for (unsigned int m = 0; m < KM; ++m) {
        cplx_t acc = cmul(cconj(fc[m]), fv);
        for (unsigned int n = 0; n < KM; ++n) {
            const cplx_t fmn =
                s2a(Fu[(unsigned long)iu[m*KM + n]*GP + g]);
            acc += cmul((n >= m) ? cconj(fmn) : fmn, u[n]);
        }
        accu[(unsigned long)m*GP + g] = acc;
    }

    cplx_t af = cmul(fc[0], u[0]);
    for (unsigned int m = 1; m < KM; ++m)
        af += cmul(fc[m], u[m]);
    accf[g] = af;
}

/* coefficients (ncell,) -> one grid of a padded stack, at element
   offset `off`. The offset is explicit rather than taken from a
   sliced device array, whose .data is the whole buffer. */
__kernel void scatter_cells(__global const cplx_t *src,
                            __global const long *gflat,
                            __global cplx_t *pad,
                            const unsigned int ncell,
                            const unsigned long off)
{
    const unsigned int c = get_global_id(0);
    if (c >= ncell) return;
    pad[off + (unsigned long)gflat[c]] = src[c];
}

/* one grid of a padded stack -> coefficients */
__kernel void gather_cells(__global const cplx_t *pad,
                           __global const long *gflat,
                           __global cplx_t *dst,
                           const unsigned int ncell,
                           const unsigned long off)
{
    const unsigned int c = get_global_id(0);
    if (c >= ncell) return;
    dst[c] = pad[off + (unsigned long)gflat[c]];
}
"""


class ModeApply(object):
    """Device state and apply for one enrichment's mode blocks."""

    def __init__(self, enr):
        self.km = int(enr.km)
        self.pad = tuple(int(v) for v in enr.pad)
        self.GP = int(np.prod(self.pad))
        self.sdt = np.dtype(enr.Fu.dtype)
        self.acc = np.dtype(np.complex128)
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
        self.k_contract = ocl_core.kernel(self.prg, 'mode_contract')
        self.k_scatter = ocl_core.kernel(self.prg, 'scatter_cells')
        self.k_gather = ocl_core.kernel(self.prg, 'gather_cells')
        self._U = ocl_core.zeros((self.km,) + self.pad, self.acc)
        self._F = ocl_core.zeros(self.pad, self.acc)
        self._au = ocl_core.zeros((self.km,) + self.pad, self.acc)
        self._af = ocl_core.zeros(self.pad, self.acc)
        self._cell = ocl_core.empty((self.ncell,), self.acc)
        self.fft_k = ocl_core.fft_app((self.km,) + self.pad, self.acc,
                                      ndim=3)
        self.fft_1 = ocl_core.fft_app(self.pad, self.acc, ndim=3)

    def _cells_to_grid(self, host_vals, grid, row=0):
        """Scatter ``host_vals`` onto grid ``row`` of a padded stack."""
        q = ocl_core.queue()
        self._cell.set(np.ascontiguousarray(host_vals, dtype=self.acc),
                       queue=q)
        self.k_scatter(q, (self.ncell,), None, self._cell.data,
                       self._gflat.data, grid.data, np.uint32(self.ncell),
                       np.uint64(row*self.GP))

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
        self._U.fill(self.acc.type(0), queue=q)
        for m in range(km):
            self._cells_to_grid(u[m::km], self._U, row=m)
        self.fft_k.fft(self._U)
        self._F.fill(self.acc.type(0), queue=q)
        self._cells_to_grid(i_f, self._F)
        self.fft_1.fft(self._F)

        self.k_contract(q, (self.GP,), None, self._Fu.data, self._Fc.data,
                        self._U.data, self._F.data, self._iu.data,
                        self._au.data, self._af.data, np.uint64(self.GP))

        self.fft_k.ifft(self._au)
        self.fft_1.ifft(self._af)
        out_u = np.empty(km*self.ncell, dtype=self.acc)
        for m in range(km):
            self.k_gather(q, (self.ncell,), None, self._au.data,
                          self._gflat.data, self._cell.data,
                          np.uint32(self.ncell), np.uint64(m*self.GP))
            out_u[m::km] = self._cell.get(queue=q)
        self.k_gather(q, (self.ncell,), None, self._af.data,
                      self._gflat.data, self._cell.data,
                      np.uint32(self.ncell), np.uint64(0))
        out_f = self._cell.get(queue=q)
        if self.mask is not None:
            out_u = out_u[self.mask]
        return out_u, out_f
