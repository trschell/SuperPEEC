# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for the OPT-IN c64 mid-level transfer tables (SPPEEC_MID_FP32=1).

The default stays fp64 -- the c64 CPU kernel measured 0.82x (the table
is fixed-size and L3-resident, the kernel compute-bound, and the
mixed-precision inner loop pays a per-element conversion) -- but the
normalised-c64 machinery is kept for a future GPU mid-M2L port, and
this gate keeps it from rotting:

  A. the normalisation survives the ftrans dynamic-range trap: raw
     mid tables carry 1/r**(j+n+1) in SI metres and overflow float32
     (measured 5.2e38 at a 1e-5 m pitch); the stored c64 table must be
     finite with the magnitude factored into the fp64 u/v scales;
  B. the c64 m2l matches the fp64 m2l to fp32 storage rounding on a
     multipole-scaled random input;
  C. farneighbors is int32 (matches the Fortran INTEGER argument --
     int64 made f2py copy-convert the whole array every m2l call);
  D. the default (no env) really is fp64 with no scale vectors.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import os
import sys

import numpy as np

sys.path.insert(0, 'src')

FAIL = []


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


def build(fp32):
    import voxmodel
    if fp32:
        os.environ['SPPEEC_MID_FP32'] = '1'
    else:
        os.environ.pop('SPPEEC_MID_FP32', None)
    m = voxmodel.VoxelModel('block')
    m.dims = (64, 64, 16)
    m.d = 1e-5              # the pitch class whose raw tables hit 5e38
    m.sigma = np.full(m.dims, 5.8e7, dtype=np.float32)
    m.freq = np.array([1e8])
    return m.build_tree(nleaf=[4, 4, 4], numlevels=3)


def main():
    M64, M32 = build(False), build(True)
    mid64, mid32 = M64.lv[1], M32.lv[1]

    check('default stores fp64, no scales',
          mid64.transfer.dtype == np.complex128 and mid64._mid_u is None,
          str(mid64.transfer.dtype))
    check('opt-in stores c64 + fp64 scales',
          mid32.transfer.dtype == np.complex64
          and mid32._mid_u is not None
          and mid32._mid_u.dtype == np.float64,
          str(mid32.transfer.dtype))
    raw_max = np.abs(mid64.transfer).max()
    check('raw fp64 table exceeds float32 range (the trap is live '
          'on this geometry)', raw_max > np.finfo(np.float32).max,
          'max %.2e' % raw_max)
    check('normalised c64 table all finite',
          bool(np.all(np.isfinite(mid32.transfer))),
          'max %.2e' % np.abs(mid32.transfer).max())
    check('farneighbors int32',
          mid32.farneighbors.dtype == np.int32
          and mid64.farneighbors.dtype == np.int32,
          str(mid32.farneighbors.dtype))

    ng = np.size(mid64.idx)
    rng = np.random.default_rng(3)
    d = (rng.standard_normal((ng, mid64.nnmax))
         + 1j*rng.standard_normal((ng, mid64.nnmax)))
    deg = np.floor(np.sqrt(np.arange(mid64.nnmax)))
    d = d * (2e-5)**deg[None, :]        # q*r^n multipole magnitudes
    mid64.data = d.copy()
    mid64.m2l()
    mid32.data = d.copy()
    mid32.m2l()
    rel = (np.linalg.norm(mid32.data - mid64.data)
           / np.linalg.norm(mid64.data))
    check('c64 m2l == fp64 m2l to fp32 storage rounding',
          rel < 5e-7, 'rel %.2e' % rel)

    print('\n%d checks failed' % len(FAIL))
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
