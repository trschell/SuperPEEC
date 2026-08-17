# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for the GPU GeoMG apply, including SPLIT RESIDENCY (2026-08-15).

Covers, on a real (small) plaquette Gram from studies/geomg.build_gram:
  A. single-device GPU apply matches the CPU apply (fp32 rounding);
  B. the forced split (devices=(d,d)) is a valid same-device plumbing
     test and matches the single-device apply almost exactly -- the
     split adds only vector copies, no different arithmetic;
  C. the synthetic-budget auto path (SPPEEC_GPU_BUDGET_GB) either
     engages a real two-device split (multi-GPU box) or raises so the
     factor records a CPU fallback (single-GPU box) -- NEVER a silent
     wrong residency;
  D. _GeoMGFactor.gpu_state answers "did the GPU engage" (the probe
     the 2026-08-12 R4 benchmark was missing).

SKIPS (exit 0) when cupy or a CUDA device is unavailable: the suite
must stay green on CPU-only boxes; this file gates GPU boxes only.
The dual-GPU transfer-overhead numbers still need a real 2-GPU box --
this validator proves correctness of the plumbing, not the ~10 ms
transfer estimate.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import os
import sys

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'studies')

FAIL = []


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


def main():
    try:
        import cupy as cp
        ndev = cp.cuda.runtime.getDeviceCount()
        cp.zeros(4).sum()          # force a real context, not just import
    except Exception as exc:
        print('SKIP: no usable GPU (%s: %s) -- validator gates GPU '
              'boxes only' % (type(exc).__name__, exc))
        return
    print('devices: %d' % ndev)

    from geomg import build_gram
    from port_impedance import _GeoMGFactor
    from gpu_amg import GPUGeoBlock, GPUGeoCore

    # small but >= 3 MG levels, and hold back 4 plaquette columns as
    # synthetic macro columns so the Schur branch of __call__ runs too
    _, normal, base, Y = build_gram(16, 16, 6)
    nmac = 4
    nplaq = Y.shape[1] - nmac
    YT = Y.T.tocsr().astype(np.float32)

    os.environ.pop('SPPEEC_GPU_SPLIT', None)
    os.environ.pop('SPPEEC_GPU_BUDGET_GB', None)
    os.environ['SPPEEC_GPU'] = '1'
    fac = _GeoMGFactor(YT, normal[:nplaq], base[:nplaq], nplaq)
    nlev = len(fac.mg.levels)
    print('hierarchy: %d levels, nplaq %d, nmac %d'
          % (nlev, nplaq, fac.nmac))
    check('>= 2 levels (a split needs a boundary)', nlev >= 2,
          '%d' % nlev)

    rng = np.random.default_rng(7)
    b = np.float32(rng.standard_normal(fac.n))

    # D. engagement probe
    check("gpu_state reports engagement", fac.gpu_state.startswith(
        'gpu single('), fac.gpu_state)

    # A. CPU vs GPU single-device
    saved, fac._gpu = fac._gpu, None
    y_cpu = fac(b)
    fac._gpu = saved
    y_gpu = fac(b)
    rel = np.linalg.norm(y_cpu - y_gpu)/np.linalg.norm(y_cpu)
    check('GPU apply == CPU apply to fp32 rounding', rel < 5e-5,
          'rel %.2e' % rel)

    # B. forced same-device split: plumbing only, near-exact
    dsplit = GPUGeoBlock(fac, devices=(0, 0))
    check('forced split placement recorded',
          dsplit.core.placement.startswith('split(')
          and dsplit.core.devs[0] == 0 and dsplit.core.devs[-1] == 0,
          dsplit.core.placement)
    y_split = dsplit(b)
    rel = np.linalg.norm(y_split - y_gpu)/np.linalg.norm(y_gpu)
    check('split apply == single apply', rel < 1e-6, 'rel %.2e' % rel)

    # env-driven forced split reaches production without params
    os.environ['SPPEEC_GPU_SPLIT'] = '0,0'
    fac_env = _GeoMGFactor(YT, normal[:nplaq], base[:nplaq], nplaq)
    check('SPPEEC_GPU_SPLIT env engages the split',
          fac_env.gpu_state.startswith('gpu split('), fac_env.gpu_state)
    y_env = fac_env(b)
    rel = np.linalg.norm(y_env - y_split)/np.linalg.norm(y_split)
    check('env split == param split', rel < 1e-6, 'rel %.2e' % rel)
    del os.environ['SPPEEC_GPU_SPLIT']

    # C. synthetic budget: total no longer fits one device's budget.
    # Budget chosen between the biggest half and the total, so a
    # 2-device split fits but single-device residency does not.
    core = GPUGeoCore(fac.mg, fac.cycles)
    lev0 = None
    # recompute the two halves the same way the constructor does
    itm = np.dtype(core.dtype).itemsize

    def csr_bytes(M):
        M = M.tocsr()
        return M.nnz*itm + M.indices.nbytes + M.indptr.nbytes

    lev = [csr_bytes(L) for L in fac.mg.levels]
    for i, P in enumerate(fac.mg.Ps):
        lev[i] += 2*csr_bytes(P)
    for i, d in enumerate(fac.mg.dinv):
        lev[min(i, nlev - 1)] += d.size*itm
    lev[-1] += fac.mg.coarse_pinv.size*itm
    lev0, rest, total = lev[0], sum(lev[1:]), sum(lev)
    budget = max(lev0, rest)*1.05
    check('test geometry gives a splittable hierarchy',
          budget < total, 'l0 %d, rest %d bytes' % (lev0, rest))
    os.environ['SPPEEC_GPU_BUDGET_GB'] = repr(budget/1e9)
    if ndev >= 2:
        core2 = GPUGeoCore(fac.mg, fac.cycles)
        check('auto split engages on multi-GPU box',
              core2.placement.startswith('split('), core2.placement)
        fac2 = _GeoMGFactor(YT, normal[:nplaq], base[:nplaq], nplaq)
        y2 = fac2(b)
        rel = np.linalg.norm(y2 - y_gpu)/np.linalg.norm(y_gpu)
        check('auto-split apply == single apply', rel < 1e-6,
              'rel %.2e' % rel)
    else:
        try:
            GPUGeoCore(fac.mg, fac.cycles)
            check('over-budget single-GPU raises', False, 'no raise')
        except RuntimeError as exc:
            check('over-budget single-GPU raises', 'split' in str(exc),
                  str(exc)[:60])
        fac2 = _GeoMGFactor(YT, normal[:nplaq], base[:nplaq], nplaq)
        check('factor records the CPU fallback',
              fac2.gpu_state.startswith('cpu fallback ('),
              fac2.gpu_state)
        y2 = fac2(b)
        rel = np.linalg.norm(y2 - y_cpu)/np.linalg.norm(y_cpu)
        check('fallback apply is the CPU apply', rel == 0.0,
              'rel %.2e' % rel)
    del os.environ['SPPEEC_GPU_BUDGET_GB']

    print('\n%d checks failed' % len(FAIL))
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
