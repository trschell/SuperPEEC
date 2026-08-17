# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""GPU apply for the block-AMG preconditioner (``SPPEEC_GPU=1``).

RESIDENT design, the lesson of both the abandoned pycuda attempt and
the working GPU m2l: every level's ``A``/``P``/``R``, the coarse dense
inverse, and the macro coupling ``B`` upload ONCE at construction; per
apply only the right-hand side and the result cross PCIe (float32
vectors, ~7 MB each way on square_coil).

THE SMOOTHER IS DELIBERATELY NOT THE CPU ONE. pyamg's default
symmetric Gauss-Seidel is sequential by construction; this cycle uses
DAMPED JACOBI (omega = 2/3, one pre- and one post-sweep per level),
which is nothing but SpMV + axpy -- cuSPARSE's home turf. The GPU
preconditioner therefore DIFFERS from the CPU one: Krylov iteration
counts may move a little, the solved answer cannot (a preconditioner
only shapes the search path). Judge it by end-to-end solve wall clock
and matvec count against the CPU apply, never by apply-level equality.

The block structure mirrors ``port_impedance._BlockAMGFactor``: AMG on
the local block, exact (CPU, tiny) LU Schur on the macro cycles. The
macro pieces stay on the CPU on purpose -- ``n_macro`` is tens at most,
and shipping a tens-length vector is cheaper than porting an LU.
"""

import numpy as np


class GPUAMG:
    """Device-resident V-cycle engine for a built pyamg hierarchy.

    Constructed FROM the CPU factor's ``ml``, so pyamg's setup
    (aggregation, strength, hierarchy) is reused verbatim -- only the
    per-apply cycle runs on the device. Raises ImportError/CUDA errors
    at construction; the caller decides fatal vs fallback.
    """

    def __init__(self, ml, cycles, sweeps=None):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        self.cp = cp
        self.cycles = int(cycles)
        # Jacobi sweeps per pre/post smooth. One sweep converges ~2x
        # slower (Krylov iterations) than the CPU's Gauss-Seidel; the
        # GPU apply is cheap enough that extra sweeps are nearly free,
        # so the default is 2 (measured below in the module history).
        import os
        self.sweeps = int(os.environ.get('SPPEEC_GPU_AMG_SWEEPS', '2')) \
            if sweeps is None else int(sweeps)
        self.levels = []
        mllevels = ml.levels
        # dtype-preserving (2026-08-08, extended 2026-08-13): the
        # diagschur S_d hierarchy is COMPLEX symmetric -> complex128;
        # everything real keeps the CPU hierarchy's OWN dtype -- fp32
        # under the phase-1 precision policy, which halves device
        # residency AND runs SpMV at the consumer card's fp32 rate.
        dt = (np.complex128
              if np.issubdtype(mllevels[0].A.dtype, np.complexfloating)
              else mllevels[0].A.dtype)
        self.dtype = dt
        for i, lv in enumerate(mllevels):
            A = lv.A.tocsr().astype(dt)
            entry = {'A': csp.csr_matrix(A)}
            if i < len(mllevels) - 1:
                d = A.diagonal()
                entry['dinv'] = cp.asarray(((2.0/3.0)/d).astype(dt))
                entry['P'] = csp.csr_matrix(lv.P.tocsr().astype(dt))
                entry['R'] = csp.csr_matrix(lv.R.tocsr().astype(dt))
            else:
                # coarse level: dense inverse, computed once on CPU in
                # full precision (max_coarse defaults to 400 --
                # trivial), STORED in the hierarchy dtype, resident on
                # the device so the recursion never leaves it
                hi = np.complex128 if np.issubdtype(dt, np.complexfloating) \
                    else np.float64
                entry['Ainv'] = cp.asarray(
                    np.linalg.inv(A.toarray().astype(hi)).astype(dt))
            self.levels.append(entry)

    def _vcycle(self, li, b):
        lev = self.levels[li]
        if 'Ainv' in lev:
            return lev['Ainv'] @ b
        A, dinv = lev['A'], lev['dinv']
        x = dinv*b                           # damped-Jacobi presmooth, x0=0
        for _ in range(self.sweeps - 1):
            x = x + dinv*(b - A @ x)
        r = b - A @ x
        x = x + lev['P'] @ self._vcycle(li + 1, lev['R'] @ r)
        for _ in range(self.sweeps):         # postsmooth
            x = x + dinv*(b - A @ x)
        return x

    def solve(self, r):
        """``cycles`` iterations of x <- x + V(r - A x), on device."""
        cp = self.cp
        x = cp.zeros_like(r)
        A0 = self.levels[0]['A']
        for _ in range(self.cycles):
            x = x + self._vcycle(0, r - A0 @ x)
        return x


class GPUPlainAMG:
    """GPU apply for ``_AMGFactor`` (whole-system hierarchy, no blocks)."""

    def __init__(self, factor):
        import cupy as cp
        self.cp = cp
        self.core = GPUAMG(factor.ml, factor.cycles)

    def __call__(self, b):
        cp = self.cp
        x = self.core.solve(cp.asarray(np.asarray(b, self.core.dtype)))
        return np.float32(cp.asnumpy(x))


class GPUBlockAMG:
    """GPU apply for ``_BlockAMGFactor`` (AMG local block + exact macro).

    The macro pieces stay on the CPU on purpose -- ``n_macro`` is tens
    at most, and shipping a tens-length vector is cheaper than porting
    an LU.
    """

    def __init__(self, factor):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        self.cp = cp
        self.core = GPUAMG(factor.ml, factor.cycles)
        self.n = factor.n
        self.nmac = factor.nmac
        self.loc = cp.asarray(factor.loc)
        self.mac = cp.asarray(factor.mac) if factor.nmac else None
        self._lu_solve = factor._lu_solve
        self.S = factor.S
        self.B = csp.csr_matrix(
            factor.B.astype(self.core.dtype)) if factor.nmac else None

    def __call__(self, b):
        cp = self.cp
        dt = self.core.dtype
        bg = cp.asarray(np.asarray(b, dt))
        rp = bg[self.loc]
        yp = self.core.solve(rp)
        out = cp.empty(self.n, dtype=dt)
        if self.nmac:
            rm = bg[self.mac]
            ym_cpu = self._lu_solve(self.S, np.float64(
                cp.asnumpy(rm - self.B.T @ yp)))
            ym = cp.asarray(ym_cpu.astype(dt))
            out[self.loc] = yp - self.core.solve(self.B @ ym)
            out[self.mac] = ym
        else:
            out[self.loc] = yp
        return np.float32(cp.asnumpy(out))

class GPUGeoCore:
    """Device-resident V-cycle for a built :class:`loopmg.GeoMG`
    hierarchy -- the geometric sibling of :class:`GPUAMG`. Setup is
    reused verbatim (that setup being nearly free is GeoMG's whole
    case); only the apply moves to the device. Requires the dense
    coarse pseudo-inverse (max_coarse <= 2000); raises otherwise and
    the caller falls back to the CPU apply.

    SPLIT RESIDENCY (2026-08-15): when the hierarchy exceeds one
    device's budget, level 0 (the finest -- ~70% of the bytes) can
    live on one device and every coarser level on a second. This is
    VRAM POOLING, not parallelism: a V-cycle is sequential through
    levels, so the devices never compute concurrently and the only
    inter-device traffic is the two COARSE-sized vectors at the
    l0/l1 boundary (restricted residual down, correction up) --
    ~100 MB/apply at hero scale, ~10 ms. P[lv]/R[lv] live on the
    FINE side of their boundary so only coarse vectors cross.

    Placement, in order of precedence: ``devices=(fine, coarse)``
    argument; ``SPPEEC_GPU_SPLIT=fine,coarse`` env (same-device pairs
    like ``0,0`` are valid and exercise the split plumbing on one
    card); automatic -- single-device if the byte estimate fits the
    budget (85% of free VRAM, or ``SPPEEC_GPU_BUDGET_GB`` per device
    when set -- the synthetic-budget test knob), else the first
    two-device split that fits, else raise (caller falls back to the
    CPU apply). ``self.placement`` records the decision."""

    def __init__(self, mg, cycles, devices=None):
        import os
        import cupy as cp
        import cupyx.scipy.sparse as csp
        if mg.coarse_pinv is None:
            raise RuntimeError("GeoMG coarse level too big for the "
                               "dense pinv -- CPU apply only")
        self.cp = cp
        self.cycles = int(cycles)
        self.nu = int(mg.nu)
        self.omega = float(mg.omega)
        self.dtype = mg.levels[0].dtype       # fp32 under phase 1
        nlev = len(mg.levels)
        itm = np.dtype(self.dtype).itemsize

        def _csr_bytes(M):
            M = M.tocsr()
            return M.nnz*itm + M.indices.nbytes + M.indptr.nbytes

        # byte estimate per level, P/R accounted on their FINE side
        lev_bytes = [_csr_bytes(L) for L in mg.levels]
        for i, Pm in enumerate(mg.Ps):
            lev_bytes[i] += 2*_csr_bytes(Pm)
        for i, d in enumerate(mg.dinv):
            lev_bytes[min(i, nlev - 1)] += d.size*itm
        lev_bytes[-1] += mg.coarse_pinv.size*itm
        total = sum(lev_bytes)

        env = os.environ.get('SPPEEC_GPU_SPLIT')
        if devices is None and env:
            devices = tuple(int(v) for v in env.split(','))
        budget_env = os.environ.get('SPPEEC_GPU_BUDGET_GB')

        def _budget(dev):
            if budget_env:
                return float(budget_env)*1e9
            with cp.cuda.Device(dev):
                free, _ = cp.cuda.runtime.memGetInfo()
            return 0.85*free

        cur = cp.cuda.runtime.getDevice()
        # self.split: (fine_dev, coarse_dev) when the two halves are
        # assigned separately -- INCLUDING a same-device pair, which is
        # the single-card plumbing test: the boundary-copy code runs
        # (asarray aliases on the same device, so it costs nothing)
        # even though no bytes cross PCIe.
        if devices is not None:
            if len(devices) != 2:
                raise ValueError("devices must be a (fine, coarse) "
                                 "pair, got %r" % (devices,))
            self.split = (int(devices[0]), int(devices[1]))
        elif total <= _budget(cur):
            self.split = None
        else:
            picked = None
            for d2 in range(cp.cuda.runtime.getDeviceCount()):
                if d2 == cur:
                    continue
                if (lev_bytes[0] <= _budget(cur)
                        and sum(lev_bytes[1:]) <= _budget(d2)):
                    picked = d2
                    break
            if picked is None:
                raise RuntimeError(
                    "GeoMG hierarchy %.3g GB exceeds the %.3g GB "
                    "device budget and no two-device split fits"
                    % (total/1e9, _budget(cur)/1e9))
            self.split = (cur, picked)
        if self.split is None:
            self.devs = [cur]*nlev
            self.placement = 'single(dev%d, %.3g GB)' % (cur, total/1e9)
        else:
            self.devs = [self.split[0]] + [self.split[1]]*(nlev - 1)
            self.placement = ('split(l0->dev%d %.3g GB, '
                              'l1+->dev%d %.3g GB)'
                              % (self.split[0], lev_bytes[0]/1e9,
                                 self.split[1],
                                 sum(lev_bytes[1:])/1e9))

        def _on(dev, up):
            with cp.cuda.Device(dev):
                return up()

        self.A = [_on(self.devs[i],
                      lambda L=L: csp.csr_matrix(L.astype(self.dtype)))
                  for i, L in enumerate(mg.levels)]
        self.P = [_on(self.devs[i],
                      lambda P=P: csp.csr_matrix(
                          P.tocsr().astype(self.dtype)))
                  for i, P in enumerate(mg.Ps)]
        self.R = [_on(self.devs[i],
                      lambda P=P: csp.csr_matrix(
                          P.T.tocsr().astype(self.dtype)))
                  for i, P in enumerate(mg.Ps)]
        self.dinv = [_on(self.devs[min(i, nlev - 1)],
                         lambda d=d: cp.asarray(d.astype(self.dtype)))
                     for i, d in enumerate(mg.dinv)]
        self.pinv = _on(self.devs[-1],
                        lambda: cp.asarray(
                            mg.coarse_pinv.astype(self.dtype)))

    def _smooth(self, lv, x, b):
        with self.cp.cuda.Device(self.devs[lv]):
            A, di = self.A[lv], self.dinv[lv]
            for _ in range(self.nu):
                x = x + self.omega*di*(b - A @ x)
            return x

    def _vcycle(self, lv, b, x):
        cp = self.cp
        with cp.cuda.Device(self.devs[lv]):
            if lv == len(self.A) - 1:
                return self.pinv @ b
            x = self._smooth(lv, x, b)
            r = b - self.A[lv] @ x
            rc = self.R[lv] @ r               # coarse-sized, fine dev
        nxt = self.devs[lv + 1]
        # under a split, asarray moves the coarse vector across the
        # boundary; same-device pairs alias for free, so the plumbing
        # runs (and is testable) on one card too
        with cp.cuda.Device(nxt):
            if self.split is not None:
                rc = cp.asarray(rc)           # the down transfer
            z = cp.zeros(self.P[lv].shape[1], dtype=self.dtype)
        xc = self._vcycle(lv + 1, rc, z)
        with cp.cuda.Device(self.devs[lv]):
            if self.split is not None:
                xc = cp.asarray(xc)           # the up transfer
            x = x + self.P[lv] @ xc
            return self._smooth(lv, x, b)

    def solve(self, r):
        cp = self.cp
        with cp.cuda.Device(self.devs[0]):
            x = cp.zeros_like(r)
        for _ in range(self.cycles):
            x = self._vcycle(0, r, x)
        return x


class GPUGeoBlock:
    """GPU apply for ``_GeoMGFactor`` (geometric MG local block +
    exact macro Schur) -- the mirror of :class:`GPUBlockAMG`."""

    def __init__(self, factor, devices=None):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        self.cp = cp
        self.core = GPUGeoCore(factor.mg, factor.cycles,
                               devices=devices)
        self.n = factor.n
        self.nmac = factor.nmac
        # block-level arrays live on the ENTRY device (level 0's)
        with cp.cuda.Device(self.core.devs[0]):
            self.loc = cp.asarray(factor.loc)
            self.mac = cp.asarray(factor.mac) if factor.nmac else None
            self.B = csp.csr_matrix(
                factor.B.astype(self.core.dtype)) if factor.nmac \
                else None
        self._lu_solve = factor._lu_solve
        self.S = factor.S

    def __call__(self, b):
        cp = self.cp
        dt = self.core.dtype
        with cp.cuda.Device(self.core.devs[0]):
            bg = cp.asarray(np.asarray(b, dt))
            rp = bg[self.loc]
            yp = self.core.solve(rp)
            out = cp.empty(self.n, dtype=dt)
            if self.nmac:
                rm = bg[self.mac]
                ym_cpu = self._lu_solve(self.S, np.float64(
                    cp.asnumpy(rm - self.B.T @ yp)))
                ym = cp.asarray(ym_cpu.astype(dt))
                out[self.loc] = yp - self.core.solve(self.B @ ym)
                out[self.mac] = ym
            else:
                out[self.loc] = yp
            return np.float32(cp.asnumpy(out))

