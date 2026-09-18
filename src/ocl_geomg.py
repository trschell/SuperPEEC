# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The geometric-multigrid preconditioner apply on OpenCL.

Mirrors :class:`gpu_amg.GPUGeoCore` and :class:`gpu_amg.GPUGeoBlock`
onto the deterministic kernels in :mod:`ocl_sparse`. The V-cycle is the
same algorithm in the same order: ``nu`` damped-Jacobi sweeps, the
residual, restriction, a recursive coarse solve ending in the dense
pseudo-inverse, prolongation, and ``nu`` sweeps again.

Two things are different, both deliberate.

*The products are reproducible by construction.* Every row is reduced
by a fixed number of work items in a fixed binary tree, so the apply is
the same map on every call whatever the scheduler does. The CUDA path
has to work around cuSPARSE's atomics twice over to get the same
property, once by materialising the macro transpose and once by running
its rows through a Python loop. Here the macro transpose is an ordinary
product and runs at full width.

*The vectors are real.* The preconditioner is applied to a real
float32 vector and returns one, so none of this needs complex support.

What stays on the host, as in the CUDA path: the macro Schur factor and
its solve, which is small and dense, and the kept macro columns.

Single device only. The CUDA path can split a hierarchy across two
cards through implicit peer-to-peer copies; OpenCL has no equivalent
and the split is deferred.
"""
import os

import numpy as np

import ocl_core
import ocl_sparse


class _Levels(object):
    """``mg`` with a level list substituted, leaving the original alone."""

    def __init__(self, mg, levels):
        self._mg = mg
        self.levels = levels

    def __getattr__(self, k):
        return getattr(self._mg, k)


def _copy(dst, src, nbytes=None):
    """Device-to-device copy of a whole array."""
    import pyopencl as cl
    cl.enqueue_copy(ocl_core.queue(), dst.data, src.data,
                    byte_count=int(nbytes if nbytes is not None
                                   else src.nbytes))


class GeoCore(object):
    """The multigrid hierarchy, resident, with a deterministic V-cycle."""

    # level-0 Gram budget, bytes: above this the host build is the
    # very transient the memory campaign removed, so refuse and let
    # the caller fall back rather than trade one regression for another
    GRAM_BUDGET = float(os.environ.get('SPPEEC_OCL_GRAM_GB', '2'))*2**30

    def __init__(self, mg, cycles, basis=None):
        if mg.coarse_pinv is None:
            raise RuntimeError("GeoMG coarse level too big for the dense "
                               "pinv -- host apply only")
        if any(L is None for L in mg.levels[1:]):
            raise RuntimeError("the OpenCL GeoMG needs the coarse level "
                               "hierarchy on the host")
        self.dtype = np.dtype(mg.dtype)
        levels = list(mg.levels)
        if levels[0] is None:
            # The stencil path never forms level 0 as a matrix; that is
            # the memory win. The CUDA path builds the Gram on the card
            # instead, which needs a device sparse-sparse product this
            # backend does not have yet, so it is formed on the host
            # here and uploaded. That is a build transient, so it is
            # bounded: porting the level-0 stencil apply is the proper
            # fix and removes this branch.
            if basis is None:
                raise RuntimeError(
                    "the OpenCL GeoMG needs either a host level 0 or the "
                    "basis it is formed from")
            itm = np.dtype(self.dtype).itemsize
            est = float(getattr(mg, '_nnz0_est', 0) or 0)*(itm + 4)
            if est > self.GRAM_BUDGET:
                raise MemoryError(
                    "the level-0 Gram would take about %.1f GB on the host "
                    "(budget %.1f GB, SPPEEC_OCL_GRAM_GB); the device "
                    "sparse product is not ported yet"
                    % (est/2**30, self.GRAM_BUDGET/2**30))
            levels[0] = (basis @ basis.T).tocsr().astype(self.dtype)
        mg = _Levels(mg, levels)
        self.cycles = int(cycles)
        self.nu = int(mg.nu)
        self.omega = float(mg.omega)
        dt = self.dtype
        self.A = [ocl_sparse.CSR(L, dt) for L in mg.levels]
        self.P = [ocl_sparse.CSR(P, dt) for P in mg.Ps]
        self.R = [ocl_sparse.CSR(P.T.tocsr(), dt) for P in mg.Ps]
        self.dinv = [ocl_core.to_device(np.asarray(d, dt)) for d in mg.dinv]
        pinv = np.ascontiguousarray(mg.coarse_pinv, dtype=dt)
        self.pinv = ocl_core.to_device(pinv)
        self.pinv_shape = pinv.shape
        self.sizes = [int(A.shape[0]) for A in self.A]
        # per-level workspace: solution, right-hand side, residual and
        # the Jacobi ping-pong partner
        self._x = [ocl_core.zeros((n,), dt) for n in self.sizes]
        self._b = [ocl_core.zeros((n,), dt) for n in self.sizes]
        self._r = [ocl_core.zeros((n,), dt) for n in self.sizes]
        self._t = [ocl_core.zeros((n,), dt) for n in self.sizes]

    def _smooth(self, lv, x, b):
        """``nu`` damped-Jacobi sweeps, leaving the result in ``x``."""
        cur, alt = x, self._t[lv]
        for _ in range(self.nu):
            self.A[lv].jacobi(cur, b, self.dinv[lv], alt, self.omega)
            cur, alt = alt, cur
        if cur is not x:
            _copy(x, cur)

    def _vcycle(self, lv, b, x):
        """One V-cycle at level ``lv``; ``x`` is updated in place."""
        if lv == len(self.A) - 1:
            m, n = self.pinv_shape
            ocl_sparse.dense_gemv(self.pinv, b, x, m, n, self.dtype,
                                  ocl_sparse.CSR.WG)
            return
        self._smooth(lv, x, b)
        self.A[lv].residual(x, b, self._r[lv])
        self.R[lv].spmv(self._r[lv], self._b[lv + 1])
        self._x[lv + 1].fill(self.dtype.type(0), queue=ocl_core.queue())
        self._vcycle(lv + 1, self._b[lv + 1], self._x[lv + 1])
        self.P[lv].spmv_add(self._x[lv + 1], x)
        self._smooth(lv, x, b)

    def solve(self, r, out=None):
        """``cycles`` V-cycles from a zero start, on a device vector."""
        x = self._x[0]
        x.fill(self.dtype.type(0), queue=ocl_core.queue())
        for _ in range(self.cycles):
            self._vcycle(0, r, x)
        if out is not None:
            _copy(out, x)
            return out
        return x


class GeoBlock(object):
    """The local block plus the exact macro Schur, as a host callable."""

    def __init__(self, factor):
        self.core = GeoCore(factor.mg, factor.cycles,
                            basis=getattr(factor, '_basis', None))
        dt = self.core.dtype
        self.dtype = dt
        self.n = int(factor.n)
        self.nmac = int(factor.nmac)
        self.loc = ocl_core.to_device(np.asarray(factor.loc, np.int32))
        self.mac = (ocl_core.to_device(np.asarray(factor.mac, np.int32))
                    if self.nmac else None)
        rest = getattr(factor, 'rest', None)
        self.rest = (ocl_core.to_device(np.asarray(rest, np.int32))
                     if rest is not None and np.size(rest) else None)
        self.B = ocl_sparse.CSR(factor.B, dt) if self.nmac else None
        self.BT = (ocl_sparse.CSR(factor.B.T.tocsr(), dt)
                   if self.nmac else None)
        self._lu_solve = factor._lu_solve
        self.S = factor.S
        self.MB = getattr(factor, '_MB', None)
        nloc = int(np.size(factor.loc))
        if self.core.sizes[0] != nloc:
            raise RuntimeError(
                "local block size %d does not match the hierarchy's level 0 "
                "(%d)" % (nloc, self.core.sizes[0]))
        self._k_sub = ocl_core.kernel(
            ocl_sparse.program(dt, ocl_sparse.CSR.WG), 'scatter_sub')
        self._rest_tmp = (ocl_core.zeros((int(np.size(rest)),), dt)
                          if self.rest is not None else None)
        self._bg = ocl_core.zeros((self.n,), dt)
        self._out = ocl_core.zeros((self.n,), dt)
        self._rp = ocl_core.zeros((nloc,), dt)
        self._yp = ocl_core.zeros((nloc,), dt)
        self._sub = ocl_core.zeros((nloc,), dt)
        self._rm = ocl_core.zeros((self.nmac,), dt) if self.nmac else None
        self._bt = ocl_core.zeros((self.nmac,), dt) if self.nmac else None
        self._ym = ocl_core.zeros((self.nmac,), dt) if self.nmac else None

    def solve_local(self, rp):
        """The local block solve for a host vector."""
        d = ocl_core.to_device(np.asarray(rp, self.dtype))
        return self.core.solve(d).get(queue=ocl_core.queue())

    def __call__(self, b):
        dt = self.dtype
        q = ocl_core.queue()
        self._bg.set(np.ascontiguousarray(np.asarray(b, dt)), queue=q)
        ocl_sparse.gather(self._bg, self.loc, self._rp, dt)
        self.core.solve(self._rp, out=self._yp)
        out = self._out
        out.fill(dt.type(0), queue=q)
        if self.nmac:
            ocl_sparse.gather(self._bg, self.mac, self._rm, dt)
            self.BT.spmv(self._yp, self._bt)
            ym_cpu = self._lu_solve(
                self.S, np.float64(self._rm.get(queue=q)
                                   - self._bt.get(queue=q)))
            if self.MB is not None:
                # the kept macro columns: a dense host product instead
                # of a second V-cycle, as on the CUDA path
                self._sub.set(np.ascontiguousarray(
                    (self.MB @ ym_cpu.astype(np.float32)).astype(dt)),
                    queue=q)
            else:
                self._ym.set(np.ascontiguousarray(ym_cpu.astype(dt)),
                             queue=q)
                self.B.spmv(self._ym, self._rp)
                self.core.solve(self._rp, out=self._sub)
            nloc = int(self._yp.size)
            self._k_sub(q, (nloc,), None, self._yp.data, self._sub.data,
                        self.loc.data, out.data, np.uint32(nloc))
            self._ym.set(np.ascontiguousarray(ym_cpu.astype(dt)), queue=q)
            ocl_sparse.scatter(self._ym, self.mac, out, dt)
        else:
            ocl_sparse.scatter(self._yp, self.loc, out, dt)
        if self.rest is not None:
            # the identity set passes through unchanged
            ocl_sparse.gather(self._bg, self.rest, self._rest_tmp, dt)
            ocl_sparse.scatter(self._rest_tmp, self.rest, out, dt)
        return np.float32(out.get(queue=q))
