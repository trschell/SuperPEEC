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


def _sub_flat(a, b, out, dtype):
    """``out = a - b`` on flat device vectors."""
    k = ocl_core.kernel(ocl_sparse.program(dtype, ocl_sparse.CSR.WG),
                        'sub_contig')
    k(ocl_core.queue(), (int(out.size),), None, a.data, b.data, out.data,
      np.uint32(out.size))


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
    # distinct columns a Gram row may have; the plaquette Gram is a
    # 36-slot stencil, so this is loose
    GRAM_MAXC = int(os.environ.get('SPPEEC_OCL_GRAM_MAXC', '128'))

    def __init__(self, mg, cycles, basis=None):
        if mg.coarse_pinv is None:
            raise RuntimeError("GeoMG coarse level too big for the dense "
                               "pinv -- host apply only")
        if any(L is None for L in mg.levels[1:]):
            raise RuntimeError("the OpenCL GeoMG needs the coarse level "
                               "hierarchy on the host")
        self.dtype = np.dtype(mg.dtype)
        dt = self.dtype
        levels = list(mg.levels)
        A0 = None
        sten = getattr(mg, '_sten0', None)
        wdi = getattr(mg, '_wdi0_t', None)
        if levels[0] is None and sten is not None and wdi is not None:
            # The stencil path never forms level 0 as a matrix; that is
            # the memory win the compression campaign bought. Apply it
            # as a stencil here too, which is both cheaper and exact:
            # the slot order matches the Fortran kernel's, so the two
            # agree bit for bit.
            import ocl_stencil
            A0 = ocl_stencil.Stencil0(sten, wdi, dt)
        elif levels[0] is None:
            # No certified stencil: form the Gram on the host and
            # upload it. That is the build transient the campaign
            # removed, so it is bounded rather than silently paid.
            if basis is None:
                raise RuntimeError(
                    "the OpenCL GeoMG needs a host level 0, a certified "
                    "stencil, or the basis level 0 is formed from")
            # formed on the card and kept there: a Gram is gigabytes
            # at flagship scale and has no business on the host
            try:
                A0 = ocl_sparse.CSR(
                    ocl_sparse.gram_device(basis, dt, self.GRAM_MAXC), dt)
            except ocl_sparse.SpGEMMOverflow as exc:
                itm = np.dtype(dt).itemsize
                est = float(getattr(mg, '_nnz0_est', 0) or 0)*(itm + 4)
                if est > self.GRAM_BUDGET:
                    raise MemoryError(
                        "the level-0 Gram needs more than %d columns in a "
                        "row (%s) and would take about %.1f GB on the host "
                        "(budget %.1f GB, SPPEEC_OCL_GRAM_GB)"
                        % (self.GRAM_MAXC, exc, est/2**30,
                           self.GRAM_BUDGET/2**30))
                levels[0] = (basis @ basis.T).tocsr().astype(dt)
        self.level0 = ('stencil' if isinstance(A0, object)
                       and type(A0).__module__ == 'ocl_stencil'
                       else ('device Gram' if A0 is not None else 'matrix'))
        self.cycles = int(cycles)
        self.nu = int(mg.nu)
        self.omega = float(mg.omega)
        self.A = [A0 if (i == 0 and A0 is not None)
                  else ocl_sparse.CSR(L, dt)
                  for i, L in enumerate(levels)]
        sweeps0 = hasattr(self.A[0], 'sweeps')
        # An aggregation prolongator is 0/1 with exactly one entry per
        # row, so its values and its row pointers carry nothing: it is
        # one column index per row, and the product is a gather. Its
        # transpose keeps the pointers but still needs no values. At R5
        # that is 689 MB of the hierarchy. The general form stays as
        # the fallback for an aggregation that is not of that shape.
        # Level 0 in tiles (2026-09-25): when level 0 is the stencil the
        # V-cycle keeps x, its Jacobi partner and the residual in the
        # stencil's two tile grids and never forms a flat level-0
        # vector of its own. The restriction then reads the residual
        # from the tiles, through the flat map folded into its index
        # array once here; the prolongation adds into the tiles.
        self.tiled0 = hasattr(A0, 't_sweeps')
        self.P, self.R = [], []
        self.aggregation0 = 'explicit'
        for i, Pm in enumerate(mg.Ps):
            remap = (np.asarray(sten.flat) if (i == 0 and self.tiled0)
                     else None)
            if remap is not None and getattr(mg, 'divs', None) \
                    and os.environ.get('SPPEEC_OCL_IMPLICIT_P0', '1') != '0':
                # the aggregation from tile geometry: tables instead of
                # a per-plaquette column array and a fine index list,
                # when the host verifies it computes the same thing
                try:
                    import ocl_stencil
                    agg = ocl_stencil.ImplicitAggregation(
                        Pm, sten.flat, mg.divs[0], A0.TL, A0.nt)
                    self.P.append(agg)
                    self.R.append(agg)
                    self.aggregation0 = 'implicit'
                    continue
                except ValueError as exc:
                    self.aggregation0 = 'explicit (%s)' % exc
            try:
                self.P.append(ocl_sparse.OnesProlong(Pm, dt))
                self.R.append(ocl_sparse.OnesRestrict(Pm, dt, remap=remap))
            except ValueError:
                if remap is not None:
                    raise RuntimeError("the tiled level 0 needs a "
                                       "one-per-row aggregation")
                self.P.append(ocl_sparse.CSR(Pm, dt))
                self.R.append(ocl_sparse.CSR(Pm.T.tocsr(), dt))
        # level 0's inverse diagonal is read only by the generic
        # smoother, which never runs when level 0 sweeps itself
        self.dinv = [None if (i == 0 and sweeps0)
                     else ocl_core.to_device(np.asarray(d, dt))
                     for i, d in enumerate(mg.dinv)]
        pinv = np.ascontiguousarray(mg.coarse_pinv, dtype=dt)
        self.pinv = ocl_core.to_device(pinv)
        self.pinv_shape = pinv.shape
        self.sizes = [int(A.shape[0]) for A in self.A]
        # per-level workspace: solution, right-hand side, residual and
        # the Jacobi ping-pong partner
        # Level 0's right-hand side is the caller's vector, never one
        # of ours, and its Jacobi partner is unused when level 0 runs
        # its own sweeps (the stencil does). Both were allocated and
        # never read: 198 MB each at R5.
        # with level 0 in tiles, its flat solution and residual (198
        # MB each at R5) do not exist either
        self._x = [None if (i == 0 and self.tiled0)
                   else ocl_core.zeros((n,), dt)
                   for i, n in enumerate(self.sizes)]
        self._b = [None] + [ocl_core.zeros((n,), dt)
                            for n in self.sizes[1:]]
        self._r = [None if (i == 0 and self.tiled0)
                   else ocl_core.zeros((n,), dt)
                   for i, n in enumerate(self.sizes)]
        self._t = [None if (i == 0 and sweeps0)
                   else ocl_core.zeros((n,), dt)
                   for i, n in enumerate(self.sizes)]

    def parts(self):
        """Per-matrix device bytes, with what the host held."""
        rows = []
        for tag, lst in (('A', self.A), ('P', self.P), ('R', self.R)):
            for i, M in enumerate(lst):
                if not hasattr(M, 'device_bytes') \
                        or getattr(M, 'nnz', None) is None:
                    continue          # level 0 may be a stencil, not a matrix
                rows.append((('%s%d' % (tag, i)), M.device_bytes(), M.nnz,
                             getattr(M, 'src_dtype', '?'),
                             bool(getattr(M, 'ones_only', False)),
                             bool(getattr(M, 'int8_ok', False))))
        rows.append(('dinv', sum(d.nbytes for d in self.dinv
                                 if d is not None), 0, '-', False, False))
        rows.append(('pinv', self.pinv.nbytes, 0, '-', False, False))
        return rows

    def device_bytes(self):
        """Resident device bytes, split into operator and workspace."""
        op = sum(A.device_bytes() for A in self.A)
        seen = set()
        for M in list(self.P) + list(self.R):
            if id(M) not in seen:          # the implicit level-0 aggregation is one object for both
                seen.add(id(M))
                op += M.device_bytes()
        op += sum(d.nbytes for d in self.dinv
                  if d is not None) + self.pinv.nbytes
        ws = sum(v.nbytes for lst in (self._x, self._b, self._r, self._t)
                 for v in lst if v is not None)
        return dict(operator=int(op), workspace=int(ws))

    def _smooth(self, lv, x, b):
        """``nu`` damped-Jacobi sweeps, leaving the result in ``x``."""
        A = self.A[lv]
        if hasattr(A, 'sweeps'):
            # the stencil packs once and sweeps on tiles, so it runs
            # the whole set of sweeps in a single call
            A.sweeps(x, b, self.nu)
            return
        cur, alt = x, self._t[lv]
        for _ in range(self.nu):
            self.A[lv].jacobi(cur, b, self.dinv[lv], alt, self.omega)
            cur, alt = alt, cur
        if cur is not x:
            _copy(x, cur)

    def _vcycle0(self, b):
        """One V-cycle with level 0 in the stencil's tiles; ``b`` flat."""
        A0 = self.A[0]
        A0.t_sweeps(b, self.nu)
        rt = A0.t_residual(b)
        A0.t_restrict(self.R[0], rt, self._b[1])
        self._x[1].fill(self.dtype.type(0), queue=ocl_core.queue())
        self._vcycle(1, self._b[1], self._x[1])
        A0.t_prolong_add(self.P[0], self._x[1])
        A0.t_sweeps(b, self.nu)

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

    def _cycles(self, r):
        if self.tiled0:
            self.A[0].t_zero()
            for _ in range(self.cycles):
                self._vcycle0(r)
            return None
        x = self._x[0]
        x.fill(self.dtype.type(0), queue=ocl_core.queue())
        for _ in range(self.cycles):
            self._vcycle(0, r, x)
        return x

    def solve(self, r, out):
        """``cycles`` V-cycles from a zero start, on a device vector;
        the result is written to ``out`` (which may alias ``r``: the
        answer is unpacked after the last cycle)."""
        x = self._cycles(r)
        if self.tiled0:
            return self.A[0].t_unpack(out)
        if out is not x:
            _copy(out, x)
        return out

    def solve_sub(self, r, yp, out):
        """``out = yp - M r``: the solve and the subtraction in one,
        read straight from the tiles when level 0 lives there. ``out``
        may alias ``r``."""
        x = self._cycles(r)
        if self.tiled0:
            return self.A[0].t_unpack_sub(yp, out)
        _sub_flat(yp, x, out, self.dtype)
        return out


class GeoBlock(object):
    """The local block plus the exact macro Schur, as a host callable."""

    def __init__(self, factor):
        self.core = GeoCore(factor.mg, factor.cycles,
                            basis=getattr(factor, '_basis', None))
        dt = self.core.dtype
        self.dtype = dt
        self.n = int(factor.n)
        self.nmac = int(factor.nmac)
        loc_h = np.asarray(factor.loc)
        nloc = int(loc_h.size)
        # The production split is the historical contiguous one: the
        # local set is the leading range of the vector. Then the local
        # right-hand side IS a view of the input, the local index array
        # (198 MB at R5) need not exist on the card, and the gather and
        # scatter through it are a view and a copy. A caller-supplied
        # macro set keeps the indexed path.
        self.contig = bool(nloc and loc_h[0] == 0
                           and loc_h[-1] == nloc - 1
                           and bool(np.all(np.diff(loc_h) == 1)))
        self.loc = (None if self.contig
                    else ocl_core.to_device(loc_h.astype(np.int32)))
        self.mac = (ocl_core.to_device(np.asarray(factor.mac, np.int32))
                    if self.nmac else None)
        # The identity set is never indexed on the device any more:
        # the apply leaves those entries where they already are. It is
        # kept only to check the partition below, on the host.
        rest = getattr(factor, 'rest', None)
        self.nrest = int(np.size(rest)) if rest is not None else 0
        # the coupling block by its nonempty rows: six columns do not
        # need a row pointer over 49.6 M rows
        self.B = ocl_sparse.CSRRows(factor.B, dt) if self.nmac else None
        self.BT = (ocl_sparse.CSR(factor.B.T.tocsr(), dt)
                   if self.nmac else None)
        self._lu_solve = factor._lu_solve
        self.S = factor.S
        self.MB = getattr(factor, '_MB', None)
        if self.core.sizes[0] != nloc:
            raise RuntimeError(
                "local block size %d does not match the hierarchy's level 0 "
                "(%d)" % (nloc, self.core.sizes[0]))
        self._k_sub = ocl_core.kernel(
            ocl_sparse.program(dt, ocl_sparse.CSR.WG), 'scatter_sub')
        self._k_sub_c = ocl_core.kernel(
            ocl_sparse.program(dt, ocl_sparse.CSR.WG), 'sub_contig')
        # The apply writes its output over its input, which is only
        # sound if the three sets cover the vector: the identity set
        # is then the positions nothing else writes, and leaving the
        # input there is exactly the pass-through. The factor builds
        # `rest` as the complement of loc and mac, so this holds by
        # construction -- but it is the whole safety argument for the
        # aliasing below, so it is checked rather than assumed.
        cov = np.zeros(self.n, dtype=bool)
        cov[np.asarray(factor.loc)] = True
        if self.nmac:
            cov[np.asarray(factor.mac)] = True
        if rest is not None and np.size(rest):
            cov[np.asarray(rest)] = True
        if not bool(cov.all()):
            raise RuntimeError(
                "the local, macro and identity sets leave %d of %d entries "
                "unwritten; the in-place apply needs them to partition the "
                "vector" % (int((~cov).sum()), self.n))
        del cov
        self._bg = ocl_core.zeros((self.n,), dt)
        self._rp = (self._bg[:nloc] if self.contig
                    else ocl_core.zeros((nloc,), dt))
        self._yp = ocl_core.zeros((nloc,), dt)
        self._sub = None
        self._rm = ocl_core.zeros((self.nmac,), dt) if self.nmac else None
        self._bt = ocl_core.zeros((self.nmac,), dt) if self.nmac else None
        self._ym = ocl_core.zeros((self.nmac,), dt) if self.nmac else None

    def solve_local(self, rp):
        """The local block solve for a host vector."""
        d = ocl_core.to_device(np.asarray(rp, self.dtype))
        return self.core.solve(d, d).get(queue=ocl_core.queue())

    def __call__(self, b):
        dt = self.dtype
        q = ocl_core.queue()
        # The output is the input buffer. The local and macro sets are
        # overwritten below and the identity set keeps the value it
        # was handed, which is what the explicit pass-through used to
        # copy; the zero fill goes with it, since every position is
        # now either written or deliberately kept.
        out = self._bg
        out.set(np.ascontiguousarray(np.asarray(b, dt)), queue=q)
        if not self.contig:
            ocl_sparse.gather(out, self.loc, self._rp, dt)
        nloc = int(self._yp.size)
        if self.nmac:
            yp = self.core.solve(self._rp, self._yp)
            ocl_sparse.gather(out, self.mac, self._rm, dt)
            self.BT.spmv(yp, self._bt)
            ym_cpu = self._lu_solve(
                self.S, np.float64(self._rm.get(queue=q)
                                   - self._bt.get(queue=q)))
            if self.MB is not None:
                # the kept macro columns: a dense host product instead
                # of a second V-cycle, as on the CUDA path. One flat
                # buffer for it, kept only where this branch exists
                # (the small models whose MB fits the budget)
                if self._sub is None:
                    self._sub = ocl_core.zeros((nloc,), dt)
                self._sub.set(np.ascontiguousarray(
                    (self.MB @ ym_cpu.astype(np.float32)).astype(dt)),
                    queue=q)
                if self.contig:
                    self._k_sub_c(q, (nloc,), None, yp.data,
                                  self._sub.data, out.data,
                                  np.uint32(nloc))
                else:
                    self._k_sub(q, (nloc,), None, yp.data, self._sub.data,
                                self.loc.data, out.data, np.uint32(nloc))
            else:
                self._ym.set(np.ascontiguousarray(ym_cpu.astype(dt)),
                             queue=q)
                # rp was consumed by the first solve: it takes the
                # correction's right-hand side, and the combined result
                # yp - M(B ym) is written over it -- the local range of
                # `out` itself when the split is contiguous -- straight
                # from the tiles, with no flat solution buffer
                self.B.spmv(self._ym, self._rp)
                self.core.solve_sub(self._rp, yp, self._rp)
                if not self.contig:
                    ocl_sparse.scatter(self._rp, self.loc, out, dt)
            self._ym.set(np.ascontiguousarray(ym_cpu.astype(dt)), queue=q)
            ocl_sparse.scatter(self._ym, self.mac, out, dt)
        else:
            # no macro set: the answer goes straight to the local range
            self.core.solve(self._rp, self._rp)
            if not self.contig:
                ocl_sparse.scatter(self._rp, self.loc, out, dt)
        return np.float32(out.get(queue=q))
