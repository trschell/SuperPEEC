# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Geometric multigrid on the LOOP (plaquette) basis of the LpR solve.

The loop space is the cycle space ``ker(A^T) = im(d_2)``, so a plaquette
basis IS the discrete curl of face variables on a uniform voxel lattice:
the coarsening that smoothed-aggregation AMG has to discover by
inspecting matrix entries is already known here -- it is the lattice,
halved. This module supplies the two pieces that exploit that, and
:class:`port_impedance._GeoMGFactor` wires them in behind the same
float32-callable interface a cholmod Factor presents.

See studies/geomg.py for the A/B harness and the measured comparison
against pyamg (including the correction that the STORED AMG hierarchy
is ~240 B/loop and flat -- not the terabyte-scale wall it was once
extrapolated to be).
"""
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Threaded CSR matvec (mp_fortran CSRMV_*, 2026-08-25): row-parallel,
# so threaded results are BIT-IDENTICAL to serial AND measured exactly
# equal to scipy's csr_matvec (same ordered accumulation; maxrel 0.0
# on a 14M-nnz probe) -- and this lives inside a PRECONDITIONER, where
# rounding can only move iteration counts, never converged answers.
# The v-cycle's scipy SpMV was the largest single-threaded line left
# in a CPU solve cycle (7.9 s of ~26 at R4). SPPEEC_SPMV=0 opts out;
# thread count follows OMP_NUM_THREADS.
_CSRMV, _CSRMV8, _JACOBI8 = {}, {}, {}
if os.environ.get('SPPEEC_SPMV') != '0':
    try:
        import mp_fortran as _mpf
        _CSRMV = {(np.float32, np.int32): _mpf.csrmv_s,
                  (np.float64, np.int32): _mpf.csrmv_d,
                  (np.float32, np.int64): _mpf.csrmv_sl,
                  (np.float64, np.int64): _mpf.csrmv_dl}
        # int8-data twins (tier 1, 2026-08-26): the loop Gram's
        # entries are exactly {4, +-1} and the aggregation
        # prolongators are 0/1, so int8 storage is LOSSLESS -- the
        # mixed int8*real product promotes to the same reals -- and
        # streams 4x fewer data bytes through a bandwidth-bound
        # kernel. JACOBI8 additionally fuses the damped-Jacobi
        # update, replacing four NROW-sized numpy temporaries per
        # sweep with one pass.
        _CSRMV8 = {(np.float32, np.int32): _mpf.csrmv8_s,
                   (np.float64, np.int32): _mpf.csrmv8_d,
                   (np.float32, np.int64): _mpf.csrmv8_sl,
                   (np.float64, np.int64): _mpf.csrmv8_dl}
        _JACOBI8 = {(np.float32, np.int32): _mpf.jacobi8_s,
                    (np.float64, np.int32): _mpf.jacobi8_d,
                    (np.float32, np.int64): _mpf.jacobi8_sl,
                    (np.float64, np.int64): _mpf.jacobi8_dl}
    except (ImportError, AttributeError):  # old .so: quiet fallback
        _CSRMV, _CSRMV8, _JACOBI8 = {}, {}, {}


def _int8_ok(data):
    """True when the entries store losslessly in int8 (the {4, +-1}
    Gram and 0/1 prolongators do; a deep Galerkin level could in
    principle outgrow the range and must then stay in fp)."""
    return bool(np.all(np.abs(data) <= 127)
                and np.all(data == np.rint(data)))


_STEN = {}
if os.environ.get('SPPEEC_STENCIL') != '0' and _CSRMV8:
    try:
        _STEN = {np.float32: (_mpf.stenjac_s, _mpf.stenmv_s),
                 np.float64: (_mpf.stenjac_d, _mpf.stenmv_d)}
    except AttributeError:
        _STEN = {}


class _Stencil0:
    """Tiled-stencil form of the level-0 Gram (tier 2, 2026-08-26).

    The plaquette Gram is a translation-invariant stencil (constant
    coefficient per (normal->normal, offset); verified on real Grams:
    36 offsets, coefficients +-1, diagonal 4). This packs the
    plaquette vector into per-normal dense 16^3 tiles over the
    occupied lattice and applies the stencil in Fortran with a 1-cell
    halo -- streaming only vectors, no matrix. Engagement is gated on
    EXACT probe equality against the csr path in :func:`build`, so a
    geometry that breaks any assumption silently keeps the csr.
    """
    TL = 16

    def __init__(self, dtype, tables, tile_of, loc, shape):
        self.jac, self.mv = _STEN[dtype.type]
        (self.nbt, self.nsrc, self.of, self.cf, self.sptr) = tables
        # ONE flat intp map replaces tile_of + the four loc arrays
        # (tier 3 bundle): five int64 vectors measured 549 MB at R4,
        # the flat index is 92 MB -- and a single fancy index is
        # faster than a five-array one
        TL = self.TL
        self.flat = (((tile_of*3 + loc[0])*TL + loc[1])*TL
                     + loc[2])*TL + loc[3]
        self.n = int(self.flat.size)
        self.shape = shape              # (NT, 3, TL, TL, TL)
        self.dtype = dtype
        self._buf = {}

    def _tiled(self, key):
        b = self._buf.get(key)
        if b is None:
            b = self._buf[key] = np.zeros(self.shape, dtype=self.dtype)
        return b

    def pack(self, v, key):
        t = self._tiled(key)
        t.reshape(-1)[self.flat] = v
        return t

    def unpack(self, t):
        return np.ascontiguousarray(t.reshape(-1)[self.flat])

    def matvec(self, x):
        xt = self.pack(x, 'x')
        yt = self.mv(xt.T, self.nbt, self.nsrc, self.of, self.cf,
                     self.sptr)
        return self.unpack(yt.T)

    def jacobi(self, x, b, wdi_t, nu):
        xt = self.pack(x, 'x')
        bt = self.pack(b, 'b')
        for _ in range(nu):
            xt = self.jac(xt.T, bt.T, wdi_t.T, self.nbt, self.nsrc,
                          self.of, self.cf, self.sptr).T
        return self.unpack(xt)

    @staticmethod
    def build(A, normal, base, dtype, jacobi_probe, basis=None,
              omega=None):
        """Extract + certify; returns (_Stencil0, wdi_tiled) or None.

        Two constructions share this body. With ``A`` (a materialised
        Gram): certification is bitwise in 'exact' mode, reorder
        tolerance in 'reordered' mode, against ``jacobi_probe``'s
        csr-path results. With ``basis`` (tier 3, ``A=None``): the
        Gram is NEVER materialised -- sampled rows come from a thin
        ``basis[rows] @ basis.T`` product, mode is 'reordered' by
        construction (a stored Gram and the two-step apply sum rows
        in different orders), certification runs against
        ``basis @ (basis.T @ x)`` at reorder tolerance, and the
        diagonal slots are asserted == 4 (every plaquette has four
        edges), which licenses the caller's analytic dinv.
        """
        if not _STEN or dtype.type not in _STEN:
            return None
        TL = _Stencil0.TL
        if basis is None:
            A = A.tocsr()
            n = A.shape[0]
        else:
            n = basis.shape[0]
        rows = None
        if n > 500_000:
            rows = np.random.default_rng(7).choice(n, 500_000,
                                                   replace=False)
            rows.sort()
        # extraction can run on a row SAMPLE: the exact probes below
        # certify the FULL reconstruction, so an offset the sample
        # missed or a coefficient it got wrong fails the probe and
        # falls back -- sampling costs correctness nothing and keeps
        # the setup temporaries bounded at rung scale.
        # 500k rows: at R4 the 2M-row extraction stacked +2.5 GB of
        # COO/key intermediates ON TOP of the getmesh_full peak
        # (measured; getmesh owns the build HWM at +6.6 GB) -- the
        # probes certify the full reconstruction either way, so the
        # sample only needs to SEE every slot, and 500k rows of a
        # 36-slot stencil oversamples that by orders of magnitude
        if basis is not None:
            # TIER 3: the Gram is never materialised. Sampled rows
            # come from a THIN product basis[rows] @ basis.T (~tens
            # of MB where the full Gram is a GB); certification below
            # runs against the matrix-free two-step apply.
            src = basis[rows] if rows is not None else basis
            coo = (src @ basis.T).tocoo()
            if rows is not None:
                del src
        elif rows is not None:
            As = A[rows]
            coo = As.tocoo()
            del As
        else:
            coo = A.tocoo()
        di = rows[coo.row] if rows is not None else coo.row
        dj, dv = coo.col, coo.data.astype(np.float64)
        del coo
        # slot key per nnz: (ni, nj, dx+1, dy+1, dz+1); diagonal is
        # the (nn, nn, 0,0,0) slot with coefficient 4
        dvec = base[dj] - base[di]
        if dvec.min() < -1 or dvec.max() > 1:
            return None                       # not a 1-halo stencil
        key = (((normal[di]*3 + normal[dj])*3 + dvec[:, 0] + 1)*3
               + dvec[:, 1] + 1)*3 + dvec[:, 2] + 1
        uk, inv = np.unique(key, return_inverse=True)
        # constancy: one coefficient per slot
        vmin = np.full(uk.size, np.inf)
        vmax = np.full(uk.size, -np.inf)
        np.minimum.at(vmin, inv, dv)
        np.maximum.at(vmax, inv, dv)
        if not np.all(vmin == vmax) or abs(vmin).max() > 127:
            return None
        # order: row-adjacent precedence pairs (csr columns ascend);
        # dedupe in numpy before touching python sets. On a single
        # solid block the pairs are conflict-free and one global slot
        # order reproduces every row's CSR summation EXACTLY. On
        # multi-block geometry they conflict (measured: 30 conflicts
        # on the halfbridge -- plaquette column ids interleave
        # differently per region), so no such order exists; the
        # stencil then engages in 'reordered' mode -- certified to
        # TOLERANCE instead of bitwise, which is sound for a
        # preconditioner (a reordered preconditioner cannot change
        # the converged answer, only the iteration count -- the
        # off-by-nholes lineage).
        same = (di[1:] == di[:-1])
        pair = np.unique(inv[:-1][same].astype(np.int64)*64
                         + inv[1:][same])
        prec = {(int(p)//64, int(p) % 64) for p in pair}
        conf = {(a_, b_) for (a_, b_) in prec if (b_, a_) in prec}
        mode = 'exact' if not conf and basis is None else 'reordered'
        prec -= conf
        del di, dj, dv, dvec, key, inv, same, pair
        if basis is not None:
            # every plaquette has exactly 4 edges: its diagonal Gram
            # entry is 4, or the analytic dinv upstream would be wrong
            for nn in range(3):
                u_diag = nn*81 + nn*27 + 13
                s = np.flatnonzero(uk == u_diag)
                if s.size != 1 or vmin[int(s[0])] != 4.0:
                    return None
        # toposort per output normal. Key decode (see encode above):
        #   u = ni*81 + nj*27 + (dx+1)*9 + (dy+1)*3 + (dz+1)
        onrm = uk//81
        nsrc, of, cf, sptr = [], [], [], [0]
        for nn in range(3):
            rest = {int(s) for s in np.flatnonzero(onrm == nn)}
            order = []
            while rest:
                free = [s for s in rest
                        if not any((t, s) in prec for t in rest
                                   if t != s)]
                if not free:
                    return None               # cycle: bad derivation
                s = min(free)
                order.append(s)
                rest.discard(s)
            for s in order:
                u = int(uk[s])
                nsrc.append((u//27) % 3 + 1)
                of.append([(u//9) % 3 - 1, (u//3) % 3 - 1,
                           u % 3 - 1])
                cf.append(int(vmin[s]))
            sptr.append(len(nsrc))
        nsrc = np.asarray(nsrc, np.int32)
        of = np.asarray(of, np.int64)
        cf = np.asarray(cf, np.int8)
        sptr = np.asarray(sptr, np.int32)
        # tiles
        tc = base//TL
        tkey = (tc[:, 0].astype(np.int64) << 42) \
            | (tc[:, 1].astype(np.int64) << 21) | tc[:, 2]
        ut, tile_of = np.unique(tkey, return_inverse=True)
        nt = ut.size
        # neighbour table: 27 offsets per tile
        nbt = np.zeros((27, nt), np.int32, order='F')
        tx = (ut >> 42) & 0x1FFFFF
        ty = (ut >> 21) & 0x1FFFFF
        tz = ut & 0x1FFFFF
        pos = {int(k): i for i, k in enumerate(ut)}
        for hx in (-1, 0, 1):
            for hy in (-1, 0, 1):
                for hz in (-1, 0, 1):
                    slot = 9*(hx + 1) + 3*(hy + 1) + (hz + 1)
                    for i in range(nt):
                        k = ((int(tx[i]) + hx) << 42) \
                            | ((int(ty[i]) + hy) << 21) \
                            | (int(tz[i]) + hz)
                        nbt[slot, i] = pos.get(k, -1) + 1
        # pack layout (NT, 3, TL, TL, TL) C-ordered; its .T is the
        # Fortran (X, Y, Z, ON, T) view, so the Fortran X axis is the
        # base-x local coordinate and OF rows are (dx, dy, dz)
        loc = (normal.astype(np.intp),
               (base[:, 2] % TL).astype(np.intp),
               (base[:, 1] % TL).astype(np.intp),
               (base[:, 0] % TL).astype(np.intp))
        shape = (nt, 3, TL, TL, TL)
        s2 = _Stencil0(dtype, (np.asfortranarray(nbt), nsrc,
                               np.asfortranarray(
                                   of.T.astype(np.int32)),
                               cf, sptr),
                       tile_of.astype(np.intp), loc, shape)
        s2.mode = mode
        # ---- certification ----------------------------------------
        # 'exact' mode: bitwise probe equality or bust. 'reordered'
        # mode: tolerance equality -- the reconstruction must be the
        # same OPERATOR (any mismatch of pattern or coefficients
        # shows up far above summation-reorder rounding, which for a
        # 13-term fp32 row sits at ~1e-7 relative).
        tol = 1e-5 if dtype.itemsize == 4 else 1e-11

        def _agree(got, ref):
            if mode == 'exact':
                return np.array_equal(got, ref)
            nr = float(np.linalg.norm(ref))
            return nr == 0.0 or \
                float(np.linalg.norm(got - ref)) <= tol*nr

        if basis is None:
            def _ref_mv(x):
                return _spmv(A, x)
        else:
            def _ref_mv(x):
                return np.asarray(basis @ (basis.T @ x), dtype=dtype)
        rng = np.random.default_rng(17)
        for _ in range(2):
            x = rng.standard_normal(n).astype(dtype)
            if not _agree(s2.matvec(x), _ref_mv(x)):
                return None
        x = rng.standard_normal(n).astype(dtype)
        b = rng.standard_normal(n).astype(dtype)
        if basis is None:
            ref, wdi = jacobi_probe(x, b)
        else:
            dinv0 = np.full(n, 0.25, dtype=dtype)
            wdi = (omega*dinv0).astype(dtype)
            ref = np.asarray(x + omega*dinv0*(b - _ref_mv(x)),
                             dtype=dtype)
        wdi_t = s2.pack(np.asarray(wdi, dtype), 'w').copy()
        got = s2.jacobi(x, b, wdi_t, 1)
        if not _agree(got, ref):
            return None
        return s2, wdi_t


def _spmv(A, x):
    """``A @ x`` through the threaded kernel when the types fit,
    scipy otherwise. CSR only; x must match the vector dtype the
    kernel expects (int8-data matrices pair with x's own dtype)."""
    if not x.flags.c_contiguous or A.format != 'csr':
        return A @ x
    if A.data.dtype == np.int8:
        f = _CSRMV8.get((x.dtype.type, A.indices.dtype.type))
        # int8 levels only exist when the kernels loaded, so f is
        # only None for an exotic x dtype -- restore fp then
        if f is None:
            return sp.csr_matrix(
                (A.data.astype(x.dtype), A.indices, A.indptr),
                shape=A.shape) @ x
        return f(A.indptr, A.indices, A.data, x)
    f = _CSRMV.get((A.data.dtype.type, A.indices.dtype.type))
    if f is None or x.dtype != A.data.dtype:
        return A @ x
    return f(A.indptr, A.indices, A.data, x)


def plaquette_geometry(Y, fil_axis, fil_cell, nplaq):
    """(normal, cell) per plaquette column of ``Y``.

    A lattice face has 4 edges spanning exactly two axes; the third is
    its normal. Position is the componentwise min of the 4 edge cells.
    Vectorised over all columns at once -- every column has exactly 4
    nonzeros, so the index array reshapes to (nplaq, 4).
    """
    Yc = Y.tocsc()
    # Slice by INDPTR, not by assuming 4*nplaq contiguous entries: in
    # equiterminal the same Y carries hole cycles, port cycles and
    # redistribution modes after the plaquettes, and extra ROWS below
    # them, so only the first nplaq columns are lattice faces.
    ptr = Yc.indptr[:nplaq + 1]
    if not np.all(np.diff(ptr) == 4):
        bad = int(np.count_nonzero(np.diff(ptr) != 4))
        raise RuntimeError("%d of the first %d columns are not 4-edge "
                           "plaquettes -- geometry cannot be recovered"
                           % (bad, nplaq))
    rows = Yc.indices[ptr[0]:ptr[-1]].reshape(nplaq, 4)
    ax = np.asarray(fil_axis)[rows]                 # (nplaq, 4)
    cells = np.asarray(fil_cell)[rows]              # (nplaq, 4, 3)
    # normal = the axis not present among the 4 edges: 0+1+2 minus the
    # two distinct axes present
    amin = ax.min(axis=1)
    amax = ax.max(axis=1)
    normal = 3 - amin - amax
    base = cells.min(axis=1)
    ok = amin != amax
    if not np.all(ok):
        raise RuntimeError("%d plaquette(s) span a single axis -- not a "
                           "lattice face" % int((~ok).sum()))
    return normal.astype(np.int64), base.astype(np.int64)


class GeoMG:
    """Matrix-free-hierarchy geometric multigrid on plaquette DOFs.

    Aggregates 2x2x2 blocks of same-normal faces per level (SEMI-
    COARSENING: an axis whose extent is already small is not coarsened,
    which is the right behaviour for the pancake geometries SuperPEEC
    partitions into anyway -- a 4-cell-thick board must not be coarsened
    in z more than once). Coarse operators are Galerkin ``P^T A P`` with
    a PIECEWISE-CONSTANT P: unlike smoothed aggregation, P is not
    smoothed, so the coarse operators do not densify and the hierarchy
    stays cheap -- which is the entire point.

    Smoother: damped Jacobi. Deliberately not Gauss-Seidel: Jacobi is
    what parallelises on the many-core machine this exists to reach, and
    the GPU AMG experiment already had to swap GS out for the same
    reason.

    NOTE ON THE KERNEL. ``Y^T Y`` is SINGULAR here -- its kernel is the
    cube boundaries (a current differing by a cube boundary is the same
    current). Unlike the classic H(curl) case there is no mass term, so
    the kernel is EXACT rather than near-kernel: A annihilates it, a
    consistent rhs never excites it, and no Hiptmair-style auxiliary
    space correction is needed to damp it. The coarsest level is solved
    with a pseudo-inverse for the same reason.
    """

    def __init__(self, A, normal, base, nu=2, omega=2.0/3.0,
                 max_coarse=400, max_levels=12, basis=None):
        self.nu = int(nu)
        self.omega = float(omega)
        self.levels = []
        self.Ps = []
        self._sten0 = None
        self._wdi0_t = None
        self._nnz0_est = 0
        # TIER 3 (2026-08-27): with ``basis`` (the plaquette rows Yp,
        # A=None) the level-0 Gram is NEVER materialised -- its
        # stencil is extracted from a thin sampled product and
        # certified matrix-free, and level 1 is the Gram of the
        # AGGREGATED basis, A1 = (Yp^T P)^T (Yp^T P), a pair of thin
        # SpGEMMs. The full-Gram SpGEMM was the measured R4 build
        # peak-setter. Any certification failure (or a stalled first
        # coarsening) falls back to materialising A classically.
        use_basis = False
        P0 = nrm1 = bs1 = None
        if A is None:
            got = None
            if _STEN:
                try:
                    got = _Stencil0.build(
                        None, normal, base, np.dtype(basis.dtype),
                        None, basis=basis, omega=self.omega)
                except Exception:       # never break a solve
                    got = None
            if got is not None:
                P0, nrm1, bs1 = self._aggregate(normal.copy(),
                                                base.copy())
                if P0.shape[1] < basis.shape[0]:
                    use_basis = True
            if not use_basis:
                got = None
                A = (basis @ basis.T).tocsr()
        # The hierarchy is stored and applied in the operator's OWN
        # dtype (fp32 under the phase-1 precision policy -- see
        # port_impedance._PRECOND_DT). P must be cast BEFORE the
        # Galerkin product or scipy silently upcasts every coarse
        # operator back to float64.
        if use_basis:
            self._sten0, self._wdi0_t = got
            self.dtype = basis.dtype
            n0 = basis.shape[0]
            self.levels.append(None)     # level 0 lives in the stencil
            P0 = P0.tocsr().astype(self.dtype)
            self.Ps.append(P0)
            A = self._probe_coarse(P0, nrm1, bs1)
            nrm, bs = nrm1, bs1
            # level-0 nnz never counted exactly (no matrix); ~10.7
            # entries/row measured across the geometry family --
            # DIAGNOSTIC only (nnz/nnz_ratio reporting)
            self._nnz0_est = int(n0*10.7)
        else:
            A = A.tocsr()
            self.dtype = A.dtype
            nrm, bs = normal.copy(), base.copy()
        for _ in range(max_levels):
            self.levels.append(A)
            if A.shape[0] <= max_coarse:
                break
            P, nrm, bs = self._aggregate(nrm, bs)
            if P.shape[1] >= A.shape[0]:
                break                      # coarsening stalled
            P = P.tocsr().astype(self.dtype)
            self.Ps.append(P)
            A = (P.T @ A @ P).tocsr()
        self.dinv = [np.full(n0, 0.25, dtype=self.dtype) if L is None
                     else (1.0/np.where(np.abs(L.diagonal()) > 0,
                                        L.diagonal(), 1.0)
                           ).astype(self.dtype)
                     for L in self.levels]
        # CSR transposes of the prolongators, so the restriction
        # P.T @ r runs through the row-parallel threaded kernel too
        # (a CSC transpose apply scatters and cannot thread
        # deterministically). P is 1-nnz-per-row aggregation, so the
        # extra storage is negligible.
        self.PTs = [P.T.tocsr() for P in self.Ps]
        # wdi = omega*dinv precomputed per level for the fused Jacobi
        # kernel; numerically identical to evaluating omega*di inside
        # the sweep (same multiply, same values)
        self._wdi = [(self.omega*di).astype(self.dtype)
                     for di in self.dinv]
        # LOSSLESS int8 conversion of the hierarchy data (tier 1,
        # 2026-08-26). Every level but the coarsest (which lsqr/pinv
        # must read numerically) and every prolongator whose entries
        # fit int8 exactly -- {4, +-1} Gram, 0/1 aggregation -- drops
        # its fp data for int8: 4x fewer data bytes streamed per
        # sweep, identical floats out. Only done when the kernels
        # loaded (an int8-data csr must never reach plain scipy `@`).
        if _CSRMV8:
            for L in self.levels[:-1]:
                if L is not None and _int8_ok(L.data):
                    L.data = L.data.astype(np.int8)
            for P in list(self.Ps) + list(self.PTs):
                if _int8_ok(P.data):
                    P.data = P.data.astype(np.int8)
        # tier 2: certified tiled stencil for level 0. build() gates
        # engagement on EXACT probe equality against the csr path
        # (matvec and one fused sweep), so any geometry that breaks a
        # stencil assumption silently keeps the csr. The probe below
        # reproduces exactly what _smooth would compute.
        if self._sten0 is None and len(self.levels) > 1 and _STEN:
            def _probe(x, b):
                A0 = self.levels[0]
                wdi = self._wdi[0]
                if A0.data.dtype == np.int8:
                    f = _JACOBI8.get((x.dtype.type,
                                      A0.indices.dtype.type))
                    if f is not None:
                        return f(A0.indptr, A0.indices, A0.data,
                                 x, b, wdi), wdi
                return np.asarray(
                    x + self.omega*self.dinv[0]*(b - _spmv(A0, x)),
                    dtype=self.dtype), wdi
            try:
                got = _Stencil0.build(self.levels[0], normal, base,
                                      np.dtype(self.dtype), _probe)
            except Exception:          # a stencil must never break
                got = None             # a solve; the csr path stands
            if got is not None:
                self._sten0, self._wdi0_t = got
        Ac = self.levels[-1]
        # pinv rank decisions in float64 (the kernel is exact and must
        # be cut cleanly); STORAGE in the hierarchy dtype
        self.coarse_pinv = np.linalg.pinv(
            Ac.toarray().astype(np.float64)).astype(self.dtype) \
            if Ac.shape[0] <= 2000 else None
        self.nnz = self._nnz0_est + sum(L.nnz for L in self.levels
                                        if L is not None)
        self.nnz_ratio = self.nnz/max(
            self._nnz0_est or (self.levels[0].nnz
                               if self.levels[0] is not None else 1), 1)
        self.sizes = [self.dinv[i].size if L is None else L.shape[0]
                      for i, L in enumerate(self.levels)]
        if self._sten0 is not None and \
                os.environ.get('SPPEEC_GPU') == '0':
            # CPU-only run with the stencil certified: the level-0
            # csr has no remaining reader (smoother/residual go
            # through the stencil; only the coarsest level feeds
            # lsqr/pinv) -- release it. Any accidental use fails
            # loudly rather than silently slowly.
            self.levels[0] = None

    def _probe_coarse(self, P0, nrm1, bs1):
        """A1 = P0^T A P0 by COLOUR PROBING through the level-0
        stencil -- no Gram-sized SpGEMM anywhere (the W1 = Yp^T P0
        route measured a peak-neutral 1.8 GB at R4, replacing the
        Gram spike 1:1).

        The coarse operator's reach is one coarse cell (fine offsets
        are +-1, aggregates 2x2x2), so colouring coarse columns by
        (normal, base mod 3) makes same-colour columns non-
        interacting: 81 probes u = P0 @ 1_colour, w = A u (the
        certified stencil), z = P0^T w, and each nonzero z[r]
        IS the single entry A1[r, d] whose column d is the unique
        same-colour neighbour of r -- decoded arithmetically. The
        probes are 0/1 vectors, so every sum is an exact integer:
        the assembled values equal the SpGEMM's exactly (verified),
        only the storage order differs. Transient: two fine vectors
        per probe (~100 MB at R4 vs ~1.8 GB)."""
        nc = P0.shape[1]
        m1 = int(bs1[:, 1].max()) + 2
        m2 = int(bs1[:, 2].max()) + 2
        ckey = ((nrm1*(int(bs1[:, 0].max()) + 2)
                 + bs1[:, 0])*m1 + bs1[:, 1])*m2 + bs1[:, 2]
        order = np.argsort(ckey, kind='stable')
        skey = ckey[order]
        P0T = P0.T.tocsr()
        rows_i, cols_i, vals_i = [], [], []
        mod = bs1 % 3
        for cn in range(3):
            base_sel = nrm1 == cn
            for mx in range(3):
                for my in range(3):
                    for mz in range(3):
                        sel = np.flatnonzero(
                            base_sel & (mod[:, 0] == mx)
                            & (mod[:, 1] == my) & (mod[:, 2] == mz))
                        if sel.size == 0:
                            continue
                        e = np.zeros(nc, dtype=self.dtype)
                        e[sel] = 1.0
                        w = self._sten0.matvec(
                            np.ascontiguousarray(P0 @ e))
                        z = P0T @ w
                        r = np.flatnonzero(z)
                        if r.size == 0:
                            continue
                        # unique same-colour column in r's 3^3
                        # neighbourhood: per axis the one offset in
                        # {-1,0,1} congruent to the colour mod 3
                        off = np.stack(
                            [(m - bs1[r, k]) % 3
                             for k, m in enumerate((mx, my, mz))],
                            axis=1).astype(np.int64)
                        off[off == 2] = -1
                        pos = bs1[r] + off
                        # a negative or overflowing coordinate could
                        # ALIAS a valid key through the mixed-radix
                        # encoding -- mask before looking up
                        inb = ((pos >= 0).all(axis=1)
                               & (pos[:, 1] < m1 - 1)
                               & (pos[:, 2] < m2 - 1))
                        r, pos = r[inb], pos[inb]
                        if r.size == 0:
                            continue
                        pk = ((np.full(r.size, cn, np.int64)
                               * (int(bs1[:, 0].max()) + 2)
                               + pos[:, 0])*m1 + pos[:, 1])*m2 \
                            + pos[:, 2]
                        i = np.searchsorted(skey, pk)
                        i = np.minimum(i, skey.size - 1)
                        ok = skey[i] == pk
                        rows_i.append(r[ok])
                        cols_i.append(order[i[ok]])
                        vals_i.append(z[r[ok]].astype(self.dtype))
        A1 = sp.csr_matrix(
            (np.concatenate(vals_i), (np.concatenate(rows_i),
                                      np.concatenate(cols_i))),
            shape=(nc, nc))
        A1.sum_duplicates()
        return A1

    def _aggregate(self, normal, base):
        """2x2x2 geometric agglomeration, per face orientation."""
        span = base.max(axis=0) - base.min(axis=0) + 1
        div = np.where(span > 2, 2, 1)            # semi-coarsening
        cb = base//div[None, :]
        # ENCODED scalar key instead of np.unique(..., axis=0): the
        # axis-0 unique views rows as void records and copies its way
        # through the sort (~1.5 GB at R4, and it runs on EVERY
        # hierarchy build -- the transient every Gram-formation route
        # kept hitting). The mixed-radix encode is order-preserving
        # (components are bounded non-negatives), so unique/first/inv
        # are BIT-IDENTICAL to the axis-0 version.
        # np.unique returns (unique, index, inverse) IN THAT ORDER when
        # both flags are set -- binding them the other way round gives a
        # P with the coarse count as its ROW dimension and a dimension
        # mismatch two levels later.
        r0 = np.int64(cb[:, 0].max()) + 1
        r1 = np.int64(cb[:, 1].max()) + 1
        r2 = np.int64(cb[:, 2].max()) + 1
        key = ((normal.astype(np.int64)*r0 + cb[:, 0])*r1
               + cb[:, 1])*r2 + cb[:, 2]
        _, first, inv = np.unique(key, return_index=True,
                                  return_inverse=True)
        del key
        inv = np.asarray(inv).ravel()
        nc = int(inv.max()) + 1
        P = sp.coo_matrix((np.ones(inv.size), (np.arange(inv.size), inv)),
                          shape=(inv.size, nc))
        return P.tocsc(), normal[first], cb[first]

    def _smooth(self, lv, x, b):
        if lv == 0 and self._sten0 is not None:
            return self._sten0.jacobi(x, b, self._wdi0_t, self.nu)
        A, di = self.levels[lv], self.dinv[lv]
        if A.data.dtype == np.int8:
            f = _JACOBI8.get((x.dtype.type, A.indices.dtype.type))
            if f is not None and x.flags.c_contiguous \
                    and b.flags.c_contiguous:
                # fused sweep: one pass instead of four NROW-sized
                # numpy temporaries; same per-element operation order,
                # so the floats are identical
                wdi = self._wdi[lv]
                for _ in range(self.nu):
                    x = f(A.indptr, A.indices, A.data, x, b, wdi)
                return x
        for _ in range(self.nu):
            x = x + self.omega*di*(b - _spmv(A, x))
        return x

    def _vcycle(self, lv, b, x):
        if lv == len(self.levels) - 1:
            if self.coarse_pinv is not None:
                return self.coarse_pinv @ b
            return spla.lsqr(self.levels[lv], b, atol=1e-10, btol=1e-10)[0]
        x = self._smooth(lv, x, b)
        if lv == 0 and self._sten0 is not None:
            r = b - self._sten0.matvec(x)
        else:
            r = b - _spmv(self.levels[lv], x)
        P = self.Ps[lv]
        xc = self._vcycle(lv + 1, _spmv(self.PTs[lv], r),
                          np.zeros(P.shape[1], dtype=self.dtype))
        x = x + _spmv(P, xc)
        return self._smooth(lv, x, b)

    def __call__(self, b, cycles=1):
        x = np.zeros(b.shape[0], dtype=self.dtype)
        for _ in range(cycles):
            x = self._vcycle(0, np.asarray(b, dtype=self.dtype), x)
        return x
