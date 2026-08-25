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
_CSRMV = {}
if os.environ.get('SPPEEC_SPMV') != '0':
    try:
        import mp_fortran as _mpf
        _CSRMV = {(np.float32, np.int32): _mpf.csrmv_s,
                  (np.float64, np.int32): _mpf.csrmv_d,
                  (np.float32, np.int64): _mpf.csrmv_sl,
                  (np.float64, np.int64): _mpf.csrmv_dl}
    except (ImportError, AttributeError):  # old .so: quiet fallback
        _CSRMV = {}


def _spmv(A, x):
    """``A @ x`` through the threaded kernel when the types fit,
    scipy otherwise. CSR only; x must match A's real dtype."""
    f = _CSRMV.get((A.data.dtype.type, A.indices.dtype.type))
    if (f is None or A.format != 'csr'
            or x.dtype != A.data.dtype or not x.flags.c_contiguous):
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
                 max_coarse=400, max_levels=12):
        self.nu = int(nu)
        self.omega = float(omega)
        self.levels = []
        self.Ps = []
        # The hierarchy is stored and applied in A's OWN dtype (fp32
        # under the phase-1 precision policy -- see
        # port_impedance._PRECOND_DT). P must be cast BEFORE the
        # Galerkin product or scipy silently upcasts every coarse
        # operator back to float64.
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
        self.dinv = [(1.0/np.where(np.abs(L.diagonal()) > 0,
                                   L.diagonal(), 1.0)).astype(self.dtype)
                     for L in self.levels]
        # CSR transposes of the prolongators, so the restriction
        # P.T @ r runs through the row-parallel threaded kernel too
        # (a CSC transpose apply scatters and cannot thread
        # deterministically). P is 1-nnz-per-row aggregation, so the
        # extra storage is negligible.
        self.PTs = [P.T.tocsr() for P in self.Ps]
        Ac = self.levels[-1]
        # pinv rank decisions in float64 (the kernel is exact and must
        # be cut cleanly); STORAGE in the hierarchy dtype
        self.coarse_pinv = np.linalg.pinv(
            Ac.toarray().astype(np.float64)).astype(self.dtype) \
            if Ac.shape[0] <= 2000 else None
        self.nnz = sum(L.nnz for L in self.levels)
        self.nnz_ratio = self.nnz/max(self.levels[0].nnz, 1)
        self.sizes = [L.shape[0] for L in self.levels]

    def _aggregate(self, normal, base):
        """2x2x2 geometric agglomeration, per face orientation."""
        span = base.max(axis=0) - base.min(axis=0) + 1
        div = np.where(span > 2, 2, 1)            # semi-coarsening
        cb = base//div[None, :]
        key = np.stack([normal, cb[:, 0], cb[:, 1], cb[:, 2]], axis=1)
        # np.unique returns (unique, index, inverse) IN THAT ORDER when
        # both flags are set -- binding them the other way round gives a
        # P with the coarse count as its ROW dimension and a dimension
        # mismatch two levels later.
        _, first, inv = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
        inv = np.asarray(inv).ravel()
        nc = int(inv.max()) + 1
        P = sp.coo_matrix((np.ones(inv.size), (np.arange(inv.size), inv)),
                          shape=(inv.size, nc))
        return P.tocsc(), normal[first], cb[first]

    def _smooth(self, lv, x, b):
        A, di = self.levels[lv], self.dinv[lv]
        for _ in range(self.nu):
            x = x + self.omega*di*(b - _spmv(A, x))
        return x

    def _vcycle(self, lv, b, x):
        if lv == len(self.levels) - 1:
            if self.coarse_pinv is not None:
                return self.coarse_pinv @ b
            return spla.lsqr(self.levels[lv], b, atol=1e-10, btol=1e-10)[0]
        x = self._smooth(lv, x, b)
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
