# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Hierarchical (HODLR) compression + factorization of the branch impedance
Z = R + jw*Lp, for use as the branch-block solve of an LpPR preconditioner.

Stage 1 of the H-LU program (see the stage-0 gate in the project memory /
precond whitepaper trail): the iteration-growth census showed every
short-range preconditioner leaks an O(n) population of long-range modes,
while stage 0 measured that an eps-truncated ALL-RANGE approximate inverse
holds iteration counts flat in N at fixed eps. This module supplies that
inverse scalably:

* per orientation (e/f/g are block-diagonal in Lp), filaments are ordered by
  Morton code and split into a binary cluster tree;
* sibling off-diagonal blocks of Lp are compressed by ADAPTIVE RANDOMIZED
  SVD with a rigorous Frobenius tail bound, assembled chunk-wise from the
  closed-form genL3D offset tables (entries are table lookups, essentially
  free -- so explicit blocks are affordable and no pivoting heuristic is
  needed; the full dense Lp is still never assembled, only one off-diagonal
  block at a time, transiently). Partially pivoted ACA (:func:`_aca`) was
  tried first and MEASURED to miss rank on these weakly admissible touching
  blocks (solve errors 10-40x the tolerance, growing with N); it is kept in
  the module for future entry-expensive kernels (e.g. layered-media
  dielectrics) where explicit assembly is not free, with the caveat that it
  needs ACA+/hybrid pivoting there. The compression is FREQUENCY-INDEPENDENT
  (Lp only);
* per frequency, Z = R + jw*Lp is factored by recursive Woodbury: dense LU
  at the leaves, and at each internal node the low-rank off-diagonal
  correction [0, jw*U V^T; jw*V U^T, 0] (Lp symmetry: the (2,1) factors are
  the (1,2) factors swapped) is folded in by the Woodbury identity, with the
  small (r1+r2) core LU'd.

HODLR uses WEAK admissibility (sibling halves touch), so interface ranks
grow with cluster size on 3-D volumes -- accepted for stage 1 (rank stats
are reported; the production-grade endgame is strong-admissibility H^2 via
a library). At the loose eps of a preconditioner this is measured to be
affordable through 23^3 (see profile_stage1_hodlr.py).
"""
import ctypes as _ct
import os as _os
import ctypes as _ct
import os as _os
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from greens import genL3D
from reluctance import filament_geometry

try:
    # native (C + BLAS/LAPACK) forward solve -- see hodlr_native.c; the
    # Python tree walk costs ~10x the flops on the S_c sampling path
    _nat = _ct.CDLL(_os.path.join(_os.path.dirname(_os.path.abspath(
        __file__)), "libhodlrnat.so"))
    _nat.hodlr_solve.argtypes = [_ct.c_void_p]*5 + [_ct.c_void_p]*7 + \
        [_ct.c_int, _ct.c_int, _ct.c_void_p, _ct.c_int, _ct.c_int]
    _nat.hodlr_solve_t.argtypes = [_ct.c_void_p]*5 + [_ct.c_void_p]*7 + \
        [_ct.c_double, _ct.c_double,
         _ct.c_int, _ct.c_int, _ct.c_void_p, _ct.c_int, _ct.c_int]
except OSError:
    _nat = None


def _aca(entry_rows, entry_cols, m, n, eps, rmax):
    """Partially pivoted ACA of an m x n block. entry_rows(i) -> row i,
    entry_cols(j) -> column j. Returns (U, V) with block ~= U @ V.T."""
    U = np.zeros((m, 0))
    V = np.zeros((n, 0))
    used = np.zeros(m, bool)
    i = 0
    fro2 = 0.0
    for _ in range(min(rmax, m, n)):
        r = entry_rows(i) - U[i] @ V.T if U.shape[1] else entry_rows(i)
        j = int(np.argmax(np.abs(r)))
        piv = r[j]
        used[i] = True
        if abs(piv) < 1e-300:
            free = np.nonzero(~used)[0]
            if free.size == 0:
                break
            i = int(free[0])
            continue
        v = r/piv
        c = entry_cols(j) - U @ V[j] if U.shape[1] else entry_cols(j)
        u = c
        unorm2 = float(np.real(u @ np.conj(u)))
        vnorm2 = float(np.real(v @ np.conj(v)))
        fro2 += unorm2*vnorm2
        U = np.column_stack([U, u])
        V = np.column_stack([V, v])
        if unorm2*vnorm2 <= (eps*eps)*fro2:
            break
        au = np.abs(u)
        au[used] = -1.0
        i = int(np.argmax(au))
        if au[i] < 0:
            break
    return U, V


def _lowrank(B, eps, rmax, rng):
    """Adaptive randomized low-rank factorization: B ~= U @ V.T with
    ||B - U V^T||_F <= eps*||B||_F, rank-adaptive via the exact projection
    identity ||B - Q Q^T B||_F^2 = ||B||_F^2 - ||Q^T B||_F^2."""
    m, n = B.shape
    nB2 = float(np.sum(B*B))
    if nB2 == 0.0:
        return np.zeros((m, 0)), np.zeros((n, 0))
    k = 32
    while True:
        k = min(k, m, n, rmax)
        G = rng.standard_normal((n, k))
        Q, _ = np.linalg.qr(B @ G)
        C = Q.T @ B
        tail2 = max(nB2 - float(np.sum(C*C)), 0.0)
        if tail2 <= (eps*eps)*nB2 or k >= min(m, n, rmax):
            break
        k *= 2
    Us, s, Vh = np.linalg.svd(C, full_matrices=False)
    s2 = s*s
    resid = tail2 + np.concatenate([np.cumsum(s2[::-1])[::-1][1:], [0.0]])
    r = int(np.searchsorted(-resid, -(eps*eps)*nB2) + 1)
    r = min(max(r, 1), s.size)
    return Q @ (Us[:, :r]*s[:r]), Vh[:r].T


class _Node:
    __slots__ = ('lo', 'hi', 'left', 'right', 'U', 'V', 'lu', 'Y', 'G',
                 'r', 'Yt')


def _morton(pos):
    key = np.zeros(pos.shape[0], np.int64)
    for b in range(10):
        for d in range(3):
            key |= ((pos[:, d].astype(np.int64) >> b) & 1) << (3*b + d)
    return key


class HodlrZ:
    """HODLR-compressed Z = R + jw*Lp for one tree M: build once
    (frequency-independent Lp compression), factor per frequency, solve.

    Parameters: eps (ACA tolerance, the preconditioner-quality knob;
    stage-0 plateau guidance ~1e-2), leaf (dense leaf size), rmax (rank
    cap per block).
    """

    def __init__(self, M, eps=1e-2, leaf=64, rmax=400):
        self.eps = eps
        pos, orient = filament_geometry(M)
        l = M.e.l
        maxsep = int(pos.max()) + 2
        ie = genL3D(l[0], l[1], l[2], maxsep, maxsep, maxsep) \
            / getattr(M.e, 'lscale', 1.0)
        ff = np.transpose(genL3D(l[1], l[0], l[2], maxsep, maxsep, maxsep),
                          (1, 0, 2)) / getattr(M.f, 'lscale', 1.0)
        gg = np.transpose(genL3D(l[0], l[2], l[1], maxsep, maxsep, maxsep),
                          (0, 2, 1)) / getattr(M.g, 'lscale', 1.0)
        self.tables = [ie, ff, gg]
        self.trees = []
        self.perms = []
        self.stats = []          # (level, rank, m, n) per compressed block
        for o in range(3):
            idx = np.nonzero(orient == o)[0]
            if idx.size == 0:
                self.trees.append(None)
                self.perms.append(idx)
                continue
            key = _morton(pos[idx])
            order = np.argsort(key, kind='stable')
            pidx = idx[order]
            self.perms.append(pidx)
            P = pos[pidx]
            T = self.tables[o]
            rng = np.random.default_rng(12345 + o)

            def fill_block(rpos, cpos):
                m = rpos.shape[0]
                B = np.empty((m, cpos.shape[0]))
                step = max(1, int(4e6 // max(cpos.shape[0], 1)))
                for s0 in range(0, m, step):
                    d = np.abs(rpos[s0:s0+step, None, :] - cpos[None, :, :])
                    B[s0:s0+step] = T[d[..., 0], d[..., 1], d[..., 2]]
                return B

            def build(lo, hi, level):
                nd = _Node()
                nd.lo, nd.hi = lo, hi
                if hi - lo <= leaf:
                    nd.left = nd.right = None
                    nd.U = fill_block(P[lo:hi], P[lo:hi])  # dense Lp leaf
                    nd.V = None
                    return nd
                mid = (lo + hi)//2
                nd.left = build(lo, mid, level+1)
                nd.right = build(mid, hi, level+1)
                B = fill_block(P[lo:mid], P[mid:hi])
                U, V = _lowrank(B, eps, rmax, rng)
                del B
                nd.U, nd.V = U, V
                self.stats.append((level, U.shape[1], mid-lo, hi-mid))
                return nd

            self.trees.append(build(0, pidx.size, 0))

    def nbytes(self):
        tot = 0

        def walk(nd):
            nonlocal tot
            if nd is None:
                return
            if nd.left is None:
                tot += nd.U.nbytes
                return
            tot += nd.U.nbytes + nd.V.nbytes
            walk(nd.left)
            walk(nd.right)
        for t in self.trees:
            walk(t)
        return tot

    def factor(self, jomega, rvals, transpose=True):
        """Factor Z = diag(r_o) + jw*Lp per orientation (rvals = (r_e, r_f,
        r_g)). Recursive Woodbury; call before solve() at each frequency.

        transpose=False skips the Yt = D^{-T} W^T arrays (halving the
        correction-factor memory; solve_t then raises). Measured note: for
        the S_c sampling use, serving 'T' ops from the eps-level complex
        symmetry instead of solve_t made NO difference to preconditioner
        quality (42/149-class counts identical) -- exact transpose is
        kept for operators that need it, not required there."""
        self.jw = jomega
        self._has_t = transpose

        def fac(nd, r):
            if nd.left is None:
                Z = self.jw*nd.U + r*np.eye(nd.hi - nd.lo)
                nd.lu = lu_factor(Z)
                return
            fac(nd.left, r)
            fac(nd.right, r)
            m1 = nd.left.hi - nd.left.lo
            m2 = nd.right.hi - nd.right.lo
            r1 = nd.U.shape[1]
            # correction: [[0, jw U V^T],[jw V U^T, 0]] -> Ucat Wcat with
            # Ucat = [[jw U, 0],[0, jw V]], Wcat = [[0, V^T],[U^T, 0]]
            Ucat = np.zeros((m1+m2, 2*r1), np.complex128)
            Ucat[:m1, :r1] = self.jw*nd.U
            Ucat[m1:, r1:] = self.jw*nd.V
            # chunk the correction solves by columns: the recursion holds a
            # temporary of the RHS width at EVERY level simultaneously, so
            # full-width solves cost depth x (n x 2r) transient memory --
            # measured as the 12 GB machine's OOM driver at 19^3, rank ~700
            Y = np.empty_like(Ucat)
            for lo in range(0, Ucat.shape[1], 128):
                Y[:m1, lo:lo+128] = self._solve_node(nd.left,
                                                     Ucat[:m1, lo:lo+128])
                Y[m1:, lo:lo+128] = self._solve_node(nd.right,
                                                     Ucat[m1:, lo:lo+128])
            nd.r = r1
            if r1 == 0:
                nd.Y = nd.G = nd.Yt = None
                return
            W = np.zeros((2*r1, m1+m2), np.complex128)
            W[:r1, m1:] = nd.V.T
            W[r1:, :m1] = nd.U.T
            nd.Y = Y
            nd.G = lu_factor(np.eye(2*r1) + W @ Y)
            # transpose-solve data: Z^T = D^T + W^T Ucat^T, whose Woodbury
            # core is I + Ucat^T D^{-T} W^T = G^T -- so G is REUSED with
            # trans=1 solves; only Yt = D^{-T} W^T is new. Needed because
            # the randomized-SVD factors leave Ztil only eps-symmetric,
            # and serving S_c's transpose sampling ops from that
            # approximate symmetry measurably degraded the matvec-built
            # Schur (HODLR 150 vs 30 mv at 4fres).
            if not self._has_t:
                nd.Yt = None
                return
            Wt = np.zeros((m1+m2, 2*r1), np.complex128)
            Wt[:m1, r1:] = nd.U
            Wt[m1:, :r1] = nd.V
            Yt = np.empty_like(Wt)
            for lo in range(0, 2*r1, 128):
                Yt[:m1, lo:lo+128] = self._solve_node_t(nd.left,
                                                        Wt[:m1, lo:lo+128])
                Yt[m1:, lo:lo+128] = self._solve_node_t(nd.right,
                                                        Wt[m1:, lo:lo+128])
            nd.Yt = Yt
        for o, t in enumerate(self.trees):
            if t is not None:
                fac(t, rvals[o])
        self._pack()

    def _pack(self):
        """Flatten the factored trees into per-orientation pointer tables
        for the native C solve (hodlr_native.c). Leaf LU and Woodbury-core
        G get Fortran-ordered copies with 1-based int32 pivots (LAPACK
        zgetrs convention); U/V/Y are referenced in place (numpy C order =
        the Fortran-ordered transpose the C GEMMs expect)."""
        self._packed = []
        if _nat is None:
            return
        for t in self.trees:
            if t is None:
                self._packed.append(None)
                continue
            kind = []; m = []; r = []; left = []; right = []
            ptrs = {k: [] for k in
                    ('lu', 'piv', 'U', 'V', 'Y', 'G', 'Gpiv', 'Yt')}
            keep = []

            def zero():
                for k in ptrs:
                    ptrs[k].append(0)

            def walk(nd):
                if nd.left is None:
                    lu_f = np.asfortranarray(nd.lu[0])
                    piv = (nd.lu[1] + 1).astype(np.int32)
                    keep.extend([lu_f, piv])
                    kind.append(0)
                    m.append(nd.hi - nd.lo)
                    r.append(0)
                    left.append(-1)
                    right.append(-1)
                    zero()
                    i = len(kind) - 1
                    ptrs['lu'][i] = lu_f.ctypes.data
                    ptrs['piv'][i] = piv.ctypes.data
                    return i
                il = walk(nd.left)
                ir = walk(nd.right)
                kind.append(1)
                m.append(nd.hi - nd.lo)
                r.append(nd.r)
                left.append(il)
                right.append(ir)
                zero()
                i = len(kind) - 1
                if nd.r > 0:
                    G_f = np.asfortranarray(nd.G[0])
                    Gpiv = (nd.G[1] + 1).astype(np.int32)
                    # U/V are stored REAL (Lp factors); the C kernel reads
                    # complex128 buffers, so complex copies are mandatory
                    U = np.ascontiguousarray(nd.U, np.complex128)
                    V = np.ascontiguousarray(nd.V, np.complex128)
                    Y = np.ascontiguousarray(nd.Y)
                    keep.extend([G_f, Gpiv, U, V, Y])
                    ptrs['G'][i] = G_f.ctypes.data
                    ptrs['Gpiv'][i] = Gpiv.ctypes.data
                    ptrs['U'][i] = U.ctypes.data
                    ptrs['V'][i] = V.ctypes.data
                    ptrs['Y'][i] = Y.ctypes.data
                    if self._has_t:
                        Yt = np.ascontiguousarray(nd.Yt)
                        keep.append(Yt)
                        ptrs['Yt'][i] = Yt.ctypes.data
                return i

            root = walk(t)
            pk = dict(
                kind=np.asarray(kind, np.int32),
                m=np.asarray(m, np.int32),
                r=np.asarray(r, np.int32),
                left=np.asarray(left, np.int32),
                right=np.asarray(right, np.int32),
                root=root, rmax=int(max(r) if r else 0), keep=keep)
            for k in ptrs:
                pk[k] = np.asarray(ptrs[k], np.int64)
            self._packed.append(pk)

    def _native_solve(self, oi, b):
        pk = self._packed[oi]
        n = int(pk['m'][pk['root']])
        x = np.asfortranarray(b.reshape(n, -1))
        _nat.hodlr_solve(
            pk['kind'].ctypes.data, pk['m'].ctypes.data,
            pk['r'].ctypes.data, pk['left'].ctypes.data,
            pk['right'].ctypes.data, pk['lu'].ctypes.data,
            pk['piv'].ctypes.data, pk['U'].ctypes.data,
            pk['V'].ctypes.data, pk['Y'].ctypes.data,
            pk['G'].ctypes.data, pk['Gpiv'].ctypes.data,
            pk['root'], n, x.ctypes.data_as(_ct.c_void_p),
            x.shape[1], pk['rmax'])
        return x.reshape(b.shape)

    def _native_solve_t(self, oi, b):
        pk = self._packed[oi]
        n = int(pk['m'][pk['root']])
        x = np.asfortranarray(b.reshape(n, -1))
        _nat.hodlr_solve_t(
            pk['kind'].ctypes.data, pk['m'].ctypes.data,
            pk['r'].ctypes.data, pk['left'].ctypes.data,
            pk['right'].ctypes.data, pk['lu'].ctypes.data,
            pk['piv'].ctypes.data, pk['U'].ctypes.data,
            pk['V'].ctypes.data, pk['Yt'].ctypes.data,
            pk['G'].ctypes.data, pk['Gpiv'].ctypes.data,
            float(np.real(self.jw)), float(np.imag(self.jw)),
            pk['root'], n, x.ctypes.data_as(_ct.c_void_p),
            x.shape[1], pk['rmax'])
        return x.reshape(b.shape)

    def _solve_node(self, nd, b):
        if nd.left is None:
            return lu_solve(nd.lu, b)
        m1 = nd.left.hi - nd.left.lo
        t = np.empty_like(b)
        t[:m1] = self._solve_node(nd.left, b[:m1])
        t[m1:] = self._solve_node(nd.right, b[m1:])
        r1 = nd.r
        if r1 == 0:
            return t
        Wt = np.empty((2*r1,) + b.shape[1:], np.complex128)
        Wt[:r1] = nd.V.T @ t[m1:]
        Wt[r1:] = nd.U.T @ t[:m1]
        return t - nd.Y @ lu_solve(nd.G, Wt)

    def _solve_node_t(self, nd, b):
        if nd.left is None:
            return lu_solve(nd.lu, b, trans=1)
        m1 = nd.left.hi - nd.left.lo
        t = np.empty_like(b)
        t[:m1] = self._solve_node_t(nd.left, b[:m1])
        t[m1:] = self._solve_node_t(nd.right, b[m1:])
        r1 = nd.r
        if r1 == 0:
            return t
        z = np.empty((2*r1,) + b.shape[1:], np.complex128)
        z[:r1] = self.jw*(nd.U.T @ t[:m1])
        z[r1:] = self.jw*(nd.V.T @ t[m1:])
        return t - nd.Yt @ lu_solve(nd.G, z, trans=1)

    def solve(self, b):
        """Apply Z^{-1} (at the factored frequency) to a branch vector in
        the solver's e|f|g ordering. Uses the native C tree walk
        (hodlr_native.c) when libhodlrnat.so is present; python fallback
        otherwise (identical results)."""
        x = np.empty_like(np.asarray(b, np.complex128))
        b = np.asarray(b, np.complex128)
        native = _nat is not None and getattr(self, '_packed', None)
        for o, t in enumerate(self.trees):
            if t is None:
                continue
            p = self.perms[o]
            if native and self._packed[o] is not None:
                x[p] = self._native_solve(o, b[p])
            else:
                x[p] = self._solve_node(t, b[p])
        return x

    def solve_t(self, b):
        """Apply Z^{-T} (exact transpose solve at the factored frequency);
        needed to serve exact 'T'/'C' sampling ops when the compressed Z
        appears inside another operator's matvec (e.g. the consistent
        Schur S_c). Roughly doubles the stored correction factors (Yt);
        requires factor(..., transpose=True)."""
        if not getattr(self, '_has_t', False):
            raise RuntimeError("factored with transpose=False; no Yt")
        x = np.empty_like(np.asarray(b, np.complex128))
        b = np.asarray(b, np.complex128)
        native = _nat is not None and getattr(self, '_packed', None)
        for o, t in enumerate(self.trees):
            if t is None:
                continue
            p = self.perms[o]
            if native and self._packed[o] is not None:
                x[p] = self._native_solve_t(o, b[p])
            else:
                x[p] = self._solve_node_t(t, b[p])
        return x
