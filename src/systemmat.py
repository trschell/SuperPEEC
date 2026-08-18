# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""``SystemMat`` -- the PEEC system operator and its preconditioners.

Extracted from main.py, which is a driver SCRIPT and therefore was not
importable: every test that needed SystemMat had to exec ~1300 lines of
main.py's head, and several ended up reimplementing pieces of it instead
(that is how SystemMat.precondinitLpPR came to have no coverage at all
while a validator's docstring claimed otherwise).

Import it directly:  from systemmat import SystemMat
"""
import os as _os
import numpy as np
import multipole as mp
import meshgraph as mg
import reluctance as rel
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr
from scipy.linalg import toeplitz
from scipy.linalg import lu_factor, lu_solve, cho_solve
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import cg
from scipy.sparse.csgraph import reverse_cuthill_mckee
import pyamg
from pyamg.krylov import gmres
from pyamg.krylov import fgmres
from pyamg.util.linalg import norm
import sksparse.cholmod as cholmod
from cvxopt import spmatrix, amd
from cvxopt import matrix as cvxmatrix
from scipy.sparse import dia_matrix, coo_matrix, issparse
from scipy.sparse.linalg import splu
import time

# Solver scale/config constants. These live here rather than in main.py
# because SystemMat is what reads them.
nfold = 1        # capacitance fold-in sweeps for the loop preconditioner
pscale = 1e+0    # potential-row scaling
bscale = 1e+0    # branch-row scaling
cscale = 1e+0    # capacitive-block scaling


def _dense_pext_fallback(M, ext):
    """Dense LU-able P_ext when the sparse Cholesky ``n2nchol`` is absent.

    REFUSES when the Cholesky failed because the operator is INDEFINITE.
    That failure is an assertion result, not a performance hint: the
    coefficient-of-potential matrix is a Gram matrix and is positive
    definite, so indefiniteness means the near-field truncation dropped
    too much. Falling back to an LU here would factor the very matrix
    just rejected -- ``Tree._factor_n2n`` warns in as many words that an
    LU "would succeed on an indefinite matrix and hide this" -- and hand
    back a preconditioner for an operator the surrounding algorithm
    assumes is SPD, visible only as poor convergence at large N with no
    diagnosis.

    A NON-indefinite failure (cholmod missing, say) is a different thing
    entirely: the matrix is fine and the dense LU is merely slower, so
    that case still falls back.
    """
    if getattr(M, 'n2nchol_indefinite', False):
        raise RuntimeError(
            "refusing to build a preconditioner from an indefinite P_ext. "
            "The near-field Cholesky failed as an ASSERTION and the dense "
            "LU fallback would factor exactly the same matrix, hiding it. "
            "Recorded reason: %s"
            % (getattr(M, 'n2nchol_error', None) or "(none recorded)"))
    if getattr(M, 'n2n', None) is None:
        raise RuntimeError(
            "this tree was built with keep_n2n=False: no stored n2n "
            "exists to factor. Use the band W (wsolveinit('band') "
            "before any preconditioner init; LpPRSolver wsolve='auto' "
            "does this when n2nchol is absent).")
    # n2n may be dense OR sparse depending on the assembly path.
    # np.asarray() on a scipy sparse matrix does NOT densify it -- it
    # returns a 0-d OBJECT array wrapping it -- and np.ndim() reports 2
    # for both, so sparsity has to be tested explicitly.
    if issparse(M.n2n):
        return np.asarray(M.n2n[ext][:, ext].toarray(), dtype=np.complex128)
    dense = np.asarray(M.n2n)
    if dense.ndim != 2:
        raise RuntimeError(
            "no P_ext factorisation available: n2nchol is None and this "
            "tree has neither a sparse nor a dense n2n to fall back on "
            "(ndim=%s). That combination means a CIRCULANT tree whose "
            "near-field Cholesky failed -- the circulant path deliberately "
            "never forms the dense n2n, which is the whole point at large "
            "N. Fix the factorisation rather than the fallback: see "
            "Tree._factor_n2n.%s"
            % (dense.ndim,
               (" Recorded reason: " + M.n2nchol_error)
               if getattr(M, 'n2nchol_error', None) else ""))
    return np.asarray(dense[np.ix_(ext, ext)], dtype=np.complex128)


def print_res(rk):
    # if S.numiters % 10 == 0 or S.numiters < 10:
    print(np.linalg.norm(rk))


class SystemMat:
    def __init__(self, M, jomega):
        self.M = M
        self.jomega = jomega
        # How the W rescale applies P_ext^{-1}. Default 'exact' (the
        # factored near-field P_ext); see :meth:`wsolveinit` for 'band'.
        self._wsolve = 'exact'
        self._Wband = None
        self.neumanniter = 2
        self.M.jomega = jomega
        self.esize = np.size(M.e.struc)
        self.fsize = np.size(M.f.struc)
        self.gsize = np.size(M.g.struc)
        self.nodesize = np.size(M.lv[0].struc)
        self.efsize = self.esize + self.fsize
        self.efgsize = self.efsize + self.gsize
        self.wholesize = self.efgsize + self.nodesize
        self.wholedata = np.zeros((self.wholesize,), dtype=np.complex128)
        self.M.e.data = self.wholedata[:self.esize]
        self.M.f.data = self.wholedata[self.esize:self.efsize]
        self.M.g.data = self.wholedata[self.efsize:self.efgsize]
        self.M.lv[0].data = self.wholedata[self.efgsize:self.wholesize]
        self.numiters = 0
        # Loop (mesh) basis and its transpose, used by matvecLpR. The
        # driver builds them after this object exists (they need esize/
        # efgsize), so they are attached then rather than constructed
        # here -- they used to be read as module GLOBALS, which is one of
        # the reasons this class could not be imported.
        self.Zmesh = None
        self.ZmeshT = None
        # External/internal node masks for the mixed-row LpPR node
        # equations (capacitive trees only; an LpR tree has no panels and
        # never calls matvecLpPR).
        if hasattr(M, 'external'):
            self.intmask = np.ones((self.nodesize,), dtype=bool)
            self.intmask[M.external] = False
            self.extmask = ~self.intmask
        self._Lp = None            # cached dense partial inductance

    def precondinitLpPR(self):
        """Build the block-triangular Schur preconditioner for matvecLpPR.

        Built for the W-RESCALED operator ``rescaleLpPR``, the same K' the
        other three preconditioners (reluctance, diagschur, loop) target:
            K' = [[Z, A], [A^T, -jw W PiE]],   W = (PiE P + PiJ)^-1,
        with Z = R + jw*Lp (impedance) and A the incidence.

        IT MUST NOT BE BUILT FROM THE RAW BLOCKS [[Z, A], [C, D]] with
        C = (PiE P + PiJ)A^T and D = -jw PiE. Those carry P ~ 1e14 against
        jw ~ 1e11 and condition at ~1e22 -- past double precision -- so the
        exact Schur complement still drives the PRECONDITIONED residual to
        1e-13 while the true solution is wrong in a near-null direction:
        measured 100% current error under 'cell' and 1e-6 under 'edge'.
        That is what this method used to do; ``validate_lppr_precond.py``
        now calls it directly and would catch a regression.

        The preconditioner is the block-upper-triangular
            P = [[Z, A], [0, S]],  S = -jw W PiE - A^T Z^{-1} A,
        for which K P^{-1} = [[I, 0], [C Z^{-1}, I]] has minimal polynomial of
        degree 2, so preconditioned GMRES converges in ~2 iterations
        independent of frequency. This targets the 100 MHz - 10 GHz band where
        Z is dense and far from diagonally dominant, so a diagonal
        preconditioner is useless.

        The dense partial inductance Lp, the incidence A, and the potential
        operator P are assembled once (Lp is frequency-independent and cached);
        each call refactors Z and the Schur complement S at the current
        ``jomega``. The dense Z factorization is the seam where an FMM/
        loop-space inner solve would go for O(N) scaling; here it is a direct
        LU, exact and cheap at present sizes and reused across a frequency
        sweep.
        """
        M = self.M
        efg = self.efgsize
        nn = self.nodesize
        self._Rdiag = np.concatenate([
            np.full(self.esize, M.e.r), np.full(self.fsize, M.f.r),
            np.full(self.gsize, M.g.r)]).astype(np.complex128)
        if self._Lp is None:
            jw_save = M.jomega
            M.jomega = 1j                 # probe traverseRL -> R + 1j*Lp
            Lp = np.zeros((efg, efg), dtype=np.complex128)
            for k in range(efg):
                self.wholedata[:efg] = 0
                self.wholedata[k] = 1.0
                M.traverseRL()
                Lp[:, k] = self.wholedata[:efg]
            Lp = (Lp - np.diag(self._Rdiag)) / 1j
            self._Lp = 0.5*(Lp + Lp.T)    # symmetrize (partial inductance)
            Amat = np.zeros((efg, nn), dtype=np.complex128)
            for k in range(nn):
                M.lv[0].data[:] = 0
                M.lv[0].data[k] = 1.0
                (ae, af, ag) = M.connectA()
                Amat[:, k] = np.concatenate([ae, af, ag])
            self._Amat = Amat
            self._P = np.asarray(M.n2n, dtype=np.complex128)
            self._CopA = ((np.diag(self.extmask.astype(np.complex128)) @ self._P
                          + np.diag(self.intmask.astype(np.complex128)))
                          @ Amat.T)      # C = (PiE P + PiJ) A^T (raw form)
            # W PiE, with W = (PiE P + PiJ)^-1: the capacitive block of the
            # W-RESCALED operator is -jw * this, so it is frequency
            # independent and can be cached alongside Lp.
            self._WPiE = np.linalg.solve(
                np.diag(self.extmask.astype(np.complex128)) @ self._P
                + np.diag(self.intmask.astype(np.complex128)),
                np.diag(self.extmask.astype(np.complex128)))
            self.wholedata[:] = 0
            M.jomega = jw_save
        jw = self.jomega
        Z = np.diag(self._Rdiag) + jw*self._Lp
        self._luZ = lu_factor(Z)
        ZinvA = lu_solve(self._luZ, self._Amat)
        # Schur complement of the W-RESCALED operator K' = rescaleLpPR,
        # i.e. node rows [A^T, -jw W PiE] -- NOT the raw [C, D]. See the
        # docstring: on the raw blocks this preconditioner drives the
        # PRECONDITIONED residual to 1e-13 while leaving the true solution
        # 100% wrong, because raw K conditions at ~1e22.
        S = -jw*self._WPiE - self._Amat.T @ ZinvA
        self._luS = lu_factor(S)

    def precondLpPR(self, v):
        """Apply the block-triangular Schur preconditioner P^{-1}.

        P^{-1}[b_i; b_v] = [Z^{-1}(b_i - A S^{-1} b_v); S^{-1} b_v].
        """
        efg = self.efgsize
        xv = lu_solve(self._luS, v[efg:])
        xi = lu_solve(self._luZ, v[:efg] - self._Amat @ xv)
        return np.concatenate([xi, xv])

    # ---- Loop-reduction (inductive-saddle) preconditioner --------------
    def loopprecinit(self, saddle_rtol=1e-6):
        """Build the scalable loop-reduction preconditioner for matvecLpPR.

        ``saddle_rtol`` is the INNER loop-solve tolerance, i.e. how
        accurate the preconditioner is; see :meth:`_saddle_solve` for the
        measurements behind the 1e-6 default (the old 1e-3 stagnated at
        the top of the band).

        Rescaling the external node rows of the mixed-row system by
        ``P_ext**-1`` turns it into the standard PEEC MNA
        ``[[Z, A], [A^T, -jw C]]`` with a tiny nodal capacitance
        ``C = P_ext**-1``; the inductive saddle ``[[Z, A], [A^T, 0]]`` then
        preconditions it, and that saddle is solved by the divergence-free
        loop reduction (loop basis ``B`` from ``getmesh_fortran`` + geometric
        Cholesky ``chol(B^T B)``) -- the same engine as the LpR solve. The
        preconditioner is iterative, so the outer solver must be flexible
        (FGMRES). Unlike the dense-LU block preconditioner
        (:meth:`precondinitLpPR`), every operator here is matrix-free/sparse,
        so it scales.
        """
        self.saddle_rtol = float(saddle_rtol)
        M = self.M
        adjmat = M.adjmats()
        Bm = mg.getmesh_fortran(adjmat, self.esize, self.efsize,
                                self.efgsize, self.nodesize)
        Bm.data = np.float64(Bm.data)
        self._Bm = Bm
        self._BmT = Bm.T.tocsc()
        self._BmT.data = np.float64(self._BmT.data)
        self._loopsize = Bm.shape[1]
        BmT32 = self._BmT.copy()
        BmT32.data = np.float32(BmT32.data)
        self._Bchol = cholmod.cholesky_AAt(BmT32, mode='supernodal',
                                           ordering_method='amd')
        # P_ext^{-1} for the W rescaling. Prefer the sparse Cholesky n2nchol
        # (the factored external block of the near-field coefficient-of-
        # potential n2n) -- scalable at multilevel, and exactly the P_ext this
        # otherwise factors densely. Fall back to a dense LU only if n2nchol is
        # unavailable (n2n[ext] not SPD).
        self._ext = M.external
        # A banded W (wsolveinit('band')) replaces _pext_solve outright, so
        # the dense fallback -- and the indefiniteness refusal inside it --
        # is neither needed nor wanted.
        if self._Wband is None and M.n2nchol is None:
            self._luPext = lu_factor(_dense_pext_fallback(M, self._ext))
        else:
            self._luPext = None
        # Ground an INTERNAL node to fix the potential gauge: W leaves internal
        # rows untouched, so the ground constraint is not mixed by the rescale.
        # Shell-like structures (e.g. the parallel-plate resonator of
        # validate_rlc_resonance.py) have NO internal nodes; any node fixes
        # the gauge, so fall back to the first external one.
        internal = np.nonzero(self.intmask)[0]
        self._gnd = int(internal[0]) if internal.size else int(self._ext[0])
        self._Aloop = LinearOperator((self._loopsize, self._loopsize),
                                     matvec=self._loopmatvec,
                                     dtype=np.complex128)
        self._Bcholop = LinearOperator((self._loopsize, self._loopsize),
                                       matvec=self._bcholapply,
                                       dtype=np.complex128)
        self._Bop = LinearOperator((self.nodesize, self.efgsize),
                                   matvec=self.connectB,
                                   rmatvec=self.connectBT, dtype=np.complex128)
        self._BTop = LinearOperator((self.efgsize, self.nodesize),
                                    matvec=self.connectBT,
                                    rmatvec=self.connectB, dtype=np.complex128)
        self._nfold = nfold      # capacitance fold-in sweeps (module-level)

    def wsolveinit(self, mode='exact', window=30, retain=18):
        """Choose how the W rescale applies ``P_ext^{-1}``.

        ``'exact'`` (default, unchanged) factors the near-field P_ext --
        the historical behaviour, and the reason LpPR did not scale: that
        factorisation is a CHOLESKY, and the truncated coefficient-of-
        potential matrix is positive definite ONLY when the near field
        spans essentially the whole domain (<~4 boxes per axis). Every
        corpus model above ~24k voxels fails it at every leaf size, so
        ``_mnaprecassembly`` refuses to build any preconditioner at all.

        ``'band'`` applies a SPARSE approximation instead, from
        :func:`reluctance.extract_ccap` -- the same windowed local
        inversion used for ``ccap='band'``, ~21 nnz/row. No factorisation,
        no definiteness requirement, O(n_ext) storage, and the apply is a
        sparse matvec.

        WHY THIS IS LEGITIMATE: the rescale is a ROW SCALING applied to
        BOTH K and the right-hand side (``rescaleLpPR`` / ``rescaleRHS``),
        so for ANY invertible W the rescaled system has the same solution
        as the original. W's quality shows up in CONDITIONING -- the size
        of the ``(WP - I)`` defect, hence the iteration count and the
        attainable residual -- never in the answer.

        Measured at 1 MHz, banded vs exact where both exist, and banded
        alone where the Cholesky refuses::

            model            unknowns   exact       band
            setup1 7^3           1225   6 matvecs   11, currents 6.7e-10
            setup3 nleaf 6        723   16          13, 2.1e-10
            setup3 nleaf 4        723   REFUSED     13, 9.7e-9
            straight_cond1      93200   REFUSED     19,   1.9 s, 0.88 GB
            circular_coil      277272   REFUSED     28,  33.3 s, 3.17 GB
            square_coil       2358320   REFUSED     40, 191.4 s, 22.5 GB

        (currents are vs a row-equilibrated dense direct solve of the same
        K; the large models have no dense reference, and were checked on
        the ORIGINAL operator instead -- branch equation
        ``|Zi+Av|/|terms|`` 5.3-6.0e-8, node rows 5e-10..1.4e-9, at every
        size.)

        CALL THIS BEFORE any ``*precinit``: the preconditioner assembly
        reads ``_pext_solve`` and skips its dense-LU fallback when a
        banded W is present.

        CAVEATS. ``window``/``retain`` were tuned for ``ccap`` on
        parallel-plate resonators, where the requirement is "good
        capacitance"; as a RESCALE the requirement is "``(WP-I)`` small",
        and that pairing has not been swept. All the validation above is
        at ONE frequency (1 MHz), deep in the resistive band -- the
        ``-jw C_cap`` term the rescale produces grows with frequency, so
        W's accuracy should matter more toward self-resonance and is
        untested there.
        """
        if mode not in ('exact', 'band'):
            raise ValueError("wsolve must be 'exact' or 'band', got %r"
                             % (mode,))
        if mode == 'exact':
            self._wsolve, self._Wband = 'exact', None
            return
        self._Wband = rel.extract_ccap(self.M, window=window,
                                       retain=retain).tocsr()
        self._Wband_params = (int(window), int(retain))
        self._wsolve = 'band'

    def _pext_solve(self, b):
        """Apply P_ext^{-1} to an external-node subvector. Uses the sparse
        Cholesky n2nchol when available -- a cholmod Factor at multilevel
        (scalable) or a numpy Cholesky L at single level -- factoring the real
        n2n[ext] block, so a complex LHS is solved by parts; else a dense LU.
        Under ``wsolveinit('band')`` it is a sparse matvec instead."""
        if self._Wband is not None:
            return self._Wband @ np.asarray(b, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        ch = self.M.n2nchol
        if ch is None:
            return lu_solve(self._luPext, b)
        if hasattr(ch, 'solve_A'):          # cholmod Factor (multilevel)
            return (np.asarray(ch.solve_A(b.real))
                    + 1j*np.asarray(ch.solve_A(b.imag)))
        L = np.asarray(ch)                  # numpy Cholesky L (single level)
        return (cho_solve((L, True), b.real)
                + 1j*cho_solve((L, True), b.imag))

    def _loopmatvec(self, y):
        # reduced inductive impedance  B^T Z B  (Z via the FMM traverseRL)
        self.wholedata[:self.efgsize] = self._Bm.dot(y)
        self.M.traverseRL()
        return self._BmT.dot(self.wholedata[:self.efgsize])

    def _bcholapply(self, y):
        # geometric loop preconditioner (B^T B)^{-1}, single precision, split
        return (np.float64(self._Bchol(np.float32(np.real(y))))
                + 1j*np.float64(self._Bchol(np.float32(np.imag(y)))))

    def _saddle_solve(self, ri, rv):
        """Solve the inductive saddle [[Z,A],[A^T,0]][i;V]=[ri;rv] by loop
        reduction; return (i, V). Iterative (lsqr + preconditioned lgmres).

        INNER TOLERANCE. ``saddle_rtol`` is the accuracy of the loop
        correction, i.e. of the PRECONDITIONER itself. It used to be
        1e-3, which is inconsistent with an outer target of 1e-8: FGMRES
        tolerates a varying preconditioner, but it cannot converge past
        the level at which that operator stops being self-consistent.
        Measured on setup1 at 10 GHz under the 'cell' scheme, outer
        residual and current error against a dense equilibrated direct
        solve:

            inner rtol   1e-3      1e-5      1e-7      1e-9
            resid(K')    8.3e-4    4.3e-5    1.3e-5    2.7e-4
            vs direct    1.3e-2    5.3e-4    1.6e-4    1.1e-3

        1e-3 STAGNATES -- and non-monotonically, so more outer iterations
        make it worse, which is the signature of an inconsistent apply
        rather than slow convergence. The outer iteration count also
        FALLS (58 -> 49) with the tighter inner solve, so the extra inner
        sweeps are partly paid for. Below ~1e-7 the neglected capacitance
        dominates and tightening further stops helping.
        """
        # particular current: A^T xhat = rv
        xhat = lsqr(self._Bop, rv, atol=1e-10, btol=1e-10)[0]
        self.wholedata[:self.efgsize] = xhat
        self.M.traverseRL()
        zxhat = self.wholedata[:self.efgsize].copy()
        # loop correction: B^T Z B loop = B^T (ri - Z xhat)
        Vm = self._BmT.dot(ri - zxhat)
        loop = lgmres(self._Aloop, Vm, M=self._Bcholop,
                      rtol=self.saddle_rtol, maxiter=200, inner_m=40)[0]
        i = self._Bm.dot(loop) + xhat
        # potentials: A V = ri - Z i
        self.wholedata[:self.efgsize] = i
        self.M.traverseRL()
        V = lsqr(self._BTop, ri - self.wholedata[:self.efgsize],
                 atol=1e-10, btol=1e-10)[0]
        return i, V

    def rescaleLpPR(self, vec):
        """W-rescaled, gauge-fixed mixed-row operator K' v.

        Left-multiplies the node-row block of the mixed-row operator K
        (matvecLpPR) by W = P_ext^{-1} on external rows, turning K into the
        standard PEEC MNA K' = [[Z, A], [A^T, -jw C]] (C = P_ext^{-1}, tiny),
        and pins the ground node (identity row) to fix the potential gauge.
        K' is well conditioned (the P ~ 1e16 scale disparity is removed), so an
        FGMRES residual tolerance controls the solution error directly. K' and
        the saddle preconditioner share the same 1-D null space (a global
        potential shift), so the singular system is left as-is -- FGMRES
        converges to a particular solution and the currents are gauge
        invariant."""
        y = np.asarray(self.matvecLpPR(vec)).copy()
        yv = y[self.efgsize:]
        yv[self._ext] = self._pext_solve(yv[self._ext])
        return y

    def rescaleRHS(self, source):
        """Apply the same W rescaling to a right-hand side."""
        b = np.asarray(source, dtype=np.complex128).copy()
        b[self.efgsize:][self._ext] = self._pext_solve(b[self.efgsize:][self._ext])
        return b

    def precondSaddle(self, vec):
        """Preconditioner for K': the inductive saddle solve (loop reduction).
        K' ~ [[Z,A],[A^T,0]], so the saddle inverse clusters its spectrum."""
        i, V = self._saddle_solve(vec[:self.efgsize], vec[self.efgsize:])
        return np.concatenate([i, V])

    def precondFoldin(self, vec):
        """Preconditioner for the full K' = [[Z,A],[A^T,-jw C_cap]] with the
        capacitance folded in by defect correction.

        The inductive saddle enforces A^T i = r_v; the full system needs
        A^T i - jw C_cap V = r_v, i.e. A^T i = r_v + jw C_cap V. So iterate:
        solve the saddle with the displacement current jw*C_cap*V (=
        jw*P_ext^{-1}*V on external nodes) added to the divergence RHS, feeding
        the updated V forward. The fixed point solves K' exactly; a few sweeps
        make it a much better preconditioner at high frequency, where the
        capacitance the plain saddle ignores is largest. nfold=1 reduces to
        :meth:`precondSaddle`.
        """
        ri = vec[:self.efgsize]
        rv0 = np.asarray(vec[self.efgsize:], dtype=np.complex128)
        V = np.zeros(self.nodesize, dtype=np.complex128)
        i = np.zeros(self.efgsize, dtype=np.complex128)
        for _ in range(max(1, self._nfold)):
            rv = rv0.copy()
            rv[self._ext] += self.jomega * self._pext_solve(V[self._ext])
            i, V = self._saddle_solve(ri, rv)
        return np.concatenate([i, V])

    # ---- Reluctance (fixed block-triangular) preconditioner --------------
    def reluctanceprecinit(self, ccap='diag', stil='auto'):
        """Build the FIXED sparse block-triangular reluctance preconditioner for
        the W-rescaled mixed-row system (:meth:`rescaleLpPR`).

        The reluctance K~ = Lp^{-1} (reluctance.py: windowed extract ~50 /
        retain ~27 per orientation, symmetrized) is localized and
        FREQUENCY-INDEPENDENT, so it yields a fixed linear preconditioner and
        ordinary GMRES applies (unlike the iterative loop-reduction saddle,
        which needs FGMRES). The naive (R,K) reformulation -- left-multiply only
        the branch rows by K~ -- fails: the mixed-row system is a saddle point
        (internal nodes carry no charge, a zero (2,2) block), so block-diagonal
        preconditioning clusters the (1,1) block (cond ~6) but not the saddle's
        null space (measured cond(MK) ~ 1e20). The cure is a saddle-aware but
        still fixed block-UPPER-triangular preconditioner of the rescaled MNA
        K' = [[Z, A], [A^T, -jw C_cap]]:

            x_v = S~^{-1} r_v ;   x_i = N_Z (r_i - A x_v).

        Since K~ Z ~ jw I (i.e. K~ ~ jw Z^{-1}), the consistent choices are
            N_Z = (1/jw) K~,   S~ = -jw C_cap - (1/jw) A^T K~ A,
        the nodal reluctance admittance. (The jw factors are essential -- the
        naive S~ = -jw C_cap - A^T K~ A converges the preconditioned residual to
        the WRONG currents.) K~ and A^T K~ A are frequency-independent and
        assembled once; only the scalar jw and the sparse factorization of S~
        change with frequency. No gauge fixing is applied: the -jw C_cap
        diagonal breaks the null space of the nodal reluctance Laplacian, so
        S~ is nonsingular as-is (see :meth:`_factorReluctanceSchur` for why
        pinning a node here would be actively harmful).

        Everything is sparse: K~ (~32 nnz/row), the incidence A, and the nodal
        S~ (factored by sparse LU). This is the high-frequency solver -- cond(MK)
        and the iteration count IMPROVE with frequency (setup1: ~27 outer iters
        at 10 GHz, best in the band), the opposite of the loop preconditioner.
        Below ~300 MHz it degrades (N_Z is a poor Z^{-1} once R dominates); use
        the loop preconditioner there.

        Parameters
        ----------
        ccap : {'diag', 'band', 'full'}
            Capacitance block of S~. 'diag' (default, scalable) uses
            diag(P_ext^{-1}) -- ample far below self-resonance, where the
            capacitance is a small perturbation. 'band' (scalable, the
            near-resonance tool) uses the sparse windowed inverse
            reluctance.extract_ccap (window=30/retain=18 from the
            profile_ccap_band.py study): within ~1.2x of 'full' iteration
            counts at ~19 nnz/row, S~ stays sparse, and the dense
            P_ext^{-1} probe is skipped. 'full' embeds the full dense
            P_ext^{-1} block (O(n_ext^2) storage, dense factor of S~): near
            a RESONANCE the off-diagonal capacitive coupling is what the
            preconditioner must balance against the inductive term, and the
            full block cuts iterations several-fold there (resonator
            validation: 16 vs ~40 at f_res); prefer 'band' unless the node
            count is small.
        stil : {'auto', 'splu', 'amg'}
            How the nodal Schur S~ is inverted (see
            :meth:`_factorReluctanceSchur`). 'auto' (default): whenever
            jomega is purely imaginary, factor the equivalent REAL symmetric
            system (S~ = j*(A^T K~ A / w - w*C_cap)) with the symmetric
            MMD_AT_PLUS_A ordering -- exact at every frequency and measured
            4.7x faster than the complex LU (41.1 -> 8.7 s at 23^3, which
            recurs PER FREQUENCY); falls back to the complex LU otherwise.
            'splu' forces the complex LU. 'amg' is EXPERIMENTAL: a fixed
            k-V-cycle pyamg iteration (still a fixed linear operator, so
            standard GMRES survives), with an empirical contraction guard;
            measured NOT competitive here (smoothed aggregation contracts
            only ~0.5-0.6 per cycle on the non-M-matrix A^T K~ A, the guard
            is unreliable near/above resonance, and the outer iteration
            count roughly doubles) -- kept for future multigrid work.
        """
        self._mnaprecassembly(ccap=ccap)
        # reluctance K~ (real, symmetric, sparse); frequency independent.
        self._Ktilde = rel.extract_reluctance(self.M).tocsr()
        # nodal reluctance admittance A^T K~ A (sparse, frequency independent).
        self._AtKA = (self._Ainc.T @ self._Ktilde @ self._Ainc).tocsc()
        self._stil = stil
        self._factorReluctanceSchur()

    def _mnaprecassembly(self, ccap='diag'):
        """Frequency-independent assembly shared by the FIXED sparse LpPR
        preconditioners (:meth:`reluctanceprecinit`, :meth:`diagschurprecinit`):
        P_ext^{-1} access for the W rescaling (sparse n2nchol preferred, dense
        LU fallback), the explicit sparse incidence A (two-probe parity build
        with one-hot fallback and sanity check), the capacitance block
        C_cap = P_ext^{-1} (diagonal by default, full dense block on
        ccap='full'), and the fallback gauge node."""
        M = self.M
        efg = self.efgsize
        nn = self.nodesize
        self._ext = M.external
        # P_ext^{-1} for the W rescaling and C_cap: prefer the sparse Cholesky
        # n2nchol (scalable at multilevel); dense LU fallback only if absent.
        if self._Wband is None and M.n2nchol is None:
            self._luPext = lu_factor(_dense_pext_fallback(M, self._ext))
        else:
            self._luPext = None
        # Explicit sparse incidence A (efg x nn). Built by Tree.incidence()
        # (the two-probe parity trick, with the one-hot fallback and the
        # per-row +beta/-beta sanity check) -- shared with the spanning-tree
        # particular solution in port_impedance, which needs the same
        # matrix. Kept as a Tree method because it is a property of the
        # tree, not of this preconditioner.
        self._Ainc = M.incidence()
        self.wholedata[:] = 0.0
        # C_cap = P_ext^{-1} on external nodes (the -jw capacitance block of
        # the rescaled MNA): diagonal approximation by default (scalable,
        # ample below self-resonance), full dense block on request (near
        # resonance; see the ccap parameter).
        self._Ccap = np.zeros(nn, dtype=np.complex128)
        self._CcapFull = None
        self._CcapBand = None
        if self._ext.size and ccap == 'band':
            # Sparse banded C_cap ~ P_ext^{-1} (reluctance.extract_ccap):
            # windowed inversion on the external-node surface, the
            # electrostatic twin of K~. The window/retain study
            # (profile_ccap_band.py, parallel-plate resonators 8x4x3 and
            # 16x8x3 through resonance) settled window=30/retain=18: within
            # ~1.2x of the dense ccap='full' iteration counts at ~19 nnz/row
            # (resonator at f_res: diag 40 -> band 17 vs full 16; larger
            # 16x8x3: diag 131 -> band 27 vs full 26), PD at every setting
            # tested (truncation strengthens the Maxwell matrix's diagonal
            # dominance), currents vs dense direct < 1e-5. The surface
            # analog of the volume 27-stencil is a ~3x3 patch: retain
            # saturates at 8-18, growing mildly with plate size (open-
            # surface screening is algebraic, not exponential -- rechecked
            # if geometries grow). Skips the dense P_ext^{-1} eye-probe of
            # the other paths; keeps S~ / S_d sparse (sdamg-compatible).
            if self._Wband is not None and \
                    getattr(self, '_Wband_params', None) == (30, 18):
                # SHARE the banded extraction with the W rescale
                # (2026-08-08): wsolveinit('band') and this branch ran
                # the SAME extract_ccap (same window/retain) twice --
                # identical windows, blocks, and batched solves. The W
                # is the ext-indexed band; C_cap is the same matrix
                # embedded into full node space. Halves the setup's
                # extraction time AND its transient memory.
                Cb = self._Wband.tocoo()
            else:
                Cb = rel.extract_ccap(M, window=30, retain=18).tocoo()
            self._CcapBand = coo_matrix(
                (np.real(Cb.data), (self._ext[Cb.row], self._ext[Cb.col])),
                shape=(nn, nn)).tocsr()
            self._Ccap = np.asarray(self._CcapBand.diagonal(),
                                    dtype=np.complex128)
        elif self._ext.size:
            Pinv_ext = np.asarray(self._pext_solve(np.eye(self._ext.size)))
            self._Ccap[self._ext] = np.real(np.diag(Pinv_ext))
            if ccap == 'full':
                self._CcapFull = np.zeros((nn, nn), dtype=np.complex128)
                self._CcapFull[np.ix_(self._ext, self._ext)] = Pinv_ext
        # fallback gauge node, used only if the plain S~ factorization fails
        # (prefer internal; shell-like structures have none -- any node works)
        internal = np.nonzero(self.intmask)[0]
        self._gnd = int(internal[0]) if internal.size else int(self._ext[0])

    def _factorReluctanceSchur(self):
        """Assemble and factor S~ = -jw C_cap - (1/jw) A^T K~ A at the current
        jomega, storing the solve as ``self._stil_solve``.

        S~ is factored AS-IS, with no gauge pinning: the -jw C_cap diagonal on
        the external nodes breaks the null space (global potential shift) of
        the nodal reluctance Laplacian A^T K~ A, so S~ is nonsingular. Do NOT
        pin a node and zero the matching residual entry in the apply -- that
        makes the whole preconditioner M SINGULAR along that node, and left-
        preconditioned GMRES then converges the preconditioned residual while
        the true solution stays wrong wherever the node carries capacitive
        physics. Discovered on the all-external-node parallel-plate resonator
        (validate_resonator_precond.py): with the old pin+zero scheme the
        currents were off by up to 140% at a converged 1e-8 preconditioned
        residual -- the same failure class as dropping the 1/jw factors. It
        went unnoticed on setup1 because the pinned node was INTERNAL there (a
        pure-continuity row with no capacitance). A pinned refactor is kept
        only as a fallback if the plain factorization fails (S~ can be
        numerically near-singular far below resonance, where jw*C_cap is
        ~1e-7 of the Laplacian term).

        Inversion routes (the ``stil`` parameter of
        :meth:`reluctanceprecinit`). All exact routes exploit jw being
        purely imaginary: with jw = j*w,

            S~ = -j*w*C_cap + j*(A^T K~ A)/w = j*Shat,
            Shat = (A^T K~ A)/w - w*C_cap   (REAL symmetric),

        so S~^{-1} b = -j * Shat^{-1} b, with complex b handled by real/imag
        parts. The DEFAULT factors the real Shat by sparse LU with the
        symmetric MMD_AT_PLUS_A ordering: real arithmetic quarters the
        flops and the symmetric ordering beats COLAMD on this 3-D volume
        graph -- measured 41.1 -> 8.7 s at 23^3 (and 58 -> 31 ms per
        apply), exact at every frequency including the indefinite
        near/above-resonance regime. The complex-LU route remains for a
        general (non-imaginary) jomega and as the pinned-fallback path.

        Symbolic reuse for the per-frequency factorization was tried
        (UMFPACK backend, 2026-07-28) and REVERTED. What the attempt
        measured, so it is not retried naively: the symbolic phase is
        NOT the cost (0.03 s of the 8.9 s factor at 23^3 -- numeric
        dominates); UMFPACK's multifrontal numeric alone was 3.1x
        faster than splu but its default adaptive iterative refinement
        makes the apply a VARYING operator (data-dependent stopping
        inside the solve), stagnating standard GMRES at ~1e-9 -- the
        fixedness lesson again, caught by validate_lppr_multilevel's
        tol=1e-10 canary -- and it is single-threaded. The accepted
        route to a faster S~ factor is a PARALLEL solver (e.g. PARDISO
        via pydiso, which brings a phase split anyway), not symbolic
        reuse. Also measured and rejected: reusing only the PERMUTATION
        under SuperLU permc_spec='NATURAL' (114 s, 13x WORSE -- partial
        pivoting without the fill ordering destroys sparsity), cholmod
        simplicial LDL' reuse (19 s, no BLAS-3), cholmod supernodal LL'
        (provably breaks down: Shat is ALWAYS indefinite, since the
        gauge vector v0 has A v0 = 0 exactly, so v0' Shat v0 < 0).

        The EXPERIMENTAL 'amg' route runs a fixed k-V-cycle pyamg
        Richardson iteration on Shat (stationary and linear in b, so the
        preconditioner M stays a fixed operator and standard GMRES
        survives), guarded by an empirical contraction test. Measured
        honestly: smoothed aggregation contracts only ~0.5-0.6 per cycle
        on this operator (A^T K~ A is NOT an M-matrix -- the reluctance
        stencil has mixed-sign off-diagonals), the random-probe guard
        misses low-dimensional indefinite modes near/above resonance, and
        the outer iteration count roughly doubles even below resonance --
        so it currently loses to the real-symmetric LU everywhere tested.
        Kept behind the flag for future multigrid work (energy-min
        aggregation, complex-shifted-Laplacian ideas).
        """
        jw = self.jomega
        nn = self.nodesize
        if self._CcapFull is not None:
            # dense path (ccap='full'): the capacitance block is dense
            Sd = -jw*self._CcapFull - self._AtKA.toarray()/jw
            luSd = lu_factor(Sd)
            self._stil_solve = lambda rv: lu_solve(luSd, rv)
            return
        stil = getattr(self, '_stil', 'auto')
        imag_jw = abs(jw.real) <= 1e-12*abs(jw)
        if imag_jw and stil != 'splu':
            w = float(jw.imag)
            CcapS = (np.real(self._CcapBand) if self._CcapBand is not None
                     else dia_matrix((np.real(self._Ccap), 0),
                                     shape=self._AtKA.shape))
            Shat = (np.real(self._AtKA)/w - w*CcapS).tocsc()
            if stil == 'amg':
                # EXPERIMENTAL (see docstring): guarded fixed-cycle AMG.
                ml = pyamg.smoothed_aggregation_solver(
                    Shat.tocsr(), B=np.ones((nn, 1)), max_coarse=100)
                cyc = ml.aspreconditioner(cycle='V')
                rngt = np.random.default_rng(0)
                bt = rngt.standard_normal(nn)
                nb = np.linalg.norm(bt)
                xt = cyc*bt
                r1 = bt - Shat*xt
                xt = xt + cyc*r1
                r2 = bt - Shat*xt
                rho = max(np.linalg.norm(r1)/nb,
                          np.linalg.norm(r2)/max(np.linalg.norm(r1), 1e-300))
                if rho < 0.85:
                    k = int(min(12,
                                max(2, np.ceil(np.log(1e-3)/np.log(rho)))))

                    def shat_solve(b, cyc=cyc, k=k):
                        x = cyc*b
                        for _ in range(k - 1):
                            x = x + cyc*(b - Shat*x)
                        return x

                    self._stil_solve = \
                        lambda rv: -1j*(shat_solve(np.real(rv))
                                        + 1j*shat_solve(np.imag(rv)))
                    return
                # cycle not solidly convergent: fall through to the exact
                # real-symmetric factorization below
            try:
                # ORDERING (measured 2026-08-08, 160^2 pdn boards):
                # MMD_AT_PLUS_A -- chosen on 23^3 solid cubes, where it
                # still wins (22.8 s vs MMD_ATA 41.8 s on an unbroken
                # 102k-node slab) -- is CATASTROPHIC on PERFORATED
                # planes: antipad/via/slot holes derail the greedy
                # minimum-degree elimination (708 s at 48k nodes, 2550
                # s at 99k, factor fill 661M vs 85M). MMD_ATA is
                # robust across all three graph classes (10/42/44 s);
                # cholmod-METIS is comparable (7.7/43/33 s) but needs
                # a new code path. Dielectrics were exonerated: they
                # only add nodes to the graph the FILIGREE poisons.
                luS = splu(Shat, permc_spec='MMD_ATA')
                self._stil_solve = \
                    lambda rv: -1j*(luS.solve(np.real(rv))
                                    + 1j*luS.solve(np.imag(rv)))
                return
            except RuntimeError:
                pass                    # singular: complex pinned path below
        CcapC = (self._CcapBand if self._CcapBand is not None
                 else dia_matrix((self._Ccap, 0), shape=self._AtKA.shape))
        Stil = (-jw*CcapC - self._AtKA/jw).tocsc()
        try:
            self._luStil = splu(Stil)
        except RuntimeError:                        # singular: pin the gauge
            Stil = Stil.tolil()
            g = self._gnd
            Stil.rows[g] = [g]; Stil.data[g] = [1.0]
            Stil = Stil.tocsc()
            Stil[:, g] = 0.0
            Stil[g, g] = 1.0
            self._luStil = splu(Stil.tocsc())
        self._stil_solve = self._luStil.solve

    def precondReluctance(self, vec):
        """Apply the fixed block-upper-triangular reluctance preconditioner
        M^{-1} to a residual of the W-rescaled system K'. The full residual is
        used untouched -- see :meth:`_factorReluctanceSchur` for why no entry
        may be zeroed here."""
        efg = self.efgsize
        ri = vec[:efg]
        xv = self._stil_solve(np.asarray(vec[efg:], dtype=np.complex128))
        xi = (self._Ktilde @ (ri - self._Ainc @ xv)) / self.jomega
        return np.concatenate([xi, xv])

    # ---- Diagonal-admittance Schur (low-frequency) preconditioner --------
    def diagschurprecinit(self, ccap='diag', sdsolve='splu'):
        """Build the FIXED sparse diagonal-admittance Schur preconditioner for
        the W-rescaled mixed-row system (:meth:`rescaleLpPR`) -- the LOW-
        frequency counterpart of :meth:`reluctanceprecinit`.

        Mathematical framework (after PyPEEC's sparse preconditioner; see
        docs/pypeec_comparison_whitepaper.pdf): approximate the dense branch
        impedance Z = R + jw Lp of the rescaled MNA
        K' = [[Z, A], [A^T, -jw C_cap]] by its DIAGONAL
        D = R + jw diag(Lp) (self partial inductances only), and the
        potential block by a diagonal C_cap (ccap='diag' -- exactly PyPEEC's
        second diagonalization; 'full' keeps the dense P_ext^{-1} block).
        The sparsified MNA M = [[D, A], [A^T, -jw C_cap]] is then inverted
        EXACTLY through the Schur complement with respect to the diagonal
        block,

            S_d = -jw C_cap - A^T D^{-1} A,

        a complex-symmetric nodal admittance Laplacian on the 7-point node
        graph -- far sparser than the reluctance S~ (whose A^T K~ A stencil
        couples next-nearest nodes), so the per-frequency factorization is
        much cheaper. The apply is the exact block LU of M (BOTH triangles,
        unlike the block-upper-triangular reluctance apply, because D^{-1}
        is free):

            x_v = S_d^{-1} (r_v - A^T D^{-1} r_i);  x_i = D^{-1} (r_i - A x_v).

        M is a FIXED linear operator, so STANDARD GMRES applies -- unlike the
        loop-reduction path, which buys the same low-frequency band with an
        ITERATIVE inner saddle solve (lsqr + lgmres per apply) under FGMRES.

        Regime: LOW frequency, wL/R <~ 1 (crossover ~1 GHz at 10-12.5 um
        cells). There R dominates the mutual inductive coupling that the
        diagonalization discards, so M ~ K' and the Krylov iterations absorb
        the leftover mutual terms. Once wL/R >> 1 the neglected coupling
        dominates and iterations grow: hand over to 'reluctance' there. The
        two fixed preconditioners share the same assembly
        (:meth:`_mnaprecassembly`) and bracket the band from opposite ends.

        On the uniform grid every filament of one orientation is congruent,
        so diag(Lp) takes exactly three values (e/f/g self partial
        inductances), measured here by three single-filament traverseRL
        probes at the unit probe frequency (entry = r + 1j*Lp_ii; the self
        term is near-field and exact at any tree depth).

        Parameters
        ----------
        ccap : {'diag', 'full'}
            Capacitance block of S_d; see :meth:`reluctanceprecinit`.
        sdsolve : {'splu', 'amg'}
            How S_d is inverted per frequency (see
            :meth:`_factorDiagSchur`). 'splu' (default): exact complex
            sparse LU, MMD_AT_PLUS_A ordering -- cheap on thin/sparse
            structures (quasi-2-D node graphs), O(n^2)-flop on solid
            volumes. 'amg': fixed k-V-cycle smoothed-aggregation
            iteration. Unlike the reluctance stil='amg' (which LOSES:
            A^T K~ A is not an M-matrix, SA contracts only ~0.5-0.6),
            S_d IS the SA-friendly case -- a 7-point graph Laplacian
            whose weights 1/(r + jw*l_self) have strictly positive real
            part at every frequency (real part exactly an M-matrix,
            measured; near-common phase on the uniform grid) -- and SA
            contracts at ~0.19-0.22/cycle flat in frequency AND size,
            replacing the O(n^2) per-frequency factorization with O(n)
            cycles (23^3: 2.4 s splu factor vs ~0.1 s SA setup + ~1e-2 s
            apply). A contraction guard (rho < 0.5) falls back to splu,
            so 'amg' is always safe to request.
        """
        M = self.M
        self._sdsolve = sdsolve
        self._mnaprecassembly(ccap=ccap)
        self._Rdiag = np.concatenate([
            np.full(self.esize, M.e.r), np.full(self.fsize, M.f.r),
            np.full(self.gsize, M.g.r)]).astype(np.complex128)
        jw_save = M.jomega
        M.jomega = 1j
        Lpd = np.empty(self.efgsize, dtype=np.float64)
        for off, sz, leaf_r in ((0, self.esize, M.e.r),
                                (self.esize, self.fsize, M.f.r),
                                (self.efsize, self.gsize, M.g.r)):
            if sz == 0:
                continue
            self.wholedata[:] = 0.0
            self.wholedata[off] = 1.0
            M.traverseRL()
            # the probe reads r + 1j*Lp_ii; SUBTRACT the probed
            # filament's r rather than taking the raw imaginary part,
            # because r is COMPLEX for dielectric and superconductor
            # models (Im(r) ~ -3e7 for a dielectric half-cell) and the
            # contamination is then multiplied by jw into D: measured
            # on 1-cell-thick plates over an FR4 gap (whose FIRST
            # g-filament is metal->dielectric), every z-branch
            # admittance was wrong by NINE ORDERS and fgmres diverged.
            # Thick plates dodged it only because their first
            # z-filament is metal-metal.
            r0 = complex(np.atleast_1d(leaf_r).ravel()[0])
            Lpd[off:off+sz] = np.imag(self.wholedata[off] - r0)
        self.wholedata[:] = 0.0
        M.jomega = jw_save
        self._Lpdiag = Lpd
        self._factorDiagSchur()

    def _factorDiagSchur(self):
        """Assemble and factor S_d = -jw C_cap - A^T D^{-1} A at the current
        jomega (D = R + jw diag(Lp)), storing the solve as ``self._sd_solve``
        and the branch admittances D^{-1} as ``self._ydiag``.

        Unlike the reluctance S~, the Laplacian weights 1/(r + jw*Lp_ii) are
        fully complex, so there is no real-symmetric reformulation; S_d is
        factored as a complex symmetric sparse LU with the symmetric
        MMD_AT_PLUS_A ordering. Cheap regardless: the pattern is the 7-point
        node graph. No gauge pinning, for the same reason as
        :meth:`_factorReluctanceSchur` (-jw C_cap breaks the null space of
        the nodal Laplacian, and pin+zero would make the preconditioner
        singular along that node); the pinned refactor is kept only as a
        fallback, since S_d approaches the singular resistive Laplacian
        -A^T diag(1/r) A as w -> 0.

        sdsolve='amg' (see :meth:`diagschurprecinit`) replaces the LU by a
        FIXED k-cycle smoothed-aggregation Richardson iteration on
        -S_d = A^T D^{-1} A + jw C_cap (positive-real-part form; B = the
        constant vector, which is the w -> 0 near-null space). Fixed k and
        stationary cycles keep the preconditioner a fixed LINEAR operator,
        so it is legal under left-preconditioned GMRES as well as the
        production right-preconditioned fgmres; k targets a 1e-3 inner
        contraction (k ~ 5 at the measured rho ~ 0.2). Guarded by the same
        empirical two-cycle contraction probe as the reluctance 'amg'
        path: rho >= 0.5 falls through to the exact splu, so a request
        for 'amg' can never make the solve wrong -- only the probe cost
        (~3 cycles) is wasted. The dense ccap='full' path always uses LU.
        """
        jw = self.jomega
        nn = self.nodesize
        self._ydiag = 1.0/(self._Rdiag + jw*self._Lpdiag)
        Yd = dia_matrix((self._ydiag, 0), shape=(self.efgsize, self.efgsize))
        AtYA = (self._Ainc.T @ (Yd @ self._Ainc)).tocsc()
        if self._CcapFull is not None:
            Sd = -jw*self._CcapFull - AtYA.toarray()
            luSd = lu_factor(Sd)
            self._sd_solve = lambda rv: lu_solve(luSd, rv)
            return
        CcapD = (self._CcapBand if getattr(self, '_CcapBand', None) is not None
                 else dia_matrix((self._Ccap, 0), shape=(nn, nn)))
        Sd = (-jw*CcapD - AtYA).tocsc()
        if getattr(self, '_sdsolve', 'splu') == 'amg':
            Spos = (-Sd).tocsr()
            # SYMMETRIC DIAGONAL EQUILIBRATION before aggregation
            # (2026-08-08): on dielectric boards S_d's weights span
            # ~10 orders (metal-region rows ~1/r_metal vs dielectric
            # rows ~|jw C_exc|), and smoothed aggregation across that
            # contrast stalls -- measured rho >= 0.5 on the filled
            # 320^2 pdn, guard falling back to splu. Scaling by
            # 1/sqrt(|diag|) normalizes row magnitudes; on homogeneous
            # conductors (the setup1 validator case) the scaling is
            # ~constant and nothing changes. The near-null space
            # (w -> 0 constant vector) maps to 1/scale.
            dg = np.asarray(Spos.diagonal())
            sc = 1.0/np.sqrt(np.maximum(np.abs(dg), 1e-300))
            Dsc = dia_matrix((sc, 0), shape=Spos.shape)
            Ssc = (Dsc @ Spos @ Dsc).tocsr()
            ml = pyamg.smoothed_aggregation_solver(
                Ssc, B=(1.0/sc)[:, None].astype(Ssc.dtype),
                max_coarse=100)
            cyc = ml.aspreconditioner(cycle='V')
            rngt = np.random.default_rng(0)
            bt = rngt.standard_normal(nn).astype(np.complex128)
            nb = np.linalg.norm(bt)
            xt = cyc*bt
            r1 = bt - Ssc*xt
            xt = xt + cyc*r1
            r2 = bt - Ssc*xt
            rho = max(np.linalg.norm(r1)/nb,
                      np.linalg.norm(r2)/max(np.linalg.norm(r1), 1e-300))
            if rho < 0.5:
                k = int(min(12, max(2, np.ceil(np.log(1e-3)/np.log(rho)))))
                if _os.environ.get('SPPEEC_GPU', '0') == '1':
                    # device-resident k-cycle apply (GPUAMG preserves
                    # the complex dtype); measured motivation: the S_d
                    # apply is ~21% of every LpPR iteration at 320^2
                    # (376 ms) and cuSPARSE-friendly. Same fixed
                    # linear operator as the CPU closure (legal under
                    # standard/right-preconditioned GMRES); permanent
                    # CPU fallback on any GPU failure.
                    try:
                        from gpu_amg import GPUAMG
                        g = GPUAMG(ml, cycles=k)
                        cp = g.cp

                        def sd_amg_gpu(b, g=g, cp=cp, sc=sc):
                            bs = sc*np.asarray(b, dtype=np.complex128)
                            x = g.solve(cp.asarray(bs))
                            return -(sc*cp.asnumpy(x))  # S_d = -Spos

                        self._sd_solve = sd_amg_gpu
                        return
                    except Exception:
                        pass                # fall through to CPU cycle

                def sd_amg(b, cyc=cyc, Ssc=Ssc, k=k, sc=sc):
                    bs = sc*np.asarray(b, dtype=np.complex128)
                    x = cyc*bs
                    for _ in range(k - 1):
                        x = x + cyc*(bs - Ssc*x)
                    return -(sc*x)          # S_d = -Spos, unscale
                self._sd_solve = sd_amg
                return
            # cycle not solidly convergent: exact splu below
        try:
            luSd = splu(Sd, permc_spec='MMD_AT_PLUS_A')
        except RuntimeError:                        # singular: pin the gauge
            Sd = Sd.tolil()
            g = self._gnd
            Sd.rows[g] = [g]; Sd.data[g] = [1.0]
            Sd = Sd.tocsc()
            Sd[:, g] = 0.0
            Sd[g, g] = 1.0
            luSd = splu(Sd.tocsc())
        self._sd_solve = luSd.solve

    def precondDiagSchur(self, vec):
        """Apply the diagonal-admittance Schur preconditioner M^{-1} -- the
        EXACT inverse of the diagonalized MNA [[D, A], [A^T, -jw C_cap]] --
        to a residual of the W-rescaled system K'. The full residual is used
        untouched (no gauge zeroing; see :meth:`_factorReluctanceSchur`)."""
        efg = self.efgsize
        ri = np.asarray(vec[:efg], dtype=np.complex128)
        xv = self._sd_solve(np.asarray(vec[efg:], dtype=np.complex128)
                            - self._Ainc.T @ (self._ydiag*ri))
        xi = self._ydiag*(ri - self._Ainc @ xv)
        return np.concatenate([xi, xv])

    def precondneumann(self, v):
        self.wholedata[:] = v
        output = v.copy()
        for i in range(self.neumanniter):
            (AVne, AVnf, AVng) = self.M.connectA()
            self.M.lv[0].data[:] = self.M.connectAT()
            self.M.diaginverse()
            self.M.traverseRL(neumann=True)
            self.wholedata[:] *= -1
            output[:] += self.wholedata
            self.M.e.data += AVne
            self.M.f.data += AVnf
            self.M.g.data += AVng
        self.wholedata[:] = output
        self.M.diaginverse()
        return self.wholedata

    def connectB(self, v):
        self.wholedata[:self.efgsize] = v
        self.M.lv[0].data[:] = self.M.connectAT()
        return self.wholedata[self.efgsize:]

    def connectBT(self, v):
        self.wholedata[self.efgsize:] = v
        (self.M.e.data[:], self.M.f.data[:], self.M.g.data[:]) = \
            self.M.connectA()
        return self.wholedata[:self.efgsize]

    def matvecLpPR(self, v):
        """Mixed-row single-P LpPR MNA matvec on [currents; potentials].

        Branch rows:        Z i + A v
        External node rows: P (At i) - jw v|ext   (P applied via traverseP3)
        Internal node rows: At i                  (plain KCL; P has zero
                                                   rows there, so the
                                                   P-multiplied equation
                                                   carries no information)
        Replaces the original (P^2 + alpha)-transformed node rows, which
        were mathematically consistent but conditioned at ~1e47; see
        docs/alphap_whitepaper.pdf. The -jw sign matches the original
        code's convention; it is empirically indistinguishable from +jw at
        low frequency and still needs a high-frequency validation case.
        """
        self.wholedata[:] = v
        Vn = self.M.lv[0].data.copy()
        (AVne, AVnf, AVng) = self.M.connectA()
        AtI = self.M.connectAT().copy()
        self.M.lv[0].data[:] = AtI
        self.M.traverseP3()
        node = self.M.lv[0].data.copy()
        node[self.intmask] = AtI[self.intmask]
        node[self.extmask] -= self.jomega * Vn[self.extmask]
        self.M.traverseRL()
        self.M.e.data += AVne
        self.M.f.data += AVnf
        self.M.g.data += AVng
        if getattr(self, 'dZ_near', None) is not None:
            # subpixel stages B/C: sparse partial-cell branch-
            # impedance correction (complex, frequency-tracked --
            # geometry dL plus the imposed-profile internal
            # impedance); the far field stays pure Toeplitz
            self.wholedata[:self.efgsize] += (
                self.dZ_near @ np.asarray(v[:self.efgsize]))
        self.M.lv[0].data[:] = node
        self.numiters += 1
        return self.wholedata

    def matvecLpR(self, v):
        # (AVne, AVnf, AVng) = self.M.connectA()
        print("performing matvec #", self.numiters)
        self.wholedata[:self.efgsize] = self.Zmesh.dot(v)
        self.M.traverseRL()
        ZtAZv = self.ZmeshT.dot(self.wholedata[:self.efgsize])
        self.numiters += 1
        return ZtAZv

    def matvecLpPR2(self, v):
        self.wholedata[:self.efgsize] = v
        self.M.lv[0].data[:] = self.M.connectAT()
        self.M.traverseP3()
        (APATIfe, APATIff, APATIfg) = self.M.connectA()
        self.M.traverseRL()
        self.M.e.data += 1/self.jomega*APATIfe
        self.M.f.data += 1/self.jomega*APATIff
        self.M.g.data += 1/self.jomega*APATIfg
        self.numiters += 1
        print(str(self.numiters) + " matvec(s) completed")
        return self.wholedata[:self.efgsize]


    def matvecLpPR3(self, v):
        self.M.e.data[:] = v[:self.esize]
        self.M.f.data[:] = v[self.esize:self.efsize]
        self.M.g.data[:] = v[self.efsize:self.efgsize]
        self.M.lv[0].data[:] = self.M.connectAT()
        self.M.lv[0].data[:] /= cscale
        self.M.traverseRL()
        self.M.traverseP3()
        self.M.lv[0].data[:] /= pscale
        self.M.lv[0].data[:] /= self.jomega
        (BPCe, BPCf, BPCg) = self.M.connectA()
        BPCe = self.M.e.data - 1/bscale*BPCe
        BPCf = self.M.f.data - 1/bscale*BPCf
        BPCg = self.M.g.data - 1/bscale*BPCg
        return np.concatenate([BPCe, BPCf, BPCg])

    def RHS2(self, Source):
        self.wholedata[:] = Source
        self.M.traverseP3()
        (APIne, APInf, APIng) = self.M.connectA()
        self.M.e.data += 1/self.jomega*APIne
        self.M.f.data += 1/self.jomega*APInf
        self.M.g.data += 1/self.jomega*APIng
