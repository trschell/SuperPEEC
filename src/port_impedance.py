# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Port impedance extraction from SuperPEEC's LpR (inductance) formulation.

Turns SuperPEEC from "solves for currents" into "reports R + jwL at a
port", which is what FastHenry and VoxHenry report and therefore what a
head-to-head comparison needs.

WHAT IS SOLVED
--------------
The LpR formulation is the mesh-current PEEC system. With ``A`` the
node-to-filament incidence, ``Z = R + jw*Lp`` the (dense, FMM-applied)
branch impedance and ``s_n`` the injected nodal current:

    A^T i = s_n                      (KCL, with port injection)
    Z i + A v = 0                    (branch equations, no series EMF)

Split ``i = i_hat + Y w`` where ``A^T i_hat = s_n`` is any particular
solution (least squares) and ``Y`` spans the loop space (``A^T Y = 0``,
from meshgraph). The loop rows give the square system

    (Y^T Z Y) w = -Y^T Z i_hat

which is what ``lgmres`` solves against a Cholesky preconditioner on
``Y^T Y``. Recovering ``v`` from ``A v = -Z i`` (least squares again)
gives the NODE POTENTIALS, and the port voltage follows. This mirrors
main.py's LpR branch exactly; it is reimplemented here rather than
imported because main.py is a driver script, not a module.

NEITHER LEAST-SQUARES PROJECTION IS NECESSARY, and on a large model they
cost more than everything else put together -- 65% of a square_coil
solve, against 17% for the FMM (``profile_solve_budget.py``). Both are
now avoided by default:

* ``i_hat`` is built directly on a SPANNING FOREST (the default,
  ``ihat_method='tree'``) in O(N) instead of ~1600 lsqr iterations,
  satisfying KCL to roundoff where lsqr stagnates at 1.6e-9 on
  square_coil. Measured there: 0.85 s against 77.6 s. It is also PURELY
  TOPOLOGICAL, so :meth:`LpRSolver.particular` caches it and one vector
  serves a whole frequency sweep. ``ihat_method='lsqr'`` restores the
  minimum-norm least-squares solution.
* ``v`` is never formed: the port voltage is the work-conjugate pairing
  ``w_k . v = i_hat_k . zi`` (see :func:`impedance_matrix`), so the
  second projection is skipped entirely. ``potentials=True`` restores
  the original route, and ``LpRSolver.solve`` still defaults to
  returning ``v`` for callers that want the potential field.

PORT MODEL, AND HOW IT DIFFERS FROM VOXHENRY
--------------------------------------------
VoxHenry drives a port with a unit VOLTAGE across an EQUIPOTENTIAL
terminal -- every panel on the terminal face is held at the same
potential and the current density distributes itself -- then assembles
Y column by column and reports ``Z = inv(Y)``.

SuperPEEC's LpR takes injected nodal CURRENT, so the natural port here is
the dual: a prescribed current PROFILE over the terminal, with the
potential free to vary across the face. Z comes out directly, one solve
per port, no inversion. For a one-port (12 of the 15 shipped VoxHenry
models) ``Z = 1/Y`` makes these identical in principle; for the
two-port files current injection with the other port left open is
exactly the open-circuit Z that ``inv(Y)`` produces.

They are NOT identical in practice, because the two impose different
boundary conditions at the terminal itself. Forcing a prescribed
current profile where the true solution wants a non-uniform one adds
spurious loss local to the port. The error is a terminal effect and
decays over a few cells (Saint-Venant), so it is small for a conductor
many cells long and large for one a couple of cells long.

That error is MEASURABLE, not merely arguable: :func:`impedance_matrix`
accepts ``weight='corner'`` (area-consistent bilinear lumping) or
``'uniform'`` (equal per distinct terminal node). If the two agree, the
extraction is insensitive to the profile and terminal lumping is not
the dominant error; if they diverge, it is. :func:`profile_sensitivity`
reports the spread, and the CLI prints it alongside every result.

The port voltage is the WORK-CONJUGATE average, not an arbitrary one:
with injection ``s = I*w`` (so ``sum(w) = 0`` across the port),

    V = w . v      and      Zport = V/I = (s . v)/I^2

which is the complex power divided by I^2. This is the pairing that
makes the corner weighting the consistent lumping of its own current
profile, and it is gauge invariant because ``sum(w) = 0`` -- the
constant null space of A drops out.

Usage::

    import vhr, port_impedance as pz
    m = vhr.read_vhr('VoxHenry/Input_files/wire_len50.0u_dia10.0u.vhr')
    M = m.build_tree()
    solver = pz.LpRSolver(M)
    Z = pz.impedance_matrix(m, M, solver, 2.5e9)
    print(pz.as_r_jl(Z, 2.5e9))

Command line::

    python3 port_impedance.py <file.vhr> [freq ...]

Run inside the toolbox.
"""

import os
import time
import warnings
# Thread defaults -- see sppeec_threads.py. The library defaults cost
# 6.3x on the FMM path (OpenBLAS spawning a dozen threads for each of
# many small gemv calls) and 1.8x even on the dense LpPR path, and the
# penalty is LARGEST on the coarse meshes the skin/corner studies use.
# The runtime call works whatever the import order; the environment
# block inside the module covers OMP and FFTW for anyone importing
# early.
try:
    import sppeec_threads as _spthreads
    _spthreads.enforce_blas()
except Exception:                    # tuning must never break a solve
    _spthreads = None

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, lsqr, lgmres, bicgstab
import sksparse.cholmod as cholmod
import meshgraph as mg
import vhr
import stencils as st

MU0 = 4e-7*np.pi


# fp32 campaign PHASE 2: the Krylov BASIS in single precision.
# Below this rtol the fp32 path cannot deliver and is refused (see
# krylov_solve). MEASURED on a 200k complex-symmetric stand-in:
# at rtol 1e-4 single and double are indistinguishable (8 matvecs
# both, true residual 5.207e-05 vs 5.206e-05); at rtol 1e-8 single
# STALLS at 3.96e-08 and burns 274 matvecs into the iteration cap.
# 1e-5 sits an order of magnitude above the measured floor.
_SINGLE_RTOL_FLOOR = 1e-5


def krylov_solve(Aop, rhs, Pop, method='lgmres', rtol=1e-10,
                 maxiter=30, inner_m=None, precision='auto'):
    """One preconditioned Krylov solve; returns ``(x, flag)``.

    The single entry point for every LpR-formulation outer solve
    (LpRSolver, EquiTerminalSolver, WireEddySolver, WireBondSolver;
    the LpPR path keeps its own fgmres -- different system, different
    trade-offs).

    DEFAULT: lgmres with a SMALL BASIS, ``inner_m=10`` (decided with
    the user 2026-08-13 from the R3 frequency sweep below). Krylov
    storage is the dominant solve-time residency at hero scale, and
    the basis size is the knob that sets it: lgmres keeps ``inner_m``
    inner vectors plus 2*outer_k augmentation vectors, bicgstab ~8.
    Measured on the DBC at R3, 5 frequencies, one persistent solver so
    only the method differs (mean matvecs relative to lgmres(30), and
    the projected Krylov residency at 1e9 cells / 235M loops):

        config       mean mv    Krylov GB   GB saved per % cost
        lgmres(30)     1.00x       135.4        --
        lgmres(10)     1.08x        60.2        9.4     <- DEFAULT
        lgmres(5)      1.23x        41.4        4.1
        bicgstab       1.29x        30.1        3.6

    inner_m=10 is the efficient point by 2.6x: 75 GB for 8% of the
    iterations (~40 min on a 7-9.5 h hero frequency point, against
    bicgstab's ~2.5 h). All four configurations agree on the physics
    -- L within 0.003%, R within 0.31% across the sweep.

    ``method='bicgstab'`` stays selectable for a HARD memory ceiling:
    it is the leanest at ~8 work vectors, and its cost is strongly
    frequency-dependent -- BEST of all configs at 1e5 Hz (135 vs 161
    matvecs) and worst at 3e7 (155 vs 91). It converges to the same
    answer (8 digits at rtol 1e-8); its residual is simply not
    monotone, so at loose rtol the error left behind is not shaped
    away from the impedance functional the way a minimal-residual
    method's is (measured: R within 0.83% at rtol 1e-4 vs lgmres's
    0.006%). On bicgstab failure (breakdown or budget) this falls
    back to lgmres automatically, warm-started from the bicgstab
    iterate when it is finite.

    ``maxiter``/``inner_m`` set the lgmres budget; the bicgstab
    iteration cap is the SAME matvec budget (maxiter*inner_m total,
    at 2 matvecs/iteration). Note the default maxiter rose to 30 as
    inner_m fell to 10, holding that budget at 300 matvecs.

    ``precision`` -- fp32 campaign phase 2, the KRYLOV BASIS dtype:

      'auto'   single when ``rtol >= 1e-5``, double otherwise (the
               default: engineering solves get the lean basis,
               oracle-grade ones keep full precision automatically)
      'single' force complex64 -- REFUSED below the floor, because
               fp32 cannot reach there: measured, rtol 1e-8 in
               complex64 stalls at 3.96e-08 after 274 matvecs
      'double' force complex128 (the pre-phase-2 behaviour)

    Only the SOLVER'S OWN storage changes. The operator and the
    preconditioner still compute in double internally -- we merely
    hand scipy complex64-typed wrappers, so its basis vectors, its
    Gram-Schmidt and its axpys run in single. That halves the
    dominant solve-time residency (at 1e9 cells the inner_m=10
    basis is ~60 GB double, ~30 GB single) and, because that work
    is memory-bandwidth-bound, it is expected to cut wall clock
    too -- the Krylov algebra was the single largest line in the
    1e9 projection. The solution is returned in complex128.
    """
    # BASIS SIZE TRACKS rtol (2026-08-13). inner_m=10 is the measured
    # efficient point AT ENGINEERING TOLERANCE; a small basis costs
    # more iterations the tighter the tolerance (measured on the DBC
    # at R3: rtol 1e-4 converges in 172 matvecs, rtol 1e-10 exhausts a
    # 600-matvec budget at inner_m=10). Oracle-grade solves are rare
    # and never memory-bound, so they get the wide basis back. Passing
    # inner_m explicitly overrides this entirely.
    if inner_m is None:
        if rtol >= _SINGLE_RTOL_FLOOR:
            inner_m = 10
        else:
            # hold the caller's matvec budget (maxiter*10) constant
            inner_m, maxiter = 30, max(1, int(round(maxiter/3.0)))
    single = (precision == 'single'
              or (precision == 'auto' and rtol >= _SINGLE_RTOL_FLOOR))
    if precision not in ('auto', 'single', 'double'):
        raise ValueError("precision must be 'auto', 'single' or "
                         "'double', got %r" % (precision,))
    if precision == 'single' and rtol < _SINGLE_RTOL_FLOOR:
        raise ValueError(
            "precision='single' with rtol=%g is below the measured "
            "fp32 floor (%g): the solve would stall short of "
            "tolerance and burn the whole iteration budget. Use "
            "'auto' (which picks double here) or loosen rtol."
            % (rtol, _SINGLE_RTOL_FLOOR))
    if single:
        n = Aop.shape[0]
        c64 = np.complex64
        Aop = LinearOperator(
            (n, n), matvec=lambda v, _A=Aop: np.asarray(
                _A.matvec(np.asarray(v, np.complex128)), c64),
            dtype=c64)
        Pop = LinearOperator(
            (n, n), matvec=lambda v, _P=Pop: np.asarray(
                _P.matvec(np.asarray(v, np.complex128)), c64),
            dtype=c64)
        rhs = np.asarray(rhs, c64)

    def _finish(x, flag):
        return np.asarray(x, np.complex128), flag

    if method == 'lgmres':
        return _finish(*lgmres(Aop, rhs, M=Pop, rtol=rtol,
                               maxiter=maxiter, inner_m=inner_m))
    if method != 'bicgstab':
        raise ValueError("method must be 'bicgstab' or 'lgmres', "
                         "got %r" % (method,))
    cap = max(1, (int(maxiter)*int(inner_m))//2)
    x, flag = bicgstab(Aop, rhs, M=Pop, rtol=rtol, maxiter=cap)
    if flag != 0:
        warnings.warn("bicgstab did not converge (flag %s); falling "
                      "back to lgmres" % (flag,))
        x0 = x if np.all(np.isfinite(x)) else None
        x, flag = lgmres(Aop, rhs, M=Pop, x0=x0, rtol=rtol,
                         maxiter=maxiter, inner_m=inner_m)
    return _finish(x, flag)



# Loop-preconditioner storage/apply precision (fp32 campaign PHASE 1,
# decided with the user 2026-08-13). float32 is safe BY CONSTRUCTION
# here: a preconditioner's precision moves iteration counts only, never
# the converged answer (it stays a fixed linear operator, so lgmres and
# bicgstab are both fine), and the loop Gram's entries are exactly
# {4, +-1} -- representable without loss. The tiny dense Schur LU stays
# float64 (tens-sized; exactness there is free). SPPEEC_PRECOND_FP64=1
# restores the old double-precision hierarchy for A/B measurement.
_PRECOND_DT = (np.float64 if os.environ.get('SPPEEC_PRECOND_FP64') == '1'
               else np.float32)


def _gpu_amg_wanted():
    """GPU-apply policy for the LpR AMG preconditioners (DECIDED with
    the user 2026-08-12): DEFAULT ON when the hardware works.

    SPPEEC_GPU unset -> AUTO: try the GPU, fall back to CPU quietly
    (one info line). '1' -> force, loud warning on failure. '0' ->
    off. Rationale: the GPU apply is a numerically different smoother
    (damped Jacobi vs Gauss-Seidel) whose answer difference at
    convergence measured 8 digits -- totally swamped by any sane rtol
    (engineering default 1e-4; FEKO ships 3e-3). The old always-off
    default protected byte anchors and capped-run reproducibility;
    validators now pin tight rtol explicitly instead, and end users
    get the 19x apply. LpPR is unaffected (different preconditioners;
    "GPU harms LpPR" stands).
    """
    return os.environ.get('SPPEEC_GPU', 'auto') not in ('0',)


class _AMGFactor:
    """Callable stand-in for a cholmod Factor, backed by AMG V-cycles.

    Same interface as ``cholmod.cholesky_AAt(...)``: call it on a float32
    vector, get a float32 vector back, so ``LpRSolver._precond`` needs no
    change. Used only with ``basis='overcomplete'``, where Y^T Y is
    SINGULAR and a Cholesky is impossible -- and unnecessary, because the
    over-complete Gram is well enough conditioned for AMG to be a good
    preconditioner where it is useless on the selected basis (measured:
    10-16 iters vs 265-369; see studies/gram_amg.py).
    """

    def __init__(self, YT, cycles=2, max_coarse=400):
        import pyamg
        G = (YT @ YT.T).tocsr().astype(_PRECOND_DT)
        self._dt = G.dtype
        self.ml = pyamg.smoothed_aggregation_solver(G, max_coarse=max_coarse)
        self.cycles = int(cycles)
        self.nnz_ratio = sum(l.A.nnz for l in self.ml.levels)/G.nnz
        self.nnz = sum(l.A.nnz for l in self.ml.levels)
        # Opt-in GPU apply (SPPEEC_GPU=1): device-resident hierarchy,
        # damped-Jacobi smoother in place of Gauss-Seidel -- a DIFFERENT
        # (parallel) preconditioner, same solved answers; judged by
        # end-to-end wall clock. Failure is a warning, not fatal.
        self._gpu = None
        if _gpu_amg_wanted():
            try:
                from gpu_amg import GPUPlainAMG
                self._gpu = GPUPlainAMG(self)
            except Exception as exc:
                if os.environ.get('SPPEEC_GPU') == '1':
                    warnings.warn("SPPEEC_GPU=1 but GPU AMG setup "
                                  "failed (%s: %s) -- using the CPU "
                                  "apply" % (type(exc).__name__, exc))
                # auto mode: CPU fallback is normal on GPU-less boxes

    def __call__(self, b):
        if self._gpu is not None:
            return self._gpu(b)
        x = np.zeros(b.shape[0], dtype=self._dt)
        b = np.asarray(b, dtype=self._dt)
        for _ in range(self.cycles):
            x = self.ml.solve(b, x0=x, tol=1e-14, maxiter=1,
                              cycle='V')
        return np.float32(x)


class _BlockAMGFactor:
    """AMG on the large LOCAL block, exact Schur on the small LONG-cycle one.

    Plain AMG dies when the loop basis contains long cycles: they put
    dense rows into Y^T Y and wreck coarsening. Measured in equiterminal,
    whose 158 port cycles took the hierarchy from 1.2x to 3.0x nnz and
    stopped it converging at all, while the SAME over-complete basis
    without them ran 43-71 matvecs in port_impedance.

    But the long cycles are always FEW -- 158 of ~10000 there, 2 of 198
    on setup3's macro loops -- so split them off and treat them exactly:

        G = [A  B]      A = local (plaquettes), AMG
            [B' C]      C = macro (long cycles), tiny and dense-ish
        S = C - B' A^-1 B                      (n_macro x n_macro, dense)

    then the exact block LDU. Setup costs n_macro applications of A^-1
    (one per column of B), done once. Same float32 callable interface as
    a cholmod Factor, so ``_precond`` is unchanged.
    """

    def __init__(self, YT, macro_idx, cycles=4, max_coarse=400,
                 cycle_type='V', smoother=None, strength=None):
        import pyamg
        from scipy.linalg import lu_factor, lu_solve
        self._lu_solve = lu_solve
        G = (YT @ YT.T).tocsr().astype(_PRECOND_DT)
        self._dt = G.dtype
        n = G.shape[0]
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(macro_idx, dtype=int)] = True
        self.loc = np.flatnonzero(~mask)
        self.mac = np.flatnonzero(mask)
        self.nmac = self.mac.size
        A = G[self.loc][:, self.loc].tocsr()
        self.B = G[self.loc][:, self.mac].tocsr()
        C = G[self.mac][:, self.mac].toarray()
        _sm = {} if smoother is None else dict(presmoother=smoother,
                                               postsmoother=smoother)
        if strength is not None:
            _sm['strength'] = strength
        self.ml = pyamg.smoothed_aggregation_solver(A, max_coarse=max_coarse,
                                                    **_sm)
        self.smoother = smoother
        self.cycles = int(cycles)
        self.cycle_type = str(cycle_type)
        self.nnz = sum(l.A.nnz for l in self.ml.levels)
        self.nnz_ratio = self.nnz/max(G.nnz, 1)
        AiB = np.column_stack([self._A(self.B[:, j].toarray().ravel())
                               for j in range(self.nmac)]) \
            if self.nmac else np.zeros((self.loc.size, 0))
        self.S = lu_factor(np.float64(C - self.B.T @ AiB)) \
            if self.nmac else None
        self.n = n
        # Opt-in GPU apply (SPPEEC_GPU=1): device-resident hierarchy,
        # damped-Jacobi smoother in place of Gauss-Seidel -- a DIFFERENT
        # (parallel) preconditioner, same solved answers; judged by
        # end-to-end wall clock. Failure here is a warning, not fatal.
        self._gpu = None
        if _gpu_amg_wanted():
            try:
                from gpu_amg import GPUBlockAMG
                self._gpu = GPUBlockAMG(self)
            except Exception as exc:
                if os.environ.get('SPPEEC_GPU') == '1':
                    import warnings
                    warnings.warn("SPPEEC_GPU=1 but GPU AMG setup "
                                  "failed (%s: %s) -- using the CPU "
                                  "apply" % (type(exc).__name__, exc))
                # auto mode: CPU fallback is normal on GPU-less boxes

    def _A(self, r):
        x = np.zeros(r.shape[0], dtype=self._dt)
        r = np.asarray(r, dtype=self._dt)
        for _ in range(self.cycles):
            x = self.ml.solve(r, x0=x, tol=1e-14, maxiter=1,
                              cycle=self.cycle_type)
        return x

    def __call__(self, b):
        if self._gpu is not None:
            return self._gpu(b)
        b = np.asarray(b, dtype=self._dt)
        rp, rm = b[self.loc], b[self.mac]
        yp = self._A(rp)
        out = np.empty(self.n, dtype=self._dt)
        if self.nmac:
            ym = self._lu_solve(self.S, np.float64(rm - self.B.T @ yp))
            out[self.loc] = yp - self._A(self.B @ ym.astype(self._dt))
            out[self.mac] = ym
        else:
            out[self.loc] = yp
        return np.float32(out)


class _GeoMGFactor:
    """GEOMETRIC multigrid on the plaquette block, exact Schur on the rest.

    Drop-in for :class:`_BlockAMGFactor`: same float32 callable
    interface, so ``_precond`` and ``EquiTerminalSolver`` need no
    change. The difference is what inverts the large block --
    :class:`loopmg.GeoMG` (lattice agglomeration, unsmoothed Galerkin,
    damped Jacobi) instead of smoothed-aggregation AMG.

    THE SPLIT IS DIFFERENT, deliberately, and it is not a free choice:
    only the PLAQUETTE columns of Y are lattice faces with a recoverable
    (normal, i, j, k). Hole cycles (one per handle of a multiply
    connected conductor -- 402 on the pdn board), port cycles and
    redistribution modes have no geometry at all, and a geometric
    hierarchy cannot place them. They therefore ALL go into the exact
    Schur block, where _BlockAMGFactor put only the port cycles. That
    costs one A^-1 apply per macro column at setup, which is why the
    setup scales with the conductor's HOMOLOGY rather than its size.

    Trade-off vs AMG measured on a solid block (studies/geomg.py):
    worse per-iteration reduction (0.62 vs 0.41) but a ~4x cheaper
    apply, netting ~1.7x faster to tolerance at 19% less stored
    hierarchy. Irregular occupancy is the case where AMG is expected to
    defend itself, which is what the end-to-end A/B exists to settle.
    """

    def __init__(self, YT, geom_normal, geom_base, nplaq, cycles=2,
                 nu=2, omega=2.0/3.0, max_coarse=400):
        from scipy.linalg import lu_factor, lu_solve
        import loopmg
        self._lu_solve = lu_solve
        G = (YT @ YT.T).tocsr().astype(_PRECOND_DT)
        self._dt = G.dtype
        n = G.shape[0]
        self.loc = np.arange(min(nplaq, n))
        self.mac = np.arange(self.loc.size, n)
        self.nmac = self.mac.size
        A = G[self.loc][:, self.loc].tocsr()
        self.B = G[self.loc][:, self.mac].tocsr()
        C = G[self.mac][:, self.mac].toarray()
        self.mg = loopmg.GeoMG(A, geom_normal, geom_base, nu=nu,
                               omega=omega, max_coarse=max_coarse)
        self.cycles = int(cycles)
        self.nnz = self.mg.nnz
        self.nnz_ratio = self.nnz/max(G.nnz, 1)
        AiB = np.column_stack([self._A(self.B[:, j].toarray().ravel())
                               for j in range(self.nmac)]) \
            if self.nmac else np.zeros((self.loc.size, 0))
        self.S = lu_factor(np.float64(C - self.B.T @ AiB)) \
            if self.nmac else None
        self.n = n
        # GPU apply, same auto policy as the AMG factors (the
        # remedy the GeoMG adoption was conditioned on, 2026-08-12).
        # gpu_state answers "did the GPU apply actually engage, and
        # where does the hierarchy live" -- the AUTO policy's quiet
        # CPU fallback is otherwise invisible in production runs.
        self._gpu = None
        self.gpu_state = 'cpu (SPPEEC_GPU=0)'
        if _gpu_amg_wanted():
            try:
                from gpu_amg import GPUGeoBlock
                self._gpu = GPUGeoBlock(self)
                self.gpu_state = 'gpu ' + self._gpu.core.placement
            except Exception as exc:
                self.gpu_state = ('cpu fallback (%s: %s)'
                                  % (type(exc).__name__, exc))
                if os.environ.get('SPPEEC_GPU') == '1':
                    warnings.warn("SPPEEC_GPU=1 but GPU GeoMG setup "
                                  "failed (%s: %s) -- CPU apply"
                                  % (type(exc).__name__, exc))

    def _A(self, r):
        return self.mg(r, cycles=self.cycles)

    def __call__(self, b):
        if self._gpu is not None:
            return self._gpu(b)
        b = np.asarray(b, dtype=self._dt)
        rp, rm = b[self.loc], b[self.mac]
        yp = self._A(rp)
        out = np.empty(self.n, dtype=self._dt)
        if self.nmac:
            ym = self._lu_solve(self.S, np.float64(rm - self.B.T @ yp))
            out[self.loc] = yp - self._A(self.B @ ym.astype(self._dt))
            out[self.mac] = ym
        else:
            out[self.loc] = yp
        return np.float32(out)


class LpRSolver:
    """Frequency-independent setup for the LpR mesh solve.

    The loop basis ``Y`` (from the tree's adjacency) and the Cholesky
    factor of ``Y^T Y`` used to precondition it are both purely
    topological, so they are built ONCE here and reused across a
    frequency sweep. Only ``traverseRL`` -- which reads ``M.jomega`` --
    is frequency dependent.

    Parameters
    ----------
    M : multipole.Tree
        Must already have its data buffer allocated (``vhr.allocate`` or
        ``VhrModel.prepare``).
    verbose : bool, optional
    chol_mode : {'simplicial', 'supernodal', 'auto'}, optional
        CHOLMOD factorization mode. Default 'simplicial' -- measured 38%
        less factor memory and 3x faster (single core) than the
        'supernodal' this used to hard-code; see the table at the call
        site. Supernodal is the one that can use threaded BLAS, so
        revisit if you become time-bound rather than memory-bound.
    chol_ordering : optional
        CHOLMOD fill-reducing ordering: one of 'natural', 'amd',
        'metis', 'nesdis', 'colamd', 'default', 'best'. Default 'metis'
        (was 'amd', which measured worst). Ignored when
        ``basis='overcomplete'``.
    basis : {'auto', 'selected', 'overcomplete'}, optional
        Which cycle basis spans the loop space.

        'auto' (DEFAULT since 2026-08-05) takes 'overcomplete' and falls
        back to 'selected' only where the plaquettes cannot span the
        cycle space, warning when it does. The realised choice is on
        ``self.basis``, and ``self.basis_fallback`` records why if it
        fell back. Over-complete is preferred because it is faster AND
        lighter end to end on the large model -- square_coil, 604800
        voxels: 225.3 s / 2.92 GB against 271.6 s / 5.38 GB, despite
        needing MORE iterations (49 vs 42), because ``getmesh_fortran``
        plus the exact Cholesky cost 113 s of setup against 5.5 s. Note
        this REVERSES the older 590 s vs 630 s ranking: removing the two
        lsqr projections took a large fixed cost off both totals and
        promoted setup to the deciding term. ``getmesh_fortran`` also
        STALLS on coil topologies (>135 s and still running on
        circular_coil, against 0.5 s for ``getmesh_full``).

        'selected' is the historical one: ``getmesh_fortran``
        emits an INDEPENDENT spanning set (one quad per non-tree edge),
        preconditioned by an exact Cholesky of Y^T Y. Still required for
        topologies needing macro loops no square can reach -- vias and
        PCB stacks, e.g. setup3, where 196 plaquettes must cover a cycle
        rank of 198.

        'overcomplete' keeps EVERY plaquette (``getmesh_full``). Y^T Y is
        then singular -- its kernel is the cube boundaries, which are
        physically meaningless because currents differing by a cube
        boundary are the same current -- but singular-and-CONSISTENT is
        fine for lgmres, and the Gram is far better conditioned on its
        range. Measured on the 24k-voxel bar (studies/oc_solve.py):
        identical R and L to 7 significant figures at 1e8/2.5e9/1e10 Hz,
        the same matvec count at 2.5-10 GHz, and **12 MB of AMG hierarchy
        against 121 MB of Cholesky fill**. Costs 1.47x more unknowns and,
        at this size, more wall-clock -- it is a MEMORY win, and the
        advantage grows with problem size because the Cholesky fill grows
        superlinearly (27.7x here) while AMG stays ~1.2x.
    amg_cycles : int, optional
        AMG cycles per application of A^-1 when ``basis='overcomplete'``;
        the block preconditioner applies A^-1 TWICE, so the cost is
        ``2*amg_cycles`` cycles per preconditioner call. Default 4.

        TUNED, not guessed (equiterminal, wire_len50.0u at 10 GHz):
            n:        2     3     4     6     8    12
            matvecs 195   150   131   114   112   118
            solve s  13.7  11.2  11.0  11.1  11.5  16.1
        The optimum is BROAD (3-8 within 5%) and n=2 -- the original
        default -- was 25% off it. More cycles helps because each outer
        matvec costs a full ``apply_Z`` (FMM + Toeplitz near field),
        which dominates the AMG work: buying fewer outer iterations with
        more preconditioner effort is a straight win until the AMG cost
        overtakes it past n~8. At n=8 the count (112) nearly matches the
        exact Cholesky's 107.
    smoother : optional
        pyamg (pre|post)smoother spec, e.g. ``('chebyshev', {'degree': 4,
        'iterations': 1})``. Default None = pyamg's own (symmetric block
        Gauss-Seidel).

        SWEPT, and the smoother BARELY MATTERS here (matvecs / solve s,
        equiterminal wire_len50.0u at 10 GHz):
            smoother            n=4        n=6        n=8
            default block GS  131/10.15  114/10.39  112/11.48
            gauss_seidel sym  131/10.62  114/10.63  112/11.83
            chebyshev deg4    125/10.11  118/11.28  117/12.16
            chebyshev deg2    311=CAP    281/27.96  206/18.21
            jacobi w=4/3      311=CAP    284/23.07  231/20.73
            schwarz           119/20.48  116/24.79  116/29.98
            gauss_seidel_ne   311=CAP    311=CAP    281/41.48
        The good ones (block GS, GS, Chebyshev-4) are interchangeable
        within run-to-run noise (~8%: the SAME default config measured
        10.98 s and 10.15 s on two runs). Weak smoothers (Chebyshev-2,
        Jacobi) fail to converge at n=4; `gauss_seidel_ne` is a
        normal-equations smoother and simply wrong for an SPD operator;
        Schwarz gives the FEWEST iterations (116) but 3x the per-iteration
        cost and 4.6-8.1 s of setup, so it loses badly on wall-clock.
        CONCLUSION: keep the default. Cycle COUNT is the knob that
        matters, because ``apply_Z`` dominates -- not the smoother.
    strength : optional
        pyamg strength-of-connection spec. Default None = pyamg's own
        (symmetric, theta=0).

        SWEPT because it had a PRINCIPLED reason to help: G has 25%
        POSITIVE off-diagonals (87008 of 348320 on the 24k bar) and
        standard strength measures assume diffusion-like negative
        couplings. It does work mechanically -- and does not pay:
            strength              matvecs
            default sym theta=0     131
            sym theta=0.25          132
            sym theta=0.50          146
            evolution eps=2/4       122   <- deterministic, both eps
            algebraic_distance      122
        `evolution` reliably saves ~7% of ITERATIONS (122 vs 131, and
        `algebraic_distance` independently agrees at 122), but a clean
        alternating A/B of 3 reps gives min wall-clock 10.62 s vs the
        default's 10.33 s -- the richer measure builds a costlier
        hierarchy that eats the saving. KEEP THE DEFAULT.
    cycle_type : {'V', 'W', 'F'}, optional
        AMG cycle shape. Default 'V'. W and F were measured and buy
        essentially NOTHING here (149 iterations vs V's 150) while
        costing more work per cycle, so V wins on wall-clock at every
        cycle count.
    """

    def __init__(self, M, verbose=False, chol_mode='simplicial',
                 chol_ordering='metis', basis='auto', amg_cycles=4):
        self.M = M
        self.verbose = verbose
        self.basis = str(basis)
        self.amg_cycles = int(amg_cycles)
        self.chol_mode = chol_mode
        self.chol_ordering = chol_ordering
        self.esize = np.size(M.e.struc)
        self.fsize = np.size(M.f.struc)
        self.gsize = np.size(M.g.struc)
        self.nodesize = np.size(M.lv[0].struc)
        self.efgsize = self.esize + self.fsize + self.gsize
        self.whole = M._vhr_whole
        if self.whole is None or self.whole.size != self.efgsize + \
                self.nodesize:
            raise RuntimeError(
                "tree buffer not allocated -- call VhrModel.prepare(M, f) "
                "or vhr.allocate(M) before constructing LpRSolver")
        t0 = time.perf_counter()
        adjmat = M.adjmats()
        if self.basis not in ('auto', 'selected', 'overcomplete'):
            raise ValueError("basis must be 'auto', 'selected' or "
                             "'overcomplete', got %r" % (basis,))
        if self.basis == 'auto':
            # Prefer the over-complete plaquette basis and fall back only
            # where it cannot represent the cycle space. Measured on
            # square_coil (604800 voxels), end to end:
            #     selected     113.0 s setup + 150.2 s solve = 271.6 s,
            #                  5.38 GB, 42 matvecs
            #     overcomplete   5.5 s setup + 211.5 s solve = 225.3 s,
            #                  2.92 GB, 49 matvecs
            # i.e. 1.21x faster and 1.8x less memory despite needing MORE
            # iterations, because getmesh_fortran + the exact Cholesky
            # dominate the setup. (This REVERSES the older 590 s vs 630 s
            # ranking: eliminating the two lsqr projections removed a big
            # fixed cost from both totals, which promoted the setup to the
            # deciding term.) getmesh_fortran also STALLS on coil
            # topologies -- >135 s and still running on circular_coil,
            # against 0.5 s for getmesh_full.
            self.basis = 'overcomplete'
            auto = True
        else:
            auto = False
        self.basis_fallback = None
        if self.basis == 'overcomplete':
            self.Y = mg.getmesh_full(adjmat, self.esize,
                                     self.esize + self.fsize,
                                     self.efgsize, self.nodesize)
            # PLAQUETTES DO NOT ALWAYS SPAN THE CYCLE SPACE. Topologies
            # needing macro loops that no square can reach leave a
            # deficit -- setup3: 196 plaquettes for a cycle rank of 198,
            # which is exactly why getmesh_fortran falls back to MST
            # there. A deficient basis solves in a SUBSPACE and can
            # return a silently wrong answer; on setup3 it happened to
            # give the right one only because the two unreachable loops
            # carry no current for that excitation. Fail loudly instead.
            #
            # NOTE this compares COLUMN COUNT, which is NECESSARY but not
            # SUFFICIENT: a basis with enough columns could still be rank
            # deficient. The sufficient fix -- augmenting the plaquettes
            # with just enough MST macro cycles to close the deficit --
            # is NOT implemented.
            import meshgraph_aux as _mga
            _adjs = (adjmat + adjmat.T).tocsr()
            _rank = (self.efgsize - self.nodesize
                     + _mga.counttrees(_adjs.indices, _adjs.indptr))
            if self.Y.shape[1] < _rank and not auto:
                raise RuntimeError(
                    "over-complete plaquette basis spans only %d columns "
                    "for a cycle space of %d -- this geometry needs macro "
                    "loops no square can reach (getmesh_fortran falls back "
                    "to MST here). Use basis='selected' or 'auto'."
                    % (self.Y.shape[1], _rank))
            if self.Y.shape[1] < _rank:
                # basis='auto': the plaquettes cannot represent this
                # topology (vias/PCB stacks are the recorded example), so
                # take the selected basis rather than solve in a subspace.
                # Announced, not silent -- the two differ in memory and
                # setup cost by enough that the caller should know which
                # one ran.
                self.basis_fallback = (
                    "plaquettes span %d of %d cycle-space dimensions"
                    % (self.Y.shape[1], _rank))
                warnings.warn(
                    "basis='auto': %s, so this geometry needs macro loops "
                    "no square can reach -- falling back to the SELECTED "
                    "basis (getmesh_fortran + exact Cholesky), which is "
                    "slower to set up and heavier in memory. Pass "
                    "basis='selected' to make that explicit."
                    % self.basis_fallback, RuntimeWarning, stacklevel=2)
                self.basis = 'selected'
        if self.basis == 'selected':
            self.Y = mg.getmesh_fortran(adjmat, self.esize,
                                        self.esize + self.fsize,
                                        self.efgsize, self.nodesize)
        self.Y.data = np.float64(self.Y.data)
        self.YT = self.Y.T.tocsc()
        self.YT.data = np.float64(self.YT.data)
        self.meshsize = np.shape(self.Y)[1]
        self.t_mesh = time.perf_counter() - t0
        t0 = time.perf_counter()
        # Cholesky of Y^T Y via cholesky_AAt(Y^T). float32 is deliberate
        # and matches main.py: this is only a preconditioner, and the
        # Krylov tolerance is on the true residual.
        #
        # mode/ordering were hard-coded to ('supernodal', 'amd'), the
        # WORST of the four combinations measured on the 24k-voxel bar
        # (straight_cond1, 45201 loops; peak RSS, and factor-only after
        # subtracting the 0.11 GB tree+mesh baseline):
        #     supernodal amd    0.32 GB   0.21 GB factor   12.5 s
        #     supernodal metis  0.31 GB   0.20 GB          11.3 s
        #     simplicial amd    0.29 GB   0.18 GB           9.2 s
        #     simplicial metis  0.24 GB   0.13 GB           4.2 s
        # i.e. 38% less factor memory AND 3x faster. Now defaults, and
        # exposed rather than hard-coded so a caller can go back.
        #
        # TWO CAVEATS. (1) The times are ONE CORE. Supernodal factors
        # through BLAS-3 and can use all 16; simplicial is scalar and
        # cannot. Threaded supernodal may close or reverse the TIME gap
        # -- the MEMORY result is structural and stands either way.
        # (2) Supernodal normally wins when the factor has large dense
        # blocks, so re-measure on a chunkier geometry before assuming
        # this ranking generalises.
        YT32 = self.YT.copy()
        YT32.data = np.float32(YT32.data)
        if self.basis == 'overcomplete':
            # Y^T Y is SINGULAR here (kernel = the cube boundaries), so a
            # Cholesky does not exist. It is also not wanted: the point of
            # the over-complete basis is that AMG works on it.
            self.chol = _AMGFactor(YT32, self.amg_cycles)
        else:
            self.chol = cholmod.cholesky_AAt(YT32, mode=self.chol_mode,
                                             ordering_method=self.chol_ordering)
        self.t_chol = time.perf_counter() - t0
        self.matvecs = 0
        # Cache of particular solutions, keyed by the caller. See
        # :meth:`particular` -- these are purely topological, so one entry
        # per port serves an entire frequency sweep.
        self._ihat = {}
        if verbose:
            print("  mesh basis %d loops (%.2f s), Cholesky (%.2f s)"
                  % (self.meshsize, self.t_mesh, self.t_chol))

    # -- the pieces main.py builds inline as SystemMat methods --------

    def _precond(self, vec):
        re = self.chol(np.float32(np.real(vec)))
        im = self.chol(np.float32(np.imag(vec)))
        return np.float64(re) + 1j*np.float64(im)

    def _apply_Z(self, i):
        """Branch impedance ``Z i`` via the FMM (in place on the buffer)."""
        self.whole[:self.efgsize] = i
        self.M.traverseRL()
        return self.whole[:self.efgsize]

    def _mesh_matvec(self, w):
        self.matvecs += 1
        return self.YT.dot(self._apply_Z(self.Y.dot(w)))

    def _divergence(self, i):
        """``A^T i``: filament currents -> net current out of each node."""
        self.whole[:self.efgsize] = i
        self.M.lv[0].data[:] = self.M.connectAT()
        return self.whole[self.efgsize:].copy()

    def _gradient(self, v):
        """``A v``: node potentials -> filament potential differences."""
        self.whole[self.efgsize:] = v
        (self.M.e.data[:], self.M.f.data[:], self.M.g.data[:]) = \
            self.M.connectA()
        return self.whole[:self.efgsize].copy()

    # -- the solve ---------------------------------------------------

    def _incidence_ops(self):
        """``(A^T, A)`` as LinearOperators over the filament/node spaces."""
        n_efg, n_nod = self.efgsize, self.nodesize
        Bop = LinearOperator((n_nod, n_efg), matvec=self._divergence,
                             rmatvec=self._gradient, dtype=np.complex128)
        BTop = LinearOperator((n_efg, n_nod), matvec=self._gradient,
                              rmatvec=self._divergence, dtype=np.complex128)
        return Bop, BTop

    def _tree_particular(self, s_n):
        """``i_hat`` on a SPANNING FOREST of the filament mesh graph.

        Direct O(N) alternative to the ``lsqr`` projection, using the same
        graph the loop basis comes from. Every non-tree filament carries
        zero; the flow on the tree edge above node ``k`` is the SUBTREE
        SUM of the injection below it, which is forced -- the subtree can
        only exchange current with the rest of the network through that
        one edge. So this satisfies ``A^T i_hat = s_n`` EXACTLY (to
        roundoff), where lsqr satisfies it to its tolerance.

        It is NOT the minimum-norm solution lsqr returns -- it threads the
        whole injected current along tree paths -- so it is a different,
        equally valid, particular solution. The converged answer is
        invariant (the loop solve absorbs any change in ``i_hat``), but
        the Krylov right-hand side ``-Y^T Z i_hat`` differs and the
        iteration count may move; that is what makes this worth measuring
        rather than adopting blind.

        Raises ValueError if the injection does not sum to zero over each
        connected component, which is the exact solvability condition
        (lsqr would silently return a least-squares fit instead).
        """
        from scipy.sparse.csgraph import connected_components, \
            breadth_first_order
        A = self.M.incidence()
        beta = float(np.real(self.M.lv[0].beta))
        nn = A.shape[1]
        # Decode the two endpoints of each filament: A has exactly one
        # +beta and one -beta per row (Tree.incidence asserts this).
        cols = A.indices.reshape(-1, 2).astype(np.int64)
        vals = np.real(A.data).reshape(-1, 2)
        pos = np.where(vals[:, 0] > 0, cols[:, 0], cols[:, 1])
        neg = np.where(vals[:, 0] > 0, cols[:, 1], cols[:, 0])
        # (A^T i)_k = beta*(sum of i over filaments with pos==k
        #                   - sum over filaments with neg==k), so i flows
        # INTO its `pos` node and OUT OF its `neg` node.
        nfil = pos.size
        graph = coo_matrix((np.ones(2*nfil), (np.r_[pos, neg],
                                              np.r_[neg, pos])),
                           shape=(nn, nn)).tocsr()
        # Filament lookup by ordered node pair, both directions. int64
        # throughout: nn**2 overflows int32 above ~46000 nodes.
        nn = np.int64(nn)
        key = np.r_[pos*nn + neg, neg*nn + pos]
        fil = np.r_[np.arange(nfil), np.arange(nfil)]
        srt = np.argsort(key)
        key, fil = key[srt], fil[srt]

        ncomp, label = connected_components(graph, directed=False)
        ihat = np.zeros(nfil, dtype=np.complex128)
        s = np.asarray(s_n, dtype=np.complex128)
        for c in range(ncomp):
            members = np.nonzero(label == c)[0]
            tot = s[members].sum()
            if abs(tot) > 1e-9*max(np.abs(s).max(), 1.0):
                raise ValueError(
                    "injection does not sum to zero over connected "
                    "component %d (net %.3g) -- A^T i = s_n has no exact "
                    "solution there. A port whose + and - terminals land "
                    "in different components would do this."
                    % (c, abs(tot)))
            if members.size < 2:
                continue
            order, pred = breadth_first_order(graph, members[0],
                                              directed=False)
            ch = order[1:].astype(np.int64)      # every node but the root
            pa = pred[ch].astype(np.int64)       # its parent in the BFS tree
            # Tree edge (pa, ch) -> filament index, looked up once for the
            # whole component rather than inside the accumulation loop.
            slot = np.searchsorted(key, ch*nn + pa)
            if (slot >= key.size).any() or (key[slot] != ch*nn + pa).any():
                raise RuntimeError(
                    "spanning-tree edge with no matching filament -- the "
                    "node graph and the incidence disagree")
            rr = fil[slot]
            # i flows INTO its `pos` node, so a flow of acc[k] from the
            # parent into k is +acc[k] when pos == k and -acc[k] otherwise.
            sgn = np.where(pos[rr] == ch, 1.0, -1.0)/beta
            # Reverse BFS order visits every child before its parent, so
            # one pass accumulates each subtree sum into its parent.
            acc = np.zeros(nn, dtype=np.complex128)
            acc[members] = s[members]
            for t in range(ch.size - 1, -1, -1):
                a = acc[ch[t]]
                ihat[rr[t]] = sgn[t]*a
                acc[pa[t]] += a
        return ihat

    def particular(self, s_n, key=None, lsqr_tol=1e-12, method='tree'):
        """Minimum-norm ``i_hat`` with ``A^T i_hat = s_n``, optionally cached.

        The loop basis is divergence-free (``A^T Y = 0``), so no loop
        current can carry the injection; ``i_hat`` is what puts the port
        current into the system and makes the loop right-hand side
        ``-Y^T Z i_hat`` nonzero.

        **This is PURELY TOPOLOGICAL.** ``A`` is the filament-to-node
        incidence scaled by ``beta``, and ``Tree.RDFinit`` sets
        ``beta = 1.0`` unconditionally -- no frequency enters. The same
        vector therefore serves every point of a frequency sweep, which
        matters because the lsqr takes ~1600 iterations on a large model
        and was the single largest item in the solve budget (see
        ``profile_solve_budget.py``). Pass a ``key`` -- anything hashable
        identifying the injection, e.g. ``(port, weight)`` -- to cache it
        on the solver; ``key=None`` disables caching.

        Because the relation is linear, cache the UNIT-current solution
        and scale it: ``particular(w)*current`` solves ``A^T x = w*current``.

        Parameters
        ----------
        method : {'tree', 'lsqr'}, optional
            ``'tree'`` (DEFAULT) uses :meth:`_tree_particular`, an O(N)
            spanning-forest construction that satisfies KCL exactly.
            ``'lsqr'`` returns the minimum-norm least-squares solution --
            the original route, ~1600 iterations on square_coil, and
            historically the single largest item in the solve.

            The two are DIFFERENT particular solutions; both are valid and
            the converged impedance is the same (measured 1.2e-14 /
            1.7e-13 on the small models, and identical to 7 digits on
            square_coil), but the Krylov right-hand side differs so
            iteration counts move -- 64 -> 49 on square_coil, 53 -> 54 and
            106 -> 111 on the small ones. ``'tree'`` is the default
            because it is 91x faster there (0.85 s vs 77.6 s) AND more
            accurate: lsqr STAGNATES at a KCL residual of 1.6e-9 on that
            model against its 1e-12 request, where the tree gives 7e-15.

            One behaviour difference worth knowing: ``'tree'`` RAISES if
            the injection does not sum to zero over each connected
            component, which is the exact solvability condition. lsqr
            silently returns a least-squares fit instead. A port whose two
            terminals land in different components is the case that
            differs.
        """
        if key is not None and key in self._ihat:
            return self._ihat[key]
        if method == 'tree':
            ihat = self._tree_particular(s_n)
        elif method == 'lsqr':
            Bop, _ = self._incidence_ops()
            ihat = lsqr(Bop, s_n, atol=lsqr_tol, btol=lsqr_tol)[0]
        else:
            raise ValueError("method must be 'lsqr' or 'tree', got %r"
                             % (method,))
        if key is not None:
            self._ihat[key] = ihat
        return ihat

    def solve(self, s_n, emf=None, rtol=1e-12, maxiter=30, inner_m=None,
              lsqr_tol=1e-12, ihat=None, potentials=True,
              ihat_method='tree', method='lgmres', precision='auto'):
        """Solve the LpR system for one nodal current injection.

        Parameters
        ----------
        s_n : ndarray of complex
            Injected nodal current, in SuperPEEC's compressed node ordering
            (from ``VhrModel.source_vector``). Must sum to zero.
        emf : ndarray of complex, optional
            A KNOWN series EMF on each filament, in ``e|f|g`` order.
            The branch equation becomes ``Z i + A v = -emf``, so it
            enters the loop right-hand side and the potential recovery
            but NOT the operator -- which is the whole point: the
            terminal half-filaments couple to the interior without
            becoming unknowns.
        rtol, maxiter, inner_m : optional
            Krylov controls (``maxiter``/``inner_m`` set the matvec
            budget for either method; see :func:`krylov_solve`).
        method : {'bicgstab', 'lgmres'}, optional
            Outer Krylov method; see :func:`krylov_solve`.
        lsqr_tol : float, optional
            Tolerance for the two least-squares projections.
        ihat_method : {'tree', 'lsqr'}, optional
            How to build ``i_hat`` when it is not supplied; see
            :meth:`particular`. Default ``'tree'``.
        ihat : ndarray of complex, optional
            A particular solution of ``A^T i_hat = s_n``, supplied by the
            caller (see :meth:`particular`) instead of solved for here.
            It is the caller's job to make it consistent with ``s_n``;
            ``info['kcl']`` reports the relative residual so a mistake
            shows up rather than quietly biasing the answer.
        potentials : bool, optional
            Recover the node potentials ``v`` by the second least-squares
            projection. Default True. With False that lsqr is SKIPPED and
            ``v`` comes back as None -- use it when the caller only needs
            a work-conjugate pairing ``w . v``, which
            :func:`impedance_matrix` evaluates from ``info['zi']`` without
            ever forming ``v``.

        Returns
        -------
        i : ndarray of complex
            Filament currents (e|f|g order).
        v : ndarray of complex or None
            Node potentials, gauge-fixed by the minimum-norm lsqr
            solution. Only DIFFERENCES are meaningful. None when
            ``potentials=False``.
        info : dict
            ``matvecs``, ``flag``, ``residual`` (true relative residual
            of the loop system), ``kcl`` (relative residual of
            ``A^T i_hat = s_n``), ``zi`` (the branch voltage ``Z i + emf``,
            which satisfies ``A v = zi``), ``ihat``, and timings.
        """
        _, BTop = self._incidence_ops()
        t0 = time.perf_counter()
        # particular solution of A^T i_hat = s_n
        if ihat is None:
            ihat = self.particular(s_n, lsqr_tol=lsqr_tol,
                                   method=ihat_method)
        nsn = np.linalg.norm(s_n)
        kcl = (np.linalg.norm(self._divergence(ihat) - s_n)/nsn
               if nsn > 0 else 0.0)
        zi = np.array(self._apply_Z(ihat))
        if emf is not None:
            zi += emf
        rhs = -self.YT.dot(zi)
        Aop = LinearOperator((self.meshsize, self.meshsize),
                             matvec=self._mesh_matvec, dtype=np.complex128)
        Pop = LinearOperator((self.meshsize, self.meshsize),
                             matvec=self._precond, dtype=np.complex128)
        n0 = self.matvecs
        w, flag = krylov_solve(Aop, rhs, Pop, method=method, rtol=rtol,
                               maxiter=maxiter, inner_m=inner_m,
                               precision=precision)
        nrhs = np.linalg.norm(rhs)
        resid = np.linalg.norm(rhs - Aop*w)/nrhs if nrhs > 0 else 0.0
        i = self.Y.dot(w) + ihat
        # Recover potentials from the branch equation. main.py solves
        # `connectA(y) = -Z i` and calls y the node potential; connectA's
        # sign convention makes that y the NEGATIVE of the potential (a
        # positive injected current would otherwise come back as a
        # negative resistance). Negated here so `v` is the potential and
        # `w . v` is the port voltage with the physical sign.
        zi = np.array(self._apply_Z(i))
        if emf is not None:
            zi += emf
        v = -lsqr(BTop, -zi, atol=lsqr_tol, btol=lsqr_tol)[0] \
            if potentials else None
        info = dict(matvecs=self.matvecs - n0, flag=flag, residual=resid,
                    kcl=kcl, zi=zi, ihat=ihat,
                    time=time.perf_counter() - t0)
        if self.verbose:
            print("    solve: %d matvecs, flag %s, rel resid %.2e, %.2f s"
                  % (info['matvecs'], flag, resid, info['time']))
        return i, v, info


def impedance_matrix(model, M, solver, freq, current=1.0, weight='corner',
                     terminals=True, potentials=False,
                     ihat_method='tree', **kw):
    """Open-circuit impedance matrix at one frequency.

    Injects ``current`` into port j (all other ports open, since no
    current is injected there) and reads the work-conjugate port
    voltages, giving column j of Z. One solve per port.

    Parameters
    ----------
    model : vhr.VhrModel
    M : multipole.Tree
    solver : LpRSolver
    freq : float
        Hz.
    current : complex, optional
    weight : {'corner', 'uniform'}, optional
        Terminal current profile; see the module docstring.
    terminals : bool, optional
        Add the series impedance of the port's terminal half-filaments
        (:func:`terminal_impedance`). Default True. Without it the model
        spans terminal cell centre to centre and is one cell of length
        short, which at DC is an error of ``1/L``.
    ihat_method : {'tree', 'lsqr'}, optional
        How the per-port particular solutions are built; see
        :meth:`LpRSolver.particular`. Default ``'tree'`` (O(N) spanning
        forest, exact KCL); ``'lsqr'`` is the original least-squares
        route. The cache key includes it, so switching does not reuse the
        other method's vectors.
    potentials : bool, optional
        Recover the node potentials and read the port voltage as
        ``w_k . v``, the original route. Default FALSE, which instead
        uses the equivalent work-conjugate identity below and skips the
        second least-squares projection entirely. Keep it for debugging
        or when the potentials themselves are wanted; the two agree at
        the loop-residual level.

    Notes
    -----
    **The port voltage does not need ``v``.** ``solve`` realises
    ``A v = zi`` with ``zi = Z i + emf``, and ``i_hat_k`` satisfies
    ``A^T i_hat_k = w_k`` by construction, so

        w_k . v = (A^T i_hat_k) . v = i_hat_k . (A v) = i_hat_k . zi

    and ``zi`` is already in hand from the last FMM sweep. That removes
    the second ``lsqr`` -- ~1600 iterations on a large model -- from
    every frequency. The identity is exact up to the loop residual (the
    part of ``zi`` outside ``range(A)``), which is the same floor the
    ``v`` route sits on; measured agreement is 1e-13..1e-12 at
    ``rtol=1e-12``.

    The ``i_hat_k`` are PURELY TOPOLOGICAL (see
    :meth:`LpRSolver.particular`), so they are cached on the solver and
    one set serves an entire frequency sweep, including the off-diagonal
    entries, which need the OTHER port's ``i_hat``.
    **kw
        Forwarded to :meth:`LpRSolver.solve`.

    Returns
    -------
    Z : ndarray of complex, shape (nports, nports)
    info : list of dict
        Per-port solve diagnostics.
    """
    n = len(model.ports)
    Z = np.zeros((n, n), dtype=np.complex128)
    infos = []
    model.prepare(M, freq)
    # Measurement weights, one per port, in compressed node ordering.
    # Frequency independent, so hoisted out of the solve loop.
    wts = [model.source_vector(M, k, 1.0, weight) for k in range(n)]
    # Unit-current particular solutions, one per port. Also frequency
    # independent, and cached on the solver so a sweep pays once. Under
    # `potentials` the old route is taken and these are not needed.
    ihats = None if potentials else \
        [solver.particular(wts[k], key=('port', k, weight, ihat_method),
                           method=ihat_method) for k in range(n)]
    esz, efsz, efgsz, _ = (np.size(M.e.struc),
                           np.size(M.e.struc) + np.size(M.f.struc),
                           np.size(M.e.struc) + np.size(M.f.struc)
                           + np.size(M.g.struc), 0)
    jw = 1j*2*np.pi*freq
    for j in range(n):
        model.prepare(M, freq)
        src = model.source_vector(M, j, current, weight)
        emf = None
        lp = None
        if terminals:
            lp, axis = terminal_interior_coupling(model, M, j, weight)
            lo = {1: 0, 0: esz, 2: efsz}[axis]
            emf = np.zeros(efgsz, dtype=np.complex128)
            emf[lo:lo+lp.size] = jw*current*lp
        if potentials:
            i, v, info = solver.solve(src, emf=emf,
                                      ihat_method=ihat_method, **kw)
        else:
            # i_hat scales linearly with the injection, so the cached
            # unit solution is reused rather than re-solved.
            i, v, info = solver.solve(src, emf=emf, potentials=False,
                                      ihat=current*ihats[j], **kw)
        infos.append(info)
        for k in range(n):
            # V_k = w_k . v  is gauge invariant because sum(w_k) = 0
            Z[k, j] = (np.dot(wts[k], v)/current if potentials
                       else np.dot(ihats[k], info['zi'])/current)
        if terminals:
            # terminal <-> terminal, plus the transpose gather of the
            # terminal <-> interior coupling (same operator, by symmetry)
            Z[j, j] += terminal_impedance(model, M, j, freq, weight)
            Z[j, j] += jw*np.dot(lp, i[lo:lo+lp.size])/current
    return Z, infos


def as_r_jl(Z, freq):
    """Convert an impedance matrix to VoxHenry's ``R + jL`` reporting.

    VoxHenry prints ``abs(real(Z)) + 1j*abs(imag(Z))/(2*pi*f)``, i.e.
    resistance in ohms and inductance in henries, with the SIGN
    DISCARDED on both parts. The sign matters for mutual terms -- a
    return path shared between two conductors drives the mutual
    inductance negative at high frequency -- so this keeps the sign and
    the caller can take ``abs`` to match VoxHenry's display exactly.
    """
    return np.real(Z) + 1j*np.imag(Z)/(2*np.pi*freq)


def profile_sensitivity(model, M, solver, freq, pair=('uniform', 'rim'),
                        **kw):
    """How much does the answer depend on the terminal current profile?

    Returns ``(Z_a, Z_b, rel_spread, infos)`` for the two profiles in
    ``pair``. A small spread means the extraction is insensitive to the
    port lumping and the terminal model is not the limiting error; a
    large one means it is, which is expected when the conductor is only
    a few cells long.

    THE PAIR CHANGED 2026-08-09, and the old one was worthless: under
    the CELL scheme a port face maps to exactly ONE node, so 'corner'
    and 'uniform' hand every face the identical 1/N and the spread is
    exactly 0.000e+00 BY CONSTRUCTION -- measured on 9-face and 4-face
    ports. Since 'cell' is the default scheme, this function used to
    certify "insensitive to the port profile" on every model it was
    asked about. ('uniform', 'rim') is a genuinely different pair in
    BOTH schemes; see :meth:`voxmodel.VoxelModel.port_nodes`.
    """
    Zc, infos = impedance_matrix(model, M, solver, freq, weight=pair[0],
                                 **kw)
    Zu, _ = impedance_matrix(model, M, solver, freq, weight=pair[1], **kw)
    denom = np.abs(Zc)
    denom[denom == 0] = 1.0
    return Zc, Zu, float(np.max(np.abs(Zc - Zu)/denom)), infos


def sweep(model, M, freqs=None, current=1.0, weight='corner', verbose=True,
          sensitivity=False, **kw):
    """Impedance over a frequency list, reusing one mesh basis.

    Returns a list of ``(freq, Z, info)``.
    """
    if freqs is None:
        freqs = model.freq
    model.prepare(M, float(freqs[0]))
    solver = LpRSolver(M, verbose=verbose)
    out = []
    for f in np.atleast_1d(freqs):
        f = float(f)
        if sensitivity:
            Zc, Zu, spread, infos = profile_sensitivity(model, M, solver,
                                                        f, **kw)
            Z = Zc
            for d in infos:
                d['spread'] = spread
        else:
            Z, infos = impedance_matrix(model, M, solver, f,
                                        current=current, weight=weight, **kw)
        out.append((f, Z, infos))
    return out


def _main(argv):
    if not argv:
        print("usage: python3 port_impedance.py <file.vhr> [freq ...]")
        return 1
    path = argv[0]
    m = vhr.read_vhr(path)
    freqs = [float(a) for a in argv[1:]] or list(m.freq)
    print(m.summary())
    leaf, levels = m.partition()
    t0 = time.perf_counter()
    M = m.build_tree(leaf, levels)
    print("\n  tree: leaf %s, %d levels, pitch %g m, %.2f s"
          % ('x'.join(str(v) for v in leaf), levels, float(M.e.l[0]),
             time.perf_counter() - t0))
    m.prepare(M, freqs[0])
    solver = LpRSolver(M, verbose=True)
    n = len(m.ports)
    print("\n  %-11s %-5s %-14s %-15s %-8s %-9s %s"
          % ('freq (Hz)', 'port', 'R (ohm)', 'L (H)', 'matvecs', 'resid',
             'profile'))
    for f in freqs:
        Zc, Zu, spread, infos = profile_sensitivity(m, M, solver, f)
        rj = as_r_jl(Zc, f)
        mv = sum(d['matvecs'] for d in infos)
        rs = max(d['residual'] for d in infos)
        for k in range(n):
            for j in range(n):
                tag = '-' if n == 1 else '%d,%d' % (k, j)
                first = (k == 0 and j == 0)
                print("  %-11.4g %-5s %-14.6g %-15.6g %-8s %-9s %s"
                      % (f if first else '', tag,
                         np.real(rj[k, j]), np.imag(rj[k, j]),
                         mv if first else '',
                         ('%.1e' % rs) if first else '',
                         ('%.1e' % spread) if first else ''))
    print("\n  'profile' = relative spread between the 'corner' and "
          "'uniform' terminal\n  current profiles: how much the answer "
          "depends on the port lumping.")
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(_main(sys.argv[1:]))


# axis -> filament orientation, for the terminal half-filaments
_AXIS_ORIENT = {0: 'f', 1: 'e', 2: 'g'}


def lppr_impedance_matrix(solver, freq, ports=None, current=1.0,
                          warm=True, **kw):
    """Open-circuit impedance matrix on the LpPR (capacitive) path.

    The counterpart of :func:`impedance_matrix`, which is LpR-only. One
    solve per port: inject into port j with every other port open (no
    injection there) and read the work-conjugate voltage at port i,

        Z_ij = (s_i . v_j) / I^2

    with ``v = -x[efgsize:]`` (``matvecLpPR`` carries +A on the branch
    rows, so the physical potential is MINUS the node solution -- the
    same convention :meth:`LpPRSolver.solve` folds into its scalar Z).

    WHY THIS MATTERS MORE THAN THE ONE-PORT CASE. Every quantity that
    is a COUPLING rather than a self term needs it: a SQUID's input
    coil to washer mutual inductance, coupled RSFQ transmission lines,
    a PDN with several decap sites. None of them are reachable through
    repeated one-port solves, because the off-diagonal IS the answer.

    It also buys a free validation: the discrete system is complex
    SYMMETRIC, so Z must satisfy Z = Z^T. ``info['reciprocity']``
    reports ``max|Z - Z^T|/max|Z|`` -- a genuine check on the port
    machinery, in the spirit of validate_port_impedance's two-port
    reciprocity test on the LpR side.

    Warm starting across PORTS (not frequencies) is on by default: the
    RHS changes between columns but the OPERATOR does not, which is the
    case warm starts actually suit -- unlike a frequency sweep, where
    the operator moves and the RHS is fixed (see
    :meth:`LpPRSolver.solve`). The guess is passed unscaled
    (``x0_mode='none'``) because there is no frequency ratio to apply.
    """
    m = solver.model
    S = solver.S
    if ports is None:
        ports = list(range(len(m.ports)))
    n = len(ports)
    Z = np.zeros((n, n), dtype=np.complex128)
    srcs = [m.source_vector(solver.M, p, current,
                            weight=kw.get('weight', 'corner'))
            for p in ports]
    infos = []
    x_prev = None
    for j, pj in enumerate(ports):
        _, x, info = solver.solve(freq, port=pj, current=current,
                                  x0=x_prev if warm else None,
                                  x0_mode='none', **kw)
        if warm:
            x_prev = x
        v = -np.asarray(x[S.efgsize:])
        for i in range(n):
            Z[i, j] = complex(np.dot(srcs[i], v))/(current*current)
        infos.append(info)
    denom = max(float(np.max(np.abs(Z))), 1e-300)
    recip = float(np.max(np.abs(Z - Z.T))/denom)
    return Z, dict(reciprocity=recip, infos=infos, ports=list(ports))


def lppr_equipotential_impedance(solver, freq, port=0, current=1.0,
                                 pairing='sorted', **kw):
    """EQUIPOTENTIAL-terminal port impedance on the LpPR path.

    Removes the prescribed-profile assumption entirely: the terminal
    current SPLIT is solved for rather than assumed, which is what
    ``EquiTerminalSolver`` does on the LpR side and what LpPR has
    lacked.

    METHOD -- substructuring, not a new operator. Treat each (P face,
    N face) pair as a sub-port and extract the k x k sub-port impedance
    matrix ``Zs`` (:func:`lppr_impedance_matrix`, one solve per
    sub-port, warm started across columns). An ideal terminal forces
    every sub-port voltage to the same V while the splits sum to I, so

        Zs s = V 1,   1^T s = I    =>    Z = V/I = 1/(1^T Zs^-1 1)

    -- the parallel combination of the sub-ports. This is EXACT for the
    linear network (no new discretisation, no preconditioner surgery),
    and it needs no change to SystemMat, the W rescale or diagschur --
    which is why it is worth doing before the bordered-system version.

    THE ASSUMPTION IT STILL MAKES: equal PAIR voltages is implied by
    (both terminals equipotential) but does not by itself force each
    terminal to be separately equipotential, so for a geometrically
    asymmetric port the two differ. Pass different ``pairing`` values
    and compare -- a spread means the port is in that regime.

    THE NUMERICAL CATCH, and it is real for capacitive ports: below
    resonance every sub-port sees essentially the same 1/(jwC), so
    ``Zs`` is nearly RANK ONE (measured on a 2-port pdn: entries agree
    to 6 digits) and inverting it cancels most of your significant
    figures. ``info['cond']`` reports cond(Zs); cond ~5e7 was measured
    on a 9-face pdn port, where R converges only once the solves reach
    tol 1e-10 (R moved 1.1% between tol 1e-8 and 1e-10, then was stable
    to 4 digits through 1e-13). CHECK CONVERGENCE IN tol, not just in
    the residual, whenever you use this on a capacitive port.
    INDUCTIVE ports do not suffer this.

    MEASURED, and the reason this function had to exist (9-face pdn
    port at 1e8, tol 1e-12):
      * The solved split is [0.564, -0.092, 0.412, -0.092, -0.248,
        -0.103, 0.412, -0.103, 0.252] A per face -- it CHANGES SIGN.
        Current flows into some faces and back out of others,
        recirculating around the antipads. No non-negative prescribed
        profile can represent that.
      * So R_equipotential = 4.3511e-4 against 3.6284e-4 (uniform) and
        3.5817e-4 (rim): 20% away, and OUTSIDE the bracket the two
        prescribed profiles span. The profile-sensitivity diagnostic
        (1.28% spread) therefore UNDERSTATES the port-model error by
        more than an order of magnitude -- it samples two points in a
        family that excludes the answer.
      * C is untouched (525.9686 vs 525.9689 fF): capacitance is a
        global electrostatic quantity and does not care how the port
        current distributes. The port model is an R story.
      * Verified by superposition: driving the solved split makes the
        terminal potential spread 9.1e-8 (P) / 1.0e-7 (N) against
        7.6e-7 / 6.3e-7 for the prescribed profile, and reproduces
        this function's Z to 7 digits. The residual spread is the
        pair-constraint caveat above, whose size shows up as ~0.9%
        sensitivity to ``pairing``.
      * NOTE it is NOT true that this must lower R. Minimum-dissipation
        (Thomson) arguments apply to DC resistive networks; a
        reactance-dominated AC port has no such variational principle,
        and here the equipotential R is HIGHER.

    Returns ``(Z, info)``; ``info['split']`` is the solved per-face
    current, which is the diagnostic worth looking at.
    """
    import voxmodel as _vm
    m = solver.model
    p = m.port(port)
    pos = [tuple(int(v) for v in e) for e in p.pos]
    neg = [tuple(int(v) for v in e) for e in p.neg]
    if len(pos) != len(neg):
        raise NotImplementedError(
            "equipotential terminal needs matching P/N face counts "
            "(got %d and %d); an unpaired terminal needs the 2k-node "
            "constrained form" % (len(pos), len(neg)))
    if pairing == 'sorted':
        axis = pos[0][3]
        others = [c for c in range(3) if c != axis]
        key = lambda e: (e[others[0]], e[others[1]])
        pos, neg = sorted(pos, key=key), sorted(neg, key=key)
    elif pairing == 'reversed':
        pos = sorted(pos)
        neg = sorted(neg)[::-1]
    elif pairing != 'given':
        raise ValueError("pairing must be sorted/reversed/given")
    subs = []
    for j, (ep, en) in enumerate(zip(pos, neg)):
        q = _vm.Port('_eqsub%d' % j)
        q._add('P', ep)
        q._add('N', en)
        q._freeze()
        subs.append(q)
    saved = m.ports
    try:
        m.ports = subs
        Zs, minfo = lppr_impedance_matrix(solver, freq, current=current,
                                          **kw)
    finally:
        m.ports = saved
    one = np.ones(len(subs), dtype=np.complex128)
    y = np.linalg.solve(Zs, one)
    Z = 1.0/complex(one @ y)
    cond = float(np.linalg.cond(Zs))
    info = dict(cond=cond, nsub=len(subs), reciprocity=minfo['reciprocity'],
                matvecs=sum(i['matvecs'] for i in minfo['infos']),
                Zsub=Zs, split=Z*y*current)
    if cond > 1e10:
        import warnings
        warnings.warn(
            "lppr_equipotential_impedance: cond(Zs) = %.2e -- the "
            "sub-port matrix is nearly rank deficient (every sub-port "
            "sees the same reactance), so the parallel combination has "
            "lost most of its significant digits. Treat Z as "
            "indicative and use the constrained solve instead." % cond,
            RuntimeWarning)
    return Z, info


def lppr_profile_sensitivity(solver, freq, port=0,
                             pair=('uniform', 'rim'), **kw):
    """How much does an LpPR answer depend on the port current profile?

    The capacitive-path counterpart of :func:`profile_sensitivity`.
    Returns ``(Z_a, Z_b, rel_spread, infos)`` for the two profiles
    in ``pair`` -- 'uniform' vs 'rim' by default, because 'corner' and
    'uniform' are the SAME vector under the cell scheme (see
    :func:`profile_sensitivity`).

    This is the cheapest honest statement about the port model SuperPEEC
    can make on the LpPR path, and until the equipotential terminal
    lands it is the ONLY one: a prescribed profile is an assumption,
    and the spread between two defensible profiles bounds what that
    assumption is worth. Small spread => port lumping is not the
    limiting error; large spread => it is, and no amount of Krylov
    convergence will help.
    """
    zc, _, ic = solver.solve(freq, port=port, weight=pair[0], **kw)
    zu, _, iu = solver.solve(freq, port=port, weight=pair[1], **kw)
    # COMPONENT-WISE, not |dZ|/|Z|. A capacitive port is reactance
    # dominated (|Z| ~ 3e3 ohm with R ~ 4e-4), so the |Z|-relative
    # spread reads 1e-7 and hides a 1.3% disagreement in R -- and R is
    # exactly the quantity a port profile perturbs. Components below a
    # floor of |Z| are skipped so a lossless superconductor port (R
    # numerically 0) does not divide 0 by 0.
    floor = 1e-12*max(abs(zc), 1e-300)
    parts = []
    for a, b in ((zc.real, zu.real), (zc.imag, zu.imag)):
        if abs(a) > floor:
            parts.append(abs(a - b)/abs(a))
    spread = max(parts) if parts else abs(zc - zu)/max(abs(zc), 1e-300)
    return zc, zu, float(spread), (ic, iu)


def terminal_impedance(model, M, port=0, freq=1.0, weight='corner',
                       t_l=None):
    """Series impedance of a port's terminal half-filaments.

    ``t_l`` is the terminal length, i.e. THE PORT REFERENCE PLANE:
    ``t_l = dx/2`` (the default) puts it on the conductor face, smaller
    moves it inside the end cell, larger outside. That is what makes
    de-embedding a parameter rather than a remesh.

    With nodes at cell centres the solved network spans centre to
    centre, so a port driven face to face is one cell of length short
    (``(L-1)dx`` rather than ``L dx``). ``terminal.py`` closes that with
    a half-length filament at each port face; this evaluates what those
    add to the port impedance.

    They are NOT extra unknowns. All the current crossing a port face
    flows through that face's terminal filament into the cell behind it,
    so with a prescribed port profile the terminal currents are fixed at
    ``I*u_k`` and their contribution is a series term:

        Z_term = sum_kl u_k u_l ( R_t delta_kl + jw Lp(k,l) )

    with ``u_k`` the per-face share, normalised to sum to 1 at EACH end
    (both ends are in series along the current path, so both add). The
    terminal filaments all carry current in the same direction along the
    port axis -- the injection weights differ in sign between the two
    terminals, the CURRENTS do not -- so ``u_k = |w_k|``.

    APPROXIMATION, and the reason this is a correction rather than a
    reformulation: the mutual coupling between a terminal filament and
    the INTERIOR filaments is neglected. It is exact at DC, where only
    resistance survives and this reproduces ``L/(N^2 dx sigma)``
    exactly. At high frequency the terminal sits collinear with and
    adjacent to the first interior filament, so their mutual is not
    small; treating it properly means making the terminal currents
    genuine unknowns, which changes the unknown set (several terminals
    per end share a port node, so they redistribute rather than being
    determined). See validate_port_impedance.py for the measured size.

    Returns
    -------
    complex
        Impedance to add in series with the extracted port impedance.
    """
    import terminal as tm
    p = model.port(port)
    # Material AT THIS PORT. On a mixed-material model the terminal sits
    # in one specific metal (Cu_Al has one port on each). A SUPERCONDUCTOR
    # (or a dielectric) has no meaningful sigma -- it may be 0 everywhere
    # -- so take the general impedance density z(w) [ohm m] at the port
    # cells and use sigma_eff = 1/z, which terminal_resistance consumes
    # unchanged: R_t = t_l/(A sigma_eff) = z t_l/A is exactly the series
    # impedance of that half filament. For a normal metal z = 1/sigma and
    # this reduces to the old expression identically.
    if getattr(model, 'superconductor', False) or \
            getattr(model, 'epsilon', None) is not None:
        zd = np.asarray(model.impedance_density(freq))
        cells = [(int(e[0]), int(e[1]), int(e[2]))
                 for arr in (p.pos, p.neg) for e in arr]
        vals = np.array([zd[c] for c in cells])
        nz = vals[vals != 0]
        if nz.size == 0:
            raise ValueError("%s: port %s touches no material with a "
                             "finite impedance density" % (model.name, port))
        sigma = 1.0/np.mean(nz)
    else:
        sigma = model.port_sigma(port)
    # per-axis pitch: the terminal's axial step and its cross-section are
    # all read from l below, so anisotropic models need no special case
    # (model.dx raises on them by design)
    l = tuple(float(v) for v in model.d)
    jw = 1j*2*np.pi*freq
    # per-face share, summing to 1 at each end
    faces = []
    for arr in (p.pos, p.neg):
        if len(arr) == 0:
            continue
        share = 1.0/len(arr)
        for e in arr:
            faces.append((int(e[0]), int(e[1]), int(e[2]), int(e[3]),
                          int(e[4]), share))
    if not faces:
        return 0.0
    axis = faces[0][3]
    orient = _AXIS_ORIENT[axis]
    others = [c for c in range(3) if c != axis]
    rt = tm.terminal_resistance(l, orient, sigma, t_l=t_l)
    # axial half-slot: cell c spans [2c, 2c+2); the terminal occupies the
    # half next to its face
    slots = []
    for (ix, iy, iz, ax, sign, share) in faces:
        cell = (ix, iy, iz)
        slot = 2*cell[axis] + (1 if sign > 0 else 0)
        slots.append((slot, cell[others[0]], cell[others[1]], share))
    span = max(s[0] for s in slots) - min(s[0] for s in slots) + 2
    ncell = [int(v) for v in model.dims]
    n = [0, 0, 0]
    n[axis] = span//2 + 2
    n[others[0]] = ncell[others[0]] + 1
    n[others[1]] = ncell[others[1]] + 1
    kern = tm.axial_halfstep_kernel(l, orient, n)
    z = 0j
    for (sa, ta0, ta1, ua) in slots:
        z += ua*ua*rt
        for (sb, tb0, tb1, ub) in slots:
            z += jw*ua*ub*tm.mutual_segments(
                kern, orient, (sa, 1), (sb, 1),
                transverse=(ta0 - tb0, ta1 - tb1))
    return complex(z)


def _terminal_slots(model, port, axis):
    """(half-slot, transverse0, transverse1, share) per terminal face.

    Share is normalised to 1 at EACH end -- both ends sit in series
    along the current path, and both carry current the same way along
    the port axis even though their injection weights differ in sign.
    """
    p = model.port(port)
    others = [c for c in range(3) if c != axis]
    out = []
    for arr in (p.pos, p.neg):
        if len(arr) == 0:
            continue
        share = 1.0/len(arr)
        for e in arr:
            cell = (int(e[0]), int(e[1]), int(e[2]))
            slot = 2*cell[axis] + (1 if int(e[4]) > 0 else 0)
            out.append((slot, cell[others[0]], cell[others[1]], share))
    return out


def _interior_slots(M, axis):
    """Per-filament (low half-slot, transverse0, transverse1), compressed.

    A cell-centred filament at lattice index i along its own axis joins
    cells i and i+1, so it occupies half-slots ``2i+1`` and ``2i+2``.

    At a single level the compressed ``idx`` are already global lattice
    indices. Multilevel stores per-box indices and needs the box offset
    added -- the same decode as ``profile_fill_fraction.globkey``.
    Getting this wrong is not loud: it silently mislabels every
    filament's position, which shows up as broken reciprocity rather
    than as an error.
    """
    leaf = {0: M.f, 1: M.e, 2: M.g}[axis]
    n = np.asarray(leaf.n, dtype=np.int64)
    idx = np.asarray(leaf.idx, dtype=np.int64)
    cell = [idx//(n[1]*n[2]), (idx//n[2]) % n[1], idx % n[2]]
    if M.numlevels > 1:
        off = [np.zeros(idx.size, dtype=np.int64) for _ in range(3)]
        bx = np.asarray(leaf.xidx, dtype=np.int64)
        by = np.asarray(leaf.yidx, dtype=np.int64)
        bz = np.asarray(leaf.zidx, dtype=np.int64)
        i0 = np.asarray(leaf.idx0, dtype=np.int64)
        for g in range(i0.size - 1):
            sl = np.s_[i0[g]:i0[g+1]]
            off[0][sl] = bx[g]*n[0]
            off[1][sl] = by[g]*n[1]
            off[2][sl] = bz[g]*n[2]
        cell = [cell[c] + off[c] for c in range(3)]
    others = [c for c in range(3) if c != axis]
    return 2*cell[axis] + 1, cell[others[0]], cell[others[1]]


def terminal_interior_coupling(model, M, port=0, weight='corner'):
    """Mutual partial inductance between a port's terminals and every
    filament, as the pair of operators the correction needs.

    The terminal currents are PRESCRIBED by the port profile, so this
    coupling never enters the operator or the unknown set. It appears
    exactly twice, and by symmetry both are the same object:

      * a known EMF on every interior filament, added to the loop RHS;
      * a gather of the solved interior currents into the terminal
        voltage drops, at post-processing.

    Only filaments PARALLEL to the port axis couple: the mutual partial
    inductance of perpendicular filaments vanishes, which removes two
    thirds of the work before it starts.

    Returns
    -------
    lp : ndarray
        ``sum_t u_t Lp(j, t)`` for every filament ``j`` of the port's
        orientation, in that leaf's compressed order. Multiply by
        ``jw*I`` for the EMF; contract with the solved currents for the
        gather.
    axis : int
        The port axis, so the caller knows which leaf block ``lp``
        indexes.
    """
    import terminal as tm
    p = model.port(port)
    if len(p.pos) == 0:
        raise ValueError("port %r has no positive terminal" % port)
    axis = int(p.pos[0][3])
    orient = _AXIS_ORIENT[axis]
    dx = model.dx
    l = (dx, dx, dx)
    terms = _terminal_slots(model, port, axis)
    s0, t0, t1 = _interior_slots(M, axis)
    dims = [int(v) for v in model.dims]
    others = [c for c in range(3) if c != axis]
    n = [0, 0, 0]
    n[axis] = dims[axis] + 2
    n[others[0]] = dims[others[0]] + 1
    n[others[1]] = dims[others[1]] + 1
    kern = tm.axial_halfstep_kernel(l, orient, n)
    ka = kern.shape[AXIS_ORDER[axis][0]]
    # SIGN: returned in the SOLVER's convention, not as a bare mutual
    # inductance. LpRSolver's `v` is the NEGATED node potential -- it
    # solves `A v = +Z i`, the same flip noted in solve() -- so a
    # positive physical mutual enters the branch equation with the
    # opposite sign. Both uses (the RHS EMF and the transpose gather)
    # take it identically, so the convention is applied once here rather
    # than twice at the call sites. Verified against the uncoupled
    # Richardson limit: with this sign the coarsest mesh lands on the
    # extrapolated L, with the other it moves away by the same amount.
    # ...and by the CURRENT DIRECTION along the port axis, which the
    # terminal shares do not carry: they are |w|, since both terminals
    # of a port pass current the same way. Which way that is depends on
    # the port's geometry -- current ENTERS at the P faces, so it flows
    # ANTI-parallel to the P face normal. A port driven from the low
    # face (normal -1) sends current along +axis; one driven from the
    # high face sends it along -axis. Omitting this applies the whole
    # coupling backwards, which is worth exactly twice the correction
    # and is invisible to reciprocity (both uses share lp, so the
    # result stays symmetric either way).
    sign = -float(p.pos[0][4])
    lp = np.zeros(s0.size, dtype=np.float64)
    for (st_, a0, a1, u) in terms:
        d0 = np.abs(t0 - a0)
        d1 = np.abs(t1 - a1)
        for s in (s0, s0 + 1):          # the filament's two half-slots
            da = np.abs(s - st_)
            ok = da < ka
            sel = [None, None, None]
            sel[axis] = da[ok]
            sel[others[0]] = d0[ok]
            sel[others[1]] = d1[ok]
            lp[ok] += sign*u*kern[sel[0], sel[1], sel[2]]
    return lp, axis


# kernel axis order per port axis: (own, other0, other1) -> (x, y, z)
AXIS_ORDER = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}


class LpPRSolver:
    """Port impedance through the PRODUCTION mixed-row LpPR path.

    The full R-L-P-C solve for a :class:`voxmodel.VoxelModel` with
    ports: capacitive tree, W-rescaled mixed-row operator
    (``SystemMat.rescaleLpPR``), the diagschur (default) or reluctance
    fixed sparse preconditioner, right-preconditioned fgmres. This is
    the path that carries PER-CELL DIELECTRICS (``model.epsilon``):
    their polarization branches ride the complex filament impedances
    from ``impedance_density`` and the bound charge rides the panel
    machinery -- wired and validated 2026-08-08 against the dense
    part-D oracle (C to 5 significant digits on vacuum/half/filled
    plate pairs; see validate_dielectric part F).

    CONVENTIONS (the two traps measured while wiring this):
      * The port current injections enter the RESCALED rhs directly
        (the rescaled K' = [[Z, A], [A^T, -jw C]] is the standard MNA
        in current units). Feeding them to the raw node rows and
        rescaling would divide them by P: the drive becomes a smeared
        P^-1 s and the node solution comes out charge-like (measured
        V off by the P scale, ~6e16).
      * ``matvecLpPR`` is the part-D system with v -> -v (its branch
        rows carry +A), so the physical potential is MINUS the node
        solution; the port voltage below folds that in.

    Parameters
    ----------
    model : voxmodel.VoxelModel
    M : multipole.Tree
        Capacitive tree, already prepared (``model.prepare(M, freq)``).
    precond : {'diagschur', 'reluctance'}, optional
        diagschur (default) is the low-frequency solver -- the right
        regime for dielectric/PDN work far below resonance.
    """

    def __init__(self, model, M, precond='diagschur', wsolve='auto',
                 **precopts):
        from systemmat import SystemMat as _SystemMat
        from scipy.sparse.linalg import LinearOperator as _LO
        if np.size(M.e.struc) == 0 or np.size(M.f.struc) == 0 \
                or np.size(M.g.struc) == 0:
            raise NotImplementedError(
                "%s: an orientation has ZERO filaments (e %d, f %d, "
                "g %d) -- the fortran filament2node/connect kernels "
                "reject empty orientations. A plate pair of 1-cell "
                "plates with an empty gap is the canonical case; "
                "thicken the plates." % (
                    model.name, np.size(M.e.struc), np.size(M.f.struc),
                    np.size(M.g.struc)))
        has_diel = getattr(model, 'epsilon', None) is not None and \
            bool(np.any((np.asarray(model.epsilon) != 1.0)
                        & (model.sigma == 0.0)))
        if precond == 'reluctance' and has_diel:
            # MEASURED 2026-08-08 (160^2 filled pdn): reluctance's
            # N_Z = (1/jw) Ktilde models every branch as inductive
            # metal; a dielectric branch's true admittance is
            # ~jw C_exc -- TEN ORDERS away -- and the preconditioned
            # iteration produces garbage (306 matvecs, TRUE residual
            # 5.2e-1, C nonsense) rather than converging slowly. A
            # material-split hybrid (Ktilde on metal rows + exact
            # diagonal admittance on dielectric rows) is the docketed
            # fix for high-frequency dielectric boards; until then
            # diagschur handles all frequencies for dielectric models
            # (with growing counts above the wL/R crossover).
            raise NotImplementedError(
                "%s: precond='reluctance' is incompatible with "
                "dielectric models (measured: true residual 0.52 "
                "after full iteration budget). Use 'diagschur', or "
                "see the material-split hybrid on the docket."
                % model.name)
        self.model = model
        self.M = M
        self.S = _SystemMat(M, M.jomega)
        self._dz_static = None
        # W rescale route: 'exact' needs the near-field P_ext Cholesky,
        # which the truncated multilevel near field often cannot supply
        # -- and DIELECTRICS make it strictly worse (measured on the
        # (16,16,12) plate pair: vacuum definite at leaf [8,8,13], the
        # half-filled model indefinite even at [11,11,13]: the slab's
        # free-surface node layer adds closely-spaced couplings the
        # 27-box window truncates). 'band' (reluctance.extract_ccap
        # windowed inverse) needs no factorisation and is the scalable
        # route; 'auto' picks exact when the Cholesky exists, else band.
        if wsolve == 'auto':
            # exact iff the near-field Cholesky actually EXISTS. Keying
            # on numlevels==1 was wrong twice over: a CIRCULANT
            # single-level tree deliberately never forms the dense n2n
            # (its 'exact' path would try a 700 GB dense fallback at
            # 300k nodes), and circulant is the designed capacitive
            # route at scale -- the multilevel capacitive tree's near
            # n2n costs ~27*leaf^3 entries/node (~33 GB at a 320^2
            # board, vs 0.1 GB inductive).
            wsolve = 'exact' if M.n2nchol is not None else 'band'
        # HISTORY (2026-08-08): band W was briefly REFUSED for
        # dielectric models after measuring 49% C errors at converged
        # rescaled residuals. The real culprit was solve()'s old rhs
        # shortcut (injections straight into the rescaled rhs -- valid
        # only for the exact W); with the correct rhs = W*(P*inj), the
        # stock band W reproduces the dense-probed multilevel ground
        # truth to 6 digits on the same models. extract_ccap was never
        # at fault; the true-residual postcheck in solve() guards the
        # remaining (conditioning-level) W-quality risk.
        if wsolve == 'band':
            self.S.wsolveinit('band')
        self.wsolve = wsolve
        self.Kprime = _LO((self.S.wholesize, self.S.wholesize),
                          matvec=self.S.rescaleLpPR, dtype=np.complex128)
        if precond == 'diagschur':
            self.S.diagschurprecinit(**precopts)
            pv = self.S.precondDiagSchur
        else:
            self.S.reluctanceprecinit(**precopts)
            pv = self.S.precondReluctance
        self._precond = precond
        self.P = _LO((self.S.wholesize, self.S.wholesize),
                     matvec=pv, dtype=np.complex128)

    def solve(self, freq, port=0, current=1.0, tol=1e-10, restrt=300,
              maxiter=1, callback=None, verbose=False, x0=None,
              x0_freq=None, x0_mode='physical', weight='corner',
              terminals=False, t_l=None):
        """Solve at one frequency; returns ``(Z, x, info)``.

        ``Z`` is the port impedance V/I (physical sign). ``x`` is the
        solution of the rescaled system ([currents; -potentials]).

        PORT MODEL (and how it is still weaker than the LpR path's).
        This drives a PRESCRIBED nodal current profile; there is no
        equipotential terminal here, unlike ``EquiTerminalSolver``,
        which solves for the terminal current split. Two knobs make
        that gap measurable and partly correctable:

        ``weight`` : {'corner', 'uniform'}
            The terminal current profile, exactly as the LpR path's
            :func:`impedance_matrix` uses it. If the two profiles give
            the same answer the extraction is insensitive to port
            lumping; if they diverge, the port model IS the dominant
            error. See :func:`lppr_profile_sensitivity`.
        ``terminals`` : bool
            Add the series impedance of the port's terminal half
            filaments (:func:`terminal_impedance`). With nodes at cell
            centres the solved network spans centre to centre, so the
            port is one cell of length SHORT; the LpR path corrects
            this by default. DEFAULT FALSE HERE ON PURPOSE: every
            recorded LpPR anchor (validate_dielectric parts D-H, the
            pdn capstone, the dense oracle in studies/squid_washer.py)
            was taken centre-to-centre, and the dense oracle spans the
            same way, so flipping the default would silently move all
            of them. Pass True for a face-to-face port impedance
            comparable with the LpR convention.
        ``t_l`` : float, optional
            Terminal length = PORT REFERENCE PLANE (only with
            ``terminals=True``); ``dx/2`` puts it on the conductor
            face. This is LpPR's first de-embedding knob.

        MEASURED (copper washer, 100 nm cells, 1e10 Hz): the LpR/LpPR
        port gap is 0.0145 pH at N=20 and 0.0201 at N=32, of which the
        terminal term supplies 0.0091 / 0.0088 -- so roughly half to
        two thirds is this omission and the rest is the prescribed
        profile vs the solved equipotential split.

        WARM STARTS (frequency sweeps): pass the previous point's ``x``
        as ``x0`` and the frequency it was solved at as ``x0_freq``.
        fgmres's stopping test is ``||r|| < tol*||b||`` -- relative to
        the RHS, not to the initial residual -- so a warm start can
        never loosen the answer, only save iterations.

        WHY A PLAIN WARM START IS WORTHLESS HERE, and what works
        (measured on the 160^2 filled pdn, 2026-08-08):
          * The RHS is FREQUENCY-INDEPENDENT -- W and P are
            electrostatic and ``rescaleRHS`` touches only the node
            block -- and its branch block is exactly ZERO. So the
            branch rows are a pure near-cancellation ``Z i + A v ~ 0``
            between terms measured at ~1e4 ||b||.
          * Perturbing w by even 2% therefore leaves
            ``||K(f2) x(f1)|| = 318 ||b||``: the old solution is not
            approximately right, it is enormously wrong in the branch
            rows. Feeding it in raw, or scaled by any single global
            alpha (the least-squares alpha collapses to 1e-5), leaves
            the initial residual at 1.0000 -- exactly a cold start,
            for one wasted probe matvec. Measured across the full
            1e5-1e9 sweep: warm cost +1 matvec at EVERY point, saved
            none.
          * What restores it is BLOCK-WISE scaling, because the two
            blocks follow different frequency laws: a capacitive
            port's potentials run as 1/w while its currents do not.
            Fitting the two scales by least squares recovers
            ``beta = w1/w2`` to 3 digits (0.9804 vs 0.9804 at a 2%
            step), so the scaling is ANALYTIC and needs no probe
            matvec at all. Initial residual at a 2% step: 1.0000
            (global) -> 0.0051 (block). At 1.5x: 0.107. At 10x: 0.787
            -- the lever is real only for FINE sweeps.

        ``x0_mode``: 'physical' (default) scales the node block by
        ``w0/w``; 'global' is the one-probe-matvec least-squares alpha
        kept for comparison; 'none' feeds ``x0`` unscaled.
        """
        from pyamg.krylov import fgmres as _fgmres
        m, M, S = self.model, self.M, self.S
        # Update ONLY the frequency-dependent pieces. Do NOT call
        # model.prepare() here: its allocate() rebinds the leaf data
        # buffers to a fresh array, severing SystemMat's wholedata
        # views -- the matvec then writes one buffer while the tree
        # reads another, and fgmres makes no progress (measured: resid
        # stuck at 1.0).
        re_, rf_, rg_ = m.resistances(M, freq)
        M.e.r, M.f.r, M.g.r = re_, rf_, rg_
        jw = 1j*2*np.pi*freq
        S.jomega = jw
        M.jomega = jw
        if getattr(m, 'subpixel', None):
            # subpixel stage B: the pure-geometry dL, scaled jw per
            # point. The imposed-profile variant (build_dZ) measured
            # WORSE -- see subpixel.py's docstring; enrichment
            # amplitudes must be SOLVED (stage C.2), not imposed.
            if self._dz_static is None:
                from subpixel import build_dL
                self._dz_static = build_dL(m, M)
            S.dZ_near = None if self._dz_static is None \
                else jw*self._dz_static
        if self._precond == 'diagschur':
            # dielectric branch impedances are FREQUENCY DEPENDENT
            # (sigma_eff = jw eps0 (eps_r - 1)); refresh the diagonal
            # before refactoring S_d
            S._Rdiag = np.concatenate([
                np.full(S.esize, M.e.r), np.full(S.fsize, M.f.r),
                np.full(S.gsize, M.g.r)]).astype(np.complex128)
            self.S._factorDiagSchur()
        else:
            self.S._factorReluctanceSchur()
        src = m.source_vector(M, port, current, weight=weight)
        # RAW-system rhs: the raw node rows are P(A^T i) - jw v = s on
        # external nodes and plain KCL on internal ones, so physical
        # injections enter as s_ext = P*(injections) -- ONE traverseP3.
        # The rescaled rhs is then W*s_raw for WHATEVER W is installed.
        # The previous shortcut (injections straight into the rescaled
        # rhs) is valid ONLY for the exact W = P^-1: for any
        # approximate W it solves x = K^-1 W^-1 s instead of K^-1 P s
        # -- wrong by exactly W's deficiency. Measured: this shortcut,
        # not extract_ccap, was the root of the 'band W silently 49%
        # wrong' finding; the multilevel dielectric SYSTEM itself is
        # correct to 0.07% of single-level (dense-probe direct solve).
        M.lv[0].data[:] = src
        M.traverseP3()
        s_raw = np.asarray(M.lv[0].data).copy()
        s_raw[S.intmask] = src[S.intmask]
        raw = np.zeros(S.wholesize, dtype=np.complex128)
        raw[S.efgsize:] = s_raw
        rhs = S.rescaleRHS(raw)
        n0 = S.numiters
        guess = None
        if x0 is not None:
            guess = np.asarray(x0, dtype=np.complex128).copy()
            if x0_mode == 'physical':
                if x0_freq is None:
                    raise ValueError(
                        "x0_mode='physical' needs x0_freq (the "
                        "frequency x0 was solved at) -- the node block "
                        "is scaled by w0/w")
                guess[S.efgsize:] *= float(x0_freq)/float(freq)
            elif x0_mode == 'global':
                kx0 = np.asarray(self.Kprime*guess)
                den = complex(np.vdot(kx0, kx0))
                alpha = (complex(np.vdot(kx0, rhs))/den
                         if abs(den) > 0.0 else 0.0)
                guess *= alpha
            elif x0_mode != 'none':
                raise ValueError("x0_mode must be physical/global/none")
        # maxiter counts RESTART CYCLES of length restrt. The Krylov
        # memory is ~2 * restrt * wholesize * 16 B (flexible GMRES
        # keeps two bases): at large sizes prefer restrt ~100 with
        # several cycles over one long cycle -- a 1.5M-unknown board
        # at restrt 200 was OOM-killed by the basis alone.
        # fgmres fills `rlist` with true residual norms; rlist[0] is the
        # INITIAL residual, so the warm start's worth is measured for
        # free (no probe matvec) -- the diagnostic that turned the
        # global-scaling null into the block-scaling fix.
        rlist = []
        x, flag = _fgmres(self.Kprime, rhs, x0=guess, M=self.P, tol=tol,
                          maxiter=maxiter, restrt=restrt,
                          callback=callback, residuals=rlist)
        warm_r0 = (float(rlist[0]/np.linalg.norm(rhs))
                   if (guess is not None and rlist) else None)
        resid = float(np.linalg.norm(rhs - self.Kprime*x)
                      / np.linalg.norm(rhs))
        # TRUE-residual postcheck on the UNRESCALED operator: the
        # rescaled residual is measured through W and a deficient W
        # can hide arbitrarily large true defects (that is exactly how
        # the rhs-shortcut bug stayed silent). One extra matvec.
        rtrue = np.asarray(S.matvecLpPR(x.copy())) - raw
        # per-block relative: branch rows vs |Z i| scale, node rows vs
        # |s_raw| scale
        num = np.linalg.norm(rtrue)
        den = max(np.linalg.norm(raw), 1e-300)
        true_resid = float(num/den)
        # physical potential is -x_node (see class docstring); the
        # energy pairing against the injections gives V*I, so Z = V/I
        # = pairing / I^2
        z = -complex(np.dot(x[S.efgsize:], src))/(current*current)
        zterm = 0j
        if terminals:
            # series correction: the solved network spans centre to
            # centre, the physical port spans face to face
            zterm = terminal_impedance(m, M, port, freq, weight=weight,
                                       t_l=t_l)
            z += zterm
        info = dict(matvecs=S.numiters - n0, flag=flag, residual=resid,
                    true_residual=true_resid, warm_residual=warm_r0,
                    z_terminal=complex(zterm), weight=weight)
        if true_resid > 1e3*max(tol, 1e-12):
            import warnings
            warnings.warn(
                "LpPRSolver: TRUE residual %.2e far exceeds the "
                "rescaled residual %.2e -- the W rescale is too weak "
                "for this model; treat the answer as unconverged."
                % (true_resid, resid), RuntimeWarning)
        if verbose:
            print("    %.4g Hz: %d matvecs, flag %s, resid %.2e "
                  "(true %.2e)" % (freq, info['matvecs'], flag, resid,
                                   true_resid))
        return z, x, info

    def impedance_matrix(self, freq, current=1.0, weight='corner',
                         keep_drive=None, verbose=False, **kw):
        """Open-circuit impedance matrix at one frequency.

        One :meth:`solve` per driven port (column j; the other ports
        inject nothing, i.e. are open); the off-diagonal V_i is the
        work-conjugate pairing of port i's injection profile with the
        driven node solution -- the :func:`impedance_matrix`
        construction carried to the LpPR path. Reciprocity
        (Z_ij == Z_ji) is therefore a SOLVED property, not an imposed
        one; the input validator gates it.

        ``keep_drive=j`` attaches the port-j-driven filament currents
        to ``infos[j]['i_f']`` (for field export) -- opt-in because
        that vector is efg-sized. ``terminals=True`` (via kw) corrects
        DIAGONAL entries only, like the LpR twin. Returns
        ``(Z, infos)`` with Z ``(nports, nports)`` complex in port
        declaration order.
        """
        n = len(self.model.ports)
        srcs = [np.asarray(self.model.source_vector(
                    self.M, i, current, weight=weight))
                for i in range(n)]
        Z = np.zeros((n, n), dtype=np.complex128)
        infos = []
        for j in range(n):
            zjj, x, info = self.solve(freq, port=j, current=current,
                                      weight=weight, verbose=verbose,
                                      **kw)
            Z[j, j] = zjj
            xn = x[self.S.efgsize:]
            for i in range(n):
                if i != j:
                    Z[i, j] = -complex(np.dot(xn, srcs[i])) \
                        / (current*current)
            if keep_drive == j:
                info['i_f'] = np.asarray(x[:self.S.efgsize])
            infos.append(info)
        return Z, infos
