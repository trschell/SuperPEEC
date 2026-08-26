# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""ASSEMBLY, stage A1: bond wires and the voxel lattice in ONE solve.

The hybrid's last layer. wirekernel supplies exact element kernels,
wirecoupler the near/far coupling into the FMM sweep; this module joins
the two current systems into a single complex-symmetric solve:

  * WIRE unknowns -- element currents of every segment, constrained so
    each segment's elements are PARALLEL BRANCHES between its end nodes
    (the FastHenry connectivity, which is what lets rings carry
    different sector counts and frames rotate freely between segments).
  * VOXEL unknowns -- plaquette mesh (loop) currents, the repo's
    standard divergence-free basis, closed eddy loops only.

STAGE A1 IS GALVANICALLY ISOLATED: wires couple to the voxel bulk
inductively (image screening, eddy losses) but no current crosses
between the systems -- the wire loop is closed EXTERNALLY, exactly the
idealisation of studies/wire_proximity.py, now with the metal
environment present. Stage A2 adds the feet (foot constriction R +
current return through the bulk), which turns the wire loop into an
equiterminal-style port cycle.

THE CURRENT BASIS. With wires w1..wN carrying prescribed series signs
p_j and sharing groups (e.g. A,B parallel against return C), the wire
current vector decomposes as

  i_w = ihat*I  +  S d  +  T s

  ihat  prescribed unit-loop pattern: within a group the total splits
        uniformly over wires, within a segment uniformly over elements
        (any feasible split works; the homogeneous DOFs correct it)
  S     per-segment ZERO-SUM columns (element e_k - e_{k+1}): the
        cross-section redistribution -- skin and proximity
  T     sharing loops (wire j vs wire k within a group, uniform over
        elements): the current-sharing DOF the module question needs
  d, s  solved; the residual equations S^T v_w = 0 and T^T v_w = 0
        enforce equal element voltages within a segment and equal chain
        voltages within a group -- i.e. the parallel connections.

The voxel side contributes Y m (plaquettes); the coupled Galerkin
system [Y|S|T]^T Z [Y|S|T] is complex symmetric and solved with lgmres,
preconditioned by the basis Gram's Cholesky (the repo's standard
mesh-basis preconditioner). Every coupling flows through ONE
traverseRL sweep with the WireCoupler hooks: wire->voxel rides the
injected locals (picking up jomega inside the sweep), voxel->wire
comes back raw in out_w and is scaled here.

LOOP READOUT. V = ihat^T v_w (work-conjugate, bilinear -- the system
is complex symmetric, no conjugation anywhere), which at the solved
point equals the common chain voltage of every wire in the driven
group. Z_loop = V/I. Current sharing is read from the solved segment
totals.

FREQUENCY NOTE (Tier 1, 2026-08-12). The wire shapes enter every
coupling block BILINEARLY as quadrature weights, so WireCoupler
caches the point-resolved geometry kernels once and a frequency
retune is a REWEIGHTING: WireBondSolver.set_frequency(f) ==
prepare + Wire.set_delta + WireCoupler.reweight + r_w -- measured
BIT-IDENTICAL to a fresh build at the new frequency and ~500x
cheaper (0.02 s vs 10.6 s on the validator geometry). Basis,
Gram/AMG, paths, feet, far tables and Wff persist across the sweep.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lgmres

import meshgraph as mg
import wirekernel as wk
from equiterminal import sparse_incidence, filament_cells, node_cells
from wirecoupler import WireCoupler

MU0 = 4e-7*np.pi


# ------------------------------------------------- lattice path machinery

def _forest(B, nn):
    """Spanning forest of the lattice node graph.

    ``B`` is the filament incidence (efg x nn, +1 on the low cell).
    Returns (parent, pedge, psign, comp): per node its BFS parent, the
    filament joining them, the SIGN a unit current takes on that
    filament when flowing parent -> node, and the component label.
    """
    Bc = B.tocoo()
    pos = Bc.data > 0
    k = Bc.row[pos]
    lo = np.zeros(B.shape[0], dtype=np.int64)
    hi = np.zeros(B.shape[0], dtype=np.int64)
    lo[k] = Bc.col[pos]
    hi[Bc.row[~pos]] = Bc.col[~pos]
    # adjacency with edge ids
    import collections
    adj = collections.defaultdict(list)
    for e in range(B.shape[0]):
        adj[lo[e]].append((hi[e], e, +1.0))   # traversing lo->hi = +1
        adj[hi[e]].append((lo[e], e, -1.0))
    parent = np.full(nn, -1, dtype=np.int64)
    pedge = np.full(nn, -1, dtype=np.int64)
    psign = np.zeros(nn)
    comp = np.full(nn, -1, dtype=np.int64)
    ncomp = 0
    for root in range(nn):
        if comp[root] >= 0 or not adj[root]:
            continue
        comp[root] = ncomp
        stack = [root]
        while stack:
            u = stack.pop()
            for v, e, sgn in adj[u]:
                if comp[v] < 0:
                    comp[v] = ncomp
                    parent[v] = u
                    pedge[v] = e
                    psign[v] = sgn
                    stack.append(v)
        ncomp += 1
    return parent, pedge, psign, comp


def _tree_path(parent, pedge, psign, comp, u, v):
    """Signed filament path u -> v through the forest: list of
    (filament, sign). Both nodes must share a component."""
    if comp[u] != comp[v]:
        raise ValueError("nodes are on different conductors")
    seen = {}
    a = u
    d = 0
    while a >= 0:
        seen[a] = d
        if parent[a] < 0:
            break
        a = parent[a]
        d += 1
    b = v
    up_v = []
    while b not in seen:
        up_v.append((pedge[b], psign[b]))     # parent->b, i.e. toward v
        b = parent[b]
    path = []
    a = u
    while a != b:
        path.append((pedge[a], -psign[a]))    # a->parent, against psign
        a = parent[a]
    path.extend(reversed(up_v))
    return path


class Wire:
    """A bond wire: polyline -> chain of straight segments, each with
    the settled 1-4-8-12 cross-section (25 elements).

    ``max_seglen`` subdivides each straight piece; keep it at or below
    the tree's leaf-box extent (WireCoupler refuses longer -- the far
    point source would be wrong by construction) and short enough for
    the O((l/r)^2) far error to sit below the cross-section model's
    ~0.5%.
    """

    def __init__(self, points, radius, sigma, nring=3, nsect=(4, 8, 12),
                 delta=None, max_seglen=None, nr=3, nth=3):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            raise ValueError("a wire needs at least two polyline points")
        self.radius = float(radius)
        self.sigma = float(sigma)
        self.rho = 1.0/self.sigma
        self.segments = []
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            u = (b - a)/L
            # the 1e-9 slack absorbs float slop: 8.000000000000001/2
            # must give 4 segments, not 5
            nsub = 1 if max_seglen is None else max(
                1, int(np.ceil(L/max_seglen - 1e-9)))
            for k in range(nsub):
                p0 = a + u*(L*k/nsub)
                self.segments.append(wk.round_wire(
                    p0, u, L/nsub, radius, nring=nring, nsect=nsect,
                    delta=delta, nr=nr, nth=nth))

    def set_delta(self, delta):
        """Retune every element's skin shape to the given depth (None
        = uniform). Geometry untouched; cached coupling kernels stay
        valid -- pair with WireCoupler.reweight()."""
        for seg in self.segments:
            for f in seg:
                if delta is None or delta <= 0:
                    f.set_shape(None)
                else:
                    rr = np.linalg.norm(f.off, axis=1)
                    f.set_shape(np.exp(-(self.radius - rr)/delta))

    def relements(self):
        """Per-element series resistance, chain order (shape-aware:
        Filament.resistance carries int(phi^2)/int(phi)^2)."""
        return np.array([f.resistance(self.rho)
                         for seg in self.segments for f in seg])


class WireEddySolver:
    """Stage A1: wires + voxel eddy currents, externally closed loop.

    Parameters
    ----------
    model, M : the prepared VoxelModel and Tree (prepare() is called
        per solve; the tree must be multilevel for the far coupling).
    wires : list of :class:`Wire`
    loop : sequence of +-1, one per wire -- the prescribed series
        direction of the unit loop current along each chain.
    groups : list of lists of wire indices, optional. Wires in one
        group connect in PARALLEL (share the group's total current,
        split solved); default: every wire its own group. The loop
        pattern gives each GROUP a total of +-1 times the drive.
    """

    def __init__(self, model, M, wires, loop, groups=None, nq=4, ng=16,
                 verbose=False):
        t0 = time.perf_counter()
        self.model, self.M = model, M
        self.wires = wires
        self.loop = np.asarray(loop, dtype=float)
        if self.loop.size != len(wires):
            raise ValueError("one loop sign per wire")
        if groups is None:
            groups = [[j] for j in range(len(wires))]
        self.groups = groups
        seen = sorted(j for g in groups for j in g)
        if seen != list(range(len(wires))):
            raise ValueError("groups must partition the wires")
        segments = [seg for w in wires for seg in w.segments]
        self.wire_of_seg = np.concatenate(
            [np.full(len(w.segments), j) for j, w in enumerate(wires)])
        self.wc = WireCoupler(M, segments, nq=nq, ng=ng)
        wcs = self.wc
        self.nwel = wcs.nwel
        self.r_w = np.concatenate([w.relements() for w in wires])
        # element bookkeeping: which wire, which segment, element count
        self.elem_wire = np.concatenate(
            [np.full(wcs.seg0[s + 1] - wcs.seg0[s], self.wire_of_seg[s])
             for s in range(len(segments))]).astype(int)
        # prescribed unit-loop pattern
        self.ihat = np.zeros(self.nwel)
        for g in self.groups:
            for j in g:
                pj = self.loop[j]/len(g)
                for s in np.where(self.wire_of_seg == j)[0]:
                    a, b = wcs.seg0[s], wcs.seg0[s + 1]
                    self.ihat[a:b] = pj/(b - a)
        # S: per-segment zero-sum chains e_k - e_{k+1}
        rows, cols, vals = [], [], []
        col = 0
        for s in range(len(segments)):
            a, b = wcs.seg0[s], wcs.seg0[s + 1]
            for e in range(a, b - 1):
                rows += [e, e + 1]
                cols += [col, col]
                vals += [1.0, -1.0]
                col += 1
        self.S = sp.csr_matrix((vals, (rows, cols)),
                               shape=(self.nwel, col))
        # T: sharing loops within each group (wire j vs the group's
        # first wire), uniform over each wire's elements, respecting
        # the chain signs so a column is a genuine circulation
        rows, cols, vals = [], [], []
        col = 0
        for g in self.groups:
            for j in g[1:]:
                for w, sgn in ((g[0], -1.0), (j, +1.0)):
                    pj = sgn*self.loop[w]
                    for s in np.where(self.wire_of_seg == w)[0]:
                        a, b = wcs.seg0[s], wcs.seg0[s + 1]
                        rows += list(range(a, b))
                        cols += [col]*(b - a)
                        vals += [pj/(b - a)]*(b - a)
                col += 1
        self.T = sp.csr_matrix((vals, (rows, cols)),
                               shape=(self.nwel, col))
        # voxel plaquette basis. The SELECTED set spans ANY topology
        # (getmesh_fortran falls back to MST fundamental cycles for
        # what squares cannot reach -- correct always, slow on
        # coil-class topologies). The rank check below is an INTEGRITY
        # assert on that construction, not a topology restriction.
        # Scaling note: AMG is only effective on the OVERCOMPLETE
        # basis (getmesh_full; recorded in the loop-preconditioner
        # work), so when models outgrow the Gram Cholesky the upgrade
        # is overcomplete + _BlockAMGFactor with the wire columns in
        # the exact macro block -- the Gram here is exactly
        # block-diagonal (Y^T S = Y^T T = S^T T = 0), so the swap is
        # contained to this block.
        self.efg = (np.size(M.e.struc) + np.size(M.f.struc)
                    + np.size(M.g.struc))
        self.nnode = np.size(M.lv[0].struc)
        self.whole = M._vhr_whole
        if self.whole is None or self.whole.size != self.efg + self.nnode:
            raise RuntimeError("tree buffer not allocated -- call "
                               "model.prepare(M, freq) first")
        self.B, _ = sparse_incidence(M, self.whole, self.efg, self.nnode)
        Y = mg.getmesh_fortran(M.adjmats(), np.size(M.e.struc),
                               np.size(M.e.struc) + np.size(M.f.struc),
                               self.efg, self.nnode)
        Y.data = np.float64(Y.data)
        self.Y = sp.csc_matrix(Y)
        self.YT = self.Y.T.tocsc()
        # completeness + divergence-free: the two checks with teeth
        import scipy.sparse.csgraph as csg
        adj = self.B.T @ self.B
        ncomp = csg.connected_components(adj, directed=False,
                                         return_labels=False)
        want = self.efg - self.nnode + ncomp
        if self.Y.shape[1] != want:
            raise RuntimeError(
                "selected cycle basis has %d columns for a cycle space "
                "of %d -- getmesh_fortran's spanning construction broke, "
                "which is a bug, not a topology limit"
                % (self.Y.shape[1], want))
        div = np.abs((self.B.T @ self.Y)).max()
        if div > 1e-12:
            raise RuntimeError("mesh columns are not divergence-free "
                               "(max %g)" % div)
        self.nm = self.Y.shape[1]
        self.nd = self.S.shape[1]
        self.ns = self.T.shape[1]
        self.size = self.nm + self.nd + self.ns
        # Gram preconditioner (block: the wire blocks are tiny)
        from sksparse.cholmod import cholesky
        G = sp.bmat([[self.YT @ self.Y, None, None],
                     [None, (self.S.T @ self.S).tocsc(),
                      (self.S.T @ self.T).tocsc()],
                     [None, (self.T.T @ self.S).tocsc(),
                      (self.T.T @ self.T).tocsc()]], format='csc')
        self.chol = cholesky(G)
        self.matvecs = 0
        self.t_setup = time.perf_counter() - t0
        if verbose:
            print("  %d wire elements (%d segments, %d wires), "
                  "%d plaquettes, %d + %d wire DOFs; near %.1f%%; "
                  "setup %.2f s"
                  % (self.nwel, len(segments), len(wires), self.nm,
                     self.nd, self.ns, 100*self.wc.near_frac,
                     self.t_setup))

    # -- the coupled operator -------------------------------------------

    def _coupled(self, i_f, i_w):
        """(v_f, v_w) = Z [i_f; i_w] through one FMM sweep."""
        wcs, M = self.wc, self.M
        self.whole[:self.efg] = i_f
        wcs.i_f = np.ascontiguousarray(i_f, dtype=np.complex128)
        wcs.i_w = np.ascontiguousarray(i_w, dtype=np.complex128)
        wcs.out_w = np.zeros(self.nwel, dtype=np.complex128)
        M.traverseRL(extra=wcs)
        v_f = np.array(self.whole[:self.efg])
        v_w = (self.r_w*i_w
               + M.jomega*(wcs.wire_matvec(i_w) + wcs.out_w))
        return v_f, v_w

    def _expand(self, x):
        w = x[:self.nm]
        d = x[self.nm:self.nm + self.nd]
        s = x[self.nm + self.nd:]
        return self.Y @ w, self.S @ d + self.T @ s

    def _matvec(self, x):
        self.matvecs += 1
        i_f, i_w = self._expand(x)
        v_f, v_w = self._coupled(i_f, i_w)
        return np.concatenate([self.YT @ v_f, self.S.T @ v_w,
                               self.T.T @ v_w])

    def _precond(self, vec):
        return (self.chol(np.real(vec))
                + 1j*self.chol(np.imag(vec)))

    # -- the solve -------------------------------------------------------

    def solve(self, freq, current=1.0, rtol=1e-10, maxiter=30,
              inner_m=None, verbose=False, method='lgmres',
              precision='auto'):
        """Solve at one frequency.

        Returns ``(Z, info)``: the driven loop's impedance V/I, and a
        dict with the solved per-wire current shares (fraction of the
        group total, from segment totals), matvec count, true residual.
        ``method``: see :func:`port_impedance.krylov_solve`.
        """
        self.model.prepare(self.M, freq)
        t0 = time.perf_counter()
        v_f0, v_w0 = self._coupled(np.zeros(self.efg, np.complex128),
                                   self.ihat*current)
        rhs = -np.concatenate([self.YT @ v_f0, self.S.T @ v_w0,
                               self.T.T @ v_w0])
        Aop = LinearOperator((self.size,)*2, matvec=self._matvec,
                             dtype=np.complex128)
        Pop = LinearOperator((self.size,)*2, matvec=self._precond,
                             dtype=np.complex128)
        n0 = self.matvecs
        from port_impedance import krylov_solve
        x, flag = krylov_solve(Aop, rhs, Pop, method=method, rtol=rtol,
                               maxiter=maxiter, inner_m=inner_m,
                               precision=precision)
        nrhs = np.linalg.norm(rhs)
        resid = (np.linalg.norm(rhs - Aop @ x)/nrhs if nrhs > 0 else 0.0)
        i_f, i_hom = self._expand(x)
        i_w = self.ihat*current + i_hom
        v_f, v_w = self._coupled(i_f, i_w)
        V = complex(self.ihat @ v_w)
        # solved sharing: per-wire chain current (mean over its
        # segments' totals; the spread is a consistency diagnostic)
        share = np.zeros(len(self.wires), dtype=np.complex128)
        spread = 0.0
        for j in range(len(self.wires)):
            tots = []
            for s in np.where(self.wire_of_seg == j)[0]:
                a, b = self.wc.seg0[s], self.wc.seg0[s + 1]
                tots.append(i_w[a:b].sum())
            tots = np.array(tots)
            share[j] = tots.mean()/current
            if np.abs(tots.mean()) > 0:
                spread = max(spread, float(np.abs(tots - tots.mean()).max()
                                           / np.abs(tots.mean())))
        info = dict(matvecs=self.matvecs - n0, flag=flag, residual=resid,
                    share=share, share_spread=spread,
                    time=time.perf_counter() - t0,
                    i_w=i_w, i_f=i_f)
        if verbose:
            print("    %.4g Hz: %d matvecs, flag %s, resid %.2e, "
                  "%.2f s; shares %s"
                  % (freq, info['matvecs'], flag, resid, info['time'],
                     np.round(np.real(share), 4)))
        return complex(V)/current, info


# ------------------------------------------------------------- stage A2

class WireBondSolver:
    """Stage A2: wires GALVANICALLY attached to the voxel bulk.

    The full module circuit: a port (strip of lattice cells on each
    side, equal prescribed split) drives current that flows through the
    bulk, up through each wire's FOOT, across the wire, and back down
    -- loop resistance, inductance and per-wire current sharing all
    solved together, with every inductive coupling (wire<->wire,
    wire<->bulk, bulk skin/eddy) carried by the one FMM sweep.

    WHAT IS DERIVED, NOT SPECIFIED:
      feet   -- the first/last polyline point of each wire is its
                contact; the foot cell is the nearest occupied cell
                within one pitch (loud error otherwise). Keep the
                endpoint a fraction of a cell ABOVE the surface so the
                end face's cross-section stays clear of the metal (the
                overlap rule); the gap is inside the constriction
                model's physics anyway.
      loops  -- conductor components + wires form a coarse circuit
                graph; a spanning tree of it routes the DRIVEN cycle
                (port -> tree wires -> port) and every chord wire
                contributes one SHARING cycle. Lattice legs come from
                a spanning forest of the node graph (path choice is
                gauge: any two routes differ by plaquettes, which are
                in the basis -- part I of the validator PROVES the
                answer is route-independent).
      foot R -- wirekernel.foot_constriction (equipotential disc,
                rho of the pad under each foot, radius ``foot_r0``,
                default 2x wire radius: the flattened-bond rule). It
                enters as a series term on the CHAIN current through
                the aggregation A_foot -- no extra unknowns, exact for
                every chain-consistent basis vector, and the
                measured sensitivity (L: 8 ppm, sharing: 0.02%, AC
                loop R: ~1.4%) bounds what the unfinished
                discretisation calibration can move.

    KCL IS ASSERTED, NOT ASSUMED: every basis column must satisfy
    B^T i_f + Bw I_chain = 0 exactly, and the prescribed pattern must
    match the port injection; a sign error in any path dies at build.

    The port is NODAL with a prescribed equal split over the strip
    (the dense_oracle 'prescribed' pattern). The recorded caveat
    applies: a small port makes the port-local term dx-dependent --
    use a strip of several cells and hold it PHYSICALLY fixed under
    refinement.
    """

    def __init__(self, model, M, wires, port_p, port_n, foot_r0=None,
                 nq=4, ng=16, root0=0, foot_model='patch',
                 basis='auto', amg_cycles=4, gram_solver='geo',
                 verbose=False):
        t0 = time.perf_counter()
        if foot_model not in ('patch', 'point'):
            raise ValueError("foot_model must be 'patch' (calibrated "
                             "footprint, the default) or 'point' "
                             "(legacy single-cell + analytic disc)")
        if basis not in ('auto', 'selected', 'overcomplete'):
            raise ValueError("basis must be 'auto' (size-evaluated, "
                             "the default), 'selected' (independent "
                             "plaquette set + exact Gram Cholesky) or "
                             "'overcomplete' (ALL plaquettes + "
                             "BlockAMG/Schur -- the memory-flat scaling "
                             "path)")
        if gram_solver not in ('geo', 'amg'):
            raise ValueError("gram_solver must be 'geo' (geometric "
                             "multigrid, the default) or 'amg' "
                             "(smoothed aggregation)")
        self.foot_model = foot_model
        self.basis = basis
        self.amg_cycles = int(amg_cycles)
        # DEFAULT 'geo' since 2026-08-13 (decided with the user).
        # Only engages under basis='overcomplete' -- below the
        # auto-basis threshold the Gram gets an exact Cholesky and
        # this knob is moot. Evidence: GeoMG's factor build is
        # near-free where SA coarsening is the dominant setup cost at
        # scale (R4 62 s vs the AMG-era 1373 s total setup), apply and
        # iteration count are at PARITY (R3 head-to-head 247 mv both),
        # and under fp32 phase 1 geo improved (173->166 mv, apply
        # 1.6x) where amg lost 15% of its iterations. AMG stays
        # selectable for geometries where lattice-regular coarsening
        # is a bad fit and algebraic aggregation can adapt.
        self.gram_solver = str(gram_solver)
        self.model, self.M = model, M
        self.wires = wires
        segments = [seg for w in wires for seg in w.segments]
        self.wire_of_seg = np.concatenate(
            [np.full(len(w.segments), j) for j, w in enumerate(wires)])
        self.nseg_of_wire = np.array([len(w.segments) for w in wires])
        self.wc = WireCoupler(M, segments, nq=nq, ng=ng)
        wcs = self.wc
        self.nwel = wcs.nwel
        self.r_w = np.concatenate([w.relements() for w in wires])
        self.efg = (np.size(M.e.struc) + np.size(M.f.struc)
                    + np.size(M.g.struc))
        self.nnode = np.size(M.lv[0].struc)
        self.whole = M._vhr_whole
        if self.whole is None or self.whole.size != self.efg + self.nnode:
            raise RuntimeError("tree buffer not allocated -- call "
                               "model.prepare(M, freq) first")
        self.B, ncell = sparse_incidence(M, self.whole, self.efg,
                                         self.nnode)
        self.node_of_cell = {tuple(c): i for i, c in enumerate(ncell)}
        # share the coupler's copies -- filament_cells is linear in N
        self.fil_axis, self.fil_cell = wcs.fil_axis, wcs.fil_cell
        self.parent, self.pedge, self.psign, self.comp = _forest(
            self.B, self.nnode)
        # rotate the BFS root: gauge for the path system, used by the
        # validator to PROVE route independence
        if root0:
            order = np.roll(np.arange(self.nnode), -root0)
            par, ped, psn, cmp_ = _forest_rooted(self.B, self.nnode,
                                                 order)
            self.parent, self.pedge, self.psign, self.comp = \
                par, ped, psn, cmp_
        # feet: anchor cell per contact, then (for the calibrated
        # 'patch' model) the coverage-weighted footprint on the pad
        # surface, with the deficit resistance from footcal. The
        # 'point' model is the same machinery degenerated to a
        # one-cell patch with the legacy analytic-disc series R.
        struc = model.struc()
        l = wcs.l
        if foot_model == 'patch' and not np.allclose(l, l[0]):
            raise NotImplementedError(
                "the foot calibration's lattice Green's function is "
                "cubic-only (pitch %s); use foot_model='point' or "
                "resample" % list(l))
        self.foot_cell = []
        self.foot_patch = []      # per wire: per end: (nodes, weights)
        self.Rfoot = np.zeros(len(wires))
        for j, w in enumerate(wires):
            f0 = w.segments[0][0]
            fN = w.segments[-1][0]
            ends = [f0.p0, fN.p0 + fN.length*fN.u]
            r0 = (2.0*w.radius if foot_r0 is None
                  else (foot_r0[j] if np.ndim(foot_r0) else foot_r0))
            cells, patches = [], []
            for p in ends:
                anchor = self._find_foot(p, struc, l, r0)
                cells.append(anchor)
                rho_pad = 1.0/float(model.sigma[tuple(anchor)])
                if foot_model == 'point':
                    patches.append(([self.node_of_cell[tuple(anchor)]],
                                    np.array([1.0])))
                    self.Rfoot[j] += wk.foot_constriction(r0, rho_pad)
                else:
                    patches.append(self._patch(anchor, p, r0, struc, l))
                    import footcal
                    self.Rfoot[j] += footcal.foot_resistance(
                        r0, float(l[0]), rho_pad)
            self.foot_cell.append(cells)
            self.foot_patch.append(patches)
        self.foot_node = [[self.node_of_cell[tuple(c)] for c in fc]
                          for fc in self.foot_cell]
        # chain aggregation and foot incidence (patch cells carry the
        # coverage weights)
        rows, cols, vals = [], [], []
        for s in range(len(segments)):
            j = self.wire_of_seg[s]
            a, b = wcs.seg0[s], wcs.seg0[s + 1]
            rows += [j]*(b - a)
            cols += list(range(a, b))
            vals += [1.0/self.nseg_of_wire[j]]*(b - a)
        self.Afoot = sp.csr_matrix((vals, (rows, cols)),
                                   shape=(len(wires), self.nwel))
        rows, cols, vals = [], [], []
        for j in range(len(wires)):
            for e, sgn in ((0, +1.0), (1, -1.0)):
                nodes, wts = self.foot_patch[j][e]
                rows += list(nodes)
                cols += [j]*len(nodes)
                vals += list(sgn*wts)
        self.Bw = sp.csr_matrix((vals, (rows, cols)),
                                shape=(self.nnode, len(wires)))
        # port strips
        self.port_p = [self.node_of_cell[tuple(c)] for c in port_p]
        self.port_n = [self.node_of_cell[tuple(c)] for c in port_n]
        # coarse circuit graph -> driven route + sharing chords
        self._bundles = {}
        self._build_cycles()
        # voxel plaquettes + per-segment zero-sum, as stage A1
        import scipy.sparse.csgraph as csg
        ncomp = csg.connected_components(self.B.T @ self.B,
                                         directed=False,
                                         return_labels=False)
        want = self.efg - self.nnode + ncomp
        if basis == 'auto':
            # Threshold SET BY THE USER 2026-08-12 at 1e6 CELLS, on
            # the rationale "users probably won't complain about
            # ~3.7 GB of RAM going to a good cause" -- the measured
            # Cholesky cost at the largest sub-threshold anchor
            # (square_coil, 605k cells: 3.5 GB precond, and FASTER
            # wall-clock than overcomplete there, 590 vs 630 s).
            # Above it, fill grows superlinearly and the memory-flat
            # overcomplete + BlockAMG(+GPU) path takes over.
            # Deliberately a round number, not a measured crossover;
            # revisit when scaling raises the memory stakes.
            basis = ('selected' if np.prod(model.dims) < 1e6
                     else 'overcomplete')
            self.basis = basis
            was_auto = True
        else:
            was_auto = False
        if basis == 'overcomplete':
            # ALL plaquettes: singular Gram (kernel = cube boundaries,
            # physically meaningless), far better conditioned on its
            # range, and AMG works there -- the memory-flat path the
            # refinement ladder needs. The count check is NECESSARY,
            # not sufficient (the recorded caveat): a multiply
            # connected bulk needs hole generators no square spans --
            # equiterminal._hole_cycles is the shipped pattern, not
            # yet wired here (the flagship's traces and plate are
            # simply connected; docketed for perforated bulks).
            Y = mg.getmesh_full(M.adjmats(), np.size(M.e.struc),
                                np.size(M.e.struc) + np.size(M.f.struc),
                                self.efg, self.nnode)
            if Y.shape[1] < want:
                if was_auto:
                    import warnings
                    warnings.warn(
                        "basis='auto': the bulk is multiply connected "
                        "(%d plaquettes for cycle space %d) and hole "
                        "generators are not wired into this solver -- "
                        "falling back to the SELECTED basis (exact "
                        "Cholesky, heavier at scale)"
                        % (Y.shape[1], want), RuntimeWarning,
                        stacklevel=2)
                    basis = self.basis = 'selected'
                else:
                    raise RuntimeError(
                        "over-complete plaquette basis spans %d columns "
                        "for a cycle space of %d -- this bulk is "
                        "multiply connected and needs hole generators "
                        "(see equiterminal._hole_cycles); use "
                        "basis='selected'" % (Y.shape[1], want))
        if basis == 'selected':
            Y = mg.getmesh_fortran(M.adjmats(), np.size(M.e.struc),
                                   np.size(M.e.struc)
                                   + np.size(M.f.struc),
                                   self.efg, self.nnode)
            if Y.shape[1] != want:
                raise RuntimeError("selected cycle basis broke: %d "
                                   "columns for cycle rank %d"
                                   % (Y.shape[1], want))
        Y.data = np.float64(Y.data)
        rows, cols, vals = [], [], []
        col = 0
        for s in range(len(segments)):
            a, b = wcs.seg0[s], wcs.seg0[s + 1]
            for e in range(a, b - 1):
                rows += [e, e + 1]
                cols += [col, col]
                vals += [1.0, -1.0]
                col += 1
        S = sp.csr_matrix((vals, (rows, cols)), shape=(self.nwel, col))
        # stacked basis over [i_f; i_w]
        Tf = sp.csr_matrix((self.efg, len(self._chords)))
        Tw = sp.csr_matrix((self.nwel, len(self._chords)))
        if self._chords:
            Tf = sp.hstack([c[0] for c in self._chords], format='csc')
            Tw = sp.hstack([c[1] for c in self._chords], format='csc')
        self.Bmat = sp.bmat([[sp.csc_matrix(Y), None, Tf],
                             [None, S, Tw]], format='csc')
        self.size = self.Bmat.shape[1]
        self.nplaq, self.nd, self.nchord = Y.shape[1], S.shape[1], \
            Tf.shape[1]
        # KCL asserts with teeth
        self._assert_kcl()
        if basis == 'overcomplete' and self.gram_solver == 'geo':
            # GEOMETRIC MG on the plaquettes, exact Schur over the
            # CHORDS ONLY, and the distribution block on its own tiny
            # Cholesky -- the exact decoupling (Y^T S = S^T T = 0)
            # makes this identical in effect to putting S in the local
            # block. DO NOT hand the S columns to _GeoMGFactor's
            # positional macro split: its Schur setup materialises a
            # DENSE (n_local x n_macro) intermediate -- 58 GB at R3
            # with the ~3200 distribution columns, which is what
            # OOM-killed the first benchmarks (2026-08-12).
            import loopmg
            from port_impedance import _GeoMGFactor
            from sksparse.cholmod import cholesky
            idx_yt = np.r_[0:self.nplaq,
                           self.nplaq + self.nd:self.size]
            Byt = self.Bmat[:, idx_yt].T.tocsr().astype(np.float32)
            nrm, bse = loopmg.plaquette_geometry(
                self.Bmat[:self.efg, :self.nplaq].tocsc(),
                self.fil_axis, self.fil_cell, self.nplaq)
            geo = _GeoMGFactor(Byt, nrm, bse, self.nplaq,
                               cycles=self.amg_cycles)
            del Byt                # setup transient: the factor holds
            del nrm, bse           # everything it needs
            if verbose:
                print("GeoMG precond apply: %s" % geo.gpu_state,
                      flush=True)
            Ssub = self.Bmat[:, self.nplaq:self.nplaq + self.nd]
            cholS = cholesky((Ssub.T @ Ssub).tocsc())
            nplaq, nd, size = self.nplaq, self.nd, self.size

            class _GeoSplit:
                nnz_ratio = geo.nnz_ratio
                nmac = geo.nmac

                def __call__(self, b):
                    b = np.float64(b)
                    yt = np.concatenate([b[:nplaq], b[nplaq + nd:]])
                    g = np.float64(geo(yt))
                    out = np.empty(size)
                    out[:nplaq] = g[:nplaq]
                    out[nplaq + nd:] = g[nplaq:]
                    out[nplaq:nplaq + nd] = cholS(b[nplaq:nplaq + nd])
                    return np.float32(out)

            self.chol = _GeoSplit()
            if verbose:
                print("    GeoMG split: hierarchy %.2fx nnz, %d chord "
                      "Schur cols, S-block exact (%d)"
                      % (geo.nnz_ratio, geo.nmac, self.nd))
        elif basis == 'overcomplete':
            # BlockAMG + exact Schur: LOCAL block = plaquettes (+ the
            # distribution columns, whose per-segment chain Gram is an
            # M-matrix SA coarsens trivially, and which couple to
            # nothing: Y^T S = S^T T = 0 identically); MACRO block =
            # the wire sharing chords -- the LONG loops, outside AMG's
            # aggregation, exactly as decided. Hole generators would
            # join the macro block when wired in.
            from port_impedance import _BlockAMGFactor
            YT32 = self.Bmat.T.tocsr().astype(np.float32)
            macro = np.arange(self.nplaq + self.nd, self.size)
            self.chol = _BlockAMGFactor(YT32, macro,
                                        cycles=self.amg_cycles)
            if verbose:
                print("    BlockAMG: hierarchy %.2fx nnz, %d macro "
                      "columns (exact Schur)"
                      % (self.chol.nnz_ratio, self.chol.nmac))
        else:
            from sksparse.cholmod import cholesky
            G = (self.Bmat.T @ self.Bmat).tocsc()
            self.chol = cholesky(G)
            del G                  # setup transient
        # the stacked basis and incidence carry +-1s and small
        # dyadics: float32 stores them exactly, halving their bytes
        # with every product bit-unchanged (refused if any entry
        # would round). AFTER the factor builds: those consumed
        # full-precision copies of their own.
        from port_impedance import shrink_exact_f32
        shrink_exact_f32(self.Bmat)
        shrink_exact_f32(self.B)
        self.matvecs = 0
        self.t_setup = time.perf_counter() - t0
        if verbose:
            print("  %d wire elements / %d wires, %d plaquettes, %d "
                  "dist + %d sharing DOFs; feet %s; near %.1f%%; "
                  "setup %.1f s"
                  % (self.nwel, len(wires), self.nplaq, self.nd,
                     self.nchord,
                     [tuple(map(tuple, fc)) for fc in self.foot_cell],
                     100*self.wc.near_frac, self.t_setup))

    # -- construction helpers -------------------------------------------

    def _find_foot(self, p, struc, l, r0):
        """Nearest occupied cell within a PHYSICAL radius of the
        contact: max(one pitch, the foot radius r0). The radius must
        be physical, not a cell count -- the contact standoff above
        the surface is fixed geometry while cells shrink under
        refinement (a 0.06 mm standoff is 0.3 cells at demo pitch and
        1.2 cells three rungs down, where a 27-neighbourhood search
        starved -- found at ladder R3). Anything within ~r0 of the
        surface is a legitimate landing; the constriction model owns
        the gap."""
        R = max(float(np.linalg.norm(l)), float(r0))
        nscan = np.maximum(1, np.ceil(R/l).astype(np.int64))
        c0 = np.floor(p/l).astype(np.int64)
        best, bestd, tie = None, np.inf, False
        for dx in range(-nscan[0], nscan[0] + 1):
            for dy in range(-nscan[1], nscan[1] + 1):
                for dz in range(-nscan[2], nscan[2] + 1):
                    c = c0 + [dx, dy, dz]
                    if np.any(c < 0) or np.any(c >= struc.shape):
                        continue
                    if not struc[tuple(c)]:
                        continue
                    d = np.linalg.norm((c + 0.5)*l - p)
                    if abs(d - bestd) < 1e-12*float(l.min()):
                        tie = True
                    elif d < bestd:
                        best, bestd, tie = c.copy(), d, False
        if best is None or bestd > R:
            raise ValueError("wire endpoint %s has no occupied cell "
                             "within %g m (max of one pitch and the "
                             "foot radius) -- feet must land on the "
                             "conductor" % (p, R))
        if tie:
            # MEASURED trap (2026-08-11): a contact exactly on a cell
            # boundary gets its foot tie-broken by iteration order --
            # half a cell of silent placement bias that showed up as a
            # 9e-4 sharing asymmetry between geometrically mirrored
            # wires and survived at DC (which is what exonerated the
            # cross-section seams and convicted the feet). Contacts
            # belong over cell CENTRES.
            import warnings
            warnings.warn(
                "wire contact %s is equidistant from several occupied "
                "cells (it sits on a cell boundary); the foot cell %s "
                "was tie-broken arbitrarily -- move the contact over a "
                "cell centre for a well-defined foot"
                % (list(p), list(best)), RuntimeWarning, stacklevel=3)
        return best

    def _patch(self, anchor, p, r0, struc, l):
        """The contact FOOTPRINT: surface cells covered by the disc of
        radius ``r0`` around the contact, with footcal's coverage
        weights -- the same offsets/weights the calibration integrates
        over, so the injected split and the calibrated deficit are the
        same object. Cells whose outward neighbour is metal are buried,
        not surface, and are dropped (with a warning if the clip loses
        real weight -- a foot hanging over a pad edge)."""
        import footcal
        d = p - (anchor + 0.5)*l
        naxis = int(np.argmax(np.abs(d)))
        sgn = 1 if d[naxis] >= 0 else -1
        taxes = [k for k in range(3) if k != naxis]
        off2, wt = footcal.patch_offsets(r0/float(l[0]))
        nodes, ws = [], []
        for (i1, i2), wcov in zip(off2, wt):
            c = anchor.copy()
            c[taxes[0]] += i1
            c[taxes[1]] += i2
            if np.any(c < 0) or np.any(c >= struc.shape):
                continue
            if not struc[tuple(c)]:
                continue
            cout = c.copy()
            cout[naxis] += sgn
            if (0 <= cout[naxis] < struc.shape[naxis]
                    and struc[tuple(cout)]):
                continue                    # buried, not surface
            nodes.append(self.node_of_cell[tuple(c)])
            ws.append(wcov)
        ws = np.asarray(ws)
        if ws.size == 0:
            raise ValueError("foot at cell %s: no surface cells under "
                             "the contact footprint" % (list(anchor),))
        if ws.sum() < 0.75:
            import warnings
            warnings.warn(
                "foot at cell %s: only %.0f%% of the contact footprint "
                "lies on the pad surface (edge overhang?) -- the "
                "calibration assumes a full disc"
                % (list(anchor), 100*ws.sum()), RuntimeWarning,
                stacklevel=3)
        return nodes, ws/ws.sum()

    def _bundle(self, j, e):
        """Sparse (efg, 1) column distributing unit current from foot
        (j, e)'s anchor node over its patch cells with the coverage
        weights -- the lattice-side half of a patch foot. Cached."""
        key = (j, e)
        if key not in self._bundles:
            nodes, ws = self.foot_patch[j][e]
            anchor = self.foot_node[j][e]
            col = sp.csc_matrix((self.efg, 1))
            for n, w in zip(nodes, ws):
                if n != anchor:
                    col = col + self._path_col(anchor, n)*float(w)
            self._bundles[key] = col
        return self._bundles[key]

    def _path_into(self, rows, vals, u, v, w=1.0):
        """Accumulate a unit-current path u -> v (scaled by ``w``)
        into row/value lists -- ONE sparse construction at the end
        instead of a sparse ADD per path (which was 62% of R3 setup:
        ~2300 port cells x O(nnz) additions)."""
        for e, sgn in _tree_path(self.parent, self.pedge, self.psign,
                                 self.comp, u, v):
            rows.append(e)
            vals.append(sgn*w)

    def _bundle_into(self, rows, vals, j, e, w=1.0):
        """Accumulate foot (j, e)'s patch distribution bundle."""
        nodes, ws = self.foot_patch[j][e]
        anchor = self.foot_node[j][e]
        for n, wc in zip(nodes, ws):
            if n != anchor:
                self._path_into(rows, vals, anchor, n, float(wc)*w)

    def _path_col(self, u, v):
        """Sparse (efg,1) column carrying unit current u -> v."""
        rows, vals = [], []
        for e, sgn in _tree_path(self.parent, self.pedge, self.psign,
                                 self.comp, u, v):
            rows.append(e)
            vals.append(sgn)
        return sp.csc_matrix((vals, (rows, [0]*len(rows))),
                             shape=(self.efg, 1))

    def _wire_col(self, j, sgn):
        """Sparse (nwel,1) column: unit chain current through wire j."""
        rows, vals = [], []
        for s in np.where(self.wire_of_seg == j)[0]:
            a, b = self.wc.seg0[s], self.wc.seg0[s + 1]
            rows += list(range(a, b))
            vals += [sgn/(b - a)]*(b - a)
        return sp.csc_matrix((vals, (rows, [0]*len(rows))),
                             shape=(self.nwel, 1))

    def _build_cycles(self):
        """Coarse circuit graph: components as supernodes, wires as
        edges. A spanning tree routes the driven port current; chord
        wires become sharing cycles."""
        nw = len(self.wires)
        wcomp = [(self.comp[fn[0]], self.comp[fn[1]])
                 for fn in self.foot_node]
        # union-find over components using wires
        root = {}

        def find(a):
            root.setdefault(a, a)
            while root[a] != a:
                root[a] = root[root[a]]
                a = root[a]
            return a
        tree_wires = []
        chord_wires = []
        for j, (c1, c2) in enumerate(wcomp):
            if find(c1) != find(c2):
                root[find(c1)] = find(c2)
                tree_wires.append(j)
            else:
                chord_wires.append(j)
        cP = self.comp[self.port_p[0]]
        cN = self.comp[self.port_n[0]]
        if any(self.comp[n] != cP for n in self.port_p) or \
           any(self.comp[n] != cN for n in self.port_n):
            raise ValueError("a port strip spans several conductors")
        if find(cP) != find(cN):
            raise ValueError("port P and N are not connected, even "
                             "through the wires: no return path")
        # coarse adjacency over tree wires only; BFS route cP -> cN
        import collections
        adj = collections.defaultdict(list)
        for j in tree_wires:
            c1, c2 = wcomp[j]
            adj[c1].append((c2, j, +1))
            adj[c2].append((c1, j, -1))
        prev = {cP: None}
        q = collections.deque([cP])
        while q:
            c = q.popleft()
            if c == cN:
                break
            for c2, j, sgn in adj[c]:
                if c2 not in prev:
                    prev[c2] = (c, j, sgn)
                    q.append(c2)
        route = []
        c = cN
        while prev[c] is not None:
            cprev, j, sgn = prev[c]
            route.append((j, sgn))
            c = cprev
        route.reverse()
        self._route = route
        # prescribed pattern: port P cells -> route -> port N cells.
        # Accumulate into lists, build ONCE: the per-cell sparse-add
        # version was 62% of R3 setup (large physical port strips).
        fr, fv = [], []
        wr, wv_ = [], []

        def _wire_into(j, sgn):
            for seg in np.where(self.wire_of_seg == j)[0]:
                a, b = self.wc.seg0[seg], self.wc.seg0[seg + 1]
                wr.extend(range(a, b))
                wv_.extend([sgn/(b - a)]*(b - a))

        nP, nN = len(self.port_p), len(self.port_n)
        for j, sgn in route:
            _wire_into(j, sgn)
        self.ihat_w = np.zeros(self.nwel)
        np.add.at(self.ihat_w, np.asarray(wr, dtype=np.intp),
                  np.asarray(wv_))
        # LATTICE side of the prescribed pattern by ONE sparse lsqr on
        # the incidence (equiterminal's construction): B^T ihat_f must
        # equal the port injection minus the wire foot flows. Any
        # feasible pattern is equally valid (the homogeneous DOFs
        # correct it); the per-port-cell TREE-PATH construction this
        # replaces walked ~4600 Python parent chains and was 47.9 s
        # of the 76.5 s R3 setup (fixed 2026-08-12).
        from scipy.sparse.linalg import lsqr
        want = np.zeros(self.nnode)
        for n in self.port_p:
            want[n] += 1.0/nP
        for n in self.port_n:
            want[n] -= 1.0/nN
        rhs = want - self.Bw @ (self.Afoot @ self.ihat_w)
        # Solve via the GRAPH LAPLACIAN L = B^T B (an M-matrix --
        # AMG's textbook case) and set ihat_f = B phi, which satisfies
        # B^T ihat_f = rhs by construction. Plain lsqr on the raw
        # incidence was condition-limited (~90-100 s at R3 regardless
        # of tolerance); Laplacian AMG converges in a handful of
        # cycles. Falls back to lsqr if pyamg is unavailable.
        #
        # GROUNDED PER COMPONENT (2026-08-13). L is singular: phi^T L
        # phi = ||B phi||^2, so its kernel is the vectors with B phi =
        # 0, i.e. ONE CONSTANT PER CONNECTED COMPONENT (8 on the DBC).
        # That is only the missing potential reference of nodal
        # analysis, and it is harmless in exact arithmetic -- but AMG's
        # Galerkin coarse operators turn those exact zero eigenvalues
        # into tiny values of EITHER SIGN, and pyamg's CG then aborts
        # on its definiteness guards: negative curvature p^T A p prints
        # "Indefinite matrix", negative z^T r prints "Indefinite
        # preconditioner". WHICH guard fires is decided by roundoff, so
        # the failure was NONDETERMINISTIC: measured at R4, one run
        # passed the residual gate in seconds and another fell into the
        # 20000-iteration lsqr fallback for ~2500 s (setup 234 s vs
        # 2755 s for the same problem).
        # Pinning phi = 0 at one node per component deletes exactly the
        # kernel and leaves a strictly positive definite system. The
        # ANSWER IS UNCHANGED: ihat_f = B phi and B annihilates
        # per-component constants, so grounding moves the gauge only.
        BT = self.B.T.tocsc().astype(np.float64)
        try:
            import pyamg
            import scipy.sparse.csgraph as _csg
            L = (BT @ self.B).tocsr()
            ncomp_L, lab = _csg.connected_components(L, directed=False)
            # one reference node per component: the first node of each
            # (isolated nodes touch no filament -- their L row is empty
            # and they are their own component, so they ground
            # themselves and contribute nothing to B phi)
            roots = np.zeros(ncomp_L, dtype=np.int64)
            roots[lab[::-1]] = np.arange(self.nnode - 1, -1, -1)
            free = np.ones(self.nnode, dtype=bool)
            free[roots] = False
            Lf = L[free][:, free].tocsr()
            ml = pyamg.smoothed_aggregation_solver(Lf, max_coarse=400)
            phi = np.zeros(self.nnode)
            phi[free] = ml.solve(rhs[free], tol=1e-12, maxiter=200,
                                 accel='cg')
            self.ihat_f = self.B @ phi
            # The grounded rows were NOT solved for -- their residual is
            # a genuine consistency check: summing a component's node
            # equations gives 0 = (net injection into that component),
            # so a nonzero here means the injections themselves violate
            # KCL (a mis-assigned wire foot, a broken chain), not that
            # the solve failed.
            resid_all = BT @ self.ihat_f - rhs
            gnd = np.abs(resid_all[~free]).max() if ncomp_L else 0.0
            if gnd > 1e-9:
                raise RuntimeError(
                    "component injections violate KCL (grounded-row "
                    "residual %g) -- the port/foot pattern is "
                    "inconsistent, not the solve" % gnd)
            resid = np.abs(resid_all).max()
            if resid > 1e-9:
                raise RuntimeError("Laplacian AMG ihat residual %g"
                                   % resid)
        except Exception:
            sol_ = lsqr(BT, rhs, atol=1e-11, btol=1e-11,
                        iter_lim=20000)
            self.ihat_f = sol_[0]
        # sharing cycles: chord wire forward + tree route back
        self._chords = []
        for j in chord_wires:
            c1, c2 = wcomp[j]
            fr, fv = [], []
            wr, wv_ = [], []
            self._bundle_into(fr, fv, j, 0, +1.0)
            self._bundle_into(fr, fv, j, 1, -1.0)
            _wire_into(j, +1.0)
            back = self._coarse_route(adj, c2, c1)
            node = self.foot_node[j][1]
            for (jj, sgn) in back:
                inn = self.foot_node[jj][0 if sgn > 0 else 1]
                self._path_into(fr, fv, node, inn)
                _wire_into(jj, sgn)
                ei, eo = (0, 1) if sgn > 0 else (1, 0)
                self._bundle_into(fr, fv, jj, ei, +1.0)
                self._bundle_into(fr, fv, jj, eo, -1.0)
                node = self.foot_node[jj][1 if sgn > 0 else 0]
            self._path_into(fr, fv, node, self.foot_node[j][0])
            f = sp.csc_matrix((fv, (fr, [0]*len(fr))),
                              shape=(self.efg, 1))
            wvcol = sp.csc_matrix((wv_, (wr, [0]*len(wr))),
                                  shape=(self.nwel, 1))
            self._chords.append((f, wvcol))

    def _coarse_route(self, adj, ca, cb):
        import collections
        prev = {ca: None}
        q = collections.deque([ca])
        while q:
            c = q.popleft()
            if c == cb:
                break
            for c2, j, sgn in adj[c]:
                if c2 not in prev:
                    prev[c2] = (c, j, sgn)
                    q.append(c2)
        out = []
        c = cb
        while prev[c] is not None:
            cp, j, sgn = prev[c]
            out.append((j, sgn))
            c = cp
        out.reverse()
        return out

    def _assert_kcl(self):
        """Every basis column divergence-free INCLUDING the wire foot
        flows; the prescribed pattern must match the port injection."""
        BT = self.B.T
        for k, (f, wv) in enumerate(self._chords):
            div = (BT @ f.toarray().ravel()
                   + self.Bw @ (self.Afoot @ wv.toarray().ravel()))
            if np.abs(div).max() > 1e-9:
                raise RuntimeError("sharing cycle %d violates KCL "
                                   "(max %g)" % (k, np.abs(div).max()))
        div = BT @ self.ihat_f + self.Bw @ (self.Afoot @ self.ihat_w)
        want = np.zeros(self.nnode)
        for n in self.port_p:
            want[n] += 1.0/len(self.port_p)
        for n in self.port_n:
            want[n] -= 1.0/len(self.port_n)
        if np.abs(div - want).max() > 1e-9:
            raise RuntimeError("prescribed pattern does not match the "
                               "port injection (max %g)"
                               % np.abs(div - want).max())
        # plaquettes/zero-sum columns are structurally safe; spot-check
        # the whole basis anyway
        X = self.Bmat[:self.efg, :]
        W = self.Bmat[self.efg:, :]
        div = np.abs(BT @ X + self.Bw @ (self.Afoot @ W)).max()
        if div > 1e-9:
            raise RuntimeError("stacked basis violates KCL (max %g)"
                               % div)

    # -- operator + solve (the A1 pattern with the foot term) -----------

    def _coupled(self, i_f, i_w):
        wcs, M = self.wc, self.M
        self.whole[:self.efg] = i_f
        wcs.i_f = np.ascontiguousarray(i_f, dtype=np.complex128)
        wcs.i_w = np.ascontiguousarray(i_w, dtype=np.complex128)
        wcs.out_w = np.zeros(self.nwel, dtype=np.complex128)
        M.traverseRL(extra=wcs)
        v_f = np.array(self.whole[:self.efg])
        v_w = (self.r_w*i_w
               + M.jomega*(wcs.wire_matvec(i_w) + wcs.out_w)
               + self.Afoot.T @ (self.Rfoot*(self.Afoot @ i_w)))
        return v_f, v_w

    def _matvec(self, x):
        self.matvecs += 1
        i = self.Bmat @ x
        v_f, v_w = self._coupled(i[:self.efg], i[self.efg:])
        return self.Bmat.T @ np.concatenate([v_f, v_w])

    def _precond(self, vec):
        return self.chol(np.real(vec)) + 1j*self.chol(np.imag(vec))

    def set_frequency(self, freq):
        """Retune the solver to a new frequency WITHOUT rebuilding
        (Tier 1, 2026-08-12): the only frequency-dependent physics is
        the wire skin shapes, and they enter the coupling blocks
        bilinearly as quadrature weights -- so the cached
        point-resolved kernels are REWEIGHTED (seconds) instead of
        requadratured, and the basis, Gram/AMG hierarchy, paths,
        feet, far tables and Wff all persist. Gated: a retuned solver
        matches a fresh build at the new frequency to ~1e-14 on the
        blocks and machine-level on Z."""
        self.model.prepare(self.M, freq)
        for w in self.wires:
            delta = (np.sqrt(1.0/(np.pi*freq*MU0*w.sigma))
                     if freq and freq > 0 else None)
            w.set_delta(delta)
        self.wc.reweight()
        self.r_w = np.concatenate([w.relements() for w in self.wires])

    def solve(self, freq, current=1.0, rtol=1e-4, maxiter=60,
              inner_m=None, verbose=False, method='lgmres',
              precision='auto'):
        """Returns ``(Z, info)`` -- port impedance V/I and diagnostics
        (per-wire chain currents as ``share`` of the drive).

        DEFAULT rtol IS THE ENGINEERING 1e-4 (decided with the user
        2026-08-12): the residual tail below it moves extracted Z by
        < 1e-3 % (the measured 81%-waste law; FEKO ships 3e-3).
        Oracle-grade comparisons pass rtol=1e-10 explicitly -- the
        validators do. ``method``: BiCGSTAB default / lgmres
        selectable, see :func:`port_impedance.krylov_solve`."""
        self.model.prepare(self.M, freq)
        t0 = time.perf_counter()
        v_f0, v_w0 = self._coupled(self.ihat_f*current,
                                   self.ihat_w*current)
        rhs = -(self.Bmat.T @ np.concatenate([v_f0, v_w0]))
        Aop = LinearOperator((self.size,)*2, matvec=self._matvec,
                             dtype=np.complex128)
        Pop = LinearOperator((self.size,)*2, matvec=self._precond,
                             dtype=np.complex128)
        n0 = self.matvecs
        from port_impedance import krylov_solve
        x, flag = krylov_solve(Aop, rhs, Pop, method=method, rtol=rtol,
                               maxiter=maxiter, inner_m=inner_m,
                               precision=precision)
        nrhs = np.linalg.norm(rhs)
        resid = (np.linalg.norm(rhs - Aop @ x)/nrhs if nrhs > 0 else 0.0)
        i = self.Bmat @ x
        i_f = self.ihat_f*current + i[:self.efg]
        i_w = self.ihat_w*current + i[self.efg:]
        v_f, v_w = self._coupled(i_f, i_w)
        V = complex(self.ihat_f @ v_f + self.ihat_w @ v_w)
        share = (self.Afoot @ i_w)/current
        info = dict(matvecs=self.matvecs - n0, flag=flag,
                    residual=resid, share=share,
                    time=time.perf_counter() - t0, i_f=i_f, i_w=i_w)
        if verbose:
            print("    %.4g Hz: %d matvecs, flag %s, resid %.2e, "
                  "%.2f s; shares %s"
                  % (freq, info['matvecs'], flag, resid, info['time'],
                     np.round(np.real(share), 4)))
        return complex(V)/current, info


def _forest_rooted(B, nn, order):
    """As :func:`_forest` but visiting roots in the given order --
    a different gauge for the path system (validator part I)."""
    Bc = B.tocoo()
    pos = Bc.data > 0
    lo = np.zeros(B.shape[0], dtype=np.int64)
    hi = np.zeros(B.shape[0], dtype=np.int64)
    lo[Bc.row[pos]] = Bc.col[pos]
    hi[Bc.row[~pos]] = Bc.col[~pos]
    import collections
    adj = collections.defaultdict(list)
    for e in range(B.shape[0]):
        adj[lo[e]].append((hi[e], e, +1.0))
        adj[hi[e]].append((lo[e], e, -1.0))
    parent = np.full(nn, -1, dtype=np.int64)
    pedge = np.full(nn, -1, dtype=np.int64)
    psign = np.zeros(nn)
    comp = np.full(nn, -1, dtype=np.int64)
    ncomp = 0
    for root in order:
        if comp[root] >= 0 or not adj[root]:
            continue
        comp[root] = ncomp
        stack = [root]
        while stack:
            u = stack.pop()
            for v, e, sgn in adj[u]:
                if comp[v] < 0:
                    comp[v] = ncomp
                    parent[v] = u
                    pedge[v] = e
                    psign[v] = sgn
                    stack.append(v)
        ncomp += 1
    return parent, pedge, psign, comp
