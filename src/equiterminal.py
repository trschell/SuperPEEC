# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Equipotential port terminals, with the terminal currents as UNKNOWNS.

Replaces the prescribed-current port model of ``port_impedance`` (which
hard-codes an equal current share over every face of a terminal) with
the boundary condition VoxHenry and FastHenry impose: all faces of a
terminal share one node, and the current distributes itself.

WHY (measured, not assumed). On setup3's wide port the solved return
split runs 0.1388 down to 0.0009 over 15 faces where equal share would
be 0.0667 -- a 159:1 crowding. Against VoxHenry the prescribed model
sits 11.2% high in L at dx=1e-5 and 9.6% at 5e-6, i.e. essentially
REFINEMENT-INDEPENDENT, the signature of a model defect; with this
boundary condition the same geometry gives 2.65% and 0.82%, i.e. it
CONVERGES. ``dense_oracle.py`` is the reference that established this
and is the acceptance test for this module.

WHAT CHANGES STRUCTURALLY
-------------------------
The terminal half-filaments stop being a post-hoc series correction and
become branches of the circuit, which is what lets the current split
solve itself. Three consequences:

  * The port injection is a single ``+I`` at the P node and ``-I`` at the
    N node. The whole corner/uniform lumping machinery
    (``port_nodes(weight=...)``) is not needed -- and under the 'cell'
    scheme those two weightings were provably identical anyway.
  * The port voltage is simply ``phi(P) - phi(N)``; no work-conjugate
    weighting, no ``terminal_impedance`` series term, no
    ``terminal_interior_coupling`` right-hand side.
  * The mesh basis must span the new cycles the shared port node
    creates: ``n_faces - 1`` per terminal. They are built here and
    APPENDED to the production plaquette basis rather than replacing it,
    so the existing basis keeps its 4-nonzero columns and its
    conditioning; the appended cycles are 3-nonzero whenever consecutive
    port faces are neighbours, which is the usual case.

CONVENTION (self-contained, and checked at build time)
------------------------------------------------------
Every filament carries current positive along ``+axis``; ``B[f, tail] =
+1``, ``B[f, head] = -1``. Then the branch equation is ``Z i = B phi``
and Kirchhoff is ``B^T i = s``, with ``phi`` the PHYSICAL potential --
no sign flip to undo at the end. The production incidence is recovered
from ``connectA`` by the two-probe parity trick and divided by ``beta``,
and :func:`_check_orientation` ASSERTS that its ``+1`` sits on the low
cell of each filament, so the terminal rows built here share one
convention with the interior. Getting that wrong is worth exactly twice
the terminal coupling and is invisible to reciprocity, so it is
asserted rather than assumed.

STATUS / COST
-------------
The terminal<->interior mutual is held as an EXPLICIT DENSE block
``C`` of shape ``(n_terminals, n_parallel_filaments)``. That is correct
and makes the operator and its transpose the same object -- no sign can
disagree between them -- but it costs ``O(n_t * N)`` memory and one
gemv per matvec. Fine for ports of tens of faces (setup3: 16 x 4996);
for large ports this block is what the FMM should supply instead --
near list direct, far field through P2M with a halved moment and a
dx/4 centroid shift. That optimisation is deliberately NOT done here:
correctness first, against the oracle.

Run inside the toolbox.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsqr, lgmres
import sksparse.cholmod as cholmod

import meshgraph as mg
import terminal as tm
import stencils as st

_AXIS_ORIENT = {0: 'f', 1: 'e', 2: 'g'}


# -- geometry / topology decode ----------------------------------------

def node_cells(M):
    """Cell coordinate of every node, in compressed node order.

    Mirrors the decode in ``systemmat._mnaprecassembly``: at a single
    level the compressed ``idx`` are global lattice indices; multilevel
    stores per-box indices and needs the box offset added.
    """
    lv0 = M.lv[0]
    if M.numlevels == 1:
        dims = M.ntotal.astype(int)
        cx = (lv0.idx // (dims[1]*dims[2])).astype(np.int64)
        cy = ((lv0.idx // dims[2]) % dims[1]).astype(np.int64)
        cz = (lv0.idx % dims[2]).astype(np.int64)
    else:
        n0 = lv0.n.astype(int)
        cx = (lv0.idx // (n0[1]*n0[2])).astype(np.int64)
        cy = ((lv0.idx // n0[2]) % n0[1]).astype(np.int64)
        cz = (lv0.idx % n0[2]).astype(np.int64)
        for g in range(np.size(lv0.idx0) - 1):
            sl = np.s_[lv0.idx0[g]:lv0.idx0[g+1]]
            cx[sl] += lv0.xidx[g]*n0[0]
            cy[sl] += lv0.yidx[g]*n0[1]
            cz[sl] += lv0.zidx[g]*n0[2]
    return np.stack([cx, cy, cz], axis=1)


def sparse_incidence(M, whole, efgsize, nodesize):
    """Explicit sparse incidence ``B`` (efg x nodes), entries +-1.

    Two ``connectA`` probes with the node keys split by coordinate
    parity decode both endpoints and both signs at once -- the same
    trick ``systemmat._mnaprecassembly`` uses, reproduced here because
    that one is a ``SystemMat`` method and this path has no SystemMat.
    The result is divided by ``beta`` so the convention is the plain
    +-1 incidence the docstring above describes.
    """
    lv0 = M.lv[0]
    cells = node_cells(M)
    par = cells.sum(axis=1) % 2
    keys = np.arange(1, nodesize + 1, dtype=np.float64)
    resp = []
    for p in (0, 1):
        lv0.data[:] = 0.0
        lv0.data[par == p] = keys[par == p]
        ae, af, ag = M.connectA()
        resp.append(np.real(np.concatenate([ae, af, ag])))
    whole[:] = 0.0
    beta = float(np.real(lv0.beta))
    r_ev, r_od = resp
    n_ev = np.rint(np.abs(r_ev)/beta).astype(np.int64) - 1
    n_od = np.rint(np.abs(r_od)/beta).astype(np.int64) - 1
    bad = (np.abs(np.abs(r_ev)/beta - (n_ev + 1)) > 1e-6).any() or \
          (np.abs(np.abs(r_od)/beta - (n_od + 1)) > 1e-6).any() or \
          (n_ev < 0).any() or (n_ev >= nodesize).any() or \
          (n_od < 0).any() or (n_od >= nodesize).any() or \
          (np.sign(r_ev)*np.sign(r_od) >= 0).any()
    if bad:
        raise RuntimeError(
            "connectA produced an invalid incidence -- under the EDGE "
            "scheme a single-level tree needs N >= NT+1")
    rows = np.repeat(np.arange(efgsize), 2)
    cols = np.column_stack([n_ev, n_od]).ravel()
    vals = np.column_stack([np.sign(r_ev), np.sign(r_od)]).ravel()
    B = sp.coo_matrix((vals, (rows, cols)),
                      shape=(efgsize, nodesize)).tocsr()
    per_row = np.diff(B.indptr)
    if (per_row != 2).any() or np.abs(B.sum(axis=1)).max() > 1e-9:
        raise RuntimeError("incidence rows are not +1/-1 pairs")
    return B, cells


def _leaf_order(M):
    """Leaf blocks in the buffer's e|f|g order, with their axis."""
    return [(M.e, 1, 0), (M.f, 0, np.size(M.e.struc)),
            (M.g, 2, np.size(M.e.struc) + np.size(M.f.struc))]


def filament_cells(M):
    """Per filament: its axis and the coordinate of its LOW cell.

    Same decode as ``port_impedance._interior_slots`` (a cell-centred
    filament at lattice index i joins cells i and i+1), returned for
    all three orientations in buffer order. Returned int32: per-axis
    cell coordinates stay far below 2**31 at any reachable scale, and
    the arrays are a linear-in-N residency term (they were 2x81 MB of
    int64 at R3 before the dedup + downcast).
    """
    axis = []
    cells = []
    for leaf, d, _ in _leaf_order(M):
        n = np.asarray(leaf.n, dtype=np.int64)
        idx = np.asarray(leaf.idx, dtype=np.int64)
        c = [idx//(n[1]*n[2]), (idx//n[2]) % n[1], idx % n[2]]
        if M.numlevels > 1:
            bx = np.asarray(leaf.xidx, dtype=np.int64)
            by = np.asarray(leaf.yidx, dtype=np.int64)
            bz = np.asarray(leaf.zidx, dtype=np.int64)
            i0 = np.asarray(leaf.idx0, dtype=np.int64)
            off = [np.zeros(idx.size, dtype=np.int64) for _ in range(3)]
            for g in range(i0.size - 1):
                sl = np.s_[i0[g]:i0[g+1]]
                off[0][sl] = bx[g]*n[0]
                off[1][sl] = by[g]*n[1]
                off[2][sl] = bz[g]*n[2]
            c = [c[k] + off[k] for k in range(3)]
        axis.append(np.full(idx.size, d, dtype=np.int32))
        cells.append(np.stack(c, axis=1))
    return np.concatenate(axis), np.concatenate(cells).astype(np.int32)


def _check_orientation(B, fil_axis, fil_cell, node_of_cell):
    """Assert ``B``'s +1 sits on each filament's LOW cell.

    If connectA ever flipped this the terminal rows built here would
    oppose the interior ones, which shifts the answer by twice the
    terminal coupling and leaves reciprocity intact -- i.e. it would not
    show up in the obvious guard. So it is measured, not assumed.
    """
    lo = np.array([node_of_cell[tuple(c)] for c in fil_cell])
    got = np.asarray(B[np.arange(B.shape[0]), lo]).ravel()
    if not np.all(got == 1.0):
        n_bad = int((got != 1.0).sum())
        if np.all(got == -1.0):
            raise RuntimeError(
                "connectA orients filaments with -1 on the low cell; "
                "flip the terminal rows to match before using this path")
        raise RuntimeError("incidence orientation is not uniform "
                           "(%d of %d filaments disagree)"
                           % (n_bad, got.size))


# -- the terminal block -------------------------------------------------

class Terminals:
    """A port's terminal half-filaments, as branches.

    Parameters
    ----------
    model : vhr.VhrModel
    M : multipole.Tree
    port : int or str
    """

    def __init__(self, model, M, port=0, t_l=None):
        p = model.port(port)
        if len(p.pos) == 0 or len(p.neg) == 0:
            raise ValueError("port %r needs both P and N faces" % port)
        axes = {int(e[3]) for e in p.pos} | {int(e[3]) for e in p.neg}
        if len(axes) != 1:
            raise ValueError("port faces must share one axis, got %s" % axes)
        self.axis = axes.pop()
        self.orient = _AXIS_ORIENT[self.axis]
        # Per-axis pitch: the terminal's AXIAL step is l[axis] (t_l
        # semantics unchanged), the kernels take the full l vector, and
        # cross-sections are transverse products. self.dx keeps the
        # axial pitch under its historical name for the t_l algebra.
        self.l = tuple(float(v) for v in model.d)
        self.dx = self.l[self.axis]
        self._model = model
        self.superconductor = bool(getattr(model, 'superconductor', False))
        # The terminal resistance needs the sigma of the metal THIS
        # PORT's faces sit in -- port_sigma, not uniform_sigma, which
        # has no answer on a mixed-material model (Cu_Al: one port on
        # copper, one on aluminium; port_sigma raises only if a single
        # port straddles two materials, where R_t is truly ambiguous).
        # On a uniform model the two return the identical float, so
        # nothing changes there. Superconductors have no meaningful
        # sigma at all (it may be 0 everywhere); their R is z(w)-based
        # and installed per frequency by set_frequency instead.
        self.sigma = None if self.superconductor else model.port_sigma(port)
        others = [c for c in range(3) if c != self.axis]
        self.faces = []                     # (cell, sign, polarity)
        for arr, pol in ((p.pos, +1), (p.neg, -1)):
            for e in arr:
                self.faces.append(((int(e[0]), int(e[1]), int(e[2])),
                                   int(e[4]), pol))
        self.n = len(self.faces)
        # ARBITRARY LENGTH. A terminal runs from the port reference plane
        # to the node it feeds, so it ENDS at the cell centre: a low face
        # spans [centre - t_l, centre] and a high face [centre, centre +
        # t_l]. t_l = dx/2 puts the plane on the conductor face (the only
        # length the half-step kernel could reach); smaller moves it
        # inside the end cell, larger outside -- i.e. t_l IS the port
        # reference-plane position, which is what makes de-embedding a
        # parameter rather than a remesh.
        self.t_l = 0.5*self.dx if t_l is None else float(t_l)
        if not 0.0 < self.t_l < self.dx:
            raise ValueError("terminal length must satisfy 0 < t_l < dx "
                             "(got %g with dx %g)" % (self.t_l, self.dx))
        self.tau = self.t_l/self.dx
        self.cax = np.array([c[self.axis] for c, _, _ in self.faces],
                            dtype=np.int64)
        self.hi = np.array([s > 0 for _, s, _ in self.faces])
        self.t0 = np.array([c[others[0]] for c, _, _ in self.faces])
        self.t1 = np.array([c[others[1]] for c, _, _ in self.faces])
        self.pol = np.array([pl for _, _, pl in self.faces])
        self.sign = np.array([s for _, s, _ in self.faces])
        l = self.l
        if self.superconductor:
            # placeholder until the first set_frequency; solve() installs
            # the real thing before any apply
            self.R = np.zeros(self.n, dtype=np.complex128)
        else:
            self.R = np.full(self.n,
                             tm.terminal_resistance(l, self.orient,
                                                    self.sigma, self.t_l))
        self._build_kernel(M)

    def set_frequency(self, freq):
        """Install the terminal series impedances for ``freq``.

        No-op for normal conductors, whose R is real and frequency
        independent. For superconductors the terminal filament -- full
        cross-section, length ``t_l``, living entirely in its end cell
        -- has ``R = z(w)*t_l/A`` with that CELL's impedance density,
        complex (the kinetic inductance rides in the imaginary part)
        and frequency dependent, so the solver calls this before every
        solve. Per-cell lookup rather than a global value, so a mixed
        normal/superconducting model keeps each port face honest.
        """
        if not self.superconductor:
            return
        z = self._model.impedance_density(freq)
        cells = np.array([c for c, _, _ in self.faces], dtype=np.int64)
        zc = z[cells[:, 0], cells[:, 1], cells[:, 2]]
        ot = [c for c in range(3) if c != self.axis]
        self.R = zc*self.t_l/(self.l[ot[0]]*self.l[ot[1]])

    def _build_kernel(self, M):
        """Unequal-length kernels, one per face-type combination.

        Axial START offsets, from the spans above with the filament
        joining cells j, j+1 spanning [j+1/2, j+3/2]:
          terminal->filament   low: s0 = t_l   high: s0 = 0
          terminal->terminal   both same type: 0; low->high: +t_l;
                               high->low: -t_l
        """
        dims = [int(v) for v in np.asarray(M.ntotal, dtype=int)]
        others = [c for c in range(3) if c != self.axis]
        n = [0, 0, 0]
        n[self.axis] = dims[self.axis] + 2
        n[others[0]] = dims[others[0]] + 1
        n[others[1]] = dims[others[1]] + 1
        self.nax = n[self.axis]
        l = self.l
        p, dx = self.t_l, self.dx
        self.kf = {False: tm.unequal_kernel(l, self.orient, n, p, dx, p),
                   True: tm.unequal_kernel(l, self.orient, n, p, dx, 0.0)}
        s0 = {0.0: tm.unequal_kernel(l, self.orient, n, p, p, 0.0),
              +1: tm.unequal_kernel(l, self.orient, n, p, p, p),
              -1: tm.unequal_kernel(l, self.orient, n, p, p, -p)}
        self.kt = {(False, False): s0[0.0], (True, True): s0[0.0],
                   (False, True): s0[+1], (True, False): s0[-1]}

    def self_block(self):
        """Dense ``Lp`` between terminal filaments (n x n, henries)."""
        L = np.zeros((self.n, self.n))
        for a in range(self.n):
            for b in range(self.n):
                L[a, b] = tm.mutual_offset(
                    self.kt[(bool(self.hi[a]), bool(self.hi[b]))],
                    self.orient, int(self.cax[b] - self.cax[a]), self.nax,
                    transverse=(int(self.t0[a] - self.t0[b]),
                                int(self.t1[a] - self.t1[b])))
        return L

    def parallel(self, fil_axis):
        """Buffer indices of the filaments PARALLEL to the port axis.

        Perpendicular filaments have zero mutual partial inductance,
        which removes two thirds of the work before it starts.
        """
        return np.flatnonzero(fil_axis == self.axis)

    def _rows(self, k, jax, u0, u1):
        """``Lp`` from terminal ``k`` to the filaments described.

        ``jax`` is each filament's LOW cell along the port axis; the
        kernel index is the SIGNED offset ``j - c``, since with unequal
        bar lengths the coupling is no longer symmetric in it.
        """
        others = [c for c in range(3) if c != self.axis]
        kern = self.kf[bool(self.hi[k])]
        idx = [None, None, None]
        idx[self.axis] = (jax - self.cax[k]) + self.nax
        idx[others[0]] = np.abs(self.t0[k] - u0)
        idx[others[1]] = np.abs(self.t1[k] - u1)
        return kern[idx[0], idx[1], idx[2]]

    def coupling(self, fil_axis, fil_cell, restrict=None):
        """``Lp`` from each terminal to the parallel filaments.

        ``restrict`` is an optional list of per-terminal index arrays
        (into the parallel-filament subset). Given it, ONLY those pairs
        are evaluated and the result comes back as a sparse matrix --
        which is the point: the dense block is O(n_t * N) and must never
        be formed when only the near pairs are wanted.
        """
        sel = self.parallel(fil_axis)
        others = [c for c in range(3) if c != self.axis]
        jax = fil_cell[sel, self.axis]           # the filament's LOW cell
        u0 = fil_cell[sel, others[0]]
        u1 = fil_cell[sel, others[1]]
        if restrict is None:
            C = np.zeros((self.n, sel.size))
            for k in range(self.n):
                C[k] = self._rows(k, jax, u0, u1)
            return C, sel
        rows, cols, vals = [], [], []
        for k, jj in enumerate(restrict):
            if jj.size == 0:
                continue
            rows.append(np.full(jj.size, k))
            cols.append(jj)
            vals.append(self._rows(k, jax[jj], u0[jj], u1[jj]))
        if not rows:
            return sp.csr_matrix((self.n, sel.size)), sel
        return sp.csr_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.n, sel.size)), sel


def skin_depth(sigma, freq, mu=4e-7*np.pi):
    """Classical skin depth ``sqrt(2/(w mu sigma))`` (metres)."""
    if freq <= 0:
        return np.inf
    return float(np.sqrt(2.0/(2*np.pi*freq*mu*sigma)))


def recommend_subdivision(dx, sigma, freq, cap=3):
    """Sub-filaments per transverse direction, from cell size vs skin depth.

    Returns 1 (no subdivision) when the mesh already resolves the skin
    depth, so switching this on costs nothing where it buys nothing.

    THE RULE AND ITS HONEST LIMITS. The classical guideline -- Coperich,
    Ruehli & Cangellaris, T-MTT 48(9) 2000, for their GSI cross-section
    mesh -- is ``h < delta/2``, i.e. ``k >= 2*dx/delta``. Measured here
    at ``dx/delta = 4.8`` on a uniform bar, R relative to no
    subdivision:

        k = 2x2  +93.9%     2x-refined reference  +98.5%
        k = 3x3  +123.6%    3x-refined reference  +130.4%

    so each k tracks the SAME-LEVEL refinement to ~95%, but the 3x
    reference is itself still climbing -- k=2 is a large cheap
    improvement, NOT convergence. The default cap keeps the cost sane
    (modes grow as k**2 per filament); raise it if you need the
    remaining accuracy and can pay for it.

    Parameters
    ----------
    dx : float
        Cell pitch (metres).
    sigma : float
    freq : float
        Reference frequency -- use the HIGHEST of interest, since the
        skin depth is smallest there. Over-subdividing at lower
        frequency is harmless: the extra modes simply carry no current
        (exactly zero at DC, measured).
    cap : int, optional

    Returns
    -------
    int
    """
    d = skin_depth(sigma, freq)
    if not np.isfinite(d) or d <= 0:
        return 1
    k = int(min(max(1, np.ceil(2.0*dx/d)), cap))
    # NEVER RETURN 2. A 2x2 split is BLIND to an axially symmetric
    # neighbourhood: all four quadrants are equivalent, so their
    # couplings to a collinear neighbour are identical and the net-zero
    # mode weights cancel them EXACTLY. Measured on setup3, whose vias
    # are collinear stacks: |Zcross|max = 1.4e-27 at k=2 (machine zero,
    # so Z is unchanged to 10 digits) against 2.2e-13 at k=3. Only an
    # ODD split has a distinguishable centre/edge/corner, which is what
    # the radially symmetric part of skin effect needs; k=2 can express
    # "more current on one side" (proximity effect) but not "more
    # current at the edges than the centre".
    return 3 if k == 2 else k


def conduction_weights(kk, dx, delta):
    """Net-zero CONDUCTION-MODE weights on the ``kk[0] x kk[1]`` sub-bar grid.

    Daniel, Sangiovanni-Vincentelli & White (EPEP 2000): inside a good
    conductor the current solves a Helmholtz equation whose solutions
    are exponentials anchored to the cross-section's faces and corners
    -- face decay rate ``p = (1+j)/delta``, corner rate ``(1+j)/
    (delta*sqrt(2))``. Those are the shapes the piecewise-constant basis
    approximates badly (measured in ``studies/modebasis2d.py``: at
    matched mode count the conduction basis misses 6.6x less of the
    skin-effect correction than 'diff', the recorded 201%% failure's
    root cause).

    The modes are anchored to the CELL faces, not the conductor surface:
    every filament must carry IDENTICAL weights or the Toeplitz/FFT
    structure of the mode tables is lost. That costs nothing in span --
    a face exponential restricted to any interior cell is a combination
    of that cell's own face exponentials, and the per-cell mean it also
    needs is exactly the aggregate current.

    Four complex shapes, following the 2-D testbed: the symmetric face
    sum (pure skin crowding), the two antisymmetric face differences
    (proximity), and the corner sum. Each complex shape enters as TWO
    REAL columns -- real and imaginary part -- because ``build_fft``
    requires a real ``W`` (one real spectrum serves both convolution
    directions) and the real pair spans strictly more than one complex
    column. Columns are mean-subtracted (net-zero, the load-bearing
    invariant), normalised, and pruned: a column that vanishes (the
    imaginary parts at delta >> dx) or is linearly dependent on the
    others (the corner mode degenerates into the face modes as
    dx/delta -> 0) would make the mode resistance block ``W^T W``
    singular, and the mode equations with it.
    """
    from scipy.linalg import qr
    k0, k1 = kk
    u = (np.arange(k0) + 0.5)/k0
    v = (np.arange(k1) + 0.5)/k1
    U, V = np.meshgrid(u, v, indexing='ij')       # matches _sub_boxes
    x = U.ravel()*dx                              # distance from the low
    y = V.ravel()*dx                              # face, per split axis
    p = (1.0 + 1.0j)/delta
    ex0, ex1 = np.exp(-p*x), np.exp(-p*(dx - x))
    ey0, ey1 = np.exp(-p*y), np.exp(-p*(dx - y))
    pc = (1.0 + 1.0j)/(delta*np.sqrt(2.0))
    corner = (np.exp(-pc*(x + y)) + np.exp(-pc*(x + (dx - y)))
              + np.exp(-pc*((dx - x) + y))
              + np.exp(-pc*((dx - x) + (dx - y))))
    shapes = [ex0 + ex1 + ey0 + ey1,              # symmetric: skin
              ex0 - ex1,                          # antisymmetric: proximity
              ey0 - ey1,
              corner]
    cols = []
    for c in shapes:
        for part in (c.real, c.imag):
            w = part - part.mean()
            n = np.linalg.norm(w)
            if n > 1e-12*max(np.linalg.norm(part), 1.0):
                cols.append(w/n)
    if not cols:
        raise ValueError("conduction modes are all constant at dx/delta = "
                         "%.3e -- no skin effect to model; use "
                         "subdivide=False instead" % (dx/delta,))
    W = np.stack(cols, axis=1)
    # Prune near-dependent columns by pivoted QR, KEEPING original
    # (physical) columns rather than mixing them, so the result is
    # deterministic and readable. Dropped columns are within ~1e-7 of
    # the span of the kept ones, so the span is preserved to that level.
    _, R, piv = qr(W, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    W = W[:, np.sort(piv[d > 1e-7*d[0]])]
    return W - W.mean(axis=0, keepdims=True)      # exact net-zero


class Redistribution:
    """Cross-section redistribution modes: skin effect without a finer mesh.

    Each filament of one orientation is replaced by ``k`` parallel
    sub-filaments spanning THE SAME TWO NODES, then the basis is changed
    to (aggregate, redistribution):

        i_p = I/k + sum_m w_mp u_m ,      sum_p w_mp = 0

    The aggregate ``I`` is exactly the original filament -- the identity
    ``W_agg^T Z_sub W_agg = Z_full`` holds exactly, because the volume
    integral over the whole cross-section is the sum over the pieces --
    so the existing Toeplitz FFT near field and FMM far field are
    UNTOUCHED. The ``k-1`` redistribution modes carry ZERO net current,
    hence zero incidence: no new nodes, invisible to KCL, and invisible
    to the nodal Schur complement.

    Physically each mode is the mesh current of the loop formed by two
    parallel branches, and its equation is KVL around that loop. At DC
    the branches are identical so the current splits evenly and the
    modes vanish; at AC the halves couple differently to the rest of the
    structure and they do not. That asymmetry IS skin and proximity
    effect, obtained from extra circuit equations instead of extra mesh.

    Because the modes are net-zero their fields are dipolar: mode-to-
    aggregate falls off as 1/r**2 and mode-to-mode as 1/r**3, against
    1/r for the ordinary near field. The blocks below are therefore
    SHORT RANGED and should eventually be truncated; they are kept dense
    here for correctness first, exactly as the terminal coupling was.
    """

    def __init__(self, model, M, axis, fil_axis, fil_cell, split_axis=None,
                 k=2, term=None, rc_uu=3, rc_cross=4,
                 use_fft=True, csr_max_gb=2.0, mode_basis='diff',
                 skin_freq=None, boundary_only=False):
        self.axis = int(axis)
        others = [c for c in range(3) if c != self.axis]
        # k may be an int (split BOTH transverse axes k x k) or a pair.
        # A one-dimensional split only lets current crowd to two faces of
        # four and saturates well short of the truth -- measured +86% of
        # the +130% a refined model gives at dx/delta = 4.8 -- so the
        # default subdivides the whole cross-section.
        kk = (int(k), int(k)) if np.isscalar(k) else tuple(int(v) for v in k)
        if min(kk) < 1 or max(kk) < 2:
            raise ValueError("k must give at least two sub-filaments")
        self.split = others
        self.kk = kk
        self.k = kk[0]*kk[1]
        self.dx = model.dx
        self.sigma = model.uniform_sigma()
        self.sel = np.flatnonzero(fil_axis == self.axis)
        self.mode_basis = str(mode_basis)
        self.boundary_only = bool(boundary_only)
        # Which filaments CARRY modes. NOTE this must NOT shrink `sel`:
        # `sel` also selects which aggregate currents the modes COUPLE
        # to (Zcross columns, and `i_f[sel]` in the matvec). Restricting
        # it would silently blind a boundary cell's mode to its interior
        # neighbours' currents -- a real drive error, not a saving. So
        # keep every parallel filament in the coupling set and mask only
        # the mode DOFs.
        self._bnd = np.ones(self.sel.size, dtype=bool)
        if self.boundary_only:
            occ = np.asarray(model.struc()).astype(bool)
            cc = np.asarray(fil_cell)[self.sel]
            s0, s1 = self.split
            keep = np.zeros(len(cc), dtype=bool)
            for t, d in ((s0, +1), (s0, -1), (s1, +1), (s1, -1)):
                nb = cc.copy()
                nb[:, t] += d
                inside = np.ones(len(cc), dtype=bool)
                for c in range(3):
                    inside &= (nb[:, c] >= 0) & (nb[:, c] < occ.shape[c])
                exposed = ~inside
                idx = np.flatnonzero(inside)
                if idx.size:
                    exposed[idx] = ~occ[nb[idx, 0], nb[idx, 1], nb[idx, 2]]
                keep |= exposed
            self._bnd = keep
        self.nfil = self.sel.size
        self.cells = fil_cell[self.sel]
        self.lo, self.hi = self._sub_boxes(self.cells)
        self.flo, self.fhi = self._full_boxes(self.cells)
        self.asub = (self.dx/kk[0])*(self.dx/kk[1])
        self.afull = self.dx*self.dx
        # Per-filament weights, IDENTICAL for every filament. Columns
        # span (part of) the NET-ZERO space -- that is the load-bearing
        # property: zero net current => zero incidence => invisible to
        # KCL and to the nodal Schur complement, and the aggregate
        # identity W_agg^T Z_sub W_agg == Z_full still holds exactly.
        #
        # 'diff'       : k-1 differences of consecutive sub-bars (original).
        # 'linear'     : 2 columns, the sub-bar centroid offsets in the two
        #                transverse directions -- a linear tilt of the
        #                current density, i.e. VoxHenry/PWL-style
        #                p-refinement discretised on the existing sub-bar
        #                grid. Mode count becomes 2 REGARDLESS of k, so k
        #                turns into pure quadrature refinement of the ramp.
        # 'conduction' : up to 8 columns, the real and imaginary parts of
        #                the face/corner exponentials the interior
        #                Helmholtz equation actually solves (see
        #                conduction_weights). Like 'linear', k is pure
        #                quadrature: it resolves the exponentials without
        #                adding unknowns, so use k ~ 6-8 with this basis.
        #                Needs ``skin_freq`` -- the shapes depend on the
        #                skin depth. ``set_frequency`` retunes W per solve
        #                point, so skin_freq only seeds the initial state.
        self.skin_freq = None if skin_freq is None else float(skin_freq)
        if self.mode_basis == 'conduction':
            if self.skin_freq is None or self.skin_freq <= 0:
                raise ValueError("mode_basis='conduction' needs skin_freq "
                                 "> 0: the mode shapes are exponentials in "
                                 "the skin depth (got %r)" % (skin_freq,))
            delta = skin_depth(self.sigma, self.skin_freq)
            W = conduction_weights(kk, self.dx, delta)
        elif self.mode_basis == 'linear':
            k0, k1 = kk
            u = (np.arange(k0) + 0.5)/k0 - 0.5
            v = (np.arange(k1) + 0.5)/k1 - 0.5
            U, V = np.meshgrid(u, v, indexing='ij')   # matches _sub_boxes
            W = np.stack([U.ravel(), V.ravel()], axis=1)
            W -= W.mean(axis=0, keepdims=True)        # enforce net-zero
        elif self.mode_basis == 'diff':
            W = np.zeros((self.k, self.k - 1))
            for m in range(self.k - 1):
                W[m, m] = +1.0
                W[m + 1, m] = -1.0
        else:
            raise ValueError("mode_basis must be 'diff', 'linear' or "
                             "'conduction', got %r" % (mode_basis,))
        self.csr_max_gb = float(csr_max_gb)
        self._check_aggregate()
        self.use_fft = bool(use_fft)
        self._term = term
        self._rc_uu, self._rc_cross = int(rc_uu), int(rc_cross)
        # Frequency-independent geometry caches, filled on first build:
        # sub-bar coupling tables keyed by the separation set, and the
        # KDTree pair lists. They are what make set_frequency cheap --
        # the kernel evaluations dominate assembly and never change.
        self._geom = {}
        self._geom_t = {}
        self._pairs = {}
        self._set_W(W)
        self._assemble()

    def _set_W(self, W):
        """Install mode weights and everything whose SHAPE follows km."""
        if np.abs(W.sum(axis=0)).max() > 1e-12:
            raise ValueError("mode weights must be net-zero")
        self.W = W
        self.km = W.shape[1]
        self.nmode_full = self.nfil*self.km
        self.mode_mask = np.repeat(self._bnd, self.km)
        self.nmode = int(self.mode_mask.sum())

    def _assemble(self):
        """Fold the current ``W`` into every block. Geometry is cached,
        so a re-assembly (set_frequency) pays only einsum contractions,
        CSR index arithmetic and -- the real cost -- the FFT spectra.

        Build ONLY the representation that will be applied. The sparse
        blocks are not merely redundant under FFT, they are ruinous:
        pairs grow as (2rc+1)**3, so at the default (3,4) radii a
        23600-filament conductor needs ~5e8 nonzeros (~4 GB) and OOMs
        -- while the FFT it would have fed costs only padding.
        """
        self.Zuu = self.Zcross = None
        self.nnz = (0, 0)
        if not self.use_fft:
            self._build_truncated(self._rc_uu, self._rc_cross)
        r_sub = self.k/(self.sigma*self.dx)          # k x a full filament's
        # resistance couples only sub-bars of the SAME filament, so Ru is
        # block diagonal with one identical block
        self.Ru = sp.kron(sp.identity(self.nfil, format='csr'),
                          r_sub*(self.W.T @ self.W), format='csr')
        self.Zt = None
        if self._term is not None and self._term.axis == self.axis:
            self._build_terminal(self._term, self._rc_cross)
        if self.use_fft:
            self.build_fft(self._rc_uu, self._rc_cross)
        if self.nmode != self.nmode_full:
            # Drop the interior modes AFTER assembly, so Zcross keeps all
            # its aggregate columns (the drive) while only mode ROWS go.
            mk = self.mode_mask
            self.Ru = self.Ru[mk][:, mk]
            if self.Zuu is not None:
                self.Zuu = self.Zuu[mk][:, mk]
            if self.Zcross is not None:
                self.Zcross = self.Zcross[mk]
            if self.Zt is not None:
                self.Zt = self.Zt[mk]

    def set_frequency(self, freq):
        """Retune the conduction modes to ``freq``; True if blocks changed.

        The mode SHAPES depend on the skin depth, but the expensive part
        of every block -- the geometric sub-bar coupling tables and the
        neighbour pair lists -- is frequency independent and cached, so
        this re-folds the new ``W`` and re-runs assembly on top of the
        cache. The caller owns the consequences of a changed ``km``
        (pruning is delta-dependent): when the return is True, check
        ``nmode`` and rebuild whatever sized itself on it.

        No-op for 'diff'/'linear' (frequency-independent weights), at
        the frequency already installed, and at freq <= 0 -- at DC the
        ``jw`` couplings vanish and the mode equations force ``u = 0``
        whatever shapes are loaded, so retuning would only lose the
        (well-defined) high-frequency W for an undefined one.
        """
        if self.mode_basis != 'conduction':
            return False
        freq = float(freq)
        if freq <= 0 or freq == self.skin_freq:
            return False
        self.skin_freq = freq
        delta = skin_depth(self.sigma, freq)
        self._set_W(conduction_weights(self.kk, self.dx, delta))
        self._assemble()
        return True

    def _sub_boxes(self, cells):
        """The parallel sub-bars of each filament: the cross-section cut
        ``kk[0] x kk[1]`` over the two transverse axes, full length."""
        dx, d = self.dx, self.axis
        s0, s1 = self.split
        k0, k1 = self.kk
        h0, h1 = dx/k0, dx/k1
        cells = np.atleast_2d(np.asarray(cells))
        lo = np.zeros((len(cells)*self.k, 3))
        hi = np.zeros((len(cells)*self.k, 3))
        for f, c in enumerate(cells):
            for p in range(k0):
                for q in range(k1):
                    a = [c[0]*dx, c[1]*dx, c[2]*dx]
                    b = [(c[0]+1)*dx, (c[1]+1)*dx, (c[2]+1)*dx]
                    a[d] = (c[d] + 0.5)*dx      # centre to centre
                    b[d] = (c[d] + 1.5)*dx
                    a[s0] = c[s0]*dx + p*h0
                    b[s0] = c[s0]*dx + (p + 1)*h0
                    a[s1] = c[s1]*dx + q*h1
                    b[s1] = c[s1]*dx + (q + 1)*h1
                    lo[f*self.k + p*k1 + q] = a
                    hi[f*self.k + p*k1 + q] = b
        return lo, hi

    def _full_boxes(self, cells):
        """The undivided filament of each subdivided cell."""
        dx, d = self.dx, self.axis
        cells = np.atleast_2d(np.asarray(cells))
        lo = np.zeros((len(cells), 3))
        hi = np.zeros((len(cells), 3))
        for f, c in enumerate(cells):
            lo[f] = [c[0]*dx, c[1]*dx, c[2]*dx]
            hi[f] = [(c[0]+1)*dx, (c[1]+1)*dx, (c[2]+1)*dx]
            lo[f][d] = (c[d] + 0.5)*dx
            hi[f][d] = (c[d] + 1.5)*dx
        return lo, hi

    def _neighbour_pairs(self, radius, other=None):
        """Filament index pairs within ``radius`` cells (inf-norm).

        Both directions plus the self pair, so the assembled block is
        symmetric without a second pass. Cached: the cells never move,
        so re-assembly (set_frequency) must not pay the KDTree again.
        """
        ck = (float(radius), other is None)
        hit = self._pairs.get(ck)
        if hit is not None:
            return hit
        from scipy.spatial import cKDTree
        a = cKDTree(self.cells.astype(float))
        if other is None:
            p = a.query_pairs(r=radius, p=np.inf, output_type='ndarray')
            self_i = np.arange(self.nfil)
            fa = np.concatenate([p[:, 0], p[:, 1], self_i])
            fb = np.concatenate([p[:, 1], p[:, 0], self_i])
            self._pairs[ck] = (fa, fb)
            return fa, fb
        b = cKDTree(np.asarray(other, dtype=float))
        pairs = a.query_ball_tree(b, r=radius, p=np.inf)
        fa = np.concatenate([np.full(len(v), i, dtype=np.int64)
                             for i, v in enumerate(pairs)]) \
            if any(pairs) else np.zeros(0, dtype=np.int64)
        fb = np.concatenate([np.asarray(v, dtype=np.int64)
                             for v in pairs if v]) \
            if any(pairs) else np.zeros(0, dtype=np.int64)
        self._pairs[ck] = (fa, fb)
        return fa, fb

    def _build_truncated(self, rc_uu, rc_cross):
        """Assemble Zuu and Zcross as SPARSE, distance-truncated blocks.

        Justified by the decay: the modes are net-zero, so mode-to-
        aggregate is dipole-to-monopole (1/r**2) and mode-to-mode is
        dipole-to-dipole (1/r**3), against 1/r for the ordinary near
        field. Only near pairs are ever evaluated -- the dense form costs
        O(n**2) box evaluations and does not scale (it hangs on a
        23600-filament conductor).

        RADII, and why the two differ (measured error vs untruncated on a
        uniform bar at dx/delta = 4.8):

            rc_uu, rc_cross    1,1     2,2     1,2     2,3     3,4
            error              9.1e-2  2.7e-2  1.2e-2  5.2e-3  4.4e-4

        Note (1,1) -> (1,2) cuts the error nearly 8x while (1,2) -> (2,2)
        makes it WORSE: the cross block needs reach, the mode-mode block
        does not. That is the 1/r**2 vs 1/r**3 asymmetry directly.

        The default (3,4) is chosen for the FFT apply, whose cost is set
        by the padded box (``shape + 2*rc``) and not by nnz, so radius is
        nearly free there. **Under ``use_fft=False`` it is NOT free**:
        pairs grow as (2rc+1)**3, so Zuu goes 27 -> 343 neighbours
        (12.7x) and Zcross 125 -> 729 (5.8x). On a large problem the CSR
        path at these radii will not fit; drop to (1,2) if you must use
        it.
        """
        km = self.km
        mr = np.arange(km)
        fa_u, fb_u = self._neighbour_pairs(rc_uu)
        fa_c, fb_c = self._neighbour_pairs(rc_cross)
        self._check_csr_size(fa_u.size, fa_c.size, rc_uu, rc_cross)
        # Table over the separations that ACTUALLY OCCUR, not over the
        # whole (2rc+1)^3 cube -- otherwise a radius larger than the
        # geometry (or an "untruncated" run) builds a table far bigger
        # than the problem and blows up.
        Du = self.cells[fb_u] - self.cells[fa_u]
        Dc = self.cells[fb_c] - self.cells[fa_c]
        span = int((self.cells.max(axis=0) - self.cells.min(axis=0)).max()) + 1
        w = 2*span + 1

        def key(D):
            return ((D[:, 0] + span)*w + (D[:, 1] + span))*w + D[:, 2] + span

        ku, kc = key(Du), key(Dc)
        uniq, inv = np.unique(np.concatenate([ku, kc]), return_inverse=True)
        iu, ic = inv[:ku.size], inv[ku.size:]
        dz = uniq % w - span
        dy = (uniq // w) % w - span
        dx_ = uniq // (w*w) - span
        Bu, Bc = self._mode_tables(np.stack([dx_, dy, dz], axis=1))
        self.ntable = int(uniq.size)

        red = Bu[iu]
        rows = np.broadcast_to(fa_u[:, None, None]*km + mr[None, :, None],
                               red.shape).ravel()
        cols = np.broadcast_to(fb_u[:, None, None]*km + mr[None, None, :],
                               red.shape).ravel()
        self.Zuu = sp.csr_matrix((red.ravel(), (rows, cols)),
                                 shape=(self.nmode_full, self.nmode_full))

        red = Bc[ic]
        rows = (fa_c[:, None]*km + mr[None, :]).ravel()
        cols = np.broadcast_to(fb_c[:, None], red.shape).ravel()
        self.Zcross = sp.csr_matrix((red.ravel(), (rows, cols)),
                                    shape=(self.nmode_full, self.nfil))
        self.nnz = (int(self.Zuu.nnz), int(self.Zcross.nnz))

    def _mode_tables(self, D):
        """Pre-contracted blocks for each distinct CELL SEPARATION ``D``.

        The sub-bars are congruent boxes on a lattice, so the coupling
        between sub-bar ``p`` of one filament and sub-bar ``q`` of
        another depends only on ``(cell separation, p, q)`` -- it is a
        Toeplitz table. Building it costs ``len(D) * k**2`` kernel
        evaluations (a few thousand) instead of one per PAIR (22 million
        on a 23600-filament conductor, which is what made the first
        version take 487 s). Assembly afterwards is pure index
        arithmetic.

        The mode weights are folded in here too, so the tables hold the
        final ``(km x km)`` and ``(km,)`` blocks rather than raw sub-bar
        couplings. The RAW couplings ``M``/``Mc`` are geometry only and
        are cached by separation set: re-folding a new ``W``
        (set_frequency) then costs two einsums instead of ``len(D)*k**2``
        kernel evaluations.
        """
        D = np.asarray(D, dtype=np.int64)
        key = D.tobytes()
        cached = self._geom.get(key)
        if cached is None:
            from greens import box_pair_stencil_pairs as spair
            k = self.k
            nD = D.shape[0]
            lo0, hi0 = self._sub_boxes(np.zeros((1, 3), dtype=np.int64))
            M = np.empty((nD, k, k))
            Mc = np.empty((nD, k))
            # CHUNK the kernel evaluations. The pair arrays below are
            # (n*k*k, 3), and an untruncated table pushes nD*k**2 into
            # the tens of millions -- unchunked that is GBs of broadcast
            # temporaries for a bounded-size answer.
            step = max(1, 2_000_000 // (k*k))
            for a in range(0, nD, step):
                Dc = D[a:a + step]
                n = Dc.shape[0]
                loD, hiD = self._sub_boxes(Dc)
                floD, fhiD = self._full_boxes(Dc)
                # mode <-> mode: k x k sub-bar pairs per separation
                A_lo = np.broadcast_to(lo0[None, :, None, :], (n, k, k, 3))
                A_hi = np.broadcast_to(hi0[None, :, None, :], (n, k, k, 3))
                B_lo = np.broadcast_to(loD.reshape(n, 1, k, 3), (n, k, k, 3))
                B_hi = np.broadcast_to(hiD.reshape(n, 1, k, 3), (n, k, k, 3))
                S = spair(A_lo.reshape(-1, 3), A_hi.reshape(-1, 3),
                          B_lo.reshape(-1, 3), B_hi.reshape(-1, 3))
                M[a:a + step] = (S/(self.asub*self.asub)).reshape(n, k, k)
                # mode <-> aggregate: k sub-bars against the full bar
                A_lo = np.broadcast_to(lo0[None, :, :], (n, k, 3))
                A_hi = np.broadcast_to(hi0[None, :, :], (n, k, 3))
                Sc = spair(A_lo.reshape(-1, 3), A_hi.reshape(-1, 3),
                           np.repeat(floD, k, axis=0),
                           np.repeat(fhiD, k, axis=0))
                Mc[a:a + step] = (Sc/(self.asub*self.afull)).reshape(n, k)
            cached = self._geom[key] = (M, Mc)
        M, Mc = cached
        W = self.W
        Bu = np.einsum('pm,dpq,qr->dmr', W, M, W)
        Bc = np.einsum('pm,dp->dm', W, Mc)
        return Bu, Bc

    def build_fft(self, rc_uu, rc_cross):
        """Spectra for applying the mode blocks as CONVOLUTIONS.

        The blocks are translation invariant, so

            out_u[f,m] = sum_g sum_n Bu[cell_g - cell_f, m, n] u[g,n]

        is a 3-D CORRELATION of ``u`` with ``Bu`` for each mode pair, and
        the transpose (modes back onto the aggregate filaments) is the
        matching CONVOLUTION. Applying them by FFT replaces ~26M nnz of
        index-chasing CSR with ``2*km`` transforms plus ``km**2``
        pointwise products.

        The kernels are REAL -- they are mutual inductances -- so ONE
        spectrum serves both directions: convolution uses it directly,
        correlation its conjugate.

        NOTE this is a WHOLE-BOUNDING-BOX transform, so its cost is
        box-proportional, not occupancy-proportional like
        ``leaf_induct.p2pcpu``. On a sparse geometry the slab-swept
        per-leaf-box form would be the architecturally consistent
        choice; this is the simple version, and it can be switched off.
        """
        from scipy import fft as sfft
        rc = max(int(rc_uu), int(rc_cross))
        self.org = self.cells.min(axis=0)
        shape = (self.cells.max(axis=0) - self.org + 1).astype(int)
        # Tabulate only separations that CAN OCCUR: the stencil clipped
        # per axis to the grid extent. Same lesson _build_truncated
        # already carries -- an rc larger than the geometry (the
        # untruncated studies) otherwise builds a (2rc+1)**3 table FAR
        # bigger than the problem: rc=51 on a 50x10x10 wire is 103**3 =
        # 1.09M separations where only 99x19x19 = 36k exist, 88M kernel
        # evaluations and ~30 GB of broadcast temporaries for a table
        # that is 97% ZEROS (there is no pair at those separations, and
        # the pad below never wraps far enough to read them). Clipping
        # also shrinks the pad, so the per-matvec transforms get cheaper
        # exactly when rc is generous.
        rcs = [min(rc, int(s) - 1) for s in shape]
        rngs = [np.arange(-r, r + 1) for r in rcs]
        D = np.stack(np.meshgrid(*rngs, indexing='ij'),
                     axis=-1).reshape(-1, 3)
        Bu, Bc = self._mode_tables(D)
        self.ntable = int(D.shape[0])
        km = self.km
        inf = np.abs(D).max(axis=1)
        Bu[inf > rc_uu] = 0.0            # respect each block's own radius
        Bc[inf > rc_cross] = 0.0
        self.grid = tuple(int(v) for v in shape)
        self.pad = tuple(sfft.next_fast_len(int(s) + 2*r)
                         for s, r in zip(shape, rcs))
        idx = self.cells - self.org
        self.gidx = ((idx[:, 0]*self.grid[1] + idx[:, 1])*self.grid[2]
                     + idx[:, 2])
        # Delta = 0 at the origin, negative offsets wrapped to the end
        Ku = np.zeros((km, km) + self.pad)
        Kc = np.zeros((km,) + self.pad)
        wrap = tuple(np.mod(D[:, a], self.pad[a]) for a in range(3))
        for m in range(km):
            Kc[(m,) + wrap] = Bc[:, m]
            for n2 in range(km):
                Ku[(m, n2) + wrap] = Bu[:, m, n2]
        self.Fu = sfft.fftn(Ku, axes=(2, 3, 4))
        self.Fc = sfft.fftn(Kc, axes=(1, 2, 3))
        self._sfft = sfft

    def _scatter(self, v):
        """Values on the mode filaments -> zero-padded grid."""
        flat = np.zeros(int(np.prod(self.grid)), dtype=np.complex128)
        flat[self.gidx] = v
        g = np.zeros(self.pad, dtype=np.complex128)
        g[:self.grid[0], :self.grid[1], :self.grid[2]] = \
            flat.reshape(self.grid)
        return g

    def _gather(self, g):
        return g[:self.grid[0], :self.grid[1],
                 :self.grid[2]].reshape(-1)[self.gidx]

    def apply_fft(self, u, i_f):
        """Mode blocks by convolution: returns (out_u, out_f).

        The convolution needs one amplitude per (filament, mode) for
        EVERY parallel filament, so a masked mode set is expanded with
        zeros here and compressed on the way out.
        """
        sfft = self._sfft
        km = self.km
        masked = self.nmode != self.nmode_full
        if masked:
            uf = np.zeros(self.nmode_full, dtype=np.complex128)
            uf[self.mode_mask] = u
            u = uf
        U = np.stack([sfft.fftn(self._scatter(u[m::km])) for m in range(km)])
        F = sfft.fftn(self._scatter(i_f))
        out_u = np.empty(u.size, dtype=np.complex128)
        for m in range(km):
            acc = self.Fc[m].conj()*F               # correlation
            for n2 in range(km):
                acc += self.Fu[m, n2].conj()*U[n2]
            out_u[m::km] = self._gather(sfft.ifftn(acc))
        accf = np.zeros(self.pad, dtype=np.complex128)
        for m in range(km):
            accf += self.Fc[m]*U[m]                 # convolution
        if masked:
            out_u = out_u[self.mode_mask]
        return out_u, self._gather(sfft.ifftn(accf))

    def _build_terminal(self, term, radius):
        km = self.km
        tcell = np.array([f[0] for f in term.faces], dtype=np.int64)
        fa, tb = self._neighbour_pairs(radius, other=tcell.astype(float))
        if fa.size == 0:
            self.Zt = sp.csr_matrix((self.nmode_full, term.n))
            self.ntable_t = 0
            return
        D, inv = self._uniq_sep(tcell[tb] - self.cells[fa])
        Bt = self._terminal_table(D, term.t_l)
        self.ntable_t = int(D.shape[0])
        si = (term.sign[tb] > 0).astype(np.int64)
        red = Bt[inv, :, si]                      # (npair, km)
        rows = (fa[:, None]*km + np.arange(km)[None, :]).ravel()
        cols = np.broadcast_to(tb[:, None], red.shape).ravel()
        self.Zt = sp.csr_matrix((red.ravel(), (rows, cols)),
                                shape=(self.nmode_full, term.n))

    def _check_csr_size(self, npair_uu, npair_cross, rc_uu, rc_cross):
        """Refuse to build sparse blocks that will not fit.

        The default radii are chosen for the FFT apply, where cost is
        padding and a wider radius is nearly free. Under
        ``use_fft=False`` the same radii are ruinous -- pairs grow as
        ``(2rc+1)**3``, so on a 23600-filament conductor at (3,4) the
        blocks want ~5e8 nonzeros, about 6 GB, and the process is
        OOM-killed with no diagnosis at all.

        Checked AFTER the pair search (whose own memory is ~1% of the
        blocks') so the count is exact rather than a bound, and a sparse
        geometry is not refused on a worst-case estimate.
        """
        km = self.km
        nnz = npair_uu*km*km + npair_cross*km
        gb = 12.0*nnz/1e9          # 8 B data + 4 B index per nonzero
        if gb <= self.csr_max_gb:
            return
        raise RuntimeError(
            "redistribution CSR blocks would need ~%.1f GB (%d nonzeros) at "
            "rc_uu=%d, rc_cross=%d with k=%d. These radii are the defaults "
            "for use_fft=True, where a wider radius costs only padding. "
            "Either keep use_fft=True, or drop to rc_uu=1, rc_cross=2 for "
            "the sparse path, or raise csr_max_gb past %.1f if you really "
            "have the memory."
            % (gb, nnz, rc_uu, rc_cross, self.k, self.csr_max_gb))

    def _uniq_sep(self, D):
        """Distinct cell separations in ``D``, and the per-row index."""
        span = int(np.abs(D).max()) + 1
        w = 2*span + 1
        key = ((D[:, 0] + span)*w + (D[:, 1] + span))*w + D[:, 2] + span
        uniq, inv = np.unique(key, return_inverse=True)
        dz = uniq % w - span
        dy = (uniq // w) % w - span
        dx_ = uniq // (w*w) - span
        return np.stack([dx_, dy, dz], axis=1), inv

    def _terminal_table(self, D, t_l):
        """Mode<->terminal blocks per (separation, face sign).

        Same translation invariance as the mode tables: a terminal sits
        at a lattice cell and its bar is fixed by the face sign, so the
        coupling depends only on ``(cell separation, sub-bar, sign)``.
        Two signs, so the table is twice the mode one -- still a few
        thousand kernel evaluations against one per PAIR, which at
        rc_cross = 4 with 800 terminals was ~2M and dominated setup.
        Geometry cached per (separations, t_l), same reason as
        ``_mode_tables``.
        """
        k, W = self.k, self.W
        nD = D.shape[0]
        key = (D.tobytes(), float(t_l))
        cached = self._geom_t.get(key)
        if cached is None:
            from greens import box_pair_stencil_pairs as spair
            lo0, hi0 = self._sub_boxes(np.zeros((1, 3), dtype=np.int64))
            A_lo = np.broadcast_to(lo0[None, :, :], (nD, k, 3)).reshape(-1, 3)
            A_hi = np.broadcast_to(hi0[None, :, :], (nD, k, 3)).reshape(-1, 3)
            Mct = np.zeros((nD, k, 2))
            for si, sgn in enumerate((-1, +1)):
                tlo, thi = self._term_boxes(D, np.full(nD, sgn), t_l)
                S = spair(A_lo, A_hi,
                          np.repeat(tlo, k, axis=0), np.repeat(thi, k, axis=0))
                Mct[:, :, si] = (S/(self.asub*self.afull)).reshape(nD, k)
            cached = self._geom_t[key] = Mct
        Bt = np.einsum('pm,dps->dms', W, cached)
        return Bt

    def _term_boxes(self, cells, signs, t_l):
        """Terminal bars at the given cells and face signs."""
        dx, d = self.dx, self.axis
        cells = np.atleast_2d(np.asarray(cells))
        lo = np.zeros((len(cells), 3))
        hi = np.zeros((len(cells), 3))
        for t, c in enumerate(cells):
            lo[t] = [c[0]*dx, c[1]*dx, c[2]*dx]
            hi[t] = [(c[0]+1)*dx, (c[1]+1)*dx, (c[2]+1)*dx]
            mid = (c[d] + 0.5)*dx
            if signs[t] > 0:
                lo[t][d], hi[t][d] = mid, mid + t_l
            else:
                lo[t][d], hi[t][d] = mid - t_l, mid
        return lo, hi

    def _terminal_boxes(self, term):
        dx, d = self.dx, self.axis
        lo = np.zeros((term.n, 3))
        hi = np.zeros((term.n, 3))
        for t, (cell, sgn, _) in enumerate(term.faces):
            a = [cell[0]*dx, cell[1]*dx, cell[2]*dx]
            b = [(cell[0]+1)*dx, (cell[1]+1)*dx, (cell[2]+1)*dx]
            mid = (cell[d] + 0.5)*dx
            if sgn > 0:
                a[d], b[d] = mid, mid + term.t_l
            else:
                a[d], b[d] = mid - term.t_l, mid
            lo[t] = a
            hi[t] = b
        return lo, hi

    def _check_aggregate(self):
        """The aggregate block must BE the full filament operator.

        ``W_agg^T Z_sub W_agg == Z_full`` holds because the volume
        integral over a cross-section is the sum over its pieces. If it
        failed, the change of basis would be wrong and the existing FFT
        near field silently replaced rather than reused. Checked on a
        small corner of the problem -- it is an algebraic identity, so a
        subset suffices to catch a wiring error.
        """
        n = min(12, self.nfil)
        ns = n*self.k
        Zs = tm.box_mutual_matrix(self.lo[:ns], self.hi[:ns], self.axis)
        Wa = np.zeros((ns, n))
        for f in range(n):
            Wa[f*self.k:(f + 1)*self.k, f] = 1.0/self.k
        full = tm.box_mutual_matrix(self.flo[:n], self.fhi[:n], self.axis)
        got = Wa.T @ Zs @ Wa
        err = np.abs(got - full).max()/np.abs(full).max()
        if err > 1e-10:
            raise RuntimeError("aggregate block does not reproduce the full "
                               "filament operator (rel %.3e)" % err)
        self.aggregate_err = float(err)


class CouplerUnavailable(RuntimeError):
    """The FMM coupling does not apply to this tree/port geometry.

    Raised only for the two GEOMETRIC preconditions, which are reasons
    to fall back to the dense block, not bugs. Anything else the coupler
    detects (a non-contiguous parallel-filament block, say) stays a bare
    RuntimeError, because it means an assumption broke.
    """


class TerminalCoupler:
    """Terminal<->interior mutual inductance, split near/far by the tree.

    The dense ``C`` block is O(n_t * N) in memory and per matvec. This
    replaces it with the FMM's own decomposition:

      NEAR (the terminal's leaf box and its 26 neighbours -- exactly what
      ``leaf.p2p`` covers): direct, with the exact half-step kernel.
      FAR (everything the M2L ladder delivers): the terminal enters the
      sweep as an ORDINARY POINT SOURCE at its centroid with weight
      ``i_t * t_l/l_axis``.

    The weight is the whole trick and it needs no new mathematics.
    ``levels.leafinit`` already models a filament as a point at its
    lattice position with prefactor ``m0 = mu0/(4pi)*l_axis**2`` -- the
    far limit of a bar of length ``l_axis`` -- so a bar of length ``t_l``
    is the same point source scaled by ``t_l/l_axis``. Measured against
    the exact kernel: ratio 0.9940 / 0.9984 / 0.9996 / 0.9999 at 3 / 6 /
    12 / 24 cells of separation, i.e. O((l/r)^2), the same order the FMM
    already carries for ordinary filaments. Reading back at a terminal
    uses ``m0`` scaled the same way, so the two directions stay
    symmetric, and terminal<->terminal FAR pairs come out with
    ``(t_l/l_axis)*(t_l*l_axis) = t_l**2`` -- correct without special
    casing, which is why :attr:`Ltt` must be restricted to NEAR pairs.

    THE BOUNDARY IS THE TREE'S, NOT A DISTANCE. Near must be exactly the
    27-box neighbourhood the FMM excludes from its interaction lists;
    anything else double counts or drops silently.
    """

    def __init__(self, term, M, fil_axis, fil_cell, csel):
        self.term = term
        self.M = M
        self.csel = csel
        leaf = {0: M.f, 1: M.e, 2: M.g}[term.axis]
        self.leaf = leaf
        self.orientation = leaf.orientation
        n = np.asarray(leaf.n, dtype=np.int64)
        self.n = n
        d = term.axis
        # box of each terminal, and of each parallel filament
        tcell = np.array([f[0] for f in term.faces], dtype=np.int64)
        self.tbox = tcell // n
        gbox, gof = self._groups(leaf)
        self.gbox, self.gof = gbox, gof
        # Checked HERE rather than in the p2m hook: a missing box is a
        # property of the geometry, so it must be known at construction
        # (when we can still fall back), not discovered mid-matvec.
        missing = [k for k in range(term.n)
                   if tuple(tcell[k] // n) not in gof]
        if missing:
            raise CouplerUnavailable(
                "%d terminal(s) sit in a leaf box holding no filament of "
                "the port's own orientation, so their multipole moments "
                "have nowhere to attach" % len(missing))
        # csel selects the filaments PARALLEL to the port axis, which is
        # exactly this leaf's block of the e|f|g buffer, in leaf order --
        # so the hooks can address leaf.data directly. Assert it rather
        # than assume: a mismatch would corrupt a different orientation.
        nfil = np.asarray(leaf.idx).size
        if csel.size != nfil or not np.array_equal(csel,
                                                   np.arange(csel[0],
                                                             csel[0] + nfil)):
            raise RuntimeError("parallel-filament block is not this leaf's "
                               "contiguous slice (%d vs %d)"
                               % (csel.size, nfil))
        self._build_near(term, M, fil_axis, fil_cell, leaf)
        self._build_tables(leaf, tcell)
        self.i_t = None
        self.i_f = None
        self.out_t = None

    def _groups(self, leaf):
        """(box coords, group index) for every leaf group."""
        i0 = np.asarray(leaf.idx0, dtype=np.int64)
        bx = np.asarray(leaf.xidx, dtype=np.int64)
        by = np.asarray(leaf.yidx, dtype=np.int64)
        bz = np.asarray(leaf.zidx, dtype=np.int64)
        box = np.stack([bx, by, bz], axis=1)
        return box, {tuple(b): g for g, b in enumerate(box)}

    def _build_near(self, term, M, fil_axis, fil_cell, leaf):
        """Evaluate ONLY the pairs inside the 27-box neighbourhood.

        The near filaments of a terminal are gathered through the leaf's
        group structure (each group is one box, ``idx0`` gives its slice)
        rather than by masking a dense array -- forming the (n_t x N)
        block even once, if only to throw most of it away, would defeat
        the purpose.
        """
        i0 = np.asarray(leaf.idx0, dtype=np.int64)
        restrict = []
        for k in range(term.n):
            b = self.tbox[k]
            parts = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        g = self.gof.get((b[0]+dx, b[1]+dy, b[2]+dz))
                        if g is not None and i0[g+1] > i0[g]:
                            parts.append(np.arange(i0[g], i0[g+1]))
            restrict.append(np.concatenate(parts) if parts
                            else np.zeros(0, dtype=np.int64))
        self.Cnear, _ = term.coupling(fil_axis, fil_cell, restrict=restrict)
        npar = self.csel.size
        self.near_frac = (float(self.Cnear.nnz)/(term.n*npar)
                          if npar else 0.0)
        # terminal<->terminal: the FMM supplies the far pairs with the
        # correct t_l**2 prefactor on its own, so the dense Ltt must keep
        # only the near ones or every far pair is counted twice
        self.tnear = (np.abs(self.tbox[None, :, :] - self.tbox[:, None, :])
                      .max(axis=2) <= 1)

    def _build_tables(self, leaf, tcell):
        """P2M and L2P harmonics at the terminal positions.

        Positions are taken in the SAME box-centred frame ``leafinit``
        uses: transversely the terminal sits on the cell lattice, exactly
        like a filament, so only the axial coordinate is off-lattice --
        a terminal on the low face of cell ``c`` is centred a quarter
        cell inside it, while a filament joining cells ``c`` and ``c+1``
        is centred on the face between them.
        """
        from special import sph_harm_of_cos
        d = self.term.axis
        others = [c for c in range(3) if c != d]
        n, l = self.n, np.asarray(leaf.l, dtype=float)
        nt = self.term.n
        pos = np.zeros((nt, 3))
        # axial: the terminal ends at the node (cell centre, c + 1/2) and
        # runs t_l outward, so its CENTROID is half a terminal length to
        # either side; the box origin is box*n + (n+1)/2
        axc = (tcell[:, d] + 0.5
               + np.where(self.term.hi, +0.5, -0.5)*self.term.tau)
        pos[:, d] = l[d]*(axc - self.tbox[:, d]*n[d] - (n[d] + 1)/2.0)
        for i in others:
            pos[:, i] = l[i]*(tcell[:, i] - self.tbox[:, i]*n[i]
                              - (n[i] - 1)/2.0 - 0.5)
        r = np.sqrt((pos**2).sum(axis=1))
        if (r == 0).any():
            raise RuntimeError("terminal sits on its box centre")
        theta = np.pi - np.arccos(pos[:, 2]/r)      # z-FLIPPED frame
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        nmax = leaf.nmax
        self.mfil = np.zeros((leaf.nnmax, nt), dtype=np.complex128)
        self.ynmr = np.zeros((nt, leaf.nnmax), dtype=np.complex128)
        MU0 = 4e-7*np.pi
        m0mix = MU0/(4*np.pi)*self.term.t_l*l[d]   # t_l * l_axis
        for nn in range(nmax + 1):
            rn = r**nn
            for m in range(-nn, nn + 1):
                idx = nn**2 + nn + m
                self.mfil[idx, :] = rn*sph_harm_of_cos(nn, -m, theta, phi)
                self.ynmr[:, idx] = m0mix*rn*sph_harm_of_cos(nn, m, theta,
                                                             phi)
        # source weight relative to an ordinary filament of this leaf:
        # the far field of a bar of length t_l is that of one of length
        # l_axis scaled by t_l/l_axis (validated to O((l/r)^2))
        self.wsrc = self.term.t_l/l[d]
        # Scatter/gather as MATRICES, so both hooks are one multiply
        # instead of a Python loop over terminals: `tgroup` is each
        # terminal's leaf-box (group) index, and `Pscatter` sums the
        # terminals sharing a box -- several usually do, so this is not
        # a permutation and cannot be done by fancy indexing alone.
        self.tgroup = np.array([self.gof[tuple(b)] for b in self.tbox])
        msize = np.size(leaf.idx0) - 1
        self.Pscatter = sp.csr_matrix(
            (np.full(nt, self.wsrc), (self.tgroup, np.arange(nt))),
            shape=(msize, nt))

    # -- the three hooks tree.traverseRL calls -------------------------

    def p2m(self, leaf):
        if leaf is not self.leaf:
            return
        leaf.above.data += self.Pscatter.dot(self.i_t[:, None]*self.mfil.T)

    def p2p(self, leaf):
        if leaf is not self.leaf:
            return
        leaf.data += self.Cnear.T.dot(self.i_t)
        self.out_t += self.Cnear.dot(self.i_f)

    def l2p(self, leaf):
        if leaf is not self.leaf:
            return
        self.out_t += (self.ynmr*leaf.above.data[self.tgroup]).sum(axis=1)


# -- the solver ---------------------------------------------------------

class EquiTerminalSolver:
    """LpR port solve with the terminal currents as unknowns.

    Parameters
    ----------
    model : vhr.VhrModel
    M : multipole.Tree
        Already prepared (``VhrModel.prepare``).
    port : int or str, optional
    verbose : bool, optional
    """

    def __init__(self, model, M, port=0, verbose=False, fmm=True, t_l=None,
                 subdivide=False, split_axis=None, rc_uu=3,
                 rc_cross=4, skin_freq=None, use_fft=True,
                 csr_max_gb=2.0, chol_mode='simplicial',
                 chol_ordering='metis', mode_basis='diff',
                 boundary_only=False, basis='selected', amg_cycles=4,
                 amg_cycle_type='V', amg_smoother=None,
                 amg_strength=None, gram_solver='amg'):
        self.M = M
        self.model = model
        self.verbose = verbose
        self.chol_mode = chol_mode
        self.chol_ordering = chol_ordering
        self.basis = str(basis)
        self.amg_cycles = int(amg_cycles)
        self.amg_cycle_type = str(amg_cycle_type)
        self.amg_smoother = amg_smoother
        self.amg_strength = amg_strength
        self.gram_solver = str(gram_solver)
        if self.basis not in ('selected', 'overcomplete'):
            raise ValueError("basis must be 'selected' or 'overcomplete', "
                             "got %r" % (basis,))
        self.efg = (np.size(M.e.struc) + np.size(M.f.struc)
                    + np.size(M.g.struc))
        self.nnode = np.size(M.lv[0].struc)
        self.whole = M._vhr_whole
        if self.whole is None or self.whole.size != self.efg + self.nnode:
            raise RuntimeError("tree buffer not allocated -- call "
                               "VhrModel.prepare(M, freq) first")
        t0 = time.perf_counter()
        self.term = Terminals(model, M, port, t_l=t_l)
        self.B, self.ncell = sparse_incidence(M, self.whole, self.efg,
                                              self.nnode)
        self.node_of_cell = {tuple(c): i for i, c in enumerate(self.ncell)}
        self.fil_axis, self.fil_cell = filament_cells(M)
        _check_orientation(self.B, self.fil_axis, self.fil_cell,
                           self.node_of_cell)
        self.csel = self.term.parallel(self.fil_axis)
        self.Ltt = self.term.self_block()
        self.C = None
        self.coupler = None
        self.fmm_reason = None
        if fmm:
            try:
                if M.numlevels < 2:
                    raise CouplerUnavailable(
                        "single-level tree: p2p already covers the whole "
                        "domain, so the dense block IS the near block")
                self.coupler = TerminalCoupler(self.term, M, self.fil_axis,
                                               self.fil_cell, self.csel)
                # far terminal<->terminal pairs now come from the ladder
                self.Ltt = np.where(self.coupler.tnear, self.Ltt, 0.0)
            except CouplerUnavailable as exc:
                # Fall back to the dense block, which is always correct --
                # but RECORD why, so a silent 30x slowdown is diagnosable.
                self.fmm_reason = str(exc)
        if self.coupler is None:
            # only now, and only if we must: this is the O(n_t * N) block
            self.C, _ = self.term.coupling(self.fil_axis, self.fil_cell)
        # The skin-effect switch. False/None/1 = off (default, and
        # exactly the behaviour without this feature); True/'auto' picks
        # k from cell size vs skin depth at `skin_freq` (default: the
        # model's highest frequency, where delta is smallest); an int is
        # taken literally. NOTE `1 == True` in Python, so the identity
        # test must come first. The reference frequency is resolved
        # unconditionally because the 'conduction' basis needs it even
        # when k is given literally.
        eps = getattr(model, 'epsilon', None)
        if eps is not None and bool(np.any(
                (np.asarray(eps) != 1.0) & (model.sigma == 0.0))):
            raise NotImplementedError(
                "this model has PURE-DIELECTRIC cells, whose physics "
                "lives in the charge/potential coupling: solve it "
                "through the capacitive LpPR path, not the LpR-only "
                "equipotential-terminal solver (an excess-capacitance "
                "branch without its bound charge is not a dielectric).")
        skin_off = (subdivide is False or subdivide is None
                    or (subdivide == 1 and subdivide is not True))
        if getattr(model, 'anisotropic', False) and not skin_off:
            raise NotImplementedError(
                "skin-effect subdivision on ANISOTROPIC cells: the "
                "redistribution engine's sub-bars, mode tables and "
                "skin-depth heuristics assume a square cross-section. "
                "Solve with subdivide=False, or resample the model to "
                "cubic voxels first.")
        if getattr(model, 'superconductor', False) and not skin_off:
            raise NotImplementedError(
                "skin-effect subdivision on a superconductor: the mode "
                "engine's Ru and the classical skin depth both assume a "
                "REAL sigma, and the London screening length (not delta) "
                "sets the current profile. Solve with subdivide=False -- "
                "the per-cell complex z(w) already carries the "
                "two-fluid physics at the filament level.")
        fref = skin_freq
        if fref is None:
            fref = float(np.max(model.freq)) if len(model.freq) else 0.0
        self.skin_freq = fref
        self.skin_k = 1
        if subdivide is True or subdivide == 'auto':
            self.skin_k = recommend_subdivision(model.dx,
                                                model.uniform_sigma(), fref)
        elif subdivide in (False, None):
            self.skin_k = 1
        else:
            self.skin_k = int(subdivide)
        subdivide = self.skin_k
        self.redist = None
        if subdivide > 1:
            self.redist = Redistribution(model, M, self.term.axis,
                                         self.fil_axis, self.fil_cell,
                                         split_axis=split_axis,
                                         k=subdivide, term=self.term,
                                         rc_uu=rc_uu, rc_cross=rc_cross,
                                         use_fft=use_fft,
                                         csr_max_gb=csr_max_gb,
                                         mode_basis=mode_basis,
                                         skin_freq=fref,
                                         boundary_only=boundary_only)
        self.nu = 0 if self.redist is None else self.redist.nmode
        self._build_augmented()
        self.t_setup = time.perf_counter() - t0
        self.matvecs = 0
        if verbose:
            print("  %d terminals, %d loops (+%d port cycles, %d hole "
                  "generators), setup %.2f s"
                  % (self.term.n, self.meshsize, self.nportcyc,
                     self.nholes, self.t_setup))
            if self.redist is None:
                print("    skin effect: OFF (no cross-section subdivision)")
            else:
                print("    skin effect: ON, %dx%d sub-filaments -> %d modes"
                      % (self.redist.kk[0], self.redist.kk[1], self.nu))
                print("      Zuu %d nnz, Zcross %d nnz, %d separations "
                      "tabulated" % (self.redist.nnz[0], self.redist.nnz[1],
                                     self.redist.ntable))
            if self.coupler is None:
                print("    terminal coupling: DENSE (%.1f MB) -- %s"
                      % (self.C.nbytes/1e6,
                         self.fmm_reason or "fmm disabled"))
            else:
                dense = 8.0*self.term.n*self.csel.size
                print("    terminal coupling: FMM, near %.1f%% "
                      "(%.1f MB, vs %.1f MB dense)"
                      % (100*self.coupler.near_frac,
                         self.coupler.Cnear.data.nbytes/1e6, dense/1e6))

    # -- topology ------------------------------------------------------

    def _build_augmented(self):
        """Augmented incidence, mesh basis and its Cholesky."""
        M, term = self.M, self.term
        nt, efg, nn = term.n, self.efg, self.nnode
        self.pnode = {+1: nn, -1: nn + 1}
        rows, cols, vals = [], [], []
        for k, (cell, sgn, pol) in enumerate(term.faces):
            cn = self.node_of_cell[cell]
            pn = self.pnode[pol]
            # current positive along +axis: a LOW face runs port -> cell,
            # a HIGH face runs cell -> port
            tail, head = (pn, cn) if sgn < 0 else (cn, pn)
            rows += [k, k]
            cols += [tail, head]
            vals += [1.0, -1.0]
        Bt = sp.coo_matrix((vals, (rows, cols)),
                           shape=(nt, nn + 2)).tocsr()
        Bi = sp.hstack([self.B, sp.csr_matrix((efg, 2))], format='csr')
        blocks = [Bi, Bt]
        if self.nu:
            # redistribution modes carry ZERO net current, so their
            # incidence rows are identically zero -- that is the whole
            # reason they need no nodes and never touch S_d
            blocks.append(sp.csr_matrix((self.nu, nn + 2)))
        self.Baug = sp.vstack(blocks, format='csr')

        # plaquette basis over the interior filaments. 'overcomplete'
        # keeps EVERY plaquette instead of an independent spanning set --
        # singular Gram (kernel = cube boundaries, physically
        # meaningless) but far better conditioned on its range, and AMG
        # then works where it is useless on the selected basis. See
        # port_impedance.LpRSolver's `basis` docs for the measurements.
        if self.basis == 'overcomplete':
            Y = mg.getmesh_full(M.adjmats(), np.size(M.e.struc),
                                np.size(M.e.struc) + np.size(M.f.struc),
                                efg, nn)
        else:
            Y = mg.getmesh_fortran(M.adjmats(), np.size(M.e.struc),
                                   np.size(M.e.struc) + np.size(M.f.struc),
                                   efg, nn)
        Y.data = np.float64(Y.data)
        self.nplaq = Y.shape[1]
        self.nholes = 0
        if self.basis == 'overcomplete':
            holes = self._hole_cycles(Y)
            if holes is not None:
                Y = sp.hstack([Y, holes], format='csc')
                self.nholes = holes.shape[1]
        self.nportcyc = 0
        newc = self._port_cycles()
        if newc is not None:
            Y = sp.hstack([sp.vstack([Y, sp.csr_matrix((nt, Y.shape[1]))],
                                     format='csr'), newc], format='csc')
            self.nportcyc = newc.shape[1]
        else:
            Y = sp.vstack([Y, sp.csr_matrix((nt, Y.shape[1]))], format='csc')
        if self.nu:
            # Each redistribution mode is a branch of ZERO incidence, so
            # it is a cycle all by itself: its basis column is a unit
            # vector. (Also orthogonal to every other column, so Y^T Y
            # just gains an identity block and the Cholesky is unharmed.)
            Y = sp.hstack([sp.vstack([Y, sp.csr_matrix((self.nu,
                                                        Y.shape[1]))],
                                     format='csr'),
                           sp.vstack([sp.csr_matrix((efg + nt, self.nu)),
                                      sp.identity(self.nu, format='csr')],
                                     format='csr')], format='csc')
        self.Y = sp.csc_matrix(Y)
        self.YT = self.Y.T.tocsc()
        self.meshsize = self.Y.shape[1]
        # Cycle rank of the augmented graph: edges - nodes + COMPONENTS.
        # The single-conductor case had components = 1 hard-coded; with
        # galvanically isolated conductors (straight_cond2) the undriven
        # ones stay their own components -- their cycles are plaquettes
        # only, which is exactly right for eddy currents. The two port
        # nodes merge with whatever conductor components the terminals
        # touch, so count with a small union-find; and if P and N end up
        # in DIFFERENT components the drive has no return path at all --
        # B^T i = s is infeasible and lsqr would silently return a
        # least-squares non-solution, so reject loudly here.
        _, _, _, comp = self._spanning_tree()
        root = list(range(self.ncomp + 2))

        def find(a):
            while root[a] != a:
                root[a] = root[root[a]]
                a = root[a]
            return a

        pP, pN = self.ncomp, self.ncomp + 1
        for cell, _, pol in term.faces:
            a = find(pP if pol > 0 else pN)
            b = find(int(comp[self.node_of_cell[cell]]))
            if a != b:
                root[a] = b
        if find(pP) != find(pN):
            raise ValueError(
                "the port's P and N faces sit on galvanically separate "
                "conductors: the driven current has no return path, so "
                "B^T i = s is infeasible")
        ncomp_aug = len({find(a) for a in range(self.ncomp + 2)})
        want = efg + nt + self.nu - (nn + 2) + ncomp_aug
        if self.basis == 'overcomplete':
            # The column count is now the PLAQUETTE count, not the cycle
            # rank, so it can only be required to SPAN. The
            # divergence-free test below is unaffected and remains the
            # check with teeth -- it is what catches a mis-signed port
            # cycle, which the count never did.
            if self.meshsize < want:
                raise RuntimeError("over-complete basis has %d columns, "
                                   "fewer than the cycle space %d"
                                   % (self.meshsize, want))
        elif self.meshsize != want:
            raise RuntimeError("mesh basis has %d columns, cycle space is "
                               "%d -- port cycles are wrong"
                               % (self.meshsize, want))
        # Every column must be a CYCLE, i.e. divergence-free. The right
        # column COUNT plus full rank (which the Cholesky below proves)
        # is not enough: a mis-signed tree path is still independent and
        # still counts, it just is not in the null space of B^T, and the
        # solve then answers a different question -- which is exactly how
        # this first failed, with L looking right to 3e-4 while R was 1%
        # off and the current split was unrecognisable.
        div = np.abs(self.Baug.T.dot(self.Y)).max()
        if div > 1e-9:
            raise RuntimeError("mesh basis is not divergence-free "
                               "(max |B^T Y| = %.3e)" % div)
        YT32 = self.YT.copy()
        YT32.data = np.float32(YT32.data)
        # ('supernodal', 'amd') was hard-coded here and is the WORST of
        # the four combinations measured on a 24k-voxel bar (45201
        # loops): simplicial+metis costs 0.13 GB of factor against
        # 0.21 GB and runs 4.2 s against 12.5 s -- 38% less memory and
        # 3x faster. Same change as port_impedance.LpRSolver; see the
        # table at that call site for the full matrix and the two
        # caveats (the times are SINGLE CORE, and supernodal is the mode
        # that can use threaded BLAS, so the TIME ranking may reverse --
        # the MEMORY result is structural). Exposed rather than
        # hard-coded so a caller can go back.
        if self.basis == 'overcomplete':
            # Y^T Y is singular here, so no Cholesky exists -- and none is
            # wanted: the point of the over-complete basis is that AMG
            # works on its Gram.
            # The port cycles are LONG and break plain AMG (3.0x nnz,
            # no convergence). Split them off and treat them exactly.
            if self.gram_solver == 'geo':
                # GEOMETRIC multigrid on the plaquette block. Only the
                # first nplaq columns are lattice faces; hole cycles,
                # port cycles and redistribution modes have no geometry
                # and go to _GeoMGFactor's exact Schur block.
                import loopmg
                from port_impedance import _GeoMGFactor
                nrm, bse = loopmg.plaquette_geometry(
                    self.Y[:efg, :self.nplaq].tocsc(), self.fil_axis,
                    self.fil_cell, self.nplaq)
                self.chol = _GeoMGFactor(YT32, nrm, bse, self.nplaq,
                                         cycles=self.amg_cycles)
                if self.verbose:
                    print("GeoMG precond apply: %s"
                          % self.chol.gpu_state, flush=True)
            else:
                from port_impedance import _BlockAMGFactor
                # OFF-BY-nholes, FIXED 2026-08-10. Y's columns run
                # [plaquettes | holes | port cycles | modes], so
                # arange(nplaq, nplaq+nportcyc) indexed the HOLE block
                # whenever a conductor was multiply connected, and the
                # PORT cycles -- the long dense ones this split exists
                # to remove -- stayed inside AMG. Measured on a 32^2
                # stitched pdn (nplaq 1490, nholes 22, nportcyc 16):
                # the macro window sat at [1490,1506) while the port
                # cycles were at [1512,1528). Answers were never wrong
                # (a preconditioner cannot change the solution), only
                # the iteration count. Holes are long cycles of the
                # same character, so they join the exact block too.
                macro = np.arange(self.nplaq,
                                  self.nplaq + self.nholes
                                  + self.nportcyc)
                self.chol = _BlockAMGFactor(YT32, macro, self.amg_cycles,
                                            cycle_type=self.amg_cycle_type,
                                            smoother=self.amg_smoother,
                                            strength=self.amg_strength)
        else:
            self.chol = cholmod.cholesky_AAt(YT32, mode=self.chol_mode,
                                             ordering_method=self.chol_ordering)

    def _spanning_tree(self):
        """BFS FOREST over the conductor graph: one tree per component.

        Galvanically isolated conductors are LEGITIMATE (VoxHenry's
        straight_cond2 pair is the corpus case): the undriven conductor
        carries closed eddy currents, which the plaquette basis spans
        without any help from port cycles. Returns ``(parent, pedge,
        depth, comp)`` with ``comp`` the component label per node;
        ``self.ncomp`` is the component count. Cached -- the graph
        never changes after construction.
        """
        if getattr(self, '_forest', None) is not None:
            return self._forest
        B = self.B.tocoo()
        order = np.argsort(B.row, kind='stable')
        r, c, v = B.row[order], B.col[order], B.data[order]
        tail = c[v > 0]
        head = c[v < 0]
        adj = [[] for _ in range(self.nnode)]
        for f in range(self.efg):
            adj[tail[f]].append((head[f], f))
            adj[head[f]].append((tail[f], f))
        parent = -np.ones(self.nnode, dtype=np.int64)
        pedge = -np.ones(self.nnode, dtype=np.int64)
        depth = -np.ones(self.nnode, dtype=np.int64)
        comp = -np.ones(self.nnode, dtype=np.int64)
        ncomp = 0
        for seed in range(self.nnode):
            if depth[seed] >= 0:
                continue
            depth[seed] = 0
            comp[seed] = ncomp
            stack = [seed]
            while stack:
                u = stack.pop()
                for v_, f in adj[u]:
                    if depth[v_] < 0:
                        depth[v_] = depth[u] + 1
                        parent[v_], pedge[v_] = u, f
                        comp[v_] = ncomp
                        stack.append(v_)
            ncomp += 1
        self.tail, self.head = tail, head
        self.ncomp = ncomp
        self._forest = (parent, pedge, depth, comp)
        return self._forest

    def _tree_path(self, u, v, parent, pedge, depth):
        """Tree path u -> v as (filament, sign) with sign along u -> v."""
        out = []
        while u != v:
            if depth[u] >= depth[v]:
                f = pedge[u]
                # traversing u -> parent[u]: WITH the filament when u is
                # its tail, since current is positive tail -> head
                s = +1.0 if self.tail[f] == u else -1.0
                out.append((f, s))
                u = parent[u]
            else:
                f = pedge[v]
                s = +1.0 if self.tail[f] == parent[v] else -1.0
                out.append((f, s))
                v = parent[v]
        return out

    def _hole_cycles(self, Y):
        """Hole-encircling generators completing the plaquette span.

        On a multiply-connected conductor (a power plane with antipad
        holes, a slot, a handle) the plaquette basis cannot span the
        cycle space: a loop around a hole is not a sum of plaquettes,
        because the hole removed exactly the plaquettes that would
        tile it (measured on pdn_planes 320: 185442 plaquettes vs a
        185844-dimensional cycle space -- one missing generator per
        antipad plus the slot).

        The completing set is found with a tree-cotree collapse
        (discrete-Morse style): walk plaquettes that currently contain
        exactly ONE unmatched non-tree filament and pair them off;
        every non-tree filament left unmatched gets its spanning-tree
        fundamental cycle appended as a generator. This SPANS by
        induction over the collapse order: each matched filament's
        fundamental cycle equals its plaquette minus fundamental
        cycles of earlier-matched or unmatched filaments, so
        plaquettes + unmatched generators reach every fundamental
        cycle. The collapse can in principle leave more generators
        than the true deficit (it is a matching, not an elimination),
        which the overcomplete basis absorbs by design; on plane
        geometry it is exact. Each generator is a genuine cycle, so
        the divergence-free check downstream applies to it unchanged.

        Returns a CSC matrix of shape ``(efg, K)``, or None when the
        plaquettes already span (every simply-connected conductor).
        """
        from collections import deque
        parent, pedge, depth, comp = self._spanning_tree()
        nontree = np.ones(self.efg, dtype=bool)
        te = pedge[pedge >= 0]
        nontree[te] = False
        Yc = Y.tocsc()
        nplaq = Yc.shape[1]
        indptr, indices = Yc.indptr, Yc.indices
        matched = np.zeros(self.efg, dtype=bool)
        members = []
        edge_plaq = {}
        free = np.zeros(nplaq, dtype=np.int64)
        for j in range(nplaq):
            mem = [int(f) for f in indices[indptr[j]:indptr[j+1]]
                   if nontree[f]]
            members.append(mem)
            free[j] = len(mem)
            for f in mem:
                edge_plaq.setdefault(f, []).append(j)
        # a plaquette's 4 filaments cannot all be tree edges (they
        # would close a cycle in the tree), so free >= 1 initially
        queue = deque(np.flatnonzero(free == 1).tolist())
        gens = []

        def drain():
            while queue:
                j = queue.popleft()
                if free[j] != 1:
                    continue
                f = next(f for f in members[j] if not matched[f])
                matched[f] = True
                for j2 in edge_plaq[f]:
                    free[j2] -= 1
                    if free[j2] == 1:
                        queue.append(j2)

        # Collapse until dry, then break each stall by DECLARING one
        # unmatched filament a generator and resuming -- one generator
        # per stall keeps the count near the true homology deficit
        # (dumping every leftover at once gave 235 generators for a
        # 40-dimensional deficit on the 48-cell pdn test, and the long
        # redundant columns degraded lgmres). Prefer the stalled
        # filament of smallest tree-depth sum: its fundamental cycle
        # is shortest.
        drain()
        loose = [int(f) for f in np.flatnonzero(nontree & ~matched)]
        while loose:
            f = min(loose, key=lambda f: depth[self.tail[f]]
                    + depth[self.head[f]])
            gens.append(f)
            matched[f] = True
            for j2 in edge_plaq.get(f, ()):
                free[j2] -= 1
                if free[j2] == 1:
                    queue.append(j2)
            drain()
            loose = [int(f) for f in np.flatnonzero(nontree & ~matched)]
        if not gens:
            return None
        rows, cols, vals = [], [], []
        for k, f in enumerate(gens):
            rows.append(int(f))
            cols.append(k)
            vals.append(1.0)
            for f2, s in self._tree_path(int(self.head[f]),
                                         int(self.tail[f]),
                                         parent, pedge, depth):
                rows.append(f2)
                cols.append(k)
                vals.append(s)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.efg, len(gens))).tocsc()

    def _port_cycles(self):
        """One cycle per consecutive pair of faces on the same terminal.

        Terminal ``k`` and ``k+1`` both meet the shared port node, so
        ``term_k -> (conductor path) -> term_{k+1} -> back`` is a cycle
        that the interior basis cannot span. There are ``n_faces - 1``
        of them per terminal, which is exactly the growth of the cycle
        space, and each is 3 nonzeros whenever the two port faces are
        neighbouring cells -- the usual case for a port face set.
        """
        term = self.term
        if term.n < 3:
            return None
        parent, pedge, depth, comp = self._spanning_tree()
        rows, cols, vals = [], [], []
        k = 0
        for pol in (+1, -1):
            grp = np.flatnonzero(term.pol == pol)
            gc = {int(comp[self.node_of_cell[term.faces[a][0]]])
                  for a in grp}
            if len(gc) > 1:
                # The consecutive-pair construction walks a CONDUCTOR
                # path between the two faces; across components there
                # is none, and the cross-component cycles (which run
                # through the shared port node) would be silently
                # missing from the basis -- the count check would fire
                # with a misleading message. Reject clearly instead.
                raise ValueError(
                    "the %s faces of the port span %d galvanically "
                    "separate conductors -- port cycles need each "
                    "face group on one conductor"
                    % ('P' if pol > 0 else 'N', len(gc)))
            for a, b in zip(grp[:-1], grp[1:]):
                ca = self.node_of_cell[term.faces[a][0]]
                cb = self.node_of_cell[term.faces[b][0]]
                # traverse terminal a from the port into the conductor,
                # cross to cell b, and come back out through terminal b
                sa = +1.0 if term.sign[a] < 0 else -1.0
                rows.append(self.efg + a)
                cols.append(k)
                vals.append(sa)
                for f, s in self._tree_path(ca, cb, parent, pedge, depth):
                    rows.append(f)
                    cols.append(k)
                    vals.append(s)
                sb = -1.0 if term.sign[b] < 0 else +1.0
                rows.append(self.efg + b)
                cols.append(k)
                vals.append(sb)
                k += 1
        if k == 0:
            return None
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.efg + term.n, k)).tocsc()

    # -- the operator --------------------------------------------------

    def apply_Z(self, x):
        """``Z_aug x`` for the augmented current vector ``[i_f; i_t]``.

        With ``fmm=True`` the terminal<->interior coupling rides along in
        the FMM sweep (near direct, far through the ladder) instead of
        being a dense gemv; the terminal contributions are injected
        BEFORE ``traverseRL``'s ``jomega`` multiply, so they pick it up
        automatically, while the terminal ROWS collect raw mutual
        inductances and are scaled here.
        """
        efg, nt = self.efg, self.term.n
        i_f, i_t = x[:efg], x[efg:efg+nt]
        u = x[efg+nt:] if self.nu else None
        self.whole[:efg] = i_f
        jw = self.M.jomega
        if self.coupler is None:
            self.M.traverseRL()
            out_f = np.array(self.whole[:efg])
            out_f[self.csel] += jw*(self.C.T @ i_t)
            out_t = (self.term.R*i_t + jw*(self.Ltt @ i_t)
                     + jw*(self.C @ i_f[self.csel]))
        else:
            c = self.coupler
            c.i_t = i_t
            c.i_f = np.ascontiguousarray(i_f[self.csel])
            c.out_t = np.zeros(self.term.n, dtype=np.complex128)
            self.M.traverseRL(extra=c)
            out_f = np.array(self.whole[:efg])
            out_t = self.term.R*i_t + jw*(self.Ltt @ i_t) + jw*c.out_t
        if not self.nu:
            return np.concatenate([out_f, out_t])
        # Redistribution modes. The AGGREGATE current already went
        # through traverseRL untouched; these are the net-zero
        # corrections, which couple back to the aggregate filaments and
        # (if parallel to them) to the terminals. Everything here is
        # dipolar and short ranged -- kept dense for correctness.
        r = self.redist
        sel = r.sel
        if r.use_fft:
            mu, mf = r.apply_fft(u, np.ascontiguousarray(i_f[sel]))
            out_u = r.Ru @ u + jw*mu
            out_f[sel] += jw*mf
        else:
            out_u = r.Ru @ u + jw*(r.Zuu @ u + r.Zcross @ i_f[sel])
            out_f[sel] += jw*(r.Zcross.T @ u)
        if r.Zt is not None:
            out_u += jw*(r.Zt @ i_t)
            out_t += jw*(r.Zt.T @ u)
        return np.concatenate([out_f, out_t, out_u])

    def _mesh_matvec(self, w):
        self.matvecs += 1
        return self.YT.dot(self.apply_Z(self.Y.dot(w)))

    # -- the solve -----------------------------------------------------

    def solve(self, freq, current=1.0, rtol=1e-12, maxiter=30, inner_m=None,
              lsqr_tol=1e-12, method='lgmres', precision='auto'):
        """Solve at one frequency; return ``(Z, i, info)``.

        ``Z`` is the port impedance ``(phi_P - phi_N)/I``. ``i`` is the
        augmented current vector; ``i[efg:]`` is the SOLVED terminal
        current split, the quantity the prescribed model had to guess.
        ``method``: see :func:`port_impedance.krylov_solve`.
        """
        self.model.prepare(self.M, freq)
        self.term.set_frequency(freq)   # superconductor z(w); else no-op
        t0 = time.perf_counter()
        # The conduction basis tracks the solve frequency: its shapes
        # are exponentials in the skin depth. `skin_freq` only seeds the
        # initial W (and the 'auto' k choice); every solve retunes. The
        # delta-dependent pruning can change km, and the augmented
        # system is sized by nu = nmode, so a changed count rebuilds the
        # mesh basis and its Cholesky -- rare across a sweep (km moves
        # only when a mode degenerates), and the retune itself only
        # re-folds cached geometry.
        if self.redist is not None and self.redist.set_frequency(freq):
            if self.redist.nmode != self.nu:
                self.nu = self.redist.nmode
                self._build_augmented()
        n = self.efg + self.term.n + self.nu
        s = np.zeros(self.nnode + 2, dtype=np.complex128)
        s[self.pnode[+1]] = current
        s[self.pnode[-1]] = -current
        BT = self.Baug.T.tocsc().astype(np.complex128)
        ihat = lsqr(BT, s, atol=lsqr_tol, btol=lsqr_tol)[0]
        rhs = -self.YT.dot(self.apply_Z(ihat))
        Aop = LinearOperator((self.meshsize,)*2, matvec=self._mesh_matvec,
                             dtype=np.complex128)
        Pop = LinearOperator((self.meshsize,)*2, matvec=self._precond,
                             dtype=np.complex128)
        n0 = self.matvecs
        from port_impedance import krylov_solve
        w, flag = krylov_solve(Aop, rhs, Pop, method=method, rtol=rtol,
                               maxiter=maxiter, inner_m=inner_m,
                               precision=precision)
        nrhs = np.linalg.norm(rhs)
        resid = np.linalg.norm(rhs - Aop*w)/nrhs if nrhs > 0 else 0.0
        i = self.Y.dot(w) + ihat
        # Z i = B phi with phi the PHYSICAL potential (see the module
        # docstring): no sign flip to undo here.
        zi = self.apply_Z(i)
        phi = lsqr(self.Baug.astype(np.complex128), zi,
                   atol=lsqr_tol, btol=lsqr_tol)[0]
        v = (phi[self.pnode[+1]] - phi[self.pnode[-1]])/current
        info = dict(matvecs=self.matvecs - n0, flag=flag, residual=resid,
                    time=time.perf_counter() - t0)
        if self.verbose:
            print("    %.4g Hz: %d matvecs, flag %s, resid %.2e, %.2f s"
                  % (freq, info['matvecs'], flag, resid, info['time']))
        return complex(v), i, info

    def _precond(self, vec):
        re = self.chol(np.float32(np.real(vec)))
        im = self.chol(np.float32(np.imag(vec)))
        return np.float64(re) + 1j*np.float64(im)

    def terminal_split(self, i, current=1.0):
        """Solved per-face current entering the conductor, share of I."""
        i_t = i[self.efg:self.efg + self.term.n]
        return -self.term.sign*i_t/current
