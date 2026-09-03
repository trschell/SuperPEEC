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

import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsqr, lgmres
import sksparse.cholmod as cholmod

import meshgraph as mg
import sppeec_status as _spstatus
import terminal as tm
import stencils as st
from enrich import (Split, PairTables, unique_separations,
                    neighbour_pairs, partial_dL)

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


class CellIndex:
    """Dict-like ``cell-tuple -> node id``, backed by sorted keys.

    Replaces the ``{tuple(c): i}`` python dict over every node --
    measured ~0.9 GB of RESIDENT interpreter objects at R4 (4.6M
    tuple keys), built to serve a handful of scalar lookups plus one
    vectorised sweep. Arrays: ~75 MB. ``__getitem__`` for the scalar
    sites, ``lookup`` for whole-array queries."""

    def __init__(self, cells):
        c = np.asarray(cells, dtype=np.int64)
        self._m1 = int(c[:, 1].max()) + 1
        self._m2 = int(c[:, 2].max()) + 1
        keys = (c[:, 0]*self._m1 + c[:, 1])*self._m2 + c[:, 2]
        self._order = np.argsort(keys, kind='stable')
        self._keys = keys[self._order]

    def _key(self, c):
        return (c[..., 0]*self._m1 + c[..., 1])*self._m2 + c[..., 2]

    def __getitem__(self, cell):
        k = (int(cell[0])*self._m1 + int(cell[1]))*self._m2 \
            + int(cell[2])
        i = int(np.searchsorted(self._keys, k))
        if i >= self._keys.size or self._keys[i] != k:
            raise KeyError(tuple(cell))
        return int(self._order[i])

    def lookup(self, cells):
        """Vectorised node ids for an (n, 3) cell array."""
        k = self._key(np.asarray(cells, dtype=np.int64))
        i = np.searchsorted(self._keys, k)
        i = np.minimum(i, self._keys.size - 1)
        if not np.array_equal(self._keys[i], k):
            raise KeyError("cells outside the node set")
        return self._order[i]


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
    # DIRECT csr assembly: every row has exactly two entries, so the
    # COO->CSR sort the old construction paid (part of a measured
    # +2.6 GB build transient at R4) is a per-row column swap.
    # int32 indices and float32 +-1 data ARE the canonical stored
    # form (the shrink pass produced exactly this); the per-row sort
    # reproduces the old canonical column order, so every downstream
    # byte is unchanged.
    cols = np.column_stack([n_ev, n_od]).astype(np.int32)
    vals = np.column_stack([np.sign(r_ev), np.sign(r_od)]
                           ).astype(np.float32)
    del r_ev, r_od, n_ev, n_od, resp
    swap = cols[:, 0] > cols[:, 1]
    cols[swap] = cols[swap][:, ::-1]
    vals[swap] = vals[swap][:, ::-1]
    del swap
    B = sp.csr_matrix((vals.ravel(), cols.ravel(),
                       np.arange(0, 2*efgsize + 1, 2,
                                 dtype=np.int32)),
                      shape=(efgsize, nodesize))
    if np.abs(B.sum(axis=1)).max() > 1e-9:
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
    lo = np.asarray(node_of_cell.lookup(fil_cell))
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
        # Per-face, so a port may touch cells of different materials --
        # or, with subpixel fills, a PARTIAL cell. The superconductor
        # branch of set_frequency has always indexed z per face; this
        # makes the normal-metal branch agree. `sigma` stays the scalar
        # where one exists (terminal_impedance and reporting want it).
        try:
            self.sigma = (None if self.superconductor
                          else model.port_sigma(port))
        except ValueError:
            # a port spanning materials (or touching a partial cell) is
            # fine here: R below is per-face. Keep the scalar unset so
            # anything that still wants one fails loudly rather than
            # silently picking a material.
            self.sigma = None
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
            # per-face sigma, read from the cell each terminal
            # half-filament actually sits in (self.faces is already in
            # that order) -- the same indexing the superconductor
            # branch of set_frequency has always used for z
            cells = np.array([c for c, _, _ in self.faces],
                             dtype=np.int64)
            sig = np.asarray(self._model.sigma, dtype=float)[
                cells[:, 0], cells[:, 1], cells[:, 2]]
            self.R = np.array([tm.terminal_resistance(l, self.orient,
                                                      sv, self.t_l)
                               for sv in sig])
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
        # (p, q, s0) per face-type combination; the kernel is evaluated
        # AT THE REQUESTED POINTS (terminal.unequal_kernel_at, bit-
        # identical to the former tables) rather than tabulated over
        # the whole grid: those tables cost 5 x 27 box_selfind sweeps
        # over n_x n_y n_z entries -- 765 of 991 s of setup on the
        # 1.1M-cell RSFQ JTL, and a volume-proportional memory term --
        # while only the 27-box near pairs and the terminal pairs are
        # ever read (2026-08-26).
        p, dx = self.t_l, self.dx
        self.kf = {False: (p, dx, p), True: (p, dx, 0.0)}
        self.kt = {(False, False): (p, p, 0.0), (True, True): (p, p, 0.0),
                   (False, True): (p, p, p), (True, False): (p, p, -p)}

    def _kernel(self, params, k, m0, m1):
        p, q, s0 = params
        return tm.unequal_kernel_at(self.l, self.orient, k, m0, m1,
                                    p, q, s0)

    def self_block(self):
        """Dense ``Lp`` between terminal filaments (n x n, henries)."""
        L = np.zeros((self.n, self.n))
        a, b = np.meshgrid(np.arange(self.n), np.arange(self.n),
                           indexing='ij')
        a, b = a.ravel(), b.ravel()
        k = (self.cax[b] - self.cax[a]).astype(np.int64)
        m0 = np.abs(self.t0[a] - self.t0[b]).astype(np.int64)
        m1 = np.abs(self.t1[a] - self.t1[b]).astype(np.int64)
        hi = np.asarray(self.hi, dtype=bool)
        for key, params in self.kt.items():
            sel = np.flatnonzero((hi[a] == key[0]) & (hi[b] == key[1]))
            if sel.size:
                L[a[sel], b[sel]] = self._kernel(params, k[sel], m0[sel],
                                                 m1[sel])
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
        jax = np.atleast_1d(np.asarray(jax, dtype=np.int64))
        u0 = np.atleast_1d(np.asarray(u0, dtype=np.int64))
        u1 = np.atleast_1d(np.asarray(u1, dtype=np.int64))
        return self._kernel(self.kf[bool(self.hi[k])],
                            jax - int(self.cax[k]),
                            np.abs(int(self.t0[k]) - u0),
                            np.abs(int(self.t1[k]) - u1))

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


MU0 = 4e-7*np.pi


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


def london_rate(model):
    """``1/lambda`` for a uniform London model, else None.

    The mode palette needs the conductor's Helmholtz decay rate. For a
    superconductor that is ``1/lambda`` -- real, and frequency
    independent, which is why the London path never retunes.
    """
    lam = getattr(model, 'lambdaL', None)
    if lam is None or not getattr(model, 'superconductor', False):
        return None
    lam = np.asarray(lam, dtype=float)
    vals = np.unique(lam[lam > 0.0])
    if vals.size != 1:
        return None                  # mixed or absent: caller decides
    return 1.0/float(vals[0])


def material_response(model, freq):
    """``(p, z)`` -- the ONLY two things the mode engine asks of a material.

    ``p`` is the decay rate of the Helmholtz equation the interior
    current obeys, which sets the conduction palette's face/corner
    exponentials. ``z`` is the series impedance DENSITY in ohm*m, which
    sets the sub-bar impedance in ``Ru``.

        normal conductor   grad^2 J = j w mu sigma J   p = (1+j)/delta
                                                       z = 1/sigma
        London supercond.  grad^2 J = J/lambda^2       p = 1/lambda
                                                       z = j w mu lambda^2

    Two slots, one material law each. Expressing it this way is what
    keeps the engine below free of any superconductor branch: the mode
    shapes are ``conduction_weights(p=p)`` and the sub-bar impedance is
    ``z*k/dx``, for BOTH -- and ``1/sigma * k/dx`` is exactly the old
    ``k/(sigma*dx)``, so the normal path is unchanged arithmetic.

    ``p`` is frequency dependent for a normal conductor and not for a
    superconductor; ``z`` is the other way round. ``set_frequency``
    therefore just recomputes both and rebuilds what moved, with no
    special case for either.
    """
    lam_inv = london_rate(model)
    if lam_inv is not None:
        if not freq or float(freq) <= 0.0:
            raise ValueError(
                "a London material's impedance density is "
                "j w mu lambda^2 and needs freq > 0 (got %r)" % (freq,))
        w = 2.0*np.pi*float(freq)
        return lam_inv, 1j*w*MU0/lam_inv**2
    sigma = model.uniform_sigma()
    # z is frequency independent for a normal conductor, so it is
    # available even where the RATE is not -- which is what the
    # 'diff'/'linear' palettes need: they want no shapes, but Ru still
    # wants the material.
    p = (None if not freq or float(freq) <= 0.0
         else (1.0 + 1.0j)/skin_depth(sigma, freq))
    return p, 1.0/sigma


def conduction_weights(kk, dx, delta, p=None):
    """Net-zero CONDUCTION-MODE weights on the ``kk[0] x kk[1]`` sub-bar grid.

    ``dx`` is the physical cross-section: a scalar (square cell) or the
    ``(d0, d1)`` pair of transverse pitches (anisotropic cells,
    2026-09-01). The shapes are exponentials in PHYSICAL distance from
    each face, so a rectangular cross-section only changes where the
    faces sit -- the palette, the parity split and the pruning are
    unchanged.

    Daniel, Sangiovanni-Vincentelli & White (EPEP 2000): inside a good
    conductor the current solves a Helmholtz equation whose solutions
    are exponentials anchored to the cross-section's faces and corners
    -- face decay rate ``p = (1+j)/delta``, corner rate ``(1+j)/
    (delta*sqrt(2))``.

    ``p`` OVERRIDES that rate, which is what lets the same palette
    serve a LONDON SUPERCONDUCTOR. The governing equation is the same
    Helmholtz form with a different constant -- a normal conductor has
    ``grad^2 J = j w mu sigma J`` (so ``p = (1+j)/delta``), a London
    superconductor ``grad^2 J = J/lambda^2`` (so ``p = 1/lambda``,
    REAL and frequency independent) -- and the corner rate follows for
    any ``p`` by the same algebra, since ``exp(-a(x+y))`` has
    ``grad^2 = 2a^2`` and therefore ``a = p/sqrt(2)``.

    Passing ``delta = lambda`` instead of the real rate does NOT work
    and is the trap worth naming: it gives ``p = (1+j)/lambda``, whose
    real and imaginary parts are ``exp(-x/lam)cos(x/lam)`` and
    ``exp(-x/lam)sin(x/lam)``. Their span does not contain the London
    profile ``cosh((x-t/2)/lambda)`` -- measured residual 1.2e-2 with
    four columns, against 2.2e-15 with the two real-rate columns. Those are the shapes the piecewise-constant basis
    approximates badly (measured in ``studies/modebasis2d.py``: at
    matched mode count the conduction basis misses 6.6x less of the
    skin-effect correction than 'diff', the recorded 201%% failure's
    root cause). Since 2026-08-20 the four corner exponentials enter
    INDIVIDUALLY (not as one symmetric sum): the corner-parity split
    measured +6 delivered points at dx/delta 3-6, and the smooth
    family was shown COMPLETE for straight sections
    (``studies/xsection_tabulated.py`` -- the engine's residual is
    sub-bar resolution and rc truncation, not span).

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
    d0, d1 = ((float(dx), float(dx)) if np.isscalar(dx)
              else (float(dx[0]), float(dx[1])))
    u = (np.arange(k0) + 0.5)/k0
    v = (np.arange(k1) + 0.5)/k1
    U, V = np.meshgrid(u, v, indexing='ij')       # matches Split.boxes
    x = U.ravel()*d0                              # distance from the low
    y = V.ravel()*d1                              # face, per split axis
    p = (1.0 + 1.0j)/delta if p is None else complex(p)
    ex0, ex1 = np.exp(-p*x), np.exp(-p*(d0 - x))
    ey0, ey1 = np.exp(-p*y), np.exp(-p*(d1 - y))
    pc = p/np.sqrt(2.0)
    # INDIVIDUAL faces and corners (2026-08-20, studies/
    # palette_ablation.py + xsection_tabulated.py). The original
    # palette carried the four corner exponentials only as their fully
    # SYMMETRIC sum -- but four corner anchors span four parity
    # combinations (symmetric, x-odd, y-odd, xy-odd), and a cell at a
    # conductor corner cannot crowd toward its one exposed corner
    # without them: the missing three columns measured +6 delivered
    # points at dx/delta 3-6 (83->89 / 75->82 / 69->75 % at 2/3/4
    # cells across), neutral at dx/delta ~ 2, and behave as a pure
    # basis change under every knob (rc/k/frequency deltas match the
    # old palette's to 0.1-0.3 points). Faces enter individually too
    # (same span as the old symmetric+antisymmetric pairs). km grows
    # to <= 16, so the mode-FFT spectra cost 4x -- measured matvec
    # counts unchanged with mode_precond.
    shapes = [ex0, ex1, ey0, ey1]
    if min(k0, k1) > 1:
        # corner exponentials only when the cross-section is genuinely
        # 2-D. Under a 1-D split (the thin-film palette, kk = (1, kz))
        # a "corner" shape degenerates into a face exponential at the
        # rate p/sqrt(2) -- not new physics, a same-shaped column at a
        # slightly different rate. MEASURED (2026-09-02, XNOR symbol
        # probe): keeping them makes the translation-invariant mode
        # operator NEARLY SINGULAR -- condition 2.7e12, worst
        # directions in-plane-smooth / z-oscillating combinations of
        # the 1/lambda and 1/(lambda sqrt 2) families a factor 1000
        # below the spectral floor -- which is the ~2000-matvec
        # film-palette stall no block preconditioner could fix
        # (near-null operator directions are not preconditionable).
        # The per-cell QR prune cannot see it: the columns are
        # independent WITHIN a cell and degenerate only ACROSS cells
        # at specific k.
        shapes += [np.exp(-pc*(x + y)),
                   np.exp(-pc*(x + (d1 - y))),
                   np.exp(-pc*((d0 - x) + y)),
                   np.exp(-pc*((d0 - x) + (d1 - y)))]
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
                         "subdivide=False instead" % (max(d0, d1)/delta,))
    W = np.stack(cols, axis=1)
    # Prune near-dependent columns by pivoted QR, KEEPING original
    # (physical) columns rather than mixing them, so the result is
    # deterministic and readable. Dropped columns are within ~1e-7 of
    # the span of the kept ones, so the span is preserved to that level.
    _, R, piv = qr(W, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    # `qr(mode='economic')` returns R with min(m, n) ROWS, so diag(R) is
    # min(m, n) long while piv is n long. Slicing piv to d's length is a
    # no-op whenever m >= n (the only case reachable in production: the
    # conduction-auto k rule floors k at 7, i.e. 49 sub-bars against at
    # most 16 columns) and is what makes the k <= 3 case work at all --
    # with m rows at most m columns can be independent, and pivoted QR
    # has already ordered them by decreasing importance, so the
    # candidates are exactly the first min(m, n) pivots.
    W = W[:, np.sort(piv[:d.size][d > 1e-7*d[0]])]
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
                 skin_freq=None, boundary_only=True):
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
        self.tr = others
        self.kk = kk
        self.k = kk[0]*kk[1]
        # Per-axis pitch (2026-09-01): the filament's AXIAL step and
        # the two TRANSVERSE pitches of its cross-section. On a cubic
        # model all three coincide and every formula below reduces to
        # the old scalar-dx one; self.dx keeps the historical name for
        # the axial pitch (nothing outside this class reads it).
        d3 = np.asarray(model.d, dtype=float)
        self.d3 = d3
        self.dax = float(d3[self.axis])
        self.dt = (float(d3[others[0]]), float(d3[others[1]]))
        self.dx = self.dax
        # The material enters through material_response() alone -- see
        # there for why that leaves no superconductor branch below. A
        # lossless London model legitimately has sigma = 0 and no skin
        # depth, so sigma is kept only for reporting.
        self._model = model
        self._london = london_rate(model) is not None
        try:
            self.sigma = model.uniform_sigma()
        except ValueError:
            if not self._london:
                raise
            self.sigma = 0.0
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
        # DEFAULT TRUE since 2026-08-23. Modes belong where the physics
        # is: a cell with metal on all sides has no nearby surface for
        # current to crowd against, so face-anchored exponentials there
        # are unphysical, and because the modes are NET-ZERO their
        # leading far field is a DIPOLE. Those spurious interior dipoles
        # are what the mode-coupling truncation then mishandles -- it
        # keeps part of a shell whose tails only cancel when enough of
        # the shell is included, leaving an uncancelled field that
        # OVER-CONCENTRATES the current and overstates loss.
        #
        # MEASURED on the 20-cell-wide numex1 bar, error against a
        # converged refinement ladder:
        #     boundary_only   10 GHz   25 GHz   100 GHz    cost
        #     False            -0.2%    +4.6%    +70.9%    43.5 s
        #     True             -2.1%    -5.4%     -7.7%    36.9 s
        # True is CHEAPER and errs LOW -- the safe direction, the same
        # direction as the plain basis -- where False overstates loss by
        # 70% with no signal it has left its range.
        #
        # THIS ONLY ALIGNS THE DIRECT API WITH WHAT PRODUCTION ALREADY
        # DID: sppeec_input has defaulted boundary_only to True since the
        # 2026-08-17 referee measurement (modes-everywhere overshoots the
        # physical skin limit ~2.8x on a wide section). Callers
        # constructing EquiTerminalSolver directly -- every study in
        # studies/ -- silently got the unsafe setting instead.
        self._bnd = np.ones(self.sel.size, dtype=bool)
        if self.boundary_only:
            occ = np.asarray(model.struc()).astype(bool)
            cc = np.asarray(fil_cell)[self.sel]
            s0, s1 = self.tr
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
        self.split = Split.transverse(self.axis, kk, d3)
        self.whole = Split(self.axis, (1, 1, 1), d3)
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
        if self.mode_basis == 'conduction' and (self.skin_freq is None
                                                or self.skin_freq <= 0):
            raise ValueError(
                "mode_basis='conduction' needs skin_freq > 0: the mode "
                "shapes are exponentials in the material's Helmholtz "
                "rate (got %r)" % (skin_freq,))
        # z is wanted by EVERY palette (it is what Ru is made of); the
        # rate p only by 'conduction'. Resolving both here keeps the
        # material in one place regardless of which shapes are chosen.
        self._p, self._z = material_response(model, self.skin_freq)
        if self.mode_basis == 'conduction':
            W = conduction_weights(kk, self.dt, None, p=self._p)
        elif self.mode_basis == 'linear':
            k0, k1 = kk
            u = (np.arange(k0) + 0.5)/k0 - 0.5
            v = (np.arange(k1) + 0.5)/k1 - 0.5
            U, V = np.meshgrid(u, v, indexing='ij')   # matches Split.boxes
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
        self.use_fft = bool(use_fft)
        self._term = term
        self._rc_uu, self._rc_cross = int(rc_uu), int(rc_cross)
        # Frequency-independent geometry caches, filled on first build:
        # sub-bar coupling tables keyed by the separation set, and the
        # KDTree pair lists. They are what make set_frequency cheap --
        # the kernel evaluations dominate assembly and never change.
        self.tables = PairTables()
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
        # THE SUB-BAR SERIES IMPEDANCE. For a normal conductor this is
        # a resistance, k/(sigma*dx), real and frequency independent.
        # For a LONDON superconductor the per-cell impedance density is
        # j w mu lambda^2, not 1/sigma, so the same projection carries
        # a reactance: Ru becomes complex and proportional to w.
        #
        # That w-proportionality is the physics, not a nuisance: the
        # mode equation is (Ru + jw*Zuu) u = drive, so with Ru itself
        # linear in w the frequency cancels and the current profile is
        # FREQUENCY INDEPENDENT -- which is exactly what distinguishes
        # London screening from skin effect, and what
        # validate_superconductor PART C measures as flat L(f).
        r_sub = self._sub_impedance()
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
        p, self._z = material_response(self._model, freq)
        # Whether the SHAPES moved is a property of the material, not a
        # branch to write: a normal conductor's rate is (1+j)/delta and
        # does move, a London rate is 1/lambda and does not. z moves
        # either way (it carries j w mu lambda^2 for the superconductor),
        # so the blocks are always re-assembled; only W is conditional,
        # and only because rebuilding it can re-prune and change km.
        moved = (p != self._p)
        self._p = p
        if moved:
            self._set_W(conduction_weights(self.kk, self.dt, None, p=p))
        self._assemble()
        return bool(moved)

    def mode_precond(self, jw):
        """Per-filament block-Jacobi inverse of ``Ru + jw*Zuu_self``.

        The mesh preconditioner's mode block is the identity, so the
        mode equations run unpreconditioned; at high omega with a rich
        basis |jw Zuu| dwarfs Ru and the Krylov stalls (measured: the
        engine-only ladder rungs at 1e10 hit the 311-matvec cap; the
        same failure class as the C.2 subpixel modes before their
        block-Jacobi shipped). Every filament carries IDENTICAL
        weights, so the self block is ONE km x km matrix -- Ru's
        per-filament block plus the folded same-filament sub-bar
        mutual -- inverted once and Kronecker'd over the mode-carrying
        filaments."""
        if self.nmode == 0:
            return None
        from terminal import box_mutual_matrix
        lo, hi = self.split.boxes(self.cells[:1])
        Ls = box_mutual_matrix(lo, hi, self.axis)
        r_sub = self._sub_impedance()
        A = r_sub*(self.W.T @ self.W) + jw*(self.W.T @ (Ls @ self.W))
        Ainv = np.linalg.inv(A)
        nf = int(self._bnd.sum())
        return sp.kron(sp.identity(nf, format='csr'),
                       sp.csr_matrix(Ainv), format='csr')

    def _sub_impedance(self):
        """Series impedance of ONE sub-bar, projected later onto W.

        ONE formula for both materials: ``z * k / dx``, where z is the
        series impedance density from material_response(). For a normal
        conductor z = 1/sigma and this is exactly the old
        ``k/(sigma*dx)``; for a London superconductor z = j w mu
        lambda^2 and the same expression gives a reactance proportional
        to w. Both callers -- _assemble and mode_precond -- must use the
        SAME value or the preconditioner stops approximating the
        operator it preconditions. Per-axis ``z*l/a_sub`` -- on a cubic
        model exactly the old ``z*k/dx``.
        """
        return self._z*self.dax/self.split.area

    def _neighbour_pairs(self, radius, other=None):
        """Filament index pairs within ``radius`` cells (inf-norm),
        cached: the cells never move, so a re-assembly (set_frequency)
        must not pay the KDTree again."""
        ck = (float(radius), other is None)
        if ck not in self._pairs:
            self._pairs[ck] = neighbour_pairs(self.cells, radius, other)
        return self._pairs[ck]

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
        D, inv = unique_separations(np.concatenate([Du, Dc]))
        iu, ic = inv[:len(Du)], inv[len(Du):]
        Bu, Bc = self._mode_tables(D)
        self.ntable = int(D.shape[0])

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
        M = self.tables(self.split, self.split, D)
        Mc = self.tables(self.split, self.whole, D)[:, :, 0]
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
        # Delta = 0 at the origin, negative offsets wrapped to the end.
        #
        # BUILT ONE KERNEL AT A TIME. The batched form materialised the
        # whole real table Ku (km^2 x pad) and then its whole spectrum
        # Fu (km^2 x pad, complex) with BOTH alive across the transform.
        # This transform is whole-bounding-box, so on a large sparse
        # layout that is the dominant allocation in the entire solver:
        # measured on the RSFQ XNOR at 100 nm cubic (box 621x721x34,
        # pad 630x729x42 = 19.3M, km = 8) it is 9.9 GB of table plus
        # 19.8 GB of spectrum, and the run was OOM-killed at 35.4 GB.
        # One (m, n2) at a time keeps a single pad-sized real slab
        # (154 MB here) instead, for the same spectra: the transform is
        # independent per kernel pair, so this is a loop order change,
        # not an approximation.
        # STORED SINGLE (fp32 campaign, phase 5). After chunking, the
        # spectra ARE the allocation: km^2 x pad complex, 19.8 GB on
        # the XNOR against 154 MB for the working slab. They are
        # spectra of mutual inductances -- smooth, O(1e-12..1e-7) in
        # SI, nowhere near float32's 1e-38 floor, so unlike the leaf
        # gather tables they need no scale factored out. Built in fp64
        # by the transform and stored single; every consumer in
        # apply_fft multiplies against complex128 data and so
        # accumulates in double through numpy promotion, exactly as
        # the wire coupling caches do. SPPEEC_MODE_FP64=1 restores
        # double storage for A/B.
        dt = (np.complex128 if os.environ.get('SPPEEC_MODE_FP64') == '1'
              else np.complex64)
        self.Fu = np.empty((km, km) + self.pad, dtype=dt)
        self.Fc = np.empty((km,) + self.pad, dtype=dt)
        wrap = tuple(np.mod(D[:, a], self.pad[a]) for a in range(3))
        slab = np.zeros(self.pad)
        for m in range(km):
            slab[...] = 0.0
            slab[wrap] = Bc[:, m]
            self.Fc[m] = sfft.fftn(slab)
            for n2 in range(km):
                slab[...] = 0.0
                slab[wrap] = Bu[:, m, n2]
                self.Fu[m, n2] = sfft.fftn(slab)
        del slab
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
        D, inv = unique_separations(tcell[tb] - self.cells[fa])
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
        Mct = self._raw_terminal_table(D, t_l)
        return np.einsum('pm,dps->dms', self.W, Mct)

    def _raw_terminal_table(self, D, t_l):
        """Raw sub-bar <-> terminal-bar table ``(nD, k, 2 signs)``,
        geometry only, from the shared separation tables."""
        return np.stack([self.tables(self.split,
                                     self.split.terminal(t_l, s), D)[:, :, 0]
                         for s in (-1, +1)], axis=-1)


class SubpixelModes(Redistribution):
    """Solved net-zero SURFACE-ANCHORED modes on subpixel (fill) models.

    The coarse :class:`Redistribution` engine requires identical mode
    weights on every filament -- that is what makes its tables Toeplitz
    and its apply an FFT -- and a uniform conductivity. A subpixel model
    breaks both: partial cells carry sigma_eff = sigma*fill, and the
    physically right mode shapes are exponentials in the distance to the
    TRUE conductor surface (the resolved circle), which differs cell by
    cell. This subclass keeps the parent's augmentation contract (same
    Ru/Zuu/Zcross/Zt blocks, net-zero columns, zero incidence) and swaps
    the construction:

      * per-cell weights ``Wf[f]`` from the surface exponential
        ``exp(-(1+j) d/delta)`` at the sub-prism centroids (d = signed
        distance to the cylinder surface, from the resolved geometry the
        voxelizer stored), plus its two azimuthal (proximity) partners
        -- the circle's analog of the face/corner conduction shapes;
      * sub-prisms outside the metal (fill ~ 0) carry ZERO weight, and
        the net-zero constraint holds over the SUPPORT;
      * sub-bar resistance is fill-weighted, ``r_p = l/(sigma a fill_p)``
        -- their parallel combination reproduces the aggregate's
        sigma_eff exactly, and the resistive aggregate<->mode coupling
        vanishes identically (r_p * fill share is constant over the
        support, and the mode is net-zero);
      * the AGGREGATE side of the cross block uses the fill shares
        (the Galerkin-consistent clipped current), not the full box;
      * assembly folds the parent's cached RAW separation tables with
        the per-pair weights -- sparse/truncated only. ``use_fft`` is
        forced off: per-cell weights are exactly what the FFT fold
        cannot represent.

    Mode placement is a geometric boundary ring: a cell carries modes
    only if some supported sub-prism lies within ``bnd_reach`` cells of
    the surface. Interior modes measured actively harmful in the 2-D
    Galerkin study and unphysical at scale in 3-D (docket 2026-08-17);
    the ring rule is the subpixel analog of ``boundary_only``.
    """

    def __init__(self, model, M, axis, fil_axis, fil_cell, k=7,
                 term=None, rc_uu=3, rc_cross=4, csr_max_gb=2.0,
                 skin_freq=None, bnd_reach=1.0):
        spx = model.subpixel
        self.axis = int(axis)
        if int(spx['axis']) != self.axis:
            raise NotImplementedError(
                "subpixel modes: the terminal axis (%d) differs from "
                "the cylinder axis (%d) -- transverse mode families are "
                "future work" % (self.axis, int(spx['axis'])))
        sigs = {float(g[3]) for g in spx['geom'].values()}
        if len(sigs) != 1:
            raise NotImplementedError(
                "subpixel modes: cylinders with different sigma in one "
                "model (%d values)" % len(sigs))
        others = [c for c in range(3) if c != self.axis]
        kk = (int(k), int(k)) if np.isscalar(k) else tuple(int(v) for v in k)
        if min(kk) < 1 or max(kk) < 2:
            raise ValueError("k must give at least two sub-filaments")
        self.tr = others
        self.kk = kk
        self.k = kk[0]*kk[1]
        self.dx = model.dx        # cubic-only: model.dx raises otherwise
        self.d3 = np.asarray(model.d, dtype=float)
        self.dax = self.dx
        self.dt = (self.dx, self.dx)
        self.sigma = sigs.pop()
        self.sel = np.flatnonzero(fil_axis == self.axis)
        self.mode_basis = 'conduction'
        self.boundary_only = True
        self.nfil = self.sel.size
        self.cells = fil_cell[self.sel]
        self.split = Split.transverse(self.axis, kk, self.d3)
        self.whole = Split(self.axis, (1, 1, 1), self.d3)
        if skin_freq is None or float(skin_freq) <= 0:
            raise ValueError("SubpixelModes needs skin_freq > 0: the "
                             "shapes are exponentials in the skin depth")
        self.skin_freq = float(skin_freq)
        self.csr_max_gb = float(csr_max_gb)
        self.use_fft = False
        self._term = term
        self._rc_uu, self._rc_cross = int(rc_uu), int(rc_cross)
        self.tables = PairTables()
        self._pairs = {}
        # -- per-transverse-cell geometry, shared down the extrusion --
        # sub-prism fills at the ENGINE subdivision (resampled from the
        # resolved circle, independent of the voxelizer's k), signed
        # distance to the surface and azimuth at the centroids.
        t1, t2 = others
        keys = np.stack([self.cells[:, t1], self.cells[:, t2]], axis=1)
        self._tkey = [(int(a), int(b)) for a, b in keys]
        self._percell = {}
        k0, k1 = kk
        h0, h1 = self.dx/k0, self.dx/k1
        ns = 8                                    # samples per sub-prism axis
        u1 = (np.arange(k0*ns) + 0.5)*(h0/ns)
        u2 = (np.arange(k1*ns) + 0.5)*(h1/ns)
        cu = (np.arange(k0) + 0.5)*h0             # centroids
        cv = (np.arange(k1) + 0.5)*h1
        for key in set(self._tkey):
            g = spx['geom'].get(key)
            if g is None:
                # a conductor cell that is not part of any cylinder
                # (mixed model): full box, no surface -> no modes, but
                # it still couples as an aggregate
                self._percell[key] = None
                continue
            c1, c2, R, _ = g
            x0 = key[0]*self.dx
            y0 = key[1]*self.dx
            ins = ((x0 + u1[:, None] - c1)**2
                   + (y0 + u2[None, :] - c2)**2) <= R*R
            fill = ins.reshape(k0, ns, k1, ns).mean(axis=(1, 3)).ravel()
            XC, YC = np.meshgrid(x0 + cu, y0 + cv, indexing='ij')
            rho = np.hypot(XC.ravel() - c1, YC.ravel() - c2)
            self._percell[key] = dict(
                fill=fill, d=R - rho,             # signed: >0 inside
                phi=np.arctan2(YC.ravel() - c2, XC.ravel() - c1))
        # aggregate-side weights: fill shares (clipped current)
        G = np.full((self.nfil, self.k), 1.0/self.k)
        for f, key in enumerate(self._tkey):
            pc = self._percell[key]
            if pc is not None:
                tot = pc['fill'].sum()
                G[f] = pc['fill']/tot if tot > 0 else 0.0
        self.G = G
        # boundary ring: modes only where the surface passes nearby
        self._bnd = np.zeros(self.nfil, dtype=bool)
        reach = float(bnd_reach)*self.dx
        for f, key in enumerate(self._tkey):
            pc = self._percell[key]
            if pc is not None:
                sup = pc['fill'] > 1e-3
                self._bnd[f] = bool(np.any(np.abs(pc['d'][sup]) <= reach))
        self._set_Wf(*self._make_W(skin_depth(self.sigma, self.skin_freq)))
        self._assemble()

    # -- weights -------------------------------------------------------

    KM = 6           # 3 complex shapes (skin, 2x proximity) x (re, im)

    def _make_W(self, delta):
        """Per-cell weight stack ``(nfil, k, KM)`` and its column mask.

        Per unique transverse cell: the surface exponential and its two
        azimuthal partners, restricted to the supported (fill > 0)
        sub-prisms, mean-subtracted over the support (net-zero), then
        normalised and pruned by pivoted QR -- a pruned column is zeroed
        and MASKED rather than dropped, so ``km`` stays global and the
        parent's mode_mask machinery does the bookkeeping.
        """
        from scipy.linalg import qr
        Wu, cmask_u = {}, {}
        for key, pc in self._percell.items():
            if pc is None:
                Wu[key] = np.zeros((self.k, self.KM))
                cmask_u[key] = np.zeros(self.KM, dtype=bool)
                continue
            sup = pc['fill'] > 1e-3
            # weights are sub-bar CURRENTS = density shape x metal
            # area share. The fill factor is load-bearing twice over:
            # physically (a sliver carries a sliver's current) and
            # numerically (without it, an O(1) mode current through a
            # 1/fill resistance makes Ru's conditioning arbitrarily
            # bad -- the doctrine's sliver trap, measured here as a
            # stalled Krylov solve)
            c = (np.exp(-(1.0 + 1.0j)*np.maximum(pc['d'], 0.0)/delta)
                 * pc['fill'])
            shapes = [c, c*np.cos(pc['phi']), c*np.sin(pc['phi'])]
            W = np.zeros((self.k, self.KM))
            keep = np.zeros(self.KM, dtype=bool)
            cols = []
            for sh in shapes:
                for part in (sh.real, sh.imag):
                    w = np.zeros(self.k)
                    w[sup] = part[sup] - part[sup].mean()
                    cols.append(w)
            Wc = np.stack(cols, axis=1)
            nrm = np.linalg.norm(Wc, axis=0)
            ok = nrm > 1e-12
            Wc[:, ok] /= nrm[ok]
            if ok.any():
                _, Rq, piv = qr(Wc[:, ok], mode='economic', pivoting=True)
                dg = np.abs(np.diag(Rq))
                kept = np.flatnonzero(ok)[np.sort(piv[dg > 1e-7*dg[0]])]
                keep[kept] = True
                W[:, kept] = Wc[:, kept]
                # exact net-zero over the support after pruning
                col = W[:, kept]
                col[sup] -= col[sup].mean(axis=0, keepdims=True)
                W[:, kept] = col
            Wu[key], cmask_u[key] = W, keep
        Wf = np.zeros((self.nfil, self.k, self.KM))
        cmask = np.zeros((self.nfil, self.KM), dtype=bool)
        for f, key in enumerate(self._tkey):
            Wf[f] = Wu[key]
            cmask[f] = cmask_u[key]
        return Wf, cmask

    def _set_Wf(self, Wf, cmask):
        net = np.abs(Wf.sum(axis=1)).max()
        if net > 1e-9:
            raise RuntimeError("subpixel mode weights are not net-zero "
                               "(%.3e)" % net)
        self.Wf = Wf
        self.km = self.KM
        self.nmode_full = self.nfil*self.km
        self.mode_mask = (cmask & self._bnd[:, None]).ravel()
        self.nmode = int(self.mode_mask.sum())

    # -- assembly ------------------------------------------------------

    def _assemble(self):
        self.Zuu = self.Zcross = None
        self.nnz = (0, 0)
        self._build_truncated(self._rc_uu, self._rc_cross)
        # per-cell fill-weighted mode resistance, block diagonal
        r = np.zeros((self.nfil, self.k))
        base = self.k/(self.sigma*self.dx)        # full-fill sub-bar
        for f, key in enumerate(self._tkey):
            pc = self._percell[key]
            if pc is None:
                r[f] = base
            else:
                sup = pc['fill'] > 1e-3
                r[f, sup] = base/pc['fill'][sup]
        blocks = np.einsum('fpm,fp,fpr->fmr', self.Wf, r, self.Wf)
        self.Ru = sp.block_diag([blocks[f] for f in range(self.nfil)],
                                format='csr')
        self.Zt = None
        if self._term is not None and self._term.axis == self.axis:
            self._build_terminal(self._term, self._rc_cross)
        if self.nmode != self.nmode_full:
            mk = self.mode_mask
            self.Ru = self.Ru[mk][:, mk]
            if self.Zuu is not None:
                self.Zuu = self.Zuu[mk][:, mk]
            if self.Zcross is not None:
                self.Zcross = self.Zcross[mk]
            if self.Zt is not None:
                self.Zt = self.Zt[mk]

    def _build_truncated(self, rc_uu, rc_cross):
        km = self.km
        fa_u, fb_u = self._neighbour_pairs(rc_uu)
        fa_c, fb_c = self._neighbour_pairs(rc_cross)
        self._check_csr_size(fa_u.size, fa_c.size, rc_uu, rc_cross)
        Du = self.cells[fb_u] - self.cells[fa_u]
        Dc = self.cells[fb_c] - self.cells[fa_c]
        D, inv = unique_separations(np.concatenate([Du, Dc]))
        iu, ic = inv[:len(Du)], inv[len(Du):]
        M = self.tables(self.split, self.split, D)
        self.ntable = int(D.shape[0])
        Wf, G = self.Wf, self.G
        mr = np.arange(km)
        # mode <-> mode, folded per pair (chunked: the einsum temporary
        # is npair*k*k)
        red = np.empty((fa_u.size, km, km))
        step = max(1, 20_000_000 // (self.k*self.k))
        for a in range(0, fa_u.size, step):
            sl = np.s_[a:a + step]
            red[sl] = np.einsum('apm,apq,aqr->amr',
                                Wf[fa_u[sl]], M[iu[sl]], Wf[fb_u[sl]])
        rows = np.broadcast_to(fa_u[:, None, None]*km + mr[None, :, None],
                               red.shape).ravel()
        cols = np.broadcast_to(fb_u[:, None, None]*km + mr[None, None, :],
                               red.shape).ravel()
        self.Zuu = sp.csr_matrix((red.ravel(), (rows, cols)),
                                 shape=(self.nmode_full, self.nmode_full))
        # mode <-> aggregate, the aggregate carrying its fill shares
        red = np.empty((fa_c.size, km))
        for a in range(0, fa_c.size, step):
            sl = np.s_[a:a + step]
            red[sl] = np.einsum('apm,apq,aq->am',
                                Wf[fa_c[sl]], M[ic[sl]], G[fb_c[sl]])
        rows = (fa_c[:, None]*km + mr[None, :]).ravel()
        cols = np.broadcast_to(fb_c[:, None], red.shape).ravel()
        self.Zcross = sp.csr_matrix((red.ravel(), (rows, cols)),
                                    shape=(self.nmode_full, self.nfil))
        self.nnz = (int(self.Zuu.nnz), int(self.Zcross.nnz))

    def _build_terminal(self, term, radius):
        km = self.km
        tcell = np.array([f[0] for f in term.faces], dtype=np.int64)
        fa, tb = self._neighbour_pairs(radius, other=tcell.astype(float))
        if fa.size == 0:
            self.Zt = sp.csr_matrix((self.nmode_full, term.n))
            self.ntable_t = 0
            return
        D, inv = unique_separations(tcell[tb] - self.cells[fa])
        Mct = self._raw_terminal_table(D, term.t_l)
        self.ntable_t = int(D.shape[0])
        si = (term.sign[tb] > 0).astype(np.int64)
        red = np.einsum('apm,aps->ams', self.Wf[fa], Mct[inv])
        red = red[np.arange(fa.size), :, si]
        rows = (fa[:, None]*km + np.arange(km)[None, :]).ravel()
        cols = np.broadcast_to(tb[:, None], red.shape).ravel()
        self.Zt = sp.csr_matrix((red.ravel(), (rows, cols)),
                                shape=(self.nmode_full, term.n))

    def set_frequency(self, freq):
        freq = float(freq)
        if freq <= 0 or freq == self.skin_freq:
            return False
        self.skin_freq = freq
        self._set_Wf(*self._make_W(skin_depth(self.sigma, freq)))
        self._assemble()
        return True

    def build_fft(self, rc_uu, rc_cross):
        raise NotImplementedError(
            "subpixel modes have per-cell weights -- the mode blocks "
            "are not translation invariant and cannot be applied as "
            "convolutions; the sparse truncated path is the only one")

    def mode_precond(self, jw):
        """Block-Jacobi inverse of ``Ru + jw*Zuu`` per mode-carrying
        cell, as a sparse block-diagonal matrix in the MASKED mode
        numbering.

        The mesh preconditioner is the frequency-independent Cholesky
        of the cycle Gram, whose mode block is the identity -- mode
        equations are effectively unpreconditioned. That is survivable
        for the coarse engine's moderate mode scales, but the subpixel
        blocks span sliver-to-full fill ratios AND deep-skin solves run
        at omega ~ 1e12, where |jw*Zuu| dwarfs the identity: measured
        2078 matvecs WITHOUT convergence at dx/delta = 6. The per-cell
        inverse restores the row scales; rebuild per solve frequency
        (a few thousand <= km x km inversions, milliseconds)."""
        if self.Zuu is None or self.nmode == 0:
            return None
        A = (self.Ru + jw*self.Zuu).tocsr()
        counts = self.mode_mask.reshape(self.nfil, self.km).sum(axis=1)
        data, rows, cols = [], [], []
        pos = 0
        for c in counts:
            if c == 0:
                continue
            idx = np.arange(pos, pos + int(c))
            blk = A[idx][:, idx].toarray()
            inv = np.linalg.inv(blk)
            rows.append(np.repeat(idx, len(idx)))
            cols.append(np.tile(idx, len(idx)))
            data.append(inv.ravel())
            pos += int(c)
        return sp.csr_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nmode, self.nmode))


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
                 boundary_only=True, basis='selected', amg_cycles=4,
                 amg_cycle_type='V', amg_smoother=None,
                 amg_strength=None, gram_solver='amg',
                 corner_modes=False, nsolves=1):
        self.M = M
        self.nsolves = int(nsolves)
        self.basis_fallback = None
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
        if self.basis not in ('auto', 'selected', 'overcomplete'):
            raise ValueError("basis must be 'auto', 'selected' or "
                             "'overcomplete', got %r" % (basis,))
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
        self.node_of_cell = CellIndex(self.ncell)
        self.fil_axis, self.fil_cell = filament_cells(M)
        _check_orientation(self.B, self.fil_axis, self.fil_cell,
                           self.node_of_cell)
        self.csel = self.term.parallel(self.fil_axis)
        self.Ltt = self.term.self_block()
        self.C = None
        self.coupler = None
        self.fmm_reason = None
        # subpixel stage B: stage A made the material law exact; this is
        # the geometric footprint the mutual tables still have wrong (a
        # half-filled cell presents a full-cell bar). None on whole-cell
        # models.
        with _spstatus.task('subpixel dL'):
            self.dL_near = partial_dL(model, M)
        with _spstatus.task('terminal coupler'):
            if fmm:
                try:
                    if M.numlevels < 2:
                        raise CouplerUnavailable(
                            "single-level tree: p2p already covers the "
                            "whole domain, so the dense block IS the "
                            "near block")
                    self.coupler = TerminalCoupler(self.term, M,
                                                   self.fil_axis,
                                                   self.fil_cell,
                                                   self.csel)
                    # far terminal<->terminal pairs now come from the
                    # ladder
                    self.Ltt = np.where(self.coupler.tnear, self.Ltt,
                                        0.0)
                except CouplerUnavailable as exc:
                    # Fall back to the dense block, which is always
                    # correct -- but RECORD why, so a silent 30x
                    # slowdown is diagnosable.
                    self.fmm_reason = str(exc)
            if self.coupler is None:
                # only now, and only if we must: this is the
                # O(n_t * N) block
                self.C, _ = self.term.coupling(self.fil_axis,
                                               self.fil_cell)
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
        # ANISOTROPIC cells are supported since 2026-09-01: the box
        # builders, sub-bar areas, conduction shapes and the k rule are
        # all per-axis now (gate: studies/london_film.py, the cubic
        # recovered-fraction curve reproduced from an anisotropic mesh
        # of the same physical film). The SubpixelModes engine is still
        # cubic-only -- it raises through model.dx on its own.
        if getattr(model, 'superconductor', False) and not skin_off:
            # The two objections this guard used to raise are both
            # answered for a UNIFORM London model, and only for one:
            #   * the mode SHAPES now take the Helmholtz rate directly
            #     (conduction_weights(p=1/lambda)), so the screening
            #     length sets the profile rather than a skin depth;
            #   * Ru now carries the sub-bar impedance j w mu lambda^2
            #     rather than a real k/(sigma dx).
            # The modes remain net-zero, so they redistribute current
            # inside a cell and add none: the cell mean still carries
            # the bulk kinetic term and there is no double count. What
            # is NOT answered is a model with several lambdas -- one
            # rate cannot serve them -- so that still raises.
            if london_rate(model) is None:
                raise NotImplementedError(
                    "skin-effect subdivision on a superconductor with "
                    "no single London depth: the mode palette is "
                    "exponentials at ONE rate 1/lambda, and a mixed- "
                    "or zero-lambda model has no such rate. Solve with "
                    "subdivide=False.")
            if mode_basis != 'conduction':
                raise NotImplementedError(
                    "skin-effect subdivision on a superconductor needs "
                    "mode_basis='conduction': that palette's shapes are "
                    "the exponentials the London Helmholtz equation "
                    "actually solves, at rate 1/lambda. 'diff' and "
                    "'linear' are generic net-zero shapes with no "
                    "London content and are not validated here; got "
                    "%r." % (mode_basis,))
        fref = skin_freq
        if fref is None:
            fref = float(np.max(model.freq)) if len(model.freq) else 0.0
        self.skin_freq = fref
        self.skin_k = 1
        spx = getattr(model, 'subpixel', None)
        # THIN-FILM PALETTE ([[block]] film = "x|y|z", 2026-09-01). The
        # declared normal is the stiff axis: in-plane current varies on
        # the lambda/delta scale THROUGH the film, and (measured,
        # studies/london_film.py Stage 0) recovering it needs BOTH fine
        # quadrature along the normal and wide coupling radii -- two
        # budgets the k x k cross-section palette cannot pay at once
        # (its tables cost rc^3 * k^4-ish; the last Stage-0 bench point
        # took 73 minutes). The film palette is the same engine spending
        # the same budget where the physics is: a 1-D split along the
        # normal, kk = (1, kz), under which conduction_weights reduces
        # BY PRUNING to the normal-axis face family (the in-plane
        # columns become constants and the corner columns degenerate) --
        # ~3 modes and kz sub-bars instead of up to 16 and kz^2. kz
        # takes the floor-7 conduction rule (Stage 0: k = 3 plateaus at
        # 77% recovered, a quadrature ceiling; a 1-D kz = 7..12 is
        # cheap). Applies when the port axis is IN-PLANE; a port along
        # the normal falls back to the standard palette. Amplitudes are
        # always solved -- this is a budget hint, not a sheet model.
        fnorm = getattr(model, 'film_normal', None)
        film = (fnorm is not None and int(fnorm) != self.term.axis
                and mode_basis == 'conduction')
        if film:
            fnorm = int(fnorm)
        if subdivide is True or subdivide == 'auto':
            lp = london_rate(model)
            if lp is not None:
                # A London conductor's screening length is lambda, and
                # it does not depend on frequency, so the skin-depth
                # rule does not apply: ask the same question of the
                # right length. recommend_subdivision's rule is
                # k >= 2*dx/length, and it is fed a sigma, so invert
                # lambda into the sigma that would give delta = lambda
                # at this frequency rather than duplicate the rule.
                lam = 1.0/lp
                sig0 = 2.0/(2.0*np.pi*fref*4e-7*np.pi*lam**2) if fref > 0 \
                    else 0.0
                dtc = (float(model.d[fnorm]) if film
                       else max(float(v) for c, v in enumerate(model.d)
                                if c != self.term.axis))
                self.skin_k = (recommend_subdivision(dtc, sig0, fref)
                               if sig0 > 0 else 1)
            else:
                # fill models have no uniform sigma, but the base METAL
                # sigma (what the skin depth is made of) is well defined
                sig0 = (next(iter(spx['geom'].values()))[3] if spx
                        else model.uniform_sigma())
                dtc = (float(model.d[fnorm]) if film
                       else max(float(v) for c, v in enumerate(model.d)
                                if c != self.term.axis))
                self.skin_k = recommend_subdivision(dtc, sig0, fref)
        elif subdivide in (False, None):
            self.skin_k = 1
        else:
            self.skin_k = int(subdivide)
        if film and self.skin_k > 1:
            # 1-D split is cheap (kz, not kz^2, sub-bars), so give it
            # the conduction-quality quadrature unconditionally
            self.skin_k = int(min(12, max(7, self.skin_k)))
        subdivide = self.skin_k
        self.film_kk = None
        if film and subdivide > 1:
            others = [c for c in range(3) if c != self.term.axis]
            self.film_kk = tuple(subdivide if c == fnorm else 1
                                 for c in others)
        self.redist = None
        if subdivide > 1 and spx is not None:
            # subpixel models get the surface-anchored per-cell engine;
            # the coarse engine's identical-weight/uniform-sigma
            # assumptions do not hold here
            with _spstatus.task('skin engine k=%d' % subdivide):
                self.redist = SubpixelModes(model, M, self.term.axis,
                                            self.fil_axis,
                                            self.fil_cell,
                                            k=subdivide, term=self.term,
                                            rc_uu=rc_uu,
                                            rc_cross=rc_cross,
                                            csr_max_gb=csr_max_gb,
                                            skin_freq=fref)
        elif subdivide > 1:
            with _spstatus.task('skin engine k=%d' % subdivide):
                self.redist = Redistribution(
                    model, M, self.term.axis,
                    self.fil_axis, self.fil_cell,
                    split_axis=split_axis,
                    k=(self.film_kk or subdivide), term=self.term,
                    rc_uu=rc_uu, rc_cross=rc_cross,
                    use_fft=use_fft,
                    csr_max_gb=csr_max_gb,
                    mode_basis=mode_basis,
                    skin_freq=fref,
                    boundary_only=boundary_only)
        if corner_modes:
            if isinstance(self.redist, SubpixelModes):
                raise ValueError(
                    "corner_modes with the subpixel engine is not "
                    "composed yet (per-cell weights break the "
                    "cross-block fold) -- run one or the other")
            from cornermode import CornerModes, ModeStack
            arm = None
            if self.redist is not None:
                # match the tabulation baseline to the engine's
                # single-axis coverage (axis 2 has no in-plane modes)
                arm = {0: 'u', 1: 'v'}.get(int(self.redist.axis))
            cm = CornerModes(model, M, self.fil_axis, self.fil_cell,
                             rc_cross=rc_cross, verbose=verbose,
                             engine_arm=arm)
            if cm.nmode and self.redist is not None:
                self.redist = ModeStack(self.redist, cm)
            elif cm.nmode:
                self.redist = cm
            elif verbose:
                print("    corner modes: requested but no eligible "
                      "corners found")
        self.nu = 0 if self.redist is None else self.redist.nmode
        with _spstatus.task('assemble + preconditioner'):
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
        if self.basis == 'auto':
            # LpRSolver's rule (2026-08-22), applied here since 2026-08-26
            # when the RSFQ JTL -- moated superconducting ground planes,
            # 294 hole generators -- spent 694 s in the MST fundamental-
            # cycle fallback under the old unconditional 'selected'
            # (overcomplete: 78 s setup, same L to every printed digit).
            # The enumerator with fallback=False is a FREE probe of
            # "do the plaquettes span the cycle space?"; when they do
            # and the model is under the per-solve filament budget the
            # exact Cholesky wins, otherwise the over-complete frame.
            from port_impedance import _auto_selected_fil_budget
            self.basis = 'overcomplete'
            if efg < _auto_selected_fil_budget(self.nsolves):
                probe = mg.getmesh_fortran(
                    M.adjmats(), np.size(M.e.struc),
                    np.size(M.e.struc) + np.size(M.f.struc), efg, nn,
                    fallback=False)
                if probe is not None:
                    self.basis = 'selected'
                else:
                    self.basis_fallback = (
                        'selected basis deficient on this topology '
                        '(the MST fallback is the historic coil stall); '
                        'using the over-complete frame')
            else:
                self.basis_fallback = (
                    'over the selected-basis filament budget (%d >= %d '
                    'for %d solves); using the over-complete frame'
                    % (efg, _auto_selected_fil_budget(self.nsolves),
                       self.nsolves))
            if self.verbose and self.basis_fallback:
                print("  basis auto: %s" % self.basis_fallback)
            elif self.verbose:
                print("  basis auto: selected (plaquettes span, %d "
                      "filaments under budget)" % efg)
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
                # The basis runs [plaquettes | holes + port cycles |
                # redistribution modes]. Only the middle group belongs
                # in the exact Schur block: the modes are an identity
                # block in the cycle basis whose rows _precond
                # overwrites with mode_precond, and there are far too
                # many of them on a thin film to densify. Name the
                # macro set rather than let it be "everything past the
                # plaquettes".
                # SPPEEC_GEO_POSITIONAL=1 restores the old split for
                # A/B (it cannot run at thin-film mode counts).
                nmac = YT32.shape[0] - self.nplaq - self.nu
                mac = (None if os.environ.get('SPPEEC_GEO_POSITIONAL') == '1'
                       else np.arange(self.nplaq, self.nplaq + nmac))
                self.chol = _GeoMGFactor(
                    YT32, nrm, bse, self.nplaq, cycles=self.amg_cycles,
                    macro_idx=mac)
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
        # +-1 bases store exactly in float32 (bit-unchanged products,
        # refused if any entry would round); the factors above already
        # consumed their own full-precision copies
        from port_impedance import shrink_exact_f32
        for _mat in (self.Y, self.YT, self.B, self.Baug):
            shrink_exact_f32(_mat)

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
        import os
        from collections import deque
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
            # BREADTH-first, not depth-first. A DFS forest is long and
            # snaky, so its fundamental cycles are long, so the local
            # 4-filament plaquettes in _hole_cycles cannot match them and
            # the collapse stalls spuriously -- declaring generators that
            # no hole asked for. The path length grows with refinement,
            # which is exactly why that surplus grew 2 -> 53 going from
            # 200 nm to 100 nm. BFS gives shortest-path trees and the
            # shortest fundamental cycles, which is what a plaquette can
            # actually explain. Set SPPEEC_TREE_DFS=1 to get the old
            # order back for an A/B.
            bfs = not os.environ.get('SPPEEC_TREE_DFS')
            stack = deque([seed])
            while stack:
                u = stack.popleft() if bfs else stack.pop()
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

    def _betti1(self):
        """Independent tunnels through the conductor: the EXACT number of
        generators the plaquette basis is missing.

        The tree-cotree collapse in ``_hole_cycles`` is a MATCHING, not
        an elimination, so it can declare more generators than the
        topology needs -- and the surplus is not a fixed overhead, it
        GROWS as the mesh refines (measured on the RSFQ JTL: 294
        declared against 292 needed at 200 nm, 345 against the same 292
        at 100 nm). Those surplus columns are linearly dependent, they
        land in the macro Schur block, and its LU then inverts v-cycle
        noise along the dependent directions -- the 50 nm stall.

        The true count is a topological invariant, so it can be had
        without any linear algebra at all, from the Euler
        characteristic of the cubical complex the occupied voxels
        generate:

            chi = V - E + F - C  =  b0 - b1 + b2
            b1  = b0 + b2 - chi

        V/E/F/C are counted by OR-ing the occupancy over the neighbour
        shifts that share each cell; b0 is the conductor's component
        count (26-connected, because two voxels meeting at a corner
        share that vertex and so ARE connected in the cubical complex)
        and b2 the enclosed cavities (background components that never
        reach the border). Costs seconds and no solver.

        Returns None if the occupancy grid is unavailable, in which case
        the caller keeps the greedy count rather than guessing.
        """
        try:
            occ = np.asarray(self.model.struc()) != 0
        except Exception:
            return None
        if occ.ndim != 3 or not occ.any():
            return None
        try:
            from scipy import ndimage
        except ImportError:
            return None
        nx, ny, nz = occ.shape

        def or_over(offsets, shape):
            acc = np.zeros(shape, dtype=bool)
            for dx, dy, dz in offsets:
                acc[dx:dx + nx, dy:dy + ny, dz:dz + nz] |= occ
            return int(acc.sum())

        # a vertex lives where any of the 8 voxels around it is occupied
        V = or_over([(a, b, c) for a in (0, 1) for b in (0, 1)
                     for c in (0, 1)], (nx + 1, ny + 1, nz + 1))
        # an edge along axis a: any of the 4 voxels ringing it
        E = 0
        for a in range(3):
            offs = []
            for d0 in (0, 1):
                for d1 in (0, 1):
                    o = [0, 0, 0]
                    o[(a + 1) % 3] = d0
                    o[(a + 2) % 3] = d1
                    offs.append(tuple(o))
            shp = [nx + 1, ny + 1, nz + 1]
            shp[a] = occ.shape[a]
            E += or_over(offs, tuple(shp))
        # a face normal to axis a: either of the 2 voxels sharing it
        F = 0
        for a in range(3):
            o1 = [0, 0, 0]
            o1[a] = 1
            shp = [nx, ny, nz]
            shp[a] = occ.shape[a] + 1
            F += or_over([(0, 0, 0), tuple(o1)], tuple(shp))
        C = int(occ.sum())
        chi = V - E + F - C
        _, b0 = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        st6 = ndimage.generate_binary_structure(3, 1)
        labb, _ = ndimage.label(np.pad(~occ, 1, constant_values=True),
                                structure=st6)
        border = np.unique(np.concatenate([
            labb[0].ravel(), labb[-1].ravel(),
            labb[:, 0].ravel(), labb[:, -1].ravel(),
            labb[:, :, 0].ravel(), labb[:, :, -1].ravel()]))
        b2 = int(labb.max()) - int((border > 0).sum())
        return int(b0 + b2 - chi)

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
        # the pristine per-plaquette unmatched count, so the dedup pass
        # below can replay the collapse without rebuilding `members`
        # (which is the memory-heavy part of this routine)
        free0 = free.copy()
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

        # DEDUP to the topological truth. The collapse above is a
        # matching, so `gens` can overshoot -- and the overshoot grows
        # with refinement, which is what degrades the macro Schur block
        # (see _betti1). Drop a generator, replay the collapse, and keep
        # the drop only if the collapse still completes: completion
        # IMPLIES spanning by the induction above, so a kept drop can
        # never cost correctness. b1 supplies the stopping criterion the
        # greedy lacks; without it we would not know when to stop and
        # would pay a replay per generator for nothing.
        def replay(declared):
            """Collapse with `declared` pre-matched; True if it completes."""
            mt = np.zeros(self.efg, dtype=bool)
            fr = free0.copy()
            q = deque()

            def fire(f):
                mt[f] = True
                for j2 in edge_plaq.get(f, ()):
                    fr[j2] -= 1
                    if fr[j2] == 1:
                        q.append(j2)

            for f in declared:
                if not mt[f]:
                    fire(f)
            q.extend(np.flatnonzero(fr == 1).tolist())
            while q:
                j = q.popleft()
                if fr[j] != 1:
                    continue
                nxt = next((f for f in members[j] if not mt[f]), None)
                if nxt is None:
                    continue
                fire(nxt)
            return not bool((nontree & ~mt).any())

        target = self._betti1()
        if target is not None and len(gens) > target:
            surplus = len(gens) - target
            # Test the LAST-declared first: by then the collapse has
            # already covered the genuine holes, so late declarations
            # are the likely passengers. One replay per generator is the
            # natural budget -- there is nothing to learn from a second.
            kept = list(gens)
            dropped = 0
            # Repeat to a FIXED POINT. The replay test is sufficient but
            # not necessary, so a generator can fail while a passenger
            # is still in the set and pass once that passenger is gone
            # (measured: pass 1 caught 1 of the 2 surplus at 200 nm).
            # Each pass strictly shrinks `kept` or ends the loop, so
            # this terminates in at most `surplus` passes.
            while dropped < surplus:
                before = dropped
                for f in reversed(list(kept)):
                    if dropped == surplus:
                        break
                    cand = [g for g in kept if g != f]
                    if len(cand) == len(kept):
                        continue
                    if replay(cand):
                        kept = cand
                        dropped += 1
                if dropped == before:
                    break
            if self.verbose:
                print('  hole generators: %d declared, b1 = %d, '
                      '%d dropped -> %d' % (len(gens), target, dropped,
                                            len(kept)))
                if dropped < surplus:
                    print('    NOTE %d surplus generator(s) survived the '
                          'replay test (it is sufficient, not necessary) '
                          '-- macro block may still be singular'
                          % (surplus - dropped))
            gens = kept
        elif self.verbose and target is not None:
            print('  hole generators: %d declared, b1 = %d (no surplus)'
                  % (len(gens), target))
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
        if getattr(self, 'dL_near', None) is not None:
            # subpixel stage B on the equipotential path: sparse
            # partial-cell inductance correction (real dL, scaled jw
            # here); far field stays pure Toeplitz
            out_f += jw*(self.dL_near @ np.asarray(i_f))
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
              lsqr_tol=1e-12, method='lgmres', precision='auto',
              readout='tree'):
        """Solve at one frequency; return ``(Z, i, info)``.

        ``readout`` (added 2026-08-27): ``'tree'`` (default) threads the
        port current through one + face, the conductor spanning tree
        and one - face as the particular solution (KCL exact to
        roundoff, O(N)) and reads the port voltage through the
        work-conjugate identity ``V I = ihat . (Z i)``; ``'lsqr'`` is the
        former path -- minimum-norm ``ihat`` and a least-squares node
        potential solve, ~1600 iterations each on a large model (25 s
        apiece at 147k cells, the dominant per-frequency cost at 18M
        filaments). The converged answer is the same to O(rtol); the
        Krylov right-hand side differs, so iteration counts may move.

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
                with _spstatus.task('assemble + preconditioner'):
                    self._build_augmented()
        self._mode_pc = None
        if self.nu and hasattr(self.redist, 'mode_precond'):
            self._mode_pc = self.redist.mode_precond(self.M.jomega)
        n = self.efg + self.term.n + self.nu
        s = np.zeros(self.nnode + 2, dtype=np.complex128)
        s[self.pnode[+1]] = current
        s[self.pnode[-1]] = -current
        if readout == 'lsqr':
            BT = self.Baug.T.tocsc().astype(np.complex128)
            ihat = lsqr(BT, s, atol=lsqr_tol, btol=lsqr_tol)[0]
        elif readout == 'tree':
            ihat = self._tree_particular(current, s)
        else:
            raise ValueError("readout must be 'tree' or 'lsqr', got %r"
                             % (readout,))
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
        # READOUT, phase by phase. This tail is a handful of operations
        # and ought to cost about one matvec -- but on the RSFQ XNOR
        # with sub-cell modes it ran ~40 minutes against the same-scale
        # mode-free baseline's ~15 seconds, at the highest memory
        # watermark of the whole run and only 183% CPU. It was invisible
        # because the entire tail sat inside one 'solve f=...' task, so
        # the status file could not say which line it was in. Name them.
        with _spstatus.task('readout: residual'):
            nrhs = np.linalg.norm(rhs)
            resid = (np.linalg.norm(rhs - Aop*w)/nrhs if nrhs > 0
                     else 0.0)
        # The solved mesh vector, kept for diagnostics: its tail is the
        # redistribution-mode amplitudes, and comparing their norm to
        # the loop part is what showed the modes are barely EXCITED on
        # the RSFQ XNOR (modes/loops 1.7e-01 against equibar 1.2e+03)
        # -- the reason sub-cell London modes buy nothing there. One
        # reference to an array the caller already holds.
        self._last_w = w
        with _spstatus.task('readout: expand basis'):
            i = self.Y.dot(w) + ihat
        # Z i = B phi with phi the PHYSICAL potential (see the module
        # docstring): no sign flip to undo here.
        with _spstatus.task('readout: apply_Z'):
            zi = self.apply_Z(i)
        if readout == 'lsqr':
            with _spstatus.task('readout: lsqr potential'):
                phi = lsqr(self.Baug.astype(np.complex128), zi,
                           atol=lsqr_tol, btol=lsqr_tol)[0]
                v = (phi[self.pnode[+1]] - phi[self.pnode[-1]])/current
        else:
            # work-conjugate identity: for ANY ihat with Baug^T ihat = s,
            # ihat . (Baug phi) = s . phi = current*(phi_P - phi_N), and
            # Z i == Baug phi exactly when the mesh equations hold
            # (Y^T Z i = 0) -- LpRSolver's readout since 2026-08-04.
            # At finite residual d = Y^T Z i != 0 the identity picks up
            # a . d, where ihat = ihat_perp + Y a splits the particular
            # solution into the minimum-norm part (orthogonal to the
            # loop space, which is why the lsqr readout never saw this
            # term) and a loop-space part -- LARGE for the tree route,
            # which threads the whole current along one path: measured
            # 4.0e-3 on the equibar razor gate at rtol 1e-4 against the
            # 1e-3 it must meet. a = (Y^T Y)^-1 Y^T ihat is one Gram
            # solve, i.e. one preconditioner apply (exact with the
            # Cholesky, v-cycle accurate with GeoMG/BlockAMG), so the
            # readout is corrected by (Y^T ihat) . G^-1 d and its error
            # drops from O(|r|) to O(|r| x precond defect).
            with _spstatus.task('readout: gram correction'):
                d = self.YT.dot(zi)
                c = self._gram_solve(d, rtol=rtol)
                v = ((np.dot(ihat, zi)
                      - np.dot(self.YT.dot(ihat), c))/current)
        info = dict(matvecs=self.matvecs - n0, flag=flag, residual=resid,
                    time=time.perf_counter() - t0)
        if self.verbose:
            print("    %.4g Hz: %d matvecs, flag %s, resid %.2e, %.2f s"
                  % (freq, info['matvecs'], flag, resid, info['time']))
        return complex(v), i, info

    def _tree_particular(self, current, s):
        """Particular augmented current with ``Baug^T ihat == s``, O(N).

        The whole port current enters through the FIRST + face, follows
        the conductor spanning tree (already built for the hole and
        port cycles) to the FIRST - face and leaves through it; every
        other face and every non-tree filament carries zero. The
        equipotential solve redistributes the split across faces
        through the port cycles, so which face is chosen cannot matter
        at convergence. KCL is verified to roundoff before returning --
        the sign conventions are inherited from ``_port_cycles`` and
        checked rather than trusted.
        """
        term = self.term
        parent, pedge, depth, comp = self._spanning_tree()
        a = int(np.flatnonzero(term.pol == +1)[0])
        b = int(np.flatnonzero(term.pol == -1)[0])
        ca = int(self.node_of_cell[term.faces[a][0]])
        cb = int(self.node_of_cell[term.faces[b][0]])
        if comp[ca] != comp[cb]:
            raise ValueError(
                "the port's + and - faces sit on galvanically separate "
                "conductors (components %d and %d): Baug^T i = s has no "
                "solution, the LpR equipotential problem is not posed "
                "(a capacitive path needs LpPR)" % (comp[ca], comp[cb]))
        ihat = np.zeros(self.efg + term.n + self.nu, dtype=np.complex128)
        ihat[self.efg + a] = (+1.0 if term.sign[a] < 0 else -1.0)*current
        for f, sg in self._tree_path(ca, cb, parent, pedge, depth):
            ihat[f] = sg*current
        ihat[self.efg + b] = (-1.0 if term.sign[b] < 0 else +1.0)*current
        t = self.Baug.T @ ihat
        if abs(t[self.pnode[+1]] + current) < abs(t[self.pnode[+1]] - current):
            ihat = -ihat            # orientation convention flipped
            t = -t
        err = np.abs(t - s).max()
        if err > 1e-9*abs(current):
            raise RuntimeError("tree particular solution violates KCL "
                               "(max |Baug^T ihat - s| = %.3g)" % err)
        return ihat

    def _gram_solve(self, d, tol=None, maxiter=20, rtol=None):
        """``(Y^T Y)^-1 d`` to ``tol``: preconditioned lgmres on the Gram.

        TOLERANCE FOLLOWS THE SOLVE (2026-08-29). This used to target a
        hard-coded 1e-8 with a 60x10 = 600 iteration budget -- twice the
        mesh solve's own budget, at a tolerance four decades tighter
        than the rtol of the very solution it corrects. Each iteration
        is ``YT.(Y.x)`` over the whole basis plus a full preconditioner
        apply, so on a large basis this readout can cost more than the
        solve: measured on the RSFQ XNOR with sub-cell modes (24M mode
        columns on top of 8M plaquettes) it was 2400 s of a 2527 s
        solve -- 95%, with the Krylov loop deliberately capped at 12
        matvecs -- and it set the run's memory high-water mark.
        It was invisible until the readout was split into named status
        tasks, because the whole tail sat inside one 'solve f=...' task.
        The correction's job is to take the readout error from O(|r|)
        to O(|r| x gram defect), so a gram defect two decades below the
        mesh rtol already contributes nothing next to |r| itself;
        chasing 1e-8 buys accuracy the solution does not have.

        One preconditioner apply was exact enough at 147k cells
        (readout within 1.4e-5 of lsqr) but left a 0.5% readout error
        at 1.1M -- the v-cycle defect times the tree route's loop-space
        component. Plain iterative refinement with the preconditioner
        DIVERGED there (L 2.20 -> 3.23 pH, 2026-08-27): a preconditioner
        is only right up to scaling and the fixed-point map need not
        contract. CG is monotone on the SPD (semi-definite, consistent
        rhs) Gram and indifferent to the scaling, and a few dozen
        applies are negligible next to the Krylov solve. Returns the
        single-apply answer if CG cannot improve on it.
        """
        # NOT cg: the GeoMG/Schur apply is not a symmetric operator (cg
        # with it went from a Gram residual of 0.035 to 0.27 at 100 nm,
        # 2026-08-27); the Gram is SPD but the preconditioner only has
        # to be a good inverse, so right-preconditioned lgmres is the
        # method that is indifferent to both its scaling and its
        # asymmetry -- the same solver the mesh system uses.
        if tol is None:
            # two decades below the mesh solve's own tolerance, floored
            # so a very loose rtol cannot make the correction useless
            # and clamped so a very tight one cannot out-solve the old
            # hard-coded value
            tol = min(1e-4, max(1e-8, 1e-2*float(rtol or 1e-4)))
        nd = np.linalg.norm(d)
        c0 = self._precond(d)
        if nd == 0:
            return c0
        n = d.size
        Gop = LinearOperator((n, n), matvec=lambda x: self.YT.dot(self.Y.dot(x)),
                             dtype=np.complex128)
        Pop = LinearOperator((n, n), matvec=self._precond, dtype=np.complex128)
        c, flag = lgmres(Gop, d, x0=c0, M=Pop, rtol=tol, inner_m=10,
                         outer_k=3, maxiter=maxiter)
        r0 = np.linalg.norm(d - Gop.matvec(c0))
        r1 = np.linalg.norm(d - Gop.matvec(c))
        self._readout_gram = (r0/nd, r1/nd, int(flag), float(tol))
        if min(r0, r1) > 1e-4*nd:
            import warnings
            warnings.warn("port-voltage readout: Gram solve reached only "
                          "%.1e relative (target %g) -- the extracted Z "
                          "carries that much readout error on top of the "
                          "Krylov residual" % (min(r0, r1)/nd, tol),
                          RuntimeWarning, stacklevel=3)
        return c if r1 <= r0 else c0

    def _precond(self, vec):
        re = self.chol(np.float32(np.real(vec)))
        im = self.chol(np.float32(np.imag(vec)))
        out = np.float64(re) + 1j*np.float64(im)
        if getattr(self, '_mode_pc', None) is not None:
            # the mode tail of the cycle basis is an identity block, so
            # the Gram Cholesky leaves it unpreconditioned -- apply the
            # per-cell (Ru + jw Zuu)^-1 instead
            out[-self.nu:] = self._mode_pc @ vec[-self.nu:]
        return out

    def terminal_split(self, i, current=1.0):
        """Solved per-face current entering the conductor, share of I."""
        i_t = i[self.efg:self.efg + self.term.n]
        return -self.term.sign*i_t/current
