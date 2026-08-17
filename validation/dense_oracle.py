# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Dense LpR oracle for port/terminal models.

A direct, dense, FMM-free solve of the same physics SuperPEEC solves, small
enough to invert exactly, whose purpose is to settle questions about the
PORT MODEL that the production path cannot answer about itself.

WHY THIS EXISTS
---------------
``port_impedance`` drives a port with a PRESCRIBED current profile over
the terminal faces (``_terminal_slots`` hard-codes an equal share per
face). That is exact for a uniform bar cross-section and suspect for an
asymmetric return -- a wide ground plane under a narrow trace, where the
return current crowds. The suspicion is inductance-weighted and does not
vanish as ``dx -> 0``, which matches the refinement-independent ~10% L
discrepancy against VoxHenry on setup3.

This module COMPUTES the terminal current distribution instead of
assuming it, by giving all faces of a terminal a single shared node --
an EQUIPOTENTIAL terminal, which is also what VoxHenry imposes -- and
letting the solve distribute the current.

THE DISCRIMINATOR
-----------------
The two port models are the SAME dense system on the SAME geometry with
the SAME Lp and R. They differ in one thing only:

  ``port_bc='equipotential'``  all faces of a terminal share one node;
                               the current split is SOLVED.
  ``port_bc='prescribed'``     each face gets a PRIVATE node carrying a
                               prescribed injection, so KCL at that
                               degree-one node forces the terminal
                               current to the prescribed value.

Nothing else changes, so any difference in Z is attributable to the
boundary condition alone -- no FMM truncation, no mesh basis, no solver
tolerance in the comparison. Two consequences worth knowing:

  * prescribing the SOLVED distribution reproduces the equipotential
    answer EXACTLY (the true solution satisfies the constraint), so the
    substitution step of the investigation is a self-check, and the
    quantity of interest is the gap between equal-share and solved.
  * the port voltage is the work-conjugate ``V = (s.phi)/I`` in both
    cases, which for the equipotential model reduces to
    ``phi_P - phi_N``.

TERMINAL GEOMETRY
-----------------
``terminal=`` selects what closes the half cell between the port face
and the first node (nodes sit at cell CENTRES, so a conductor of ``L``
cells carries only ``L-1`` filaments and the chain is ``dx`` short):

  ``'exact'``      a half-length filament per face, spanning face to
                   centre -- ``terminal.py``'s model, validated against
                   6-D quadrature. Path length ``L*dx``. THE REFERENCE.
  ``'onlattice'``  a FULL-length filament per face reaching one cell
                   OUTSIDE the conductor, with half the resistance
                   (i.e. doubled conductivity). Ordinary lattice
                   filament, so in production it would need no special
                   kernel and no dense terminal<->interior gather -- but
                   the current path is ``(L+1)*dx``, one cell too long.
  ``'none'``       no terminal filaments; the port attaches directly to
                   the end-cell nodes. Path length ``(L-1)*dx``.

The three straddle the truth by +-1 cell of path length, so comparing
them measures the scalability/accuracy trade directly rather than
arguing it from a length ratio.

MODEL SCOPE
-----------
Inductive only (LpR), uniform cell size, uniform conductivity except for
the terminal filaments. Only PARALLEL filaments couple: the mutual
partial inductance of perpendicular rectangular bars vanishes, so Lp is
block diagonal in orientation. All partial inductances -- ordinary,
terminal, and mixed -- come from ``terminal.axial_halfstep_kernel``,
which puts every bar on a common half-cell axial lattice so one
identical-bar table serves all three cases.

This certifies a port model against SuperPEEC's OWN physics (same filament
basis, same partial-inductance kernel). It is not an independent
statement about reality, which is the right scope: the discrepancy under
investigation is refinement-independent, i.e. a model question.

Run inside the toolbox:  python3 dense_oracle.py
"""

import numpy as np

import terminal as tm

MU0 = 4e-7*np.pi

# axis -> filament orientation letter, matching port_impedance._AXIS_ORIENT
AXIS_ORIENT = {0: 'f', 1: 'e', 2: 'g'}


class Geometry:
    """Occupied cells on a uniform cubic lattice.

    Parameters
    ----------
    cells : iterable of (int, int, int)
        Occupied cell indices.
    dx : float
        Cell pitch (metres).
    sigma : float
        Conductivity (S/m).
    """

    def __init__(self, cells, dx, sigma):
        self.cells = sorted({(int(a), int(b), int(c)) for a, b, c in cells})
        self.dx = float(dx)
        self.sigma = float(sigma)
        self.node = {c: i for i, c in enumerate(self.cells)}

    def __len__(self):
        return len(self.cells)

    @property
    def dims(self):
        a = np.array(self.cells)
        return tuple(int(v) + 1 for v in a.max(axis=0))


class Port:
    """A port as a set of voxel faces, in ``.vhr``'s own description.

    Parameters
    ----------
    pos, neg : iterable of (cell, axis, sign)
        ``cell`` is a 3-tuple, ``axis`` is 0/1/2 and ``sign`` is -1 for
        the low face of the cell or +1 for the high face. Current is
        injected across ``pos`` and extracted across ``neg``.
    """

    def __init__(self, pos, neg, name='port'):
        self.name = name
        self.pos = [(tuple(int(v) for v in c), int(a), int(s))
                    for c, a, s in pos]
        self.neg = [(tuple(int(v) for v in c), int(a), int(s))
                    for c, a, s in neg]
        axes = {a for _, a, _ in self.pos + self.neg}
        if len(axes) != 1:
            raise ValueError("port faces must share one axis, got %s" % axes)
        self.axis = axes.pop()

    @property
    def faces(self):
        return [(f, +1) for f in self.pos] + [(f, -1) for f in self.neg]


class System:
    """Dense LpR system for one geometry, port and terminal model.

    Frequency-independent assembly (filaments, incidence, Lp, R) happens
    once here; :meth:`solve` applies a frequency and an injection.

    Parameters
    ----------
    geom : Geometry
    port : Port
    terminal : {'exact', 'onlattice', 'none'}
    port_bc : {'equipotential', 'prescribed'}
    """

    def __init__(self, geom, port, terminal='exact',
                 port_bc='equipotential'):
        if terminal not in ('exact', 'onlattice', 'none'):
            raise ValueError("terminal must be exact/onlattice/none")
        if port_bc not in ('equipotential', 'prescribed'):
            raise ValueError("port_bc must be equipotential/prescribed")
        self.geom = geom
        self.port = port
        self.terminal = terminal
        self.port_bc = port_bc
        self._build_filaments()
        self._build_incidence()
        self._build_lp()

    # -- assembly -------------------------------------------------------

    def _build_filaments(self):
        """Interior filaments between adjacent occupied cells, then the
        port's terminal filaments.

        Every filament is a contiguous run of half-slots on an axial
        lattice of pitch ``dx/2``: a cell-centred filament joining cells
        ``i`` and ``i+1`` occupies half-slots ``2i+1`` and ``2i+2``, and
        a terminal occupies the single half-slot next to its face. This
        is the same layout ``port_impedance._interior_slots`` and
        ``_terminal_slots`` use, so indices are directly comparable.
        """
        g = self.geom
        axis, off, ln, t0, t1, tail, head, res = [], [], [], [], [], [], [], []
        r_full = 1.0/(g.sigma*g.dx)          # dx/(sigma*dx^2)
        for d in range(3):
            others = [c for c in range(3) if c != d]
            step = [0, 0, 0]
            step[d] = 1
            for c in g.cells:
                nb = (c[0] + step[0], c[1] + step[1], c[2] + step[2])
                if nb not in g.node:
                    continue
                axis.append(d)
                off.append(2*c[d] + 1)
                ln.append(2)
                t0.append(c[others[0]])
                t1.append(c[others[1]])
                tail.append(g.node[c])
                head.append(g.node[nb])
                res.append(r_full)
        self.n_interior = len(axis)

        # Terminal filaments and the port nodes they attach to.
        d = self.port.axis
        others = [c for c in range(3) if c != d]
        nn = len(g.cells)
        self.port_node = {}
        self.face_fil = []               # filament index per port face
        for k, ((cell, _, sign), pol) in enumerate(self.port.faces):
            if cell not in g.node:
                raise ValueError("port face on empty cell %s" % (cell,))
            if self.port_bc == 'equipotential':
                key = pol
            else:
                key = ('face', k)
            if key not in self.port_node:
                self.port_node[key] = nn + len(self.port_node)
            pnode = self.port_node[key]
            if self.terminal == 'none':
                self.face_fil.append(None)
                continue
            if self.terminal == 'exact':
                o = 2*cell[d] + (1 if sign > 0 else 0)
                length = 1
            else:                        # 'onlattice'
                o = 2*cell[d] + (1 if sign > 0 else -1)
                length = 2
            # current is positive along +d for every filament, so a low
            # face runs port -> cell and a high face runs cell -> port
            if sign > 0:
                tl, hd = g.node[cell], pnode
            else:
                tl, hd = pnode, g.node[cell]
            self.face_fil.append(len(axis))
            axis.append(d)
            off.append(o)
            ln.append(length)
            t0.append(cell[others[0]])
            t1.append(cell[others[1]])
            tail.append(tl)
            head.append(hd)
            res.append(0.5*r_full)       # half length, or doubled sigma

        self.axis = np.array(axis, dtype=np.int64)
        self.off = np.array(off, dtype=np.int64)
        self.len = np.array(ln, dtype=np.int64)
        self.t0 = np.array(t0, dtype=np.int64)
        self.t1 = np.array(t1, dtype=np.int64)
        self.tail = np.array(tail, dtype=np.int64)
        self.head = np.array(head, dtype=np.int64)
        self.R = np.array(res, dtype=np.float64)
        self.nf = self.axis.size
        self.nn = nn + len(self.port_node)

    def _build_incidence(self):
        """``B[f, n]``: +1 at the filament's tail, -1 at its head.

        With current positive from tail to head, KCL reads
        ``B^T i = s`` (current leaving each node) and the branch
        equation reads ``Z i = B phi``.
        """
        B = np.zeros((self.nf, self.nn))
        B[np.arange(self.nf), self.tail] += 1.0
        B[np.arange(self.nf), self.head] -= 1.0
        self._merge = {}
        if self.terminal == 'none':
            # With no terminal filament the port attaches straight to
            # the end-cell nodes, so the port node is an ideal
            # (zero-impedance) link and is merged away. Under the
            # equipotential BC the terminal itself is a short across
            # every end cell of one polarity, so those cell nodes merge
            # into one representative too -- that IS the equipotential
            # terminal in this model, and getting it wrong turns the
            # port into a single-cell injection with a spurious
            # constriction resistance.
            reps = {}
            for k, ((cell, _, _), pol) in enumerate(self.port.faces):
                cnode = self.geom.node[cell]
                if self.port_bc == 'equipotential':
                    rep = reps.setdefault(pol, cnode)
                    if cnode != rep:
                        self._merge[cnode] = rep
                    self._merge[self.port_node[pol]] = rep
                else:
                    self._merge[self.port_node[('face', k)]] = cnode
        self.B = B

    def _build_lp(self):
        """Dense partial inductance, one block per orientation.

        Only parallel filaments couple. Each pair is the double sum over
        half-slots of the identical-bar kernel -- the identity
        ``terminal.py`` is built on and ``validate_terminal.py`` checks
        against 6-D quadrature.
        """
        g = self.geom
        self.Lp = np.zeros((self.nf, self.nf))
        for d in range(3):
            sel = np.flatnonzero(self.axis == d)
            if sel.size == 0:
                continue
            others = [c for c in range(3) if c != d]
            off, ln = self.off[sel], self.len[sel]
            t0, t1 = self.t0[sel], self.t1[sel]
            n = [0, 0, 0]
            # kernel extents: axial in half slots (the table holds
            # 2*n[d]+1), transverse in whole cells
            n[d] = int((off.max() + ln.max() - off.min())//2) + 2
            n[others[0]] = int(t0.max() - t0.min()) + 2
            n[others[1]] = int(t1.max() - t1.min()) + 2
            kern = tm.axial_halfstep_kernel((g.dx,)*3, AXIS_ORIENT[d], n)
            blk = _pair_block(kern, d, off, ln, t0, t1)
            self.Lp[np.ix_(sel, sel)] = blk
        asym = np.abs(self.Lp - self.Lp.T).max()
        if asym > 1e-18*max(1.0, np.abs(self.Lp).max()):
            raise RuntimeError("Lp not symmetric: %.3e" % asym)

    # -- solve ----------------------------------------------------------

    def injection(self, current=1.0, shares=None):
        """Nodal injection vector for the port.

        ``shares`` (prescribed BC only) is the per-face current split,
        normalised to sum to 1 within EACH terminal; ``None`` gives the
        equal share ``port_impedance`` hard-codes.

        Shares may be COMPLEX. The solved split has a phase that varies
        across a terminal at high frequency, and feeding it back is the
        exactness check on this module -- prescribing the true split
        must reproduce the equipotential answer identically. (Production
        can only prescribe a real profile, which is a separate
        limitation of the model, not of this check.)
        """
        s = np.zeros(self.nn, dtype=np.complex128)
        faces = self.port.faces
        if self.port_bc == 'equipotential':
            s[self.port_node[+1]] += current
            s[self.port_node[-1]] -= current
        else:
            npos = sum(1 for _, pol in faces if pol > 0)
            nneg = len(faces) - npos
            for k, (_, pol) in enumerate(faces):
                if shares is None:
                    u = 1.0/(npos if pol > 0 else nneg)
                else:
                    u = complex(shares[k])
                s[self.port_node[('face', k)]] += pol*current*u
        if self._merge:
            out = np.zeros(self.nn, dtype=np.complex128)
            for i, v in enumerate(s):
                out[self._merge.get(i, i)] += v
            s = out
        return s

    def solve(self, freq, current=1.0, shares=None):
        """Solve at one frequency.

        Returns
        -------
        z : complex
            Port impedance ``V/I`` with the work-conjugate voltage
            ``V = (s.phi)/I``.
        i : ndarray
            Filament currents.
        phi : ndarray
            Node potentials, gauge-fixed at node 0.
        """
        jw = 2j*np.pi*freq
        Z = np.diag(self.R.astype(np.complex128)) + jw*self.Lp
        B = self.B.copy()
        s = self.injection(current, shares)
        for src, dst in self._merge.items():
            B[:, dst] += B[:, src]
            B[:, src] = 0.0
        nf, nn = self.nf, self.nn
        K = np.zeros((nf + nn, nf + nn), dtype=np.complex128)
        K[:nf, :nf] = Z
        K[:nf, nf:] = -B
        K[nf:, :nf] = B.T
        rhs = np.concatenate([np.zeros(nf, dtype=np.complex128), s])
        # Gauge: ONE node per CONNECTED COMPONENT. A galvanically
        # isolated conductor -- a floating shield, an unconnected pad, a
        # refined degenerate cell -- is its own component with its own
        # free potential constant, so pinning a single global node
        # leaves the system singular. Isolated nodes (no filament at
        # all) come out as their own components, which also covers the
        # dead-node case this used to special-case.
        import scipy.sparse as _sp
        import scipy.sparse.csgraph as _csg
        adj = _sp.coo_matrix(
            (np.ones(self.nf), (self.tail, self.head)),
            shape=(self.nn, self.nn))
        ncomp, lab = _csg.connected_components(adj, directed=False)
        pins = [int(np.flatnonzero(lab == c)[0]) for c in range(ncomp)]
        self.ncomponents = ncomp
        for p in np.unique(pins):
            K[nf + p, :] = 0.0
            K[nf + p, nf + p] = 1.0
            rhs[nf + p] = 0.0
        x = np.linalg.solve(K, rhs)
        i, phi = x[:nf], x[nf:]
        z = complex(np.dot(s, phi)/current)
        return z, i, phi

    def prescribable(self, u):
        """A solved split from :meth:`terminal_currents` as ``shares``.

        :meth:`terminal_currents` reports current ENTERING the conductor
        (so ``neg`` faces come out negative) while :meth:`injection`
        wants a per-terminal share that is positive on both, with the
        polarity applied separately. Getting this by hand is easy to get
        backwards, so it lives here.
        """
        pol = np.array([p for _, p in self.port.faces])
        return pol*np.asarray(u)

    def terminal_currents(self, i, current=1.0):
        """Per-face current entering the conductor, as a share of ``I``.

        Positive means current flowing INTO the conductor at that face,
        so the ``pos`` faces sum to +1 and the ``neg`` faces to -1. This
        is the quantity ``port_impedance._terminal_slots`` assumes to be
        uniform.
        """
        if self.terminal == 'none':
            raise ValueError("no terminal filaments to report")
        out = []
        for k, ((_, _, sign), _) in enumerate(self.port.faces):
            out.append(-sign*i[self.face_fil[k]]/current)
        return np.array(out)


def _pair_block(kern, axis, off, ln, t0, t1, chunk=512):
    """Mutual partial inductance between every pair of parallel bars.

    Vectorised form of ``terminal.mutual_segments`` over a whole block:
    each bar is a run of half-slots, and the pair value is the double
    sum of the identical-bar kernel over those half-slots.
    """
    others = [c for c in range(3) if c != axis]
    n = off.size
    out = np.zeros((n, n))
    lmax = int(ln.max())
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d0 = np.abs(t0[lo:hi, None] - t0[None, :])
        d1 = np.abs(t1[lo:hi, None] - t1[None, :])
        acc = np.zeros((hi - lo, n))
        for a in range(lmax):
            for b in range(lmax):
                m = (a < ln[lo:hi, None]) & (b < ln[None, :])
                if not m.any():
                    continue
                da = np.abs((off[None, :] + b) - (off[lo:hi, None] + a))
                sel = [None, None, None]
                sel[axis] = da
                sel[others[0]] = d0
                sel[others[1]] = d1
                acc += np.where(m, kern[sel[0], sel[1], sel[2]], 0.0)
        out[lo:hi] = acc
    return out


# -- electrostatics ------------------------------------------------------

def potential_matrix(geom):
    """Dense coefficient-of-potential matrix between cell charges.

    P[i,j] relates node charges to node potentials, ``phi = P q``, for
    charge distributed uniformly through each cell.

    NO NEW KERNEL IS NEEDED. The coefficient of potential between two
    uniformly charged cells and the partial mutual inductance between
    the same two cells are the SAME integral,

        I = Int_a Int_b dV dV' / |r - r'|,

    differing only by their normalising constants:

        Mp = (mu0/4pi)/(A_a A_b) I        P = (1/4pi eps0)/(V_a V_b) I

    so for identical cubes of side ``dx`` (A = dx^2, V = dx^3)

        P = Mp * A^2 / (mu0 eps0 V_a V_b) = Mp * c^2 / dx^2.

    Checked against the far limit: Mp -> (mu0/4pi) dx^2 / r gives
    P -> 1/(4 pi eps0 r), as it must. This reuses ``genL3D``'s exact
    closed form -- including the self term -- rather than quadrature.

    SCOPE: charge is VOLUME distributed per cell, the volumetric-PEEC
    convention (PyPEEC's). SuperPEEC production puts charge on SURFACE
    panels, so absolute capacitances differ; what this oracle is for is
    the structural question -- how many independent charge degrees of
    freedom a conductor has, and what that costs -- which is
    convention-independent.
    """
    from greens import genL3D
    c0 = 299792458.0
    cells = np.array(geom.cells)
    span = cells.max(axis=0) - cells.min(axis=0) + 2
    K = genL3D(geom.dx, geom.dx, geom.dx,
               int(span[0]), int(span[1]), int(span[2]))
    d = np.abs(cells[:, None, :] - cells[None, :, :])
    P = K[d[:, :, 0], d[:, :, 1], d[:, :, 2]]*c0**2/geom.dx**2
    asym = np.abs(P - P.T).max()
    if asym > 1e-6*np.abs(P).max():
        raise RuntimeError("potential matrix not symmetric: %.3e" % asym)
    return P


def box_potential_matrix(lo, hi):
    """P between uniformly charged axis-aligned boxes of ARBITRARY size.

    The fully general form of the identity :func:`potential_matrix`
    uses: the double integral of ``1/|r-r'|`` over two boxes is a mixed
    second difference, in EACH of the three axes, of the box function
    ``greens.box_selfind``. For an axis where box A spans ``[0,p]`` and
    box B spans ``[s,s+q]`` the four terms are ``|s+q|, |s|, |s+q-p|,
    |s-p|`` with coefficients ``+1/2,-1/2,-1/2,+1/2``; taking the tensor
    product over three axes gives 4**3 = 64 terms. With equal boxes on a
    lattice each axis collapses to genL3D's symmetric ``1/2 [1,-2,1]``
    and this reduces to :func:`potential_matrix` -- checked in
    ``validate``-style form by the caller.

    This is what lets charge sites of DIFFERENT sizes coexist, i.e. lets
    one conductor's charge be refined while the rest of the geometry is
    left alone.

    Parameters
    ----------
    lo, hi : ndarray, shape (n, 3)
        Lower and upper corners of each box (metres).

    Returns
    -------
    ndarray, shape (n, n)
        Coefficients of potential (1/F).
    """
    from greens import box_selfind
    c0 = 299792458.0
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = lo.shape[0]
    ext = hi - lo
    vol = np.prod(ext, axis=1)
    lens = []
    for d in range(3):
        p = ext[:, d][:, None]
        q = ext[:, d][None, :]
        s = lo[None, :, d] - lo[:, None, d]
        lens.append(np.stack([np.abs(s + q), np.abs(s),
                              np.abs(s + q - p), np.abs(s - p)], axis=-1))
    coef = np.array([0.5, -0.5, -0.5, 0.5])
    S = np.zeros((n, n))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                S += (coef[i]*coef[j]*coef[k])*box_selfind(
                    lens[0][:, :, i], lens[1][:, :, j], lens[2][:, :, k])
    P = S*c0**2/(vol[:, None]*vol[None, :])
    asym = np.abs(P - P.T).max()
    if asym > 1e-6*np.abs(P).max():
        raise RuntimeError("box potential matrix not symmetric: %.3e" % asym)
    return P


class Electrostatic:
    """Capacitance with driven and FLOATING conductors.

    Each conductor is a set of cells at one potential. Driven ones have
    it prescribed; a floating one has it as an UNKNOWN, subject to net
    charge zero -- which is the whole point, because that is what lets a
    parasitic patch polarise.

    Parameters
    ----------
    geom : Geometry
    conductors : list of iterable of cell
        One entry per conductor.
    """

    def __init__(self, geom, conductors):
        self.geom = geom
        self.P = potential_matrix(geom)
        self.groups = []
        for cs in conductors:
            idx = np.array(sorted(geom.node[tuple(c)] for c in cs))
            self.groups.append(idx)

    def solve(self, driven):
        """``driven`` maps conductor index -> potential; others float.

        Returns ``(q, potentials)``: the node charges and the potential
        of every conductor (solved for the floating ones).
        """
        n = self.P.shape[0]
        free = [k for k in range(len(self.groups)) if k not in driven]
        K = np.zeros((n + len(free), n + len(free)))
        rhs = np.zeros(n + len(free))
        K[:n, :n] = self.P
        for k, idx in enumerate(self.groups):
            if k in driven:
                rhs[idx] = driven[k]
            else:
                col = n + free.index(k)
                K[idx, col] = -1.0          # phi_i - V_float = 0
                K[col, idx] = 1.0           # sum of charge = 0
        x = np.linalg.solve(K, rhs)
        q = x[:n]
        pot = dict(driven)
        for k in free:
            pot[k] = float(x[n + free.index(k)])
        return q, pot

    def capacitance(self, a=0, b=1):
        """Two-terminal capacitance between conductors ``a`` and ``b``."""
        q, _ = self.solve({a: +0.5, b: -0.5})
        return float(q[self.groups[a]].sum())


# -- geometries ---------------------------------------------------------

def from_vhr(path, port=0):
    """Build a :class:`Geometry` and :class:`Port` from a ``.vhr`` file.

    Lets the oracle run the SAME input the production path and VoxHenry
    run, so a three-way comparison has no geometry translation in it.
    Requires a single conductivity, which is SuperPEEC's own limit anyway.
    """
    import vhr
    m = vhr.read_vhr(path)
    sig = m.sigma_values()
    if sig.size != 1:
        raise ValueError("%s: %d conductivities, oracle needs one"
                         % (path, sig.size))
    cells = np.argwhere(m.struc() != 0)
    p = m.port(port)
    pos = [((int(e[0]), int(e[1]), int(e[2])), int(e[3]), int(e[4]))
           for e in p.pos]
    neg = [((int(e[0]), int(e[1]), int(e[2])), int(e[3]), int(e[4]))
           for e in p.neg]
    return (Geometry(cells, m.dx, float(sig[0])),
            Port(pos, neg, p.name), m)



def straight_bar(nx=12, ny=3, nz=3, dx=1e-6, sigma=5.8e7):
    """Uniform bar with a full-cross-section axial port at each end.

    The control case: equal share IS the true distribution by symmetry,
    so every port model must agree here, and DC resistance must come out
    exactly ``l/(sigma*A)``.
    """
    cells = [(i, j, k) for i in range(nx) for j in range(ny)
             for k in range(nz)]
    pos = [((0, j, k), 0, -1) for j in range(ny) for k in range(nz)]
    neg = [((nx-1, j, k), 0, +1) for j in range(ny) for k in range(nz)]
    return Geometry(cells, dx, sigma), Port(pos, neg, 'axial')


def trace_over_plane(nx=10, ny=9, dx=1e-6, sigma=5.8e7):
    """Trace over a ground plane, shorted at one end, port at the other.

    setup3 in miniature and the geometry the hypothesis is about: the
    port is TRANSVERSE (vertical, into the plane) and ASYMMETRIC -- one
    face on the trace against ``ny`` faces spread across the plane
    width, where the return current crowds under the trace rather than
    spreading evenly. No existing test covers this regime.

    Plane at ``z=0`` (``nx`` by ``ny``), trace at ``z=2`` (``nx`` by 1,
    centred), joined by a via at ``x=0``. Port at ``x=nx-1`` between the
    trace's underside and the plane's upper face.
    """
    yc = ny//2
    cells = [(i, j, 0) for i in range(nx) for j in range(ny)]
    cells += [(i, yc, 2) for i in range(nx)]
    cells += [(0, yc, 1)]
    pos = [((nx-1, yc, 2), 2, -1)]
    neg = [((nx-1, j, 0), 2, +1) for j in range(ny)]
    return Geometry(cells, dx, sigma), Port(pos, neg, 'vertical')


def _selftest():
    """DC exactness on the straight bar, for all three terminal models."""
    nx, ny, nz, dx, sigma = 12, 3, 3, 1e-6, 5.8e7
    geom, port = straight_bar(nx, ny, nz, dx, sigma)
    exact = (nx*dx)/(sigma*(ny*dx)*(nz*dx))
    print("straight bar %dx%dx%d, %d cells, %d nodes"
          % (nx, ny, nz, len(geom), len(geom)))
    print("  closed-form DC R = %.12e ohm  (l/(sigma*A))" % exact)
    print("  %-11s %-15s %-18s %s" % ('terminal', 'port_bc', 'R_dc', 'ratio'))
    bad = 0
    for term in ('exact', 'onlattice', 'none'):
        for bc in ('equipotential', 'prescribed'):
            sysd = System(geom, port, terminal=term, port_bc=bc)
            z, _, _ = sysd.solve(0.0)
            r = z.real
            print("  %-11s %-15s %-18.12e %.10f"
                  % (term, bc, r, r/exact))
            if term in ('exact', 'onlattice') and abs(r/exact - 1) > 1e-12:
                bad += 1
    return bad


if __name__ == '__main__':
    import sys
    sys.exit(1 if _selftest() else 0)
