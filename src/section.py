# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Section cuts: geometry INVARIANT along one lattice axis and resolved
below the cell in the plane across it (docs/trace_plan.md).

A round conductor (``[[cylinder]]``) and a routed trace (``[[trace]]``)
are the same object here: a list of convex pieces in the section
plane, ``('circle', c1, c2, R)`` or ``('poly', vertices)``, whose
UNION is the metal. Everything asks the pieces two questions --
is this point inside, and how far is it from the boundary (signed,
positive inside) -- through :func:`field`; the painter and the
surface palette share them, so a bend, a trace end, two traces meeting
or a trace over a cylinder need no case analysis.

The model record it produces (``VoxelModel.cut``)::

    dict(kind='section', axis=a,        # the invariance axis
         shapes=[...],                  # every piece, in metres
         k=ks, cells={(t1, t2): bins})  # ks x ks sub-fill bins of every
                                        # claimed transverse cell

with ``model.fill`` the covered fraction per cell. Cells the union
covers whole are ``fill == 1`` with all-one bins; the fill and bins of
a boundary cell are sampled (``S`` points per cell axis). A cell is
classified from the union's signed distance at its centre: at least
half a diagonal inside -> whole, at least half a diagonal outside ->
empty (the distance is exact inside a convex piece and a lower bound
outside it, so both tests are safe); only the rest is sampled.
"""
import numpy as np

S = 64          # samples per cell axis on boundary cells (16 per bin at k=4)
KS = 4          # sub-fill bins per cell axis (measured NOT the accuracy
                # limiter: k = 8 left the Kelvin gate unchanged-to-worse)


# --- pieces -------------------------------------------------------------

def _ccw(v):
    v = np.asarray(v, dtype=float)
    area = 0.5*np.sum(v[:, 0]*np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1)*v[:, 1])
    return v if area >= 0 else v[::-1]


def trace_pieces(path, width):
    """Convex pieces of a polyline trace: one rectangle per segment and,
    at every interior vertex, the mitre quadrilateral that closes the
    outer corner (a bevel triangle when the mitre would reach farther
    than two widths, i.e. bends sharper than ~30 degrees)."""
    P = np.asarray(path, dtype=float)
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] != 2:
        raise ValueError("trace path_m: at least two [x, y] points")
    hw = 0.5*float(width)
    seg = np.diff(P, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    if np.any(ln <= 0):
        raise ValueError("trace path_m repeats a point")
    t = seg/ln[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)         # left normals
    pieces = []
    for i in range(len(seg)):
        a, b = P[i], P[i + 1]
        pieces.append(('poly', _ccw([a + n[i]*hw, b + n[i]*hw,
                                     b - n[i]*hw, a - n[i]*hw])))
    for j in range(1, len(P) - 1):
        n0, n1 = n[j - 1], n[j]
        cross = t[j - 1, 0]*t[j, 1] - t[j - 1, 1]*t[j, 0]
        if abs(cross) < 1e-12:
            if np.dot(t[j - 1], t[j]) < 0:
                raise ValueError("trace path_m reverses on itself at "
                                 "point %d" % j)
            continue                                   # collinear
        s = -1.0 if cross > 0 else 1.0                 # the OUTER side
        A = P[j] + s*n0*hw
        B = P[j] + s*n1*hw
        m = s*(n0 + n1)/(1.0 + float(np.dot(n0, n1)))*hw
        if np.hypot(*m) > 4*hw:
            pieces.append(('poly', _ccw([P[j], A, B])))
        else:
            pieces.append(('poly', _ccw([P[j], A, P[j] + m, B])))
    return pieces


def field(shapes, x, y):
    """Signed distance to the union boundary (> 0 inside) and the
    outward-normal angle of the piece that sets it, at the points
    ``(x, y)`` (broadcast). Exact inside every convex piece; outside,
    a lower bound on the distance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.full(np.broadcast(x, y).shape, -np.inf)
    phi = np.zeros_like(d)
    for sh in shapes:
        if sh[0] == 'circle':
            _, c1, c2, R = sh
            dxp, dyp = x - c1, y - c2
            ds = R - np.hypot(dxp, dyp)
            ps = np.arctan2(dyp, dxp)
        else:
            v = np.asarray(sh[1], dtype=float)
            e = np.roll(v, -1, axis=0) - v
            el = np.hypot(e[:, 0], e[:, 1])
            nx, ny = e[:, 1]/el, -e[:, 0]/el          # outward (ccw poly)
            ds = np.full(d.shape, np.inf)
            ps = np.zeros(d.shape)
            for k in range(len(v)):
                sk = -((x - v[k, 0])*nx[k] + (y - v[k, 1])*ny[k])
                take = sk < ds
                ds = np.where(take, sk, ds)
                ps = np.where(take, np.arctan2(ny[k], nx[k]), ps)
        take = ds > d
        d = np.where(take, ds, d)
        phi = np.where(take, ps, phi)
    return d, phi


def inside(shapes, x, y):
    return field(shapes, x, y)[0] > 0.0


def lattice(s):
    """``s x s`` sample offsets in [0, 1)^2, row ``k`` of the regular
    lattice shifted along x by the golden fraction of a spacing times
    k (mod 1). The unshifted lattice counts a 45-degree edge with a
    +1/(4s) bias per cell (every lattice diagonal flips at once);
    the shifted rows never share a line of rational slope, so the
    count error averages out. Bin membership follows the shifted
    coordinate."""
    j = np.arange(s) + 0.5
    k = np.arange(s)
    ox = np.mod(j[:, None] + 0.6180339887498949*k[None, :], s)/s
    oy = np.broadcast_to(j[None, :]/s, (s, s))
    return ox, oy


# --- the painter --------------------------------------------------------

def paint(m, prims, axis, ks=KS, s=S):
    """Paint section primitives into ``m`` (sigma, fill, cut).

    ``prims`` is a list of ``(pieces, sigma, a0, a1)``: the convex
    pieces of one primitive, its conductivity and its half-open cell
    span along ``axis``. Union rule: a cell is metal if any primitive
    claims it (whole, or with fill >= 1e-3); a later primitive's sigma
    wins where they overlap; a cell some BLOCK already fills whole is
    left alone (no cut). If no cell ends up partial the cut record is
    dropped: a commensurate trace IS a block, to every reader.
    """
    t1, t2 = [c for c in range(3) if c != axis]
    n1, n2 = int(m.dims[t1]), int(m.dims[t2])
    p1, p2 = float(m.d[t1]), float(m.d[t2])
    half = 0.5*np.hypot(p1, p2)
    xc = (np.arange(n1) + 0.5)*p1
    yc = (np.arange(n2) + 0.5)*p2
    XC, YC = np.meshgrid(xc, yc, indexing='ij')
    whole = np.zeros((len(prims), n1, n2), dtype=bool)
    bnd = np.zeros((n1, n2), dtype=bool)
    for q, (pieces, _, _, _) in enumerate(prims):
        dc, _ = field(pieces, XC, YC)
        whole[q] = dc >= half
        bnd |= (dc > -half) & ~whole[q]
    bnd &= ~whole.any(axis=0)
    # sample the boundary cells: fill and bins of the UNION, and each
    # primitive's own claim
    ox, oy = lattice(s)
    bi, bj = np.nonzero(bnd)
    claim = whole.copy()
    fill = whole.any(axis=0).astype(np.float64)
    bins = {}
    if bi.size:
        xs = (bi[:, None, None] + ox[None, :, :])*p1        # (nb, s, s)
        ys = (bj[:, None, None] + oy[None, :, :])*p2
        ins = np.zeros((bi.size, s, s), dtype=bool)
        for q, (pieces, _, _, _) in enumerate(prims):
            iq = inside(pieces, xs, ys)
            claim[q, bi, bj] = iq.reshape(bi.size, -1).mean(axis=1) >= 1e-3
            ins |= iq
        f = ins.reshape(bi.size, -1).mean(axis=1)
        fill[bi, bj] = f
        # bins by the shifted x coordinate: count per (bx, by), divided
        # by the samples that landed there
        bx = np.minimum((ox*ks).astype(int), ks - 1)          # (s, s)
        by = np.minimum((oy*ks).astype(int), ks - 1)
        flat = (bx*ks + by).ravel()
        cnt = np.bincount(flat, minlength=ks*ks).astype(float)
        sub = np.stack([np.bincount(flat, weights=r.ravel(),
                                    minlength=ks*ks)/cnt
                        for r in ins]).reshape(bi.size, ks, ks)
        for r, (i, j) in enumerate(zip(bi, bj)):
            if f[r] >= 1e-3:
                bins[(int(i), int(j))] = sub[r].astype(np.float64)
    for q in range(len(prims)):
        wq = whole[q]
        for i, j in zip(*np.nonzero(wq)):
            bins.setdefault((int(i), int(j)), np.ones((ks, ks)))
    # spans: primitives that overlap in the section must share a span
    # (the record is one bin pattern down the extrusion)
    for qa in range(len(prims)):
        for qb in range(qa + 1, len(prims)):
            if (prims[qa][2:] != prims[qb][2:]
                    and np.any(claim[qa] & claim[qb])):
                raise ValueError(
                    "section primitives %d and %d overlap in the section "
                    "but span different cells along the axis -- v1 keeps "
                    "one bin pattern per transverse cell" % (qa, qb))
    occ0 = np.asarray(m.struc()) > 0
    if m.fill is None:
        m.fill = occ0.astype(np.float64)
    claimed = np.zeros((n1, n2), dtype=bool)
    for q, (pieces, sig, a0, a1) in enumerate(prims):
        for i, j in zip(*np.nonzero(claim[q])):
            pos = [None]*3
            pos[axis] = slice(a0, a1)
            pos[t1], pos[t2] = int(i), int(j)
            pos = tuple(pos)
            blockfull = occ0[pos] & (m.fill[pos] >= 1.0)
            if np.all(blockfull):
                continue                      # a block owns it whole
            m.sigma[pos] = np.where(blockfull, m.sigma[pos],
                                    np.float32(sig))
            m.fill[pos] = np.where(blockfull, 1.0, fill[i, j])
            claimed[i, j] = True
    cells = {key: b for key, b in bins.items() if claimed[key]}
    if not any(b.min() < 1.0 - 1e-12 for b in cells.values()):
        if np.all(m.fill[np.asarray(m.struc()) > 0] >= 1.0):
            m.fill = None
        return None
    shapes = [p for pr in prims for p in pr[0]]
    # FACE fills for the in-plane orientations: the metal fraction of
    # the + face of each transverse cell along t1 and t2. A filament
    # through a tilted cut takes the conductance of the face it
    # crosses (VoxelModel.resistances): the half-cell series rule
    # charges a link between two cells of unequal fill an O(1) excess,
    # and on a 45-degree edge every link is such a pair (measured:
    # DC R ratio 1.029 staircase -> 1.034 with cell fills at 16 across,
    # and FIRST order). The face rule makes a uniform flow's
    # dissipation exact to O(h^2) at any angle. A face with a whole
    # cell on either side is whole.
    part = np.zeros((n1, n2), dtype=bool)
    for key in cells:
        part[key] = cells[key].min() < 1.0 - 1e-12
    o = (np.arange(s) + 0.5)/s
    faces = {}
    for a, (na, pa, pb) in ((t1, (n1, p1, p2)), (t2, (n2, p2, p1))):
        G = np.ones((n1, n2))
        nb = np.roll(part, -1, axis=0 if a == t1 else 1)
        both = part & nb
        if a == t1:
            both[-1, :] = False
        else:
            both[:, -1] = False
        for i, j in zip(*np.nonzero(both)):
            if a == t1:
                xs, ys = (i + 1.0)*p1, (j + o)*p2
            else:
                xs, ys = (i + o)*p1, (j + 1.0)*p2
            G[i, j] = float(inside(shapes, xs, ys).mean())
        faces[int(a)] = G
    return dict(kind='section', axis=int(axis), shapes=shapes,
                k=int(ks), cells=cells, faces=faces)
