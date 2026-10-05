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
import os

import numpy as np

S = 64          # samples per cell axis on boundary cells (16 per bin at k=4)
CLAIM_MIN = 1e-3   # a primitive claims a boundary cell from this fill up
SLIVER = float(os.environ.get('SPPEEC_SLIVER', '0.05'))
                   # floor on the fill along the cut axis (the record's
                   # 'axial_floor', read by VoxelModel.impedance_scale;
                   # 0 disables; the env is a study override)
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


def _bbox(sh):
    """Axis-aligned bounding box (x0, y0, x1, y1) of one piece."""
    if sh[0] == 'circle':
        _, c1, c2, R = sh
        return c1 - R, c2 - R, c1 + R, c2 + R
    v = np.asarray(sh[1], dtype=float)
    return v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()


def _boxes_meet(sh, bx0, by0, bx1, by1):
    """Rows whose axis-aligned box ``[bx0, bx1] x [by0, by1]`` can meet
    piece ``sh``: bounding boxes overlap AND (for a polygon) no edge of
    the piece has all four box corners strictly outside it -- the
    separating-axis test, exact for a convex piece against a box. A
    small guard keeps a sample that rounds onto an edge in. The long
    45-degree sides of a spiral have bounding boxes spanning several
    turns; this keeps only the rows along the side itself."""
    x0, y0, x1, y1 = _bbox(sh)
    hit = (bx1 >= x0) & (bx0 <= x1) & (by1 >= y0) & (by0 <= y1)
    if sh[0] != 'poly' or not hit.any():
        return np.flatnonzero(hit)
    idx = np.flatnonzero(hit)
    v = np.asarray(sh[1], dtype=float)
    e = np.roll(v, -1, axis=0) - v
    el = np.hypot(e[:, 0], e[:, 1])
    nx, ny = e[:, 1]/el, -e[:, 0]/el          # outward (ccw poly)
    guard = 1e-9*max(x1 - x0, y1 - y0, 1e-30)
    ax0, ay0, ax1, ay1 = bx0[idx], by0[idx], bx1[idx], by1[idx]
    keep = np.ones(idx.size, dtype=bool)
    for k in range(len(v)):
        # the corner most INSIDE edge k (largest signed value sk)
        cx = np.where(nx[k] > 0, ax0, ax1)
        cy = np.where(ny[k] > 0, ay0, ay1)
        sk = -((cx - v[k, 0])*nx[k] + (cy - v[k, 1])*ny[k])
        keep &= sk >= -guard
    return idx[keep]


def _piece(sh, x, y, shape):
    """Signed distance (> 0 inside) and outward-normal angle of ONE
    piece at the points (x, y), broadcast to ``shape``."""
    if sh[0] == 'circle':
        _, c1, c2, R = sh
        dxp, dyp = x - c1, y - c2
        return R - np.hypot(dxp, dyp), np.arctan2(dyp, dxp)
    v = np.asarray(sh[1], dtype=float)
    e = np.roll(v, -1, axis=0) - v
    el = np.hypot(e[:, 0], e[:, 1])
    nx, ny = e[:, 1]/el, -e[:, 0]/el          # outward (ccw poly)
    ds = np.full(shape, np.inf)
    ps = np.zeros(shape)
    for k in range(len(v)):
        sk = -((x - v[k, 0])*nx[k] + (y - v[k, 1])*ny[k])
        take = sk < ds
        ds = np.where(take, sk, ds)
        ps = np.where(take, np.arctan2(ny[k], nx[k]), ps)
    return ds, ps


# below this many point-piece pairs the plain all-pairs pass is cheaper
# than sorting the points
_CULL_MIN = 1 << 21


def field(shapes, x, y, reach=None):
    """Signed distance to the union boundary (> 0 inside) and the
    outward-normal angle of the piece that sets it, at the points
    ``(x, y)`` (broadcast). Exact inside every convex piece; outside,
    a lower bound on the distance.

    ``reach`` CULLS (2026-10-04): a piece is evaluated only at points
    within ``reach`` of its bounding box. Wherever the true distance is
    >= -reach the result is bit-identical to the all-pairs pass (the
    winning piece is never culled, every piece's arithmetic is the
    same, and a culled piece could only have lost); farther out the
    distance may read -inf. ``None`` keeps the all-pairs pass. At the
    IHP spiral's 0.25 um rung the all-pairs pass was ~160 pieces x
    ~5e8 sample points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = np.broadcast(x, y).shape
    d = np.full(shape, -np.inf)
    phi = np.zeros_like(d)
    n = int(np.prod(shape))
    if reach is None or n*len(shapes) < _CULL_MIN:
        for sh in shapes:
            ds, ps = _piece(sh, x, y, shape)
            take = ds > d
            d = np.where(take, ds, d)
            phi = np.where(take, ps, phi)
        return d, phi
    xf = np.broadcast_to(x, shape).ravel()
    yf = np.broadcast_to(y, shape).ravel()
    df, pf = d.ravel(), phi.ravel()
    order = np.argsort(xf, kind='stable')
    xs = xf[order]
    r = float(reach)
    for sh in shapes:
        x0, y0, x1, y1 = _bbox(sh)
        lo = np.searchsorted(xs, x0 - r, side='left')
        hi = np.searchsorted(xs, x1 + r, side='right')
        if hi <= lo:
            continue
        cand = order[lo:hi]
        yc = yf[cand]
        sel = cand[(yc >= y0 - r) & (yc <= y1 + r)]
        if sel.size == 0:
            continue
        ds, ps = _piece(sh, xf[sel], yf[sel], sel.shape)
        take = ds > df[sel]
        df[sel] = np.where(take, ds, df[sel])
        pf[sel] = np.where(take, ps, pf[sel])
    return df.reshape(shape), pf.reshape(shape)


def inside(shapes, x, y):
    """Point in the union. Culled at reach 0: a point inside a convex
    piece is inside its bounding box, so the answer is exact."""
    return field(shapes, x, y, reach=1e-12)[0] > 0.0


def _field_grid(pieces, xc, yc, reach):
    """``field`` on the cell-centre grid ``xc[:, None], yc[None, :]``,
    each piece evaluated on the index block its bounding box (plus
    ``reach``) covers -- pure slicing, the same per-cell arithmetic as
    the all-pairs pass, so the same values wherever the distance is
    >= -reach (see :func:`field`)."""
    d = np.full((xc.size, yc.size), -np.inf)
    for sh in pieces:
        x0, y0, x1, y1 = _bbox(sh)
        i0 = int(np.searchsorted(xc, x0 - reach, side='left'))
        i1 = int(np.searchsorted(xc, x1 + reach, side='right'))
        j0 = int(np.searchsorted(yc, y0 - reach, side='left'))
        j1 = int(np.searchsorted(yc, y1 + reach, side='right'))
        if i1 <= i0 or j1 <= j0:
            continue
        X = xc[i0:i1, None]
        Y = yc[None, j0:j1]
        ds, _ = _piece(sh, X, Y, (i1 - i0, j1 - j0))
        blk = d[i0:i1, j0:j1]
        d[i0:i1, j0:j1] = np.where(ds > blk, ds, blk)
    return d


def _inside_cells(pieces, ci, cj, ox, oy, p1, p2):
    """Inside-the-union of the ``s x s`` samples of cells ``(ci, cj)``
    (``(n, s, s)`` bool): each piece is sampled only on the cells whose
    square meets its bounding box -- a sample outside that box cannot
    be inside the piece, so the answer is exact."""
    ins = np.zeros((ci.size,) + ox.shape, dtype=bool)
    cx0, cy0 = ci*p1, cj*p2
    cx1, cy1 = (ci + 1)*p1, (cj + 1)*p2
    for sh in pieces:
        hit = _boxes_meet(sh, cx0, cy0, cx1, cy1)
        if hit.size == 0:
            continue
        xs = (ci[hit, None, None] + ox[None, :, :])*p1
        ys = (cj[hit, None, None] + oy[None, :, :])*p2
        ds, _ = _piece(sh, xs, ys, xs.shape)
        ins[hit] |= ds > 0.0
    return ins


def inside_rows(shapes, xs, ys, box):
    """Inside-the-union of sample rows ``xs[r], ys[r]`` (arrays of shape
    ``(n, ...)``) whose samples all lie in ``box[r] = (x0, y0, x1, y1)``:
    each piece is evaluated only on the rows whose box meets its own --
    exact, a sample outside a piece's box is outside the piece."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    shape = np.broadcast(xs, ys).shape
    xs = np.broadcast_to(xs, shape)
    ys = np.broadcast_to(ys, shape)
    ins = np.zeros(shape, dtype=bool)
    bx0, by0, bx1, by1 = (np.asarray(box)[:, c] for c in range(4))
    for sh in shapes:
        hit = _boxes_meet(sh, bx0, by0, bx1, by1)
        if hit.size == 0:
            continue
        ds, _ = _piece(sh, xs[hit], ys[hit], (hit.size,) + shape[1:])
        ins[hit] |= ds > 0.0
    return ins


def partial_map(cut):
    """``(pmap, bins)`` of a section cut's PARTIAL cells: ``pmap`` the
    (n1, n2) row index of each transverse cell (-1 where the cell is
    whole or not metal), ``bins`` the (n_partial, k, k) sub-fill bins in
    that row order. Built once per cut and cached on it -- readers that
    asked ``cut['cells']`` one filament at a time (a Python loop over
    ~10^8 filaments at the IHP spiral's 0.25 um rung) gather instead."""
    got = cut.get('_pmap')
    if got is not None:
        return got
    n1, n2 = next(iter(cut['faces'].values())).shape
    keys = [key for key, b in cut['cells'].items() if b.min() < 1.0 - 1e-12]
    k = int(cut['k'])
    pmap = np.full((n1, n2), -1, dtype=np.int64)
    bins = np.empty((len(keys), k, k))
    for r, key in enumerate(keys):
        pmap[key] = r
        bins[r] = cut['cells'][key]
    cut['_pmap'] = (pmap, bins)
    return cut['_pmap']


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
    whole = np.zeros((len(prims), n1, n2), dtype=bool)
    bnd = np.zeros((n1, n2), dtype=bool)
    # SCALE (2026-10-04): every pass below is culled per piece -- a
    # piece only touches the points near its own bounding box -- and
    # vectorised; the results are bit-identical to the all-pairs form
    # (git 04b7538), which spent >10 min and 10 GB on the 1 um IHP
    # spiral (160 pieces, 29 k boundary cells x 4096 samples).
    for q, (pieces, _, _, _) in enumerate(prims):
        # reach `half`: the tests are dc >= half and dc > -half
        dc = _field_grid(pieces, xc, yc, half*(1 + 1e-9))
        whole[q] = dc >= half
        bnd |= (dc > -half) & ~whole[q]
        del dc
    bnd &= ~whole.any(axis=0)
    # sample the boundary cells: fill and bins of the UNION, and each
    # primitive's own claim -- in chunks of cells (the sample cloud of
    # every boundary cell at once is ~4 GB per coordinate at 0.25 um)
    ox, oy = lattice(s)
    bi, bj = np.nonzero(bnd)
    claim = whole.copy()
    fill = whole.any(axis=0).astype(np.float64)
    bins = {}
    if bi.size:
        bx = np.minimum((ox*ks).astype(int), ks - 1)          # (s, s)
        by = np.minimum((oy*ks).astype(int), ks - 1)
        flat = (bx*ks + by).ravel()
        cnt = np.bincount(flat, minlength=ks*ks).astype(float)
        # sample -> bin incidence: the per-bin sums of 0/1 samples are
        # small integers, exact in any summation order
        onehot = np.zeros((s*s, ks*ks))
        onehot[np.arange(s*s), flat] = 1.0
        f = np.empty(bi.size)
        sub = np.empty((bi.size, ks, ks))
        CH = 4096
        for c0 in range(0, bi.size, CH):
            ci, cj = bi[c0:c0 + CH], bj[c0:c0 + CH]
            ins = np.zeros((ci.size, s, s), dtype=bool)
            for q, (pieces, _, _, _) in enumerate(prims):
                iq = _inside_cells(pieces, ci, cj, ox, oy, p1, p2)
                claim[q, ci, cj] = iq.reshape(ci.size, -1).mean(axis=1) >= CLAIM_MIN
                ins |= iq
            insf = ins.reshape(ci.size, -1)
            f[c0:c0 + ci.size] = insf.mean(axis=1)
            sub[c0:c0 + ci.size] = ((insf.astype(np.float64) @ onehot)
                                    / cnt).reshape(ci.size, ks, ks)
            del ins, insf
        fill[bi, bj] = f
        for r in np.flatnonzero(f >= CLAIM_MIN):
            bins[(int(bi[r]), int(bj[r]))] = sub[r]
        bpart = np.zeros((n1, n2), dtype=bool)
        bpart[bi, bj] = (f >= CLAIM_MIN) & (f < 1.0)
    else:
        bpart = np.zeros((n1, n2), dtype=bool)
    # whole cells share ONE read-only all-ones pattern (a fresh array per
    # whole cell was ~3.5 M arrays at 0.25 um); readers never write bins
    ones = np.ones((ks, ks))
    ones.flags.writeable = False
    for q in range(len(prims)):
        for i, j in zip(*np.nonzero(whole[q])):
            bins.setdefault((int(i), int(j)), ones)
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
        ii, jj = np.nonzero(claim[q])
        if ii.size == 0:
            continue
        # every claimed column at once: index arrays (column, span)
        ax = np.arange(a0, a1)
        idx = [None]*3
        idx[axis] = ax[None, :]
        idx[t1], idx[t2] = ii[:, None], jj[:, None]
        idx = tuple(np.broadcast_arrays(*idx))
        blockfull = occ0[idx] & (m.fill[idx] >= 1.0)
        keep = ~np.all(blockfull, axis=1)        # a block owns it whole
        if not keep.any():
            continue
        sel = tuple(v[keep] for v in idx)
        bf = blockfull[keep]
        m.sigma[sel] = np.where(bf, m.sigma[sel], np.float32(sig))
        m.fill[sel] = np.where(bf, 1.0, fill[ii[keep], jj[keep]][:, None])
        claimed[ii[keep], jj[keep]] = True
    cells = {key: b for key, b in bins.items() if claimed[key]}
    part = bpart & claimed
    if not part.any():
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
    o = (np.arange(s) + 0.5)/s
    faces = {}
    for a in (t1, t2):
        G = np.ones((n1, n2))
        nb = np.roll(part, -1, axis=0 if a == t1 else 1)
        both = part & nb
        if a == t1:
            both[-1, :] = False
        else:
            both[:, -1] = False
        fi, fj = np.nonzero(both)
        if fi.size:
            if a == t1:
                xs = np.broadcast_to(((fi + 1.0)*p1)[:, None], (fi.size, s))
                ys = (fj[:, None] + o[None, :])*p2
            else:
                xs = (fi[:, None] + o[None, :])*p1
                ys = np.broadcast_to(((fj + 1.0)*p2)[:, None], (fi.size, s))
            G[fi, fj] = inside(shapes, xs, ys).mean(axis=1)
        faces[int(a)] = G
    return dict(kind='section', axis=int(axis), shapes=shapes,
                k=int(ks), cells=cells, faces=faces, axial_floor=SLIVER)
