# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Sub-cell enrichment: the geometry and kernel tables that every
beyond-voxel correction in SuperPEEC is built from.

Every such correction -- the cross-section skin engine, the thin-film
palette, surface-anchored subpixel modes, corner circulation modes and
the partial-cell inductance correction -- is one construction: a
filament is cut into SUB-PRISMS, weights map sub-prisms to unknowns
(one aggregate column plus net-zero mode columns), and exact
box-mutual tables over the sub-prisms are folded through those
weights. This module holds the two pieces that construction shares
and that used to exist three times over (docs/enrichment_plan.md,
phase 1):

* :class:`Split` -- how a filament of one orientation is cut. One
  vectorised box builder serves the k x k transverse split, the 1-D
  film and slab splits, the in-plane corner split, the undivided
  filament and the equipotential terminal bar.
* :class:`PairTables` -- raw sub-prism partial-mutual tables between
  two splits at a set of cell separations, cached by separation set.
  Geometry only; weight folding is the caller's, so a re-fold at a new
  frequency never re-evaluates a kernel.

:func:`partial_dL` is the first consumer written on them: the
partial-cell inductance correction (subpixel stage B) for cylinders
and axis-aligned slabs alike, expressed as the aggregate-aggregate
block of the fold, ``dL = w'Tw - u'Tu``.
"""
import warnings

import numpy as np
import scipy.sparse as sp

import sppeec_status as _spstatus


class Split:
    """Decomposition of one filament orientation into sub-prisms.

    A filament along ``axis`` at lattice cell ``c`` is the box
    ``[c, c+1)*d`` on the two transverse axes and, along its own axis,
    ``[c + axial[0], c + axial[1])*d`` -- centre to centre,
    ``(0.5, 1.5)``, by default; an equipotential terminal bar of length
    ``t_l`` at a low face is ``(0.5 - t_l/d, 0.5)`` and at a high face
    ``(0.5, 0.5 + t_l/d)``. ``n`` cuts each axis into that many equal
    pieces; sub-prisms are numbered in C order over ``n``, so a
    transverse ``(k0, k1)`` split lists them as ``p*k1 + q`` with ``p``
    on the lower-indexed transverse axis.
    """

    def __init__(self, axis, n=(1, 1, 1), d=None, axial=(0.5, 1.5),
                 offset=None):
        self.axis = int(axis)
        self.n = tuple(int(v) for v in n)
        self.d = tuple(float(v) for v in np.asarray(d, dtype=float))
        self.axial = (float(axial[0]), float(axial[1]))
        # ``offset``: axial extent as METRES from the cell centre instead
        # of pitch fractions (the terminal bar: (0, t_l) or (-t_l, 0)).
        self.offset = None if offset is None else (float(offset[0]),
                                                   float(offset[1]))
        if min(self.n) < 1:
            raise ValueError("every sub-count must be >= 1")
        self.nsub = self.n[0]*self.n[1]*self.n[2]
        oth = [c for c in range(3) if c != self.axis]
        # cross-section of ONE sub-prism, perpendicular to the current
        self.area = ((self.d[oth[0]]/self.n[oth[0]])
                     * (self.d[oth[1]]/self.n[oth[1]]))
        self.key = (self.axis, self.n, self.d, self.axial, self.offset)

    @classmethod
    def transverse(cls, axis, kk, d):
        """``kk[0] x kk[1]`` split over the two transverse axes, in
        increasing axis order (the skin engine's cross-section grid)."""
        n = [1, 1, 1]
        for t, k in zip([c for c in range(3) if c != int(axis)], kk):
            n[t] = int(k)
        return cls(axis, n, d)

    def terminal(self, t_l, sign):
        """The undivided terminal bar of length ``t_l`` at a face of
        the given sign, for filaments of this split's orientation."""
        off = (0.0, float(t_l)) if sign > 0 else (-float(t_l), 0.0)
        return Split(self.axis, (1, 1, 1), self.d, offset=off)

    def boxes(self, cells):
        """``(lo, hi)`` of every sub-prism of every cell, each of shape
        ``(ncell*nsub, 3)``, cell-major then C order over ``n``."""
        c = np.atleast_2d(np.asarray(cells, dtype=float))
        ncell = c.shape[0]
        g = np.stack(np.meshgrid(*[np.arange(v, dtype=float)
                                   for v in self.n], indexing='ij'),
                     axis=-1).reshape(-1, 3)
        lo = np.empty((ncell, self.nsub, 3))
        hi = np.empty((ncell, self.nsub, 3))
        for a in range(3):
            d, n = self.d[a], self.n[a]
            f0, f1 = (self.axial if a == self.axis else (0.0, 1.0))
            if a == self.axis and self.offset is not None:
                mid = ((c[:, a] + 0.5)*d)[:, None]
                lo[:, :, a] = mid + self.offset[0]
                hi[:, :, a] = mid + self.offset[1]
            elif n == 1:
                lo[:, :, a] = ((c[:, a] + f0)*d)[:, None]
                hi[:, :, a] = ((c[:, a] + f1)*d)[:, None]
            else:
                h = (f1 - f0)*d/n
                base = ((c[:, a] + f0)*d)[:, None]
                lo[:, :, a] = base + g[None, :, a]*h
                hi[:, :, a] = base + (g[None, :, a] + 1.0)*h
        return lo.reshape(-1, 3), hi.reshape(-1, 3)


class PairTables:
    """Raw sub-prism partial-mutual tables, cached by separation set.

    ``tables(sa, sb, D)`` is the ``(len(D), sa.nsub, sb.nsub)`` array of
    partial mutual inductances between sub-prism ``p`` of a filament at
    the origin cell (split ``sa``) and sub-prism ``q`` of a filament at
    cell offset ``D[i]`` (split ``sb``). The sub-prisms are congruent
    boxes on a lattice, so the coupling depends only on ``(separation,
    p, q)`` and a table over the separations that occur replaces one
    kernel evaluation per PAIR (22 million on a 23600-filament
    conductor). Both splits must share the axis: perpendicular bars
    have zero mutual partial inductance and no table.

    Instances are per solver, not global: at k = 12 a table over a
    (2*12+1)**3 stencil is ~2.6 GB, and it must die with the solve.
    """

    def __init__(self):
        self._cache = {}

    def __call__(self, sa, sb, D):
        if sa.axis != sb.axis:
            raise ValueError("box tables need parallel bars (axes %d, %d)"
                             % (sa.axis, sb.axis))
        D = np.asarray(D, dtype=np.int64).reshape(-1, 3)
        key = (sa.key, sb.key, D.tobytes())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        from greens import box_pair_stencil_pairs as spair
        na, nb = sa.nsub, sb.nsub
        nD = D.shape[0]
        lo0, hi0 = sa.boxes(np.zeros((1, 3), dtype=np.int64))
        T = np.empty((nD, na, nb))
        # CHUNKED: the pair arrays are (n*na*nb, 3), and an untruncated
        # table pushes nD*na*nb into the tens of millions -- unchunked
        # that is GBs of broadcast temporaries for a bounded answer.
        step = max(1, 2_000_000 // (na*nb))
        with _spstatus.task('mode tables',
                            ticks=(nD + step - 1)//step) as _t:
            for a in range(0, nD, step):
                Dc = D[a:a + step]
                n = Dc.shape[0]
                loD, hiD = sb.boxes(Dc)
                A_lo = np.broadcast_to(lo0[None, :, None, :], (n, na, nb, 3))
                A_hi = np.broadcast_to(hi0[None, :, None, :], (n, na, nb, 3))
                B_lo = np.broadcast_to(loD.reshape(n, 1, nb, 3),
                                       (n, na, nb, 3))
                B_hi = np.broadcast_to(hiD.reshape(n, 1, nb, 3),
                                       (n, na, nb, 3))
                S = spair(A_lo.reshape(-1, 3), A_hi.reshape(-1, 3),
                          B_lo.reshape(-1, 3), B_hi.reshape(-1, 3))
                T[a:a + step] = (S/(sa.area*sb.area)).reshape(n, na, nb)
                _t.tick()
        self._cache[key] = T
        return T


def unique_separations(D):
    """Distinct rows of the integer ``(n, 3)`` separations ``D`` and the
    per-row index into them."""
    D = np.asarray(D, dtype=np.int64)
    span = int(np.abs(D).max()) + 1 if D.size else 1
    w = 2*span + 1
    key = ((D[:, 0] + span)*w + (D[:, 1] + span))*w + D[:, 2] + span
    uniq, inv = np.unique(key, return_inverse=True)
    dz = uniq % w - span
    dy = (uniq // w) % w - span
    dx_ = uniq // (w*w) - span
    return np.stack([dx_, dy, dz], axis=1), inv


def neighbour_pairs(cells, radius, other=None):
    """Index pairs ``(a, b)`` of ``cells`` within ``radius`` (inf-norm).

    With ``other=None`` both orderings and the self pairs are returned,
    so a block assembled from them is symmetric without a second pass;
    with ``other`` given, pairs are ``(cells index, other index)``.
    """
    from scipy.spatial import cKDTree
    a = cKDTree(np.asarray(cells, dtype=float))
    if other is None:
        p = a.query_pairs(r=radius, p=np.inf, output_type='ndarray')
        self_i = np.arange(len(cells))
        return (np.concatenate([p[:, 0], p[:, 1], self_i]),
                np.concatenate([p[:, 1], p[:, 0], self_i]))
    b = cKDTree(np.asarray(other, dtype=float))
    pairs = a.query_ball_tree(b, r=radius, p=np.inf)
    if not any(pairs):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    fa = np.concatenate([np.full(len(v), i, dtype=np.int64)
                         for i, v in enumerate(pairs)])
    fb = np.concatenate([np.asarray(v, dtype=np.int64)
                         for v in pairs if v])
    return fa, fb


# ------------------------------------------------ partial-cell inductance

def slab_weights(fill, k):
    """1-D sub-prism current shares of a cell filled ``fill`` of its
    extent along the cut axis (uniform density over the clipped part),
    or ``None`` for a whole cell, where the correction is identically
    zero."""
    f = float(fill)
    if f >= 1.0 - 1e-12:
        return None
    edges = np.arange(k + 1)/k
    w = np.clip(np.minimum(edges[1:], f) - edges[:-1], 0.0, None)
    tot = w.sum()
    return None if tot <= 0.0 else w/tot


def partial_dL(model, M, window=2, tables=None, max_pairs=250_000_000):
    """Sparse partial-cell inductance correction (subpixel stage B).

    Stage A folds a partial cell's RESISTANCE into its material law;
    this is the matching INDUCTANCE, an additive correction over near
    pairs of parallel filaments

        dL_ij = w_i' T(sep) w_j  -  u' T(sep) u

    with ``T`` the exact box-mutual table over sub-prisms, ``w`` a
    cell's normalised sub-fill shares (uniform current density over the
    CLIPPED cross-section) and ``u`` the uniform whole-cell shares.
    Both terms come from the same kernel, so dL vanishes identically on
    whole cells and the Toeplitz far field is untouched -- the
    correction is local by construction, decaying like the
    shape-difference multipoles (measured on a z-cut at fill 0.5:
    -21.8%% of the pair at one cell, -12.1%% at two, -8.2%% at three,
    absolute 2.14 -> 0.60 -> 0.27 e-15 H, which is why a 2-cell window
    is enough).

    Two geometries, one formula: a ``[[cylinder]]`` model
    (``model.subpixel``) corrects the filaments ALONG the cylinder over
    its k x k transverse sub-fill bins; an axis-aligned slab cut
    (``model.slab_fill``) corrects the two IN-PLANE orientations over a
    1-D split along the cut. A filament running ACROSS a cut is a
    length problem, not a cross-section one, and is not corrected.

    The pair value depends only on (orientation, offset, w_i, w_j) and
    the weight vectors are few (a layer stack quantises fills to n/N;
    a cylinder has one vector per partial transverse cell), so the
    values are computed once per distinct triple and gathered. Every
    pair with AT LEAST ONE partial end is emitted: a partial cell
    couples to the whole slab under it and to the ground plane, none of
    which are partial, and pairing only partials recovered ~10%%.

    ``max_pairs`` bounds the OUTPUT (about 16 B per emitted pair while
    the COO arrays live); over it the function warns and returns
    ``None``, and the solve proceeds with stage A alone -- a deliberate
    degradation at a threshold that admits the RSFQ XNOR (~2.9 GB).

    An imposed complex skin PROFILE in ``w`` (stage C.1, the Kelvin
    Bessel average per sub-prism) was measured WORSE than this
    geometry-only correction -- the lattice already carries the
    between-cell phase evolution and imposing the intra-cell phase on
    top double-counts it. Enrichment amplitudes are SOLVED (net-zero
    modes over these same tables), never imposed.

    Returns a real symmetric CSR ``(efg, efg)`` matrix, or ``None``.
    """
    spx = getattr(model, 'subpixel', None)
    sf = getattr(model, 'slab_fill', None)
    if spx and spx['cells']:
        fams = [(int(spx['axis']), 'cylinder')]
    elif sf:
        fams = [(o, 'slab') for o in range(3) if o != int(sf['axis'])]
    else:
        return None
    from equiterminal import filament_cells
    fil_axis, fil_cell = filament_cells(M)
    d = np.asarray(model.d, dtype=float)
    dims = tuple(int(v) for v in model.dims)
    tables = PairTables() if tables is None else tables
    parts = []
    for orient, kind in fams:
        sel = np.flatnonzero(fil_axis == orient)
        if sel.size == 0:
            continue
        cells = np.asarray(fil_cell[sel], dtype=np.int64)
        if kind == 'cylinder':
            k = int(spx['k'])
            t1, t2 = [c for c in range(3) if c != orient]
            split = Split.transverse(orient, (k, k), d)
            # a filament's weights: its transverse cell's normalised
            # sub-fill bins (both end cells share them)
            W = np.full((cells.shape[0], k*k), 1.0/(k*k))
            partial = np.zeros(cells.shape[0], dtype=bool)
            for f, c in enumerate(cells):
                bins = spx['cells'].get((int(c[t1]), int(c[t2])))
                if bins is not None:
                    b = np.asarray(bins, dtype=float).ravel()
                    W[f] = b/b.sum()
                    partial[f] = True
        else:
            k = 8
            n = [1, 1, 1]
            n[int(sf['axis'])] = k
            split = Split(orient, n, d)
            fv = np.asarray(sf['fill'])[cells[:, 0], cells[:, 1],
                                        cells[:, 2]]
            W = np.full((cells.shape[0], k), 1.0/k)
            partial = np.zeros(cells.shape[0], dtype=bool)
            for f, fill in enumerate(fv):
                w = slab_weights(fill, k)
                if w is not None:
                    W[f] = w
                    partial[f] = True
        if not partial.any():
            continue
        cand = int(partial.sum())*(2*int(window) + 1)**3
        if cand > int(max_pairs):
            warnings.warn(
                "subpixel stage B skipped: %d partial filaments give "
                "%.1e pair candidates, over the %.1e cap. The solve keeps "
                "stage A (the material law); raise max_pairs deliberately "
                "if you can pay for it." % (int(partial.sum()), cand,
                                            max_pairs),
                RuntimeWarning, stacklevel=2)
            return None
        parts += _pair_correction(sel, cells, W, partial, split, window,
                                  dims, tables)
    if not parts:
        return None
    n = fil_axis.size
    rows = np.concatenate([p[0] for p in parts])
    cols = np.concatenate([p[1] for p in parts])
    vals = np.concatenate([p[2] for p in parts])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _pair_correction(sel, cells, W, partial, split, window, dims, tables):
    """COO parts of ``w_i'T w_j - u'T u`` over the offset cube.

    Only the lexicographically non-negative half of the cube is
    tabulated; each pair is emitted in both orderings with the SAME
    value, so the result is exactly symmetric and the zero offset
    (self pairs) is emitted once."""
    nf = cells.shape[0]
    uw, inv = np.unique(W, axis=0, return_inverse=True)
    inv = inv.ravel()
    u = np.full(split.nsub, 1.0/split.nsub)
    rng = np.arange(-int(window), int(window) + 1)
    offs = np.stack(np.meshgrid(rng, rng, rng, indexing='ij'),
                    axis=-1).reshape(-1, 3)
    offs = offs[np.lexsort(offs.T[::-1])]
    offs = offs[len(offs)//2:]                 # (0,0,0) and the half after
    T = tables(split, split, offs)
    V = (np.einsum('ap,opq,bq->oab', uw, T, uw)
         - np.einsum('p,opq,q->o', u, T, u)[:, None, None])
    loc = np.full(dims, -1, dtype=np.int64)
    loc[cells[:, 0], cells[:, 1], cells[:, 2]] = np.arange(nf)
    out = []
    for o, off in enumerate(offs):
        c = cells + off
        ok = np.all((c >= 0) & (c < np.asarray(dims)), axis=1)
        j = np.full(nf, -1, dtype=np.int64)
        j[ok] = loc[c[ok, 0], c[ok, 1], c[ok, 2]]
        ii = np.flatnonzero(j >= 0)
        jj = j[ii]
        keep = partial[ii] | partial[jj]
        ii, jj = ii[keep], jj[keep]
        v = V[o, inv[ii], inv[jj]]
        nz = v != 0.0
        if not nz.any():
            continue
        ri, rj, rv = sel[ii[nz]].astype(np.int32), sel[jj[nz]].astype(np.int32), v[nz]
        out.append((ri, rj, rv))
        if np.any(off):
            out.append((rj, ri, rv))
    return out
