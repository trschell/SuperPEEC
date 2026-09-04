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
    this is the matching INDUCTANCE, over near pairs of parallel
    filaments::

        dL_ij = w_i' T(sep) w_j  -  u' T(sep) u

    with ``T`` the exact box-mutual table over sub-prisms, ``w`` the
    cell's normalised sub-fill shares (uniform density over the CLIPPED
    cross-section) and ``u`` the whole-cell shares. Both terms come
    from the same kernel, so dL vanishes on whole cells and the
    Toeplitz far field is untouched; the correction is local (decay
    measured in docs/enrichment_history.md, a 2-cell window suffices).
    A ``[[cylinder]]`` model corrects the filaments ALONG the cylinder
    over its k x k sub-fill bins; an axis-aligned slab cut corrects the
    two IN-PLANE orientations over a 1-D split. Values depend only on
    (orientation, offset, w_i, w_j) and the weight vectors are few, so
    they are computed once per distinct triple and gathered; every pair
    with at least one partial end is emitted. ``max_pairs`` bounds the
    OUTPUT (~16 B per pair); over it the function warns and returns
    ``None`` and the solve keeps stage A. An imposed skin profile in
    ``w`` measured WORSE (the lattice already carries the between-cell
    phase; enrichment amplitudes are solved, never imposed).

    Returns a real symmetric CSR ``(efg, efg)`` matrix, or ``None``.
    """
    cut = getattr(model, 'cut', None)
    if cut is None:
        return None
    if cut['kind'] == 'cylinder':
        fams = [(int(cut['axis']), 'cylinder')]
    else:
        fams = [(o, 'slab') for o in range(3) if o != int(cut['axis'])]
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
            k = int(cut['k'])
            t1, t2 = [c for c in range(3) if c != orient]
            split = Split.transverse(orient, (k, k), d)
            # a filament's weights: its transverse cell's normalised
            # sub-fill bins (both end cells share them)
            W = np.full((cells.shape[0], k*k), 1.0/(k*k))
            partial = np.zeros(cells.shape[0], dtype=bool)
            for f, c in enumerate(cells):
                bins = cut['cells'].get((int(c[t1]), int(c[t2])))
                if bins is not None:
                    b = np.asarray(bins, dtype=float).ravel()
                    W[f] = b/b.sum()
                    partial[f] = True
        else:
            k = 8
            n = [1, 1, 1]
            n[int(cut['axis'])] = k
            split = Split(orient, n, d)
            fv = np.asarray(model.fill)[cells[:, 0], cells[:, 1], cells[:, 2]]
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


# ------------------------------------------------------------------ material

# ------------------------------------------------------------------ palettes

def netzero_prune(cols, support=None, tol=1e-7):
    """Net-zero, normalised, pivoted-QR-pruned columns over ``support``.

    Each candidate column is mean-subtracted over the support (the
    load-bearing invariant: zero net current => zero incidence =>
    invisible to KCL and to the nodal Schur complement), normalised,
    and dropped if it vanishes or lies within ``tol`` of the span of
    the others -- a dependent column would make the mode resistance
    block ``W'W`` singular. ORIGINAL columns are kept, never mixed, so
    the result is deterministic and readable; pivoted QR orders them
    by importance and the first ``min(m, n)`` pivots are the only ones
    that can be independent. Returns ``(W, kept)`` with ``W`` of shape
    ``(m, n)`` holding zeros in the dropped columns and ``kept`` the
    column mask -- callers that want a compact basis take ``W[:, kept]``.
    """
    from scipy.linalg import qr
    C = np.asarray(cols, dtype=float)
    m, n = C.shape
    sup = np.ones(m, dtype=bool) if support is None else support
    W = np.zeros((m, n))
    nrm = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for j in range(n):          # per column: symmetric shapes tie in norm,
        w = C[sup, j]           # and the pivot order must not depend on
        W[sup, j] = w - w.mean()    # the last bit of a 2-D reduction
        nrm[j] = np.linalg.norm(W[:, j])
        ok[j] = nrm[j] > 1e-12*max(np.linalg.norm(w), 1.0)
    kept = np.zeros(n, dtype=bool)
    if ok.any():
        W[:, ok] /= nrm[ok]
        _, R, piv = qr(W[:, ok], mode='economic', pivoting=True)
        d = np.abs(np.diag(R))
        kept[np.flatnonzero(ok)[np.sort(piv[:d.size][d > tol*d[0]])]] = True
        W[:, ~kept] = 0.0
        # exact net-zero over the support after pruning
        W[np.ix_(sup, kept)] -= W[np.ix_(sup, kept)].mean(axis=0,
                                                          keepdims=True)
    return W, kept


def conduction_weights(kk, d, p):
    """Net-zero CONDUCTION-MODE weights on the ``kk[0] x kk[1]`` sub-bar
    grid of a ``d[0] x d[1]`` cross-section, at Helmholtz rate ``p``.

    Daniel, Sangiovanni-Vincentelli & White (EPEP 2000): inside a
    conductor the current solves a Helmholtz equation whose solutions
    are exponentials anchored to the cross-section's faces (rate ``p``)
    and corners (rate ``p/sqrt(2)``, since ``exp(-a(x+y))`` has
    ``grad^2 = 2a^2``). The four faces and four corners enter as
    INDIVIDUAL complex shapes (the corner-parity split measured +6
    delivered points at dx/delta 3-6), each as two REAL columns --
    ``build_fft`` needs a real W, and the real pair spans strictly more
    than one complex column. The shapes are anchored to the CELL faces,
    not the conductor surface: every filament must carry identical
    weights or the Toeplitz/FFT structure of the tables is lost, and
    it costs nothing in span (a face exponential restricted to an
    interior cell is a combination of that cell's own face
    exponentials plus a constant, and the constant IS the aggregate).

    Corner columns only under a genuinely 2-D split: under a 1-D split
    (the thin-film palette, ``kk = (1, kz)``) a "corner" shape is a
    face exponential at a slightly different rate, and keeping it makes
    the translation-invariant mode operator NEARLY SINGULAR (symbol
    condition 2.7e12; the ~2000-matvec film stall no preconditioner
    could fix) -- the per-cell prune cannot see a degeneracy that only
    appears ACROSS cells.
    """
    k0, k1 = kk
    d0, d1 = float(d[0]), float(d[1])
    U, V = np.meshgrid((np.arange(k0) + 0.5)/k0, (np.arange(k1) + 0.5)/k1,
                       indexing='ij')                  # matches Split.boxes
    x, y = U.ravel()*d0, V.ravel()*d1
    p = complex(p)
    shapes = [np.exp(-p*x), np.exp(-p*(d0 - x)),
              np.exp(-p*y), np.exp(-p*(d1 - y))]
    if min(k0, k1) > 1:
        pc = p/np.sqrt(2.0)
        shapes += [np.exp(-pc*(x + y)), np.exp(-pc*(x + (d1 - y))),
                   np.exp(-pc*((d0 - x) + y)),
                   np.exp(-pc*((d0 - x) + (d1 - y)))]
    cols = np.stack([part for c in shapes for part in (c.real, c.imag)],
                    axis=1)
    W, kept = netzero_prune(cols)
    if not kept.any():
        raise ValueError("conduction modes are all constant at this rate "
                         "-- nothing to redistribute; run without modes")
    return W[:, kept]


def _surface_geometry(spx, cells, tr, kk, dx):
    """Per transverse cell of a cylinder model: sub-prism fills at the
    ENGINE subdivision (resampled from the resolved circle), signed
    distance to the surface and azimuth at the centroids. Shared down
    the extrusion. ``None`` for a conductor cell outside every cylinder
    (it couples as an aggregate and carries no modes)."""
    t1, t2 = tr
    tkey = [(int(a), int(b)) for a, b in zip(cells[:, t1], cells[:, t2])]
    k0, k1 = kk
    h0, h1 = dx/k0, dx/k1
    ns = 8                                      # samples per sub-prism axis
    u1 = (np.arange(k0*ns) + 0.5)*(h0/ns)
    u2 = (np.arange(k1*ns) + 0.5)*(h1/ns)
    cu = (np.arange(k0) + 0.5)*h0
    cv = (np.arange(k1) + 0.5)*h1
    percell = {}
    for key in set(tkey):
        g = spx['geom'].get(key)
        if g is None:
            percell[key] = None
            continue
        c1, c2, R, _ = g
        x0, y0 = key[0]*dx, key[1]*dx
        ins = ((x0 + u1[:, None] - c1)**2
               + (y0 + u2[None, :] - c2)**2) <= R*R
        XC, YC = np.meshgrid(x0 + cu, y0 + cv, indexing='ij')
        rho = np.hypot(XC.ravel() - c1, YC.ravel() - c2)
        percell[key] = dict(
            fill=ins.reshape(k0, ns, k1, ns).mean(axis=(1, 3)).ravel(),
            d=R - rho,                          # signed: > 0 inside
            phi=np.arctan2(YC.ravel() - c2, XC.ravel() - c1))
    return tkey, percell


SURFACE_KM = 6      # 3 complex shapes (skin, 2x proximity) x (re, im)


def surface_weights(percell, tkey, k, p):
    """Per-cell weight stack ``(nfil, k, SURFACE_KM)`` and column mask:
    the surface exponential ``exp(-p d)`` (d = distance to the resolved
    surface) and its two azimuthal (proximity) partners, restricted to
    the supported sub-prisms and weighted by their fill -- physically a
    sliver carries a sliver's current, numerically an O(1) mode current
    through a 1/fill resistance would make Ru arbitrarily ill
    conditioned (measured as a stalled Krylov solve). A pruned column
    is zeroed and MASKED rather than dropped, so km stays global."""
    Wu, mu = {}, {}
    for key, pc in percell.items():
        if pc is None:
            Wu[key] = np.zeros((k, SURFACE_KM))
            mu[key] = np.zeros(SURFACE_KM, dtype=bool)
            continue
        sup = pc['fill'] > 1e-3
        c = np.exp(-p*np.maximum(pc['d'], 0.0))*pc['fill']
        cols = np.stack([part for sh in (c, c*np.cos(pc['phi']),
                                         c*np.sin(pc['phi']))
                         for part in (sh.real, sh.imag)], axis=1)
        Wu[key], mu[key] = netzero_prune(cols, support=sup)
    Wf = np.stack([Wu[key] for key in tkey])
    cmask = np.stack([mu[key] for key in tkey])
    return Wf, cmask


def _near_surface(occ, cells, tr, reach):
    """Filaments within ``reach`` transverse 4-neighbour steps of an
    empty cell or the box boundary (reach 0 = an exposed face)."""
    e = np.pad(~occ, 1, constant_values=True)
    for _ in range(int(reach) + 1):
        g = e.copy()
        for t in tr:
            g |= np.roll(e, 1, axis=t) | np.roll(e, -1, axis=t)
        e = g
    c = np.asarray(cells) + 1
    return e[c[:, 0], c[:, 1], c[:, 2]]


class ConductionPalette:
    """Shared weights: :func:`conduction_weights` on every filament of
    one orientation; the rate moves with frequency."""
    shared = True
    moves = True

    def __init__(self, kk, dt):
        self.kk, self.dt = kk, dt

    def weights(self, p):
        return conduction_weights(self.kk, self.dt, p)


class SurfacePalette:
    """Per-cell weights anchored to a resolved cylinder surface
    (:func:`surface_weights`); fill-share aggregates ``G`` and
    fill-weighted sub-bar impedance factors ``rfac``; placement within
    ``reach`` cells of the surface layer."""
    shared = False
    moves = True

    def __init__(self, spx, cells, tr, kk, dx, reach):
        self.tkey, self.percell = _surface_geometry(spx, cells, tr, kk, dx)
        k = kk[0]*kk[1]
        n = len(self.tkey)
        self.k = k
        self.G = np.full((n, k), 1.0/k)
        self.rfac = np.ones((n, k))
        self.bnd = np.zeros(n, dtype=bool)
        lim = (float(reach) + 1.0)*dx if reach is not None else np.inf
        for f, key in enumerate(self.tkey):
            pc = self.percell[key]
            if pc is None:
                continue
            sup = pc['fill'] > 1e-3
            tot = pc['fill'].sum()
            self.G[f] = pc['fill']/tot if tot > 0 else 0.0
            self.rfac[f] = np.where(sup, 1.0/np.where(sup, pc['fill'], 1.0),
                                    0.0)
            self.bnd[f] = bool(np.any(np.abs(pc['d'][sup]) <= lim))

    def weights(self, p):
        Wf, cmask = surface_weights(self.percell, self.tkey, self.k, p)
        return Wf, cmask & self.bnd[:, None]


class FixedPalette:
    """Frequency-independent per-entry weights tied into PATCH modes.

    ``sel`` lists the filament of each entry (a filament may appear
    under several patches), ``Wf`` is ``(nentry, k, kmf)`` with the
    patch fields restricted to each entry's sub-bars, and ``P`` is the
    ``(nentry*kmf, nmode)`` 0/1 prolongation that makes entry column
    ``m`` of every entry of patch ``c`` the SAME unknown: the blocks a
    family assembles per entry are folded as ``P' Z P``. ``groups``
    are the mode index ranges per patch, the block-Jacobi cells."""
    shared = False
    moves = False

    def __init__(self, sel, Wf, P, groups):
        self.sel = np.asarray(sel, dtype=np.int64)
        self.Wf = np.asarray(Wf)
        self.P = sp.csr_matrix(P)
        self.groups = groups

    def weights(self, p):
        return self.Wf, np.ones(self.Wf.shape[:1] + self.Wf.shape[2:],
                                dtype=bool)


# --------------------------------------------------------------- the family

class Enrichment:
    """Net-zero sub-cell redistribution modes on a set of filaments.

    Each filament ENTRY (a filament under one palette; the same
    filament may be entered under several corner patches) is cut into
    ``kk[0] x kk[1]`` sub-bars spanning THE SAME TWO NODES and the
    basis is changed to (aggregate, redistribution)::

        i_p = I*g_p + sum_m W_pm u_m ,      sum_p W_pm = 0

    The aggregate ``I`` is exactly the original filament (``g``
    uniform, or the fill shares on a partial cell): ``W_agg' Z_sub
    W_agg == Z_full`` (validate_enrich D), so the Toeplitz near field
    and the FMM far field are untouched. The modes carry ZERO net
    current, hence zero incidence: no new nodes, invisible to KCL and
    to the nodal Schur complement. Each mode is the mesh current of the
    loop formed by two parallel branches; at DC the branches are
    identical and the modes vanish, at AC they couple differently to
    the rest of the structure -- that asymmetry IS skin, proximity,
    London screening and current turning at a bend. Net-zero modes are
    dipoles (mode-aggregate 1/r^2, mode-mode 1/r^3), so the blocks are
    truncated at ``rc = (rc_uu, rc_cross)``.

    Blocks in the augmented system ``[i_f; i_t; u]``: ``Ru`` (sub-bar
    series impedance ``z*l/a`` through W: ``1/sigma`` for a normal
    conductor, ``j w mu lambda^2`` for a London one), ``Zuu``,
    ``Zcross`` (modes <-> the aggregates ``agg``), ``Zt`` (terminals).
    Three palettes, one fold: :class:`ConductionPalette` (shared W on
    one orientation -> Toeplitz tables, FFT apply; ``kk = (1, kz)`` is
    the thin-film palette), :class:`SurfacePalette` (per-cell weights
    on a resolved cylinder, sparse), :class:`FixedPalette` (tabulated
    corner fields on both in-plane orientations, tied into 3 unknowns
    per patch by ``P``, sparse). Placement: entries within ``reach``
    cells of the conductor surface (``None``: everywhere; interior
    modes over-crowd, docs/enrichment_history.md). ``set_frequency``
    recomputes ``(p, z)``, regenerates the weights only if the
    palette's rate moved and re-folds the cached tables either way;
    True means ``nmode`` may have changed.
    """

    def __init__(self, model, M, axis, fil_axis, fil_cell, kk, term=None,
                 rc=(3, 4), reach=0, use_fft=None, csr_max_gb=2.0,
                 freq=None, palette=None):
        kk = ((int(kk), int(kk)) if np.isscalar(kk)
              else tuple(int(v) for v in kk))
        if min(kk) < 1 or max(kk) < 2:
            raise ValueError("kk must give at least two sub-filaments")
        self.kk = kk
        d3 = np.asarray(model.d, dtype=float)
        self.d3 = d3
        fil_axis = np.asarray(fil_axis)
        fil_cell = np.asarray(fil_cell)
        self._model = model
        self.freq = float(freq) if freq else 0.0
        self._p, self._z = model.material_response(self.freq)
        cut = getattr(model, 'cut', None)
        spx = cut if cut is not None and cut['kind'] == 'cylinder' else None
        if palette is None:
            self.axis = int(axis)
            self.sel = np.flatnonzero(fil_axis == self.axis)
            self.cells = fil_cell[self.sel]
            tr = [c for c in range(3) if c != self.axis]
            if spx is not None:
                if int(spx['axis']) != self.axis:
                    raise NotImplementedError(
                        "surface modes: the terminal axis (%d) differs "
                        "from the cylinder axis (%d) -- transverse mode "
                        "families are future work"
                        % (self.axis, int(spx['axis'])))
                palette = SurfacePalette(spx, self.cells, tr, kk, model.dx,
                                         reach)
                bnd = palette.bnd
            else:
                palette = ConductionPalette(
                    kk, (float(d3[tr[0]]), float(d3[tr[1]])))
                bnd = (np.ones(self.sel.size, dtype=bool) if reach is None
                       else _near_surface(np.asarray(model.struc())
                                          .astype(bool), self.cells, tr,
                                          reach))
        else:
            self.sel = palette.sel
            self.cells = fil_cell[self.sel]
            axes = np.unique(fil_axis[self.sel])
            self.axis = int(axes[0]) if axes.size == 1 else None
            bnd = np.ones(self.sel.size, dtype=bool)
        self.palette = palette
        self.shared = palette.shared
        self.fax = fil_axis[self.sel]
        self.nfil = self.sel.size
        self.splits = {int(a): Split.transverse(int(a), kk, d3)
                       for a in np.unique(self.fax)}
        self.wholes = {a: Split(a, (1, 1, 1), d3) for a in self.splits}
        self.k = kk[0]*kk[1]
        if self.shared and self.axis is None:
            raise ValueError("shared weights need one orientation")
        self.split = self.splits.get(self.axis)
        self.whole = self.wholes.get(self.axis)
        self.tr = ([c for c in range(3) if c != self.axis]
                   if self.axis is not None else None)
        self.dt = (self.split.d[self.tr[0]], self.split.d[self.tr[1]]) \
            if self.split is not None else None
        self._bnd = bnd
        self.G = getattr(palette, 'G', None)
        self._rfac = getattr(palette, 'rfac', None)
        self.P = getattr(palette, 'P', None)
        self.reach = reach
        self.use_fft = self.shared and (True if use_fft is None
                                        else bool(use_fft))
        self.csr_max_gb = float(csr_max_gb)
        self._term = term
        self._rc = (int(rc[0]), int(rc[1]))
        # THE AGGREGATES the modes couple to (Zcross columns, the
        # drive): every filament of the family's orientations within
        # rc_cross of an entry -- for a whole-orientation family that
        # is the entries themselves; a patch-subset family (corners)
        # must still see the filaments just outside its patch, or its
        # modes lose most of their drive (measured: the corner bands
        # collapsed from x0.66 to x0.35 without them)
        if palette is None:
            self.agg = self.sel
        else:
            cand = np.flatnonzero(np.isin(fil_axis, list(self.splits)))
            fa, fb = neighbour_pairs(self.cells, self._rc[1],
                                     other=fil_cell[cand].astype(float))
            same = self.fax[fa] == fil_axis[cand[fb]]
            self.agg = cand[np.unique(fb[same])]
        self.agg_cells = fil_cell[self.agg]
        self.agg_axis = fil_axis[self.agg]
        self.tables = PairTables()
        self._pairs = {}
        self._set_weights()
        self._assemble()

    # -- weights -------------------------------------------------------

    def _set_weights(self):
        """(Re)generate the weights at the current rate and everything
        whose SHAPE follows km."""
        if self.shared:
            self.W = self.palette.weights(self._p)
            self.Wf = None
            if np.abs(self.W.sum(axis=0)).max() > 1e-12:
                raise ValueError("mode weights must be net-zero")
            self.km = self.W.shape[1]
            self.mode_mask = np.repeat(self._bnd, self.km)
        else:
            self.W = None
            self.Wf, cmask = self.palette.weights(self._p)
            if np.abs(self.Wf.sum(axis=1)).max() > 1e-9:
                raise RuntimeError("mode weights are not net-zero")
            self.km = self.Wf.shape[2]
            self.mode_mask = (cmask & self._bnd[:, None]).ravel()
        self.nmode_full = self.nfil*self.km
        self.nmode = (int(self.mode_mask.sum()) if self.P is None
                      else self.P.shape[1])

    def weights_of(self, idx):
        """``(len(idx), k, km)`` weights of the given entries (broadcast
        for a shared palette): what a cross block between families
        folds."""
        if self.shared:
            return np.broadcast_to(self.W, (len(idx),) + self.W.shape)
        return self.Wf[idx]

    def set_frequency(self, freq):
        freq = float(freq)
        if freq <= 0 or freq == self.freq:
            return False
        self.freq = freq
        p, self._z = self._model.material_response(freq)
        moved = self.palette.moves and (p != self._p)
        self._p = p
        if moved:
            self._set_weights()
        self._assemble()
        return bool(moved)

    # -- assembly ------------------------------------------------------

    def _sub_impedance(self):
        """Series impedance of ONE whole sub-bar per entry, ``z*l/a``;
        the same values feed ``Ru`` and ``mode_precond`` -- they must
        agree or the preconditioner stops approximating its operator."""
        r = np.array([self._z*self.d3[a]/self.splits[a].area
                      for a in self.fax])
        if self.shared:
            return r
        return r[:, None]*(np.ones(self.k) if self._rfac is None
                           else self._rfac)

    def _fold_modes(self, A, square):
        """Restrict a per-entry mode block (rows, and columns too when
        ``square``) to the retained modes: the mask, or the
        prolongation."""
        if self.P is None:
            mk = self.mode_mask
            return A[mk][:, mk] if square else A[mk]
        return self.P.T @ A @ self.P if square else self.P.T @ A

    def _assemble(self):
        """Fold the current weights into every block. Geometry is
        cached, so a re-assembly pays only contractions, CSR index
        arithmetic and -- the real cost -- the FFT spectra. Only the
        representation that will be applied is built: the sparse blocks
        are ruinous under FFT (pairs grow as (2rc+1)^3; ~5e8 nonzeros on
        a 23600-filament conductor at the default radii)."""
        self.Zuu = self.Zcross = None
        self.nnz = (0, 0)
        if not self.use_fft:
            self._build_truncated()
        r = self._sub_impedance()
        if self.shared:
            self.Ru = sp.kron(sp.identity(self.nfil, format='csr'),
                              r[0]*(self.W.T @ self.W), format='csr')
        else:
            blocks = np.einsum('fpm,fp,fpr->fmr', self.Wf, r, self.Wf)
            self.Ru = sp.block_diag(list(blocks), format='csr')
        self.Zt = None
        if self._term is not None and self._term.axis in self.splits:
            self._build_terminal()
        if self.use_fft:
            self.build_fft()
        # drop the interior modes AFTER assembly, so Zcross keeps all
        # its aggregate columns (the drive) while only mode ROWS go
        if self.P is not None or self.nmode != self.nmode_full:
            for nm in ('Ru', 'Zuu', 'Zcross', 'Zt'):
                v = getattr(self, nm)
                if v is not None:
                    setattr(self, nm, self._fold_modes(v, nm in ('Ru', 'Zuu')))

    def _neighbour_pairs(self, radius, other=None, other_axis=None):
        """Entry pairs (or entry-``other`` pairs) within ``radius``
        cells (inf-norm) and of the same orientation; cached."""
        ck = (float(radius), other is None)
        if ck not in self._pairs:
            fa, fb = neighbour_pairs(self.cells, radius, other)
            oax = self.fax if other is None else other_axis
            same = self.fax[fa] == oax[fb]
            self._pairs[ck] = (fa[same], fb[same])
        return self._pairs[ck]

    def _mode_tables(self, D):
        """Folded ``(km x km)`` and ``(km,)`` blocks per separation
        (shared weights): the Toeplitz tables the FFT and the CSR path
        both read."""
        W = self.W
        Bu = np.einsum('pm,dpq,qr->dmr', W,
                       self.tables(self.split, self.split, D), W)
        Bc = np.einsum('pm,dp->dm', W,
                       self.tables(self.split, self.whole, D)[:, :, 0])
        return Bu, Bc

    def _build_truncated(self):
        """Zuu and Zcross as SPARSE, distance-truncated blocks (the
        radii ladder is in docs/enrichment_history.md; under CSR pairs
        grow as (2rc+1)^3, hence ``_check_csr_size``)."""
        km, mr = self.km, np.arange(self.km)
        rc_uu, rc_cross = self._rc
        fa_u, fb_u = self._neighbour_pairs(rc_uu)
        fa_c, fb_c = self._neighbour_pairs(rc_cross, self.agg_cells.astype(float),
                                           self.agg_axis)
        self._check_csr_size(fa_u.size, fa_c.size)
        dt = float if self.shared else np.result_type(self.Wf, float)
        Bu = np.empty((fa_u.size, km, km), dtype=dt)
        Bc = np.empty((fa_c.size, km), dtype=dt)
        self.ntable = 0
        for a, split in self.splits.items():
            su = self.fax[fa_u] == a
            sc = self.fax[fa_c] == a
            Du = self.cells[fb_u[su]] - self.cells[fa_u[su]]
            Dc = self.agg_cells[fb_c[sc]] - self.cells[fa_c[sc]]
            D, inv = unique_separations(np.concatenate([Du, Dc]))
            iu, ic = inv[:len(Du)], inv[len(Du):]
            self.ntable += int(D.shape[0])
            if self.shared:
                Tu, Tc = self._mode_tables(D)
                Bu[su], Bc[sc] = Tu[iu], Tc[ic]
                continue
            M = self.tables(split, split, D)
            G = (self.G if self.G is not None
                 else np.full((self.agg.size, self.k), 1.0/self.k))
            iu_, ic_ = np.flatnonzero(su), np.flatnonzero(sc)
            step = max(1, 20_000_000 // (self.k*self.k))
            for s0 in range(0, iu_.size, step):
                s = iu_[s0:s0 + step]
                Bu[s] = np.einsum('apm,apq,aqr->amr', self.Wf[fa_u[s]],
                                  M[iu[s0:s0 + step]], self.Wf[fb_u[s]])
            for s0 in range(0, ic_.size, step):
                s = ic_[s0:s0 + step]
                Bc[s] = np.einsum('apm,apq,aq->am', self.Wf[fa_c[s]],
                                  M[ic[s0:s0 + step]], G[fb_c[s]])
        rows = np.broadcast_to(fa_u[:, None, None]*km + mr[None, :, None],
                               Bu.shape).ravel()
        cols = np.broadcast_to(fb_u[:, None, None]*km + mr[None, None, :],
                               Bu.shape).ravel()
        self.Zuu = sp.csr_matrix((Bu.ravel(), (rows, cols)),
                                 shape=(self.nmode_full, self.nmode_full))
        rows = (fa_c[:, None]*km + mr[None, :]).ravel()
        cols = np.broadcast_to(fb_c[:, None], Bc.shape).ravel()
        self.Zcross = sp.csr_matrix((Bc.ravel(), (rows, cols)),
                                    shape=(self.nmode_full, self.agg.size))
        self.nnz = (int(self.Zuu.nnz), int(self.Zcross.nnz))

    def _check_csr_size(self, npair_uu, npair_cross):
        """Refuse sparse blocks that will not fit (checked AFTER the pair
        search so the count is exact, not a bound)."""
        nnz = npair_uu*self.km*self.km + npair_cross*self.km
        gb = 12.0*nnz/1e9          # 8 B data + 4 B index per nonzero
        if gb > self.csr_max_gb:
            raise RuntimeError(
                "redistribution CSR blocks would need ~%.1f GB (%d "
                "nonzeros) at rc=%s with k=%d. Keep use_fft=True, drop to "
                "rc=(1, 2) for the sparse path, or raise csr_max_gb past "
                "%.1f." % (gb, nnz, self._rc, self.k, self.csr_max_gb))

    def _build_terminal(self):
        """Mode <-> terminal block, per (separation, face sign), over
        the entries parallel to the terminal axis."""
        term, km = self._term, self.km
        tcell = np.array([f[0] for f in term.faces], dtype=np.int64)
        fa, tb = neighbour_pairs(self.cells, self._rc[1], tcell.astype(float))
        par = self.fax[fa] == term.axis
        fa, tb = fa[par], tb[par]
        if fa.size == 0:
            self.Zt = sp.csr_matrix((self.nmode_full, term.n))
            return
        split = self.splits[term.axis]
        D, inv = unique_separations(tcell[tb] - self.cells[fa])
        Mct = np.stack([self.tables(split, split.terminal(term.t_l, s),
                                    D)[:, :, 0] for s in (-1, +1)], axis=-1)
        si = (term.sign[tb] > 0).astype(np.int64)
        if self.shared:
            red = np.einsum('pm,dps->dms', self.W, Mct)[inv, :, si]
        else:
            red = np.einsum('apm,aps->ams', self.Wf[fa], Mct[inv])
            red = red[np.arange(fa.size), :, si]
        rows = (fa[:, None]*km + np.arange(km)[None, :]).ravel()
        cols = np.broadcast_to(tb[:, None], red.shape).ravel()
        self.Zt = sp.csr_matrix((red.ravel(), (rows, cols)),
                                shape=(self.nmode_full, term.n))

    # -- the apply -----------------------------------------------------

    def apply(self, u, i_f):
        """``(Zuu u + Zcross i_f, Zcross' u)`` with ``i_f`` the currents
        of this family's aggregates (``agg``), by FFT or by the sparse
        blocks."""
        if self.use_fft:
            return self.apply_fft(u, i_f)
        return self.Zuu @ u + self.Zcross @ i_f, self.Zcross.T @ u

    def build_fft(self):
        """Spectra for applying the mode blocks as CONVOLUTIONS: the
        blocks are translation invariant, the kernels real (one
        spectrum serves correlation and convolution). Tabulated only
        over separations that can occur (stencil clipped to the grid),
        built one kernel at a time and stored single precision -- the
        spectra ARE the allocation (km^2 x pad complex, 19.8 GB on the
        RSFQ XNOR) and are smooth mutual inductances far from float32's
        floor; ``SPPEEC_MODE_FP64=1`` for A/B. Whole-bounding-box, so
        box-proportional rather than occupancy-proportional."""
        import os
        from scipy import fft as sfft
        rc_uu, rc_cross = self._rc
        rc = max(rc_uu, rc_cross)
        self.org = self.cells.min(axis=0)
        shape = (self.cells.max(axis=0) - self.org + 1).astype(int)
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
        """Mode blocks by convolution: ``(Zuu u + Zcross i_f, Zcross' u)``.
        A masked mode set is expanded with zeros here and compressed on
        the way out."""
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

    # -- preconditioning -------------------------------------------------

    def mode_precond(self, jw):
        """Block-Jacobi inverse of ``Ru + jw*Zuu_self`` per group of
        modes (per filament; per patch for a prolonged palette), in the
        retained mode numbering. The mesh preconditioner's mode block
        is the identity, so without this the mode equations run
        unpreconditioned and stall at high omega with a rich basis
        (docs/enrichment_history.md). Shared weights: ONE km x km
        inverse, Kronecker'd."""
        if self.nmode == 0:
            return None
        if self.shared:
            Ls = self.tables(self.split, self.split,
                             np.zeros((1, 3), dtype=np.int64))[0]
            W = self.W
            A = self._sub_impedance()[0]*(W.T @ W) + jw*(W.T @ (Ls @ W))
            return sp.kron(sp.identity(int(self._bnd.sum()), format='csr'),
                           sp.csr_matrix(np.linalg.inv(A)), format='csr')
        A = (self.Ru + jw*self.Zuu).tocsr()
        if self.P is not None:
            groups = self.palette.groups
        else:
            counts = self.mode_mask.reshape(self.nfil, self.km).sum(axis=1)
            ends = np.cumsum(counts)
            groups = [np.arange(e - c, e) for c, e in zip(counts, ends)
                      if c]
        data, rows, cols = [], [], []
        for idx in groups:
            idx = np.asarray(idx)
            inv = np.linalg.inv(A[idx][:, idx].toarray())
            rows.append(np.repeat(idx, len(idx)))
            cols.append(np.tile(idx, len(idx)))
            data.append(inv.ravel())
        return sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows),
                                    np.concatenate(cols))),
            shape=(self.nmode, self.nmode))


# ----------------------------------------------------------------- stacks

class ModeStack:
    """Several families as ONE ``redist`` object: ``u = [u_1; u_2; ...]``.

    Each family keeps its own apply (FFT or sparse); the mode<->mode
    coupling BETWEEN families is included (dropping mode-mode dipole
    couplings over-crowds, the C.2 lesson) as sparse cross blocks over
    parallel entry pairs within the larger of the two families'
    ``rc_uu`` -- beyond that the coupling is dipole-dipole 1/r^3, the
    class each family itself truncates. The raw pair tables are
    cached; a retune of any family only re-folds the weights.
    """

    def __init__(self, families):
        self.fam = list(families)
        self.agg = np.unique(np.concatenate([f.agg for f in self.fam]))
        self._pos = [np.searchsorted(self.agg, f.agg) for f in self.fam]
        self._raw = {}
        for a in range(len(self.fam)):
            for b in range(a + 1, len(self.fam)):
                A, B = self.fam[a], self.fam[b]
                radius = max(A._rc[0], B._rc[0])
                fa, fb = neighbour_pairs(A.cells, radius,
                                         other=B.cells.astype(float))
                same = A.fax[fa] == B.fax[fb]
                fa, fb = fa[same], fb[same]
                self._raw[a, b] = (fa, fb)
        self._restack()

    def _restack(self):
        """(Re)fold everything that depends on the families' weights."""
        self.n = [f.nmode for f in self.fam]
        self.off = np.concatenate([[0], np.cumsum(self.n)])
        self.nmode = int(self.off[-1])
        self.Ru = sp.block_diag([f.Ru for f in self.fam], format='csr')
        self.cross = {}
        for (a, b), (fa, fb) in self._raw.items():
            A, B = self.fam[a], self.fam[b]
            Z = sp.csr_matrix((A.nmode_full, B.nmode_full))
            if fa.size:
                Wa, Wb = A.weights_of(fa), B.weights_of(fb)
                vals = np.empty((fa.size, A.km, B.km),
                                dtype=np.result_type(Wa, Wb, float))
                for ax in np.unique(A.fax[fa]):
                    s = np.flatnonzero(A.fax[fa] == ax)
                    D, inv = unique_separations(B.cells[fb[s]] - A.cells[fa[s]])
                    T = A.tables(A.splits[ax], B.splits[ax], D)
                    vals[s] = np.einsum('apm,apq,aqr->amr', Wa[s], T[inv],
                                        Wb[s])
                rows = np.broadcast_to(
                    fa[:, None, None]*A.km + np.arange(A.km)[None, :, None],
                    vals.shape).ravel()
                cols = np.broadcast_to(
                    fb[:, None, None]*B.km + np.arange(B.km)[None, None, :],
                    vals.shape).ravel()
                Z = sp.csr_matrix((vals.ravel(), (rows, cols)), shape=Z.shape)
            Z = A._fold_modes(Z.tocsr(), False) if A.P is not None or \
                A.nmode != A.nmode_full else Z
            Z = (Z @ B.P if B.P is not None
                 else Z[:, B.mode_mask] if B.nmode != B.nmode_full else Z)
            self.cross[a, b] = Z.tocsr()
        self.kk = self.fam[0].kk
        self.nnz = (sum(getattr(f, 'nnz', (0, 0))[0] for f in self.fam),
                    sum(getattr(f, 'nnz', (0, 0))[1] for f in self.fam))
        self.ntable = sum(getattr(f, 'ntable', 0) for f in self.fam)
        Zts = [f.Zt for f in self.fam]
        nt = next((z.shape[1] for z in Zts if z is not None), 0)
        self.Zt = (None if nt == 0 else
                   sp.vstack([z if z is not None else sp.csr_matrix((f.nmode, nt))
                              for z, f in zip(Zts, self.fam)], format='csr'))

    def apply(self, u, i_f):
        """(Zuu@u + Zcross@i_f, Zcross.T@u) over the stacked layout;
        ``i_f`` is the aggregate slice over the UNION of ``agg``."""
        parts = [u[self.off[a]:self.off[a + 1]] for a in range(len(self.fam))]
        out_u = []
        mf = np.zeros(self.agg.size, dtype=np.complex128)
        for a, f in enumerate(self.fam):
            mu, mfa = f.apply(parts[a], np.ascontiguousarray(i_f[self._pos[a]]))
            for (i, j), Z in self.cross.items():
                if i == a:
                    mu = mu + Z @ parts[j]
                elif j == a:
                    mu = mu + Z.T @ parts[i]
            out_u.append(mu)
            np.add.at(mf, self._pos[a], mfa)
        return np.concatenate(out_u), mf

    def set_frequency(self, freq):
        changed = [f.set_frequency(freq) for f in self.fam]
        if any(changed):
            self._restack()
        return any(changed)

    def mode_precond(self, jw):
        """Block-diagonal: each family's own preconditioner (identity
        where it has none)."""
        blocks = []
        for f in self.fam:
            Pf = f.mode_precond(jw)
            blocks.append(sp.csr_matrix(Pf) if Pf is not None
                          else sp.identity(f.nmode, format='csr',
                                           dtype=np.complex128))
        return sp.block_diag(blocks, format='csr')


# ------------------------------------------------------------- the front end

def skin_depth(sigma, freq):
    """Classical skin depth ``sqrt(2/(w mu sigma))`` (metres)."""
    from voxmodel import MU0
    return float(np.sqrt(2.0/(2*np.pi*freq*MU0*sigma))) if freq > 0 \
        else np.inf


def _auto_rc(occ, axis):
    """Width-scaled coupling radii ``(rc_uu, rc_cross)``.

    The mode couplings correlate over the cross-section WIDTH: measured
    on the straight-bar ladder, (3,4) is fine at 2 cells across yet
    silently truncates ~20 delivered points at 4 across, where (6,8) --
    1.5-2x the width -- recovers +14 at unchanged apply cost. Take the
    median transverse run length perpendicular to the port axis (so one
    wide pour does not inflate rc everywhere), rc = (ceil 1.5W, ceil 2W)
    off the THIN dimension, floors (3,4), cap (12,16) (table setup
    grows as rc^3), and fall back to (3,4) whenever the scaled radii
    cannot clear 1.5x the WIDE dimension under the cap -- the rc
    ladder's measured mid-shell damage zone (a 20x20 bar reads WORSE at
    rc 12-20 than at (3,4): a hard cutoff mid-shell leaves an
    unbalanced residue rather than dropping negligible terms).
    """
    occ = np.asarray(occ).astype(bool)
    runs = []
    for t in [c for c in range(3) if c != int(axis)]:
        rl = []
        for line in np.moveaxis(occ, t, -1).reshape(-1, occ.shape[t]):
            n = 0
            for v in line:
                if v:
                    n += 1
                elif n:
                    rl.append(n)
                    n = 0
            if n:
                rl.append(n)
        runs.append(float(np.median(rl)) if rl else 1.0)
    w_thin, w_wide = min(runs), max(runs)
    ru = max(3, int(np.ceil(1.5*w_thin)))
    rc = max(4, int(np.ceil(2.0*w_thin)))
    if ru > 12 or rc > 16:
        return 3, 4
    if rc > 0.5*w_wide and ru < 1.5*w_wide:
        return 3, 4
    return ru, rc


ENRICH_KEYS = ('families', 'k', 'reach', 'rc', 'f_ref', 'use_fft',
               'csr_max_gb')


class EnrichConfig(dict):
    """Resolved enrichment settings (attribute access over a dict)."""
    __getattr__ = dict.__getitem__


def check_request(req):
    """Validate an enrichment table without a model (the TOML parser
    calls this at load time; :func:`resolve` again at build)."""
    bad = set(req) - set(ENRICH_KEYS)
    if bad:
        raise ValueError("enrich: unknown key(s) %s -- allowed: %s"
                         % (sorted(bad), ', '.join(ENRICH_KEYS)))
    if any(f not in ('section', 'corner')
           for f in req.get('families', [])):
        raise ValueError("enrich.families: 'section' and/or 'corner'")
    k = req.get('k')
    if k is not None:
        if int(k) == 2:
            raise ValueError("enrich.k = 2 is BLIND to axially symmetric "
                             "neighbourhoods (cross-couplings cancel "
                             "exactly, measured at machine zero); use 3 "
                             "or more, odd preferred")
        if int(k) < 3:
            raise ValueError("enrich.k must be >= 3 (or omit it)")
    reach = req.get('reach', 0)
    if reach != 'all' and int(reach) < 0:
        raise ValueError("enrich.reach must be >= 0 or 'all'")
    rc = req.get('rc')
    if rc is not None and (len(rc) != 2 or min(int(v) for v in rc) < 1):
        raise ValueError("enrich.rc is two radii >= 1")
    if req.get('f_ref') is not None and float(req['f_ref']) <= 0:
        raise ValueError("enrich.f_ref must be > 0")


def resolve(model, request, port_axis):
    """The one place the engagement rules live.

    ``request`` is ``None`` or ``'off'`` (no enrichment), ``'auto'``
    (the defaults, engaged only when the cell size justifies it, and
    never by default on a superconductor -- London modes are opt-in),
    or a table with any of ``families`` (``'section'``, ``'corner'``),
    ``k``, ``reach`` (cells, or ``'all'``), ``rc`` (two radii),
    ``f_ref`` and the internals ``use_fft`` / ``csr_max_gb``; a table
    is explicit, so a model the rules cannot serve raises instead of
    degrading. Returns an :class:`EnrichConfig` or ``None``.

    THE RULES. The section family engages when a transverse cell
    exceeds half the length the current varies on (the skin depth at
    ``f_ref``, or lambda), and its quadrature is k = min(12, max(7,
    ceil(2 dx/length))) -- a sub-bar no coarser than half that length
    (k is quadrature, km drives cost; measured +4 delivered points at
    dx/delta = 6 for k 7 -> 12). A given ``k`` is honoured (k = 2 is
    refused: a 2x2 split cannot express "more current at the edges
    than the centre"). A declared film normal off the port axis gets
    the 1-D film palette ``(k, 1)`` along the normal. Radii: given, or
    (12,16) on films (aligned dipoles), (3,4) on cylinder fills (the
    sparse path), else width-scaled (:func:`_auto_rc`).
    """
    if request is None or request == 'off':
        return None
    if isinstance(request, EnrichConfig):     # already resolved
        return request
    explicit = isinstance(request, dict)
    if not explicit and request != 'auto':
        raise ValueError("enrich must be 'off', 'auto' or a table, got %r"
                         % (request,))
    req = dict(request) if explicit else {}
    check_request(req)
    fam = list(req.get('families', ['section']))
    f_ref = (float(req['f_ref']) if req.get('f_ref') is not None
             else float(np.max(model.freq)) if len(model.freq) else 0.0)
    if f_ref <= 0:
        raise ValueError("enrich needs a frequency: f_ref, or a sweep")
    reach = req.get('reach', 0)
    reach = None if reach == 'all' else int(reach)
    k = None if req.get('k') is None else int(req['k'])
    d3 = np.asarray(model.d, dtype=float)
    tr = [c for c in range(3) if c != int(port_axis)]
    fnorm = getattr(model, 'film_normal', None)
    film = fnorm is not None and int(fnorm) != int(port_axis)
    kk = None
    if 'section' in fam:
        length = None
        if getattr(model, 'superconductor', False):
            lp = model.london_rate()
            if lp is None and explicit:
                raise NotImplementedError(
                    "sub-cell modes on a superconductor with no single "
                    "London depth: the palette is exponentials at ONE "
                    "rate 1/lambda")
            length = 1.0/lp if (lp is not None and explicit) else None
        else:
            try:
                length = skin_depth(model.uniform_sigma(), f_ref)
            except ValueError:
                if explicit:
                    raise
        if length is None:
            fam.remove('section')
        else:
            dtc = float(d3[int(fnorm)]) if film else max(float(d3[c])
                                                          for c in tr)
            if k is None:
                k = (int(min(12, max(7, np.ceil(2*dtc/length))))
                     if 2*dtc/length > 1 else 1)
            if k > 1:
                kk = tuple(k if (not film or c == int(fnorm)) else 1
                           for c in tr)
            else:
                fam.remove('section')
    rc = req.get('rc')
    if rc is None:
        cyl = getattr(model, 'cut', None) is not None and \
            model.cut['kind'] == 'cylinder'
        rc = ((12, 16) if film else (3, 4) if cyl
              else _auto_rc(model.struc(), port_axis))
    rc = (int(rc[0]), int(rc[1]))
    if not fam:
        return None
    return EnrichConfig(families=fam, k=k, kk=kk, reach=reach, rc=rc,
                        f_ref=f_ref, use_fft=req.get('use_fft'),
                        csr_max_gb=float(req.get('csr_max_gb', 2.0)))


def build(model, M, fil_axis, fil_cell, term, cfg, verbose=False):
    """The families a resolved config asks for, as one ``redist``
    object (an :class:`Enrichment` or a :class:`ModeStack`), or None."""
    red = None
    if cfg.kk is not None:
        with _spstatus.task('skin engine k=%d' % cfg.k):
            red = Enrichment(model, M, term.axis, fil_axis, fil_cell,
                             cfg.kk, term=term, rc=cfg.rc, reach=cfg.reach,
                             use_fft=cfg.use_fft, csr_max_gb=cfg.csr_max_gb,
                             freq=cfg.f_ref)
    if 'corner' in cfg.families:
        from cornermode import corner_palette
        # tabulate against the engine's single-axis coverage (axis 2 has
        # no in-plane modes) so the tables do not double count the
        # near-corner crowding the engine already fixes
        arm = {0: 'u', 1: 'v'}.get(term.axis) if red is not None else None
        cp = corner_palette(model, M, fil_axis, fil_cell, engine_arm=arm,
                            verbose=verbose)
        if cp is not None:
            # mode-mode dense across a patch (4W), mode-aggregate at the
            # engine's own radius (measured: the composed bands sit at
            # x0.46 here against x0.41 at 2W + rc_cross)
            Wm = max(c[4] for c in cp.corners)
            cm = Enrichment(model, M, None, fil_axis, fil_cell,
                            (cp.k_in, 1), palette=cp, term=term,
                            rc=(4*Wm, cfg.rc[1]), use_fft=False,
                            csr_max_gb=cfg.csr_max_gb, freq=cfg.f_ref)
            red = cm if red is None else ModeStack([red, cm])
    return red
