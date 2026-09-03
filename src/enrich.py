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


# ------------------------------------------------------------------ material

MU0 = 4e-7*np.pi


def base_sigma(model):
    """The metal's conductivity: the single value of a uniform model, or
    the cylinder metal of a fill model (whose per-cell ``sigma`` carries
    ``sigma*fill`` and is mixed by construction)."""
    spx = getattr(model, 'subpixel', None)
    if spx and spx['geom']:
        sigs = {float(g[3]) for g in spx['geom'].values()}
        if len(sigs) != 1:
            raise ValueError("cylinders with different sigma in one model "
                             "(%d values)" % len(sigs))
        return sigs.pop()
    return model.uniform_sigma()


def london_rate(model):
    """``1/lambda`` for a uniform London model, else None (mixed or
    absent: the caller decides)."""
    lam = getattr(model, 'lambdaL', None)
    if lam is None or not getattr(model, 'superconductor', False):
        return None
    vals = np.unique(np.asarray(lam, dtype=float)[np.asarray(lam) > 0.0])
    return 1.0/float(vals[0]) if vals.size == 1 else None


def material_response(model, freq):
    """``(p, z)`` -- the ONLY two things the mode engine asks of a material.

    ``p`` is the decay rate of the Helmholtz equation the interior
    current obeys (it sets the palette's face/corner exponentials);
    ``z`` is the series impedance density in ohm*m (it sets the sub-bar
    impedance ``z*l/a``)::

        normal conductor   grad^2 J = j w mu sigma J   p = (1+j)/delta
                                                       z = 1/sigma
        London supercond.  grad^2 J = J/lambda^2       p = 1/lambda
                                                       z = j w mu lambda^2

    Two slots, one material law each, and no superconductor branch
    anywhere below: ``p`` moves with frequency for a normal conductor
    and not for a superconductor, ``z`` the other way round, and
    :meth:`Enrichment.set_frequency` recomputes both and rebuilds what
    moved. Passing ``delta = lambda`` into the complex rate is the trap
    worth naming: ``(1+j)/lambda`` spans ``exp(-x/lam)cos/sin``, which
    does NOT contain the London profile ``cosh((x-t/2)/lambda)``
    (residual 1.2e-2 against 2.2e-15 for the two real-rate columns).
    """
    if not freq or float(freq) <= 0.0:
        raise ValueError("the mode shapes are exponentials in the "
                         "material's Helmholtz rate and need freq > 0 "
                         "(got %r)" % (freq,))
    w = 2.0*np.pi*float(freq)
    lam_inv = london_rate(model)
    if lam_inv is not None:
        return lam_inv, 1j*w*MU0/lam_inv**2
    sigma = base_sigma(model)
    return (1.0 + 1.0j)/np.sqrt(2.0/(w*MU0*sigma)), 1.0/sigma


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


# --------------------------------------------------------------- the family

class Enrichment:
    """Net-zero sub-cell redistribution modes on one filament orientation.

    Each filament parallel to ``axis`` is cut into ``kk[0] x kk[1]``
    sub-bars spanning THE SAME TWO NODES and the basis is changed to
    (aggregate, redistribution)::

        i_p = I*g_p + sum_m W_pm u_m ,      sum_p W_pm = 0

    The aggregate ``I`` is exactly the original filament (``g`` uniform,
    or the fill shares on a partial cell) -- ``W_agg' Z_sub W_agg ==
    Z_full`` because the volume integral over the cross-section is the
    sum over its pieces (validate_enrich D) -- so the Toeplitz near
    field and the FMM far field are UNTOUCHED. The ``km`` modes carry
    ZERO net current, hence zero incidence: no new nodes, invisible to
    KCL and to the nodal Schur complement. Physically each mode is the
    mesh current of the loop formed by two parallel branches and its
    equation is KVL around that loop: at DC the branches are identical
    and the modes vanish; at AC they couple differently to the rest of
    the structure and do not. That asymmetry IS skin, proximity and
    London screening, from extra circuit equations instead of extra
    mesh. Net-zero modes are dipoles, so mode<->aggregate falls as
    1/r^2 and mode<->mode as 1/r^3 and the blocks are truncated at the
    radii ``rc = (rc_uu, rc_cross)``.

    Blocks, in the solver's augmented system ``[i_f; i_t; u]``: ``Ru``
    (sub-bar series impedance ``z*l/a`` folded through W -- real
    ``1/sigma`` for a normal conductor, ``j w mu lambda^2`` for a London
    superconductor, whose w-proportionality makes the profile
    frequency independent), ``Zuu`` (mode-mode), ``Zcross``
    (mode-aggregate), ``Zt`` (mode-terminal). Two configurations of
    the same fold:

    * SHARED weights (the default): every filament carries the same
      ``W`` from :func:`conduction_weights`, so the folded tables are
      Toeplitz and the blocks apply as FFT convolutions (``use_fft``);
      ``kk = (1, kz)`` is the thin-film palette.
    * PER-CELL weights on a resolved cylinder (``model.subpixel``):
      :func:`surface_weights` anchored to the true surface, fill-
      weighted ``Ru`` and fill-share aggregates ``G``; the blocks are
      sparse and truncated (a per-cell fold is exactly what a
      convolution cannot represent).

    Placement: modes live on filaments within ``reach`` cells of the
    conductor surface (``reach=None``: everywhere). Interior modes
    are unphysical -- a cell with metal on all sides has no surface to
    crowd against -- and their spurious dipoles, mishandled by the
    truncation, OVER-concentrate current: on a 20-cell-wide bar at
    dx/delta = 4.8 modes-everywhere overstated loss by 70%% at 100 GHz
    where reach 0 erred -7.7%%, the safe direction, and was cheaper.

    Frequency: ``set_frequency`` recomputes the material's ``(p, z)``,
    regenerates the weights only if the rate ``p`` moved (a normal
    conductor's does, a London rate does not) and re-folds the cached
    tables either way (``z`` moves for the superconductor). Returns
    True when the weights moved -- pruning is rate-dependent, so
    ``nmode`` may change and the caller rebuilds what sized itself on
    it.
    """

    def __init__(self, model, M, axis, fil_axis, fil_cell, kk, term=None,
                 rc=(3, 4), reach=0, use_fft=None, csr_max_gb=2.0,
                 freq=None):
        self.axis = int(axis)
        self.tr = [c for c in range(3) if c != self.axis]
        kk = ((int(kk), int(kk)) if np.isscalar(kk)
              else tuple(int(v) for v in kk))
        if min(kk) < 1 or max(kk) < 2:
            raise ValueError("kk must give at least two sub-filaments")
        self.kk = kk
        d3 = np.asarray(model.d, dtype=float)
        self.d3, self.dax = d3, float(d3[self.axis])
        self.dt = (float(d3[self.tr[0]]), float(d3[self.tr[1]]))
        self.split = Split.transverse(self.axis, kk, d3)
        self.whole = Split(self.axis, (1, 1, 1), d3)
        self.k = self.split.nsub
        self._model = model
        self.sel = np.flatnonzero(fil_axis == self.axis)
        self.nfil = self.sel.size
        self.cells = np.asarray(fil_cell)[self.sel]
        self.freq = float(freq) if freq else 0.0
        self._p, self._z = material_response(model, self.freq)
        spx = getattr(model, 'subpixel', None)
        self.shared = not (spx is not None and spx['cells'])
        self.reach = reach
        if self.shared:
            self.G = None
            self._rfac = None
            self._bnd = (np.ones(self.nfil, dtype=bool) if reach is None
                         else _near_surface(np.asarray(model.struc())
                                            .astype(bool), self.cells,
                                            self.tr, reach))
        else:
            if int(spx['axis']) != self.axis:
                raise NotImplementedError(
                    "surface modes: the terminal axis (%d) differs from "
                    "the cylinder axis (%d) -- transverse mode families "
                    "are future work" % (self.axis, int(spx['axis'])))
            self._tkey, self._percell = _surface_geometry(
                spx, self.cells, self.tr, kk, model.dx)
            self.G = np.full((self.nfil, self.k), 1.0/self.k)
            self._rfac = np.ones((self.nfil, self.k))
            self._bnd = np.zeros(self.nfil, dtype=bool)
            lim = (float(reach) + 1.0)*model.dx if reach is not None \
                else np.inf
            for f, key in enumerate(self._tkey):
                pc = self._percell[key]
                if pc is None:
                    continue
                sup = pc['fill'] > 1e-3
                tot = pc['fill'].sum()
                self.G[f] = pc['fill']/tot if tot > 0 else 0.0
                self._rfac[f] = np.where(sup, 1.0/np.where(sup, pc['fill'],
                                                          1.0), 0.0)
                self._bnd[f] = bool(np.any(np.abs(pc['d'][sup]) <= lim))
        self.use_fft = self.shared and (True if use_fft is None
                                        else bool(use_fft))
        self.csr_max_gb = float(csr_max_gb)
        self._term = term
        self._rc = (int(rc[0]), int(rc[1]))
        self.tables = PairTables()
        self._pairs = {}
        self._set_weights()
        self._assemble()

    # -- weights -------------------------------------------------------

    def _set_weights(self):
        """(Re)generate the weights at the current rate and everything
        whose SHAPE follows km."""
        if self.shared:
            self.W = conduction_weights(self.kk, self.dt, self._p)
            self.Wf = None
            if np.abs(self.W.sum(axis=0)).max() > 1e-12:
                raise ValueError("mode weights must be net-zero")
            self.km = self.W.shape[1]
            self.mode_mask = np.repeat(self._bnd, self.km)
        else:
            self.W = None
            self.Wf, cmask = surface_weights(self._percell, self._tkey,
                                             self.k, self._p)
            if np.abs(self.Wf.sum(axis=1)).max() > 1e-9:
                raise RuntimeError("surface mode weights are not net-zero")
            self.km = SURFACE_KM
            self.mode_mask = (cmask & self._bnd[:, None]).ravel()
        self.nmode_full = self.nfil*self.km
        self.nmode = int(self.mode_mask.sum())

    def set_frequency(self, freq):
        freq = float(freq)
        if freq <= 0 or freq == self.freq:
            return False
        self.freq = freq
        p, self._z = material_response(self._model, freq)
        moved = (p != self._p)
        self._p = p
        if moved:
            self._set_weights()
        self._assemble()
        return bool(moved)

    # -- assembly ------------------------------------------------------

    def _sub_impedance(self):
        """Series impedance of ONE whole sub-bar, ``z*l/a``; the same
        value feeds ``Ru`` and ``mode_precond`` -- they must agree or the
        preconditioner stops approximating its operator."""
        return self._z*self.dax/self.split.area

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
        r_sub = self._sub_impedance()
        if self.shared:
            self.Ru = sp.kron(sp.identity(self.nfil, format='csr'),
                              r_sub*(self.W.T @ self.W), format='csr')
        else:
            blocks = np.einsum('fpm,fp,fpr->fmr', self.Wf,
                               r_sub*self._rfac, self.Wf)
            self.Ru = sp.block_diag(list(blocks), format='csr')
        self.Zt = None
        if self._term is not None and self._term.axis == self.axis:
            self._build_terminal()
        if self.use_fft:
            self.build_fft()
        if self.nmode != self.nmode_full:
            # drop the interior modes AFTER assembly, so Zcross keeps all
            # its aggregate columns (the drive) while only mode ROWS go
            mk = self.mode_mask
            self.Ru = self.Ru[mk][:, mk]
            if self.Zuu is not None:
                self.Zuu = self.Zuu[mk][:, mk]
            if self.Zcross is not None:
                self.Zcross = self.Zcross[mk]
            if self.Zt is not None:
                self.Zt = self.Zt[mk]

    def _neighbour_pairs(self, radius, other=None):
        ck = (float(radius), other is None)
        if ck not in self._pairs:
            self._pairs[ck] = neighbour_pairs(self.cells, radius, other)
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
        """Zuu and Zcross as SPARSE, distance-truncated blocks.

        Measured error vs untruncated on a uniform bar at dx/delta 4.8::

            rc_uu, rc_cross    1,1     2,2     1,2     2,3     3,4
            error              9.1e-2  2.7e-2  1.2e-2  5.2e-3  4.4e-4

        The cross block needs reach, the mode-mode block does not --
        the 1/r^2 vs 1/r^3 asymmetry directly. The (3,4) default is
        chosen for the FFT apply, where a radius costs only padding;
        under CSR pairs grow as (2rc+1)^3 (``_check_csr_size``).
        """
        km, mr = self.km, np.arange(self.km)
        rc_uu, rc_cross = self._rc
        fa_u, fb_u = self._neighbour_pairs(rc_uu)
        fa_c, fb_c = self._neighbour_pairs(rc_cross)
        self._check_csr_size(fa_u.size, fa_c.size)
        Du = self.cells[fb_u] - self.cells[fa_u]
        Dc = self.cells[fb_c] - self.cells[fa_c]
        D, inv = unique_separations(np.concatenate([Du, Dc]))
        iu, ic = inv[:len(Du)], inv[len(Du):]
        self.ntable = int(D.shape[0])
        if self.shared:
            Bu, Bc = self._mode_tables(D)
            Bu, Bc = Bu[iu], Bc[ic]
        else:
            M = self.tables(self.split, self.split, D)
            Wf, G = self.Wf, self.G
            Bu = np.empty((fa_u.size, km, km))
            Bc = np.empty((fa_c.size, km))
            step = max(1, 20_000_000 // (self.k*self.k))
            for a in range(0, fa_u.size, step):
                s = np.s_[a:a + step]
                Bu[s] = np.einsum('apm,apq,aqr->amr', Wf[fa_u[s]],
                                  M[iu[s]], Wf[fb_u[s]])
            for a in range(0, fa_c.size, step):
                s = np.s_[a:a + step]
                Bc[s] = np.einsum('apm,apq,aq->am', Wf[fa_c[s]],
                                  M[ic[s]], G[fb_c[s]])
        rows = np.broadcast_to(fa_u[:, None, None]*km + mr[None, :, None],
                               Bu.shape).ravel()
        cols = np.broadcast_to(fb_u[:, None, None]*km + mr[None, None, :],
                               Bu.shape).ravel()
        self.Zuu = sp.csr_matrix((Bu.ravel(), (rows, cols)),
                                 shape=(self.nmode_full, self.nmode_full))
        rows = (fa_c[:, None]*km + mr[None, :]).ravel()
        cols = np.broadcast_to(fb_c[:, None], Bc.shape).ravel()
        self.Zcross = sp.csr_matrix((Bc.ravel(), (rows, cols)),
                                    shape=(self.nmode_full, self.nfil))
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
        """Mode <-> terminal block, per (separation, face sign)."""
        term, km = self._term, self.km
        tcell = np.array([f[0] for f in term.faces], dtype=np.int64)
        fa, tb = self._neighbour_pairs(self._rc[1], other=tcell.astype(float))
        if fa.size == 0:
            self.Zt = sp.csr_matrix((self.nmode_full, term.n))
            return
        D, inv = unique_separations(tcell[tb] - self.cells[fa])
        Mct = np.stack([self.tables(self.split,
                                    self.split.terminal(term.t_l, s),
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

    # -- the FFT apply (shared weights) ---------------------------------

    def build_fft(self):
        """Spectra for applying the mode blocks as CONVOLUTIONS.

        The blocks are translation invariant, so ``out_u[f, m] = sum_g
        sum_n Bu[cell_g - cell_f, m, n] u[g, n]`` is a 3-D correlation
        of ``u`` with ``Bu`` per mode pair and its transpose the
        matching convolution; the kernels are real, so ONE spectrum
        serves both directions. Tabulated only over separations that
        can occur (the stencil clipped per axis to the grid extent),
        built one kernel at a time and stored single precision: the
        spectra ARE the allocation (km^2 x pad complex, 19.8 GB on the
        RSFQ XNOR against a 154 MB working slab) and they are smooth
        O(1e-12..1e-7) mutual inductances nowhere near float32's floor.
        ``SPPEEC_MODE_FP64=1`` restores double storage for A/B. NOTE
        this is a whole-bounding-box transform: box-proportional, not
        occupancy-proportional.
        """
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
        """Block-Jacobi inverse of ``Ru + jw*Zuu_self`` per filament, in
        the masked mode numbering. The mesh preconditioner's mode block
        is the identity, so without this the mode equations run
        unpreconditioned and stall at high omega with a rich basis
        (measured: 311-matvec cap on the engine-only ladder at 1e10;
        2078 unconverged matvecs on the surface palette at dx/delta 6).
        Shared weights: ONE km x km inverse, Kronecker'd."""
        if self.nmode == 0:
            return None
        if self.shared:
            Ls = self.tables(self.split, self.split,
                             np.zeros((1, 3), dtype=np.int64))[0]
            W = self.W
            A = self._sub_impedance()*(W.T @ W) + jw*(W.T @ (Ls @ W))
            return sp.kron(sp.identity(int(self._bnd.sum()), format='csr'),
                           sp.csr_matrix(np.linalg.inv(A)), format='csr')
        A = (self.Ru + jw*self.Zuu).tocsr()
        counts = self.mode_mask.reshape(self.nfil, self.km).sum(axis=1)
        data, rows, cols = [], [], []
        pos = 0
        for c in counts:
            if c == 0:
                continue
            idx = np.arange(pos, pos + int(c))
            inv = np.linalg.inv(A[idx][:, idx].toarray())
            rows.append(np.repeat(idx, len(idx)))
            cols.append(np.tile(idx, len(idx)))
            data.append(inv.ravel())
            pos += int(c)
        return sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows),
                                    np.concatenate(cols))),
            shape=(self.nmode, self.nmode))
