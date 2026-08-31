# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Subpixel stage B: sparse near-field inductance corrections.

Stage A folds a partial cell's RESISTANCE into sigma_eff =
sigma*fill; this module supplies the matching partial-cell
INDUCTANCE as a sparse additive correction

    dL_ij = w_i^T T(sep) w_j - u^T T(sep) u

over near pairs of AXIS-ALIGNED filaments, where T is the exact
box-mutual table over sub-prisms (terminal.box_mutual_matrix, one
vectorized call on a template neighbourhood -- translation-invariant
on the k x k fine lattice), w are the cell's normalised sub-fill
weights (uniform current density over the CLIPPED cross-section) and
u the uniform full-cell weights. Both terms come from the SAME
kernel, so dL == 0 identically on full cells and the Toeplitz far
field is untouched -- the correction is local by construction, decaying
like the shape-difference multipoles.

Stage C.1 -- the imposed-profile experiment, MEASURED AND NOT
WIRED. ``build_dZ(model, M, freq)`` replaces the uniform clipped
weights with the exact complex Kelvin/Bessel profile (J0(kb*r),
kb = (1-j)/delta) averaged per sub-prism, plus the profile's
resistive contraction on the diagonal. Against the exact round-wire
internal impedance this made things WORSE than stage B alone
(R-ratio error -5.2%/-6.7% vs B's -1.1%/-2.4% at dx/delta = 1/2;
k = 8 did not help): the solver's cell-level solution already
carries the between-cell phase evolution of the crowding, and
imposing the global profile's intra-cell phase on top, renormalised
per cell, double-counts it -- the unconjugated w^2 contraction then
yields destructive real-part artifacts. CONCLUSION, the measured
justification for stage C.2: enrichment amplitudes must be SOLVED
(net-zero modes over these same sub-prism tables, amplitudes from
the system), never imposed. The machinery here is C.2's testbed.

Scope: the CYLINDER path covers filaments along the cylinder axis
only; transverse and cross-orientation corrections remain future work
on that table structure.

``slab_dL`` (2026-08-30) is the same formula for an AXIS-ALIGNED SLAB
cut, which is the RSFQ layer-stack case and is easier in two ways: the
fill varies along ONE axis only, so the sub-division is 1-D rather than
k x k, and both filament orientations in the plane of the cut take the
correction with the same weights. It covers the two IN-PLANE
orientations; a filament running ACROSS the cut is a length problem,
not a cross-section one, and is not handled here (nor is it in the
cylinder path).
"""
import numpy as np
import scipy.sparse as sp

from terminal import box_mutual_matrix


def _profile_weights(model, freq):
    """Per-cell complex sub-prism weight vectors + areas at freq."""
    from scipy.special import jv
    spx = model.subpixel
    axis, k = int(spx['axis']), int(spx['k'])
    t1, t2 = [c for c in range(3) if c != axis]
    d = np.asarray(model.d, dtype=float)
    s = 64
    o = (np.arange(s) + 0.5)/s
    out = {}
    for cell, sub in spx['cells'].items():
        c1, c2, R, sig = spx['geom'][cell]
        x = (cell[0] + o)*d[t1] - c1
        y = (cell[1] + o)*d[t2] - c2
        rr = np.sqrt(x[:, None]**2 + y[None, :]**2)
        inside = rr < R
        if freq and freq > 0:
            delta = np.sqrt(2.0/(2*np.pi*freq*4e-7*np.pi*sig))
            kb = (1.0 - 1.0j)/delta
            J = np.where(inside, jv(0, kb*rr), 0.0)
        else:
            J = inside.astype(complex)
        Jb = J.reshape(k, s//k, k, s//k).sum(axis=(1, 3))
        w = Jb.ravel()/Jb.sum()
        area = sub.ravel()*(d[t1]/k)*(d[t2]/k)
        out[cell] = (w.astype(complex), area)
    return out


def slab_weights(fill, k):
    """1-D sub-prism weights for a cell filled ``fill`` of its extent.

    Uniform current density over the clipped part, normalised. Returns
    ``None`` for a whole cell, where the correction is identically zero.
    """
    f = float(fill)
    if f >= 1.0 - 1e-12:
        return None
    h = 1.0/k
    edges = np.arange(k + 1)*h
    w = np.clip(np.minimum(edges[1:], f) - edges[:-1], 0.0, None)
    tot = w.sum()
    if tot <= 0.0:
        return None
    return w/tot


def slab_dL(model, M, fill, axis, k=8, window=2):
    """Partial-cell inductance correction for an AXIS-ALIGNED SLAB cut.

    ``fill`` is the per-cell filled fraction along ``axis`` (1.0 where
    whole); ``axis`` is the cut direction, so the corrected filaments
    are the two IN-PLANE orientations. Same identity as the cylinder
    path::

        dL_ij = w_i^T T(sep) w_j - u^T T(sep) u

    with T the exact box-mutual table over sub-prisms. Both terms use
    the SAME kernel, so dL vanishes identically on whole cells and the
    Toeplitz far field is untouched: the correction is local, decaying
    like the shape-difference multipoles (measured on a z-cut at fill
    0.5: -21.8% of the pair at one cell, -12.1% at two, -8.2% at three,
    but the ABSOLUTE dL falls 2.14 -> 0.60 -> 0.27 e-15, which is what
    makes a small window enough).

    Cheaper than the cylinder case: the fill varies along one axis only,
    so the sub-division is 1-D (k pieces, not k*k), and a filament's two
    end cells share a fill because they differ only in an in-plane
    index.

    Returns a sparse (efg, efg) real matrix, or None if nothing is
    partial.
    """
    from equiterminal import filament_cells
    fill = np.asarray(fill, dtype=float)
    axis = int(axis)
    d = np.asarray(model.d, dtype=float)
    fil_axis, fil_cell = filament_cells(M)
    u = np.full(k, 1.0/k)

    rows, cols, vals = [], [], []
    Tcache = {}

    def table(orient, off):
        """Sub-prism mutual block for two `orient` filaments at cell
        offset `off`. Translation invariant, so cached per offset."""
        key = (orient, off)
        if key in Tcache:
            return Tcache[key]
        h = d[axis]/k
        lo = np.zeros((2*k, 3))
        hi = np.zeros((2*k, 3))
        for b in range(2):
            base = np.array(off, dtype=float)*d if b else np.zeros(3)
            for s in range(k):
                r = b*k + s
                lo[r] = base
                hi[r] = base + d
                # the filament spans centre to centre along `orient`
                lo[r, orient] = base[orient] + 0.5*d[orient]
                hi[r, orient] = base[orient] + 1.5*d[orient]
                lo[r, axis] = base[axis] + s*h
                hi[r, axis] = base[axis] + (s + 1)*h
        T = box_mutual_matrix(lo, hi, orient)[:k, k:]
        Tcache[key] = T
        return T

    for orient in [c for c in range(3) if c != axis]:
        sel = np.nonzero(fil_axis == orient)[0]
        if sel.size == 0:
            continue
        cells = fil_cell[sel]
        # a filament's two end cells differ only in `orient`, so they
        # share the cut-axis index and therefore the fill
        wv = {}
        for n, c in enumerate(cells):
            w = slab_weights(fill[c[0], c[1], c[2]], k)
            if w is not None:
                wv[n] = w
        if not wv:
            continue
        # EVERY PAIR WITH AT LEAST ONE PARTIAL END, not just
        # partial-partial. dL_ij = w_i^T T w_j - u^T T u is nonzero as
        # soon as ONE of the two differs from uniform, and those pairs
        # dominate: a partial cell couples to the whole slab under it
        # and to the ground plane, none of which are partial. (Pairing
        # only partials recovered ~10% of the correction, measured.)
        by_cell = {}
        for n, c in enumerate(cells):
            by_cell[(int(c[0]), int(c[1]), int(c[2]))] = n
        rng = range(-int(window), int(window) + 1)
        seen = set()
        for ia in sorted(wv):
            ca = cells[ia]
            for dx0 in rng:
                for dx1 in rng:
                    for dx2 in rng:
                        key = (int(ca[0]) + dx0, int(ca[1]) + dx1,
                               int(ca[2]) + dx2)
                        ib = by_cell.get(key)
                        if ib is None:
                            continue
                        for i, j, off in ((ia, ib, (dx0, dx1, dx2)),
                                          (ib, ia, (-dx0, -dx1, -dx2))):
                            if (i, j) in seen:
                                continue
                            seen.add((i, j))
                            T = table(orient, off)
                            wi = wv.get(i, u)
                            wj = wv.get(j, u)
                            v = (wi @ T @ wj) - (u @ T @ u)
                            if v != 0.0:
                                rows.append(int(sel[i]))
                                cols.append(int(sel[j]))
                                vals.append(v)
    if not vals:
        return None
    n = fil_axis.size
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_dZ(model, M, freq, window=2):
    """Sparse complex branch-impedance correction at ``freq``."""
    return _build(model, M, window, freq)


def build_dL(model, M, window=2):
    """Stage-B view: the pure-geometry real dL (uniform weights)."""
    return _build(model, M, window, 0.0)


def _build(model, M, window, freq):
    """Sparse (efg x efg) real dL for a subpixel model, or None."""
    spx = getattr(model, 'subpixel', None)
    if not spx or not spx['cells']:
        return None
    axis, k = int(spx['axis']), int(spx['k'])
    from equiterminal import filament_cells
    fil_axis, fil_cell = filament_cells(M)
    d = np.asarray(model.d, dtype=float)
    t1, t2 = [c for c in range(3) if c != axis]

    # axial filaments, keyed by (axial low cell, transverse cell)
    sel = np.nonzero(fil_axis == axis)[0]
    cells = fil_cell[sel]
    # a filament's weights: mean of its two half-cells' profiles
    # (identical for a constant-cross-section cylinder); cells not in
    # the subpixel dict are FULL (uniform weights). At freq > 0 the
    # weights are the COMPLEX Bessel-profile averages.
    u = np.full(k*k, 1.0/(k*k))
    prof = _profile_weights(model, freq)

    def wvec(c):
        ent = prof.get((int(c[t1]), int(c[t2])))
        return None if ent is None else ent

    weights = {}
    areas = {}
    partial = []
    for n, c in enumerate(cells):
        ea = wvec(c)
        cb = c.copy()
        cb[axis] += 1
        eb = wvec(cb)
        if ea is None and eb is None:
            continue
        wa, aa = ea if ea is not None else (u, None)
        wb, ab = eb if eb is not None else (u, None)
        weights[n] = 0.5*(wa + wb)
        areas[n] = aa if aa is not None else ab
        partial.append(n)
    if not partial:
        return None
    jomega = 2j*np.pi*freq if freq else 0.0

    W = int(window)
    # per-cell-offset exact sub-prism blocks: one (2k^2, 2k^2)
    # box_mutual_matrix call per offset (the cross quadrant is the
    # block), cached -- translation invariant, so ~ (2W+1)^3 calls
    # of k^4 pair integrals total
    la, lt1, lt2 = d[axis], d[t1]/k, d[t2]/k
    g1, g2 = np.meshgrid(np.arange(k), np.arange(k), indexing='ij')
    base_lo = np.zeros((k*k, 3))
    base_lo[:, t1] = g1.ravel()*lt1
    base_lo[:, t2] = g2.ravel()*lt2
    ext = np.zeros(3)
    ext[axis], ext[t1], ext[t2] = la, lt1, lt2
    blocks = {}

    def getB(da, d1, d2):
        key = (da, d1, d2)
        if key not in blocks:
            shift = np.zeros(3)
            shift[axis] = da*la
            shift[t1] = d1*d[t1]
            shift[t2] = d2*d[t2]
            lo = np.vstack([base_lo, base_lo + shift])
            Mm = box_mutual_matrix(lo, lo + ext, axis)
            blocks[key] = Mm[:k*k, k*k:]
        return blocks[key]

    # near pairs: at least one partial filament, cell offset within
    # the window; each unordered pair contributes once (plus its
    # symmetric entry)
    index = {}
    for n, c in enumerate(cells):
        index[(int(c[0]), int(c[1]), int(c[2]))] = n
    efg = int(np.size(M.e.struc) + np.size(M.f.struc)
              + np.size(M.g.struc))
    part_set = set(partial)
    rows, cols, vals = [], [], []
    done = set()
    for n in partial:
        c = cells[n]
        wn = weights[n]
        for da in range(-W, W + 1):
            for d1 in range(-W, W + 1):
                for d2 in range(-W, W + 1):
                    cc = c.copy()
                    cc[axis] += da
                    cc[t1] += d1
                    cc[t2] += d2
                    mth = index.get((int(cc[0]), int(cc[1]),
                                     int(cc[2])))
                    if mth is None:
                        continue
                    pair = (min(n, mth), max(n, mth))
                    if pair in done:
                        continue
                    done.add(pair)
                    wm = weights.get(mth, u)
                    B = getB(da, d1, d2)
                    dl = complex(wn @ B @ wm) - float(u @ B @ u)
                    if freq:
                        dl = jomega*dl
                    if dl == 0.0:
                        continue
                    i, j = int(sel[n]), int(sel[mth])
                    rows.append(i)
                    cols.append(j)
                    vals.append(dl)
                    if i != j:
                        rows.append(j)
                        cols.append(i)
                        vals.append(dl)
    if freq:
        # diagonal resistive contraction of the imposed profile:
        # rho*l*(sum w^2/A - 1/sum A); zero at the uniform profile
        for n in partial:
            w = weights[n]
            A = areas[n]
            ok = A > 0
            c = cells[n]
            _, _, _, sig = spx['geom'][(int(c[t1]), int(c[t2]))]
            rho_l = d[axis]/sig
            dr = rho_l*(complex(np.sum(w[ok]**2/A[ok]))
                        - 1.0/float(A.sum()))
            if dr != 0.0:
                i = int(sel[n])
                rows.append(i)
                cols.append(i)
                vals.append(dr)
    if not vals:
        return None
    dz = sp.csr_matrix((np.asarray(vals, dtype=complex),
                        (rows, cols)), shape=(efg, efg))
    return dz if freq else sp.csr_matrix(
        (np.real(dz.data), dz.indices, dz.indptr), shape=dz.shape)
