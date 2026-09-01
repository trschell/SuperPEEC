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


def slab_dL(model, M, fill, axis, k=8, window=2,
            max_pairs=250_000_000):
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
    partial -- or if the work exceeds ``max_pairs``, in which case it
    WARNS AND RETURNS None rather than exhausting the machine, and the
    solve proceeds with stage A alone.

    THE CAP MEANS SOMETHING DIFFERENT NOW (2026-08-31). It used to
    bound COMPUTE: the pair loop was Python, one small triple product
    per candidate, so (partial filaments) x 2 x (2*window+1)**3 dots
    ran one at a time behind a `seen` set holding every ordered pair.
    At 2.0e7 that cap silently excluded the case the feature exists
    for -- the RSFQ XNOR at pitch 200 / pz 67.5 nm reports 706254
    partial cells and 1.8e8 candidates, 9x over, so every XNOR subpixel
    run to date got stage A alone. Stage A fixes the material law and
    leaves the mutual footprint wrong (a half-filled cell still
    presents a full-cell bar), which is a HALF-APPLIED correction, and
    that is how subpixel measured slightly WORSE than no subpixel.

    The loop is vectorised now and the value is memoised over
    (orient, off, fill_i, fill_j) -- see the comment at the pair loop.
    Compute is no longer the binding cost; the OUTPUT is. Budget about
    16 bytes per emitted pair (int32 row + int32 col + float64 value)
    while the COO arrays live, and roughly 12 bytes/nnz once it is CSR.
    The default admits the XNOR at ~2.9 GB transient. Raise it only
    with that arithmetic in hand.

    Degrading to stage A remains a real answer rather than a failure
    (on the slab bench, stage A carries 16.7% of the R error against
    stage B's further 0.3% of L) -- but it is now a deliberate choice
    at a much higher threshold, not an accident at a low one.
    """
    from equiterminal import filament_cells
    fill = np.asarray(fill, dtype=float)
    axis = int(axis)
    d = np.asarray(model.d, dtype=float)
    fil_axis, fil_cell = filament_cells(M)
    u = np.full(k, 1.0/k)

    npart_est = int(np.count_nonzero((fill > 1e-12) & (fill < 1.0 - 1e-12)))
    cand = npart_est*2*(2*int(window) + 1)**3
    if cand > int(max_pairs):
        import warnings
        warnings.warn(
            "subpixel stage B skipped: %d partial cells give %.1e pair "
            "candidates, over the %.1e cap. The solve keeps stage A "
            "(the material law), which carries most of the correction; "
            "raise max_pairs deliberately if you can pay for it."
            % (npart_est, cand, max_pairs), RuntimeWarning, stacklevel=2)
        return None

    parts = []
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

    # THE PAIR VALUE IS A FUNCTION OF FOUR SMALL THINGS, not of the
    # pair: dL = (w_i^T T(orient, off) w_j) - (u^T T u) depends only on
    # (orient, off, fill_i, fill_j), and `fill` is QUANTIZED -- a layer
    # stack of thickness t at pitch dz yields fills on the 1/n lattice
    # (the RSFQ XNOR at pz = 67.5 nm has 13 distinct values, all n/27).
    # So the 1.8e8 "pair candidates" that used to trip the cap collapse
    # to at most (2 orientations) x (2w+1)^3 offsets x F^2 distinct
    # triple products -- 2 x 125 x 196 = 49000 for that model. The old
    # loop recomputed the same handful of dots a hundred million times
    # in Python, and paid a `seen` set of every ordered pair on top.
    #
    # Values are BIT-IDENTICAL to the scalar loop: the per-(a, b) dot is
    # evaluated exactly as before, (W[a] @ T) @ W[b] - (u @ T) @ u, one
    # entry at a time rather than as a matmul, so no BLAS re-association
    # can shift a ulp. Only the emission ORDER changes, and duplicate-free
    # COO -> CSR is order-independent.
    dims = fill.shape
    for orient in [c for c in range(3) if c != axis]:
        sel = np.nonzero(fil_axis == orient)[0]
        if sel.size == 0:
            continue
        cells = np.asarray(fil_cell[sel], dtype=np.int64)
        # a filament's two end cells differ only in `orient`, so they
        # share the cut-axis index and therefore the fill
        fv = fill[cells[:, 0], cells[:, 1], cells[:, 2]]
        uf, inv = np.unique(fv, return_inverse=True)
        W = np.empty((uf.size, k), dtype=float)
        partial = np.zeros(uf.size, dtype=bool)
        for a, f in enumerate(uf):
            w = slab_weights(f, k)
            if w is None:
                W[a] = u
            else:
                W[a] = w
                partial[a] = True
        if not partial.any():
            continue
        # cell -> local filament index; a later duplicate wins, exactly
        # as the old dict-building loop did
        loc = np.full(dims, -1, dtype=np.int64)
        loc[cells[:, 0], cells[:, 1], cells[:, 2]] = np.arange(cells.shape[0])
        ispart = partial[inv]
        rng = range(-int(window), int(window) + 1)
        for dx0 in rng:
            for dx1 in rng:
                for dx2 in rng:
                    off = (dx0, dx1, dx2)
                    c0 = cells[:, 0] + dx0
                    c1 = cells[:, 1] + dx1
                    c2 = cells[:, 2] + dx2
                    ok = ((c0 >= 0) & (c0 < dims[0])
                          & (c1 >= 0) & (c1 < dims[1])
                          & (c2 >= 0) & (c2 < dims[2]))
                    j = np.full(cells.shape[0], -1, dtype=np.int64)
                    if ok.any():
                        j[ok] = loc[c0[ok], c1[ok], c2[ok]]
                    # EVERY PAIR WITH AT LEAST ONE PARTIAL END, not just
                    # partial-partial. dL is nonzero as soon as ONE end
                    # differs from uniform, and those pairs dominate: a
                    # partial cell couples to the whole slab under it and
                    # to the ground plane, none of which are partial.
                    # (Pairing only partials recovered ~10%, measured.)
                    m = j >= 0
                    if not m.any():
                        continue
                    ii = np.nonzero(m)[0]
                    jj = j[ii]
                    keep = ispart[ii] | ispart[jj]
                    if not keep.any():
                        continue
                    ii = ii[keep]
                    jj = jj[keep]
                    T = table(orient, off)
                    uTu = (u @ T) @ u
                    V = np.empty((uf.size, uf.size), dtype=float)
                    for a in range(uf.size):
                        Ta = W[a] @ T
                        for b in range(uf.size):
                            V[a, b] = (Ta @ W[b]) - uTu
                    v = V[inv[ii], inv[jj]]
                    nz = v != 0.0
                    if not nz.any():
                        continue
                    parts.append((sel[ii[nz]].astype(np.int32),
                                  sel[jj[nz]].astype(np.int32),
                                  v[nz]))
    if not parts:
        return None
    n = fil_axis.size
    rows = np.concatenate([p[0] for p in parts])
    cols = np.concatenate([p[1] for p in parts])
    vals = np.concatenate([p[2] for p in parts])
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
