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

v1 scope: filaments along the cylinder axis only (the dominant
current direction of the geometry class); transverse-filament and
cross-orientation corrections belong to stage C's fine-lattice
machinery, which will reuse exactly this table structure.
"""
import numpy as np
import scipy.sparse as sp

from terminal import box_mutual_matrix


def build_dL(model, M, window=2):
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
    # a filament's weights: mean of its two half-cells' sub-fills
    # (identical for a constant-cross-section cylinder); cells not in
    # the subpixel dict are FULL (uniform weights)
    u = np.full(k*k, 1.0/(k*k))

    def wvec(c):
        tw = spx['cells'].get((int(c[t1]), int(c[t2])))
        if tw is None:
            return None                      # full cell
        w = tw.ravel().astype(float)
        s = w.sum()
        return w/s if s > 0 else None

    weights = {}
    partial = []
    for n, c in enumerate(cells):
        wa = wvec(c)
        cb = c.copy()
        cb[axis] += 1
        wb = wvec(cb)
        if wa is None and wb is None:
            continue
        wa = u if wa is None else wa
        wb = u if wb is None else wb
        weights[n] = 0.5*(wa + wb)
        partial.append(n)
    if not partial:
        return None

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
                    dl = float(wn @ B @ wm) - float(u @ B @ u)
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
    if not vals:
        return None
    return sp.csr_matrix((vals, (rows, cols)), shape=(efg, efg))
