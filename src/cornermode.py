# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Corner circulation modes: current turning at 90-degree bends without
a finer mesh (corner program phase 1).

THE PROBLEM (Z-trace ladder + corner referee, 2026-08-19): at coarse
grids the PWC basis cannot redistribute current around a bend -- the
corner-attributable R error is 3.2/5.8/19.1% of total R at 2 cells
across (1e5/1e9/1e10 Hz) and the corner increment has the WRONG SIGN.
The referee (studies/corner_referee.py) measured the fix: TABULATED
per-corner-class correction fields at three skin depths, tapered to 2W
support, deliver 91-99% of the corner error with 3 modes per corner.

DESIGN. Follows the net-zero mode architecture (enrich.Enrichment):

* A corner mode is a KCL-invisible current pattern over the SUB-BARS
  (in-plane transverse split, z unsplit) of the filaments in a compact
  patch around an inner-corner vertex. Per-filament net current is
  ZERO (weights are mean-removed per filament), so the aggregate
  identity holds and the mesh/Schur machinery never sees the modes.
* Shapes are TABULATED, not guessed: a small local stream-function
  solve (fine truth minus coarse-tent Galerkin, the referee distilled)
  on a canonical L-patch at the instance's physical dx and sigma, at
  dx/delta = 0, 2, 4. The fields carry the true local phase -- the
  referee measured imposed-guess phase HARMFUL but tabulated true
  phase fine. Amplitudes are SOLVED (C.1's lesson).
* The stream function maps exactly onto sub-bar currents: a solver
  x-filament at cell c is the tabulation lattice's link between cells
  c and c+x (same centre-to-centre box), and a sub-strip's current
  telescopes to psi(top) - psi(bottom). Handedness classes are the
  four vacuum quadrants; psi transforms with det(map) under the
  reflections.
* Mutuals fold from terminal.box_mutual_matrix over PARALLEL pairs
  only (perpendicular partial inductance is exactly zero), dense
  within the small patch + an rc_cross aggregate window -- the same
  radii discipline as the shipped engine (economy radii measured
  harmful in C.2).

MEASURED STRUCTURAL LIMIT (2026-08-19, and it is a THEOREM of the
architecture, not a defect of the shapes): net-zero modes DECOUPLE at
DC. The resistive mode<->aggregate cross term is exactly zero (the
aggregate current is uniform across sub-bars and the weights sum to
zero per filament), so mode amplitudes are driven only through jw
couplings and DC dissipation can only increase -- R(DC) is invariant
to the last digit (the same property that makes the skin engine's DC
gate exact). VoxHenry's linear basis shares it: J2d/J3d are
R-orthogonal to the PWC basis by parity, so their bends are AC-only
fixes too. The DC corner error (~3.2% of R at 2 cells across, O(h^1.4))
therefore needs a RESISTIVE correction -- the subpixel/sigma_eff
machinery is its natural home; docketed, not attempted here.

V1 SCOPE: xy-plane corners with z-uniform occupancy through the patch,
equal arm widths of 2..4 cells (wider corners have small error --
ladder: <= 1.2% at 8 across), uniform sigma, engine off (composition
with the engine = phase 2 -- NOTE phase 2 must
re-tabulate against a coarse+engine baseline, or the tables
double-count the near-corner crowding the engine fixes). Corners that
fail a check are SKIPPED with a warning, never fatal.

Interface: :func:`corner_palette` returns an ``enrich.FixedPalette``;
the solver wraps it in an ``enrich.Enrichment`` and, with the skin
engine present, an ``enrich.ModeStack``.
"""
import warnings

import numpy as np
import scipy.sparse as sp

import terminal as tm

MU0 = 4e-7*np.pi
_TABLES = {}                # (W_cells, dx, sigma, KF, ratios) -> fields


# --------------------------------------------------------------- tables
def _tabulate(W_cells, dx, sigma, KF=6, ratios=(1e-3, 2.0, 4.0),
              nz=None, engine_arm=None):
    """Canonical L-patch correction fields (fine minus coarse tents).

    Arms 3W, corner vertex at fine (3*WF, WF), WF = W_cells*KF fine
    cells per width; taper (1 - r/2W). Returns list of complex fields
    on the (nx+1, ny+1) fine-vertex grid, one per ratio, plus WF.

    ``engine_arm``: None tabulates against the bare coarse basis
    (engine-off solver). 'u' / 'v' adds an ENGINE ANALOG to the
    baseline -- per-station transverse modes carrying the exact 1-D
    strip skin profile -- on the u-arm (world x) or v-arm (world y)
    respectively, matching the coarse engine's SINGLE-AXIS coverage
    (engine modes live only on port-axis-parallel filaments).
    Without this the tables double-count the near-corner crowding the
    engine already fixes (phase-2 requirement)."""
    nz = W_cells if nz is None else int(nz)
    key = (int(W_cells), float(dx), float(sigma), int(KF),
           tuple(ratios), nz, engine_arm)
    if key in _TABLES:
        return _TABLES[key]
    a = dx/KF
    WF = W_cells*KF
    F, ARM = WF, 3
    nx = ny = (ARM + 1)*F
    occ = np.zeros((nx, ny), dtype=bool)
    occ[:, :F] = True
    occ[ARM*F:, :] = True
    T = nz*dx                               # thick slab, z-uniform

    occp = np.zeros((nx + 2, ny + 2), dtype=bool)
    occp[1:-1, 1:-1] = occ
    touch = (occp[1:, 1:] | occp[:-1, 1:] | occp[1:, :-1]
             | occp[:-1, :-1])
    freev = (occp[1:, 1:] & occp[:-1, 1:] & occp[1:, :-1]
             & occp[:-1, :-1])
    vs = np.argwhere(touch)
    nvert = len(vs)
    vid = -np.ones((nx + 1, ny + 1), dtype=int)
    for n, (i, j) in enumerate(vs):
        vid[i, j] = n
    isfree = freev[tuple(vs.T)]
    FREE, PRES = np.where(isfree)[0], np.where(~isfree)[0]
    XI, YI = ARM*F, F
    psi_b = np.zeros(nvert)
    for n in PRES:
        i, j = vs[n]
        if (j == YI and i <= XI) or (i == XI and j >= YI):
            psi_b[n] = 1.0
        elif i == 0:
            psi_b[n] = j/float(F)
        elif j == ny:
            psi_b[n] = (nx - i)/float(F)

    rows, cols, vals, xlo, xhi, ylo, yhi = [], [], [], [], [], [], []
    nl = 0
    for i in range(nx - 1):
        for j in range(ny):
            if occ[i, j] and occ[i + 1, j]:
                rows += [nl, nl]
                cols += [vid[i + 1, j + 1], vid[i + 1, j]]
                vals += [1.0, -1.0]
                xlo.append(((i + 0.5)*a, j*a, 0.0))
                xhi.append(((i + 1.5)*a, (j + 1)*a, T))
                nl += 1
    nxl = nl
    for i in range(nx):
        for j in range(ny - 1):
            if occ[i, j] and occ[i, j + 1]:
                rows += [nl, nl]
                cols += [vid[i, j + 1], vid[i + 1, j + 1]]
                vals += [1.0, -1.0]
                ylo.append((i*a, (j + 0.5)*a, 0.0))
                yhi.append(((i + 1)*a, (j + 1.5)*a, T))
                nl += 1
    C = sp.csr_matrix((vals, (rows, cols)), shape=(nl, nvert))
    Lf = np.zeros((nl, nl))
    Lf[:nxl, :nxl] = tm.box_mutual_matrix(np.array(xlo), np.array(xhi), 0)
    Lf[nxl:, nxl:] = tm.box_mutual_matrix(np.array(ylo), np.array(yhi), 1)
    Cd = C.toarray()
    GL = Cd.T @ (Lf @ Cd)
    GR = Cd.T @ ((1.0/(sigma*T))*Cd)
    del Lf, Cd
    GL, GR = 0.5*(GL + GL.T), 0.5*(GR + GR.T)

    K = KF
    tents, off = [], np.zeros(nvert)
    for n, (i, j) in enumerate(vs):
        if i % K or j % K:
            continue
        wx = np.maximum(0.0, 1 - np.abs(vs[:, 0] - i)/float(K))
        wy = np.maximum(0.0, 1 - np.abs(vs[:, 1] - j)/float(K))
        t = wx*wy
        if isfree[n]:
            tents.append(t)
        elif psi_b[n] != 0.0:
            off += psi_b[n]*t
    Phi = np.array(tents).T

    def _engine_analog(delta):
        """Per-station sinh strip-profile modes on ONE arm family."""
        if engine_arm is None:
            return []
        kw = (1 + 1j)/delta
        Wm = W_cells*dx

        def prof(v_um_frac):
            vm = v_um_frac*Wm
            return (np.sinh(kw*(vm - Wm/2))/(2*np.sinh(kw*Wm/2))
                    + 0.5 - vm/Wm)

        out = []
        iu = 0 if engine_arm == 'u' else 1
        stations = (range(K, ARM*F + 1, K) if engine_arm == 'u'
                    else range(F, (ARM + 1)*F, K))
        for cs in stations:
            t = np.maximum(0.0, 1 - np.abs(vs[:, iu] - cs)/float(K))
            # transverse coordinate across the arm's width, 0..1,
            # measured from the outer bank (psi = 0 side)
            if engine_arm == 'u':
                vfr = vs[:, 1]/float(F)
            else:
                vfr = ((ARM + 1)*F - vs[:, 0])/float(F)
            m = t*prof(np.clip(vfr, 0.0, 1.0))
            m[PRES] = 0.0
            if np.abs(m).max() > 1e-9:
                out.append(m)
        return out

    r = np.hypot(vs[:, 0] - XI, vs[:, 1] - YI)/float(WF)
    taper = np.maximum(0.0, 1 - r/2.0)
    fields = []
    for ratio in ratios:
        delta = dx/max(ratio, 1e-12)
        w = 2*np.pi/(np.pi*MU0*sigma*delta*delta)
        A = GR + 1j*w*GL
        rhs = -(A[np.ix_(FREE, PRES)] @ psi_b[PRES])
        xf = psi_b.astype(complex).copy()
        xf[FREE] = np.linalg.solve(A[np.ix_(FREE, FREE)], rhs)
        eng = _engine_analog(delta)
        Phi_b = (np.hstack([Phi.astype(complex)]
                           + [e[:, None] for e in eng])
                 if eng else Phi.astype(complex))
        Mg = Phi_b.T @ (A @ Phi_b)
        rhsg = -(Phi_b.T @ (A @ off.astype(complex)))
        aa, *_ = np.linalg.lstsq(Mg, rhsg, rcond=1e-13)
        xb = off.astype(complex) + Phi_b @ aa
        v = (xf - xb)*taper
        v[PRES] = 0.0
        grid = np.zeros((nx + 1, ny + 1), dtype=complex)
        grid[tuple(vs.T)] = v/np.abs(v).max()
        fields.append(grid)
    _TABLES[key] = (fields, WF)
    return _TABLES[key]


# ------------------------------------------------------------ detection
def find_corners(struc):
    """xy-plane inner-corner vertices with z-uniform occupancy.

    Returns list of (Ix, Iy, sx, sy, W_cells, zs): lattice vertex,
    signs pointing INTO the vacuum quadrant, equal arm width in cells,
    and the occupied z layers. Corners failing the v1 checks (unequal
    or out-of-range widths, z-nonuniform patch) are skipped with a
    warning."""
    occ = np.asarray(struc).astype(bool)
    nx, ny, nz = occ.shape
    o2 = occ.any(axis=2)
    uni = occ.all(axis=2) | ~o2             # per-column: full or empty
    out = []
    for Ix in range(1, nx):
        for Iy in range(1, ny):
            q = [o2[Ix - 1, Iy - 1], o2[Ix - 1, Iy],
                 o2[Ix, Iy - 1], o2[Ix, Iy]]
            if sum(q) != 3:
                continue
            miss = q.index(False)
            ax, ay = ((-1, -1), (-1, 0), (0, -1), (0, 0))[miss]
            sx, sy = 2*ax + 1, 2*ay + 1
            # arm widths: march away from the vacuum quadrant
            ci, cj = Ix - 1 if sx > 0 else Ix, Iy - 1 if sy > 0 else Iy
            wa = 0
            j = cj
            while 0 <= j < ny and o2[ci, j]:
                wa += 1
                j -= sy
            wb = 0
            i = ci
            while 0 <= i < nx and o2[i, cj]:
                wb += 1
                i -= sx
            if wa != wb or not 2 <= wa <= 4:
                warnings.warn("corner at (%d,%d): arm widths (%d,%d) "
                              "outside v1 scope (equal, 2..4) -- skipped"
                              % (Ix, Iy, wa, wb))
                continue
            R = 2*wa
            xs = slice(max(0, Ix - R), min(nx, Ix + R))
            ys = slice(max(0, Iy - R), min(ny, Iy + R))
            if not uni[xs, ys].all():
                warnings.warn("corner at (%d,%d): occupancy not "
                              "z-uniform in the patch -- skipped"
                              % (Ix, Iy))
                continue
            zcol = np.flatnonzero(occ[ci, cj, :])
            out.append((Ix, Iy, sx, sy, wa, zcol))
    return out


# -------------------------------------------------------------- palette
def corner_palette(model, M, fil_axis, fil_cell, engine_arm=None,
                   ratios=(1e-3, 2.0, 4.0), verbose=False):
    """The tabulated corner fields as an :class:`enrich.FixedPalette`,
    or None when the model has no eligible corner.

    Per corner: the patch filaments (both in-plane orientations within
    2W of the vertex, on the occupied z layers) become ENTRIES; each
    entry carries the ``KF`` in-plane sub-strip currents that the three
    tabulated stream functions telescope to (``psi(top) - psi(bottom)``
    per sub-strip, averaged over the fine planes inside the filament's
    centre-to-centre span), mean-removed per filament -- the net-zero
    invariant. The prolongation ``P`` ties field ``m`` of every entry of
    the corner into ONE unknown, so a corner is 3 solved amplitudes
    whatever its patch size. Handedness classes are the four vacuum
    quadrants; ``psi`` flips sign under a reflection. Overlapping
    patches enter the shared filament once per corner.
    """
    from enrich import FixedPalette
    corners = find_corners(model.struc())
    if not corners:
        if verbose:
            print("    corner modes: requested but no eligible corners "
                  "found")
        return None
    fil_cell = np.asarray(fil_cell)
    fil_axis = np.asarray(fil_axis)
    dx = model.dx
    sigma = model.uniform_sigma()
    KF = 6
    sel, Wf, prow, pcol, groups = [], [], [], [], []
    nmode = 0
    for cn, (Ix, Iy, sx, sy, Wc, zs) in enumerate(corners):
        # engine coverage maps to canonical arms axis-wise (the
        # handedness maps are axis-aligned): engine axis 0 covers the
        # u-arm, axis 1 the v-arm, anything else neither
        fields, WF = _tabulate(Wc, dx, sigma, KF, tuple(ratios),
                               nz=len(zs), engine_arm=engine_arm)
        sgn = float((-sx)*sy)               # psi flips under reflection
        R = 2*Wc
        inw = ((np.abs(fil_cell[:, 0] - Ix + 0.5) <= R)
               & (np.abs(fil_cell[:, 1] - Iy + 0.5) <= R)
               & (fil_axis < 2))
        idx = np.flatnonzero(inw & np.isin(fil_cell[:, 2], zs))
        if idx.size == 0:
            continue
        XI, YI = 3*WF, WF                   # canonical corner (fine)

        def psi(fld, xr, yr):
            """Tabulated psi at solver-relative in-plane point (xr, yr)
            in dx units; 0 outside the table."""
            u = int(round(XI + (-sx)*xr*KF))
            v = int(round(YI + sy*yr*KF))
            if 0 <= u < fld.shape[0] and 0 <= v < fld.shape[1]:
                return sgn*fld[u, v]
            return 0.0

        Wm = np.zeros((idx.size, KF, len(fields)), dtype=complex)
        for fi, f in enumerate(idx):
            ax = int(fil_axis[f])
            cx, cy = (fil_cell[f, 0] - Ix, fil_cell[f, 1] - Iy)
            for m, fld in enumerate(fields):
                for p in range(KF):
                    acc = 0.0
                    for s_ in range(KF):    # fine planes in the span
                        ln = (0.5 + (s_ + 1.0)/KF)
                        if ax == 0:
                            acc += (psi(fld, cx + ln, cy + (p+1)/KF)
                                    - psi(fld, cx + ln, cy + p/KF))
                        else:
                            acc -= (psi(fld, cx + (p+1)/KF, cy + ln)
                                    - psi(fld, cx + p/KF, cy + ln))
                    Wm[fi, p, m] = acc/KF
            Wm[fi] -= Wm[fi].mean(axis=0, keepdims=True)   # net-zero
        keep = np.abs(Wm).max(axis=(0, 1)) > 1e-12
        Wm[:, :, keep] /= np.abs(Wm[:, :, keep]).max(axis=(0, 1))
        Wm[:, :, ~keep] = 0.0
        e0 = len(sel)
        sel += list(idx)
        Wf.append(Wm)
        cols = np.flatnonzero(keep)
        for j, m in enumerate(cols):
            prow += list(np.arange(idx.size)*len(fields) + m + e0*len(fields))
            pcol += [nmode + j]*idx.size
        groups.append(np.arange(nmode, nmode + cols.size))
        nmode += cols.size
    if not sel:
        return None
    Wf = np.concatenate(Wf, axis=0)
    P = sp.csr_matrix((np.ones(len(prow)), (prow, pcol)),
                      shape=(Wf.shape[0]*Wf.shape[2], nmode))
    pal = FixedPalette(sel, Wf, P, groups)
    pal.corners = corners
    pal.k_in = KF
    return pal
