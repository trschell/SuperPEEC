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

DESIGN. Follows the net-zero mode architecture (Redistribution / C.2):

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
with Redistribution/SubpixelModes = phase 2 -- NOTE phase 2 must
re-tabulate against a coarse+engine baseline, or the tables
double-count the near-corner crowding the engine fixes). Corners that
fail a check are SKIPPED with a warning, never fatal.

Interface: duck-types the solver's ``redist`` slot (use_fft=False
path): nmode/sel/Ru/Zuu/Zcross/Zt/set_frequency/mode_precond.
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
    (Redistribution modes live only on port-axis-parallel filaments).
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


# ---------------------------------------------------------------- modes
class CornerModes:
    """Tabulated corner circulation modes; duck-types the solver's
    ``redist`` slot (dense/sparse path, ``use_fft = False``)."""

    use_fft = False
    Zt = None

    def __init__(self, model, M, fil_axis, fil_cell, rc_cross=4,
                 ratios=(1e-3, 2.0, 4.0), verbose=False,
                 engine_arm=None):
        self.dx = model.dx
        self.sigma = model.uniform_sigma()
        self.engine_arm = engine_arm
        corners = find_corners(model.struc())
        self.corners = corners
        self.ratios = tuple(ratios)
        self.ncorner = len(corners)
        self.kk = (0, 0)                    # print compatibility
        self.nnz = (0, 0)
        self.ntable = 0
        if not corners:
            self.nmode = 0
            self.sel = np.zeros(0, dtype=np.int64)
            return
        fil_cell = np.asarray(fil_cell)
        fil_axis = np.asarray(fil_axis)

        # one KF/k_in per construction: sub-strips == tabulation columns
        self.KF = 6
        k_in = self.KF

        # ---- per-corner patches: filaments, sub-strips, mode weights
        # global lists across corners (sub-bar space is their union)
        pf_idx = []                         # filament index per patch fil
        pf_axis = []
        pf_cell = []
        pf_corner = []
        weights = []                        # per corner: (npf*k_in, nmod)
        for cn, (Ix, Iy, sx, sy, Wc, zs) in enumerate(corners):
            # engine coverage maps to canonical arms axis-wise (the
            # handedness maps are axis-aligned): engine axis 0 covers
            # the u-arm, axis 1 the v-arm, anything else neither
            fields, WF = _tabulate(Wc, self.dx, self.sigma, self.KF,
                                   self.ratios, nz=len(zs),
                                   engine_arm=engine_arm)
            sgn = float((-sx)*sy)           # psi flips under reflection
            R = 2*Wc
            inw = ((np.abs(fil_cell[:, 0] - Ix + 0.5) <= R)
                   & (np.abs(fil_cell[:, 1] - Iy + 0.5) <= R)
                   & (fil_axis < 2))
            idx = np.flatnonzero(inw & np.isin(fil_cell[:, 2], zs))
            if idx.size == 0:
                continue
            XI, YI = 3*WF, WF               # canonical corner (fine)

            def psi(fld, xr, yr):
                """Tabulated psi at solver-relative in-plane point
                (xr, yr) in dx units; 0 outside the table."""
                u = int(round(XI + (-sx)*xr*self.KF))
                v = int(round(YI + sy*yr*self.KF))
                if 0 <= u < fld.shape[0] and 0 <= v < fld.shape[1]:
                    return sgn*fld[u, v]
                return 0.0

            Wm = np.zeros((idx.size*k_in, len(fields)), dtype=complex)
            for fi, f in enumerate(idx):
                ax = int(fil_axis[f])
                cx, cy = (fil_cell[f, 0] - Ix, fil_cell[f, 1] - Iy)
                for m, fld in enumerate(fields):
                    for p in range(k_in):
                        acc = 0.0
                        # planes: the KF integer fine planes inside the
                        # centre-to-centre span (KF even -> integers)
                        for s in range(self.KF):
                            ln = (0.5 + (s + 1.0)/self.KF)
                            if ax == 0:
                                acc += (psi(fld, cx + ln, cy + (p+1)/k_in)
                                        - psi(fld, cx + ln, cy + p/k_in))
                            else:
                                acc -= (psi(fld, cx + (p+1)/k_in, cy + ln)
                                        - psi(fld, cx + p/k_in, cy + ln))
                        Wm[fi*k_in + p, m] = acc/self.KF
                # per-filament net-zero: the load-bearing invariant
                blk = slice(fi*k_in, (fi + 1)*k_in)
                Wm[blk] -= Wm[blk].mean(axis=0, keepdims=True)
            keep = np.abs(Wm).max(axis=0) > 1e-12
            Wm = Wm[:, keep]/np.abs(Wm[:, keep]).max(axis=0)
            pf_idx.append(idx)
            pf_axis.append(fil_axis[idx])
            pf_cell.append(fil_cell[idx])
            pf_corner.append(np.full(idx.size, cn))
            weights.append(Wm)
        if not weights:
            self.nmode = 0
            self.sel = np.zeros(0, dtype=np.int64)
            return
        pf_idx = np.concatenate(pf_idx)
        pf_axis = np.concatenate(pf_axis)
        pf_cell = np.concatenate(pf_cell, axis=0)
        pf_corner = np.concatenate(pf_corner)
        npf = pf_idx.size
        # block-diagonal mode weights over the union sub-bar space
        self.nmode = sum(w.shape[1] for w in weights)
        Wall = np.zeros((npf*k_in, self.nmode), dtype=complex)
        self._blocks = []                   # (mode slice) per corner
        r0, m0 = 0, 0
        for w in weights:
            nf, nm = w.shape[0]//k_in, w.shape[1]
            Wall[r0:r0 + nf*k_in, m0:m0 + nm] = w
            self._blocks.append(slice(m0, m0 + nm))
            r0 += nf*k_in
            m0 += nm

        # ---- sub-bar boxes (in-plane split only, z unsplit)
        dx = self.dx
        ns = npf*k_in
        lo = np.zeros((ns, 3))
        hi = np.zeros((ns, 3))
        for fi in range(npf):
            ax = int(pf_axis[fi])
            tr = 1 - ax                     # the in-plane transverse axis
            c = pf_cell[fi]
            for p in range(k_in):
                a = [c[0]*dx, c[1]*dx, c[2]*dx]
                b = [(c[0]+1)*dx, (c[1]+1)*dx, (c[2]+1)*dx]
                a[ax] = (c[ax] + 0.5)*dx
                b[ax] = (c[ax] + 1.5)*dx
                a[tr] = c[tr]*dx + p*dx/k_in
                b[tr] = c[tr]*dx + (p + 1)*dx/k_in
                lo[fi*k_in + p] = a
                hi[fi*k_in + p] = b

        # ---- aggregate window (both in-plane axes, all z)
        selm = np.zeros(len(fil_axis), dtype=bool)
        for (Ix, Iy, sx, sy, Wc, zs) in corners:
            Rw = 2*Wc + int(rc_cross)
            selm |= ((np.abs(fil_cell[:, 0] - Ix + 0.5) <= Rw)
                     & (np.abs(fil_cell[:, 1] - Iy + 0.5) <= Rw)
                     & (fil_axis < 2))
        self.sel = np.flatnonzero(selm)
        nsel = self.sel.size
        slo = np.zeros((nsel, 3))
        shi = np.zeros((nsel, 3))
        for si, f in enumerate(self.sel):
            c = fil_cell[f]
            d = int(fil_axis[f])
            slo[si] = [c[0]*dx, c[1]*dx, c[2]*dx]
            shi[si] = [(c[0]+1)*dx, (c[1]+1)*dx, (c[2]+1)*dx]
            slo[si][d] = (c[d] + 0.5)*dx
            shi[si][d] = (c[d] + 1.5)*dx

        # ---- L blocks, per orientation (perpendicular pairs are zero)
        sub_ax = np.repeat(pf_axis, k_in)
        Lsub = np.zeros((ns, ns))
        Lx = np.zeros((ns, nsel))
        for ax in (0, 1):
            si = np.flatnonzero(sub_ax == ax)
            gi = np.flatnonzero(np.asarray(fil_axis)[self.sel] == ax)
            if si.size == 0:
                continue
            both_lo = np.vstack([lo[si], slo[gi]])
            both_hi = np.vstack([hi[si], shi[gi]])
            Lb = tm.box_mutual_matrix(both_lo, both_hi, ax)
            Lsub[np.ix_(si, si)] = Lb[:si.size, :si.size]
            if gi.size:
                Lx[np.ix_(si, gi)] = Lb[:si.size, si.size:]
        self.Zuu = Wall.T @ (Lsub @ Wall)
        self.Zcross = Wall.T @ Lx
        r_sub = k_in/(self.sigma*dx)
        self.Ru = sp.csr_matrix((Wall.T*r_sub) @ Wall)
        self.nnz = (self.Zuu.size, self.Zcross.size)
        self.ntable = len(self.ratios)
        self.kk = (k_in, 1)
        self._Wall = Wall
        self._pf_idx = pf_idx
        self._pf_axis = pf_axis
        self._pf_cell = pf_cell
        self._k_in = k_in
        self._lo, self._hi = lo, hi
        self._sub_ax = sub_ax

    # -- solver interface ----------------------------------------------
    def set_frequency(self, freq):
        """Tables span the frequency band by construction; the solved
        amplitudes retune, the shapes do not."""
        return False

    def mode_precond(self, jw):
        """Per-corner block-Jacobi inverse of (Ru + jw Zuu)."""
        A = self.Ru.toarray() + jw*self.Zuu
        out = np.zeros_like(A)
        for b in self._blocks:
            out[b, b] = np.linalg.inv(A[b, b])
        return out


# ---------------------------------------------------------------- stack
class ModeStack:
    """Engine + corner modes as ONE duck-typed ``redist`` object.

    u = [u_engine; u_corner]. The engine keeps its own apply path (FFT
    or sparse); corner blocks are dense-small; the engine<->corner
    mode-mode coupling Zec is INCLUDED (the C.2 lesson: dropping
    mode-mode dipole couplings over-crowds) but truncated to engine
    filaments INSIDE the corner patches -- beyond the patch the
    coupling is dipole-dipole 1/r^3, the same class the engine itself
    truncates at rc_uu. The raw geometry of Zec is cached; an engine
    retune (conduction shapes track the solve frequency) only refolds
    the weights.

    The stack always advertises ``use_fft = True`` and routes the
    engine inside :meth:`apply_fft`, so the solver's FFT branch is the
    single integration point."""

    use_fft = True

    def __init__(self, engine, corner):
        import greens
        self.engine = engine
        self.corner = corner
        self._greens = greens
        # aggregate window: union of the two sel sets
        self.sel = np.union1d(engine.sel, corner.sel)
        self._pe = np.searchsorted(self.sel, engine.sel)
        self._pc = np.searchsorted(self.sel, corner.sel)
        # engine<->corner cross geometry: corner sub-bars parallel to
        # the engine axis x engine sub-bars of PATCH filaments
        a0 = engine.axis
        cs = np.flatnonzero(corner._sub_ax == a0)
        self._cs = cs
        epos = []
        if cs.size:
            pcell = corner._pf_cell
            # engine.sel positions whose cell is a corner-patch cell
            patch = {tuple(c) for c in pcell}
            for p, c in enumerate(engine.cells):
                if tuple(c) in patch:
                    epos.append(p)
        self._epos = np.asarray(epos, dtype=int)
        self._Lec_raw = None
        if cs.size and self._epos.size:
            elo, ehi = engine._sub_boxes(engine.cells[self._epos])
            clo, chi = corner._lo[cs], corner._hi[cs]
            ncs, nes = cs.size, elo.shape[0]
            ii = np.repeat(np.arange(ncs), nes)
            jj = np.tile(np.arange(nes), ncs)
            S = greens.box_pair_stencil_pairs(clo[ii], chi[ii],
                                              elo[jj], ehi[jj])
            oth = [c for c in range(3) if c != a0]
            ca = ((chi - clo)[:, oth[0]]*(chi - clo)[:, oth[1]])
            ea = ((ehi - elo)[:, oth[0]]*(ehi - elo)[:, oth[1]])
            self._Lec_raw = (S/(ca[ii]*ea[jj])).reshape(ncs, nes)
        self._restack()

    def _restack(self):
        """(Re)build everything that depends on the engine's W/km."""
        import scipy.sparse as _sp
        e, c = self.engine, self.corner
        self.ne, self.nc = e.nmode, c.nmode
        self.nmode = self.ne + self.nc
        self.Ru = _sp.block_diag([e.Ru, c.Ru], format='csr')
        self.Zt = None
        if e.Zt is not None:
            Zt_e = e.Zt.toarray() if _sp.issparse(e.Zt) else \
                np.asarray(e.Zt)
            self.Zt = np.vstack([Zt_e,
                                 np.zeros((self.nc, Zt_e.shape[1]),
                                          dtype=Zt_e.dtype)])
        # fold Zec: (engine modes) x (corner modes)
        self.Zec = None
        if self._Lec_raw is not None:
            ke, km = e.k, e.km
            nes = self._epos.size
            # per patch filament: W_e^T (km x ke) @ L^T (ke x ncs)
            fold = np.zeros((nes*km, self._cs.size), dtype=complex)
            for q in range(nes):
                blk = self._Lec_raw[:, q*ke:(q + 1)*ke]
                fold[q*km:(q + 1)*km] = e.W.T @ blk.T
            Wc = c._Wall[self._cs]
            Zec_full = np.zeros((e.nmode_full, c.nmode), dtype=complex)
            rows = (np.repeat(self._epos*km, km)
                    + np.tile(np.arange(km), nes))
            Zec_full[rows] = fold @ Wc
            self.Zec = Zec_full[e.mode_mask]
        # print compatibility
        self.kk = e.kk
        self.nnz = (getattr(e, 'nnz', (0, 0))[0] + c.nnz[0],
                    getattr(e, 'nnz', (0, 0))[1] + c.nnz[1])
        self.ntable = getattr(e, 'ntable', 0) + c.ntable

    def apply_fft(self, u, i_f):
        """(Zuu@u + Zcross@i_f, Zcross.T@u) over the stacked layout;
        ``i_f`` is the aggregate slice over the UNION sel."""
        e, c = self.engine, self.corner
        ue, uc = u[:self.ne], u[self.ne:]
        if e.use_fft:
            mue, mfe = e.apply_fft(ue, np.ascontiguousarray(
                i_f[self._pe]))
        else:
            mue = e.Zuu @ ue + e.Zcross @ i_f[self._pe]
            mfe = e.Zcross.T @ ue
        muc = c.Zuu @ uc + c.Zcross @ i_f[self._pc]
        mfc = c.Zcross.T @ uc
        if self.Zec is not None:
            mue = mue + self.Zec @ uc
            muc = muc + self.Zec.T @ ue
        mf = np.zeros(self.sel.size, dtype=np.complex128)
        np.add.at(mf, self._pe, mfe)
        np.add.at(mf, self._pc, mfc)
        return np.concatenate([mue, muc]), mf

    def set_frequency(self, freq):
        changed = self.engine.set_frequency(freq)
        if changed:
            self._restack()
        return bool(changed)

    def mode_precond(self, jw):
        """Block-diagonal: the engine's own preconditioner when it has
        one (identity otherwise -- the coarse engine's status quo) and
        the corner blocks' exact inverse."""
        import scipy.sparse as _sp
        Pc = self.corner.mode_precond(jw)
        Pe = None
        if hasattr(self.engine, 'mode_precond'):
            Pe = self.engine.mode_precond(jw)
        if Pe is None:
            Pe = _sp.identity(self.ne, format='csr',
                              dtype=np.complex128)
        return _sp.block_diag([_sp.csr_matrix(Pe),
                               _sp.csr_matrix(Pc)], format='csr')
