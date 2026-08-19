# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Corner-patch Galerkin referee: judge circulation-mode subspaces BEFORE
any solver work (corner program phase 0).

THE QUESTION (from the Z-trace ladder, 2026-08-19): at dx/W = 0.5 the
coarse PWC basis gets the corner increment qualitatively WRONG (wrong
sign at high dx/delta), with corner-attributable R error 1.6-19%
through dx/delta = 1-6. Can a few solved net-zero circulation modes
per corner recover it? Kill criterion: if no <=4-mode family delivers
>=80% of the CORNER share of the coarse error at dx/delta <= 3, the
subspace idea is wrong.

FORMULATION -- discrete stream function. In-plane current on a slab is
the discrete curl of a scalar psi on fine-lattice vertices:
    I_xlink(i,j) = psi(top) - psi(bottom),  I_ylink = psi(L) - psi(R),
so EVERY psi field is KCL-exact, and:
  * the solver's coarse PWC cell-current basis == bilinear tents on
    coarse nodes (boundary coarse nodes carry the boundary data as the
    affine offset -- pinning the boundary at FINE granularity instead
    creates a bogus one-fine-cell current sheet, measured +1022%);
  * a net-zero corner mode == a compact psi bump near the corner;
  * the drive == boundary values: psi = 0 on one bank, I on the other,
    linear ramps across the end faces (uniform inflow).
Slab is THICK (T = W) with z-UNIFORM current: the honest in-plane
reduction of the ladder's 8x8 bar. z-redistribution is the conduction
engine's job and is excluded from BOTH truth and bases. (A thin slab
was measured to wash out the skin-regime corner physics: coarse R
error +0.1% at dx/delta = 2 vs the ladder's 5.8%.)
Mutuals: parallel links via terminal.box_mutual_matrix (staggered
center-to-center boxes); PERPENDICULAR PAIRS ARE ZERO -- L_xy is never
assembled. R = 1/(sigma*T) per link. Galerkin impedance is the
variational quadratic form Z = i^T (R + jwL) i / I^2 (unconjugated,
same as spx_galerkin/mode_referee).

DIFFERENTIAL METRIC (same trick as the ladder). The raw coarse error
of the L-junction mixes ARM lateral-crowding error (the engine's
domain, distributed over 6W of arms) with the corner error. A STRAIGHT
strip of the same width and centreline length (7W) through the same
machinery corrects the denominator to the CORNER SHARE:
    denom = (Zc_L - Zf_L) - (Zc_S - Zf_S),   dlv = (Zc_L - Ze_L)/denom
DC sanity: the true corner increment (Zf_L - Zf_S)/sheet must land at
Hall's 0.559 - 1 = -0.441 squares.

CANDIDATE MODE FAMILIES (psi bumps, zeroed on prescribed vertices):
  M1  corner-square-center pyramid (the J2d analog at corner scale).
      MEASURED NULL at F=12 thin-slab: 0.0% at every frequency -- kept
      as the recorded negative; modes must be ANCHORED AT THE CORNER.
  Mb1 tent at the inner corner vertex, radius dx.
  Mb2 tent at the inner corner vertex, radius 2dx.
  M2  inner-corner crowding: complex exp(-(1+j) r/delta), taper to
      zero at r = W. Retunes with frequency.
  M3a/M3b Gaussian partners (sigma = delta/sqrt2) straddling the
      corner, one per arm along the inner bank.
Sets: C0={M1} C1={Mb1} C2={Mb2} C3={M2} C4={Mb2,M2}
      C5={Mb2,M2,M3a,M3b}.

Usage:  PYTHONPATH=src python3 studies/corner_referee.py
Env: NC=2 (coarse cells across W -> dx/W = 1/NC), F=24 (fine cells
     across W; a/delta = (dx/delta)*NC/F caps trustworthy dx/delta),
     RATIOS="0.02,1,2,3,4,6" (dx/delta), W_UM=8, SIG=5.8e7.
"""
import os
import sys

sys.path.insert(0, 'src')
import numpy as np
from scipy.sparse import csr_matrix

import terminal as tm

MU0 = 4e-7*np.pi
SIG = float(os.environ.get('SIG', 5.8e7))
W_UM = float(os.environ.get('W_UM', 8.0))
NC = int(os.environ.get('NC', 2))
F = int(os.environ.get('F', 24))
RATIOS = [float(s) for s in
          os.environ.get('RATIOS', '0.02,1,2,3,4,6').split(',')]
W = W_UM*1e-6
a = W/F
dx = W/NC
K = F//NC
assert NC*K == F, "F must be a multiple of NC"
T = W
ARM = 3
SHEET = 1.0/(SIG*T)


class Trace:
    """Stream-function network on an occupancy mask."""

    def __init__(self, occ, bval):
        """``occ``: (nx, ny) bool cells. ``bval(i, j)``: prescribed psi
        at boundary vertex (i, j) in fine-lattice units."""
        self.occ = occ
        nx, ny = occ.shape
        occp = np.zeros((nx + 2, ny + 2), dtype=bool)
        occp[1:-1, 1:-1] = occ
        touch = (occp[1:, 1:] | occp[:-1, 1:]
                 | occp[1:, :-1] | occp[:-1, :-1])
        freev = (occp[1:, 1:] & occp[:-1, 1:]
                 & occp[1:, :-1] & occp[:-1, :-1])
        self.vs = np.argwhere(touch)
        self.nvert = len(self.vs)
        vid = -np.ones((nx + 1, ny + 1), dtype=int)
        for n, (i, j) in enumerate(self.vs):
            vid[i, j] = n
        self.isfree = freev[tuple(self.vs.T)]
        self.FREE = np.where(self.isfree)[0]
        self.PRES = np.where(~self.isfree)[0]
        self.psi_b = np.zeros(self.nvert)
        for n in self.PRES:
            i, j = self.vs[n]
            self.psi_b[n] = bval(i, j)

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
        C = csr_matrix((vals, (rows, cols)), shape=(nl, self.nvert))
        Lxx = tm.box_mutual_matrix(np.array(xlo), np.array(xhi), 0)
        Lyy = tm.box_mutual_matrix(np.array(ylo), np.array(yhi), 1)
        Lf = np.zeros((nl, nl))
        Lf[:nxl, :nxl] = Lxx
        Lf[nxl:, nxl:] = Lyy               # cross block: exactly zero
        del Lxx, Lyy
        Cd = C.toarray()
        self.GL = Cd.T @ (Lf @ Cd)
        self.GR = Cd.T @ ((1.0/(SIG*T))*Cd)
        del Lf, Cd
        for G in (self.GL, self.GR):
            assert abs(G - G.T).max() < 1e-8*abs(G).max()
        self.GL = 0.5*(self.GL + self.GL.T)
        self.GR = 0.5*(self.GR + self.GR.T)

        # coarse space: tents on the K-lattice; boundary nodes -> offset
        def tent(ci, cj):
            wx = np.maximum(0.0, 1 - np.abs(self.vs[:, 0] - ci)/float(K))
            wy = np.maximum(0.0, 1 - np.abs(self.vs[:, 1] - cj)/float(K))
            return wx*wy

        cols, off = [], np.zeros(self.nvert)
        for n, (i, j) in enumerate(self.vs):
            if i % K or j % K:
                continue
            if self.isfree[n]:
                cols.append(tent(i, j))
            elif self.psi_b[n] != 0.0:
                off += self.psi_b[n]*tent(i, j)
        self.Phi_c, self.off_c = np.array(cols).T, off
        assert abs(off[self.PRES] - self.psi_b[self.PRES]).max() < 1e-12

    def A(self, w):
        return self.GR + 1j*w*self.GL

    def fine_Z(self, A):
        ix = np.ix_(self.FREE, self.PRES)
        rhs = -(A[ix] @ self.psi_b[self.PRES])
        x = self.psi_b.astype(complex).copy()
        x[self.FREE] = np.linalg.solve(A[np.ix_(self.FREE, self.FREE)],
                                       rhs)
        return x @ (A @ x), x

    def galerkin_Z(self, A, extra=()):
        Phi = self.Phi_c.astype(complex)
        if len(extra):
            Phi = np.hstack([Phi] + [e[:, None] for e in extra])
        M = Phi.T @ (A @ Phi)
        rhs = -(Phi.T @ (A @ self.off_c.astype(complex)))
        aa, *_ = np.linalg.lstsq(M, rhs, rcond=1e-13)
        x = self.off_c.astype(complex) + Phi @ aa
        return x @ (A @ x), x

    def bump(self, fn):
        v = np.zeros(self.nvert, dtype=complex)
        um = a*1e6
        for m in self.FREE:
            i, j = self.vs[m]
            v[m] = fn(i*um, j*um)
        mx = np.abs(v).max()
        return v if mx == 0 else v/mx


# ------------------------------------------------------------ L-junction
nx, ny = (ARM + 1)*F, (ARM + 1)*F
occL = np.zeros((nx, ny), dtype=bool)
occL[:, :F] = True
occL[ARM*F:, :] = True
XI, YI = ARM*F, F                          # inner corner vertex


def bvalL(i, j):
    if (j == YI and i <= XI) or (i == XI and j >= YI):
        return 1.0                         # inner bank
    if i == 0:
        return j/float(F)                  # end ramp, arm A
    if j == ny:
        return (nx - i)/float(F)           # end ramp, arm B
    return 0.0                             # outer bank


# ------------------------------------- straight control, same centreline
nS = (2*ARM + 1)*F                         # 7W
occS = np.ones((nS, F), dtype=bool)


def bvalS(i, j):
    if j == F:
        return 1.0
    if j == 0:
        return 0.0
    return j/float(F)                      # both end ramps


def strip_profile(delta):
    """Exact 1-D strip AC correction: psi_true(v) - v/W for a width-W
    strip at skin depth delta (v = transverse position, 0..W). Zero at
    both banks; ~zero at DC. This is what the conduction engine
    delivers on straight runs -- the referee's ENGINE ANALOG."""
    k = (1 + 1j)/delta

    def c(v):
        vm = v*1e-6                        # um -> m
        return (np.sinh(k*(vm - W/2))/(2*np.sinh(k*W/2))
                + 0.5 - vm/W)
    return c


def engine_modes(tr, delta, kind):
    """Per-station transverse skin modes on every straight run: tent
    along the run (width dx) x the exact strip profile across it.
    kind 'L': arm A (v = y) stations x = K..3W, arm B (v = 4W - x)
    stations y = W..4W-K; kind 'S': stations along the full strip.
    With these in the BASELINE, arm crowding belongs to the baseline
    (as it does in the real solver) and corner modes are judged on the
    corner alone."""
    prof = strip_profile(delta)
    out = []

    def add(fn):
        v = tr.bump(fn)
        if np.abs(v).max() > 1e-9:
            out.append(v)

    Fu = K*a*1e6                           # dx in um

    def tent(u, cu):
        return max(0.0, 1 - abs(u - cu)/Fu)

    if kind == 'S':
        for ci in range(K, nS, K):
            cu = ci*a*1e6
            add(lambda x, y, cu=cu: tent(x, cu)*prof(y))
    else:
        for ci in range(K, ARM*F + 1, K):
            cu = ci*a*1e6
            add(lambda x, y, cu=cu: tent(x, cu)*prof(y))
        for cj in range(F, (ARM + 1)*F, K):
            cu = cj*a*1e6
            add(lambda x, y, cu=cu:
                tent(y, cu)*prof(4*W_UM - x))
    return out


def modes(tr, delta):
    """Corner-mode candidates, anchored at the inner corner."""
    Wu, du = W*1e6, min(delta*1e6, W_UM/2)
    xc, yc = ARM*Wu, Wu
    m1c = (xc + Wu/2, yc - Wu/2)
    m3b_c = (xc + 0.5*du, yc + 1.5*du)

    def m1(x, y):
        return max(0.0, 1 - abs(x - m1c[0])/(Wu/2)) * \
               max(0.0, 1 - abs(y - m1c[1])/(Wu/2))

    def mb(d):
        return lambda x, y: max(0.0, 1 - abs(x - xc)/d) * \
                            max(0.0, 1 - abs(y - yc)/d)

    def m2(x, y):
        r = np.hypot(x - xc, y - yc)
        return 0.0 if r >= Wu else np.exp(-(1 + 1j)*r/du)*(1 - r/Wu)

    def g(cx, cy):
        s = du/np.sqrt(2)
        return lambda x, y: np.exp(-((x - cx)**2 + (y - cy)**2)/(2*s*s))

    def mw(nu, rmax):
        """Laplace wedge harmonics of the 270-deg inner corner:
        r^nu sin(nu phi), nu = 2/3, 4/3, ... phi = 0 on the arm-A
        inner bank and 3pi/2 on the arm-B bank -- every harmonic
        vanishes on BOTH banks (sin(0), sin(nu*3pi/2) = sin(k pi)).
        The nu = 2/3 term is the exact leading DC corner singularity;
        tapered to zero by r = rmax."""
        Ru = rmax*1e6

        def f(x, y):
            r = np.hypot(x - xc, y - yc)
            if r >= Ru or r == 0.0:
                return 0.0
            phi = (np.arctan2(y - yc, x - xc) - np.pi) % (2*np.pi)
            return (r/Ru)**nu*np.sin(nu*phi)*(1 - r/Ru)**2
        return f

    def mo(x, y):
        """Outer-corner dead-zone harmonic: the convex 90-deg corner at
        (4W, 0) has wedge exponent nu = 2; psi ~ r^2 sin(2 phi_o),
        phi_o = 0 on the x = 4W bank, pi/2 on the y = 0 bank."""
        xo, yo = (ARM + 1)*Wu, 0.0
        r = np.hypot(x - xo, y - yo)
        if r >= Wu or r == 0.0:
            return 0.0
        phi = np.arctan2(y - yo, x - xo) - np.pi/2
        return (r/Wu)**2*np.sin(2*phi)*(1 - r/Wu)**2

    return dict(M1=tr.bump(m1), Mb1=tr.bump(mb(dx*1e6)),
                Mb2=tr.bump(mb(2*dx*1e6)), M2=tr.bump(m2),
                Mw1=tr.bump(mw(2.0/3, W)),
                Mw3=tr.bump(mw(2.0, W)),
                Mo=tr.bump(mo),
                M3a=tr.bump(g(xc - 1.5*du, yc - 0.5*du)),
                M3b=tr.bump(g(*m3b_c)))


def tabulated_modes(tr):
    """THE PHASE-1 CANDIDATE: true corner correction fields (fine minus
    coarse+engine solution), TAPERED to compact support -- the referee
    analog of tabulated per-corner-class shapes computed once offline
    from a small local solve. MEASURED (iteration 3): locality HOLDS
    (the DC table at radius W delivers 93.5% at DC, 98.7% at 2W), but
    ONE DC table degrades across frequency (166% overshoot at
    dx/delta=1, 19-57% at 2-6): the correction changes character with
    delta, so tabulate at several dx/delta and let Galerkin pick the
    amplitudes -- 3 tables at dx/delta = 0, 2, 4 here."""
    xc, yc = ARM*F, F                      # inner corner, lattice units
    r = np.hypot(tr.vs[:, 0] - xc, tr.vs[:, 1] - yc)/float(F)  # in W
    taper = np.maximum(0.0, 1 - r/2.0)
    out = {}
    for nm, ratio in (('T0', 1e-3), ('T2', 2.0), ('T4', 4.0)):
        delta = dx/ratio
        A = tr.A(2*np.pi/(np.pi*MU0*SIG*delta*delta))
        _, xf = tr.fine_Z(A)
        _, xb = tr.galerkin_Z(A, engine_modes(tr, delta, 'L'))
        v = (xf - xb)*taper
        v[tr.PRES] = 0.0
        out[nm] = v/np.abs(v).max()
    return out


# Measured nulls so far: M1 (centered tent), Mw2 (nu=4/3, antisymmetric
# about the bisector -- symmetry-exact zero), 2W-taper Mw1, Mw3, Mo
# (outer-corner dead zone: 1-12%). M2 (imposed complex phase) HURTS at
# AC. Analytic family ceiling ~62-90%; single-DC-table 93.5% at DC but
# not across frequency.
SETS = [('C1', ['T0']), ('C2', ['T2']), ('C3', ['T4']),
        ('C4', ['T0', 'T2']), ('C5', ['T0', 'T2', 'T4']),
        ('C6', ['T0', 'T2', 'T4', 'Mb1']),
        ('C7', ['Mw1', 'Mb1', 'Mb2']),
        ('C8', ['T0', 'T2', 'Mb1', 'Mb2'])]

print("corner referee: W=%g um, T=W, arms %dW, dx/W=1/%d, fine %d "
      "across (a/delta=ratio*%d/%d)" % (W_UM, ARM, NC, F, NC, F))
print("building L-junction + straight control ...", flush=True)
L = Trace(occL, bvalL)
S = Trace(occS, bvalS)
print("L: %d free verts, %d coarse DOF; S: %d free verts, %d coarse DOF"
      % (len(L.FREE), L.Phi_c.shape[1], len(S.FREE), S.Phi_c.shape[1]),
      flush=True)
print("DC anchor: true corner increment should be %.4g Ohm "
      "(0.559-1 squares)\n" % (-0.441*SHEET))
TAB = tabulated_modes(L)
hdr = "%-7s %-9s %-8s %-8s %-8s %-9s" % ("dx/dlt", "f[Hz]", "sq_true",
                                         "sq_crs", "sq_base", "crn_shr")
hdr += "".join("  %-8s" % ("dlv[%s]" % nm) for nm, _ in SETS)
print(hdr)
for ratio in RATIOS:
    delta = dx/ratio
    f = 1.0/(np.pi*MU0*SIG*delta*delta)
    w = 2*np.pi*f
    AL, AS = L.A(w), S.A(w)
    ZfL, ZcL = L.fine_Z(AL)[0], L.galerkin_Z(AL)[0]
    ZfS, ZcS = S.fine_Z(AS)[0], S.galerkin_Z(AS)[0]
    egL, egS = engine_modes(L, delta, 'L'), engine_modes(S, delta, 'S')
    ZbL = L.galerkin_Z(AL, egL)[0]
    ZbS = S.galerkin_Z(AS, egS)[0]
    denom = (ZbL.real - ZfL.real) - (ZbS.real - ZfS.real)
    mdL = modes(L, delta)
    mdL.update(TAB)
    line = "%-7g %-9.3g %-8.4f %-8.4f %-8.4f %-+8.2f%%" \
        % (ratio, f, 1 + (ZfL.real - ZfS.real)/SHEET,
           1 + (ZcL.real - ZcS.real)/SHEET,
           1 + (ZbL.real - ZbS.real)/SHEET, 100*denom/ZfL.real)
    for nm, keys in SETS:
        ZeL = L.galerkin_Z(AL, egL + [mdL[k] for k in keys])[0]
        dlv = (ZbL.real - ZeL.real)/denom
        line += "  %-8s" % ("%6.1f%%" % (100*dlv))
    print(line, flush=True)
print("\nsq_true/sq_crs/sq_base = corner effective squares: fine truth"
      "\n  vs bare coarse vs coarse+ENGINE-ANALOG baseline (DC truth"
      "\n  -> 0.559; the ladder pathology = coarse >> true at high f)."
      "\ncrn_shr = corner share of the BASELINE R error (dlv"
      " denominator, % of R_fine)."
      "\ndlv = share of that corner error fixed by adding the corner"
      " modes on top of the baseline. 100 = corner fixed.")
