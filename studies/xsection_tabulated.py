# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Evidence experiment: are TABULATED section profiles the road past the
conduction palette's ceiling? (rank-2, run in scratchpad -- not repo)

Extends studies/modebasis2d.py (the harness that originally justified
the conduction basis) with tabulated per-section-class modes, mirroring
the corner program's measured design:
  cond4       shipped palette (4 shapes)
  cond8       + individual faces/corners (the P1 ablation variant)
  tab1(same)  ONE mode = the fine solution's profile aggregated to the
              k x k sub-bar grid at the SAME frequency (the ceiling of
              tabulation at this sub-bar resolution)
  tab1(4.8)   one FIXED table (tabulated at w/d=4.8) across the band
              (the corner referee's measured failure mode: one table
              does not span frequency)
  tab3        tables at w/d = 2, 4.8, 10 used at every frequency (the
              corner program's shipped design, cross-section version)
CLASS TEST (2:1 rectangle): tab3 tabulated on the rectangle itself vs
tab3 tabulated on the SQUARE and reused index-wise (wrong class).

Metric identical to modebasis2d: err% = (R_basis - R_ref)/(R_ref - R_dc),
0 = exact, -100 = captured none of the skin correction.
"""
import sys

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import modebasis2d as mb

MU0, SIGMA = mb.MU0, mb.SIGMA
RATIOS = (2.0, 3.0, 4.8, 7.0, 10.0)
KX = KY = 8


def geometry(nx, ny, wx, wy):
    ax, ay = wx/nx, wy/ny
    cx = (np.arange(nx) + 0.5)*ax
    cy = (np.arange(ny) + 0.5)*ay
    X, Y = np.meshgrid(cx, cy, indexing='ij')
    return X.ravel(), Y.ravel(), ax, ay


def impedance(nx, ny, wx, wy, freq):
    x, y, ax, ay = geometry(nx, ny, wx, wy)
    d = np.sqrt((x[:, None] - x[None, :])**2
                + (y[:, None] - y[None, :])**2)
    np.fill_diagonal(d, 0.44705*np.sqrt(ax*ay))
    L = -(MU0/(2*np.pi))*np.log(d/max(wx, wy))
    R = np.diag(np.full(x.size, 1.0/(SIGMA*ax*ay)))
    return R + 1j*2*np.pi*freq*L


def submap(nx, ny):
    rx, ry = nx//KX, ny//KY
    I, J = np.meshgrid(np.arange(nx)//rx, np.arange(ny)//ry,
                       indexing='ij')
    return (I*KY + J).ravel()


def cond_basis(nx, ny, wx, wy, delta, individual):
    x, y, _, _ = geometry(nx, ny, wx, wy)
    p = (1 + 1j)/delta
    pc = p/np.sqrt(2.0)
    xx = {0: x, 1: wx - x}
    yy = {0: y, 1: wy - y}
    if individual:
        shapes = [np.exp(-p*xx[0]), np.exp(-p*xx[1]),
                  np.exp(-p*yy[0]), np.exp(-p*yy[1])]
        for sx in (0, 1):
            for sy in (0, 1):
                shapes.append(np.exp(-pc*(xx[sx] + yy[sy])))
    else:
        ex = np.exp(-p*xx[0]) + np.exp(-p*xx[1])
        ey = np.exp(-p*yy[0]) + np.exp(-p*yy[1])
        corner = sum(np.exp(-pc*(xx[sx] + yy[sy]))
                     for sx in (0, 1) for sy in (0, 1))
        shapes = [ex + ey, np.exp(-p*xx[0]) - np.exp(-p*xx[1]),
                  np.exp(-p*yy[0]) - np.exp(-p*yy[1]), corner]
    cols = []
    for c in shapes:
        for part in (c.real, c.imag):
            part = part - part.mean()
            if np.abs(part).max() > 1e-30:
                cols.append(part)
    W = np.stack(cols, axis=1)
    return W - W.mean(axis=0, keepdims=True)


def tab_mode(Z, sub, nsub):
    """The fine profile aggregated to the sub-bar grid, as 2 real
    net-zero columns."""
    i = np.linalg.solve(Z, np.ones(Z.shape[0], dtype=complex))
    prof = np.zeros(Z.shape[0], dtype=complex)
    for s in range(nsub):
        m = sub == s
        prof[m] = i[m].mean()
    cols = []
    for part in (prof.real, prof.imag):
        part = part - part.mean()
        if np.abs(part).max() > 1e-30:
            cols.append(part/np.abs(part).max())
    return np.stack(cols, axis=1)


def run(name, nx, ny, wx, wy, tab_from=None):
    """tab_from: (nx2,ny2,wx2,wy2) to tabulate on a DIFFERENT section
    (wrong-class test); default tabulates on this section."""
    sub = submap(nx, ny)
    nsub = KX*KY
    Zs, Rref = {}, {}
    for r in RATIOS:
        delta = wx/r
        f = 1.0/(np.pi*MU0*SIGMA*delta*delta)
        Zs[r] = impedance(nx, ny, wx, wy, f)
        Rref[r] = mb.solve_Z(Zs[r]).real
    Rdc = 1.0/(SIGMA*wx*wy)

    def tab_at(rt):
        if tab_from is None:
            return tab_mode(Zs[rt], sub, nsub)
        nx2, ny2, wx2, wy2 = tab_from
        delta = wx2/rt
        f = 1.0/(np.pi*MU0*SIGMA*delta*delta)
        Z2 = impedance(nx2, ny2, wx2, wy2, f)
        sub2 = submap(nx2, ny2)
        Wsub = tab_mode(Z2, sub2, nsub)      # fine cols on section 2
        # compress to per-sub-bar weights, lift onto THIS section
        out = np.zeros((nx*ny, Wsub.shape[1]))
        for s in range(nsub):
            m2 = sub2 == s
            out[sub == s] = Wsub[m2].mean(axis=0)
        return out - out.mean(axis=0, keepdims=True)

    cands = [('cond4 (shipped)',
              lambda r: cond_basis(nx, ny, wx, wy, wx/r, False)),
             ('cond8 (P1 split)',
              lambda r: cond_basis(nx, ny, wx, wy, wx/r, True)),
             ('tab1 (same freq)', lambda r: tab_at(r)),
             ('tab1 (fixed 4.8)', lambda r: tab_at(4.8)),
             ('tab3 (2,4.8,10)',
              lambda r: np.hstack([tab_at(2.0), tab_at(4.8),
                                   tab_at(10.0)]))]
    print("\n== %s ==" % name)
    print("  %-18s %-3s" % ("basis", "km")
          + "".join("  w/d=%-5.1f" % r for r in RATIOS))
    for label, fn in cands:
        cells, km = [], 0
        for r in RATIOS:
            Wf = fn(r)
            km = Wf.shape[1]
            T = np.concatenate([np.ones((nx*ny, 1))/(nx*ny), Wf],
                               axis=1).astype(complex)
            Rb = mb.solve_Z(Zs[r], T).real
            cells.append("  %-+9.1f" % (100*(Rb - Rref[r])
                                        / (Rref[r] - Rdc)))
        print("  %-18s %-3d" % (label, km) + "".join(cells))


W = 1e-5
run("square %gx%g um, fine 48x48, sub 8x8" % (W*1e6, W*1e6),
    48, 48, W, W)
run("rect 2:1 %gx%g um (tables from ITSELF)" % (W*1e6, W*5e5*1e0),
    48, 24, W, W/2)
run("rect 2:1 (tables from the SQUARE -- wrong class)",
    48, 24, W, W/2, tab_from=(48, 48, W, W))
print("\nerr%%: 0 = exact, -100 = none of the skin correction captured.")
