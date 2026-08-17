# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Surface-node prototype vs the BEM reference (dielectric phase 2).

SuperPEEC's excess-capacitance scheme rebuilt panel-exactly (same panel
enumeration and closed-form kernels as studies/bem_dielectric.py) so
the only variable is the CONDENSATION SCHEME:

  'today'   -- what SuperPEEC does now: every panel condenses onto its
               cell's centre node with fixed equal weights; dielectric
               free-surface panels share the cell node. Excess
               branches conductor<->cell and cell<->cell.
  'surface' -- the phase-2 fix candidate: each dielectric FREE-SURFACE
               panel keeps its own charge DOF (a surface node, weight
               1); the cell centre becomes a chargeless junction
               joined to each of its surface nodes by a HALF-CELL
               excess branch  C = eps0 (eps_r - 1) A_face / (d/2).

Port charge = conductor panel charge + excess-branch charge into the
body (the branch IS the (eps_r - 1) polarization share). Metric:
2-terminal Maxwell, identical to part D's readout and the BEM table.

Run: python3 studies/proto_surface_nodes.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from bem_dielectric import build_panels, rect_VE, EPS0, D

EPSR = 4.0


def panel_P(p):
    npan = len(p['ctr'])
    P = np.zeros((npan, npan))
    for j in range(npan):
        ax = int(np.flatnonzero(np.abs(p['nrm'][j]) > .5)[0])
        V, _ = rect_VE(p['ctr'][j], ax, p['ctr'])
        P[:, j] = V
    return P


def solve(ids, scheme):
    p = build_panels(ids, EPSR)
    npan = len(p['ctr'])
    Ppan = panel_P(p)
    dims = ids.shape

    nodes = {}

    def node(key):
        if key not in nodes:
            nodes[key] = len(nodes)
        return nodes[key]

    owner = np.zeros(npan, dtype=int)
    for k in range(npan):
        c = tuple(p['cell'][k])
        if p['kind'][k] == 0:
            owner[k] = node(('cell',) + c)
        elif scheme == 'surface':
            owner[k] = node(('surf', k))
        else:
            owner[k] = node(('cell',) + c)
    # dielectric centre nodes exist in both schemes (chargeless
    # junctions under 'surface' when the cell has no panels)
    dcells = np.argwhere(ids == 2)
    for c in dcells:
        node(('cell',) + tuple(c))
    nn = len(nodes)

    # excess branches
    branches = []
    cexc = 2*EPS0*(EPSR - 1.0)*D
    for c in dcells:
        me = node(('cell',) + tuple(c))
        for ax in range(3):
            for s in (+1, -1):
                nb = c.copy()
                nb[ax] += s
                inside = 0 <= nb[ax] < dims[ax]
                nid = ids[tuple(nb)] if inside else 0
                if nid == 1:
                    branches.append((node(('cell',) + tuple(nb)), me,
                                     cexc))
                elif nid == 2:
                    if s > 0:       # count each diel-diel pair once
                        branches.append((node(('cell',) + tuple(nb)),
                                         me, cexc/2.0))
                elif scheme == 'surface':
                    # free face -> half-cell branch to its surface node
                    pk = np.flatnonzero(
                        (p['kind'] == 1)
                        & (p['cell'] == c).all(axis=1)
                        & (p['nrm'][:, ax] == float(s)))
                    if pk.size:
                        branches.append((node(('surf', int(pk[0]))),
                                         me, cexc))

    W = np.zeros((npan, nn))
    for k in range(nn):
        sel = owner == k
        if sel.any():
            W[sel, k] = 1.0/sel.sum()
    Pn = W.T.dot(Ppan).dot(W)
    charged = np.flatnonzero((W != 0).any(axis=0))

    Cb = np.zeros((nn, nn))
    for a, b, cv in branches:
        Cb[a, a] += cv
        Cb[b, b] += cv
        Cb[a, b] -= cv
        Cb[b, a] -= cv

    # body of each conductor node
    bodyof = -np.ones(nn, dtype=int)
    for k in range(npan):
        if p['kind'][k] == 0:
            bodyof[owner[k]] = p['body'][k]
    nb_ = p['nbody']
    fix = np.flatnonzero(bodyof >= 0)
    free = np.array([k for k in range(nn) if bodyof[k] < 0], dtype=int)

    Cm = np.zeros((nb_, nb_))
    ne = charged.size
    Iall = np.eye(nn)
    A = np.block([
        [Pn[np.ix_(charged, charged)], -Iall[np.ix_(charged, free)]],
        [Iall[np.ix_(free, charged)], Cb[np.ix_(free, free)]]])
    rs = 1.0/np.maximum(np.abs(A).max(axis=1), 1e-300)
    lu = np.linalg.inv(A*rs[:, None])
    for col in range(nb_):
        phi = np.zeros(nn)
        phi[fix] = np.where(bodyof[fix] == col, 1.0, 0.0)
        rhs = np.concatenate([phi[charged],
                              -Cb[np.ix_(free, fix)].dot(phi[fix])])
        x = lu.dot(rhs*rs)
        q = np.zeros(nn)
        q[charged] = x[:ne]
        phin = phi.copy()
        phin[free] = x[ne:]
        for b in range(nb_):
            sel = np.flatnonzero(bodyof == b)
            Cm[b, col] = q[sel].sum() + Cb[sel].dot(phin).sum()
    return Cm


def twoterm(Cm):
    Ci = np.linalg.inv(Cm)
    return 1.0/(Ci[0, 0] + Ci[1, 1] - 2*Ci[0, 1])


def main():
    bem = {16: (1.5886, 3.0992), 24: (1.6386, 3.3106),
           32: (1.6690, 3.4369)}
    oct_now = {16: (1.062, 3.110), 24: (1.056, 3.398),
               32: (1.050, 3.580)}
    for n in (16, 24, 32):
        dims = (n, n, 4)
        base = np.zeros(dims, dtype=int)
        base[:, :, 0] = 1
        base[:, :, 3] = 1
        half = base.copy()
        half[:, :, 1] = 2
        fill = base.copy()
        fill[:, :, 1] = 2
        fill[:, :, 2] = 2
        for scheme in ('today', 'surface'):
            cv = twoterm(solve(base, scheme))
            ch = twoterm(solve(half, scheme))
            cf = twoterm(solve(fill, scheme))
            print("n=%2d %-7s: half %.4f  fill %.4f   (BEM %.4f/%.4f, "
                  "SuperPEEC-measured %.3f/%.3f)"
                  % (n, scheme, ch/cv, cf/cv, bem[n][0], bem[n][1],
                     oct_now[n][0], oct_now[n][1]), flush=True)


if __name__ == '__main__':
    main()
