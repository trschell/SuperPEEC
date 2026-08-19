# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Zero-truncation Galerkin referee for subpixel enrichment bases.

The offline harness that settled C.2 phase 2 (2026-08-18): build the
DENSE sub-bar impedance of a short cylinder wire (no truncation, no
solver machinery, no FMM), then compare -- against the fine sub-bar
truth, under the same translation-symmetric drive -- the Galerkin
projections onto (a) aggregates only (= stage B) and (b) aggregates
plus the per-cell SubpixelModes weights. Measured verdict: the mode
subspace tracks fine truth to ~2% through dx/delta = 8 where the
coarse basis collapses to -54%, which exonerated the basis and
localized the 3-D deep-skin overshoot to solver-side effects
(convergence -- fixed by the mode block-Jacobi -- then protocol and
transverse-path physics; see the docket and doctrine rule 14).

Reuse it whenever a new mode family (transverse, edge-anchored,
cross-orientation) needs judging BEFORE solver integration -- the
same xsection.py discipline, one level up.
"""

import sys, math
sys.path.insert(0, 'src')
import numpy as np, sppeec_input, equiterminal
import terminal as tm
from scipy.special import jv

NX, DX, R = 12, 1e-6, 3e-6
def doc():
    return '\n'.join(['[grid]', 'dims = [%d, 8, 8]' % NX, 'pitch = 1e-6',
        '[[cylinder]]', 'axis = "x"', 'center = [4e-6, 4e-6]',
        'radius = 3e-6', 'sigma = 5.8e7',
        '[port]', 'equipotential = true',
        'p_faces = [[0, 3, 3, "-x"]]', 'n_faces = [[%d, 3, 3, "+x"]]' % (NX-1),
        '[solve]', 'freq = [1e7]'])

prob = sppeec_input.loads(doc())
m = prob.model()
sw = prob.sweeper(m, prob.tree(m))
# the engine at k=4 for a tractable dense space (axis 0 = x)
r2 = equiterminal.SubpixelModes(m, sw.S.M, 0, sw.S.fil_axis,
                                sw.S.fil_cell, k=4, term=None,
                                skin_freq=1e7)
n, k = r2.nfil, r2.k
ns = n*k
print('%d filaments, %d sub-bars' % (n, ns), flush=True)
L = tm.box_mutual_matrix(r2.lo, r2.hi, r2.axis)     # dense, area-normalised
# fill-weighted sub-bar resistance (l = dx)
rvec = np.zeros(ns)
base = k/(r2.sigma*r2.dx)
for f, key in enumerate(r2._tkey):
    pc = r2._percell[key]
    fill = pc['fill']
    rr = np.full(k, np.inf)
    sup = fill > 1e-3
    rr[sup] = base/fill[sup]
    rvec[f*k:(f+1)*k] = rr
free = np.isfinite(rvec)                            # exclude empty slivers

def wire_Z(A_L, A_R, basis, w):
    """Impedance under same-E drive + prescribed total current."""
    nb = basis.shape[1]
    A = basis.T @ (np.diag(A_R) + 1j*w*A_L) @ basis
    c = basis.T @ np.ones(basis.shape[0])           # current content
    M = np.zeros((nb+1, nb+1), dtype=complex)
    M[:nb, :nb] = A
    M[:nb, nb] = -c
    M[nb, :nb] = c
    rhs = np.zeros(nb+1, dtype=complex)
    rhs[nb] = 1.0
    x = np.linalg.solve(M, rhs)
    return x[nb]                                    # E*len per unit I -> Z

def bases():
    Bc = np.zeros((ns, n))
    for f in range(n):
        Bc[f*k:(f+1)*k, f] = r2.G[f]
    cols, colf = [], []
    for f in range(n):
        for c in range(r2.km):
            if r2.mode_mask[f*r2.km + c]:
                v = np.zeros(ns)
                v[f*k:(f+1)*k] = r2.Wf[f][:, c]
                cols.append(v)
    Be = np.concatenate([Bc, np.stack(cols, axis=1)], axis=1) if cols else Bc
    return Bc, Be

rho, mu0 = 1/5.8e7, 4e-7*math.pi
def f_of(nn): return rho/(math.pi*mu0*(DX/nn)**2)
def kelvin(f):
    d = math.sqrt(2*rho/(2*math.pi*f*mu0))
    kb = (1-1j)/d
    return (kb*rho/(2*math.pi*R))*jv(0, kb*R)/jv(1, kb*R)

Lf = L[np.ix_(free, free)]
rf = rvec[free]
idx = np.flatnonzero(free)
for nn in (2, 4, 6, 8):
    f = f_of(nn)
    w = 2*math.pi*f
    r2.set_frequency(f)
    Bc, Be = bases()
    Bcf, Bef = Bc[free], Be[free]
    Ifine = np.eye(free.sum())
    z_f = wire_Z(Lf, rf, Ifine, w)
    z_c = wire_Z(Lf, rf, Bcf, w)
    z_e = wire_Z(Lf, rf, Bef, w)
    z0_f = wire_Z(Lf, rf, Ifine, 2*math.pi*1e4)
    z0_c = wire_Z(Lf, rf, Bcf, 2*math.pi*1e4)
    z0_e = wire_Z(Lf, rf, Bef, 2*math.pi*1e4)
    rat = lambda z, z0: z.real/z0.real
    ka = (kelvin(f)/kelvin(1e4)).real
    print('dx/d %-3g fine %7.3f  coarse %7.3f (%+5.1f%%)  enrich %7.3f '
          '(%+5.1f%%)  [Kelvin %7.3f] modes %d'
          % (nn, rat(z_f, z0_f), rat(z_c, z0_c),
             100*(rat(z_c, z0_c)/rat(z_f, z0_f) - 1),
             rat(z_e, z0_e), 100*(rat(z_e, z0_e)/rat(z_f, z0_f) - 1),
             ka, Be.shape[1] - n), flush=True)
