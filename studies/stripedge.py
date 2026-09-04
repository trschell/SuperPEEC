# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Is the RSFQ gap in-plane EDGE CROWDING?

The XNOR sits ~11% under InductEx and the two obvious causes are ruled
out by measurement: geometry (subpixel made it WORSE, -0.5%) and
in-plane mesh (2.2x the cells bought 0.73%). The surviving candidate is
the current singularity at a conductor EDGE -- for in-plane current in a
thin film the density diverges as d^-1/2 from a free edge, which is
2-D potential flow, NOT the Helmholtz screening the redistribution mode
palette is derived from. Projected onto that palette an idealised
d^-1/2 profile is 83% captured against 99.6% for the London profile.

That projection is a proxy, not a diagnosis. THIS is the diagnosis: a
wide superconducting strip over a ground plane, where edge crowding is
the dominant in-plane physics and nothing else competes. Hold the
PHYSICAL geometry and the z pitch fixed, vary only the cells across the
width, and fit the convergence order:

    smooth solution           L error ~ h^2
    d^-1/2 edge singularity   L error ~ h^1 or slower

If this strip converges at ~h^1 and the XNOR's ladder does too, edge
crowding is the shared cause and an edge mode family is worth building.
If the strip converges cleanly at h^2, the XNOR gap is elsewhere and a
mode family would be the wrong thing to build.

Run: PYTHONPATH=src python3 studies/stripedge.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), 'src')]

import equiterminal as eq        # noqa: E402
import vhr                       # noqa: E402

LAM = 9e-8                       # London depth
FREQ = 1e10
DZ = 6.75e-8                     # z pitch: 135 nm strip = 2 cells, FIXED
WIDTH = 1.62e-6                  # strip width, fixed physical
LENGTH = 4.05e-6                 # strip length, fixed physical
TH = 1.35e-7                     # strip thickness = 2 z cells
GAP = 2.7e-7                     # strip-to-ground gap = 4 z cells


def run(nw, modes=0):
    """nw cells across the strip width; everything physical is fixed."""
    dxy = WIDTH/nw
    # LENGTH must divide by dxy exactly too, or the strip changes
    # physical length between rungs and that alone moves L. LENGTH =
    # 2.5*WIDTH with EVEN nw keeps both commensurate. (The first cut
    # used nw = 9, where 4.05/0.18 = 22.5 rounds to 22 cells -- a 2.2%
    # length error that showed up as a non-monotone sequence.)
    ratio = LENGTH/dxy
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("nw=%d gives a non-integer length %.3f cells "
                         "-- pick nw so LENGTH/(WIDTH/nw) is exact"
                         % (nw, ratio))
    nx = int(round(ratio))
    ngnd, ngap = 2, int(round(GAP/DZ))
    nt = int(round(TH/DZ))
    pad = max(2, nw//2)                       # ground extends past the strip
    ny = nw + 2*pad
    nz = ngnd + ngap + nt
    struc = np.zeros((nx, ny, nz), dtype=np.int8)
    struc[:, :, :ngnd] = 1                    # ground plane, full width
    z0 = ngnd + ngap
    struc[:, pad:pad + nw, z0:z0 + nt] = 1    # the strip
    lm = np.zeros((nx, ny, nz))
    lm[struc > 0] = LAM
    ports = []
    for j in range(pad, pad + nw):
        for k in range(z0, z0 + nt):
            ports.append(('p1', 'P', 0, j, k, '-x'))
            ports.append(('p1', 'N', nx - 1, j, k, '+x'))
    p = os.path.join(HERE, 'stripedge.vhr')
    vhr.write_vhr(p, struc, dxy, 0.0, (FREQ,), ports, lambdaL=lm)
    m = vhr.read_vhr(p)
    # ANISOTROPIC: the z pitch is pinned so only the in-plane
    # resolution varies -- the whole point of the experiment
    m.d = np.array([dxy, dxy, DZ])
    leaf, lv = m.partition()
    M = m.build_tree(leaf, lv)
    m.prepare(M, FREQ)
    kw = dict(enrich=dict(k=modes, f_ref=FREQ)) if modes else {}
    Z, _, _ = eq.EquiTerminalSolver(m, M, 0, **kw).solve(FREQ)
    os.remove(p)
    return abs(Z.imag)/(2*np.pi*FREQ), nx*ny*nz


def main():
    print("strip %.2f um wide x %.0f nm thick over a ground plane, "
          "lambda %.0f nm" % (WIDTH*1e6, TH*1e9, LAM*1e9))
    print("z pitch PINNED at %.1f nm; only the in-plane pitch varies\n"
          % (DZ*1e9))
    print("%6s %9s %10s %13s %10s"
          % ('cells', 'dxy(nm)', 'lattice', 'L (pH)', 'delta'))
    rows = []
    for nw in (4, 8, 12, 16):
        L, n = run(nw)
        d = '' if not rows else '%+.4f' % (1e12*(L - rows[-1][1]))
        print("%6d %9.1f %10d %13.5f %10s"
              % (nw, 1e9*WIDTH/nw, n, 1e12*L, d), flush=True)
        rows.append((nw, L))
    # Richardson on the last three: E(h) = E0 + C h^p
    (n1, L1), (n2, L2), (n3, L3) = rows[-3:]
    h1, h2, h3 = 1.0/n1, 1.0/n2, 1.0/n3
    r = (L2 - L1)/(L3 - L2)
    p = np.log(r)/np.log((h1 - h2)/(h2 - h3)) if r > 0 else float('nan')
    print("\nconvergence order p ~ %.2f   (smooth ~2, edge singularity ~1)"
          % p)


if __name__ == '__main__':
    main()
