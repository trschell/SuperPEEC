# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Is a DIAGONAL trace handled? The straight-bar rotation ladder.

THE QUESTION. Diagonal PCB traces are common, and a voxel PEEC code
staircases them: the cross-section AREA converges with refinement,
the edge PERIMETER never does (a 45-degree staircase edge is sqrt(2)
times the true edge at every pitch), and the piecewise-constant
filament basis cannot turn current inside a cell, so a 45-degree trace
is a corner per cell. Inductance is rotation invariant, so the same
bar solved axis-aligned IS the reference at every pitch and
frequency: what a diagonal costs is the difference.

THE MEASUREMENT. One copper bar (W x T x LEN), solved (a) along x,
driven face to face, and (b) rotated 45 degrees in the xy-plane,
rasterised at the same cubic pitch (a cell is metal iff its centre
lies inside the rotated rectangle) and driven by the STAIRCASED END
CUT: every boundary face whose outside neighbour lies beyond the end
plane, x-faces and y-faces alike, at equal share -- the mixed-
orientation prescribed-current port (2026-09-04). Both at DC, in the
skin transition and deep skin, over three pitches (cells across the
width). Reported: R and L per case, the diagonal/aligned ratios, the
DC closed form, and the observed convergence order of each ratio.

Run: PYTHONPATH=src python3 studies/diagonal_bar.py [--nw 4 8 16]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), 'src')]

import voxmodel                    # noqa: E402
import port_impedance as pz        # noqa: E402

SIGMA = 5.8e7
W, T, LEN = 100e-6, 50e-6, 1.6e-3   # T = 2 cells at the coarsest pitch (a 1-cell slab has no z filaments)


def make_model(occ, d, pfaces, nfaces, freq, tag):
    m = voxmodel.VoxelModel(tag)
    m.dims = tuple(int(v) for v in occ.shape)
    m.d = np.full(3, d)
    m.sigma = np.where(occ, np.float32(SIGMA), np.float32(0.0))
    m.freq = np.asarray([freq], dtype=float)
    p = voxmodel.Port('p1')
    for f in pfaces:
        p._add('P', f)
    for f in nfaces:
        p._add('N', f)
    p._freeze()
    m.ports = [p]
    return m


def aligned(nw, freq):
    h = W/nw
    nt = max(1, int(round(T/h)))
    nl = int(round(LEN/h))
    occ = np.ones((nl, nw, nt), dtype=bool)
    pf = [(0, j, k, 0, -1) for j in range(nw) for k in range(nt)]
    nf = [(nl - 1, j, k, 0, +1) for j in range(nw) for k in range(nt)]
    return make_model(occ, h, pf, nf, freq, 'aligned_%d' % nw)


def diagonal(nw, freq):
    """The same bar along (1, 1)/sqrt(2), staircased at pitch h."""
    h = W/nw
    nt = max(1, int(round(T/h)))
    margin = 2*h
    # bar frame: u along the bar, v across; start point (x0, y0)
    c, s_ = 1/np.sqrt(2.0), 1/np.sqrt(2.0)
    ext = (LEN + W)*c + 2*margin              # bounding box side
    n = int(np.ceil(ext/h))
    x0 = margin + W*c/2                        # bar centre-line start
    y0 = margin + W*c/2
    xc = (np.arange(n) + 0.5)*h
    X, Y = np.meshgrid(xc, xc, indexing='ij')
    u = (X - x0)*c + (Y - y0)*s_
    v = -(X - x0)*s_ + (Y - y0)*c
    inside = (u >= 0) & (u <= LEN) & (np.abs(v) <= W/2)
    occ = np.repeat(inside[:, :, None], nt, axis=2)
    # END CUTS: boundary faces whose OUTSIDE neighbour lies beyond the
    # end plane (u < 0 at the P end, u > LEN at the N end); the side
    # edges' faces have outside neighbours with 0 <= u <= LEN
    pf, nf = [], []
    for i, j in zip(*np.nonzero(inside)):
        for ax, dlt in ((0, -1), (0, +1), (1, -1), (1, +1)):
            ni, nj = (i + dlt, j) if ax == 0 else (i, j + dlt)
            if 0 <= ni < n and 0 <= nj < n and inside[ni, nj]:
                continue
            un = (xc[ni] - x0)*c + (xc[nj] - y0)*s_ if (0 <= ni < n and 0 <= nj < n) else \
                ((xc[i] + dlt*h - x0)*c + (xc[j] - y0)*s_ if ax == 0 else
                 (xc[i] - x0)*c + (xc[j] + dlt*h - y0)*s_)
            for k in range(nt):
                if un < 0:
                    pf.append((int(i), int(j), k, ax, dlt))
                elif un > LEN:
                    nf.append((int(i), int(j), k, ax, dlt))
    return make_model(occ, h, pf, nf, freq, 'diag_%d' % nw)


def solve(m, freq):
    M = m.build_tree()
    m.prepare(M, freq)
    sol = pz.LpRSolver(M)
    t0 = time.perf_counter()
    Z, infos = pz.impedance_matrix(m, M, sol, freq)
    z = complex(Z[0, 0])
    return z.real, z.imag/(2*np.pi*freq), infos[0], time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--nw', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--freqs', type=float, nargs='+',
                    default=[1e3, 1e7, 1e8, 1e9])
    a = ap.parse_args(argv)
    r_dc = LEN/(SIGMA*W*T)
    print("bar %g x %g um, length %g mm, copper; DC R = %.6g ohm"
          % (W*1e6, T*1e6, LEN*1e3, r_dc))
    print("delta(f): " + ", ".join(
        "%.0e -> %.2f um" % (f, 1e6*np.sqrt(2/(2*np.pi*f*4e-7*np.pi*SIGMA)))
        for f in a.freqs))
    print()
    print("%4s %8s | %-32s | %-32s | %8s %8s | %s"
          % ('nw', 'f', 'ALIGNED R, L', 'DIAGONAL R, L', 'R_d/R_a',
             'L_d/L_a', 'cells (a/d), faces, mv, s'))
    hist = {}
    dc = {}
    for nw in a.nw:
        for f in a.freqs:
            ma = aligned(nw, f)
            md = diagonal(nw, f)
            ra, la, ia, ta = solve(ma, f)
            rd, ld, idd, td = solve(md, f)
            npf = len(md.ports[0].pos)
            hist[(nw, f)] = (rd/ra, ld/la)
            if f < 1e5:
                dc[nw] = (ra/r_dc, rd/r_dc)
            print("%4d %8.0e | R %.5e L %.5e | R %.5e L %.5e | %8.4f %8.4f "
                  "| %d/%d cells, %d P faces, mv %d/%d, %.0f+%.0f s"
                  % (nw, f, ra, la, rd, ld, rd/ra, ld/la,
                     int(ma.struc().sum()), int(md.struc().sum()), npf,
                     ia['matvecs'], idd['matvecs'], ta, td), flush=True)
    for nw, (qa, qd) in dc.items():
        print("DC check nw=%d: aligned R/R_exact = %.6f  diagonal R/R_exact "
              "= %.5f" % (nw, qa, qd))
    if len(a.nw) >= 3:
        print("\nobserved order of the diagonal/aligned RATIO error "
              "(|ratio - 1| vs h, last two refinements):")
        for f in a.freqs:
            e = [abs(hist[(nw, f)][0] - 1) for nw in a.nw]
            el = [abs(hist[(nw, f)][1] - 1) for nw in a.nw]
            def order(err):
                return (np.log(err[-3]/err[-2])/np.log(2), np.log(err[-2]/err[-1])/np.log(2)) \
                    if min(err) > 0 else (float('nan'), float('nan'))
            print("  f %.0e: R ratio err %s  order %.2f/%.2f | L ratio err %s  order %.2f/%.2f"
                  % (f, ' '.join('%.4f' % v for v in e), *order(e),
                     ' '.join('%.4f' % v for v in el), *order(el)))


if __name__ == '__main__':
    main()
