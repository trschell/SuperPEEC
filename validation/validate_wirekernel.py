# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for wirekernel.py -- the arbitrary-orientation filament kernels.

THE CHECK THAT MATTERS IS THE FIRST ONE: in the degenerate axis-aligned
case the new kernel must reproduce ``terminal.box_mutual_matrix``, the
exact rectangular-bar formula SuperPEEC has used all along. New code
reproducing old code where the two overlap is the cheapest sharp test
available, and it catches sign, factor and geometry errors that a
self-consistency check never would.

Run: PYTHONPATH=src python3 validate_wirekernel.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np

import wirekernel as wk

FAIL = []


def check(name, ok, detail=""):
    print("  %-4s %-46s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def part_a():
    """GMD against the disc's closed form a*exp(-1/4)."""
    print("\nPART A -- geometric mean distance")
    a = 1.0
    for n in (8, 12, 16):
        off, ar = wk.sector_points(0.0, a, 0.0, 2*np.pi, n, 4*n)
        g = wk.gmd(off, ar)
        want = a*np.exp(-0.25)
        check("disc GMD, %dx%d quadrature" % (n, 4*n),
              abs(g/want - 1) < 0.02,
              "%.6f vs %.6f (%.2f%%)" % (g, want, 100*(g/want - 1)))


def part_b():
    """Parallel closed form against brute-force numerical integration."""
    print("\nPART B -- parallel axial kernel vs brute force")
    s1a, s1b, s2a, s2b, d = 0.0, 1.0, 0.3, 2.1, 0.25
    n = 4000
    z1 = np.linspace(s1a, s1b, n)
    z2 = np.linspace(s2a, s2b, n)
    U = z1[:, None] - z2[None, :]
    brute = np.trapezoid(np.trapezoid(1.0/np.sqrt(U*U + d*d), z2),
                         z1)
    exact = wk.mutual_parallel(s1a, s1b, s2a, s2b, d)
    check("closed form == brute force",
          abs(exact/brute - 1) < 1e-6,
          "%.8f vs %.8f" % (exact, brute))


def part_c():
    """Skew kernel against brute force, and its parallel limit."""
    print("\nPART C -- skew line kernel")
    a1 = np.array([0.0, 0.0, 0.0]); u1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.2, 0.7, 0.4])
    u2 = np.array([0.3, 0.8, 0.5]); u2 = u2/np.linalg.norm(u2)
    l1, l2 = 1.0, 1.3
    val = wk.mutual_skew_lines(a1, u1, l1, a2, u2, l2, ng=16)
    n = 1200
    s = np.linspace(0, l1, n)[:, None]
    t = np.linspace(0, l2, n)[None, :]
    P = a1[None, None, :] + s[..., None]*u1
    Q = a2[None, None, :] + t[..., None]*u2
    R = np.linalg.norm(P - Q, axis=-1)
    brute = np.trapezoid(np.trapezoid(1.0/R, np.linspace(0, l2, n)),
                         np.linspace(0, l1, n))
    check("skew == brute force", abs(val/brute - 1) < 1e-5,
          "%.8f vs %.8f" % (val, brute))
    # parallel limit: skew path with nearly-parallel axes must approach
    # the exact parallel closed form
    u2p = np.array([1.0, 0.0, 0.0])
    a2p = np.array([0.0, 0.3, 0.0])
    sk = wk.mutual_skew_lines(a1, u1, 1.0, a2p, u2p, 1.0, ng=64)
    ex = wk.mutual_parallel(0.0, 1.0, 0.0, 1.0, 0.3)
    check("skew path -> parallel closed form",
          abs(sk/ex - 1) < 1e-4, "%.8f vs %.8f" % (sk, ex))


def part_d():
    """THE GATE: axis-aligned bars vs terminal.box_mutual_matrix."""
    print("\nPART D -- degenerate limit vs the exact bar formula")
    import terminal as tm
    d = 1e-4
    # two x-directed bars, one cell long, offset in y and z
    cases = [((0, 0, 0), (0, 2, 0)), ((0, 0, 0), (0, 1, 3)),
             ((0, 0, 0), (2, 3, 1))]
    for c1, c2 in cases:
        lo = np.array([[c1[0]*d, c1[1]*d, c1[2]*d],
                       [c2[0]*d, c2[1]*d, c2[2]*d]])
        hi = lo + d
        M = tm.box_mutual_matrix(lo, hi, 0)
        want = float(M[0, 1])
        off, ar = wk.rect_points(d, d, 6, 6)
        f1 = wk.Filament([lo[0, 0], lo[0, 1] + d/2, lo[0, 2] + d/2],
                         [1, 0, 0], d, off, ar, v=[0, 1, 0])
        f2 = wk.Filament([lo[1, 0], lo[1, 1] + d/2, lo[1, 2] + d/2],
                         [1, 0, 0], d, off, ar, v=[0, 1, 0])
        got = wk.mutual(f1, f2)
        check("bar offset %s vs box_mutual_matrix" % (str(c2),),
              abs(got/want - 1) < 5e-3,
              "%.6e vs %.6e (%.3f%%)" % (got, want, 100*(got/want - 1)))


def part_e():
    """Reciprocity, and the round-wire assembly."""
    print("\nPART E -- reciprocity and the 13-filament round wire")
    a = 1.5e-4
    fs = wk.round_wire([0, 0, 0], [0, 0, 1], 1e-3, a, nsect=12)
    check("round wire has 13 filaments", len(fs) == 13,
          "%d" % len(fs))
    atot = sum(f.A for f in fs)
    check("sector areas sum to pi a^2 (exact area, curved edges)",
          abs(atot/(np.pi*a*a) - 1) < 1e-12,
          "%.10e vs %.10e" % (atot, np.pi*a*a))
    m12 = wk.mutual(fs[1], fs[2])
    m21 = wk.mutual(fs[2], fs[1])
    check("reciprocity M12 == M21", abs(m12/m21 - 1) < 1e-12,
          "%.6e vs %.6e" % (m12, m21))
    # a skewed pair must also be reciprocal
    g1 = wk.Filament([0, 0, 0], [1, 0, 0], 1e-3, *wk.rect_points(a, a))
    g2 = wk.Filament([2e-4, 3e-4, 1e-4], [0.3, 0.9, 0.3], 1.2e-3,
                     *wk.rect_points(a, a))
    check("reciprocity, skewed pair",
          abs(wk.mutual(g1, g2)/wk.mutual(g2, g1) - 1) < 1e-9,
          "%.6e" % wk.mutual(g1, g2))


def part_f():
    """Self-inductance sanity: a round wire against the classic
    L = (mu l/2pi)(ln(2l/a) - 3/4) for l >> a."""
    print("\nPART F -- self inductance of a long round wire")
    a, l = 1e-4, 1e-1
    off, ar = wk.sector_points(0.0, a, 0.0, 2*np.pi, 8, 32)
    f = wk.Filament([0, 0, 0], [0, 0, 1], l, off, ar)
    g = wk.gmd(off, ar)
    L = wk.self_inductance(f, g)
    want = (wk.MU0*l/(2*np.pi))*(np.log(2*l/a) - 0.75)
    check("L_self vs (mu l/2pi)(ln(2l/a) - 3/4)",
          abs(L/want - 1) < 0.02,
          "%.6e vs %.6e (%.2f%%)" % (L, want, 100*(L/want - 1)))


def part_g():
    """WIRE-TO-VOXEL: a skewed wire filament against a lattice bar,
    checked against a brute-force 6-D volume integral that shares no
    code with the kernel under test."""
    print("\nPART G -- wire filament vs voxel bar")
    d = 1e-4
    bar = wk.voxel_bar((3, 2, 1), d, 0, nq=4)
    u = np.array([0.4, 0.8, 0.45]); u /= np.linalg.norm(u)
    off, ar = wk.sector_points(0.0, 0.3*d, 0.0, 2*np.pi, 3, 8)
    fil = wk.Filament([1.5*d, 0.4*d, 1.2*d], u, 2.0*d, off, ar)
    got = wk.mutual(fil, bar, ng=24)

    # independent reference: midpoint 6-D sum over both volumes
    def vol_points(f, na, nl):
        s = (np.arange(nl) + 0.5)/nl*f.length
        p = f.points()
        P = p[:, None, :] + s[None, :, None]*f.u[None, None, :]
        # volume weight = dA * dl; forgetting the axial factor
        # inflates the reference by 1/(l1*l2) and looks exactly
        # like a broken kernel
        w = np.repeat(f.area*(f.length/nl), nl).reshape(
            len(f.area), nl)
        return P.reshape(-1, 3), w.ravel()
    P1, w1 = vol_points(fil, None, 40)
    P2, w2 = vol_points(bar, None, 40)
    R = np.linalg.norm(P1[:, None, :] - P2[None, :, :], axis=-1)
    brute = (wk.MU0/(4*np.pi))*float(u @ bar.u)*float(
        (w1[:, None]*w2[None, :]/R).sum())/(fil.A*bar.A)
    check("skew wire vs voxel bar == 6-D brute force",
          abs(got/brute - 1) < 5e-3,
          "%.6e vs %.6e (%.3f%%)" % (got, brute, 100*(got/brute - 1)))
    check("wire/voxel reciprocity",
          abs(wk.mutual(fil, bar, ng=24)/wk.mutual(bar, fil, ng=24) - 1)
          < 1e-6, "")

    import time
    fils = wk.round_wire([0, 0, 0], u, 2*d, 0.3*d, nsect=12)
    t0 = time.perf_counter()
    npair = 0
    for f in fils:
        for k in range(20):
            wk.mutual(f, wk.voxel_bar((k, 2, 1), d, 0), ng=16)
            npair += 1
    dt = (time.perf_counter() - t0)/npair
    check("cost per filament-voxel pair < 1 ms", dt < 1e-3,
          "%.0f us -> ~%.0f s for 1e6 near pairs" % (1e6*dt, 1e6*dt))


def part_h():
    """BATCHED kernels (the near-field workhorses) against the scalar
    ones they vectorise. Same mathematics, different array plumbing, so
    the agreement bar is machine precision, not model accuracy."""
    print("\nPART H -- batched near-field kernels == scalar")
    rng = np.random.default_rng(11)
    d = 1e-4
    u = np.array([0.5, 0.7, 0.51]); u /= np.linalg.norm(u)
    fs = wk.round_wire([3.2*d, 4.1*d, 9.3*d], u, 2*d, 0.4*d,
                       nring=3, nsect=(4, 8, 12), delta=0.3*d)
    cells = rng.integers(0, 12, size=(30, 3))
    worst = 0.0
    for axis in range(3):
        B = wk.mutual_voxels(fs, cells, d, axis)
        for e in rng.choice(len(fs), 4, replace=False):
            for c in rng.choice(len(cells), 4, replace=False):
                ref = wk.mutual(fs[e], wk.voxel_bar(cells[c], d, axis))
                worst = max(worst, abs(B[e, c] - ref)/abs(ref))
    check("mutual_voxels (skew) == scalar mutual", worst < 1e-12,
          "max rel %.2e" % worst)
    fsy = wk.round_wire([2.5*d, 3.5*d, 2.5*d], [0, 1, 0], 2*d, 0.4*d,
                        nring=2, nsect=(6, 12))
    B = wk.mutual_voxels(fsy, cells, d, 1)
    worst = max(abs(B[e, c] - wk.mutual(fsy[e],
                                        wk.voxel_bar(cells[c], d, 1)))
                / abs(wk.mutual(fsy[e], wk.voxel_bar(cells[c], d, 1)))
                for e in range(0, len(fsy), 5) for c in range(0, 30, 7))
    check("mutual_voxels (parallel) == scalar mutual", worst < 1e-12,
          "max rel %.2e" % worst)
    check("mutual_voxels perpendicular orientation is exactly zero",
          not wk.mutual_voxels(fsy, cells, d, 0).any(), "")
    fsB = wk.round_wire([6*d, 2*d, 4*d], [0.1, -0.9, 0.4], 1.5*d, 0.3*d,
                        nring=3, nsect=(4, 8, 12))
    Bb = wk.mutual_block(fs, fsB)
    worst = max(abs(Bb[i, j] - wk.mutual(fs[i], fsB[j]))
                / abs(wk.mutual(fs[i], fsB[j]))
                for i in range(0, len(fs), 6) for j in range(0, len(fsB), 6))
    check("mutual_block (skew) == scalar mutual", worst < 1e-12,
          "max rel %.2e" % worst)
    fsC = wk.round_wire([6*d, 2*d, 4*d], u, 1.5*d, 0.3*d,
                        nring=2, nsect=(6, 12))
    Bc = wk.mutual_block(fs, fsC)
    worst = max(abs(Bc[i, j] - wk.mutual(fs[i], fsC[j]))
                / abs(wk.mutual(fs[i], fsC[j]))
                for i in range(0, len(fs), 6) for j in range(0, len(fsC), 5))
    check("mutual_block (parallel) == scalar mutual", worst < 1e-12,
          "max rel %.2e" % worst)
    S = wk.segment_self_block(fsy)
    sym = np.abs(S - S.T).max()/np.abs(S).max()
    check("segment_self_block symmetric", sym < 1e-12, "%.2e" % sym)
    g0 = wk.gmd(fsy[0].off, fsy[0].area)
    ref = wk.mutual(fsy[0], fsy[0], self_gmd=g0)
    check("self block diagonal uses the GMD self term",
          abs(S[0, 0]/ref - 1) < 1e-12, "%.6e" % S[0, 0])


if __name__ == '__main__':
    part_a(); part_b(); part_c(); part_d(); part_e(); part_f(); part_g()
    part_h()
    print("\n%d checks failed" % len(FAIL))
    raise SystemExit(1 if FAIL else 0)
