# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""London kinetic sheet inductance vs an EXACT reference.

THE INSTRUMENT THAT SETTLED THE RSFQ GAP (2026-08-31). A campaign of
XNOR-vs-InductEx comparisons produced three misleading conclusions in
one session because the reference -- a back-annotated netlist -- carries
no mutual terms and so is biased high by an unknown amount. This bench
removes the external reference entirely: a superconducting microstrip's
kinetic sheet inductance has a closed form,

    L_kin = 2 * mu0 * lambda * coth(t/lambda)   per square,

(one coth per film; field on one side of each), which for the SFQ5ee-like
t = 200 nm, lambda = 90 nm pair is 0.22906 pH/sq -- about 48% of the
total, so mis-recovering it is the dominant error term.

ISOLATION IS A DOUBLE DIFFERENCE. A lambda difference (lam vs a tiny
lambda) cancels every geometric term including fringing; a length
difference on top cancels the far-end shorting via and the port. What
survives is kinetic inductance per square, and the model's own
lambda->0 geometric difference normalises it (the parallel-plate value
mu0*h is NOT used as the denominator -- the mesh's geometric term
converges downward with refinement, and dividing by the ideal value
would fold that unrelated convergence into the kinetic answer):

    recovered = dL_kin(measured) / [L_kin_exact/(mu0*h) * dL_geo(measured)]

MEASURED LAW (this bench, in-plane 100 nm fixed, z varying):
    cells/film   2     3      4     5     8     10
    recovered   54%  ~62%   71%   75%   81%   84%     (modes off)
    error ~ dz^0.65 -- 5% needs dz ~ 3 nm, unreachable by mesh alone.
The XNOR runs pz = 67.5 nm = ~3 cells/film. Missing kinetic inductance
is missing L: the sign and rough size of the "InductEx gap".

TWO CORRECTIONS, both measured here:
  * sub-cell London modes (skin = { mode = "on", basis = "conduction" }):
    +26 points at 2 cells/film, +12 at 4 (cubic cells; --modes on).
  * subpixel alignment (--offset dz/2): the same physical film held by
    partial rim cells recovers ~85-100% at 4-cells-of-metal thickness --
    not by intra-cell physics (each cell stays uniform-J) but because
    fractional boundaries put thin sample cells where cosh(z/lambda)
    bends hardest.

NORMALISATION CAVEAT (2026-09-02). ``recovered`` is measured against
the per-square sheet formula, which is exact for infinite films (and
in studies/london1d.py) but OVERSTATES the kinetic term of the finite
strip: the true value from the 2-D cross-section oracle
(studies/london_oracle2d.py) is 68.7% / 78.9% of per-square at
W = 2 / 4 um, H = 200 nm. So the recovered-% printed here saturates
near ~79-90%, NOT 100%, when the solver is exact; use the oracle for
absolute accuracy claims and this bench for RELATIVE comparisons
(mesh, palette, rc) where the fixed normalisation cancels. Measured
against the oracle: the film palette is ~94% of truth at nt = 2,
k = 7 and ~98% at nt = 4, k = 12.

USE AS A GATE. Any change to the mode engine or the subpixel path that
touches cell geometry (per-axis pitch, anisotropic cross-sections)
must reproduce the cubic recovered-fraction curve: run the same
physical dz once with cubic cells and once with --dx coarser than dz,
and the two recovered fractions must agree to a couple of points.

  python3 studies/london_film.py --dz 100e-9 50e-9 [--dx 100e-9]
        [--modes] [--offset 25e-9] [--lam 9e-8] [--freq 1e10]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'src')]

MU0 = 4e-7*np.pi

# The physical object, SFQ5ee-like (M0/M1: 200 nm films, 200 nm gap).
T = 200e-9        # film thickness, both films
H = 200e-9        # dielectric gap
W = 4e-6          # strip width
WG = 8e-6         # ground-plane width
NY = 10e-6        # lattice extent across the strip
ZMARG = 400e-9    # empty z margin below and above
LENS = (4e-6, 8e-6)   # the length pair whose difference cancels the via


def toml_text(dx, dz, ln, lam, offset, modes):
    """One microstrip as doctrine TOML. offset > 0 shifts both films by
    that much in z and emits physical bounds + subpixel, with whole-cell
    pads at the port end (ports need deliberate face lists; the length
    difference cancels the pads)."""
    L = int(round(ln/dx))
    MX = max(4, int(round(0.5e-6/dx)))
    nx = 2*MX + L
    ny = int(round(NY/dx))
    jg0, jg1 = 10, 10 + int(round(WG/dx))
    js0 = (ny - int(round(W/dx)))//2
    js1 = js0 + int(round(W/dx))
    zp0 = ZMARG + offset          # ground film, physical z
    zp1 = zp0 + T
    zg0 = zp1 + H                 # strip film
    zg1 = zg0 + T
    nz = int(np.ceil((zg1 + ZMARG)/dz))
    w = ["[grid]",
         "dims  = [%d, %d, %d]" % (nx, ny, nz),
         "pitch = [%g, %g, %g]" % (dx, dx, dz)]
    w.append("")
    fl = lambda v: int(np.floor(v/dz + 1e-6))       # noqa: E731
    ce = lambda v: int(np.ceil(v/dz - 1e-6))        # noqa: E731
    blocks = []
    if offset:
        # whole-cell pads under the port faces; physical bounds for the
        # films themselves so the rims are genuinely partial
        xp = (MX + 2)*dx
        blocks += [("padG", MX*dx, jg0*dx, fl(zp0)*dz, xp, jg1*dx, ce(zp1)*dz),
                   ("padS", MX*dx, js0*dx, fl(zg0)*dz, xp, js1*dx, ce(zg1)*dz)]
    blocks += [("M0", MX*dx, jg0*dx, zp0, (MX + L)*dx, jg1*dx, zp1),
               ("M1", MX*dx, js0*dx, zg0, (MX + L)*dx, js1*dx, zg1),
               ("via", (MX + L - 2)*dx, js0*dx, zp1, (MX + L)*dx, js1*dx, zg0)]
    filmline = ('film = "z"'
                if getattr(ARGS, 'film', False) else None)
    for n, x0, y0, z0, x1, y1, z1 in blocks:
        fln = [filmline] if (filmline and n in ('M0', 'M1')) else []
        if offset:
            w += ["[[block]]", 'name = "%s"' % n,
                  "from_m = [%.12g, %.12g, %.12g]" % (x0, y0, z0),
                  "to_m   = [%.12g, %.12g, %.12g]" % (x1, y1, z1),
                  "lambda_l = %g" % lam] + fln + [""]
        else:
            c = lambda v, p: int(round(v/p))        # noqa: E731
            if abs(round(z0/dz) - z0/dz) > 1e-9 or \
               abs(round(z1/dz) - z1/dz) > 1e-9:
                raise SystemExit("aligned mode needs dz dividing the "
                                 "stack; %g does not (use --offset)" % dz)
            w += ["[[block]]", 'name = "%s"' % n,
                  "from = [%d, %d, %d]" % (c(x0, dx), c(y0, dx), c(z0, dz)),
                  "to   = [%d, %d, %d]" % (c(x1, dx), c(y1, dx), c(z1, dz)),
                  "lambda_l = %g" % lam] + fln + [""]
    kg = range(fl(zp0), ce(zp1))
    ks = range(fl(zg0), ce(zg1))
    fmt = lambda F: "[" + ", ".join(                # noqa: E731
        '[%d, %d, %d, "-x"]' % t for t in F) + "]"
    pf = [(MX, j, k) for j in range(js0, js1) for k in ks]
    nf = [(MX, j, k) for j in range(jg0, jg1) for k in kg]
    w += ["[port]", 'name = "P"', "equipotential = true",
          "p_faces = " + fmt(pf), "n_faces = " + fmt(nf), "",
          "[solve]", "freq = [%g]" % ARGS.freq]
    if modes:
        w.append('enrich = { families = ["section"] }')
    return "\n".join(w) + "\n"


def solve_L(dx, dz, ln, lam, offset, modes, freq):
    import sppeec_input as si
    prob = si.loads(toml_text(dx, dz, ln, lam, offset, modes))
    m = prob.model()
    M = prob.tree(m)
    sw = prob.sweeper(m, M)
    Z, _ = sw.solve(freq)
    return complex(Z).imag/(2*np.pi*freq)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dx', type=float, default=100e-9,
                    help='in-plane pitch, HELD FIXED (m)')
    ap.add_argument('--dz', type=float, nargs='+',
                    default=[100e-9, 50e-9, 25e-9])
    ap.add_argument('--lam', type=float, default=9e-8)
    ap.add_argument('--lam-tiny', type=float, default=1e-9)
    ap.add_argument('--offset', type=float, default=0.0,
                    help='z offset of the whole stack (m); nonzero '
                         'engages subpixel with partial rim cells')
    ap.add_argument('--modes', action='store_true',
                    help='sub-cell London modes on '
                         '(skin mode="on", basis="conduction")')
    ap.add_argument('--film', action='store_true',
                    help='declare M0/M1 as thin films (film = "z"): the '
                         'engine spends its sub-cell budget on the '
                         'normal -- 1-D kz split, z-face shapes, wide '
                         'rc defaults')
    ap.add_argument('--freq', type=float, default=1e10)
    global ARGS
    ARGS = ap.parse_args(argv)

    lam, tiny = ARGS.lam, ARGS.lam_tiny
    coth = lambda t, l: l/np.tanh(t/l)              # noqa: E731
    kin_exact = 2*MU0*(coth(T, lam) - coth(T, tiny))
    print("exact kinetic sheet inductance: %.5f pH/sq "
          "(2 x mu0*lambda*coth(t/lambda), lam %g vs %g)"
          % (kin_exact*1e12, lam, tiny))
    print("modes %s, offset %g nm, in-plane dx %g nm\n"
          % ("ON" if ARGS.modes else "off", ARGS.offset*1e9, ARGS.dx*1e9))
    print("%8s %8s | %10s | %9s %8s %10s"
          % ("dz nm", "cells/t", "geometric", "kinetic", "target",
             "recovered"))
    for dz in ARGS.dz:
        q = {}
        for ln in LENS:
            for lm in (lam, tiny):
                q[(ln, lm)] = solve_L(ARGS.dx, dz, ln, lm, ARGS.offset,
                                      ARGS.modes, ARGS.freq)
        geo = q[(LENS[1], tiny)] - q[(LENS[0], tiny)]
        kin = ((q[(LENS[1], lam)] - q[(LENS[1], tiny)])
               - (q[(LENS[0], lam)] - q[(LENS[0], tiny)]))
        tgt = kin_exact/(MU0*H)*geo
        print("%8.4g %8.3g | %9.4f pH | %8.4f %8.4f %9.1f%%"
              % (dz*1e9, T/dz, geo*1e12, kin*1e12, tgt*1e12,
                 100*kin/tgt))


if __name__ == '__main__':
    main()
