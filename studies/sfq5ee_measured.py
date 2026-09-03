# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""SuperPEEC vs MEASURED SFQ5ee silicon: M1aM0bM2 striplines.

THE REFERENCE THAT SETTLES ACCURACY ARGUMENTS: experiment. Tolpygo,
Golden, Weir & Bolkhovsky, "Inductance and mutual inductance of
superconductor integrated circuit features with sizes down to 120 nm.
Part I" (arXiv:2101.07457), measured the linear inductance of SFQ5ee
inductors with a SQUID-based differential method (quoted accuracy
~1.4%). For the symmetric stripline M1aM0bM2 -- signal on M1 between
M0 and M2 ground planes, all films 200 nm, both gaps 200 nm, the
planes edge-connected -- the paper gives wafer-mean MEASURED values
(Fig. 5 wafermaps, nine PCM sites, central die excluded):

    w = 0.35 um : 0.41  pH/um   (sigma 4.3% across the wafer)
    w = 1.00 um : 0.189 pH/um   (sigma 4.5%)

and process-target curves for w = 0.25..2 um (Fig. 6). lambda = 90 nm
at 4.2 K for all Nb layers (their standard value, from prior microwave
and inductance measurements; the wafer CENTRE runs ~8-13% high,
consistent with lambda ~ 96 nm there -- which is why the centre die is
excluded from the means we compare against).

THE MODEL. Doctrine TOML built directly (no GDS): three films with
film = "z", strip length-differenced (two lengths, far-end via wall
shorting strip to BOTH planes -- it also ties the planes together, as
the paper's edge connection does; the length difference cancels the
wall and the port). Film palette modes on, per-axis pitch: dz = 100 nm
(2 cells/film -- measured ~94% of true kinetic on the film bench
oracle), dx chosen so the strip is >= 7 cells wide.

This structure is a BEST CASE on purpose: axis-aligned, w >= 250 nm
(above the ~36 nm nonsuperconducting-sidewall regime the paper
documents for narrower lines), on the exact stack the converter
models. Vias and narrow lines are later chapters.

RESULTS (2026-09-02, dz = 100 nm, dx = 50 nm, film palette):

    w um   SuperPEEC   paper    ratio
    0.35     0.4242    0.410    1.035   measured, sigma 4.3%
    0.50     0.3363    0.33     1.02    target-curve readoff
    0.70     0.2654    0.26     1.02    target-curve readoff
    1.00     0.2020    0.189    1.069   measured, sigma 4.5%
    2.00     0.1125    0.105    1.071   target-curve readoff

TWO-PART INTERPRETATION, and the order matters:
  * SOLVER vs TRUTH: the 2-D cross-section oracle (the
    london_oracle2d method with a second ground plane) gives the
    exact answer for the MODELED geometry at w = 1 um:
    0.2017-0.2018 pH/um total (geo 0.099-0.101, kin 0.101-0.103).
    SuperPEEC: 0.2020 = geo 0.1007 + kin 0.1012. Agreement 0.15%.
    The solver is exact here; a 65-second solve.
  * MODEL vs SILICON: the +3.5..7% above the wafer means is the
    difference between NOMINAL process parameters (Table I, lambda =
    90 nm, design linewidth) and the wafers: the paper's own FIB
    cross-sections measured interlayer dielectrics 8-10% THIN and
    linewidths ~35 nm WIDE of design, and lambda runs 90..96 nm
    across the wafer -- each pushes measured L below nominal by a few
    percent, together bracketing the observed offset. (Their own
    wxLL-vs-measured comparisons show the same size of deviation,
    e.g. ~7% for M6aM0, attributed to dielectric thickness; their
    calibrated-InductEx effort needed "dozens of free fitting
    parameters".) Feeding actual wafer dimensions instead of nominal
    is the honest next refinement, exactly as the paper does.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'src')]

MU0 = 4e-7*np.pi
T = 200e-9          # film thickness, M0/M1/M2
G = 200e-9          # each gap
LAM = 9e-8
ZMARG = 400e-9
LENS = (4e-6, 8e-6)

# (w, measured pH/um or None, label)
CASES = [
    (0.35e-6, 0.410, 'measured, sigma 4.3%'),
    (0.50e-6, 0.33,  'target-curve readoff'),
    (0.70e-6, 0.26,  'target-curve readoff'),
    (1.00e-6, 0.189, 'measured, sigma 4.5%'),
    (2.00e-6, 0.105, 'target-curve readoff'),
]


def toml_text(w, ln, dx, dz, freq):
    wp = w + 12e-6                       # plane width (wide, as on PCM)
    nyw = int(round(w/dx))
    nyp = int(round(wp/dx))
    ny = nyp + 8
    j0p = 4                              # plane y-range
    j1p = j0p + nyp
    j0s = (ny - nyw)//2                  # strip centred
    j1s = j0s + nyw
    MX = max(4, int(round(0.5e-6/dx)))
    L = int(round(ln/dx))
    nx = 2*MX + L
    z0 = int(round(ZMARG/dz))
    tz = int(round(T/dz))
    gz = int(round(G/dz))
    zM0 = (z0, z0 + tz)
    zM1 = (zM0[1] + gz, zM0[1] + gz + tz)
    zM2 = (zM1[1] + gz, zM1[1] + gz + tz)
    nz = zM2[1] + z0
    blk = lambda n, x0, y0, zz, x1, y1: (      # noqa: E731
        '[[block]]\nname = "%s"\nfrom = [%d, %d, %d]\n'
        'to   = [%d, %d, %d]\nlambda_l = %g\nfilm = "z"\n'
        % (n, x0, y0, zz[0], x1, y1, zz[1], LAM))
    out = ["[grid]",
           "dims  = [%d, %d, %d]" % (nx, ny, nz),
           "pitch = [%g, %g, %g]" % (dx, dx, dz), "",
           blk("M0", MX, j0p, zM0, MX + L, j1p),
           blk("M2", MX, j0p, zM2, MX + L, j1p),
           blk("M1", MX, j0s, zM1, MX + L, j1s),
           # far-end wall: strip footprint through the whole stack --
           # shorts M1 to both planes and ties the planes together
           '[[block]]\nname = "wall"\nfrom = [%d, %d, %d]\n'
           'to   = [%d, %d, %d]\nlambda_l = %g\n'
           % (MX + L - 2, j0s, zM0[0], MX + L, j1s, zM2[1], LAM)]
    fmt = lambda F: "[" + ", ".join(            # noqa: E731
        '[%d, %d, %d, "-x"]' % t for t in F) + "]"
    pf = [(MX, j, k) for j in range(j0s, j1s) for k in range(*zM1)]
    nf = [(MX, j, k) for j in range(j0p, j1p)
          for zz in (zM0, zM2) for k in range(*zz)]
    out += ["[port]", 'name = "P"', "equipotential = true",
            "p_faces = " + fmt(pf), "n_faces = " + fmt(nf), "",
            "[solve]", "freq = [%g]" % freq,
            'skin = { mode = "on", basis = "conduction" }', ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dz', type=float, default=100e-9)
    ap.add_argument('--freq', type=float, default=1e10)
    ap.add_argument('--w', type=float, nargs='+', default=None,
                    help='subset of widths to run (m)')
    a = ap.parse_args(argv)
    import sppeec_input as si
    print("M1aM0bM2 stripline vs Tolpygo et al. arXiv:2101.07457")
    print("%8s | %12s %12s %8s | %s"
          % ("w um", "SuperPEEC", "paper", "ratio", "reference class"))
    for w, ref, label in CASES:
        if a.w and not any(abs(w - x) < 1e-12 for x in a.w):
            continue
        # dx = 100 nm measured +5% high against the dx = 50 nm rows
        # (in-plane discretisation of the geometric term); 50 nm for all
        dx = 50e-9
        t0 = time.time()
        Ls = []
        for ln in LENS:
            prob = si.loads(toml_text(w, ln, dx, a.dz, a.freq))
            m = prob.model()
            M = prob.tree(m)
            Z, _ = prob.sweeper(m, M).solve(a.freq)
            Ls.append(complex(Z).imag/(2*np.pi*a.freq))
        lpul = (Ls[1] - Ls[0])/(LENS[1] - LENS[0])*1e12/1e6   # pH/um
        print("%8.2f | %9.4f pH/um %9.3f %8.3f | %s   (%.0f s)"
              % (w*1e6, lpul, ref, lpul/ref, label, time.time() - t0),
              flush=True)


if __name__ == '__main__':
    main()
