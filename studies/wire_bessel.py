# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Does the 13-filament cross-section actually reproduce SKIN EFFECT?

wirekernel.py's gate validates the KERNELS -- geometry, quadrature,
mutuals, self terms, wire-to-voxel coupling. It says nothing about the
cross-section DESIGN, because it contains no frequency and no solve:
the 13-element decomposition was only ever checked for exact area and
reciprocity. This file supplies the missing test, and it is the one
that decides whether the discretisation is right before anything is
built on top of it.

THE TEST. Assemble the 13 filaments of a straight round wire into
Z = R + jw L (R from each element's own area, L from wirekernel),
drive them with a COMMON emf -- they are joined at both ends -- and
solve for the current split:

    Z i = V 1   =>   Z_total = V/I = 1/(1^T Z^-1 1)

the parallel combination. Then compare against the exact solution for a
round conductor, Z_int = (k/(2 pi a sigma)) I0(ka)/I1(ka) with
k = sqrt(j w mu sigma).

WHY Re(Z) IS THE RIGHT COMPARAND. The external partial inductance is
large, common to every filament and purely reactive, so it cancels out
of the resistance entirely: Re(Z_total)/R_dc is skin effect and nothing
else. It needs no reference length, no subtraction of an external term,
and no gauge choice -- which makes it a clean pass/fail on the
discretisation.

WHAT IT IS ALSO FOR: choosing core_frac. The core radius was picked as
0.5 with no justification when round_wire() was written. Sweeping it
here against the exact answer replaces a guess with a measurement.

Run: PYTHONPATH=src python3 studies/wire_bessel.py
Env: RADIUS (um, 200), SIGMA (3.5e7 Al), LEN (mm, 5), NSECT (12),
     CORE (core_frac; 0 sweeps it)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.special import iv

import wirekernel as wk


def bessel_zint(freq, a, sigma, mu=wk.MU0):
    """Exact internal impedance per unit length of a round conductor."""
    w = 2*np.pi*freq
    k = np.sqrt(1j*w*mu*sigma)
    return (k/(2*np.pi*a*sigma))*iv(0, k*a)/iv(1, k*a)


def bundle(a, length, nsect, core_frac, nr=3, nth=3, gmd_nr=10,
           gmd_nth=24, nring=1, grade=0.5):
    """The 13 filaments plus their (R-less) inductance matrix."""
    fils = wk.round_wire([0, 0, 0], [0, 0, 1], length, a, nsect=nsect,
                         core_frac=core_frac, nr=nr, nth=nth,
                         nring=nring, grade=grade)
    n = len(fils)
    # each element's self term needs its OWN gmd, computed on a finer
    # quadrature of the same shape (gmd converges slowly -- log
    # singularity -- and is a per-shape constant, so pay once)
    rc = core_frac*a
    g = []
    off, ar = wk.sector_points(0.0, rc, 0.0, 2*np.pi, gmd_nr, gmd_nth)
    g.append(wk.gmd(off, ar))
    t = np.array([grade**k for k in range(nring)], dtype=float)
    t = t/t.sum()
    edges = rc + (a - rc)*np.concatenate([[0.0], np.cumsum(t)])
    for m in range(nring):
        for k in range(nsect):
            off, ar = wk.sector_points(edges[m], edges[m+1],
                                       2*np.pi*k/nsect,
                                       2*np.pi*(k+1)/nsect,
                                       gmd_nr, 8)
            g.append(wk.gmd(off, ar))
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = (wk.mutual(fils[i], fils[i], self_gmd=g[i]) if i == j
                 else wk.mutual(fils[i], fils[j]))
            L[i, j] = L[j, i] = v
    return fils, L


def sweep(a, sigma, length, nsect, core_frac, freqs):
    fils, L = bundle(a, length, nsect, core_frac)
    areas = np.array([f.A for f in fils])
    R = length/(sigma*areas)
    rdc = length/(sigma*np.pi*a*a)
    one = np.ones(len(fils))
    out = []
    for f in freqs:
        Z = np.diag(R) + 1j*2*np.pi*f*L
        ztot = 1.0/complex(one @ np.linalg.solve(Z, one))
        exact = bessel_zint(f, a, sigma)*length
        out.append((f, ztot.real/rdc, exact.real/rdc))
    return np.array(out), rdc


def main():
    a = float(os.environ.get('RADIUS', '200'))*1e-6
    sigma = float(os.environ.get('SIGMA', '3.5e7'))
    length = float(os.environ.get('LEN', '5'))*1e-3
    nsect = int(os.environ.get('NSECT', '12'))
    core = float(os.environ.get('CORE', '0'))
    freqs = np.logspace(4, 8, 9)
    delta = 1.0/np.sqrt(np.pi*freqs*wk.MU0*sigma)
    print("round wire a = %.0f um, sigma = %.3g S/m, l = %.1f mm, "
          "%d sectors" % (1e6*a, sigma, 1e3*length, nsect), flush=True)

    fracs = [core] if core else [0.3, 0.5, 0.7, 0.85]
    print("\n%10s %7s " % ("freq", "a/delta")
          + " ".join("%9s" % ("cf=%.2f" % c) for c in fracs)
          + "     exact", flush=True)
    tables = {}
    for c in fracs:
        tab, rdc = sweep(a, sigma, length, nsect, c, freqs)
        tables[c] = tab
    for i, f in enumerate(freqs):
        row = " ".join("%9.4f" % tables[c][i, 1] for c in fracs)
        print("%10.3g %7.2f %s %9.4f"
              % (f, a/delta[i], row, tables[fracs[0]][i, 2]), flush=True)
    print("\nmax |error| in R_ac/R_dc vs the exact Bessel solution:",
          flush=True)
    for c in fracs:
        t = tables[c]
        err = np.abs(t[:, 1]/t[:, 2] - 1)
        print("   core_frac %.2f : %6.2f%%   (worst at a/delta = %.2f)"
              % (c, 100*err.max(), a/delta[int(np.argmax(err))]),
              flush=True)
    print("\nNOTE R_ac/R_dc is skin effect alone -- the external "
          "inductance is common to every filament and purely reactive, "
          "so it cancels from the real part.", flush=True)


if __name__ == '__main__':
    main()
