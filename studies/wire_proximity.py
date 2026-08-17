# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""PROXIMITY: does the ANGULAR discretisation earn its keep?

Everything measured so far (wire_bessel, wire_converge) is an ISOLATED
wire, where the solution is axisymmetric and all sectors are equivalent
by symmetry. The angular sectors -- the entire reason the cross-section
is not just a stack of annuli -- have therefore never been exercised.
Proximity is what they exist for, and it governs current sharing
between parallel bond wires, which is one of the two design questions
behind the power-module example.

THE SETUP. Two parallel round wires, anti-parallel currents (a
go-return pair, the strongest proximity case). Every element of wire 1
shares one terminal voltage and every element of wire 2 shares another;
the totals are constrained to +I and -I:

    [u1' Z^-1 u1  u1' Z^-1 u2] [V1]   [ I]
    [u2' Z^-1 u1  u2' Z^-1 u2] [V2] = [-I]

and the loop impedance is (V1 - V2)/I. R_loop/R_dc is then skin AND
proximity together.

THE BUILT-IN NULL CASE. nsect=1 gives full annuli with no angular
freedom at all, so proximity is structurally unrepresentable. The gap
between nsect=1 and converged nsect IS the proximity effect, measured
rather than assumed -- and it says directly how much the sectors buy.

Run: PYTHONPATH=.:studies python3 studies/wire_proximity.py
Env: SEP (centre spacing / radius, default 3), FREQ (1e7),
     NRING (2), SECTS (comma list)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import wirekernel as wk

A = 200e-6
SIGMA = 3.5e7
LEN = 80e-3          # l/a = 400: end effects ~0.1%, per wire_converge


def loop_impedance(sep, freq, nsect, nring=2, nr=3, nth=8, grade=0.5,
                   core_frac=0.5, shape=True, gmd_nth=64, angle=0.0):
    """``angle`` (degrees) rotates the SECOND wire around the first.

    This matters because the sectors start at theta=0 on every ring, so
    a coarsely sectored ring has SEAMS at fixed angles and the accuracy
    depends on whether a seam faces the neighbouring wire or bisects the
    gap. A single-angle table flatters whichever arrangement happens to
    be favourably aligned; a real module has wires at every angle, so
    the WORST case over orientation is the honest figure.
    """
    d = 1.0/np.sqrt(np.pi*freq*wk.MU0*SIGMA)
    th = np.radians(angle)
    fils, tag = [], []
    for w, ctr in enumerate(((0.0, 0.0),
                             (sep*A*np.cos(th), sep*A*np.sin(th)))):
        f = wk.round_wire([ctr[0], ctr[1], 0], [0, 0, 1], LEN, A,
                          nsect=nsect,
                          core_frac=core_frac, nring=nring, grade=grade,
                          nr=nr, nth=nth, delta=(d if shape else None))
        fils += f
        tag += [w]*len(f)
    tag = np.array(tag)
    n = len(fils)
    # self gmd per distinct shape (both wires share shapes)
    rc = core_frac*A
    t = np.array([grade**k for k in range(nring)]); t /= t.sum()
    edges = rc + (A - rc)*np.concatenate([[0.0], np.cumsum(t)])
    ns = ([int(nsect)]*nring if np.isscalar(nsect)
          else [int(v) for v in nsect])
    gs = [wk.gmd(*wk.sector_points(0.0, rc, 0.0, 2*np.pi, 6, gmd_nth))]
    for m in range(nring):
        # FIXED per-sector gmd order. Scaling it as gmd_nth//nsect
        # (as this first did) makes the gmd accuracy vary along the very
        # axis being converged -- 64/32/16/8/8/8/8 for nsect
        # 1/2/4/8/12/16/24 -- and produced a steady -0.035 drift in
        # R_loop per step that looked like non-convergence of the model.
        gs += [wk.gmd(*wk.sector_points(edges[m], edges[m+1], 0.0,
                                        2*np.pi/ns[m], 6,
                                        gmd_nth))]*ns[m]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = (wk.mutual(fils[i], fils[i], self_gmd=gs[i % (len(gs))])
                 if i == j else wk.mutual(fils[i], fils[j]))
            L[i, j] = L[j, i] = v
    rho = 1.0/SIGMA
    R = np.array([f.resistance(rho) for f in fils])
    Z = np.diag(R) + 1j*2*np.pi*freq*L
    u1 = (tag == 0).astype(float)
    u2 = (tag == 1).astype(float)
    Zi = np.linalg.solve(Z, np.column_stack([u1, u2]))
    M = np.array([[u1 @ Zi[:, 0], u1 @ Zi[:, 1]],
                  [u2 @ Zi[:, 0], u2 @ Zi[:, 1]]])
    V = np.linalg.solve(M, np.array([1.0, -1.0]))
    return (V[0] - V[1])


def main():
    sep = float(os.environ.get('SEP', '3'))
    freq = float(os.environ.get('FREQ', '1e7'))
    nring = int(os.environ.get('NRING', '2'))
    sects = [int(v) for v in
             os.environ.get('SECTS', '1,2,4,8,12,16,24').split(',')]
    d = 1.0/np.sqrt(np.pi*freq*wk.MU0*SIGMA)
    rdc = 2*LEN/(SIGMA*np.pi*A*A)
    print("two parallel wires, anti-parallel currents: a = %.0f um, "
          "centre spacing %.1f a, %.3g Hz (a/delta = %.2f), %d rings"
          % (1e6*A, sep, freq, A/d, nring), flush=True)
    print("%7s %7s %14s %12s %s"
          % ("sects", "elems", "R_loop/R_dc", "L_loop nH", "vs converged"),
          flush=True)
    vals = {}
    for ns in sects:
        t0 = time.perf_counter()
        z = loop_impedance(sep, freq, ns, nring=nring)
        vals[ns] = z
        print("%7d %7d %14.5f %12.5f          (%.0f s)"
              % (ns, 2*(1 + nring*ns), z.real/rdc,
                 1e9*z.imag/(2*np.pi*freq), time.perf_counter() - t0),
              flush=True)
    ref = vals[sects[-1]]
    print("\nrelative to the finest (nsect = %d):" % sects[-1], flush=True)
    for ns in sects:
        print("   nsect %3d : R %+8.3f%%   L %+8.4f%%"
              % (ns, 100*(vals[ns].real/ref.real - 1),
                 100*(vals[ns].imag/ref.imag - 1)), flush=True)
    print("\nnsect=1 is the NULL CASE -- full annuli, no angular freedom, "
          "proximity structurally absent. Its gap to the converged value "
          "is the proximity effect itself.", flush=True)


if __name__ == '__main__':
    main()
