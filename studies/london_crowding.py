# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Does the voxel model develop the LONDON CURRENT PROFILE, and at what
mesh?

THE QUESTION. A superconducting cell carries a bulk series impedance
z = j w mu lambda^2 (voxmodel.zdensity). The London profile is not
imposed by that term -- it EMERGES, the way skin effect emerges in a
normal-metal voxel model, because interior filaments see more mutual
inductance than surface ones and the network redistributes current.
validate_superconductor PART D shows the profile is there. PART A
measures the kinetic term on a 2x2 cross-section and finds it EXACTLY
mu*lambda^2*l/A -- the bulk value -- which is not a contradiction:
2x2 is the one cross-section where symmetry PINS the split, so no
crowding is possible and bulk is correct by construction.

So neither existing part answers the question that matters for a real
film: with a cross-section that CAN redistribute, how much kinetic
inductance does the model actually produce, and how many cells across
the conductor does it take to get there?

THE MEASUREMENT. Hold the physical cross-section and length FIXED and
vary only the number of cells across it. Kinetic inductance is
isolated by a lambda difference, which cancels the geometric term
(including any fringing) exactly as PART A's does:

    L_kin(nt) = [Im Z(lambda) - Im Z(lambda_tiny)] / w

and compared against the bulk value mu*lambda^2*l/A, which is
independent of nt. The ratio is the whole result:

    ratio -> 1.0 flat     the model cannot crowd; kinetic is bulk
    ratio > 1, rising     the profile develops, and the curve says
                          how many cells it needs

REFERENCE. A square bar has no closed form, so the equal-area cylinder
is the yardstick: for radius R and London depth lambda the kinetic
inductance per unit length is (mu/2pi)(lambda/R) I0(x)/I1(x) with
x = R/lambda, against a bulk (mu/pi)(lambda^2/R^2)/... -- their ratio
is (x/2) I0(x)/I1(x), which is what `--ratio-ref` prints. It is an
analogy, not an identity: PART D records that the 1-D slab model
over-predicts the contrast and the cylinder analog is the one that
matches, so the cylinder number is quoted as the expected SCALE.

  python3 studies/london_crowding.py [--side 3.6e-7] [--lam 9e-8]
        [--nt 2 4 6 8 12] [--freq 2.5e9]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'src')]

import equiterminal as eq            # noqa: E402
import vhr                           # noqa: E402

MU0 = 4e-7*np.pi


def bar(tag, lam, freq, nx, nt, dx):
    """nx x nt x nt lossless London bar, driven end to end."""
    struc = np.ones((nx, nt, nt), dtype=np.int8)
    ports = []
    for j in range(nt):
        for k in range(nt):
            ports.append(('p1', 'P', 0, j, k, '-x'))
            ports.append(('p1', 'N', nx - 1, j, k, '+x'))
    p = os.path.join(HERE, 'lc_%s.vhr' % tag)
    vhr.write_vhr(p, struc, dx, 0.0, (freq,), ports, lambdaL=lam)
    m = vhr.read_vhr(p)
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, freq)
    try:
        os.remove(p)
    except OSError:
        pass
    return m, M


def imz(lam, freq, nx, nt, dx, modes=0):
    m, M = bar('%g_%d_%d' % (lam, nt, modes), lam, freq, nx, nt, dx)
    kw = {}
    if modes:
        kw = dict(subdivide=int(modes), mode_basis='conduction',
                  skin_freq=freq)
    S = eq.EquiTerminalSolver(m, M, 0, **kw)
    Z, _, _ = S.solve(freq)
    return Z.imag


def cylinder_ratio(x):
    """(kinetic / bulk) for a London cylinder, x = R/lambda."""
    from scipy.special import iv
    return 0.5*x*iv(0, x)/iv(1, x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', type=float, default=3.6e-7,
                    help='physical cross-section side, m (held FIXED)')
    ap.add_argument('--length', type=float, default=3.6e-6)
    ap.add_argument('--lam', type=float, default=9e-8)
    ap.add_argument('--lam-tiny', type=float, default=1e-9,
                    help='kinetic is lambda^2, so this is ~0 kinetic')
    ap.add_argument('--nt', type=int, nargs='+',
                    default=[2, 4, 6, 8, 12])
    ap.add_argument('--freq', type=float, default=2.5e9)
    ap.add_argument('--modes', type=int, default=0,
                    help='sub-bar quadrature k for the conduction '
                         'basis (0 = engine off, the plain mesh)')
    a = ap.parse_args(argv)

    w = 2*np.pi*a.freq
    A = a.side**2
    R = a.side/np.sqrt(np.pi)            # equal-area cylinder
    print("bar %g x %g x %g m, lambda %g m  (side/lambda = %.2f)"
          % (a.length, a.side, a.side, a.lam, a.side/a.lam))
    print("bulk kinetic  mu*lam^2*l/A = %.6g H" % (MU0*a.lam**2*a.length/A))
    print("equal-area cylinder says kinetic/bulk ~ %.3f (R/lam = %.2f)"
          % (cylinder_ratio(R/a.lam), R/a.lam))
    print()
    print("%4s %8s %8s %12s %12s %9s"
          % ('nt', 'dx(nm)', 'cells', 'L_kin(H)', 'bulk(H)', 'ratio'))
    for nt in a.nt:
        dx = a.side/nt
        nx = int(round(a.length/dx))
        lk = (imz(a.lam, a.freq, nx, nt, dx, a.modes)
              - imz(a.lam_tiny, a.freq, nx, nt, dx, a.modes))/w
        l_eff = nx*dx
        bulk = MU0*a.lam**2*l_eff/A
        print("%4d %8.1f %8d %12.6g %12.6g %9.4f"
              % (nt, dx*1e9, nx*nt*nt, lk, bulk, lk/bulk), flush=True)


if __name__ == '__main__':
    main()
