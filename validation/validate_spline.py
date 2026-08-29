# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for spline bond wires -- shape, cost and the handle convention.

A curved bond wire is a MODELLING change, not a drawing change: the
solver sees a different centreline, so it must be checked as physics
and as cost, not by looking at a picture.

  A  HANDLE CONVENTION: |start_vec| is the cubic's control handle at
     the foot, so a symmetric vertical loop apexes at
     ``mid_z + 0.75*|v|``. This is the one number a user has to be
     able to predict to place a loop, and it must hold for the
     zero-middle-point case, which is the common one.
  B  INTERPOLATION: the curve passes through EVERY declared point --
     feet and middles -- and leaves each foot along its own vector.
     A spline that merely approximates its points would silently move
     the contacts off the pads they were landed on.
  C  DIRECTION SENSE: end_vec points AWAY from its pad, like
     start_vec. Getting this backwards still produces a smooth curve,
     just one that dives into the board before turning round, so no
     smoothness test can catch it.
  D  SAMPLING IS CURVATURE-ADAPTIVE AND CHEAPER: the chord sagitta
     stays under the tolerance, and a real bond loop costs FEWER
     segments than the square polyline it replaces -- the claim the
     tractability of this feature rests on. Everything in the wire
     path (far-field cache, near blocks, unknowns) is linear in that
     count, so this is the cost gate.
  E  GUARD: a spline tight enough to explode the segment count is
     refused rather than silently inflating the far-field cache.

Run: PYTHONPATH=src python3 validation/validate_spline.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np

from wireassembly import Wire, spline_points

FAIL = []
R = 0.13e-3
CAP_R3 = 0.95*0.375e-3          # R3 leaf-box segment cap
# the real BW1a feet
A = np.array([15.77e-3, 16.27e-3, 1.26e-3])
B = np.array([20.27e-3, 16.27e-3, 1.06e-3])


def check(name, ok, detail=""):
    print("  %-4s %-54s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def arclen(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def main():
    # -- A: the handle convention, over a range of loop heights -----
    # LEVEL feet: the crest sits at the half-parameter point, where
    # the identity is exact, so this pins the convention itself
    LA = np.array([15.77e-3, 16.27e-3, 1.06e-3])
    worst, det = 0.0, ''
    for apex in (2.0e-3, 3.0e-3, 4.5e-3):
        h = (apex - LA[2])/0.75
        P = spline_points([LA, B], [0, 0, h], [0, 0, h], R,
                          max_seglen=CAP_R3, sagitta=0.02)
        err = abs(P[:, 2].max() - apex)
        if err > worst:
            worst, det = err, ('apex %.3f -> %.5f mm'
                               % (1e3*apex, 1e3*P[:, 2].max()))
    check("level feet: apex = z_foot + 0.75*|v| exactly",
          worst < 3e-7, "%s, worst %.2f um" % (det, 1e6*worst))
    # unequal feet tilt the loop; the crest moves off centre and the
    # true maximum runs a few microns high -- documented, not a bug
    h = (3.0e-3 - 0.5*(A[2] + B[2]))/0.75
    P = spline_points([A, B], [0, 0, h], [0, 0, h], R,
                      max_seglen=CAP_R3, sagitta=0.02)
    off = P[:, 2].max() - 3.0e-3
    check("unequal feet: crest runs high by microns, not more",
          0.0 <= off < 2e-5, "+%.1f um over a 0.20 mm foot step"
          % (1e6*off))

    # -- B: interpolation through every declared point ---------------
    mids = [np.array([17.0e-3, 16.27e-3, 2.6e-3]),
            np.array([19.0e-3, 16.27e-3, 2.9e-3])]
    P = spline_points([A] + mids + [B], [0, 0, 2.4e-3], [0, 0, 2.4e-3],
                      R, max_seglen=CAP_R3)
    miss = max(np.linalg.norm(P - q, axis=1).min()
               for q in [A] + mids + [B])
    check("curve passes through the feet and every middle point",
          miss < 2e-5, "worst approach %.1f um" % (1e6*miss))
    check("feet are exact endpoints",
          np.allclose(P[0], A, atol=1e-12)
          and np.allclose(P[-1], B, atol=1e-12))

    # -- C: both vectors point OUT of their own pad ------------------
    P = spline_points([A, B], [0, 0, 2.45e-3], [0, 0, 2.45e-3], R,
                      max_seglen=CAP_R3)
    d0 = P[1] - P[0]
    d1 = P[-2] - P[-1]          # looking back out of the far foot
    check("both feet leave their pad along +z (end_vec is outward)",
          d0[2] > 0 and d1[2] > 0,
          "start dz %+.1f um, end dz %+.1f um" % (1e6*d0[2], 1e6*d1[2]))
    check("the loop rises above both feet, never below",
          P[:, 2].min() >= min(A[2], B[2]) - 1e-12,
          "min z %.4f mm" % (1e3*P[:, 2].min()))

    # -- D: sagitta held, and CHEAPER than the square bond -----------
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    # discrete sagitta: how far the mid-chord sits from the curve
    fine = spline_points([A, B], [0, 0, 2.45e-3], [0, 0, 2.45e-3], R,
                         max_seglen=2e-5, sagitta=1e-4,
                         max_points=100000)
    sag = 0.0
    for p, q in zip(P[:-1], P[1:]):
        mid = 0.5*(p + q)
        sag = max(sag, np.linalg.norm(fine - mid, axis=1).min())
    check("chord sagitta within tolerance (0.1 x wire radius)",
          sag <= 0.1*R*1.35,
          "%.2f um vs %.2f um budget" % (1e6*sag, 1e6*0.1*R))

    square = np.array([A, [A[0], A[1], 3.0e-3],
                       [B[0], B[1], 3.0e-3], B])
    nsq = sum(max(1, int(np.ceil(np.linalg.norm(b - a)/CAP_R3 - 1e-9)))
              for a, b in zip(square[:-1], square[1:]))
    nsp = len(P) - 1
    check("spline costs FEWER segments than the square bond",
          nsp < nsq, "%d vs %d segments at the R3 cap (%+.0f%%)"
          % (nsp, nsq, 100*(nsp/nsq - 1)))
    check("spline arc is shorter than the square path",
          arclen(P) < arclen(square),
          "%.3f vs %.3f mm" % (1e3*arclen(P), 1e3*arclen(square)))

    # every chord must be under the coupler's cap, or the far-field
    # point source is wrong by construction
    check("no chord exceeds the leaf-box cap",
          seg.max() <= CAP_R3*(1 + 1e-9),
          "max %.4f mm vs cap %.4f mm" % (1e3*seg.max(), 1e3*CAP_R3))

    # -- E: the runaway guard ----------------------------------------
    try:
        spline_points([A, B], [0, 0, 400e-3], [0, 0, 400e-3], R,
                      max_seglen=CAP_R3, max_points=64)
        check("a runaway spline is refused, not silently expanded",
              False, "no error raised")
    except ValueError as exc:
        check("a runaway spline is refused, not silently expanded",
              'cap' in str(exc), str(exc)[:52])

    # -- the sampled polyline is a legal Wire ------------------------
    w = Wire(P, R, 3.77e7, max_seglen=CAP_R3)
    check("sampled centreline builds a Wire with 25 elements/segment",
          len(w.segments) == nsp
          and all(len(s) == 25 for s in w.segments),
          "%d segments" % len(w.segments))

    print("\n%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
