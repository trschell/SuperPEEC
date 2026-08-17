# SPDX-License-Identifier: MIT
"""Does the k=3 overshoot need a STAIRCASE boundary, or just dx/delta ~ 1.5?

The wire overshot at dx/delta = 1.513; the earlier bar was tested only at
dx/delta = 0.757, where k=3 was accurate. Same test on a FLAT-boundary
bar at dx/delta = 1.513. If the flat bar also overshoots, the boundary
hypothesis is dead and it is a regime effect.
"""
import os, numpy as np, vhr, equiterminal as eq

SP = os.path.dirname(os.path.abspath(__file__))
SIG = 5.8e7
FREQ = 1e10
LEN, CS = 20e-6, 8e-6           # flat-boundary bar, 8x8 um cross-section


def build(nper):
    dx = 1e-6/nper
    nx, nt = int(round(LEN/dx)), int(round(CS/dx))
    struc = np.ones((nx, nt, nt), dtype=np.int8)
    ports = [(('p1', 'P', 0, j, k, '-x') if s == 0 else
              ('p1', 'N', nx-1, j, k, '+x'))
             for s in (0, 1) for j in range(nt) for k in range(nt)]
    p = os.path.join(SP, 'bar2_%d.vhr' % nper)
    vhr.write_vhr(p, struc, dx, SIG, (FREQ,), ports)
    return p, dx, nt*nt


d = eq.skin_depth(SIG, FREQ)
print("FLAT bar %gum long, %gx%gum, copper, f=%.3g Hz, delta=%.4g m"
      % (LEN*1e6, CS*1e6, CS*1e6, FREQ, d))
print("%-8s %-11s %-8s %-9s %-4s %-13s" %
      ("dx[um]", "grid", "cs[vox]", "dx/delta", "k", "R_10GHz"))
for nper in (1, 2, 3):
    p, dx, ncs = build(nper)
    m = vhr.read_vhr(p)
    for k in ((1, 3) if nper == 1 else (1,)):
        try:
            M = m.build_tree()
            m.prepare(M, FREQ)
            S = eq.EquiTerminalSolver(m, M, 0, subdivide=(k > 1),
                                      skin_freq=(FREQ if k > 1 else None))
            R = S.solve(FREQ)[0].real
            print("%-8.4f %-11s %-8d %-9.3f %-4d %-13.7g"
                  % (dx*1e6, "x".join(map(str, m.dims)), ncs, dx/d, k, R),
                  flush=True)
            del M, S
        except Exception as e:
            print("%-8.4f k=%d FAILED %s: %s"
                  % (dx*1e6, k, type(e).__name__, str(e)[:60]), flush=True)
