# SPDX-License-Identifier: MIT
"""What does the mode model give UNTRUNCATED? And do rc_uu / rc_cross
act in opposite directions?

Truncating these blocks changes the OPERATOR, not a sum: dropping
mode-aggregate (Zcross) under-drives the modes; dropping mode-mode (Zuu)
removes the mutual opposition limiting their response. Previous sweeps
moved both together so the two effects were confounded.

Wire is 50 cells long, 10 across -> max separation 49, so rc >= 50 is
UNTRUNCATED. k=1 = 0.03800266; converged = 0.04032062 from the 5-mesh
re-run (dx to 0.2um) -- the earlier 0.04022542 was a 3-mesh value.
"""
import os, numpy as np, vhr, equiterminal as eq
CONV, K1 = 0.04032062, 0.03800266   # CONV from the 5-mesh re-run (dx=0.2um)
m = vhr.read_vhr(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'wire_1.vhr'))
print("%-14s %-13s %-9s %-8s %s" %
      ("rc_uu,rc_cross", "R", "vs conv", "corr", "resid"))
# Domain is 50 x 10 x 10 cells, so max separation is 49 along x:
# rc >= 50 is UNTRUNCATED. Ascending so partial results survive.
for ru, rc in ((16, 16), (20, 20), (25, 25), (30, 30), (40, 40),
               (50, 50), (55, 55)):
    try:
        M = m.build_tree()
        m.prepare(M, 1e10)
        S = eq.EquiTerminalSolver(m, M, 0, subdivide=True, skin_freq=1e10,
                                  rc_uu=ru, rc_cross=rc)
        Z, ii, info = S.solve(1e10)
        print("%-14s %-13.7g %-+9.2f%% %-8.0f%% %.1e"
              % ("%d,%d" % (ru, rc), Z.real, 100*(Z.real/CONV-1),
                 100*(Z.real-K1)/(CONV-K1), info['residual']), flush=True)
        del M, S
    except Exception as e:
        print("%-14s FAILED %s: %s" % ("%d,%d" % (ru, rc),
                                       type(e).__name__, str(e)[:70]), flush=True)
