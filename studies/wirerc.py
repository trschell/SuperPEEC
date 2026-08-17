# SPDX-License-Identifier: MIT
"""Is the wire's k=3 overshoot TRUNCATION of the mode couplings?

Mode fields are dipolar (mode-aggregate ~1/r^2, mode-mode ~1/r^3) so the
blocks are truncated at rc_uu / rc_cross. On a staircase boundary the
neighbourhood is irregular and that truncation error may not cancel.
If raising the radii collapses the overshoot -> truncation. If not ->
the mode BASIS/geometry, and the boundary-cell test is next.
Converged voxel answer for this wire at 10 GHz = 0.04022542.
"""
import os, numpy as np, vhr, equiterminal as eq
CONV = 0.04022542
K1 = 0.03800266
m = vhr.read_vhr(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'wire_1.vhr'))
print("wire dx=1um, k=3, 10 GHz.  k=1 = %.7g   converged = %.7g" % (K1, CONV))
print("%-14s %-13s %-10s %-s" % ("rc_uu,rc_cross", "R", "vs conv", "correction"))
for ru, rc in ((3, 4), (5, 6), (7, 8), (10, 12)):
    try:
        M = m.build_tree()
        m.prepare(M, 1e10)
        S = eq.EquiTerminalSolver(m, M, 0, subdivide=True, skin_freq=1e10,
                                  rc_uu=ru, rc_cross=rc)
        Z, ii, info = S.solve(1e10)
        R = Z.real
        print("%-14s %-13.7g %-+10.2f%% %-6.0f%% info=%s"
              % ("%d,%d" % (ru, rc), R, 100*(R/CONV-1),
                 100*(R-K1)/(CONV-K1), info), flush=True)
        del M, S
    except Exception as e:
        print("%-14s FAILED %s: %s" % ("%d,%d" % (ru, rc),
                                       type(e).__name__, str(e)[:60]), flush=True)
