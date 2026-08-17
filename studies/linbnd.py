# SPDX-License-Identifier: MIT
"""Test lin_bnd in the real solver on the 50um wire.

Offline prediction (studies/xsection.py, infinite-wire 2-D): linear modes
on BOUNDARY cells recover ~98% of the sub-cell error, beating linear
modes everywhere (91%) at fewer DOF. Absolute % will differ here -- the
3-D wire is only 5 diameters long and has less skin effect to correct
(gap 6.1% of k=1 vs 14.1% in 2-D) -- but the RANKING should hold.

Reference, same geometry, 3-D solver:
    k=1                     0.03800266
    converged (5 meshes)    0.04032062     needed = 0.00231796
    shipped diff basis, k=3, untruncated   0.04265113  = 201%
"""
import os, time, numpy as np, vhr, equiterminal as eq

K1, CONV = 0.03800266, 0.04032062
m = vhr.read_vhr(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'wire_1.vhr'))
print("%-9s %-6s %-6s %-9s %-7s %-13s %-9s %s"
      % ("basis", "bnd", "k", "rc", "nmode", "R_10GHz", "of needed", "min"))
cases = [('diff',   False, 3, (3, 4)),
         ('diff',   False, 3, (20, 20)),
         ('linear', False, 3, (3, 4)),
         ('linear', False, 3, (20, 20)),
         ('linear', True,  3, (3, 4)),
         ('linear', True,  3, (20, 20)),
         ('linear', True,  3, (50, 50)),
         ('linear', True,  6, (20, 20))]
for mb, bo, k, rc in cases:
    t0 = time.perf_counter()
    try:
        M = m.build_tree()
        m.prepare(M, 1e10)
        S = eq.EquiTerminalSolver(m, M, 0, subdivide=k, skin_freq=1e10,
                                  rc_uu=rc[0], rc_cross=rc[1],
                                  mode_basis=mb, boundary_only=bo)
        R = S.solve(1e10)[0].real
        print("%-9s %-6s %-6d %-9s %-7d %-13.7g %-9.0f%% %.1f"
              % (mb, bo, k, str(rc), S.redist.nmode, R,
                 100*(R-K1)/(CONV-K1), (time.perf_counter()-t0)/60.0),
              flush=True)
        del M, S
    except Exception as e:
        print("%-9s %-6s %-6d %-9s FAILED %s: %s"
              % (mb, bo, k, str(rc), type(e).__name__, str(e)[:60]),
              flush=True)
