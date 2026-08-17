# SPDX-License-Identifier: MIT
"""GROUND TRUTH for the subdivision question, independent of VoxHenry.

Fixed PHYSICAL bar (10 um long, 2x2 um cross-section, copper) at 10 GHz,
solved on progressively finer meshes with subdivision OFF, plus the
coarsest mesh WITH subdivision. If mesh refinement drives R toward the
k=3 value, subdivision is right and the coarse mesh is under-resolved.
If it drives R toward the k=1 value, subdivision over-corrects.
Nothing here refers to VoxHenry.
"""
import os, numpy as np, vhr, equiterminal as eq

SP = os.path.dirname(os.path.abspath(__file__))
SIGMA = 5.8e7
FREQ = 1e10
LEN, CS = 10e-6, 2e-6          # physical bar: 10 um long, 2x2 um


def build(nx_per_um):
    """Bar meshed at dx = 1/nx_per_um um."""
    dx = 1e-6/nx_per_um
    nx, nt = int(round(LEN/dx)), int(round(CS/dx))
    struc = np.ones((nx, nt, nt), dtype=np.int8)
    ports = []
    for j in range(nt):
        for k in range(nt):
            ports.append(('p1', 'P', 0, j, k, '-x'))
            ports.append(('p1', 'N', nx-1, j, k, '+x'))
    p = os.path.join(SP, 'bar_%d.vhr' % nx_per_um)
    vhr.write_vhr(p, struc, dx, SIGMA, (FREQ,), ports)
    return p, dx, struc.sum()


print("bar %gum long, %gx%gum cross-section, copper, f=%.3g Hz"
      % (LEN*1e6, CS*1e6, CS*1e6, FREQ))
print("skin depth = %.4g m\n" % eq.skin_depth(SIGMA, FREQ))
print("%-9s %-9s %-8s %-8s %-6s %-13s" %
      ("dx[um]", "grid", "voxels", "dx/delta", "k", "R [ohm]"))
base = {}
for n in (2, 3, 4, 6):
    p, dx, nv = build(n)
    m = vhr.read_vhr(p)
    for k in ((1, 3) if n == 2 else (1,)):
        M = m.build_tree()
        m.prepare(M, FREQ)
        S = eq.EquiTerminalSolver(m, M, 0, subdivide=(k > 1),
                                  skin_freq=(FREQ if k > 1 else None))
        Z, i, info = S.solve(FREQ)
        d = eq.skin_depth(SIGMA, FREQ)
        print("%-9.4f %-9s %-8d %-8.3f %-6d %-13.7g"
              % (dx*1e6, "x".join(map(str, m.dims)), nv, dx/d, k, Z.real),
              flush=True)
        base[(n, k)] = Z.real
        del M, S
print()
r1 = base[(2, 1)]; r3 = base[(2, 3)]
print("coarsest mesh:  k=1 -> %.7g   k=3 -> %.7g   (k=3 is %+.2f%%)"
      % (r1, r3, 100*(r3/r1 - 1)))
fine = base[(6, 1)]
print("finest mesh (k=1, 3x finer): %.7g  -> %+.2f%% vs coarse k=1"
      % (fine, 100*(fine/r1 - 1)))
