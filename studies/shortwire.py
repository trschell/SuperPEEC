# SPDX-License-Identifier: MIT
"""SHORT wire: small enough that the UNTRUNCATED mode model fits under the
6 GB cap, with its own mesh-refinement reference.

Settles the fork left open on the 50um wire:
  (a) the k x k piecewise-constant basis over-predicts on a staircase
      boundary  -> untruncated k=3 will sit well ABOVE the converged
      mesh-refined answer;
  (b) the reference was soft     -> untruncated k=3 will sit ON it.

Wire radius 3um, length 24um, copper, 10 GHz (delta = 0.6609um), so the
coarse mesh dx=1um is at dx/delta = 1.513 -- the same regime as before.
Domain is 24 x 6 x 6 cells, so rc = 40 is UNTRUNCATED.
"""
import os, numpy as np, vhr, equiterminal as eq
from scipy.special import iv

SPD = os.path.dirname(os.path.abspath(__file__))
SIG, MU = 5.8e7, 4*np.pi*1e-7
LEN, RAD, FREQ = 24e-6, 3e-6, 1e10
RHO = 1.0/SIG


def analytic(f):
    w = 2*np.pi*f
    g = np.sqrt(1j*w*MU*SIG)
    return ((g*RHO)/(2*np.pi*RAD)*iv(0, g*RAD)/iv(1, g*RAD)*LEN).real


def build(nper):
    dx = 1e-6/nper
    nx, nt = int(round(LEN/dx)), int(round(2*RAD/dx))
    c = (np.arange(nt) + 0.5 - nt/2.0)*dx
    yy, zz = np.meshgrid(c, c, indexing='ij')
    disc = (yy**2 + zz**2) < RAD**2
    struc = np.tile(disc[None, :, :], (nx, 1, 1)).astype(np.int8)
    ports = [(('p1', 'P', 0, j, k, '-x') if s == 0 else
              ('p1', 'N', nx-1, j, k, '+x'))
             for s in (0, 1) for j in range(nt) for k in range(nt)
             if disc[j, k]]
    p = os.path.join(SPD, 'sw_%d.vhr' % nper)
    vhr.write_vhr(p, struc, dx, SIG, (FREQ,), ports)
    return p, dx, int(disc.sum())


d = eq.skin_depth(SIG, FREQ)
print("short wire L=%gum r=%gum copper  delta=%.4g m  analytic(round)=%.7g"
      % (LEN*1e6, RAD*1e6, d, analytic(FREQ)))
print("%-8s %-11s %-7s %-8s %-5s %-9s %-13s %s"
      % ("dx[um]", "grid", "cs", "dx/delta", "k", "rc", "R", "resid"))
runs = [(1, 1, None), (2, 1, None), (3, 1, None), (4, 1, None),
        (1, 3, (3, 4)), (1, 3, (6, 6)), (1, 3, (8, 8)), (1, 3, (12, 12)),
        (1, 3, (16, 16)), (1, 3, (20, 20)), (1, 3, (25, 25))]
for nper, k, rc in runs:
    p, dx, ncs = build(nper)
    m = vhr.read_vhr(p)
    try:
        M = m.build_tree()
        m.prepare(M, FREQ)
        kw = dict(subdivide=(k > 1), skin_freq=(FREQ if k > 1 else None))
        if rc:
            kw.update(rc_uu=rc[0], rc_cross=rc[1])
        S = eq.EquiTerminalSolver(m, M, 0, **kw)
        Z, ii, info = S.solve(FREQ)
        print("%-8.4f %-11s %-7d %-8.3f %-5d %-9s %-13.7g %.1e"
              % (dx*1e6, "x".join(map(str, m.dims)), ncs, dx/d, k,
                 str(rc) if rc else "-", Z.real, info['residual']), flush=True)
        del M, S
    except Exception as e:
        print("%-8.4f k=%d rc=%s FAILED %s: %s"
              % (dx*1e6, k, rc, type(e).__name__, str(e)[:60]), flush=True)
