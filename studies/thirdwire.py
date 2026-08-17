# SPDX-License-Identifier: MIT
"""THIRD GEOMETRY: is the untruncated direction split driven by
CROSS-SECTION RESOLUTION (r/dx)?

Measured untruncated, same dx=1um and dx/delta=1.513, aspect ratios
already similar (4 and 5), so end effects are unlikely to be the driver:
    r/dx = 3  (24um wire, 32 cs)   ~70%   UNDER-corrects
    r/dx = 5  (50um wire, 80 cs)    201%  OVER-corrects
This runs r/dx = 4 (aspect 5, matching the 50um wire) with its OWN
multi-mesh reference. If the delivered fraction lands monotonically
between 70% and 201%, cross-section resolution is the driver.

FIVE-ish mesh points for the reference, NOT two: R oscillates with dx
because the voxelised circle's AREA does, so two adjacent points are
weak evidence either way (learned the hard way 2026-08-02).
"""
import os, time, numpy as np, vhr, equiterminal as eq
from scipy.special import iv

SPD = os.path.dirname(os.path.abspath(__file__))
SIG, MU = 5.8e7, 4*np.pi*1e-7
RAD, LEN, FREQ = 4e-6, 40e-6, 1e10          # r/dx = 4 at dx = 1um, aspect 5
RHO = 1.0/SIG


def analytic(f):
    w = 2*np.pi*f
    g = np.sqrt(1j*w*MU*SIG)
    return ((g*RHO)/(2*np.pi*RAD)*iv(0, g*RAD)/iv(1, g*RAD)*LEN).real


def hwm():
    for ln in open('/proc/self/status'):
        if ln.startswith('VmHWM:'):
            return int(ln.split()[1])/1e6
    return 0.0


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
    p = os.path.join(SPD, 'tw_%d.vhr' % nper)
    vhr.write_vhr(p, struc, dx, SIG, (FREQ,), ports)
    return p, dx, int(disc.sum())


d = eq.skin_depth(SIG, FREQ)
print("third wire: L=%gum r=%gum (r/dx=%g at dx=1um)  delta=%.4g m"
      % (LEN*1e6, RAD*1e6, RAD/1e-6, d))
print("ANALYTIC round wire @10GHz = %.7g ohm\n" % analytic(FREQ))
print("%-8s %-11s %-6s %-8s %-4s %-9s %-13s %-7s %s"
      % ("dx[um]", "grid", "cs", "dx/delta", "k", "rc", "R_10GHz",
         "peakGB", "min"))
# max separation on the dx=1um grid is 39 along x -> rc >= 45 untruncated
runs = [(1, 1, None), (2, 1, None), (3, 1, None), (4, 1, None),
        (1, 3, (3, 4)), (1, 3, (20, 20)), (1, 3, (45, 45)),
        (1, 3, (50, 50))]
res = {}
for nper, k, rc in runs:
    p, dx, ncs = build(nper)
    m = vhr.read_vhr(p)
    t0 = time.perf_counter()
    try:
        M = m.build_tree()
        m.prepare(M, FREQ)
        kw = dict(subdivide=(k > 1), skin_freq=(FREQ if k > 1 else None))
        if rc:
            kw.update(rc_uu=rc[0], rc_cross=rc[1])
        S = eq.EquiTerminalSolver(m, M, 0, **kw)
        R = S.solve(FREQ)[0].real
        res[(nper, k, rc)] = R
        print("%-8.4f %-11s %-6d %-8.3f %-4d %-9s %-13.7g %-7.2f %.1f"
              % (dx*1e6, "x".join(map(str, m.dims)), ncs, dx/d, k,
                 str(rc) if rc else "-", R, hwm(),
                 (time.perf_counter()-t0)/60.0), flush=True)
        del M, S
    except Exception as e:
        print("%-8.4f k=%d rc=%s FAILED %s: %s"
              % (dx*1e6, k, rc, type(e).__name__, str(e)[:60]), flush=True)

k1 = res.get((1, 1, None))
ref = [res[key] for key in res if key[1] == 1 and key[0] > 1]
if k1 and ref:
    conv = float(np.mean(ref[-2:])) if len(ref) >= 2 else ref[-1]
    print("\nk=1 (dx=1um) = %.7g" % k1)
    print("converged reference (mean of two finest k=1) = %.7g" % conv)
    for key in sorted(res):
        if key[1] == 3:
            R = res[key]
            print("  k=3 rc=%-9s R=%.7g  vs conv %+7.2f%%  %.0f%% of needed"
                  % (str(key[2]), R, 100*(R/conv-1),
                     100*(R-k1)/(conv-k1)))
