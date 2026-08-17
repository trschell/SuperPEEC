# SPDX-License-Identifier: MIT
"""Mesh-convergence ground truth for the WIRE, plus an ANALYTIC reference.

Round copper wire, 50 um long, 10 um diameter, at DC and 10 GHz. The
voxelised circle is refined; unlike the rectangular bar this also
improves the GEOMETRY (voxel area -> pi a^2), so both discretisation and
shape converge together, and the limit is a genuine round wire -- for
which the exact answer is known in closed form. Two independent ground
truths, neither of which is VoxHenry.

  DC   : R = rho L / (pi a^2)
  AC   : Z' = (gamma rho)/(2 pi a) * I0(gamma a)/I1(gamma a),
         gamma = sqrt(j w mu sigma)      (exact round-wire skin effect)
"""
import os, numpy as np, vhr, equiterminal as eq
from scipy.special import iv

SP = os.path.dirname(os.path.abspath(__file__))
SIG, MU = 5.8e7, 4*np.pi*1e-7
LEN, RAD = 50e-6, 5e-6
RHO = 1.0/SIG


def analytic(f):
    if f <= 0:
        return RHO*LEN/(np.pi*RAD**2)
    w = 2*np.pi*f
    g = np.sqrt(1j*w*MU*SIG)
    zp = (g*RHO)/(2*np.pi*RAD)*iv(0, g*RAD)/iv(1, g*RAD)
    return (zp*LEN).real


def build(nper):
    """Voxelised wire at dx = 1/nper um; voxel counts as conductor when
    its CENTRE lies inside the radius (reproduces the shipped 80 at
    nper=1)."""
    dx = 1e-6/nper
    nx, nt = int(round(LEN/dx)), int(round(2*RAD/dx))
    c = (np.arange(nt) + 0.5 - nt/2.0)*dx
    yy, zz = np.meshgrid(c, c, indexing='ij')
    disc = (yy**2 + zz**2) < RAD**2
    struc = np.tile(disc[None, :, :], (nx, 1, 1)).astype(np.int8)
    ports = []
    for j in range(nt):
        for k in range(nt):
            if disc[j, k]:
                ports.append(('p1', 'P', 0, j, k, '-x'))
                ports.append(('p1', 'N', nx-1, j, k, '+x'))
    p = os.path.join(SP, 'wire_%d.vhr' % nper)
    vhr.write_vhr(p, struc, dx, SIG, (1.0, 1e10), ports)
    return p, dx, int(disc.sum())


print("round wire: L=%gum  dia=%gum  copper   skin depth @10GHz = %.4g m"
      % (LEN*1e6, 2*RAD*1e6, eq.skin_depth(SIG, 1e10)))
print("ANALYTIC   DC = %.7g ohm    10 GHz = %.7g ohm"
      % (analytic(0), analytic(1e10)))
print("shipped VoxHenry            0.0109123           0.0402259\n")
def hwm():
    for ln in open('/proc/self/status'):
        if ln.startswith('VmHWM:'):
            return int(ln.split()[1])/1e6
    return 0.0


import time as _t
print("%-8s %-11s %-8s %-8s %-4s %-13s %-13s %-8s %s"
      % ("dx[um]", "grid", "cs[vox]", "dx/delta", "k", "R_DC", "R_10GHz",
         "peakGB", "min"))
for nper in (1, 2, 3, 4, 5):
    p, dx, ncs = build(nper)
    m = vhr.read_vhr(p)
    for k in ((1, 3) if nper == 1 else (1,)):
        _t0 = _t.perf_counter()
        try:
            M = m.build_tree()
            out = []
            for f in (1.0, 1e10):
                m.prepare(M, f)
                S = eq.EquiTerminalSolver(
                    m, M, 0, subdivide=(k > 1),
                    skin_freq=(1e10 if k > 1 else None))
                out.append(S.solve(f)[0].real)
                del S
            print("%-8.4f %-11s %-8d %-8.3f %-4d %-13.7g %-13.7g %-8.2f %.1f"
                  % (dx*1e6, "x".join(map(str, m.dims)), ncs,
                     dx/eq.skin_depth(SIG, 1e10), k, out[0], out[1],
                     hwm(), (_t.perf_counter()-_t0)/60.0),
                  flush=True)
            del M
        except Exception as e:
            print("%-8.4f k=%d FAILED %s: %s"
                  % (dx*1e6, k, type(e).__name__, str(e)[:60]), flush=True)
