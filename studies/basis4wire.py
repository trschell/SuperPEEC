# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""PHASE 4 of the conduction-mode program: do the new bases fix the wires?

THE QUESTION. The k x k piecewise-constant ('diff') basis is erratic
across geometries at the same dx/delta = 1.513 (assembly exact,
truncation exonerated -- see memory/skin-engine-status and README):

    24um wire (r/dx=3, 32 cs)  delivers  ~67% of the correction  UNDER
    50um wire (r/dx=5, 80 cs)  delivers  201% untruncated        OVER
                               (244% at the default rc=(3,4))

'delivered' = (R_basis - R_k1)/(R_conv - R_k1) at 10 GHz, against each
wire's own refined-mesh reference (0.0403 +/- 0.4% for the 50um wire
from the 5-mesh re-run; ~0.0340 for the 24um). A basis that fixes the
engine should land near 100% on BOTH wires -- the direction split is
the discriminating observable, not either wire alone.

CANDIDATES. 'diff' k=3 (shipped, km=8), 'linear' k=6 (PWL tilt, km=2),
'conduction' k=6 and k=8 (Daniel/Sangiovanni-Vincentelli/White face and
corner exponentials, km=8 at this dx/delta; k is pure quadrature). Each
at the default rc=(3,4) AND untruncated (rc = domain size), because
rc_uu/rc_cross act in opposite directions and the recorded anchors are
untruncated. The untruncated diff rows must REPRODUCE the recorded
0.04265113 / 0.03223 -- they double as a regression check on the
Phase 2/3 Redistribution refactor.

RESULT 2026-08-05 -- THE BASIS IS EXONERATED AS THE DRIVER. Delivered
fraction of the correction, both wires, dx/delta = 1.513, 10 GHz:

    basis                50um rc(3,4)  50um untrunc  24um rc(3,4)  24um untrunc
    diff   k=3  km=8     246.4%        202.3%        22.1%         67.1%
    linear k=6  km=2     253.3%        203.5%        20.0%         68.1%
    cond   k=6  km=8     263.8%        216.2%        23.6%         72.2%
    cond   k=8  km=8     267.1%        --            24.1%         --

Both recorded anchors reproduced exactly (0.04265113, 0.03223224), so
the Phase 2/3 engine refactor is regression-clean. THE READING: the
delivered fraction is BASIS-INDEPENDENT to ~10 points on both wires --
untruncated 202-216% (over) on the 50um wire, 67-72% (under) on the
24um -- while in the 2-D single-bar testbed the same conduction basis
beats diff 6.6x. So the 3-D direction split is NOT a cross-section
span problem: it is driven by something these per-cell bases cannot
express regardless of shape. Note also the rc(3,4) -> untruncated gap
is LARGE and OPPOSITE-SIGNED on the two wires (246 -> 202 down, 22 ->
67 up): the default truncation radii do real, geometry-dependent
damage on wires.

Next probe (the Phase-1 fallback, cheap): extend studies/modebasis2d.py
to TWO COUPLED BARS -- if mutual proximity reproduces an over-shoot no
single-bar basis can fix, the driver is identified.

Usage:  PYTHONPATH=src python3 studies/basis4wire.py
"""
import faulthandler
import os
import signal
import time

import numpy as np
from scipy.special import iv

import vhr
import equiterminal as eq

faulthandler.register(signal.SIGUSR1)      # kill -USR1 <pid> = live stack

SPD = os.path.dirname(os.path.abspath(__file__))
SIG, MU = 5.8e7, 4*np.pi*1e-7
RHO = 1.0/SIG
F = 1e10

WIRES = [
    dict(tag='50um', LEN=50e-6, RAD=5e-6, rconv=0.0403,
         anchor=('diff k=3 untrunc', 0.04265113)),
    dict(tag='24um', LEN=24e-6, RAD=3e-6, rconv=0.0340,
         anchor=('diff k=3 untrunc', 0.03223)),
]


def analytic(LEN, RAD):
    w = 2*np.pi*F
    g = np.sqrt(1j*w*MU*SIG)
    return ((g*RHO)/(2*np.pi*RAD)*iv(0, g*RAD)/iv(1, g*RAD)*LEN).real


def build(tag, LEN, RAD):
    """Same voxelisation rule as wireconv/shortwire at dx = 1um."""
    dx = 1e-6
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
    p = os.path.join(SPD, 'b4w_%s.vhr' % tag)
    vhr.write_vhr(p, struc, dx, SIG, (F,), ports)
    return p, int(disc.sum()), nx


for cfg in WIRES:
    tag, LEN, RAD, rconv = cfg['tag'], cfg['LEN'], cfg['RAD'], cfg['rconv']
    path, ncs, nx = build(tag, LEN, RAD)
    rcU = nx + 1                      # > max cell separation: untruncated
    m = vhr.read_vhr(path)
    M = m.build_tree()
    m.prepare(M, F)
    print("\n== %s wire: r/dx=%g, %d cs voxels, dx/delta=%.3f, "
          "analytic %.7g, R_conv ~ %.4g ==" %
          (tag, RAD*1e6, ncs, 1e-6/eq.skin_depth(SIG, F), analytic(LEN, RAD),
           rconv), flush=True)
    print("%-22s %-4s %-13s %-9s %-10s %-6s %-7s %-7s %s"
          % ("basis", "km", "R_10GHz", "vs conv", "delivered", "mv",
             "setup_s", "solve_s", "GB"))
    runs = [
        ('k=1 (no modes)', dict(subdivide=False)),
        ('diff k=3 rc(3,4)', dict(subdivide=3)),
        ('diff k=3 untrunc', dict(subdivide=3, rc_uu=rcU, rc_cross=rcU)),
        ('linear k=6 rc(3,4)', dict(subdivide=6, mode_basis='linear')),
        ('linear k=6 untrunc', dict(subdivide=6, mode_basis='linear',
                                    rc_uu=rcU, rc_cross=rcU)),
        ('cond k=6 rc(3,4)', dict(subdivide=6, mode_basis='conduction')),
        ('cond k=6 untrunc', dict(subdivide=6, mode_basis='conduction',
                                  rc_uu=rcU, rc_cross=rcU)),
        ('cond k=8 rc(3,4)', dict(subdivide=8, mode_basis='conduction')),
    ]
    r1 = None
    for label, kw in runs:
        t0 = time.perf_counter()
        try:
            S = eq.EquiTerminalSolver(m, M, 0, skin_freq=F, **kw)
            ts = time.perf_counter() - t0
            Z, _, info = S.solve(F)
            R = Z.real
            km = 0 if S.redist is None else S.redist.km
            del S
        except Exception as e:
            print("%-22s FAILED %s: %s"
                  % (label, type(e).__name__, str(e)[:60]), flush=True)
            continue
        gb = 0.0
        for ln in open('/proc/self/status'):
            if ln.startswith('VmHWM:'):
                gb = int(ln.split()[1])/1e6
        stats = "%-6d %-7.1f %-7.1f %.1f" % (info['matvecs'], ts,
                                             info['time'], gb)
        if r1 is None:
            r1 = R
            print("%-22s %-4d %-13.7g %+-9.2f%% %-10s %s"
                  % (label, km, R, 100*(R/rconv - 1), "-", stats),
                  flush=True)
            continue
        dv = 100*(R - r1)/(rconv - r1)
        note = ""
        if cfg['anchor'][0] == label:
            note = "  (recorded %.7g)" % cfg['anchor'][1]
        print("%-22s %-4d %-13.7g %+-9.2f%% %-10.1f %s%s"
              % (label, km, R, 100*(R/rconv - 1), dv, stats, note),
              flush=True)
    del M
