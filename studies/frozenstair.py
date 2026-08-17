# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""FROZEN-STAIRCASE convergence: refine the MESH while FREEZING the shape.

THE QUESTION. basis4wire.py showed the skin engine's delivered fraction
is BASIS-INDEPENDENT (202-216% over on the 50um wire, 67-72% under on
the 24um, untruncated), so the defect is not cross-section span. The
staircase hypothesis: the converged references (0.0403 / 0.0340) come
from re-voxelising at finer dx, which converges the GEOMETRY (voxel
disc -> true circle) along with the discretisation -- so the "needed
correction" conflates sub-cell current redistribution (which per-cell
modes can deliver) with SHAPE convergence (which they never can).

THE TEST. Refine each dx=1um coarse voxel into nper^3 children that
INHERIT the coarse cell's occupancy: the staircase cross-section is
bit-frozen while the mesh refines. The limit R_frozen is the answer to
the question the mode engine is actually being asked. Then:

    R_frozen ~ engine's untruncated R  -> staircase IS the driver;
        the engine is (approximately) right about its own geometry and
        the "erratic delivered fraction" is a category error in the
        reference. Subpixel geometry is the right program.
    R_frozen ~ circle-converged R      -> staircase is INNOCENT;
        back to proximity / terminal coupling.

SELF-CHECK built in: with the shape frozen and sigma uniform, R_DC is
exactly length/(sigma*A) independent of nper -- any drift is a meshing
artefact, not physics.

RESULT 2026-08-05 -- STAIRCASE CONFIRMED; THE ENGINE WAS NEVER ERRATIC.
R_DC frozen to all digits at every nper (meshing clean). R_10GHz
converges as dx**2 (3-point Richardson fits to 1%):

    50um: 0.03800266 / 0.04188511 / 0.04268677 / 0.0429652 -> ~0.04332
    24um: 0.02862556 / 0.03168498 / 0.03229478 / 0.03251211 -> ~0.03279

The frozen-staircase truth differs from the circle truth in OPPOSITE
directions on the two wires -- +7.4%% (50um: corner/edge crowding on
the jagged boundary raises R; area excess only 1.9%%) and -3.6%% (24um:
13%% voxel area excess lowers R and dominates). That sign flip IS the
recorded "direction split". Re-based against the frozen (same-shape)
references, the untruncated delivered fractions collapse to geometry-
INDEPENDENT values:

    basis          50um    24um    (flat bar, recorded: 86%%)
    diff k=3       87.4%%   86.7%%
    linear k=6     87.9%%   88.0%%
    cond k=6       93.4%%   93.2%%

So: (1) the mode engine is a well-behaved ~87-93%% under-correction of
the question it is actually asked, on every geometry tested; (2) the
conduction basis IS the best cross-section basis, +6 points over
diff/linear, exactly as the 2-D span testbed predicted; (3) the
remaining wire error vs PHYSICAL truth is the staircase GEOMETRY error
(+-4-7%% in R here), which no fixed-shape basis can fix -- that is the
subpixel/partial-fill program's target, now quantified.

Usage:  PYTHONPATH=src python3 studies/frozenstair.py
"""
import os
import time

import numpy as np

import vhr
import equiterminal as eq

SPD = os.path.dirname(os.path.abspath(__file__))
SIG = 5.8e7
F = 1e10

#           tag    LEN    RAD   engine untrunc   circle-conv
WIRES = [('50um', 50e-6, 5e-6, 0.04265113, 0.0403),
         ('24um', 24e-6, 3e-6, 0.03223224, 0.0340)]


def build(tag, LEN, RAD, nper):
    """dx = 1um/nper, occupancy INHERITED from the dx=1um staircase."""
    dxc = 1e-6
    nxc, ntc = int(round(LEN/dxc)), int(round(2*RAD/dxc))
    c = (np.arange(ntc) + 0.5 - ntc/2.0)*dxc
    yy, zz = np.meshgrid(c, c, indexing='ij')
    disc = (yy**2 + zz**2) < RAD**2              # the frozen staircase
    fine = np.repeat(np.repeat(disc, nper, axis=0), nper, axis=1)
    nx = nxc*nper
    struc = np.tile(fine[None, :, :], (nx, 1, 1)).astype(np.int8)
    ports = []
    nt = ntc*nper
    for j in range(nt):
        for k in range(nt):
            if fine[j, k]:
                ports.append(('p1', 'P', 0, j, k, '-x'))
                ports.append(('p1', 'N', nx - 1, j, k, '+x'))
    p = os.path.join(SPD, 'fs_%s_%d.vhr' % (tag, nper))
    vhr.write_vhr(p, struc, dxc/nper, SIG, (1.0, F), ports)
    return p, int(disc.sum())


def hwm():
    for ln in open('/proc/self/status'):
        if ln.startswith('VmHWM:'):
            return int(ln.split()[1])/1e6
    return 0.0


for tag, LEN, RAD, r_engine, r_circle in WIRES:
    ncs = None
    print("\n== %s wire, staircase FROZEN at dx=1um; engine untrunc "
          "%.7g, circle-conv ~%.4g ==" % (tag, r_engine, r_circle),
          flush=True)
    print("%-5s %-11s %-9s %-13s %-13s %-7s %s"
          % ("nper", "grid", "dx/delta", "R_DC", "R_10GHz", "peakGB",
             "min"))
    for nper in (1, 2, 3, 4):
        t0 = time.perf_counter()
        try:
            p, ncs = build(tag, LEN, RAD, nper)
            m = vhr.read_vhr(p)
            M = m.build_tree()
            out = []
            for f in (1.0, F):
                m.prepare(M, f)
                S = eq.EquiTerminalSolver(m, M, 0)
                out.append(S.solve(f)[0].real)
                del S
            del M
            print("%-5d %-11s %-9.3f %-13.7g %-13.7g %-7.1f %.1f"
                  % (nper, "x".join(map(str, m.dims)),
                     (1e-6/nper)/eq.skin_depth(SIG, F), out[0], out[1],
                     hwm(), (time.perf_counter() - t0)/60.0), flush=True)
        except Exception as e:
            print("%-5d FAILED %s: %s"
                  % (nper, type(e).__name__, str(e)[:70]), flush=True)
            break
    if ncs:
        print("  (exact frozen-shape R_DC = %.7g: %d um^2 staircase area)"
              % (LEN/(SIG*ncs*1e-12), ncs), flush=True)
