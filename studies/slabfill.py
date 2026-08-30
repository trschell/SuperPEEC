# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""MOCK-UP: can a FILL FRACTION model a film whose thickness does not
divide the cell pitch?

THE DEFECT. RSFQ layers are axis-aligned slabs of fixed physical
thickness (M5 is 135 nm) and a voxel grid can only hold an integer
number of cells, so at a pitch that does not divide the thickness the
staircase is the wrong film: a 100 nm pitch models 135 nm as 100,
-26%. Measured on the XNOR ladder, rungs where the film divides
exactly converge cleanly toward InductEx (0.9246 -> 0.9463 -> 0.9627
at 2, 3, 4 cells through M5) while fractional rungs sit off that trend.

THE PROPOSAL (option 2). The subpixel program already gives boundary
cells a FILL FRACTION and carries sigma_eff = sigma*fill through the
per-cell-conductivity machinery -- but only for ``[[cylinder]]``, and
only for filaments along the cylinder axis. A slab boundary is far
easier than a circle: the fill is a 1-D length ratio, analytic. For a
London superconductor the same rule reads lambda_eff = lambda/sqrt(f):
z = j w mu lambda^2 must scale as 1/f exactly as 1/sigma does, so
lambda^2 scales as 1/f.

THE MEASUREMENT. A slab of FIXED PHYSICAL THICKNESS over a ground
return, driven end to end. The pitch is cubic and constant; what
varies is how many cells the slab is built from, i.e. its MODELLED
thickness:

  exact      the slab is an integer number of cells (reference)
  staircase  one cell short -- the current behaviour at a bad pitch
  fill       the same short grid plus one partial cell whose sigma
             (or lambda) carries the fractional remainder

If `fill` recovers `exact`, the per-cell material law is enough and
option 2 is a small change to existing, validated machinery. If it
does not, the residue is the partial cell's GEOMETRIC footprint in the
Toeplitz/FMM mutual tables, which a material law cannot reach -- and
which is what the subpixel program's stage B (build_dL) exists for.

Run: PYTHONPATH=src python3 studies/slabfill.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'src')]

import equiterminal as eq        # noqa: E402
import vhr                       # noqa: E402

SIG = 5.8e7
FREQ = 1e10
DX = 3e-8                        # cubic pitch, constant everywhere
NX, NY = 20, 6
NGND, NGAP = 2, 2


def run(nfull, frac=0.0, lam=None):
    """Slab of ``nfull`` whole cells plus an optional partial cell.

    ``frac`` in (0, 1) adds ONE more cell carrying that fraction of the
    material: sigma*frac for a normal metal, lambda/sqrt(frac) for a
    London superconductor (so that z = j w mu lam^2 scales as 1/frac
    in BOTH cases -- less material means MORE impedance).
    """
    npart = 1 if frac > 1e-9 else 0
    nz = NGND + NGAP + nfull + npart
    z0 = NGND + NGAP
    struc = np.zeros((NX, NY, nz), dtype=np.int8)
    struc[:, :, :NGND] = 1
    struc[:, :, z0:z0 + nfull + npart] = 1
    sig = np.zeros((NX, NY, nz))
    sig[struc > 0] = SIG
    lm = None
    if lam is not None:
        lm = np.zeros((NX, NY, nz))
        lm[struc > 0] = lam
        sig[:] = 0.0
    if npart:
        if lam is None:
            sig[:, :, z0 + nfull] = SIG*frac
        else:
            lm[:, :, z0 + nfull] = lam/np.sqrt(frac)
    # THE PORT DRIVES EVERY CELL OF THE SLAB, partial one included.
    # That used to be refused ("port 0 spans 2 conductivities ... the
    # terminal half-filament resistance is ambiguous there") because
    # Terminals took ONE scalar sigma for the whole port; its R is now
    # per-face, read from the cell each half-filament sits in. Nothing
    # prescribes how much current the partial face takes: on the
    # equipotential path the split across faces is SOLVED
    # (terminal_split), so a partial cell draws less because it has
    # more impedance.
    ports = []
    for j in range(NY):
        for k in range(z0, z0 + nfull + npart):
            ports.append(('p1', 'P', 0, j, k, '-x'))
            ports.append(('p1', 'N', NX - 1, j, k, '+x'))
    p = os.path.join(HERE, 'slabfill.vhr')
    vhr.write_vhr(p, struc, DX, sig, (FREQ,), ports, lambdaL=lm)
    m = vhr.read_vhr(p)
    leaf, lv = m.partition()
    M = m.build_tree(leaf, lv)
    m.prepare(M, FREQ)
    Z, _, _ = eq.EquiTerminalSolver(m, M, 0).solve(FREQ)
    os.remove(p)
    return Z


def report(tag, lam=None):
    w = 2*np.pi*FREQ
    print("\n== %s ==" % tag)
    print("%-34s %12s %12s %9s"
          % ('model', 'R (Ohm)', 'L (pH)', 'vs exact'))
    ref = None
    for label, nfull, frac in (('exact: 3 cells (reference)', 3, 0.0),
                               ('staircase: 2 cells (-33%)', 2, 0.0),
                               ('fill: 2 cells + 1.00 partial', 2, 1.0),
                               ('fill: 2 cells + 0.50 partial', 2, 0.5),
                               ('fill: 2 cells + 0.25 partial', 2, 0.25)):
        Z = run(nfull, frac, lam)
        L = abs(Z.imag)/w
        if ref is None:
            ref = L
        print("%-34s %12.5g %12.5f %8.4f"
              % (label, Z.real, 1e12*L, L/ref), flush=True)


def truth(lam=None):
    """Is the fill knob CORRECT, not merely smooth?

    A 2.5-cell slab has no integer representation at pitch DX -- but it
    is exactly 5 cells at DX/2. Solving the SAME physical slab on the
    finer grid gives an independent reference for the fractional case.
    The 3-cell/6-cell pair is solved too, so the pitch refinement's own
    error is measured rather than assumed away: whatever it does to a
    thickness the coarse grid CAN represent is the confound floor for
    the fractional comparison.
    """
    global DX, NX, NY, NGND, NGAP
    w = 2*np.pi*FREQ
    coarse = dict(DX=DX, NX=NX, NY=NY, NGND=NGND, NGAP=NGAP)

    def at_half(ncell):
        global DX, NX, NY, NGND, NGAP
        DX, NX, NY = coarse['DX']/2, coarse['NX']*2, coarse['NY']*2
        NGND, NGAP = coarse['NGND']*2, coarse['NGAP']*2
        try:
            return abs(run(ncell, 0.0, lam).imag)/w
        finally:
            DX, NX, NY = coarse['DX'], coarse['NX'], coarse['NY']
            NGND, NGAP = coarse['NGND'], coarse['NGAP']

    print("\n-- is the knob CORRECT? (reference = same slab at DX/2) --")
    print("%-30s %11s %11s %9s" % ('slab', 'coarse', 'fine ref', 'error'))
    c3 = abs(run(3, 0.0, lam).imag)/w
    f3 = at_half(6)
    print("%-30s %11.5f %11.5f %8.2f%%   <- pitch confound floor"
          % ('3.00 cells (integer)', 1e12*c3, 1e12*f3, 100*(c3/f3 - 1)))
    c25 = abs(run(2, 0.5, lam).imag)/w
    f25 = at_half(5)
    print("%-30s %11.5f %11.5f %8.2f%%   <- the FILL claim"
          % ('2.50 cells (fill 0.5)', 1e12*c25, 1e12*f25,
             100*(c25/f25 - 1)))
    c2 = abs(run(2, 0.0, lam).imag)/w
    print("%-30s %11.5f %11.5f %8.2f%%   <- what we do today"
          % ('2.50 -> staircase 2 cells', 1e12*c2, 1e12*f25,
             100*(c2/f25 - 1)))


if __name__ == '__main__':
    report('NORMAL METAL (sigma = 5.8e7)')
    truth()
    report('LONDON SUPERCONDUCTOR (lambda = 90 nm)', lam=9e-8)
    truth(lam=9e-8)
