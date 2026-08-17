# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""pdn_planes: a parametric power-plane pair -- the canonical PDN case.

THE STRUCTURE (industrially standard, every SI/PI tool's demo): two
copper planes separated by a dielectric, stitched by a via field, fed
at a "VRM" site and drawn from at a "chip" site. The FILIGREE is not
decoration -- it is mandatory board reality, and it is exactly what
shows off a multiscale solver:

  * ANTIPADS: every via passing THROUGH a plane it does not connect to
    punches a clearance hole in it. A via field therefore perforates
    both planes with hundreds of small holes (fine detail, mils) inside
    a large extent (centimetres) -- the multiscale regime the FMM and
    the thin-axis 2-D tree exist for.
  * SLOT: an optional routing slot across the top plane (plane splits
    are how real boards break; slots reshape the return path and the
    cavity modes).
  * The via field alternates PLANE-STITCHING vias (connect both
    planes... none in a power/ground pair!) -- here the field is
    GROUND-return vias bonded to the bottom plane with antipads in the
    top, plus POWER vias bonded to the top plane with antipads in the
    bottom, checkerboarded: the real pattern under a BGA.

VARIANTS
  conductor-only (eps=None):  runs everywhere today (LpR/equiterminal)
                              -- the SIZE-LADDER performance vehicle.
  dielectric fill:            per-cell eps_r between the planes; the
                              capacitive/PDN-impedance physics. Full
                              LpPR at scale lands with dielectric
                              phase 2's completion; small configs
                              validate against closed forms via the
                              dense oracle already.

VALIDATION ANCHORS
  * static C of the filled pair: eps0*eps_r*A/d (plus perforation
    corrections -- report, not assert, at high filigree).
  * first cavity resonance f_mn = c/(2*sqrt(eps_r)) * sqrt((m/a)^2 +
    (n/b)^2) -- the plane-pair impedance peak every PDN paper quotes.
  * loop inductance VRM->chip at DC vs the plane-pair spreading-L
    estimates.

Usage:
    PYTHONPATH=src python3 studies/pdn_planes.py
Env: NPLANE (cells per plane side, default 320), LADDER=1 (run the
size ladder), EPS (dielectric eps_r, default 4.2; 0 = conductor-only),
SOLVE=1 (end-to-end LpR Z11 with an 8-via decap stitching ring),
VALIDATE=1 (oracle C on the 16x16 miniature), FREQ (solve frequency).

PORT MOVED 2026-08-08: the chip site now sits at the centre of a via
QUAD (the realistic BGA-bump location; the original placement grazed
an antipad rim -- asymmetric P/N faces and artificial current
crowding). Numbers recorded below predate the move: R shifts a few
percent (spreading term), C and L are position-insensitive at these
frequencies. Post-move anchors: 48^2 filled LpPR Z11(1e8) =
4.30e-4 - 817.322j (C_eff 1.947e-12); 320^2 filled multilevel
band-W Z11(1e8) = 6.70e-4 - 18.9322j (C 8.4066e-11), 250 matvecs.

RESULTS (2026-08-07, idle):
  Partition now takes the pancake escape (voxmodel fix: a COLLAPSED
  min-axis clamp -- leaf < 3 on a 4-thick board -- used to fall to a
  single-level tree; 320^2 board: 21 s single-level build -> 1.7 s,
  nleaf [5,5,5]/[8,8,5], z spanned, top grid 2-D). Size ladder,
  conductor-only, raw traverseRL:
    n= 320:   193652 cells  build  0.9 s  matvec  666 ms
    n= 640:   775304 cells  build  4.0 s  matvec 2433 ms
    n= 960:  1744848 cells  build  9.5 s  matvec 5233 ms
    n=1280:  3102608 cells  build 11.3 s  matvec 9058 ms
  (per-cell cost FALLS 3.4 -> 2.9 us/cell: the 2-D FMM amortizes.)
  With dielectric fill the 1280 board is ~6.4M total cells.
  SOLVE=1, n=320 (193668 cells, 186k loops): Z11(1e8 Hz) =
  7.55e-4 + 4.14e-2j ohm -> L = 65.9 pH chip->decap-ring loop
  (analytic plane-pair spreading ~55 pH + via barrels: consistent),
  311 matvecs, resid 6.4e-9, setup 667 s + solve 419 s.
  A port straddling the planes WITHOUT stitching is correctly
  REFUSED by the no-return-path guard (the galvanic loop does not
  exist; the return is displacement current = the LpPR path).
  VALIDATE=1 report-only anchor: C(oracle)/plate-formula = 1.296.
"""
import os
import time

import numpy as np

import voxmodel


def build_pdn(nplane=320, gap_layers=2, via_pitch=16, antipad_r=3,
              slot=True, eps_r=4.2, dx=1e-4/2, sigma=5.8e7, stitch=0):
    """Build the plane-pair model.

    nplane      cells per plane side (planes are nplane x nplane x 1)
    gap_layers  dielectric cells between the planes
    via_pitch   cells between vias (checkerboard power/ground field)
    antipad_r   clearance radius, cells, around a via in the plane it
                does NOT bond to
    slot        cut a routing slot across the top plane
    eps_r       dielectric fill; 0/None -> conductor-only variant
    """
    nz = 2 + gap_layers
    dims = (nplane, nplane, nz)
    sig = np.zeros(dims)
    sig[:, :, 0] = sigma                  # bottom (ground) plane
    sig[:, :, nz - 1] = sigma             # top (power) plane

    # via field: checkerboard of power (bond top, antipad bottom) and
    # ground (bond bottom, antipad top) vias
    rng = range(via_pitch//2, nplane - via_pitch//4, via_pitch)
    yy, xx = np.meshgrid(np.arange(nplane), np.arange(nplane),
                         indexing='ij')
    for i, vx in enumerate(rng):
        for j, vy in enumerate(rng):
            power = (i + j) % 2 == 0
            # the via barrel through the gap
            sig[vx, vy, 1:nz - 1] = sigma
            # antipad: clearance in the NON-bonded plane
            hole = (xx - vx)**2 + (yy - vy)**2 <= antipad_r**2
            if power:
                sig[:, :, 0][hole.T] = 0.0          # hole in ground
            else:
                sig[:, :, nz - 1][hole.T] = 0.0     # hole in power
    if slot:
        # routing slot: a 2-cell-wide cut across 60% of the top plane
        s0 = nplane//3
        sig[nplane//5:nplane - nplane//5, s0:s0 + 2, nz - 1] = 0.0
    if stitch:
        # decap-site stitching: full-height vias bonded to BOTH planes
        # in a ring around the chip site. Physically these are the
        # decoupling capacitors' mounting vias (shorted, i.e. the
        # above-series-resonance limit); they give the LpR path its
        # return, so the port loop is the chip->decap plane loop --
        # the number every PDN designer actually chases. Placed AFTER
        # the antipad pass: the column overwrite re-bonds anything an
        # antipad grazed.
        cc = nplane//2
        rr = max(3*via_pitch//2, 8)
        for k in range(stitch):
            ang = 2*np.pi*k/stitch
            sx = min(max(int(round(cc + rr*np.cos(ang))), 0), nplane - 1)
            sy = min(max(int(round(cc + rr*np.sin(ang))), 0), nplane - 1)
            sig[sx, sy, :] = sigma

    m = voxmodel.VoxelModel('pdn_planes_%d' % nplane)
    m.dims = dims
    m.d = dx
    m.sigma = sig
    if eps_r:
        eps = np.ones(dims)
        for z in range(1, nz - 1):
            # dielectric everywhere in the gap except via barrels
            eps[:, :, z] = np.where(sig[:, :, z] == 0.0, eps_r, 1.0)
        m.epsilon = eps
    m.freq = np.array([1e8])

    # port: PDN input impedance Z11 at the "chip" site -- P faces on
    # the power (top) plane, N faces on the ground (bottom) plane at
    # the SAME (x,y) footprint, the standard plane-pair port.
    p = voxmodel.Port('chip')
    # chip site at the centre of a via QUAD -- equidistant from the
    # four surrounding vias, the realistic BGA-bump location, and
    # maximally clear of antipad rims. (The first version aimed
    # "beside the centre via" assuming a via at nplane//2; the grid
    # has none there, and the cluster landed grazing an antipad --
    # asymmetric P/N footprints and port-local current crowding so
    # strong it flattened the colour scale of every field plot.)
    # Via x-positions are via_pitch//2 + k*via_pitch, so quad centres
    # are the multiples of via_pitch; take the one nearest mid-board.
    mid = int(round(nplane/2.0/via_pitch))*via_pitch
    c = min(max(mid - 1, 2), nplane - 4)
    for ox in range(3):
        for oy in range(3):
            if sig[c + ox, c + oy, nz - 1] != 0:
                p._add('P', (c + ox, c + oy, nz - 1, 2, 1))
            if sig[c + ox, c + oy, 0] != 0:
                p._add('N', (c + ox, c + oy, 0, 2, -1))
    p._freeze()
    m.ports = [p]
    return m


def stats(m):
    ids = m.material_struc()
    ncond = int((ids == 1).sum())
    ndiel = int((ids == 2).sum())
    return ncond, ndiel


def validate_small():
    """Oracle-grade C of a miniature pdn (4 vias, antipads, filled gap)
    vs the perforation-corrected plate formula. REPORT-ONLY while the
    dielectric phase-2 bound-charge treatment is open: the FILLED
    configuration is the physically-sane band (validator reads ~3.1 of
    ideal 4 on plain plates; the vacuum fringe stays vacuum)."""
    import sys
    sys.path.insert(0, 'src')
    from validate_dielectric import _capacitance
    m = build_pdn(nplane=16, via_pitch=8, antipad_r=2, slot=False,
                  eps_r=4.2)
    nc, nd = stats(m)
    print("validate: 16x16x4 pdn, %d conductor + %d dielectric cells"
          % (nc, nd))
    c = _capacitance(m, 1e8)
    eps0 = 1/(voxmodel.MU0*299792458.0**2)
    d = float(m.d[2])
    # overlap area = plane area minus the 4 antipad holes (each hole
    # kills the overlap under it once -- one hole per via, in one
    # plane or the other)
    ids = m.material_struc()
    a_top = float((ids[:, :, 3] == 1).sum())*d*d
    a_bot = float((ids[:, :, 0] == 1).sum())*d*d
    a_over = float(((ids[:, :, 3] == 1) & (ids[:, :, 0] == 1)).sum())*d*d
    ideal = eps0*4.2*a_over/(2*d)      # face-to-face gap = 2 cells
    print("  C(oracle) %.4g F   eps*A_overlap/d %.4g F   ratio %.3f"
          % (c, ideal, c/ideal))
    print("  (report-only anchor: measured 1.296, 2026-08-07. Two "
          "opposing effects -- fringing on a W/d=8 plate INFLATES "
          "(part-B window allows up to 2.5x), the open phase-2 "
          "bound-charge gap DEFLATES the dielectric's share. Re-check "
          "when phase 2 lands. Plate areas top %.3g bot %.3g overlap "
          "%.3g m^2)" % (a_top, a_bot, a_over))


def solve_inductive():
    """End-to-end LpR port solve on the conductor variant: the power
    and ground plane are SEPARATE components joined only through the
    port (the multi-conductor spanning-forest case) -- Z11 here is the
    plane-pair loop R + jwL seen at the chip site."""
    import equiterminal as eq
    nplane = int(os.environ.get('NPLANE', '320'))
    freq = float(os.environ.get('FREQ', '1e8'))
    m = build_pdn(nplane=nplane, eps_r=None, stitch=8)
    nc, _ = stats(m)
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, freq)
    t0 = time.perf_counter()
    S = eq.EquiTerminalSolver(m, M, 0, verbose=True)
    print("  solver setup %.1f s (%d cells)"
          % (time.perf_counter() - t0, nc), flush=True)
    z, i, info = S.solve(freq)
    w = 2*np.pi*freq
    print("  Z11(%.3g Hz) = %.6g + %.6gj ohm   R %.4g ohm  "
          "L %.4g H   (%d matvecs, resid %.1e, %.1f s)"
          % (freq, z.real, z.imag, z.real, z.imag/w,
             info['matvecs'], info['residual'], info['time']))


def main():
    if os.environ.get('VALIDATE', '0') == '1':
        validate_small()
        return
    if os.environ.get('SOLVE', '0') == '1':
        solve_inductive()
        return
    nplane = int(os.environ.get('NPLANE', '320'))
    eps = float(os.environ.get('EPS', '4.2'))
    m = build_pdn(nplane=nplane, eps_r=eps or None)
    nc, nd = stats(m)
    print("pdn_planes: dims %s  conductor %d  dielectric %d  total %d "
          "cells" % (m.dims, nc, nd, nc + nd), flush=True)
    t0 = time.perf_counter()
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    print("  tree: nleaf %s, %d levels, %.1f s build"
          % (list(int(v) for v in leaf), levels,
             time.perf_counter() - t0), flush=True)
    if os.environ.get('LADDER', '0') == '1':
        for npl in (320, 640, 960, 1280):
            mm = build_pdn(nplane=npl, eps_r=None)
            ncc, _ = stats(mm)
            t0 = time.perf_counter()
            lf, lv = mm.partition()
            MM = mm.build_tree(lf, lv)
            tb = time.perf_counter() - t0
            whole, (esz, efsz, efgsz, _) = mm.prepare(MM, 1e8)
            whole[:efgsz] = 1.0 + 0.0j
            t0 = time.perf_counter()
            MM.traverseRL()
            tm_ = time.perf_counter() - t0
            print("  LADDER n=%d: %8d cells  nleaf %s lv %d  build "
                  "%5.1f s  matvec %6.0f ms"
                  % (npl, ncc, list(int(v) for v in lf), lv, tb,
                     1e3*tm_), flush=True)
            del MM, mm, whole


if __name__ == '__main__':
    main()
