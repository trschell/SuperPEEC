# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""GDSII (MIT-LL SFQ5ee layout) -> SuperPEEC doctrine TOML.

Turns an RSFQlib cell (github.com/sunmagnetics/RSFQlib, GPL-3; the
ColdFlux cell library for the MIT Lincoln Laboratory SFQ5ee process)
into a voxel model: every drawn polygon is rasterised at the requested
pitch on the layer's z-range from the published SFQ5ee stack (Tolpygo
et al., arXiv:2112.08457 Fig. 1: all Nb layers 200 nm except M5 at
135 nm, all SiO2 spacers 200 nm except I5 at 280 nm; London depth
90 nm), and the InductEx port labels the layout carries on GDS layer
182 ("NAME LAYER_POS LAYER_NEG" at a point) become doctrine ports.

The GDS is read at run time from the RSFQlib checkout -- nothing from
the GPL library is copied into this tree; the emitted TOML names its
source.

Junctions (J5 + C5J at a junction footprint) are geometry only: the
2 nm AlOx barrier is not a voxel object. Each junction, identified by
its "Jn M6 M5" label, is either SHORTED (the counter-electrode stack
is drawn, M5 and M6 galvanically joined -- the junction carries
current as InductEx's port would) or OPEN (the stack is omitted).
Combining short/open patterns with a driven edge port gives loop
inductances that are exact sums of the InductEx partial inductances
of the back-annotated netlist, e.g. on the JTL: drive P1 with J1
shorted and J2 open -> L1 + LP1. Resistors (R5 + C5R) are omitted by
default: InductEx does not extract their geometry either (the
extracted netlist computes LRB analytically from squares).

Usage:
  python3 studies/rsfq_gds2toml.py GDS --pitch 100e-9 --out cell.toml \
      [--drive P1] [--short J1] [--open J2] [--freq 1e10] \
      [--resistors omit|metal] [--margin 1e-6] [--pz 100e-9]
"""
import argparse
import collections
import os
import sys

import numpy as np

try:
    import gdstk
except ImportError:  # pragma: no cover
    sys.exit("gdstk is required (pip install gdstk)")

LAMBDA_NB = 90e-9          # London depth used by the SFQ5ee design kit
SIGMA_MO = 1.0/(2.0*40e-9)  # R5: 2 ohm/sq Mo at 40 nm

# SFQ5ee stack, nm, bottom to top (Tolpygo et al. 2021 Fig. 1). The
# junction counter-electrode (J5) thickness and the resistor's position
# inside the I5 spacer are nominal: the public figure gives the spacer
# total (280 nm) but not their split. z0/z1 of vias span the spacer
# they fill.
_STACK = [
    # name, gds layer, kind, z0, z1
    ('M0',  1,  'metal',    0,    200),
    ('I0',  2,  'via',      200,  400),
    ('M1',  10, 'metal',    400,  600),
    ('I1',  11, 'via',      600,  800),
    ('M2',  20, 'metal',    800,  1000),
    ('I2',  21, 'via',      1000, 1200),
    ('M3',  30, 'metal',    1200, 1400),
    ('I3',  31, 'via',      1400, 1600),
    ('M4',  40, 'metal',    1600, 1800),
    ('I4',  41, 'via',      1800, 2000),
    ('M5',  50, 'metal',    2000, 2135),
    ('J5',  51, 'junction', 2135, 2285),
    ('R5',  52, 'resistor', 2195, 2235),
    ('I5',  54, 'via',      2135, 2415),
    ('C5J', 55, 'junction', 2285, 2415),
    ('C5R', 56, 'resistor', 2235, 2415),
    ('M6',  60, 'metal',    2415, 2615),
    ('I6',  61, 'via',      2615, 2815),
    ('M7',  70, 'metal',    2815, 3015),
]
STACK = {s[0]: s for s in _STACK}
BY_GDS = {s[1]: s for s in _STACK}
LABEL_LAYER = 182

# HOW CLOSE A PORT LABEL MUST SIT TO THE LAYOUT BOUNDARY to count as an
# edge port, and how far inward to scan for that layer's first metal.
# PHYSICAL, in metres, and that is the whole point: both were once
# counted in CELLS (1.5 and 4), which is a shrinking physical distance
# as the pitch falls, so a converter that worked at 100 nm rejected the
# same port at 30 nm. The offset it was tripping over is fixed and
# geometric -- marker polygons (IXPORT and friends) push the GDS bbox
# 0.05 um past the drawn metal, so a port drawn exactly on the layout
# edge sits 0.05 um inside the bbox at EVERY pitch: 0.50 cells at
# 100 nm, 1.48 at 33.75, 1.67 at 30 -- which is where it started
# failing.
#
# 0.5 um separates the two populations by a wide margin on the RSFQlib
# cells: every signal port (P1..P4) sits 0.050 um inside the bbox and
# the nearest BIAS port (PB4) 1.700 um, so the window is 10x above the
# marker overhang and 3.4x below the first interior port.
EDGE_TOL_M = 0.5e-6
EDGE_SCAN_M = 0.5e-6


def _read(path):
    lib = gdstk.read_gds(path)
    tops = lib.top_level()
    if len(tops) != 1:
        raise SystemExit("expected one top cell, found %s"
                         % [c.name for c in tops])
    top = tops[0]
    polys = top.get_polygons(apply_repetitions=True, include_paths=True,
                             depth=None)
    labels = top.get_labels(apply_repetitions=True, depth=None)
    return lib, top, polys, labels


def _raster(polys_xy, origin, pitch, dims):
    """Occupancy (nx, ny) of a layer: a cell is filled when its centre
    lies inside any polygon of the layer (per-polygon bbox cropping
    keeps the point-in-polygon tests local)."""
    occ = np.zeros(dims, dtype=bool)
    for p in polys_xy:
        (x0, y0), (x1, y1) = p.bounding_box()
        i0 = max(int(np.floor((x0 - origin[0])/pitch[0] - 0.5)), 0)
        i1 = min(int(np.ceil((x1 - origin[0])/pitch[0] + 0.5)), dims[0])
        j0 = max(int(np.floor((y0 - origin[1])/pitch[1] - 0.5)), 0)
        j1 = min(int(np.ceil((y1 - origin[1])/pitch[1] + 0.5)), dims[1])
        if i1 <= i0 or j1 <= j0:
            continue
        xs = origin[0] + (np.arange(i0, i1) + 0.5)*pitch[0]
        ys = origin[1] + (np.arange(j0, j1) + 0.5)*pitch[1]
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = np.fromiter(gdstk.inside(pts, [p]), dtype=bool,
                             count=len(pts))
        occ[i0:i1, j0:j1] |= inside.reshape(i1 - i0, j1 - j0)
    return occ


def _boxes(occ):
    """Greedy rectangle cover of a 2-D occupancy: x-runs per row,
    merged over consecutive rows with identical extents."""
    out = []
    open_runs = {}      # (x0, x1) -> y0 of a run still growing
    for j in range(occ.shape[1] + 1):
        runs = set()
        if j < occ.shape[1]:
            row = occ[:, j]
            d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)
            runs = set(zip(starts.tolist(), ends.tolist()))
        for r in list(open_runs):
            if r not in runs:
                out.append((r[0], open_runs.pop(r), r[1], j))
        for r in runs:
            open_runs.setdefault(r, j)
    return out


def _zcells(z0_nm, z1_nm, pz):
    c0 = int(round(z0_nm*1e-9/pz))
    c1 = int(round(z1_nm*1e-9/pz))
    return c0, max(c1, c0 + 1)


def _parse_ports(labels, unit):
    """InductEx port labels: 'NAME LPOS LNEG' on layer 182; positions
    returned in metres."""
    ports = {}
    for l in labels:
        if l.layer != LABEL_LAYER:
            continue
        parts = l.text.split()
        if len(parts) == 3 and parts[1] in STACK and parts[2] in STACK:
            ports[parts[0]] = (parts[1], parts[2],
                               float(l.origin[0])*unit,
                               float(l.origin[1])*unit)
    return ports


def _run_containing(mask1d, j):
    """[j0, j1) of the True run of mask1d containing index j (or the
    nearest run when j itself is False)."""
    if not mask1d.any():
        return None
    idx = np.flatnonzero(mask1d)
    if not mask1d[j]:
        j = idx[np.argmin(np.abs(idx - j))]
    j0 = j
    while j0 > 0 and mask1d[j0 - 1]:
        j0 -= 1
    j1 = j + 1
    while j1 < mask1d.size and mask1d[j1]:
        j1 += 1
    return j0, j1


def _edge_port(name, spec, occ, origin, pitch, dims, zc,
               gnd_strip='local'):
    """Edge port: the label sits on the layout boundary; the terminal
    faces are the boundary-facing faces of the LPOS trace's cells in
    the boundary column (the run containing the label) and of the LNEG
    plane's cells directly beneath the same run."""
    lpos, lneg, x, y = spec
    ix = (x - origin[0])/pitch[0]
    iy = (y - origin[1])/pitch[1]
    nx, ny = dims
    # which boundary, if any -- ranked in METRES, so the answer does not
    # change with the pitch (see EDGE_TOL_M)
    cand = []
    for axis, val, lo_face, hi_face, extent in (
            (0, ix, '-x', '+x', nx), (1, iy, '-y', '+y', ny)):
        cand.append((abs(val - _EDGE[axis][0])*pitch[axis], axis,
                     _EDGE[axis][0], lo_face))
        cand.append((abs(val - _EDGE[axis][1])*pitch[axis], axis,
                     _EDGE[axis][1] - 1, hi_face))
    dist, axis, col, face = min(cand)
    if dist > EDGE_TOL_M:
        return None
    other = 1 - axis
    along = int(np.floor(iy if axis == 0 else ix))
    along = min(max(along, 0), dims[other] - 1)

    def faces(layer, run_ref=None):
        # marker polygons (IXPORT etc.) can push the bbox past the
        # metal: scan inward from the edge to the layer's first
        # occupied column on the label's row
        o = occ[layer]
        step = 1 if face[0] == '-' else -1
        nscan = max(4, int(np.ceil(EDGE_SCAN_M/pitch[axis])))
        for c in range(col, col + nscan*step, step):
            line = o[c, :] if axis == 0 else o[:, c]
            if line[along]:
                break
        else:
            raise SystemExit("port %s: no %s metal within %.2f um (%d "
                             "cells) of the %s edge at the label row"
                             % (name, layer, EDGE_SCAN_M*1e6, nscan,
                                face))
        run = _run_containing(line, along)
        if run_ref is not None:
            # the reference plane's terminal is the strip under the
            # driven trace (InductEx references a port to the plane
            # at the label), not the plane's whole edge
            run = (max(run[0], run_ref[0]), min(run[1], run_ref[1]))
        z0, z1 = zc[layer]
        out = []
        for k in range(run[0], run[1]):
            for z in range(z0, z1):
                cell = (c, k) if axis == 0 else (k, c)
                out.append([cell[0], cell[1], z, face])
        return out, run
    pf, run = faces(lpos)
    nf, _ = faces(lneg, run if gnd_strip == 'local' else None)
    return pf, nf


_EDGE = {}


def convert(args):
    lib, top, polys, labels = _read(args.gds)
    unit = lib.unit                       # metres per GDS user unit
    bylayer = collections.defaultdict(list)
    unknown = collections.Counter()
    for p in polys:
        if p.layer in BY_GDS and p.datatype == 0:
            bylayer[BY_GDS[p.layer][0]].append(p)
        else:
            unknown[(p.layer, p.datatype)] += 1
    (bx0, by0), (bx1, by1) = top.bounding_box()
    pitch = (args.pitch, args.pitch, args.pz or args.pitch)
    # world origin: bbox min minus the margin, snapped so the layout
    # bbox corners land on cell boundaries
    mrg = int(np.ceil(args.margin/args.pitch))
    origin = ((bx0*unit) - mrg*pitch[0], (by0*unit) - mrg*pitch[1])
    nx = int(np.ceil((bx1 - bx0)*unit/pitch[0])) + 2*mrg
    ny = int(np.ceil((by1 - by0)*unit/pitch[1])) + 2*mrg
    _EDGE[0] = (mrg, nx - mrg)
    _EDGE[1] = (mrg, ny - mrg)
    ztop = max(s[4] for s in _STACK)
    nz = _zcells(0, ztop, pitch[2])[1] + 2*args.zmargin

    ports = _parse_ports(labels, unit)
    junctions = {n: s for n, s in ports.items() if n.startswith('J')}
    short = set(args.short.split(',')) - {''}
    opn = set(args.open.split(',')) - {''}
    missing = (short | opn) - set(junctions)
    if missing:
        raise SystemExit("unknown junction(s) %s; layout has %s"
                         % (sorted(missing), sorted(junctions)))
    default_j = 'short' if args.junctions == 'short' else 'open'
    jstate = {n: ('short' if n in short else 'open' if n in opn
                  else default_j) for n in junctions}

    # scale polygons to metres once (gdstk works in user units)
    def to_m(p):
        return gdstk.Polygon(np.asarray(p.points)*unit, p.layer,
                             p.datatype)
    occ, zc, mat, zm = {}, {}, {}, {}
    report = []
    j5_polys = [to_m(p) for p in bylayer.get('J5', [])]
    for name, gds, kind, z0, z1 in _STACK:
        if kind == 'resistor' and args.resistors == 'omit':
            continue
        pm = [to_m(p) for p in bylayer.get(name, [])]
        if kind == 'junction':
            # keep only the junction footprints that are SHORTED
            keep = []
            for p in pm:
                st = None
                for jn, (lp, ln, x, y) in junctions.items():
                    if gdstk.inside([(x, y)], [p])[0] or \
                       _near(p, x, y, 0.5*pitch[0]):
                        st = jstate[jn]
                        break
                if st is None:
                    # FakeJJ / unlabeled junction-layer shapes: drawn
                    # as-is (they are vias in disguise)
                    keep.append(p)
                elif st == 'short':
                    keep.append(p)
            pm = keep
        o = _raster(pm, origin, pitch, (nx, ny))
        c0, c1 = _zcells(z0, z1, pitch[2])
        c0 += args.zmargin
        c1 += args.zmargin
        occ[name] = o
        zc[name] = (c0, c1)
        # the layer's TRUE z span in metres, offset by the z margin --
        # what --subpixel emits instead of the snapped cell pair, so a
        # 135 nm M5 stays 135 nm at any pitch
        zm[name] = (z0*1e-9 + args.zmargin*pitch[2],
                    z1*1e-9 + args.zmargin*pitch[2])
        mat[name] = ('mo' if kind == 'resistor' else 'nb')
        report.append((name, gds, kind, z0, z1, c0 - args.zmargin,
                       c1 - args.zmargin, int(o.sum()), len(pm)))

    # blocks
    blocks = []
    ncell = 0
    for name in occ:
        c0, c1 = zc[name]
        zlo, zhi = zm[name]
        for (x0, y0, x1, y1) in _boxes(occ[name]):
            blocks.append((name, x0, y0, c0, x1, y1, c1, mat[name],
                           zlo, zhi))
            ncell += (x1 - x0)*(y1 - y0)*(c1 - c0)

    # ports. In --subpixel mode the metal REALLY spans the floor/ceil
    # cover of its physical z bounds -- the rim cells exist with
    # sigma_eff = sigma*fill, and the terminal machinery reads R per
    # face from the cell it sits in -- so the port must drive the rim
    # cells too, not just the round()-snapped core that zc holds.
    # Skipping a rim cell forces its current to enter laterally PAST
    # the reference plane (a port-model error of the gnd-strip class:
    # at pz = 67.5 nm the driven span covered 2.74 of 2.96 cells of
    # metal). The same floor/ceil arithmetic as sppeec_input._cover
    # keeps the two views of "which cells hold metal" identical; the
    # 1e-6-cell guards keep an exactly-on-boundary layer from growing
    # a zero-fill sliver either side. Non-subpixel ports keep zc.
    if args.subpixel:
        zp = {}
        for name, (zlo, zhi) in zm.items():
            c0 = int(np.floor(zlo/pitch[2] + 1e-6))
            c1 = int(np.ceil(zhi/pitch[2] - 1e-6))
            zp[name] = (c0, max(c1, c0 + 1))
    else:
        zp = zc
    drive = args.drive
    if drive not in ports:
        raise SystemExit("drive port %r not in layout ports %s"
                         % (drive, sorted(ports)))
    ep = _edge_port(drive, ports[drive], occ, origin, pitch, (nx, ny), zp,
                    args.gnd_strip)
    if ep is None:
        raise SystemExit("port %s at (%.2f, %.2f) um is not on the "
                         "layout edge; interior ports are not "
                         "supported yet" % (drive, ports[drive][2]*1e6,
                                            ports[drive][3]*1e6))
    pf, nf = ep

    # emit
    src = os.path.relpath(args.gds, os.path.expanduser('~'))
    L = []
    w = L.append
    w("# SPDX-License-Identifier: MIT")
    w("# Generated by studies/rsfq_gds2toml.py from ~/%s" % src)
    w("#   (RSFQlib, SUN Magnetics / Stellenbosch ColdFlux team, GPL-3;")
    w("#    the GDS is read at conversion time, not redistributed)")
    w("# Top cell %s, bbox %.2f x %.2f um, GDS unit %g m" %
      (top.name, (bx1 - bx0)*unit*1e6, (by1 - by0)*unit*1e6, unit))
    w("# Pitch %g x %g x %g m, margin %d cells (xy) / %d (z); world "
      "origin = layout bbox corner - margin" % (pitch + (mrg, args.zmargin)))
    w("# SFQ5ee stack (Tolpygo et al. 2021 Fig. 1), snapped to z cells:")
    w("#   layer gds kind      z0..z1 nm  -> cells   filled(xy) polys")
    for r in report:
        w("#   %-5s %3d %-9s %4d..%4d   %3d..%3d  %8d %5d" % r)
    if unknown:
        w("# Ignored GDS layers (not in the stack table): %s"
          % dict(sorted(unknown.items())))
    w("# Junctions: %s" % ", ".join("%s=%s" % kv for kv in sorted(jstate.items())))
    w("# Resistors (R5/C5R): %s" % args.resistors)
    w("# Ports in layout: %s; driven: %s (%s vs %s, %d + %d faces)"
      % (", ".join("%s(%s/%s)" % (n, s[0], s[1]) for n, s in sorted(ports.items())),
         drive, ports[drive][0], ports[drive][1], len(pf), len(nf)))
    w("# %d blocks, %d occupied cells of %d lattice (%.1f%%)"
      % (len(blocks), ncell, nx*ny*nz, 100.0*ncell/(nx*ny*nz)))
    w("")
    w("[grid]")
    w("dims  = [%d, %d, %d]" % (nx, ny, nz))
    if pitch[2] == pitch[0]:
        w("pitch = %g" % pitch[0])
    else:
        w("pitch = [%g, %g, %g]" % pitch)
    if args.subpixel:
        w("subpixel = true")
    w("")
    for (name, x0, y0, z0, x1, y1, z1, m, zlo, zhi) in blocks:
        w("[[block]]")
        w("name = \"%s\"" % name)
        if args.subpixel:
            # PHYSICAL z bounds, so the layer keeps its true thickness
            # at any pitch and [grid] subpixel handles the boundary
            # cells. x/y stay on the grid: the rasteriser works in whole
            # cells there, and subpixel v1 cuts ONE axis.
            w("from_m = [%.12g, %.12g, %.12g]"
              % (x0*pitch[0], y0*pitch[1], zlo))
            w("to_m   = [%.12g, %.12g, %.12g]"
              % (x1*pitch[0], y1*pitch[1], zhi))
        else:
            w("from = [%d, %d, %d]" % (x0, y0, z0))
            w("to   = [%d, %d, %d]" % (x1, y1, z1))
        if m == 'nb':
            w("lambda_l = %g" % LAMBDA_NB)
        else:
            w("sigma = %g" % SIGMA_MO)
        w("")
    w("[port]")
    w("name = \"%s\"" % drive)
    w("equipotential = true")
    w("p_faces = %s" % _faces_toml(pf))
    w("n_faces = %s" % _faces_toml(nf))
    w("")
    w("[solve]")
    w("freq = [%g]" % args.freq)
    if args.rtol:
        w("rtol = %g" % args.rtol)
    with open(args.out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s: %d blocks, %d cells (%d x %d x %d lattice, %.1f%%), "
          "port %s %d+%d faces" % (args.out, len(blocks), ncell, nx, ny, nz,
                                   100.0*ncell/(nx*ny*nz), drive,
                                   len(pf), len(nf)))
    return ncell


def _near(p, x, y, tol):
    (x0, y0), (x1, y1) = p.bounding_box()
    return x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol


def _faces_toml(faces):
    return "[" + ", ".join('[%d, %d, %d, "%s"]' % tuple(f) for f in faces) + "]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('gds')
    ap.add_argument('--pitch', type=float, required=True, help='m')
    ap.add_argument('--pz', type=float, default=None,
                    help='z pitch (default: --pitch)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--drive', default='P1')
    ap.add_argument('--short', default='', help='junctions to short, e.g. J1,J2')
    ap.add_argument('--open', default='', help='junctions to open')
    ap.add_argument('--junctions', choices=('short', 'open'), default='open',
                    help='state of junctions not named in --short/--open')
    ap.add_argument('--resistors', choices=('omit', 'metal'), default='omit')
    ap.add_argument('--gnd-strip', choices=('local', 'edge'), default='local',
                    help='reference-plane terminal: under the trace or '
                         'the whole plane edge')
    ap.add_argument('--freq', type=float, default=1e10)
    ap.add_argument('--rtol', type=float, default=None)
    ap.add_argument('--subpixel', action='store_true',
                    help='emit PHYSICAL z bounds and [grid] subpixel = '
                         'true, so a layer keeps its real thickness at '
                         'a pitch that does not divide it')
    ap.add_argument('--margin', type=float, default=1e-6,
                    help='empty margin around the layout, m (xy)')
    ap.add_argument('--zmargin', type=int, default=2,
                    help='empty cells below M0 and above M7')
    args = ap.parse_args(argv)
    convert(args)


if __name__ == '__main__':
    main()
