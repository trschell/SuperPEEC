# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""IHP SG13G2 spiral inductor GDS -> SuperPEEC doctrine TOML.

Reads the GDS that ``studies/ihp_spiral_gds.py`` writes through the IHP
gdsfactory PDK, plus its JSON sidecar (layer numbers, the PDK's BEOL
stack, the port positions), and rasterises the conductors on a CUBIC
lattice of the requested pitch:

  TopMetal1  [0, 2.0) um      (PDK: z 6.23, 2.0 um; origin moved here)
  TopVia2    [2.0, 5.0) um    (PDK: 2.8 um, ROUNDED to 3.0 so every
                               ladder pitch 1.0 / 0.5 / 0.25 um is
                               commensurate; TopMetal2 sits 0.2 um high)
  TopMetal2  [5.0, 8.0) um    (PDK: 3.0 um)

The coil (with its P lead) is one [[trace]] by default -- a section
cut: sub-cell fills, the face rule, the edge palette, so the 45-degree
sides carry the right skin current (a staircase is first order in
deep-skin R, docs/trace_plan.md); --coil blocks rasterises it instead
(cells filled where their centre is inside, merged into boxes) as the
reference arm. Everything else is axis-aligned on a 1 um grid and is a
block: the N lead, the TopMetal1 underpass, and each TopVia2 ARRAY as
one solid block over its landing with sigma_W scaled by the array's
metal fraction (the array's DC resistance kept; its 0.9 um squares are
not resolvable at the coarse rungs).

The port is the lead pair, EQUIPOTENTIAL terminals (a probe on two
pads): P on the -y faces of the outer lead's end row, N on the
underpass lead's, through the TopMetal2 thickness. (The box-terminal
wire-bond path needs at least one bond wire.)

No substrate, no oxide: the coil in free space (the inductance and the
conductor loss; substrate coupling is a later rung).

Usage:
  python3 studies/ihp_spiral_gds2toml.py spiral.gds --pitch 0.5e-6 \
      --out spiral.toml [--freq 1e9 5e9] [--margin 2e-6]
"""
import argparse
import json
import os
import sys

import numpy as np
import gdstk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsfq_gds2toml import _raster, _boxes          # noqa: E402

# nominal conductivities, S/m: AlCu top metals (~11 / 18 mOhm/sq at
# 3.0 / 2.0 um), tungsten vias
SIGMA_TM2 = 3.0e7
SIGMA_TM1 = 2.8e7
SIGMA_W = 1.8e7
Z_UM = dict(TopMetal1=(0.0, 2.0), TopVia2=(2.0, 5.0), TopMetal2=(5.0, 8.0))


def _cell(path):
    lib = gdstk.read_gds(path)
    tops = [c for c in lib.top_level() if not c.name.startswith('$$$')]
    if len(tops) != 1:
        sys.exit("expected one top cell, found %s" % [c.name for c in tops])
    return lib, tops[0]


def convert(a):
    side = json.load(open(a.gds.rsplit('.', 1)[0] + '.json'))
    lay = {k: tuple(v) for k, v in side['layers'].items()}
    lib, top = _cell(a.gds)
    um = lib.unit/1e-6                           # GDS user unit in um
    polys = {k: [gdstk.Polygon(p.points*um) for p in top.get_polygons()
                 if (p.layer, p.datatype) == v] for k, v in lay.items()}
    p_um = a.pitch*1e6
    allp = polys['TopMetal2'] + polys['TopMetal1']
    (bx0, by0), (bx1, by1) = gdstk.Polygon(
        np.vstack([q.points for q in allp])).bounding_box()
    m_um = a.margin*1e6
    # origin on the 1 um grid: the axis-aligned features (underpass,
    # landings, N lead) sit on it, so they are commensurate at 1.0 /
    # 0.5 / 0.25 um and the trace's lead end lies on a cell boundary
    origin = (float(np.floor(bx0 - m_um)), float(np.floor(by0 - m_um)))
    nx = int(np.ceil((bx1 + m_um - origin[0])/p_um))
    ny = int(np.ceil((by1 + m_um - origin[1])/p_um))
    zc = {k: (int(round(z0/p_um)), int(round(z1/p_um)))
          for k, (z0, z1) in Z_UM.items()}
    for k, (z0, z1) in Z_UM.items():
        if abs(z0/p_um - zc[k][0]) > 1e-9 or abs(z1/p_um - zc[k][1]) > 1e-9:
            sys.exit("pitch %g um does not divide the %s z range" % (p_um, k))
    nz = zc['TopMetal2'][1]

    def cells(r):
        """A 1 um-grid rectangle [x0, y0, x1, y1] (um) in cell indices."""
        v = [(r[0] - origin[0])/p_um, (r[1] - origin[1])/p_um,
             (r[2] - origin[0])/p_um, (r[3] - origin[1])/p_um]
        if any(abs(x - round(x)) > 1e-9 for x in v):
            sys.exit("rectangle %s is not commensurate with %g um" % (r, p_um))
        return [int(round(x)) for x in v]

    rects = side['rects_um']
    blocks, ncell = [], 0

    def rect_block(name, r, layer, sig):
        nonlocal ncell
        i0, j0, i1, j1 = cells(r)
        z0, z1 = zc[layer]
        ncell += (i1 - i0)*(j1 - j0)*(z1 - z0)
        blocks.append((name, (i0, j0, z0), (i1, j1, z1), sig))

    traces = []
    occ2 = _raster(polys['TopMetal2'], origin, (p_um, p_um), (nx, ny))
    if a.coil == 'blocks':
        # staircase: every TopMetal2 polygon (coil + its P lead, N lead)
        # rasterised at cell centres, merged into boxes
        ncell += int(occ2.sum())*(zc['TopMetal2'][1] - zc['TopMetal2'][0])
        for (i0, j0, i1, j1) in _boxes(occ2):
            blocks.append(('TopMetal2', (i0, j0, zc['TopMetal2'][0]),
                           (i1, j1, zc['TopMetal2'][1]), SIGMA_TM2))
    else:
        # the coil and its P lead as ONE section-cut trace (sub-cell
        # fills, face rule, edge palette); the N lead is a block
        path = np.asarray(side['coil_path_um'])
        path_m = ((path - np.asarray(origin))*1e-6).tolist()
        traces.append(dict(name='coil', path_m=path_m,
                           width_m=side['coil_width_um']*1e-6,
                           z_m=[Z_UM['TopMetal2'][0]*1e-6,
                                Z_UM['TopMetal2'][1]*1e-6],
                           sigma=SIGMA_TM2))
        # occupied count: the coil's cells as the staircase sees them
        # (the trace's own partial cells are within its rim)
        lead = np.zeros_like(occ2)
        i0, j0, i1, j1 = cells(rects['lead_n'])
        lead[i0:i1, j0:j1] = True
        ncell += int((occ2 & ~lead).sum())*(zc['TopMetal2'][1]
                                            - zc['TopMetal2'][0])
        rect_block('TopMetal2_lead_n', rects['lead_n'], 'TopMetal2',
                   SIGMA_TM2)
    rect_block('TopMetal1_underpass', rects['underpass'], 'TopMetal1',
               SIGMA_TM1)
    for k, key in enumerate(('via_in', 'via_land')):
        # the array as one block over its landing, sigma_W times the
        # array's metal fraction of that landing (DC resistance kept)
        rect_block('TopVia2_array%d' % k, rects[key], 'TopVia2',
                   SIGMA_W*side['via_fill'][k])

    def faces(x, y, w):
        """The lead's end: the -y faces of the first row's cells lying
        WHOLLY inside the lead (full cells: the equipotential terminal
        on a section cut wants fill 1), through the metal thickness."""
        j = int(round((y - origin[1])/p_um))
        xl = origin[0] + np.arange(nx)*p_um
        ii = np.flatnonzero((xl >= x - w/2 - 1e-9) & (xl + p_um <= x + w/2 + 1e-9))
        if ii.size == 0 or not occ2[ii, j].all() or occ2[ii, j - 1].any():
            sys.exit("lead end at (%g, %g) um is not a clean row" % (x, y))
        return [[int(i), j, k, "-y"] for i in ii
                for k in range(*zc['TopMetal2'])]

    P, N = side['ports']['P'], side['ports']['N']
    out = []
    out.append("# IHP SG13G2 octagonal spiral inductor -- SuperPEEC model")
    out.append("# generated by studies/ihp_spiral_gds2toml.py from %s "
               "(coil as %s)" % (os.path.basename(a.gds), a.coil))
    out.append("# (layout: studies/ihp_spiral_gds.py on the IHP gdsfactory "
               "PDK, ihp-gdsfactory)")
    out.append("#")
    out.append("# %d turns, width %.1f um, space %.1f um, inner diameter "
               "%.0f um, outer %.0f um flat to flat; coil centreline %.2f mm"
               % (side['turns'], side['width_um'], side['space_um'],
                  side['d_in_um'], side['outer_flat_to_flat_um'],
                  side['length_um']/1e3))
    out.append("# stack (PDK, um): TopMetal1 %.2f+%.1f, TopVia2 %.2f+%.1f, "
               "TopMetal2 %.2f+%.1f; modelled TM1 [0,2), TV2 [2,5) (2.8 -> "
               "3.0), TM2 [5,8)" % tuple(v for k in ('topmetal1', 'topvia2',
                                                      'topmetal2')
                                         for v in side['stack_um'][k]))
    out.append("# sigma (nominal, S/m): TM2 %.2g, TM1 %.2g, W %.2g x array "
               "fill (%s); no substrate/oxide"
               % (SIGMA_TM2, SIGMA_TM1, SIGMA_W,
                  ", ".join("%.3f" % f for f in side['via_fill'])))
    out.append("#")
    out.append("# ATTRIBUTION: layer numbers and the SG13G2 BEOL stack come from the")
    out.append("# IHP Open PDK through ihp-gdsfactory (Apache-2.0,")
    out.append("# https://github.com/gdsfactory/IHP); the spiral itself is generated")
    out.append("# by studies/ihp_spiral_gds.py (the PDK's inductor2/3 PCells are")
    out.append("# 1-3 turn cells). No PDK file is copied here.")
    out.append("#")
    out.append("# WHAT THIS MODEL IS (and is not): the conductors in free space --")
    out.append("# the coil's inductance and its conductor loss with skin and")
    out.append("# proximity effect. A real SG13G2 coil sits on ~6 um of oxide over a")
    out.append("# lossy silicon substrate, which sets its self-resonance and most of")
    out.append("# its Q loss above a few GHz; neither is represented here (no")
    out.append("# capacitance, no substrate). Use L and the conductor R; take Q and")
    out.append("# SRF from a substrate-aware solver.")
    out.append("#")
    out.append("# pitch %.3g um cubic: lattice %d x %d x %d = %.1f M, "
               "staircase estimate ~%.2f M occupied cells, %d blocks, %d traces"
               % (p_um, nx, ny, nz, nx*ny*nz/1e6, ncell/1e6, len(blocks),
                  len(traces)))
    out.append("")
    out.append("[grid]")
    out.append("dims  = [%d, %d, %d]" % (nx, ny, nz))
    out.append("pitch = [%.6g, %.6g, %.6g]" % ((a.pitch,)*3))
    out.append("")
    out.append("[port]")
    out.append("name  = \"P1\"")
    out.append("equipotential = true")
    for key, (x, y, w) in (('p_faces', P), ('n_faces', N)):
        fs = faces(x, y, w)
        out.append("%s = [%s]" % (key, ",\n  ".join(
            "[%d, %d, %d, \"%s\"]" % tuple(f) for f in fs)))
    out.append("")
    out.append("[solve]")
    out.append("freq  = [%s]" % ", ".join("%g" % f for f in a.freq))
    out.append("rtol  = %g" % a.rtol)
    if a.method:
        out.append("method = \"%s\"" % a.method)
    out.append("basis = \"overcomplete\"")
    out.append("")
    for tr in traces:
        out.append("[[trace]]")
        out.append("name    = \"%s\"" % tr['name'])
        out.append("width_m = %.9g" % tr['width_m'])
        out.append("z_m     = [%.9g, %.9g]" % tuple(tr['z_m']))
        out.append("sigma   = %.6g" % tr['sigma'])
        out.append("path_m  = [%s]" % ",\n  ".join(
            "[%.12g, %.12g]" % tuple(q) for q in tr['path_m']))
        out.append("")
    for name, lo, hi, sig in blocks:
        out.append("[[block]]")
        out.append("name  = \"%s\"" % name)
        out.append("from  = [%d, %d, %d]" % lo)
        out.append("to    = [%d, %d, %d]" % hi)
        out.append("sigma = %.6g" % sig)
        out.append("")
    with open(a.out, 'w') as f:
        f.write("\n".join(out))
    print("%s: lattice %d x %d x %d (%.1f M), occupied ~%.2f M cells, "
          "%d blocks, %d traces" % (a.out, nx, ny, nz, nx*ny*nz/1e6,
                                    ncell/1e6, len(blocks), len(traces)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('gds')
    ap.add_argument('--pitch', type=float, required=True, help='metres')
    ap.add_argument('--out', required=True)
    ap.add_argument('--freq', type=float, nargs='+', default=[1e9])
    ap.add_argument('--margin', type=float, default=2e-6)
    ap.add_argument('--rtol', type=float, default=1e-4)
    ap.add_argument('--method', default=None,
                    help='Krylov method, e.g. gmres_stream for the large '
                         'rungs (the basis parked on disk)')
    ap.add_argument('--coil', choices=('blocks', 'trace'), default='trace',
                    help='the coil as a [[trace]] (section cut, default) '
                         'or as a staircase of [[block]]s')
    convert(ap.parse_args(argv))


if __name__ == '__main__':
    main()
