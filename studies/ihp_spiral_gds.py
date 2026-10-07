# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""A large single-ended octagonal spiral inductor on the IHP SG13G2
top metals, written as GDS through the IHP gdsfactory PDK.

Runs in the PDK's own environment (ihp-gdsfactory needs Python < 3.14):

    ~/.venvs/ihp/bin/python studies/ihp_spiral_gds.py --out spiral.gds

The PDK's ``inductor2``/``inductor3`` cells are built for 1-3 turns: at
ten turns they draw closed rings joined across the bottom, not a
spiral. This draws the standard industry topology instead, on the
PDK's layers: the coil in TopMetal2, the inner end brought out by a
TopMetal1 underpass with TopVia2 arrays at both ends, and two parallel
TopMetal2 leads whose ends carry the port (TopMetal2pin squares; the
right one is P -- the outer end -- and the left one N -- the
underpass).

Geometry: side j of the polyline (j = 0 .. 8N) is the line n_j . x =
a_j with outward normal at -90 + 45 j degrees and apothem a_j = a0 +
(w + s) j / 8, so every side stays parallel to the octagon's and
consecutive turns sit one pitch (w + s) apart on every side. Corners
are mitred exactly (offset lines intersected).
"""
import argparse
import json
import math

import numpy as np


def centreline(n_turns, w, s, d_in, x_end, x_start=None):
    """Polyline of the coil centre (um): inner start -> outer end."""
    p = w + s
    a0 = d_in/2 + w/2
    nrm = [np.array([math.cos(-math.pi/2 + j*math.pi/4),
                     math.sin(-math.pi/2 + j*math.pi/4)])
           for j in range(8*n_turns + 1)]
    ap = [a0 + p*j/8.0 for j in range(8*n_turns + 1)]
    pts = [np.array([-w/2 if x_start is None else x_start, -a0])]
    for j in range(8*n_turns):
        A = np.vstack([nrm[j], nrm[j + 1]])
        pts.append(np.linalg.solve(A, [ap[j], ap[j + 1]]))
    pts.append(np.array([x_end + w/2, -ap[-1]]))
    return np.array(pts), a0, ap[-1]


def outline(pts, w):
    """Mitred outline of a polyline of width w (closed polygon)."""
    hw = w/2
    seg = np.diff(pts, axis=0)
    seg /= np.linalg.norm(seg, axis=1)[:, None]
    nl = np.column_stack([-seg[:, 1], seg[:, 0]])     # left normal
    left, right = [], []
    for side, out in ((+1, left), (-1, right)):
        out.append(pts[0] + side*hw*nl[0])
        for k in range(1, len(pts) - 1):
            # intersection of the two offset lines at vertex k
            p1 = pts[k] + side*hw*nl[k - 1]
            p2 = pts[k] + side*hw*nl[k]
            A = np.column_stack([seg[k - 1], -seg[k]])
            t = np.linalg.solve(A, p2 - p1)
            out.append(p1 + t[0]*seg[k - 1])
        out.append(pts[-1] + side*hw*nl[-1])
    return np.vstack([left, right[::-1]])


def via_array(c, layer, x0, y0, x1, y1, size=0.9, space=1.06, enc=0.5):
    """TopVia2 squares filling [x0,x1] x [y0,y1] inside an enclosure."""
    pitch = size + space
    nx = int((x1 - x0 - 2*enc + space)//pitch)
    ny = int((y1 - y0 - 2*enc + space)//pitch)
    ox = x0 + (x1 - x0 - (nx*pitch - space))/2
    oy = y0 + (y1 - y0 - (ny*pitch - space))/2
    for i in range(nx):
        for j in range(ny):
            xa, ya = ox + i*pitch, oy + j*pitch
            c.add_polygon([(xa, ya), (xa + size, ya), (xa + size, ya + size),
                           (xa, ya + size)], layer=layer)
    return nx*ny, (nx*ny*size*size)/((x1 - x0)*(y1 - y0))


def _write_ring_sidecar(a, PDK, path, pts, a_out, rects, fills, ring, xe,
                        y_port):
    """The spiral + guard ring in the GENERIC sidecar format (the one
    studies/ihp_stacked_transformer_gds.py writes): the model's own z
    stack -- Metal5 below TopMetal1 -- the coil as a trace with its
    blocks, and the ring as extra blocks. Model stack (um): M5 [0, 0.5)
    (PDK 0.49), TM1 [1.5, 3.5), TV2 [3.5, 6.5) (2.8 -> 3.0), TM2
    [6.5, 9.5); the TopVia1 gap [0.5, 1.5) is empty."""
    L = PDK.layers
    w = a.width
    st = PDK.layer_stack.layers
    blocks = [dict(name='underpass', layer='TopMetal1',
                   rect=rects['underpass'], via=False, fill=1.0),
              dict(name='via_in', layer='TopVia2', rect=rects['via_in'],
                   via=True, fill=float(fills[0])),
              dict(name='via_land', layer='TopVia2', rect=rects['via_land'],
                   via=True, fill=float(fills[1])),
              dict(name='lead_n', layer='TopMetal2', rect=rects['lead_n'],
                   via=False, fill=1.0)]
    side = dict(
        kind='spiral_ring', topology='generic', ring=a.ring,
        title="IHP SG13G2 octagonal spiral inductor inside a %s Metal5 "
              "guard ring" % a.ring.upper(),
        notes=["%d turns, width %.1f um, space %.1f um, inner diameter %.0f "
               "um, outer %.0f um flat to flat" % (a.turns, w, a.space,
                                                   a.d_in, 2*(a_out + w/2)),
               "square Metal5 guard ring %.0f um wide, %.0f um outside the "
               "coil%s" % (a.ring_width, a.ring_space,
                           ('; slotted (%.0f um) in its top side, so it '
                            'carries no circulating current' % a.ring_slot)
                           if a.ring == 'slotted' else
                           ('; CLOSED -- a shorted turn around the coil '
                            '(the conductor has a hole: b1 = 1)'
                            if a.ring == 'closed' else ''))],
        turns=a.turns, width_um=w, space_um=a.space, d_in_um=a.d_in,
        outer_flat_to_flat_um=2*(a_out + w/2),
        winding_length_um=float(np.linalg.norm(np.diff(pts, axis=0),
                                               axis=1).sum()),
        coil_width_um=w,
        windings=[dict(name='coil', trace_layer='TopMetal2',
                       coil_path_um=path.tolist(), blocks=blocks,
                       port=dict(P=[xe, y_port, w], N=[0.0, y_port, a.wu],
                                 face='-y'))],
        extra_blocks=[dict(name=k, layer='Metal5', rect=r, via=False,
                           fill=1.0) for k, r in ring],
        model_stack_um=dict(Metal5=[0.0, 0.5], TopMetal1=[1.5, 3.5],
                            TopVia2=[3.5, 6.5], TopMetal2=[6.5, 9.5]),
        layers={k: [v.layer, v.datatype] for k, v in dict(
            TopMetal2=L.TopMetal2drawing, TopMetal1=L.TopMetal1drawing,
            TopVia2=L.TopVia2drawing, Metal5=L.Metal5drawing,
            TopMetal2pin=L.TopMetal2pin).items()},
        stack_um={k: [float(st[k].zmin), float(st[k].thickness)]
                  for k in ('metal5', 'topmetal1', 'topvia2', 'topmetal2')})
    with open(a.out.rsplit('.', 1)[0] + '.json', 'w') as f:
        json.dump(side, f, indent=1)
    print(json.dumps({k: side[k] for k in ('outer_flat_to_flat_um', 'ring',
                                           'winding_length_um')}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--turns', type=int, default=10)
    ap.add_argument('--width', type=float, default=15.0)
    ap.add_argument('--space', type=float, default=3.0)
    ap.add_argument('--d-in', type=float, default=240.0,
                    help='inner diameter (flat to flat), um')
    ap.add_argument('--x-end', type=float, default=45.0,
                    help='x of the outer lead centre, um')
    ap.add_argument('--gap', type=float, default=15.0,
                    help='underpass landing clearance below the coil, um')
    ap.add_argument('--lead', type=float, default=30.0,
                    help='lead length past the landing, um')
    ap.add_argument('--wu', type=float, default=16.0,
                    help='underpass / via landing / inner lead width, um '
                         '(even, so their edges sit on a 1 um grid)')
    ap.add_argument('--ring', choices=('none', 'closed', 'slotted'),
                    default='none',
                    help='a square Metal5 guard ring around the coil: '
                         'closed (a shorted turn, b1 = 1) or slotted')
    ap.add_argument('--ring-width', type=float, default=6.0)
    ap.add_argument('--ring-space', type=float, default=15.0,
                    help='coil outer edge to ring inner edge, um')
    ap.add_argument('--ring-slot', type=float, default=4.0,
                    help='slot width in the ring\'s top side, um')
    ap.add_argument('--out', default='ihp_spiral.gds')
    a = ap.parse_args(argv)

    import gdsfactory as gf
    from ihp import PDK
    PDK.activate()
    L = PDK.layers
    TM2, TM1, TV2 = L.TopMetal2drawing, L.TopMetal1drawing, L.TopVia2drawing
    TM2PIN = L.TopMetal2pin
    M5 = L.Metal5drawing
    w = a.width

    xe = a.x_end
    wu = a.wu
    pts, a0, a_out = centreline(a.turns, w, a.space, a.d_in, xe,
                                x_start=-wu/2)
    y_in = -a0
    y_land = -(a_out + w/2 + a.gap + w/2)
    y_port = y_land - w/2 - a.lead
    # the coil's trace path: the spiral, then down the outer (P) lead
    path = np.vstack([pts[:-1], [[xe, -a_out], [xe, y_port]]])
    c = gf.Component('ihp_spiral_%dt' % a.turns)
    c.add_polygon([tuple(p) for p in outline(path, w)], layer=TM2)

    # everything else is axis-aligned on a 1 um grid (commensurate with
    # every ladder pitch): inner via landing, TopMetal1 underpass, outer
    # via landing, N lead
    rects = dict(via_in=[-wu/2, y_in - w/2, wu/2, y_in + w/2],
                 underpass=[-wu/2, y_land - w/2, wu/2, y_in + w/2],
                 via_land=[-wu/2, y_land - w/2, wu/2, y_land + w/2],
                 lead_n=[-wu/2, y_port, wu/2, y_land + w/2])
    for k, r in rects.items():
        if any(abs(v - round(v)) > 1e-9 for v in r):
            raise SystemExit("%s %s is off the 1 um grid" % (k, r))
    n1, f1 = via_array(c, TV2, *rects['via_in'])
    n2, f2 = via_array(c, TV2, *rects['via_land'])
    for k, lay in (('underpass', TM1), ('lead_n', TM2)):
        x0, y0, x1, y1 = rects[k]
        c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=lay)
    for x, hw in ((0.0, wu/2), (xe, w/2)):
        c.add_polygon([(x - hw, y_port), (x + hw, y_port),
                       (x + hw, y_port + 1.0), (x - hw, y_port + 1.0)],
                      layer=TM2PIN)
    ring = []
    if a.ring != 'none':
        # square guard ring on Metal5 around the coil's outer edge (the
        # coil's leads and underpass cross it on higher layers, without
        # contact); the slot is in the top side, away from the leads
        ri = (a_out + w/2) + a.ring_space
        ro = ri + a.ring_width
        ring = [('ring_bottom', [-ro, -ro, ro, -ri]),
                ('ring_left', [-ro, -ri, -ri, ri]),
                ('ring_right', [ri, -ri, ro, ri])]
        if a.ring == 'closed':
            ring.append(('ring_top', [-ro, ri, ro, ro]))
        else:
            g = a.ring_slot/2
            ring += [('ring_top_l', [-ro, ri, -g, ro]),
                     ('ring_top_r', [g, ri, ro, ro])]
        for k, r in ring:
            if any(abs(v - round(v)) > 1e-9 for v in r):
                raise SystemExit("%s %s is off the 1 um grid" % (k, r))
            x0, y0, x1, y1 = r
            c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=M5)
    c.write_gds(a.out)
    if ring:
        _write_ring_sidecar(a, PDK, path, pts, a_out, rects, (f1, f2), ring,
                            xe, y_port)
        return

    # the record the converter reads beside the GDS: layer numbers and
    # the PDK's BEOL stack, so the octree side needs no PDK import
    st = PDK.layer_stack.layers
    side = dict(
        turns=a.turns, width_um=w, space_um=a.space, d_in_um=a.d_in,
        outer_flat_to_flat_um=2*(a_out + w/2),
        length_um=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
        coil_path_um=path.tolist(), coil_width_um=w, rects_um=rects,
        vias=[n1, n2], via_fill=[f1, f2],
        layers=dict(TopMetal2=[TM2.layer, TM2.datatype], TopMetal1=[TM1.layer, TM1.datatype],
                    TopVia2=[TV2.layer, TV2.datatype],
                    TopMetal2pin=[TM2PIN.layer, TM2PIN.datatype]),
        stack_um={k: [float(st[k].zmin), float(st[k].thickness)]
                  for k in ('topmetal1', 'topvia2', 'topmetal2')},
        ports=dict(P=[xe, y_port, w], N=[0.0, y_port, wu]))
    with open(a.out.rsplit('.', 1)[0] + '.json', 'w') as f:
        json.dump(side, f, indent=1)
    print(json.dumps({k: side[k] for k in ('outer_flat_to_flat_um',
                                           'length_um', 'vias', 'via_fill',
                                           'stack_um')}))


if __name__ == '__main__':
    main()
