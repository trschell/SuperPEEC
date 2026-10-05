# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""A 1:1 interwound octagonal transformer on the IHP SG13G2 top metals,
written as GDS through the IHP gdsfactory PDK.

Runs in the PDK's own environment (ihp-gdsfactory needs Python < 3.14):

    ~/.venvs/ihp/bin/python studies/ihp_transformer_gds.py --out xfmr.gds

TOPOLOGY. Two single-ended octagonal spirals share the TopMetal2 lanes
alternately -- the coplanar interwound monolithic transformer (J. R.
Long, "Monolithic transformers for silicon RF IC design", IEEE JSSC 35
(9), 2000; the Shibata / Frlan family). Each winding advances TWO
lanes per turn; the secondary is the primary rotated by 180 degrees,
which interleaves the two on every side (a winding's apothem on
absolute direction k at turn t is a0 + 2pt + pk/4, the rotated one
lands at that plus p modulo 2p). No crossover is needed between the
windings: each brings its inner end out under its own outer turns by a
straight TopMetal1 underpass with TopVia2 arrays at both ends -- the
primary at the bottom, the secondary at the top. Both ports are a lead
pair; the sidecar records each port's faces and the SHORTING BAR of
each lead pair (the open / short extraction of the 2-port, see
studies/ihp_spiral_gds2toml.py --short).

Geometry is shared with studies/ihp_spiral_gds.py (exact mitred
octagonal centrelines, everything but the coils on a 1 um grid).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ihp_spiral_gds import centreline, outline, via_array   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--turns', type=int, default=5,
                    help='turns PER WINDING')
    ap.add_argument('--width', type=float, default=15.0)
    ap.add_argument('--space', type=float, default=3.0)
    ap.add_argument('--d-in', type=float, default=240.0,
                    help='inner diameter (flat to flat), um')
    ap.add_argument('--x-end', type=float, default=45.0,
                    help='x of each outer lead centre, um')
    ap.add_argument('--gap', type=float, default=15.0,
                    help='underpass landing clearance outside the coil, um')
    ap.add_argument('--lead', type=float, default=30.0,
                    help='lead length past the landing, um')
    ap.add_argument('--wu', type=float, default=16.0,
                    help='underpass / landing / inner lead width, um')
    ap.add_argument('--out', default='ihp_transformer.gds')
    a = ap.parse_args(argv)

    import gdsfactory as gf
    from ihp import PDK
    PDK.activate()
    L = PDK.layers
    TM2, TM1, TV2 = L.TopMetal2drawing, L.TopMetal1drawing, L.TopVia2drawing
    TM2PIN = L.TopMetal2pin
    w, s, wu, xe = a.width, a.space, a.wu, a.x_end
    p = w + s
    # each winding steps two lanes per turn: its own space is w + 2s
    pts, a0, a_out = centreline(a.turns, w, w + 2*s, a.d_in, xe,
                                x_start=-wu/2)
    y_in = -a0
    y_land = -(a_out + w/2 + a.gap + w/2)
    y_port = y_land - w/2 - a.lead
    path = np.vstack([pts[:-1], [[xe, -a_out], [xe, y_port]]])
    # the primary's own rectangles (1 um grid), and its port and short
    rects = dict(via_in=[-wu/2, y_in - w/2, wu/2, y_in + w/2],
                 underpass=[-wu/2, y_land - w/2, wu/2, y_in + w/2],
                 via_land=[-wu/2, y_land - w/2, wu/2, y_land + w/2],
                 lead_n=[-wu/2, y_port, wu/2, y_land + w/2])
    # shorting bar across the lead pair's ends: from lead_n's right edge
    # into the P lead up to its centreline (the overlap is the same
    # metal, and xe is on the 1 um grid where its edge is not), one
    # landing-width deep from the port end
    short = [wu/2, y_port, xe, y_port + wu]
    for k, r in list(rects.items()) + [('short', short)]:
        if any(abs(v - round(v)) > 1e-9 for v in r):
            raise SystemExit("%s %s is off the 1 um grid" % (k, r))

    def rot(xy):
        return -np.asarray(xy, dtype=float)

    def rot_rect(r):
        return [-r[2], -r[3], -r[0], -r[1]]

    c = gf.Component('ihp_xfmr_%dt' % a.turns)
    windings = []
    for k, name in enumerate(('primary', 'secondary')):
        P = path if k == 0 else rot(path)
        R = rects if k == 0 else {q: rot_rect(r) for q, r in rects.items()}
        S = short if k == 0 else rot_rect(short)
        c.add_polygon([tuple(q) for q in outline(P, w)], layer=TM2)
        n1, f1 = via_array(c, TV2, *R['via_in'])
        n2, f2 = via_array(c, TV2, *R['via_land'])
        for q, lay in (('underpass', TM1), ('lead_n', TM2)):
            x0, y0, x1, y1 = R[q]
            c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                          layer=lay)
        # port: P on the coil's own lead, N on the underpass lead; the
        # lead ends face -y (primary) or +y (secondary)
        sgn = 1.0 if k == 0 else -1.0
        port = dict(P=[sgn*xe, sgn*y_port, w], N=[0.0, sgn*y_port, wu],
                    face='-y' if k == 0 else '+y')
        for x, hw in ((port['P'][0], w/2), (0.0, wu/2)):
            y0 = sgn*y_port
            y1 = y0 + sgn*1.0
            c.add_polygon([(x - hw, min(y0, y1)), (x + hw, min(y0, y1)),
                           (x + hw, max(y0, y1)), (x - hw, max(y0, y1))],
                          layer=TM2PIN)
        windings.append(dict(name=name, coil_path_um=P.tolist(),
                             rects_um=R, short_um=S, port=port,
                             vias=[n1, n2], via_fill=[f1, f2]))
    c.write_gds(a.out)

    st = PDK.layer_stack.layers
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    side = dict(
        kind='transformer', turns_per_winding=a.turns, width_um=w,
        space_um=s, d_in_um=a.d_in, lane_pitch_um=p,
        outer_flat_to_flat_um=2*(a_out + w/2), winding_length_um=length,
        coil_width_um=w, windings=windings,
        layers=dict(TopMetal2=[TM2.layer, TM2.datatype],
                    TopMetal1=[TM1.layer, TM1.datatype],
                    TopVia2=[TV2.layer, TV2.datatype],
                    TopMetal2pin=[TM2PIN.layer, TM2PIN.datatype]),
        stack_um={k: [float(st[k].zmin), float(st[k].thickness)]
                  for k in ('topmetal1', 'topvia2', 'topmetal2')})
    with open(a.out.rsplit('.', 1)[0] + '.json', 'w') as f:
        json.dump(side, f, indent=1)
    print(json.dumps({k: side[k] for k in ('outer_flat_to_flat_um',
                                           'winding_length_um',
                                           'lane_pitch_um')}))


if __name__ == '__main__':
    main()
