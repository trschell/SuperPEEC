# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""A 1:1 STACKED octagonal transformer on the IHP SG13G2 BEOL, written as
GDS through the IHP gdsfactory PDK.

Runs in the PDK's own environment (ihp-gdsfactory needs Python < 3.14):

    ~/.venvs/ihp/bin/python studies/ihp_stacked_transformer_gds.py --out s.gds

TOPOLOGY. The broadside-coupled ("stacked") monolithic transformer: the
primary an octagonal spiral on TopMetal2, the secondary the SAME spiral
on TopMetal1 directly beneath it, turn over turn (J. R. Long, IEEE JSSC
35 (9), 2000; the stacked transformer is the high-coupling member of
the family). TopMetal1 being taken under every primary turn, the inner
ends leave through the next metal down, as in silicon: each winding's
inner end runs a short stub into the empty centre (the primary's on
TopMetal2 at x = 0, the secondary's on TopMetal1 further left), drops
by a via stack to its own Metal5 underpass, crosses under both
windings, and rises outside to its lead. Both lead pairs exit at the
bottom, side by side.

    primary   TM2 coil + lead (x = xe); stub at x = 0 -> TV2 -> TM1 pad
              -> TV1 -> M5 underpass -> TV1 -> TM1 pad -> TV2 -> TM2 lead_n
    secondary TM1 coil + lead (x = xs_end); stub at x = xs -> TV1 -> M5
              underpass -> TV1 -> TM1 lead_n

MODEL STACK (um, origin at the Metal5 bottom; the PDK's Metal5 0.49 and
TopVia1 0.85 um are rounded to 0.5 and 1.0, TopVia2 2.8 to 3.0, so 0.5
and 0.25 um pitches are commensurate): M5 [0, 0.5), TV1 [0.5, 1.5),
TM1 [1.5, 3.5), TV2 [3.5, 6.5), TM2 [6.5, 9.5). The two coils are
[[trace]]s at different heights overlapping in plan view: a multi-span
section cut (src/section.py, 2026-10-05).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ihp_spiral_gds import centreline, outline, via_array   # noqa: E402

MODEL_STACK = dict(Metal5=(0.0, 0.5), TopVia1=(0.5, 1.5),
                   TopMetal1=(1.5, 3.5), TopVia2=(3.5, 6.5),
                   TopMetal2=(6.5, 9.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--turns', type=int, default=6, help='turns PER WINDING')
    ap.add_argument('--width', type=float, default=15.0)
    ap.add_argument('--space', type=float, default=3.0)
    ap.add_argument('--d-in', type=float, default=240.0)
    ap.add_argument('--xe', type=float, default=45.0,
                    help='primary outer lead centre x, um')
    ap.add_argument('--xs', type=float, default=-45.0,
                    help='secondary inner stub / underpass centre x, um')
    ap.add_argument('--xs-end', type=float, default=-80.0,
                    help='secondary outer lead centre x, um')
    ap.add_argument('--stub', type=float, default=27.5,
                    help='inner stub length into the centre, um')
    ap.add_argument('--gap', type=float, default=15.0)
    ap.add_argument('--lead', type=float, default=30.0)
    ap.add_argument('--wu', type=float, default=16.0)
    ap.add_argument('--out', default='ihp_stacked_transformer.gds')
    a = ap.parse_args(argv)

    import gdsfactory as gf
    from ihp import PDK
    PDK.activate()
    L = PDK.layers
    lay = dict(TopMetal2=L.TopMetal2drawing, TopMetal1=L.TopMetal1drawing,
               TopVia2=L.TopVia2drawing, TopVia1=L.TopVia1drawing,
               Metal5=L.Metal5drawing, TopMetal2pin=L.TopMetal2pin,
               TopMetal1pin=L.TopMetal1pin)
    w, wu = a.width, a.wu

    def coil(x_start, x_end):
        pts, a0, a_out = centreline(a.turns, w, a.space, a.d_in, x_end,
                                    x_start=x_start)
        y_top = -a0 + a.stub
        path = np.vstack([[[x_start, y_top]], pts[:-1], [[x_end, -a_out]]])
        return path, a0, a_out

    pp, a0, a_out = coil(0.0, a.xe)
    sp_, _, _ = coil(a.xs, a.xs_end)
    y_land = -(a_out + w/2 + a.gap + w/2)
    y_port = y_land - w/2 - a.lead
    y_stub = -a0 + a.stub
    pp = np.vstack([pp, [[a.xe, y_port]]])
    sp_ = np.vstack([sp_, [[a.xs_end, y_port]]])

    def sq(x, y):          # a wu x w landing centred on (x, y)
        return [x - wu/2, y - w/2, x + wu/2, y + w/2]

    # stub squares sit at the stub's top end, inside the coil's hole
    ps_, ss_ = sq(0.0, y_stub - w/2), sq(a.xs, y_stub - w/2)
    pl_, sl_ = sq(0.0, y_land), sq(a.xs, y_land)
    blocks_p = [  # (name, layer, rect, via?)
        ('stub_TV2', 'TopVia2', ps_, True),
        ('stub_TM1', 'TopMetal1', ps_, False),
        ('stub_TV1', 'TopVia1', ps_, True),
        ('underpass_M5', 'Metal5', [-wu/2, pl_[1], wu/2, ps_[3]], False),
        ('land_TV1', 'TopVia1', pl_, True),
        ('land_TM1', 'TopMetal1', pl_, False),
        ('land_TV2', 'TopVia2', pl_, True),
        ('lead_n', 'TopMetal2', [-wu/2, y_port, wu/2, pl_[3]], False)]
    blocks_s = [
        ('stub_TV1', 'TopVia1', ss_, True),
        ('underpass_M5', 'Metal5', [a.xs - wu/2, sl_[1], a.xs + wu/2, ss_[3]],
         False),
        ('land_TV1', 'TopVia1', sl_, True),
        ('lead_n', 'TopMetal1', [a.xs - wu/2, y_port, a.xs + wu/2, sl_[3]],
         False)]
    for name, _, r, _v in blocks_p + blocks_s:
        if any(abs(v - round(v)) > 1e-9 for v in r):
            raise SystemExit("%s %s is off the 1 um grid" % (name, r))

    c = gf.Component('ihp_stacked_xfmr_%dt' % a.turns)
    c.add_polygon([tuple(q) for q in outline(pp, w)], layer=lay['TopMetal2'])
    c.add_polygon([tuple(q) for q in outline(sp_, w)], layer=lay['TopMetal1'])
    windings = []
    for wname, trace_layer, path, blocks, xP, xN in (
            ('primary', 'TopMetal2', pp, blocks_p, a.xe, 0.0),
            ('secondary', 'TopMetal1', sp_, blocks_s, a.xs_end, a.xs)):
        blist = []
        for name, layname, r, is_via in blocks:
            fill = 1.0
            if is_via:
                size, space = ((0.9, 1.06) if layname == 'TopVia2'
                               else (0.42, 0.42))
                n, fill = via_array(c, lay[layname], *r, size=size,
                                    space=space)
            else:
                x0, y0, x1, y1 = r
                c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                              layer=lay[layname])
            blist.append(dict(name=name, layer=layname, rect=r,
                              via=bool(is_via), fill=float(fill)))
        for x, hw in ((xP, w/2), (xN, wu/2)):
            c.add_polygon([(x - hw, y_port), (x + hw, y_port),
                           (x + hw, y_port + 1.0), (x - hw, y_port + 1.0)],
                          layer=lay[trace_layer + 'pin'])
        windings.append(dict(name=wname, trace_layer=trace_layer,
                             coil_path_um=path.tolist(), blocks=blist,
                             port=dict(P=[xP, y_port, w], N=[xN, y_port, wu],
                                       face='-y')))
    c.write_gds(a.out)

    st = PDK.layer_stack.layers
    length = float(np.linalg.norm(np.diff(pp, axis=0), axis=1).sum())
    side = dict(
        kind='transformer', topology='stacked',
        turns_per_winding=a.turns, width_um=w, space_um=a.space,
        d_in_um=a.d_in, outer_flat_to_flat_um=2*(a_out + w/2),
        winding_length_um=length, coil_width_um=w, windings=windings,
        model_stack_um={k: list(v) for k, v in MODEL_STACK.items()},
        layers={k: [v.layer, v.datatype] for k, v in lay.items()},
        stack_um={k: [float(st[k].zmin), float(st[k].thickness)]
                  for k in ('metal5', 'topvia1', 'topmetal1', 'topvia2',
                            'topmetal2')})
    with open(a.out.rsplit('.', 1)[0] + '.json', 'w') as f:
        json.dump(side, f, indent=1)
    print(json.dumps({k: side[k] for k in ('outer_flat_to_flat_um',
                                           'winding_length_um')}))


if __name__ == '__main__':
    main()
