# SPDX-License-Identifier: MIT
"""Is the wire10 'engine overshoot' a basis defect or a geometry gap?

THE QUESTION. In the current-flow narrative the conduction-mode engine
overshoots R on the staircase round wire (+7.2% at dx/delta 1.51,
+15.8% at 4.79) -- the one engine failure that survived correcting the
defaults. But the truth ladder in skinnarr.py RE-VOXELIZES the circle
at every rung (round_wire recomputes the disc mask on the finer grid),
so the reference converges to the FINE-staircase shape while the engine
solved the COARSE 10-cell staircase. frozenstair.py measured the gap
between those two shapes at r/dx=5, 10 GHz: the coarse staircase's own
converged R is +7.4% ABOVE the re-voxelized reference -- the same size
as the 'overshoot'.

THE TEST. Rebuild the wire10 ladder with the OCCUPANCY FROZEN: every
rung inherits the coarse 10-across disc mask (cross-section kron-refined
by mult, length cells scaled by mult), so all rungs are the SAME SHAPE
and R_dc is pinned exactly. Refine in the PLAIN basis, extrapolate, and
score the (already-measured, default-route) engine against THAT truth.

PREDICTIONS. Theory 'geometry gap': the overshoot collapses; the engine
lands at/slightly under the frozen truth (~93-100% of the AC rise).
Theory 'corner double-anchored modes over-crowd': a residual overshoot
survives against the same-shape truth.

Usage: PYTHONPATH=src:studies python3 studies/frozenwire.py [--deep]
  --deep adds a mult=4 rung (256k cells) at 100 GHz only, where the
  mult<=3 ladder is unconverged (finest rung still dx/delta=1.6).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'studies'))
import vhr                                                    # noqa: E402
from skinnarr import (SCRATCH, SIGMA, skin_depth, solve_sp,   # noqa: E402
                      extrapolate)

FREQS = [1e9, 2.5e9, 1e10, 2.5e10, 1e11]
N = 10                    # cells across at the coarse rung (wire10)
CS = 10e-6                # physical diameter
LENGTH = 50e-6
RUNGS = (1, 2, 3)
OUT = os.path.join(ROOT, 'studies', 'frozenwire_results.json')


def coarse_disc():
    """EXACTLY skinnarr.round_wire's rule at the coarse rung."""
    c = (N - 1) / 2.0
    r = N / 2.0
    yy, zz = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    return ((yy - c) ** 2 + (zz - c) ** 2) <= r * r


def build_frozen(mult):
    """Same SHAPE at every rung: kron-refined coarse mask."""
    dx = CS / (N * mult)
    disc = np.kron(coarse_disc(), np.ones((mult, mult), dtype=bool))
    ncell_len = int(round(LENGTH / dx))
    n = N * mult
    struc = np.zeros((ncell_len, n, n), dtype=np.int8)
    struc[:, disc] = 1
    ports = []
    for j in range(n):
        for k in range(n):
            if disc[j, k]:
                ports.append(('p1', 'P', 0, j, k, '-x'))
                ports.append(('p1', 'N', ncell_len - 1, j, k, '+x'))
    p = os.path.join(SCRATCH, 'wire10_frozen_m%d.vhr' % mult)
    vhr.write_vhr(p, struc, dx, SIGMA, tuple(FREQS), ports)
    return p, dx, int(struc.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deep', action='store_true')
    args = ap.parse_args()
    plan = [(m, FREQS) for m in RUNGS]
    if args.deep:
        plan.append((4, [1e11]))
    D = {}
    if os.path.exists(OUT):
        D = json.load(open(OUT))
    rungs = D.setdefault('rungs', {})
    for mult, freqs in plan:
        if str(mult) in rungs:
            print('m=%d cached' % mult, flush=True)
            continue
        p, dx, nc = build_frozen(mult)
        print('== frozen wire10 mult=%d dx=%.4g um cells=%d'
              % (mult, dx * 1e6, nc), flush=True)
        plain = solve_sp(p, freqs, engine=False)
        # R_dc is exact and IDENTICAL on every rung (shape frozen)
        r_dc = LENGTH / (SIGMA * int(coarse_disc().sum()) * (CS / N) ** 2)
        rungs[str(mult)] = dict(dx=dx, cells=nc, r_dc=r_dc,
                                plain={str(f): v for f, v in plain.items()})
        for f in freqs:
            print('   f=%-9.3g dx/delta=%-6.2f R=%.6f'
                  % (f, dx / skin_depth(f), plain[f]['R']), flush=True)
        with open(OUT, 'w') as fh:
            json.dump(D, fh, indent=1)

    # ------------------------------------------------------------- scoring
    dflt = json.load(open(os.path.join(ROOT, 'studies',
                                       'skinnarr_default_results.json')))
    reref = json.load(open(os.path.join(ROOT, 'studies',
                                        'skinnarr_results.json')))['wire10']
    r_dc = rungs['1']['r_dc']
    print('\nR_dc (exact, all rungs): %.6f' % r_dc)
    print('%-9s %-8s %-11s %-11s %-11s %-9s %-9s %-9s'
          % ('freq', 'dx/del', 'frozenQ0', 'finest', 'engine',
             'vs_frozen', 'vs_revox', 'delivered'))
    score = {}
    for f in FREQS:
        avail = sorted(((r['dx'], r['plain'][str(f)]['R'])
                        for r in rungs.values() if str(f) in r['plain']),
                       reverse=True)
        dxs = [a[0] for a in avail]
        Rs = [a[1] for a in avail]
        ex = extrapolate(dxs, Rs)
        q0 = ex['Q0'] if ex else Rs[-1]
        fine = Rs[-1]
        e = dflt['wire10'].get(str(f), {}).get('R')
        rv = reref['rungs']['3']['plain'][str(f)]['R']
        if e is None:
            continue
        dfz = 100 * (e - q0) / q0
        drv = 100 * (e - rv) / rv
        dv = 100 * (e - r_dc) / (q0 - r_dc) if q0 > r_dc * 1.0001 else None
        score[str(f)] = dict(frozen_Q0=q0, frozen_finest=fine, p=ex and
                             ex['p'], richardson=ex and ex['richardson'],
                             engine=e, err_vs_frozen_pct=dfz,
                             err_vs_revox_pct=drv, delivered_pct=dv)
        print('%-9.3g %-8.2f %-11.6f %-11.6f %-11.6f %+-9.1f %+-9.1f %-9s'
              % (f, rungs['1']['dx'] / skin_depth(f), q0, fine, e,
                 dfz, drv, '%.0f%%' % dv if dv is not None else '-'))
    D['score'] = score
    with open(OUT, 'w') as fh:
        json.dump(D, fh, indent=1)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
