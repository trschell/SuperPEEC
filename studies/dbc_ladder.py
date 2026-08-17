# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""THE REFINEMENT LADDER on the DBC flagship -- the measurement that
decides whether 1e9 cells is days or months.

Same physical module (examples/dbc_halfbridge.toml -- fully physical:
metre-unit blocks, port boxes, wire polylines), solved at successively
finer pitch. Along THIS axis the design features are fixed -- same 8
wires, same 6 sharing chords in the exact Schur block, zero holes --
so the iteration count measures the FIELD problem, not growing
homology (the confound that poisoned the growth-axis ladder; see the
scaling memory). Flags are checked (a saturated lgmres count is
indistinguishable from a converged one otherwise), rtol is the
engineering 1e-4 (the 81%-waste law), basis is overcomplete
(memory-flat BlockAMG + Schur).

Each rung prints cells / filaments / matvecs / R / L / peak RSS and
the local iteration-growth exponent against the previous rung.

Usage: PYTHONPATH=src python3 studies/dbc_ladder.py \
           [n] [start]
       (n = number of rungs, default 2; start = first rung index,
        default 0 -- e.g. "1 2" runs only R3)
"""
import gc
import resource
import sys
import time

import numpy as np

sys.path.insert(0, 'src')
import sppeec_input  # noqa: E402

# pitch per rung; dims derive from the fixed physical envelope
# (40 x 50 x 4 mm). Aspect stays 2.5:1 (foot_model 'point' until a
# cubic rung).
RUNGS = [
    ('R1', [0.5e-3, 0.5e-3, 0.2e-3]),
    ('R2', [0.25e-3, 0.25e-3, 0.1e-3]),
    ('R3', [0.125e-3, 0.125e-3, 0.05e-3]),
    # the STRETCH rung: commensurate pitch (1/16 mm in-plane divides
    # every 1 mm feature; 0.04 mm divides the 0.2 mm layers -- the
    # 0.08 mm probe's half-cell snaps are the trap this avoids).
    # ~51M cells, ~47 GB projected peak: deliberately near this box's
    # ceiling, because the hero run needs the transients de-risked.
    ('R4', [0.0625e-3, 0.0625e-3, 0.04e-3]),
]
ENVELOPE = [40e-3, 50e-3, 4e-3]
FREQ = 1e6

def main(nrun=2, start=0):
    rows = []
    for name, pitch in RUNGS[start:start + nrun]:
        prob = sppeec_input.load('examples/dbc_halfbridge.toml')
        prob._doc['grid']['pitch'] = pitch
        prob._doc['grid']['dims'] = [int(round(e/p))
                                     for e, p in zip(ENVELOPE, pitch)]
        m = prob.model()
        cells = int(np.prod(m.dims))
        t0 = time.perf_counter()
        # tree shape from the calibrated cost-model argmin (treecost;
        # validated against the R2 sweep + check runs 2026-08-12)
        import treecost
        nl, nlv, _ = treecost.recommend(m, gpu=True)
        print("   treecost -> nleaf %s, %d levels" % (nl, nlv),
              flush=True)
        M = m.build_tree(nleaf=nl, numlevels=nlv)
        t_tree = time.perf_counter() - t0
        m.prepare(M, FREQ)
        t0 = time.perf_counter()
        sol = prob.solver(M, FREQ, model=m, nq=3, ng=8, verbose=True)
        t_setup = time.perf_counter() - t0
        Z, info = sol.solve(FREQ, rtol=prob.rtol, verbose=True)
        w = 2*np.pi*FREQ
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
        row = dict(name=name, cells=cells, fils=sol.efg,
                   mv=info['matvecs'], flag=info['flag'],
                   resid=info['residual'], R=Z.real,
                   L=abs(Z.imag)/w, share=np.real(info['share']),
                   t_tree=t_tree, t_setup=t_setup, t_solve=info['time'],
                   rss=rss)
        rows.append(row)
        print("== %s: %.2e cells, %d fils | %d mv (flag %s, resid %.1e) "
              "| R %.4f mOhm  L %.4f nH | tree %.0f s setup %.0f s "
              "solve %.0f s | peak RSS %.1f GB"
              % (name, cells, sol.efg, row['mv'], row['flag'],
                 row['resid'], 1e3*row['R'], 1e9*row['L'], t_tree,
                 t_setup, row['t_solve'], rss), flush=True)
        if len(rows) > 1:
            a, b = rows[-2], rows[-1]
            if a['flag'] == 0 and b['flag'] == 0:
                p = np.log(b['mv']/a['mv'])/np.log(b['cells']/a['cells'])
                print("   iteration-growth exponent %s->%s: %.3f  "
                      "(L moved %.3f%%)"
                      % (a['name'], b['name'], p,
                         100*abs(b['L']/a['L'] - 1)), flush=True)
            else:
                print("   NO exponent: a rung did not converge "
                      "(flags %s/%s)" % (a['flag'], b['flag']), flush=True)
        del sol, M, m
        gc.collect()
    
    print("\n name    cells      fils     mv   R(mOhm)  L(nH)   RSS(GB)")
    for r in rows:
        print(" %-6s %9.2e %8d %5d  %8.4f %7.4f  %6.1f"
              % (r['name'], r['cells'], r['fils'], r['mv'],
                 1e3*r['R'], 1e9*r['L'], r['rss']))


if __name__ == '__main__':
    import sys as _s
    main(int(_s.argv[1]) if len(_s.argv) > 1 else 2,
         int(_s.argv[2]) if len(_s.argv) > 2 else 0)
