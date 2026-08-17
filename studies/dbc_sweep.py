# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""THE FLAGSHIP DELIVERABLE: Z(f) of the DBC half-bridge at R3.

Loop R, L and per-die current sharing across 0.1-30 MHz -- through
the skin transition, which is exactly where surface-impedance methods
have no valid regime and where the module physics lives. Tree built
once (frequency-independent, treecost argmin shape); wires and their
coupling rebuilt per point (the doctrine's retune rule); GPU defaults
throughout; engineering rtol from the TOML.

Run: PYTHONPATH=src python3 studies/dbc_sweep.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, 'src')
import sppeec_input  # noqa: E402
import treecost  # noqa: E402
from studies.dbc_ladder import RUNGS, ENVELOPE  # noqa: E402

FREQS = [1e5, 1e6, 3e6, 1e7, 3e7]
name, pitch = RUNGS[2]                      # R3

prob = sppeec_input.load('examples/dbc_halfbridge.toml')
prob._doc['grid']['pitch'] = pitch
prob._doc['grid']['dims'] = [int(round(e/p))
                             for e, p in zip(ENVELOPE, pitch)]
m = prob.model()
nl, nlv, _ = treecost.recommend(m, gpu=True)
print("sweep at %s: tree %s lv%d" % (name, nl, nlv), flush=True)
M = m.build_tree(nleaf=nl, numlevels=nlv)

rows = []
sol = None
for f in FREQS:
    t0 = time.perf_counter()
    if sol is None:
        m.prepare(M, f)
        sol = prob.solver(M, f, model=m, nq=3, ng=8)
    else:
        # Tier 1 (2026-08-12): the solver persists across the sweep;
        # only the wire shapes retune (bit-identical to a fresh
        # build, 500x cheaper -- gated)
        sol.set_frequency(f)
    t_setup = time.perf_counter() - t0
    Z, info = sol.solve(f, rtol=prob.rtol, method=prob.method)
    w = 2*np.pi*f
    sh = np.real(info['share'])
    hs = (sh[0] + sh[1], sh[2] + sh[3])     # D1, D2 (2 wires each)
    ls = (sh[4] + sh[5], sh[6] + sh[7])     # D3, D4
    rows.append((f, Z.real, abs(Z.imag)/w, hs, ls, info))
    print("  %.3g Hz: R %.4f mOhm  L %.4f nH  D1/D2 %.4f/%.4f  "
          "D3/D4 %.4f/%.4f  (%d mv flag %s, setup %.0f s solve %.0f s)"
          % (f, 1e3*Z.real, 1e9*abs(Z.imag)/w, *hs, *ls,
             info['matvecs'], info['flag'], t_setup, info['time']),
          flush=True)

print("\n f (Hz)     R (mOhm)  L (nH)    D1     D2     D3     D4")
for f, R, L, hs, ls, info in rows:
    print(" %-9.3g %9.4f %8.4f  %.4f %.4f %.4f %.4f"
          % (f, 1e3*R, 1e9*L, *hs, *ls))
