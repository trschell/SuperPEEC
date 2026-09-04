# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""THE FLAGSHIP, first light: the PowerSynth-derived DBC half-bridge
through the input doctrine, at demo pitch.

Runs examples/dbc_halfbridge.toml end to end: build, sanity-report the
derived circuit structure (islands, feet, sharing DOFs -- all found,
not stated), solve the commutation loop at each [solve] frequency, and
report loop R, L and the per-die current split (BW1/BW3 = high-side
D1/D2 shares; BW5/BW7 = low-side D3/D4 shares, each a doubled bond).

This same file is the seed of the refinement-ladder driver: the TOML
is fully physical (from_m/to_m, p_box/n_box, wire polylines in
metres), so a ladder rung is a dims+pitch edit only.

Run: PYTHONPATH=src python3 studies/dbc_flagship.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, 'src')
import sppeec_input  # noqa: E402

prob = sppeec_input.load('examples/dbc_halfbridge.toml')
m = prob.model()
print("dbc_halfbridge: dims %s, pitch %s, fill %.1f%%"
      % (m.dims, list(m.d), m.fill_pct()))
t0 = time.perf_counter()
M = m.build_tree()
print("tree: %d levels, built %.1f s" % (M.numlevels,
                                         time.perf_counter() - t0))

for f in prob.freqs:
    m.prepare(M, f)
    t0 = time.perf_counter()
    sol = prob.solver(M, f, model=m, nq=3, ng=8, verbose=True)
    t1 = time.perf_counter()
    Z, info = sol.solve(f, rtol=1e-8, verbose=True)
    w = 2*np.pi*f
    sh = np.real(info['share'])
    names = [w_['name'] for w_ in prob.wire_specs]
    print("  %.3g Hz: R %.4f mOhm  L %.4f nH  (%d mv, resid %.1e, "
          "build %.0f s, solve %.0f s)"
          % (f, 1e3*Z.real, 1e9*abs(Z.imag)/w, info['matvecs'],
             info['residual'], t1 - t0, info['time']))
    hs = sh[0] + sh[1], sh[2] + sh[3]        # D1 vs D2 (2 wires each)
    ls = sh[4] + sh[5], sh[6] + sh[7]        # D3 vs D4
    print("    high-side split D1/D2: %.4f / %.4f   "
          "low-side split D3/D4: %.4f / %.4f"
          % (hs[0], hs[1], ls[0], ls[1]))
    for n, s in zip(names, sh):
        print("      %-12s %.4f" % (n, s))
