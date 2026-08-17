# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""THE MODULE CAPSTONE: three bond wires over split pads, swept.

Runs examples/module3wire.toml end to end through the input doctrine:
per frequency point the wires are rebuilt with the retuned skin shape
(doctrine rule 7 -- the wire coupling blocks are rebuilt, cost visible),
the coupled system is solved, and the table reports loop R, loop L and
the per-wire current shares. Wires A and C mirror about the pad
centreline, so share(A) == share(C) is a live symmetry check at every
point; B (centre) tells the proximity story.

Economy quadrature (ng=8, nq=3) is used for the wire near blocks --
measured IDENTICAL to full quadrature on this problem (shares to 5
decimals, Z to 7 digits) at 3.3x lower build cost.

ERROR BUDGET, oracle-anchored (dense exact assembly of the entire
system at 30 MHz, 2026-08-11): |Z| to 1.7e-3, shares to 5e-3
absolute, Re Z to 3.9% -- the far model's O(a/r) granularity class,
larger than the thin-wire validator's 2e-5 because these wires are
FAT (radius = 0.21x spacing). The printed mirror residual |A - C| is
the in-situ indicator: ~2.6e-3 of it is REAL (non-mirrored section
seams at this a/spacing -- confirmed in the oracle itself), the rest
is the far-model error. R at high frequency is the quantity the far
model limits first; L and sharing are comfortably inside the
cross-section model's own 0.5%.

Run: PYTHONPATH=src python3 studies/module_wires.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, 'src')
import sppeec_input  # noqa: E402

prob = sppeec_input.load('examples/module3wire.toml')
m = prob.model()
nleaf, numlevels = [5, 5, 3], 2   # auto partition would go single-level
M = m.build_tree(nleaf=nleaf, numlevels=numlevels)
print("module3wire: dims %s, %.0f%% fill, tree %s x %d levels"
      % (m.dims, m.fill(), nleaf, numlevels))

rows = []
for f in prob.freqs:
    m.prepare(M, f)
    t0 = time.perf_counter()
    sol = prob.solver(M, f, model=m, nq=3, ng=8)
    t1 = time.perf_counter()
    Z, info = sol.solve(f, rtol=prob.rtol)
    w = 2*np.pi*f
    sh = np.real(info['share'])
    rows.append((f, Z.real, abs(Z.imag)/w, sh, info['matvecs'],
                 t1 - t0, info['time'], info['residual']))
    print("  %.3g Hz: R %.6g Ohm  L %.6g nH  shares %s  "
          "(%d mv, build %.0f s, solve %.0f s, resid %.1e)"
          % (f, Z.real, 1e9*abs(Z.imag)/w, np.round(sh, 4),
             info['matvecs'], t1 - t0, info['time'],
             info['residual']))

print("\n f (Hz)      R (mOhm)   L (nH)     A       B       C")
for f, R, L, sh, mv, tb, ts, res in rows:
    print(" %-10.3g %-10.4f %-10.4f %-7.4f %-7.4f %-7.4f"
          % (f, 1e3*R, 1e9*L, *sh))
sym = max(abs(sh[0] - sh[2]) for _, _, _, sh, _, _, _, _ in rows)
print("\nmirror-symmetry residual max |share A - share C| = %.2e" % sym)
