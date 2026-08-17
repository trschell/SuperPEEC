# SPDX-License-Identifier: MIT
"""Which CELLS drive the wire's mode-model overshoot: boundary or interior?

Redistribution takes (fil_axis, fil_cell) as its filament list, so
wrapping it and filtering that list restricts which filaments get modes
without touching the mode machinery. A filament is classified by its LOW
cell; 'boundary' = at least one of the four TRANSVERSE neighbours empty.

  all cells  -> 0.04366244  (+8.54%)
  k=1        -> 0.03800266
  converged  -> 0.04022542
"""
import os, sys, numpy as np, vhr, equiterminal as eq

CONV, K1 = 0.04022542, 0.03800266
SPD = os.path.dirname(os.path.abspath(__file__))
m = vhr.read_vhr(os.path.join(SPD, 'wire_1.vhr'))
s = m.struc().astype(bool)
p = np.pad(s, 1)
INTERIOR = (p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1] &
            p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:]) & s
print("wire: %d cells, %d interior, %d boundary"
      % (s.sum(), INTERIOR.sum(), s.sum()-INTERIOR.sum()))

REAL = eq.Redistribution


def make(mode):
    def patched(model, M, axis, fil_axis, fil_cell, **kw):
        fa = np.asarray(fil_axis); cc = np.asarray(fil_cell)
        par = fa == axis
        isint = np.zeros(len(fa), bool)
        ok = par
        isint[ok] = INTERIOR[cc[ok, 0], cc[ok, 1], cc[ok, 2]]
        keep = (~par | ~isint) if mode == 'boundary' else \
               (~par | isint) if mode == 'interior' else \
               np.ones(len(fa), bool)
        print("   %-9s modes on %d of %d parallel filaments"
              % (mode, int((par & keep).sum()), int(par.sum())), flush=True)
        return REAL(model, M, axis, fa[keep], cc[keep], **kw)
    return patched


for mode in ('all', 'boundary', 'interior'):
    eq.Redistribution = make(mode)
    M = m.build_tree()
    m.prepare(M, 1e10)
    S = eq.EquiTerminalSolver(m, M, 0, subdivide=True, skin_freq=1e10)
    Z, ii, info = S.solve(1e10)
    R = Z.real
    print("%-9s R = %-13.7g vs conv %+7.2f%%  %.0f%% of needed  resid %.1e"
          % (mode, R, 100*(R/CONV-1), 100*(R-K1)/(CONV-K1),
             info['residual']), flush=True)
    del M, S
