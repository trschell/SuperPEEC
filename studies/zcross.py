# SPDX-License-Identifier: MIT
"""Check Zcross: the mode <- aggregate drive.

Zcross[m of filament a, b] = sum_p W[p,m] * Mp(sub_ap, FULL box of b),
Mp = S/(a_sub * a_full). Zuu (diag AND off-diag) already verified exact,
so this is the remaining assembled block.
"""
import os, numpy as np, scipy.sparse as sp, vhr, equiterminal as eq
from greens import box_pair_stencil_pairs as spair

SPD = os.path.dirname(os.path.abspath(__file__))
DX, SIG = 1e-6, 5.8e7
m = vhr.read_vhr(os.path.join(SPD, 'tiny.vhr'))
M = m.build_tree()
m.prepare(M, 1e10)
S = eq.EquiTerminalSolver(m, M, 0, subdivide=3, skin_freq=1e10, use_fft=False)
r = S.redist
km = r.k - 1
Zc = sp.csr_matrix(r.Zcross)
print("Zcross shape %s  (nmode=%d, nfil=%d)  a_full=%.4g"
      % (Zc.shape, r.nmode, r.nfil, r.afull))
lo0, hi0 = r.lo[:r.k], r.hi[:r.k]
worst = 0.0
for b in range(min(r.nfil, 8)):
    fb_lo = r.flo[b][None, :].repeat(r.k, 0)
    fb_hi = r.fhi[b][None, :].repeat(r.k, 0)
    Sab = spair(lo0, hi0, fb_lo, fb_hi)
    Mp = Sab/(r.asub*r.afull)                # (k,) sub_a -> full_b
    ref = r.W.T @ Mp                         # (km,)
    got = np.asarray(Zc[:km, b].todense()).ravel()
    den = max(np.abs(ref).max(), 1e-300)
    rel = np.abs(got-ref).max()/den
    worst = max(worst, rel if np.abs(ref).max() > 1e-25 else 0.0)
    print("   mode(fil 0) <- fil %d  sep %-12s |ref|max=%.3e |got|max=%.3e "
          "rel=%.2e" % (b, str(tuple(r.cells[b]-r.cells[0])),
                        np.abs(ref).max(), np.abs(got).max(), rel))
print("\nworst relative difference: %.3e" % worst)
