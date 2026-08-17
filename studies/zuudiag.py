# SPDX-License-Identifier: MIT
"""Check the Zuu DIAGONAL: does the shipped mode self-impedance equal the
exact wT Z_sub w for one filament's own sub-bars?

Zuu is the partial-INDUCTANCE block (applied as jw*Zuu; Ru carries the
resistance), so the reference is  W^T Lp_sub W  with Lp_sub the k x k
partial inductance matrix of ONE filament's sub-bars,
Mp = S/(A_a A_b) with S from greens.box_pair_stencil_pairs.
"""
import os, numpy as np, scipy.sparse as sp, vhr, equiterminal as eq
from greens import box_pair_stencil_pairs as spair

SPD = os.path.dirname(os.path.abspath(__file__))
DX, SIG = 1e-6, 5.8e7
struc = np.ones((4, 2, 2), dtype=np.int8)
ports = [(('p1', 'P', 0, j, k, '-x') if s == 0 else ('p1', 'N', 3, j, k, '+x'))
         for s in (0, 1) for j in range(2) for k in range(2)]
p = os.path.join(SPD, 'tiny.vhr')
vhr.write_vhr(p, struc, DX, SIG, (1e10,), ports)
m = vhr.read_vhr(p)
M = m.build_tree()
m.prepare(M, 1e10)
S = eq.EquiTerminalSolver(m, M, 0, subdivide=3, skin_freq=1e10,
                          use_fft=False)
r = S.redist
km = r.k - 1
print("k=%d (%dx%d), %d filaments, %d modes, a_sub=%.4g m^2"
      % (r.k, r.kk[0], r.kk[1], r.nfil, r.nmode, r.asub))

# --- reference: one filament's own k x k sub-bar partial inductances ---
lo, hi = r.lo[:r.k], r.hi[:r.k]
ia, ib = np.meshgrid(np.arange(r.k), np.arange(r.k), indexing='ij')
Ssten = spair(lo[ia.ravel()], hi[ia.ravel()], lo[ib.ravel()], hi[ib.ravel()])
Lp = Ssten.reshape(r.k, r.k)/(r.asub*r.asub)
ref = r.W.T @ Lp @ r.W                       # exact mode self-block

got = np.asarray(sp.csr_matrix(r.Zuu)[:km, :km].todense())
print("\nZuu self-block (filament 0), shipped vs exact wT Lp w:")
print("  shipped diag:", np.array2string(np.diag(got), precision=4))
print("  exact   diag:", np.array2string(np.diag(ref), precision=4))
e = np.abs(got-ref).max()/np.abs(ref).max()
print("  max rel diff over the whole block: %.3e" % e)

# --- OFF-DIAGONAL: mode-mode coupling between DIFFERENT filaments ---
print("\nOFF-DIAGONAL Zuu blocks (filament 0 vs filament b):")
Z = sp.csr_matrix(r.Zuu)
for b in range(1, min(r.nfil, 6)):
    lb, hb = r.lo[b*r.k:(b+1)*r.k], r.hi[b*r.k:(b+1)*r.k]
    Sab = spair(lo[ia.ravel()], hi[ia.ravel()], lb[ib.ravel()], hb[ib.ravel()])
    Lab = Sab.reshape(r.k, r.k)/(r.asub*r.asub)
    refab = r.W.T @ Lab @ r.W
    gotab = np.asarray(Z[:km, b*km:(b+1)*km].todense())
    sep = (r.cells[b] - r.cells[0])
    den = max(np.abs(refab).max(), 1e-300)
    print("   fil 0 -> %d  cell sep %-12s |ref|max=%.3e  |got|max=%.3e  "
          "rel diff=%.3e" % (b, str(tuple(sep)), np.abs(refab).max(),
                             np.abs(gotab).max(),
                             np.abs(gotab-refab).max()/den))

# --- and the physically meaningful ratio: wL vs R for one mode ---
w = 2*np.pi*1e10
Ru = np.asarray(sp.csr_matrix(r.Ru)[:km, :km].todense())
print("\nmode 0:  R=%.5g ohm   wL(shipped)=%.5g   wL(exact)=%.5g   wL/R=%.4g"
      % (Ru[0, 0], w*got[0, 0], w*ref[0, 0], w*ref[0, 0]/Ru[0, 0]))
print("r_sub expected = k/(sigma*dx) = %.5g ; Ru[0,0] should be 2*r_sub = %.5g"
      % (r.k/(SIG*DX), 2*r.k/(SIG*DX)))
