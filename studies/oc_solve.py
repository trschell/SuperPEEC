# SPDX-License-Identifier: MIT
"""Wire the OVER-COMPLETE plaquette basis into a real LpR port solve.

Swaps LpRSolver's (Y, YT, chol) for (Y_full, Y_full^T, AMG-on-Y_full^T Y_full).
The loop system Y^T Z Y is then SINGULAR but consistent (kernel = cube
boundaries = physically identical currents), so the test is whether
lgmres still converges and returns the SAME impedance as the stock
exact-Cholesky solver.
"""
import time, numpy as np, scipy.sparse as sp, pyamg
import vhr, port_impedance as pz
import meshgraph as mg   # getmesh_full: the FORTRAN enumerator

import sys
FREQS = (1e8, 2.5e9, 1e10)
name = sys.argv[1] if len(sys.argv) > 1 else 'wire_len50.0u_dia10.0u.vhr'
m = vhr.read_vhr('VoxHenry/Input_files/' + name)
M = m.build_tree()


class AMGPrec:
    """Callable stand-in for the cholmod Factor (float32 in/out)."""
    def __init__(self, G, cycles=2):
        self.ml = pyamg.smoothed_aggregation_solver(G.astype(np.float64),
                                                    max_coarse=400)
        self.nnz = sum(l.A.nnz for l in self.ml.levels)/G.nnz
        self.cycles = cycles

    def __call__(self, b):
        x = np.zeros(b.shape[0])
        for _ in range(self.cycles):
            x = self.ml.solve(np.float64(b), x0=x, tol=1e-14, maxiter=1,
                              cycle='V')
        return np.float32(x)


real_init = pz.LpRSolver.__init__
USE_OC = {'on': False, 'cycles': 2}


def patched(self, M_, verbose=False, **kw):
    real_init(self, M_, verbose=verbose, **kw)
    if not USE_OC['on']:
        return
    es_, fs2, gs2 = (np.size(M_.e.struc), np.size(M_.f.struc),
                     np.size(M_.g.struc))
    Yf = mg.getmesh_full(M_.adjmats(), es_, es_+fs2,
                         es_+fs2+gs2, np.size(M_.lv[0].struc))
    self.Y = Yf
    self.YT = Yf.T.tocsc()
    self.meshsize = Yf.shape[1]
    G = (Yf.T @ Yf).tocsr()
    self.chol = AMGPrec(G, USE_OC['cycles'])
    self.amg_nnz = self.chol.nnz
    self.amg_mb = self.chol.nnz*G.nnz*12/1e6


pz.LpRSolver.__init__ = patched
print("%-14s %-9s %-8s %-9s %-13s %-13s %s"
      % ("basis", "freq", "meshsize", "matvecs", "R [ohm]", "L [H]", "sec"))
ref = {}
for tag in ('selected', 'over-complete'):
    USE_OC['on'] = (tag == 'over-complete')
    for f in FREQS:
        m.prepare(M, f)
        t0 = time.perf_counter()
        try:
            S = pz.LpRSolver(M)
            sn = m.source_vector(M, port=0)
            Z, infos = pz.impedance_matrix(m, M, S, f)
            z = complex(np.asarray(Z).ravel()[0])
            R, L = z.real, z.imag/(2*np.pi*f)
            key = (tag, f)
            ref[key] = (R, L)
            extra = ("  AMG %.1fx = %.0f MB" % (S.amg_nnz, S.amg_mb)) if USE_OC["on"] else ("  chol %.0f MB" % (S.chol.L().nnz*12/1e6))
            print("%-14s %-9.3g %-8d %-9s %-13.7g %-13.7g %.1f%s"
                  % (tag, f, S.meshsize, str(S.matvecs), R, L,
                     time.perf_counter()-t0, extra), flush=True)
            del S
        except Exception as e:
            print("%-14s %-9.3g FAILED %s: %s"
                  % (tag, f, type(e).__name__, str(e)[:60]), flush=True)
print()
for f in FREQS:
    a, b = ref.get(('selected', f)), ref.get(('over-complete', f))
    if a and b:
        print("  f=%-9.3g  R ratio %.6f   L ratio %.6f"
              % (f, b[0]/a[0], b[1]/a[1]))
