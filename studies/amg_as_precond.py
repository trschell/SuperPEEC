# SPDX-License-Identifier: MIT
"""THE question that actually matters: chol(Y^T Y) is not used to SOLVE
G -- it is a PRECONDITIONER for the loop system Y^T Z Y. So the metric
is OUTER iterations of the real LpR solve, not PCG-on-G.

Swap LpRSolver's exact Cholesky for an AMG V-cycle and count outer
lgmres iterations at several frequencies. AMG needs 133 PCG iters to
SOLVE G (vs 1 for chol) but costs 7 MB against 131 MB -- if the outer
solve only needs a few more iterations, that trade is worth taking.
"""
import time, numpy as np, scipy.sparse as sp, vhr, pyamg
import port_impedance as pz

m = vhr.read_vhr('VoxHenry/Input_files/'
                 'straight_cond1_len30.0u_wid10.0u_dist20.0u.vhr')
M = m.build_tree()


class AMGChol:
    """Drop-in for the cholmod Factor: callable, float32 in/out."""
    def __init__(self, YT32, cycles=1):
        G = (YT32 @ YT32.T).tocsr().astype(np.float64)   # cholesky_AAt = A A^T
        t0 = time.perf_counter()
        self.ml = pyamg.smoothed_aggregation_solver(G, max_coarse=500)
        self.setup = time.perf_counter()-t0
        self.nnz = sum(l.A.nnz for l in self.ml.levels)
        self.cycles = cycles

    def __call__(self, b):
        x = np.zeros(b.shape[0])
        for _ in range(self.cycles):
            x = self.ml.solve(np.float64(b), x0=x, tol=1e-14, maxiter=1,
                              cycle='V')
        return np.float32(x)


real_init = pz.LpRSolver.__init__
MODE = {'kind': 'chol', 'cycles': 1}


def patched(self, M_, verbose=False, **kw):
    real_init(self, M_, verbose=verbose, **kw)
    if MODE['kind'] == 'amg':
        YT32 = self.YT.copy(); YT32.data = np.float32(YT32.data)
        self.chol = AMGChol(YT32, MODE['cycles'])


pz.LpRSolver.__init__ = patched
print("%-16s %-8s %-9s %-8s %s"
      % ("preconditioner", "freq", "matvecs", "resid", "solve s"))
for kind, cyc in (('chol', 0), ('amg', 1), ('amg', 2), ('amg', 4)):
    MODE['kind'], MODE['cycles'] = kind, cyc
    for f in (1e8, 2.5e9, 1e10):
        m.prepare(M, f)
        S = pz.LpRSolver(M)
        sn = m.source_vector(M, port=0)
        t0 = time.perf_counter()
        try:
            x = S.solve(sn, maxiter=10)
            info = getattr(S, 'matvecs', -1)
            print("%-16s %-8.3g %-9d %-8s %.2f"
                  % ("%s x%d" % (kind, cyc) if kind == 'amg' else kind,
                     f, info, "-", time.perf_counter()-t0), flush=True)
        except Exception as e:
            print("%-16s %-8.3g FAILED %s: %s"
                  % (kind, f, type(e).__name__, str(e)[:50]), flush=True)
        del S
