# SPDX-License-Identifier: MIT
"""Largest VoxHenry model on the new preconditioner: does it hold up?

square_coil: 400x20x400 box, 604800 voxels -- ~25x the 24k bar, and the
model that was never attempted under the old 6 GB cap. Staged so a
failure is informative rather than just an OOM: sizes, then basis, then
preconditioner, then one solve.
"""
import sys, time, numpy as np, vhr, port_impedance as pz, meshgraph as mg
import meshgraph_aux


def rss():
    for ln in open('/proc/self/status'):
        if ln.startswith('VmHWM:'):
            return int(ln.split()[1])/1e6
    return 0.0


name = sys.argv[1] if len(sys.argv) > 1 else \
    'square_coil_len100.0u_wid5.0u_heig5.0u_dist2.0u.vhr'
basis = sys.argv[2] if len(sys.argv) > 2 else 'overcomplete'
F = 2.5e9
m = vhr.read_vhr('VoxHenry/Input_files/' + name)
print("%s: dims=%s  voxels=%d  fill=%.1f%%"
      % (name, tuple(np.asarray(m.dims, int)), int(m.struc().sum()), m.fill()),
      flush=True)
t0 = time.perf_counter()
M = m.build_tree()
m.prepare(M, F)
es, fs_, gs = (np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc))
efg, nn = es+fs_+gs, np.size(M.lv[0].struc)
print("  tree built  %.1f s, peak %.1f GB | edges=%d nodes=%d"
      % (time.perf_counter()-t0, rss(), efg, nn), flush=True)

t0 = time.perf_counter()
A = M.adjmats(); adjs = (A+A.T).tocsr()
rank = efg - nn + meshgraph_aux.counttrees(adjs.indices, adjs.indptr)
print("  adjacency   %.1f s, peak %.1f GB | cycle rank = %d"
      % (time.perf_counter()-t0, rss(), rank), flush=True)

t0 = time.perf_counter()
S = pz.LpRSolver(M, basis=basis)
tsetup = time.perf_counter()-t0
pc = ("AMG %.1fx = %.0f MB, macro=%d"
      % (S.chol.nnz_ratio, S.chol.nnz*12/1e6, getattr(S.chol, 'nmac', 0))
      if basis == 'overcomplete' else
      "chol %.0f MB" % (S.chol.L().nnz*12/1e6))
print("  solver      %.1f s, peak %.1f GB | meshsize=%d (%.2fx rank) | %s"
      % (tsetup, rss(), S.meshsize, S.meshsize/rank, pc), flush=True)

t0 = time.perf_counter()
Z, infos = pz.impedance_matrix(m, M, S, F)
z = complex(np.asarray(Z).ravel()[0])
print("  SOLVE @%.3g Hz: %.1f s, %d matvecs, peak %.1f GB"
      % (F, time.perf_counter()-t0, S.matvecs, rss()), flush=True)
print("    R = %.7g ohm   L = %.7g H" % (z.real, z.imag/(2*np.pi*F)), flush=True)
