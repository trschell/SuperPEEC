# SPDX-License-Identifier: MIT
"""Stage-by-stage RSS for the LpR port solve: tree, mesh basis, Cholesky, solve."""
import sys, time, numpy as np, vhr, port_impedance as pz

def rss():
    cur = hwm = 0.0
    for ln in open('/proc/self/status'):
        if ln.startswith('VmRSS:'): cur = int(ln.split()[1])/1e6
        if ln.startswith('VmHWM:'): hwm = int(ln.split()[1])/1e6
    return cur, hwm

f = sys.argv[1]
m = vhr.read_vhr('VoxHenry/Input_files/'+f)
nvox = int(m.struc().sum())
base, _ = rss()
M = m.build_tree(capacitive=False)
efg = M.e.idx.size + M.f.idx.size + M.g.idx.size
nn = M.lv[0].idx.size
m.prepare(M, float(np.atleast_1d(m.freq)[0]))
a, _ = rss()
t0 = time.perf_counter()
S = pz.LpRSolver(M)
b, _ = rss()
t1 = time.perf_counter()
sn = m.source_vector(M, port=0)
x = S.solve(sn, maxiter=3)
c, hwm = rss()
print("RES %-44s vox=%-7d efg=%-8d nodes=%-7d loops=%-8d "
      "tree=%.3f setup=%.3f solve=%.3f peak=%.3f GB  t_mesh=%.1fs "
      "t_chol=%.1fs t_solve=%.1fs"
      % (f[:44], nvox, efg, nn, S.meshsize, a-base, b-a, c-b, hwm,
         S.t_mesh, S.t_chol, time.perf_counter()-t1))
