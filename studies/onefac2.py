# SPDX-License-Identifier: MIT
"""ONE factorization, no F.L() materialization. Clean peak-RSS datapoint."""
import sys, time, numpy as np, vhr, meshgraph as mg
from sksparse import cholmod

def hwm():
    for ln in open('/proc/self/status'):
        if ln.startswith('VmHWM:'): return int(ln.split()[1])/1e6
    return 0.0

f, om = sys.argv[1], sys.argv[2]
m = vhr.read_vhr('VoxHenry/Input_files/'+f)
M = m.build_tree(capacitive=False)
m.prepare(M, float(np.atleast_1d(m.freq)[0]))
es, fs_, gs = np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc)
efg = es+fs_+gs; nn = np.size(M.lv[0].struc)
h_tree = hwm()
Y = mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn)
Y.data = np.float64(Y.data)
YT = Y.T.tocsc(); YT.data = np.float32(YT.data)
h_mesh = hwm()
t = time.perf_counter()
F = cholmod.cholesky_AAt(YT, mode=sys.argv[3], ordering_method=om)
dt = time.perf_counter()-t
print("RES %-26s %-6s %-11s loops=%-8d efg=%-8d peak: tree=%.2f mesh=%.2f "
      "chol=%.2f GB  t_chol=%.1fs"
      % (f[:26], om, sys.argv[3], Y.shape[1], efg, h_tree, h_mesh, hwm(), dt), flush=True)
