# SPDX-License-Identifier: MIT
"""Does the CHOLMOD ordering change the fill? AMD (shipped) vs nested dissection."""
import sys, time, numpy as np, vhr, port_impedance as pz, meshgraph as mg
from sksparse import cholmod

def rss():
    for ln in open('/proc/self/status'):
        if ln.startswith('VmRSS:'): cur = int(ln.split()[1])/1e6
    return cur

f = sys.argv[1]
m = vhr.read_vhr('VoxHenry/Input_files/'+f)
M = m.build_tree(capacitive=False)
m.prepare(M, float(np.atleast_1d(m.freq)[0]))
es, fs_, gs = (np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc))
efg = es+fs_+gs; nn = np.size(M.lv[0].struc)
Y = mg.getmesh_fortran(M.adjmats(), es, es+fs_, efg, nn)
Y.data = np.float64(Y.data)
YT = Y.T.tocsc(); YT.data = np.float32(YT.data)
print("  loops=%d  nnz(Y)=%d" % (Y.shape[1], Y.nnz))
for om in ('amd', 'nesdis', 'metis', 'best'):
    try:
        a = rss(); t = time.perf_counter()
        F = cholmod.cholesky_AAt(YT, mode='supernodal', ordering_method=om)
        dt = time.perf_counter()-t; b = rss()
        nzL = F.L().nnz
        print("RES %-8s %-6s t=%7.1fs  dRSS=%6.3f GB  nnz(L)=%d"
              % (f[:8], om, dt, b-a, nzL))
        del F
    except Exception as e:
        print("RES %-8s %-6s FAILED %s: %s" % (f[:8], om, type(e).__name__, str(e)[:50]))
