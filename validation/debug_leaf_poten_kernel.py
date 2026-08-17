# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Per-offset comparator for the capacitive FFT kernel (stage-2 tool).

Reconstructs the spatial circulant C = ifftn(p2p_trans_*[neigh]) for one
neighbour block and prints it against the VALIDATED p2pinit3 oracle at
each index offset, so a wrong gather shows up as a shape mismatch and a
wrong normalisation shows up as a constant ratio. Used to find the
transverse circulant-origin bug (ramp instead of symmetric kernel).

Run inside the toolbox:  python3 debug_leaf_poten_kernel.py
"""
import numpy as np, multipole as mp
CELL=1e-5
fs=np.ones((4,4,4),np.int8); fs[3,3,3]=0
M=mp.Tree(fs,np.array([2,2,2]),np.array([4,4,4])*CELL,2,1e0,2,capacitive=True)
px=M.px
ref=dict(zip('xyz',[B.tocsr() for B in px.p2pinit3(M.px,M.py,M.pz)]))['x']
px.p2pinit()
C=np.fft.ifftn(px.p2p_trans_x[13])          # spatial circulant, self block
n=tuple(int(v) for v in px.n); sz=C.shape
print("px.n=",n," circulant sz=",sz)
g=0; lo,hi=int(px.idx0[g]),int(px.idx0[g+1])
loc=np.array([np.unravel_index(int(px.idx[k]),n) for k in range(lo,hi)])
A=ref[lo:hi,lo:hi].toarray()                # P[src,tgt] within group 0
rows=[]
for a in range(len(loc)):
    for b in range(len(loc)):
        d=tuple((loc[b]-loc[a]))
        rows.append((d, A[a,b], C[tuple(np.array(d)%np.array(sz))]))
seen={}
for d,o,c in rows: seen.setdefault(d,(o,c))
print("\n offset      oracle P[s,t]        circulant C[d]      ratio")
for d in sorted(seen)[:14]:
    o,c=seen[d]
    print("  %-11s %-19.6e %-19.6e %s"%(str(d),o.real,c.real,
          ("%.4f"%(o.real/c.real)) if abs(c.real)>1e-30 else "c=0"))
vals=np.array([(o.real,c.real) for d,(o,c) in seen.items() if abs(c.real)>1e-30])
if len(vals):
    r=vals[:,0]/vals[:,1]
    print("\n ratio over %d offsets: min=%.4f max=%.4f median=%.4f"
          %(len(r),r.min(),r.max(),np.median(r)))
