# SPDX-License-Identifier: MIT
"""Re-run the nleaf sweep with the CELL PITCH PINNED (the documented recipe)."""
import numpy as np, multipole as mp, stencils as st
CELL=1e-5; NT=np.array([15,15,15]); fs=np.ones(NT,dtype=np.int8)
def coords(idx,dims):
    return np.stack([idx//(dims[1]*dims[2]),(idx//dims[2])%dims[1],idx%dims[2]],1)
M1=mp.Tree(fs,NT,NT*CELL,1,1e0,None,capacitive=True)
assert abs(M1.e.l[0]-CELL)/CELL < 1e-12, "oracle pitch off"
c1=coords(M1.lv[0].idx,np.asarray(M1.ntotal,dtype=int))
key1=c1[:,0]*1000000+c1[:,1]*1000+c1[:,2]
nn=M1.lv[0].idx.size
rng=np.random.default_rng(7)
q1=(rng.standard_normal(nn)+1j*rng.standard_normal(nn))*1e-12
M1.lv[0].data=q1.copy(); M1.traverseP3(); phi1=M1.lv[0].data.copy()
lk={k:i for i,k in enumerate(key1)}
print(" (oracle pitch %.4g)" % (M1.e.l[0],))
print(" RES %-6s %-9s %-14s %s" % ('nleaf','pitch/CELL','occ/tot boxes','rel_err'))
for nleaf in (2,3,4,5,6):
    probe=mp.Tree(fs,np.array([nleaf]*3),NT*CELL,2,1e0,4,capacitive=True)
    fac=CELL/np.asarray(probe.e.l,dtype=float)          # per-axis pin
    M=mp.Tree(fs,np.array([nleaf]*3),NT*CELL*fac,2,1e0,8,capacitive=True)
    assert np.all(np.abs(np.asarray(M.e.l)-CELL)/CELL < 1e-9), "pitch not pinned"
    lv0=M.lv[0]; i0=np.asarray(lv0.idx0,dtype=np.int64); n=lv0.n.astype(int)
    glob=[]; occ=sum(1 for g in range(i0.size-1) if i0[g+1]>i0[g])
    for g in range(i0.size-1):
        fi=np.r_[i0[g]:i0[g+1]]
        c=coords(lv0.idx[fi],n)
        c[:,0]+=lv0.xidx[g]*n[0]; c[:,1]+=lv0.yidx[g]*n[1]; c[:,2]+=lv0.zidx[g]*n[2]
        glob.append(c)
    glob=np.concatenate(glob)
    perm=np.array([lk[k] for k in glob[:,0]*1000000+glob[:,1]*1000+glob[:,2]])
    lv0.data=q1[perm].copy()
    M.traverseP3(); phi=lv0.data.copy()
    err=np.linalg.norm(phi-phi1[perm])/np.linalg.norm(phi1[perm])
    print(" RES %-6d %-9.6f %-14s %.4e"
          % (nleaf, M.e.l[0]/CELL, "%d/%d"%(occ,i0.size-1), err))
