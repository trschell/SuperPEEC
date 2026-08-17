# SPDX-License-Identifier: MIT
# Is the cell-scheme panel far-field error a WIRING bug or the point-charge
# approximation of a whole-face panel?  Build the point-charge far field
# explicitly and compare it to (a) the FMM far field, (b) the exact far field.
import numpy as np, multipole as mp, stencils as st

CELL = 1e-5
NT = np.array([15, 15, 15])
fs = np.ones(NT, dtype=np.int8)
mu0 = 4*np.pi*1e-7; c0 = 299792458.0; eps0 = 1/(mu0*c0**2)
M0 = 1/(4*np.pi*eps0)


def coords(idx, dims):
    return np.stack([idx//(dims[1]*dims[2]), (idx//dims[2]) % dims[1],
                     idx % dims[2]], 1)


def nodemap(M, key1):
    n = M.lv[0].n.astype(int); glob = []
    for g in range(np.size(M.lv[0].idx0)-1):
        f = np.r_[M.lv[0].idx0[g]:M.lv[0].idx0[g+1]]
        c = coords(M.lv[0].idx[f], n)
        c[:, 0] += M.lv[0].xidx[g]*n[0]
        c[:, 1] += M.lv[0].yidx[g]*n[1]
        c[:, 2] += M.lv[0].zidx[g]*n[2]
        glob.append(c)
    glob = np.concatenate(glob)
    key = glob[:, 0]*100000 + glob[:, 1]*1000 + glob[:, 2]
    lk = {k: i for i, k in enumerate(key1)}
    return np.array([lk[k] for k in key])


def panel_geom(leaf, axis):
    """Global centre and box index of every occupied panel in `leaf`,
    using leafinit's OWN family convention."""
    n = leaf.n.astype(int); l = np.asarray(leaf.l, float)
    ctr = [np.arange(-(n[i]-1)/2, (n[i]-1)/2+1) for i in range(3)]
    nod = [np.arange(-(n[i]-1)/2-0.5, (n[i]-1)/2+0.5) for i in range(3)]
    panel = True
    off = False
    fam = [ctr[i] if ((i == axis) != off) else nod[i] for i in range(3)]
    g = np.meshgrid(fam[0], fam[1], fam[2], indexing='ij')
    loc = np.stack([l[0]*g[0].ravel(), l[1]*g[1].ravel(), l[2]*g[2].ravel()], 1)
    pos = np.zeros((leaf.idx.size, 3)); box = np.zeros((leaf.idx.size, 3), int)
    for gi in range(np.size(leaf.idx0)-1):
        f = np.r_[leaf.idx0[gi]:leaf.idx0[gi+1]]
        if f.size == 0:
            continue
        b = np.array([leaf.xidx[gi], leaf.yidx[gi], leaf.zidx[gi]])
        pos[f] = loc[leaf.idx[f]] + b*n*l
        box[f] = b
    return pos, box


def expand(pos, box, q, axis, l, nq):
    """Replace each panel by an nq x nq midpoint quadrature of its area."""
    t = [i for i in range(3) if i != axis]
    u = (np.arange(nq)+0.5)/nq - 0.5
    du = np.zeros((nq*nq, 3))
    g = np.meshgrid(u, u, indexing='ij')
    du[:, t[0]] = l[t[0]]*g[0].ravel(); du[:, t[1]] = l[t[1]]*g[1].ravel()
    P = (pos[:, None, :] + du[None, :, :]).reshape(-1, 3)
    B = np.repeat(box, nq*nq, axis=0)
    Q = np.repeat(q, nq*nq)/(nq*nq)
    return P, B, Q


def point_far(pos, box, q, chunk=400):
    """phi_t = M0 * sum_{s : box-dist>=2} q_s / |r_t - r_s|"""
    phi = np.zeros(pos.shape[0], dtype=q.dtype)
    for a in range(0, pos.shape[0], chunk):
        b = min(a+chunk, pos.shape[0])
        d = pos[a:b, None, :] - pos[None, :, :]
        r = np.sqrt((d*d).sum(2))
        far = (np.abs(box[a:b, None, :] - box[None, :, :]).max(2) >= 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            k = np.where(far, 1.0/r, 0.0)
        phi[a:b] = M0*(k @ q)
    return phi


for nleaf in (4, 5):
    probe = mp.Tree(fs, np.array([nleaf]*3), NT*CELL, 2, 1e0, 4,
                    capacitive=True)
    fac = CELL/np.asarray(probe.e.l, float)
    M = mp.Tree(fs, np.array([nleaf]*3), NT*CELL*fac, 2, 1e0, 8,
                capacitive=True)
    assert np.all(np.abs(np.asarray(M.e.l)-CELL)/CELL < 1e-9)
    M1 = mp.Tree(fs, NT, NT*CELL, 1, 1e0, None, capacitive=True)
    c1 = coords(M1.lv[0].idx, np.asarray(M1.ntotal, dtype=int))
    key1 = c1[:, 0]*100000 + c1[:, 1]*1000 + c1[:, 2]
    perm = nodemap(M, key1)

    rng = np.random.default_rng(7)
    q1 = rng.standard_normal(M1.lv[0].idx.size)*1e-12
    M1.lv[0].data = q1.astype(np.complex128)
    M1.traverseP3()
    exact = M1.lv[0].data.real.copy()[perm]

    q = q1[perm].astype(np.complex128)
    near = np.asarray(M.n2n.dot(q)).ravel().real
    M.lv[0].data = q.copy()
    M.traverseP3()
    fmm_far = M.lv[0].data.real.copy() - near
    ex_far = exact - near

    # point-charge far field, panels collapsed to their centres
    qp = [M.node2px.dot(q).real, M.node2py.dot(q).real, M.node2pz.dot(q).real]
    P, B = zip(*[panel_geom(L, a) for a, L in
                 enumerate((M.px, M.py, M.pz))])
    pos = np.concatenate(P); box = np.concatenate(B)
    qall = np.concatenate(qp)
    nrm = np.linalg.norm(ex_far)
    print(" nleaf=%d  |far|/|total|=%.3f" % (nleaf, nrm/np.linalg.norm(exact)))
    print("   FMM far   vs EXACT far : %.4e" % (np.linalg.norm(fmm_far-ex_far)/nrm))
    for nq in (1, 2, 3, 4):
        EP, EB, EQ = zip(*[expand(P[a], B[a], qp[a], a,
                                  np.asarray(L.l, float), nq)
                           for a, L in enumerate((M.px, M.py, M.pz))])
        ep = np.concatenate(EP); eb = np.concatenate(EB)
        eq = np.concatenate(EQ)
        phi = point_far(ep, eb, eq, chunk=200)
        phi = phi.reshape(-1, nq*nq).mean(1)   # target-side area average
        o = 0; pt_far = np.zeros(q.size)
        for L, n2p in ((M.px, M.node2px), (M.py, M.node2py),
                       (M.pz, M.node2pz)):
            k = L.idx.size
            pt_far += n2p.T.dot(phi[o:o+k]); o += k
        print("   QUAD%dx%d far vs EXACT far : %.4e   (vs FMM %.4e)"
              % (nq, nq, np.linalg.norm(pt_far-ex_far)/nrm,
                 np.linalg.norm(pt_far-fmm_far)/nrm))
