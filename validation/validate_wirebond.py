# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for WireBondSolver -- stage A2: wires galvanically in circuit.

Geometry: two copper pads separated by a 4-cell gap, bridged by three
aluminium bond wires (arcs over the gap), driven by a port strip on
each pad's far edge. All drive current crosses through the wires, so
sharing, feet and the bulk return are all load-bearing.

FOUR INDEPENDENT ANGLES, because A2 adds graph plumbing that values
alone cannot vouch for:
  B  dense oracle of the ENTIRE system (exact Lp/C/wire blocks + foot
     term, same basis, direct solve) -- the value gate.
  C  DC NODAL oracle: the same resistive network solved as a sparse
     node-potential problem -- a different FORMULATION entirely, so it
     cross-checks paths, feet and KCL, not just kernels.
  D  ROUTE INDEPENDENCE: rebuild with a rotated spanning-forest root;
     the lattice legs of every cycle change, the answer must not --
     this is the mesh-analysis gauge invariance, and it fails loudly
     if the basis is incomplete or any path sign is wrong.
  E  symmetry physics: wires A and C mirror each other about the pad
     centreline, so their shares must agree at every frequency.

Run: PYTHONPATH=src python3 validate_wirebond.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np

import terminal as tm
import voxmodel
import wirekernel as wk
from wireassembly import Wire, WireBondSolver

FAIL = []
FREQ = 1e9
D = 1e-6
SIG_CU = 5.8e7
SIG_AL = 3.77e7


def check(name, ok, detail=""):
    print("  %-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def build_model():
    # pad width ODD (9 cells) so the centreline y = 4.5 um is a CELL
    # CENTRE: the middle wire's contacts land on it with no foot-cell
    # tie, and wires A/C mirror exactly. The first two geometries here
    # were asymmetric for two different reasons -- wrongly mirrored
    # positions (1.5/3.5/5.5), then a middle wire on a cell BOUNDARY
    # whose foot was tie-broken half a cell off (the 9e-4 share
    # asymmetry that survived at DC and exonerated the section seams).
    m = voxmodel.VoxelModel('wirebond_gate')
    m.dims = (24, 9, 3)
    m.d = D
    m.sigma = np.zeros(m.dims)
    m.sigma[0:10, :, 0:2] = SIG_CU      # pad 1
    m.sigma[14:24, :, 0:2] = SIG_CU     # pad 2
    m.freq = np.array([FREQ])
    return m, m.build_tree(nleaf=[4, 9, 3], numlevels=2)


def build_wires(delta):
    a = 0.25e-6
    out = []
    for y in (1.5, 4.5, 7.5):           # cell centres, mirror pair
        pts = np.array([[9.3, y, 2.4], [10.5, y, 4.0],
                        [13.5, y, 4.0], [14.7, y, 2.4]])*D
        out.append(Wire(pts, a, SIG_AL, delta=delta,
                        max_seglen=2.5*D))
    return out


PORT_P = [(0, y, 1) for y in range(2, 7)]   # [2, 7): centred on 4.5
PORT_N = [(23, y, 1) for y in range(2, 7)]


def dense_reference(sol, freq):
    """Dense exact assembly of the ENTIRE A2 system, same basis."""
    m, M, wc = sol.model, sol.M, sol.wc
    m.prepare(M, freq)
    jw = M.jomega
    efg, nwel = sol.efg, sol.nwel
    l = wc.l
    Z = np.zeros((efg + nwel, efg + nwel), dtype=np.complex128)
    for leaf, axis, off in wc.leaves:
        size = np.size(leaf.idx)
        cells = wc.fil_cell[off:off + size]
        lo = cells*l[None, :]
        lo[:, axis] += 0.5*l[axis]
        hi = lo + l[None, :]
        Z[off:off + size, off:off + size] += \
            jw*tm.box_mutual_matrix(lo, hi, axis)
        r = np.asarray(leaf.r, dtype=float)*np.ones(size)
        Z[off:off + size, off:off + size] += np.diag(r)
    Cd = np.zeros((nwel, efg))
    for s, fs in enumerate(wc.segments):
        for leaf, axis, off in wc.leaves:
            if abs(float(fs[0].u[axis])) < 1e-14:
                continue
            size = np.size(leaf.idx)
            Cd[wc.seg0[s]:wc.seg0[s + 1], off:off + size] = \
                wk.mutual_voxels(fs, wc.fil_cell[off:off + size], l, axis)
    Z[efg:, :efg] += jw*Cd
    Z[:efg, efg:] += jw*Cd.T
    Lw = np.zeros((nwel, nwel))
    for s1 in range(len(wc.segments)):
        a1, b1 = wc.seg0[s1], wc.seg0[s1 + 1]
        Lw[a1:b1, a1:b1] = wk.segment_self_block(wc.segments[s1])
        for s2 in range(s1 + 1, len(wc.segments)):
            a2, b2 = wc.seg0[s2], wc.seg0[s2 + 1]
            B = wk.mutual_block(wc.segments[s1], wc.segments[s2])
            Lw[a1:b1, a2:b2] = B
            Lw[a2:b2, a1:b1] = B.T
    Af = sol.Afoot.toarray()
    Z[efg:, efg:] += (jw*Lw + np.diag(sol.r_w)
                      + Af.T @ np.diag(sol.Rfoot) @ Af)
    Bfull = sol.Bmat.toarray()
    ihat = np.concatenate([sol.ihat_f, sol.ihat_w])
    A = Bfull.T @ (Z @ Bfull)
    rhs = -(Bfull.T @ (Z @ ihat))
    x = np.linalg.solve(A, rhs)
    i = ihat + Bfull @ x
    v = Z @ i
    V = complex(ihat @ v)
    share = sol.Afoot @ i[efg:]
    return V, share


def dc_nodal_oracle(sol, current=1.0):
    """The same resistive network as a NODE-POTENTIAL solve -- shares
    no formulation with the mesh solver."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    m, M = sol.model, sol.M
    r_f = np.concatenate([np.asarray(leaf.r, dtype=float)
                          * np.ones(np.size(leaf.idx))
                          for leaf, _, _ in sol.wc.leaves])
    B = sol.B
    G = (B.T @ sp.diags(1.0/r_f) @ B).tocsr()
    gw = []
    for j, w in enumerate(sol.wires):
        Rch = 0.0
        for s in np.where(sol.wire_of_seg == j)[0]:
            a, b = sol.wc.seg0[s], sol.wc.seg0[s + 1]
            Rch += 1.0/np.sum(1.0/sol.r_w[a:b])
        gw.append(1.0/(Rch + sol.Rfoot[j]))
    G = G + sol.Bw @ sp.diags(gw) @ sol.Bw.T
    s = np.zeros(sol.nnode)
    for n in sol.port_p:
        s[n] += current/len(sol.port_p)
    for n in sol.port_n:
        s[n] -= current/len(sol.port_n)
    # ground one node; the wire edges make the graph one component
    Gg = G.tolil()
    Gg[0, :] = 0.0
    Gg[0, 0] = 1.0
    rhs = s.copy()
    rhs[0] = 0.0
    phi = spla.spsolve(Gg.tocsr(), rhs)
    V = float(s @ phi)/current
    # wire shares from the potential drop across each wire edge
    share = np.array([gw[j]*(phi[sol.foot_node[j][0]]
                             - phi[sol.foot_node[j][1]])
                      for j in range(len(sol.wires))])/current
    return V, share


if __name__ == '__main__':
    # the doctrine formula, so part F's wires match bit-for-bit
    delta = np.sqrt(1.0/(np.pi*FREQ*wk.MU0*SIG_AL))
    m, M = build_model()
    m.prepare(M, FREQ)
    wires = build_wires(delta)
    sol = WireBondSolver(m, M, wires, PORT_P, PORT_N, verbose=True)

    print("\nPART A -- the galvanic solve runs, converges, conserves")
    Z, info = sol.solve(FREQ, rtol=1e-10, verbose=True)
    sh = np.real(info['share'])
    check("lgmres converged", info['flag'] == 0
          and info['residual'] < 1e-8,
          "flag %s resid %.2e" % (info['flag'], info['residual']))
    check("all drive current crosses via the wires",
          abs(sh.sum() - 1) < 1e-8, "sum %.8f" % sh.sum())
    check("passivity: Re Z > 0", Z.real > 0, "Z = %s" % Z)
    check("driven route + 2 sharing chords derived",
          sol.nchord == 2, "%d chords" % sol.nchord)

    print("\nPART B -- dense oracle of the ENTIRE A2 system")
    Vd, shd = dense_reference(sol, FREQ)
    relZ = abs(Z - Vd)/abs(Vd)
    relsh = np.abs(info['share'] - shd).max()
    print("    solver Z = %.6e%+.6ej" % (Z.real, Z.imag))
    print("    oracle Z = %.6e%+.6ej" % (Vd.real, Vd.imag))
    check("Z_port matches dense oracle to 1%", relZ < 1e-2,
          "rel %.2e" % relZ)
    check("shares match dense oracle to 1e-3", relsh < 1e-3,
          "max %.2e" % relsh)

    print("\nPART C -- DC limit vs the independent NODAL oracle")
    wires_dc = build_wires(None)
    m2, M2 = build_model()
    m2.prepare(M2, 10.0)
    sol2 = WireBondSolver(m2, M2, wires_dc, PORT_P, PORT_N)
    Z2, info2 = sol2.solve(10.0, rtol=1e-10)
    Vdc, shdc = dc_nodal_oracle(sol2)
    rel = abs(Z2.real - Vdc)/Vdc
    check("Re Z(10 Hz) == nodal-network solve", rel < 1e-6,
          "%.8e vs %.8e (rel %.2e)" % (Z2.real, Vdc, rel))
    check("DC shares == nodal-network shares",
          np.abs(np.real(info2['share']) - shdc).max() < 1e-6,
          "max %.2e" % np.abs(np.real(info2['share']) - shdc).max())

    print("\nPART D -- route (gauge) independence of the answer")
    sol3 = WireBondSolver(m, M, wires, PORT_P, PORT_N,
                          root0=sol.nnode//3)
    Z3, info3 = sol3.solve(FREQ, rtol=1e-10)
    rel = abs(Z3 - Z)/abs(Z)
    check("rotated spanning-forest root: Z unchanged", rel < 1e-7,
          "rel %.2e" % rel)
    check("... and shares unchanged",
          np.abs(info3['share'] - info['share']).max() < 1e-7,
          "max %.2e" % np.abs(info3['share'] - info['share']).max())

    print("\nPART E -- symmetry physics")
    check("mirror wires A and C share equally",
          abs(sh[0] - sh[2]) < 1e-4,
          "A %.5f  B %.5f  C %.5f" % tuple(sh))

    print("\nPART F -- the input file rebuilds the same problem")
    import sppeec_input
    toml = ['[grid]', 'dims = [24, 9, 3]', 'pitch = 1e-6',
            '[[block]]', 'from = [0, 0, 0]', 'to = [10, 9, 2]',
            'sigma = 5.8e7',
            '[[block]]', 'from = [14, 0, 0]', 'to = [24, 9, 2]',
            'sigma = 5.8e7']
    for y in (1.5, 4.5, 7.5):
        toml += ['[[wire]]',
                 'points = [[9.3e-6, %ge-6, 2.4e-6],'
                 ' [10.5e-6, %ge-6, 4.0e-6],'
                 ' [13.5e-6, %ge-6, 4.0e-6],'
                 ' [14.7e-6, %ge-6, 2.4e-6]]' % (y, y, y, y),
                 'radius = 0.25e-6', 'sigma = 3.77e7',
                 'max_seglen = 2.5e-6']
    toml += ['[port]',
             'p_cells = [%s]' % ', '.join('[0, %d, 1]' % y
                                          for y in range(2, 7)),
             'n_cells = [%s]' % ', '.join('[23, %d, 1]' % y
                                          for y in range(2, 7)),
             '[solve]', 'freq = [1e9]']
    prob = sppeec_input.loads('\n'.join(toml))
    mF = prob.model()
    check("reader model struc == direct model struc",
          np.array_equal(mF.struc(), m.struc()), "")
    wF = prob.wires(FREQ)
    same = len(wF) == len(wires)
    for wa, wb in zip(wF, wires):
        same &= len(wa.segments) == len(wb.segments)
        for sa, sb in zip(wa.segments, wb.segments):
            same &= (np.allclose(sa[0].p0, sb[0].p0)
                     and np.allclose(sa[0].u, sb[0].u)
                     and abs(sa[0].length - sb[0].length) < 1e-15
                     and len(sa) == len(sb)
                     and np.allclose(sa[0].shape, sb[0].shape))
    check("reader wires == direct wires (geometry + shapes)", same, "")
    check("reader ports == direct ports",
          prob.port_p == PORT_P and prob.port_n == PORT_N, "")
    # typo protection: the doctrine's unknown-key rejection has teeth
    try:
        sppeec_input.loads('[grid]\ndims = [2,2,2]\npitch = 1e-6\n'
                           '[[wire]]\npoints = [[0,0,0],[1e-6,0,0]]\n'
                           'radiius = 1e-7\nsigma = 3e7')
        check("unknown key rejected", False, "no exception")
    except ValueError as e:
        check("unknown key rejected", 'radiius' in str(e), "")

    print("\nPART G -- basis='overcomplete': BlockAMG + Schur == "
          "selected + Cholesky")
    solG = WireBondSolver(m, M, wires, PORT_P, PORT_N,
                          basis='overcomplete', verbose=True)
    ZG, infoG = solG.solve(FREQ, rtol=1e-10)
    relZ = abs(ZG - Z)/abs(Z)
    relsh = np.abs(infoG['share'] - info['share']).max()
    check("overcomplete converged", infoG['flag'] == 0
          and infoG['residual'] < 1e-8,
          "flag %s resid %.2e, %d mv (selected: %d)"
          % (infoG['flag'], infoG['residual'], infoG['matvecs'],
             info['matvecs']))
    check("Z identical across bases", relZ < 1e-6, "rel %.2e" % relZ)
    check("shares identical across bases", relsh < 1e-6,
          "max %.2e" % relsh)
    check("macro block is exactly the sharing chords",
          solG.chol.nmac == solG.nchord,
          "%d vs %d" % (solG.chol.nmac, solG.nchord))
    check("AMG hierarchy is O(N)-flat (< 1.5x nnz)",
          solG.chol.nnz_ratio < 1.5, "%.2fx" % solG.chol.nnz_ratio)

    print("\n%d checks failed" % len(FAIL))
    raise SystemExit(1 if FAIL else 0)
