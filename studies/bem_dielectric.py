# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Ground-truth dielectric electrostatics on voxel geometry (BEM).

The REFERENCE ORACLE for dielectric phase 2: a FastCap-style
total-charge boundary-integral solve on the exact staircase geometry,
so SuperPEEC's excess-capacitance formulation can be judged against the
true answer WITH fringing -- not against infinite-plate formulas the
finite test geometries cannot reach (that mistake drove two wrong
turns in this program already).

FORMULATION (collocation, one unknown TOTAL charge q per panel):
  * conductor panel i (metal-vacuum or metal-dielectric face):
        sum_j P_ij q_j = V_body(i)          (equipotential)
  * dielectric-vacuum interface panel i (normal n from dielectric
    into vacuum): normal-D continuity across the bound-charge sheet,
        q_i = [2 eps0 (eps_a - eps_b)/(eps_a + eps_b)] * A_i *
              sum_{j != i} En_ij q_j
    with En_ij the free-space normal field at i per unit charge on j
    (the self sheet's +-q/2eps0 jump is what the condition already
    encodes, so self is excluded).
  * conductor-dielectric interfaces carry ONE panel (the conductor's)
    whose q is TOTAL (free + interface bound) charge; the free charge
    a port delivers is D_n = eps_r,outside * q, summed per body.

Kernels: 4x4 subdivided sources, observation at panel centres; the
self potential by fine quadrature. Coplanar neighbours contribute
zero normal field exactly, which collocation preserves.

SELF-CHECKS (run with this file):
  * dielectric-COATED conducting sphere against the exact series
    C = 1/(1/C_shell + 1/C_outer),  C_shell = 4 pi eps0 eps_r /
    (1/a - 1/b), C_outer = 4 pi eps0 b  (staircase-limited, few %).
  * plate ladder (n,n,4), vac/half/fill: the 2-terminal metric to
    compare directly against SuperPEEC part D's readout, and the mutual
    (Maxwell off-diagonal) for reference.

Usage: python3 studies/bem_dielectric.py  (no SuperPEEC imports; pure
numpy + scipy.ndimage for body labelling).

REFINEMENT STUDY (2026-08-08, fixed W/d = 8 shape, plates k thick,
gap 2k, dims (16k,16k,4k), 2-terminal C ratios):
  BEM   k=1..4: half 1.5886/1.5250/1.4974/1.4825  -> limit 1.44
                fill 3.0992/3.0638/3.0509/3.0445  -> limit 3.025
  SuperPEEC k=1,2: half 1.500/1.471 (+4.2%/+2.2% vs limit)
                fill 3.092/2.894 (+2.2%/-4.3%)
  (SuperPEEC k=3 is INVALID as a static probe: (48,48,12) partitions
  MULTILEVEL and the multilevel n2n is the NEAR FIELD ONLY -- the far
  field lives in the FMM ladder. Static node solves are single-level
  trees only.)
CONCLUSION: the excess-capacitance node scheme sits within +-5% of
the converged truth at practical resolutions (1-4 cells across the
dielectric), error sign configuration-dependent and non-monotone in
h. The one-time "wide-plate fill overshoot" (3.795 vs BEM-k=1 3.583
at W/d=24, gap fixed at 2 cells) is this finite-h error growing with
W/d as the ratio approaches eps_r -- an accuracy property, not a
defect. A panel-exact collocation prototype of the same scheme lands
+6% at the same points: the residual is dominated by near-field
coefficient treatment at buried metal-dielectric interfaces, so
tightening it means better interface coefficients, not a new scheme.
"""
import numpy as np
from scipy import ndimage

EPS0 = 8.8541878128e-12
D = 1e-6


def build_panels(ids, eps_r):
    """Panels from a material-id grid (0 empty / 1 conductor / 2
    dielectric). Returns dict of arrays: centre (m), normal, tangent
    axes, kind (0 conductor / 1 dielectric), body (conductor label or
    -1), eps_out (conductor: eps_r of outside medium)."""
    bodies, nb = ndimage.label(ids == 1)
    ctr, nrm, kind, body, eout, cell = [], [], [], [], [], []
    dims = ids.shape
    pad = np.pad(ids, 1)
    for ax in range(3):
        # faces between cell c (index i along ax) and c+e_ax
        lo = pad[tuple(np.s_[1:-1] if a != ax else np.s_[:-1]
                       for a in range(3))]
        hi = pad[tuple(np.s_[1:-1] if a != ax else np.s_[1:]
                       for a in range(3))]
        fx, fy, fz = np.where(lo != hi)
        for x, y, z in zip(fx, fy, fz):
            c = np.array([x + .5, y + .5, z + .5])*D
            c[ax] = [x, y, z][ax]*D          # face plane
            a_id = lo[x, y, z]
            b_id = hi[x, y, z]
            # normal pointing a -> b; cell index of the a-side cell
            # is (x,y,z) - e_ax in grid coords (lo is shifted)
            n = np.zeros(3)
            n[ax] = 1.0
            acell = [x, y, z]
            acell[ax] -= 1
            if a_id == 1 or b_id == 1:
                # conductor face; make normal point OUT of the metal
                if a_id == 1:
                    cc, other = acell, b_id
                else:
                    cc, other = [x, y, z], a_id
                    n = -n
                if other == 1:
                    continue                  # metal-metal: no panel
                kind.append(0)
                body.append(bodies[tuple(cc)] - 1)
                eout.append(eps_r if other == 2 else 1.0)
                cell.append(cc)
            elif 2 in (a_id, b_id) and 0 in (a_id, b_id):
                # dielectric free surface; normal out of dielectric
                if b_id == 2:
                    dc = [x, y, z]
                    n = -n
                else:
                    dc = acell
                kind.append(1)
                body.append(-1)
                eout.append(1.0)
                cell.append(dc)
            else:
                continue                      # 2-2 or 0-0
            ctr.append(c)
            nrm.append(n)
    return dict(ctr=np.array(ctr), nrm=np.array(nrm),
                kind=np.array(kind), body=np.array(body),
                eout=np.array(eout), cell=np.array(cell), nbody=nb)


def rect_VE(ctr, ax, obs):
    """EXACT potential and field of a uniformly charged axis-aligned
    square (side D, total charge 1, centre ``ctr``, normal along
    ``ax``) at points ``obs``. Closed forms (standard rectangle
    antiderivatives, corner-alternating sums):
        V  = (1/4 pi eps0 A) [[ u ln(v+r) + v ln(u+r)
                                - w atan2(u v, w r) ]]
        Ew = (1/4 pi eps0 A) [[ atan2(u v, w r) ]]   (solid angle)
        Eu = (1/4 pi eps0 A) [[ -ln(v+r) ]],  Ev likewise.
    Exact for touching/adjacent panels -- the near-singular geometry
    that broke the subdivided-point kernel. At w = 0 the atan2 corner
    sum is 0 outside the source and +-2 pi inside (only the self
    diagonal, which the caller zeroes)."""
    t = [a for a in range(3) if a != ax]
    u0 = obs[:, t[0]] - ctr[t[0]]
    v0 = obs[:, t[1]] - ctr[t[1]]
    w = obs[:, ax] - ctr[ax]
    h = D/2.0
    V = np.zeros(len(obs))
    Eu = np.zeros(len(obs))
    Ev = np.zeros(len(obs))
    Ew = np.zeros(len(obs))
    # corner loop (signs alternate: sign = su*sv)
    for su in (+1, -1):
        du = u0 + su*h
        for sv in (+1, -1):
            dv = v0 + sv*h
            r = np.sqrt(du*du + dv*dv + w*w)
            s = su*sv
            with np.errstate(divide='ignore', invalid='ignore'):
                lv = np.log(dv + r)
                lu = np.log(du + r)
                # PRINCIPAL arctan, not atan2: the antiderivative's
                # branch must be odd in w (atan2 put every below-plane
                # observation on the wrong branch -- x53 on the bare
                # sphere). 0/0 (w = 0 on a corner line) -> 0 is the
                # correct limit for both V (times w) and Ew (exterior
                # corner sums cancel).
                at = np.nan_to_num(np.arctan(du*dv/(w*r)))
            V += s*(du*lv + dv*lu - w*at)
            Ew += s*at
            Eu += -s*lv
            Ev += -s*lu
    scale = 1.0/(4*np.pi*EPS0*D*D)
    E = np.zeros((len(obs), 3))
    E[:, t[0]] = Eu
    E[:, t[1]] = Ev
    E[:, ax] = Ew
    return V*scale, E*scale


def assemble(p, eps_r):
    npan = len(p['ctr'])
    ctr, nrm = p['ctr'], p['nrm']
    P = np.zeros((npan, npan))
    En = np.zeros((npan, npan))
    for j in range(npan):
        ax = int(np.flatnonzero(np.abs(nrm[j]) > .5)[0])
        V, E = rect_VE(ctr[j], ax, ctr)
        P[:, j] = V
        En[:, j] = np.einsum('ik,ik->i', E, nrm)
    np.fill_diagonal(En, 0.0)
    # rows: conductor -> potential; dielectric -> jump condition
    A = np.where(p['kind'][:, None] == 0, P, -En)
    kfac = 2*EPS0*(eps_r - 1.0)/(eps_r + 1.0)*(D*D)
    dsel = p['kind'] == 1
    A[dsel] *= kfac
    A[dsel, np.flatnonzero(dsel)] = 1.0     # diagonal (En self is 0)
    return A


def _kernel_selftest():
    """Exact kernel vs brute quadrature and the point-charge limit."""
    rng = np.random.default_rng(7)
    ctr = np.zeros(3)
    m = 200
    g = ((np.arange(m) + 0.5)/m - 0.5)*D
    gx, gy = np.meshgrid(g, g)
    src = np.stack([gx.ravel(), gy.ravel(), np.zeros(m*m)], 1)
    worst = 0.0
    for k in range(8):
        ob = (rng.random(3) - 0.5)*4*D
        if abs(ob[2]) < 0.2*D:
            ob[2] = 0.25*D
        if k % 2:
            ob[2] = -abs(ob[2])      # exercise BOTH branches in w
        V, E = rect_VE(ctr, 2, ob[None, :])
        d = ob[None, :] - src
        r = np.linalg.norm(d, axis=1)
        Vq = (1.0/r).mean()/(4*np.pi*EPS0)
        Eq = (d/r[:, None]**3).mean(axis=0)/(4*np.pi*EPS0)
        worst = max(worst, abs(V[0]/Vq - 1),
                    np.abs(E[0] - Eq).max()/np.abs(Eq).max())
    far = np.array([[0.3*D, -0.2*D, 60*D]])
    V, E = rect_VE(ctr, 2, far)
    vref = 1.0/(4*np.pi*EPS0*np.linalg.norm(far))
    worst = max(worst, abs(V[0]/vref - 1))
    print("kernel self-test: worst rel err %.2e (quadrature + far "
          "limit)" % worst, flush=True)
    assert worst < 2e-3


def solve_bodies(ids, eps_r):
    """Maxwell FREE-charge matrix over conductor bodies."""
    p = build_panels(ids, eps_r)
    A = assemble(p, eps_r)
    npan, nb = len(p['ctr']), p['nbody']
    # ROW-EQUILIBRATE: conductor potential rows are O(1/4 pi eps0 D)
    # ~ 1e16 while dielectric jump rows are O(1) -- 16 orders of row
    # spread makes LU lose the dielectric physics to roundoff (the
    # same trap as the validator MNA, see the memory note). Scaling
    # rows preserves the solution exactly.
    rs = 1.0/np.abs(A).max(axis=1)
    lu = np.linalg.inv(A*rs[:, None])
    C = np.zeros((nb, nb))
    for col in range(nb):
        rhs = np.zeros(npan)
        rhs[(p['kind'] == 0) & (p['body'] == col)] = 1.0
        q = lu.dot(rhs*rs)
        qf = p['eout']*q                     # free charge per panel
        for b in range(nb):
            C[b, col] = qf[(p['kind'] == 0) & (p['body'] == b)].sum()
    return C


def twoterm(Cm):
    Ci = np.linalg.inv(Cm)
    return 1.0/(Ci[0, 0] + Ci[1, 1] - 2*Ci[0, 1])


def sphere_check(eps_r=4.0, a_c=6.5, b_c=10.5, n=24):
    g = np.arange(n) + 0.5 - n/2.0
    x, y, z = np.meshgrid(g, g, g, indexing='ij')
    r = np.sqrt(x*x + y*y + z*z)
    ids = np.zeros((n, n, n), dtype=int)
    ids[r < b_c] = 2
    ids[r < a_c] = 1
    C = solve_bodies(ids, eps_r)[0, 0]
    a, b = a_c*D, b_c*D
    c_shell = 4*np.pi*EPS0*eps_r/(1.0/a - 1.0/b)
    c_outer = 4*np.pi*EPS0*b
    c_ref = 1.0/(1.0/c_shell + 1.0/c_outer)
    # staircase sphere: effective radius of the voxelized ball is a
    # few % above the nominal; the vacuum control below cancels it
    ids_v = np.where(ids == 2, 0, ids)
    C_v = solve_bodies(ids_v, eps_r)[0, 0]
    c_ref_v = 4*np.pi*EPS0*a
    print("sphere: coated C %.4g (analytic %.4g, x%.3f) | bare C %.4g "
          "(analytic %.4g, x%.3f) | RATIO coated/bare %.4f vs analytic "
          "%.4f (staircase cancels)"
          % (C, c_ref, C/c_ref, C_v, c_ref_v, C_v/c_ref_v,
             C/C_v, c_ref/c_ref_v), flush=True)


def plate_ladder(eps_r=4.0, sizes=(16, 24, 32, 48)):
    print("plate ladder (n,n,4), 2-terminal metric (SuperPEEC part D's "
          "readout) -- SuperPEEC measured: half 1.062/1.056/1.050/1.041, "
          "fill 3.110/3.398/3.580/3.800")
    for n in sizes:
        dims = (n, n, 4)
        base = np.zeros(dims, dtype=int)
        base[:, :, 0] = 1
        base[:, :, 3] = 1
        vac = base.copy()
        half = base.copy()
        half[:, :, 1] = 2
        fill = base.copy()
        fill[:, :, 1] = 2
        fill[:, :, 2] = 2
        Cv = solve_bodies(vac, eps_r)
        Ch = solve_bodies(half, eps_r)
        Cf = solve_bodies(fill, eps_r)
        print("n=%2d: C2t vac %.5g | half ratio %.4f (series 1.600) | "
              "fill ratio %.4f (ideal 4) | mutual ratios half %.4f "
              "fill %.4f"
              % (n, twoterm(Cv), twoterm(Ch)/twoterm(Cv),
                 twoterm(Cf)/twoterm(Cv),
                 -Ch[0, 1]/-Cv[0, 1], -Cf[0, 1]/-Cv[0, 1]), flush=True)


if __name__ == '__main__':
    _kernel_selftest()
    sphere_check()
    plate_ladder()
