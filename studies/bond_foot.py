# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""THE BOND FOOT: what does attaching a 1-D wire to a 3-D mesh cost?

The wire model is now good to ~0.6% worst case over orientation. That
number is only meaningful if the FOOT -- where the filament chain lands
on the pad -- is no worse. Nothing so far has measured it, and it is
the obvious candidate for the dominant error in an assembled module.

Structurally this is the SAME problem as the port terminal measured
earlier in this project: a lower-dimensional drive meeting a
3-D conductor. That work found the answer matters a lot (a solved
equipotential split moved R by 20% against a prescribed profile) and
that a point drive has a constriction error which does NOT converge
under refinement.

THE EXACT REFERENCE. Current entering a half-space through a circular
contact of radius r0 has the classic constriction (spreading)
resistance

    R_spread = rho / (4 r0)

so a FIXED PHYSICAL footprint must converge to it as the mesh refines,
while a SINGLE-NODE attachment -- whose effective radius shrinks with
the cell -- must diverge like rho/(4 * d/2) ~ 1/d. That divergence is
the thing to avoid, and this file measures both.

No FMM and no wirekernel needed: at DC the voxel block is just a
resistive network, which isolates the foot from every other error in
the model.

Run: PYTHONPATH=src python3 studies/bond_foot.py
Env: N (block cells per side, 48), R0 (footprint radius in microns, 0
     sweeps a set), DX (cell size in microns, 0 sweeps refinement)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SIGMA = 5.8e7


def spreading_resistance(n, dx, r0, sigma=SIGMA, mode='uniform'):
    """NOTE n and dx are BOTH given: the caller must hold the PHYSICAL
    block size n*dx fixed while refining dx. Refining dx at fixed n
    shrinks the domain instead of the mesh -- measured on the first run
    of this file, where a 40 um contact ended up filling a third of a
    120 um block and the 'spreading resistance' fell 21% -> 72% below
    the half-space value. Same class of error as the shrinking port in
    the h-refinement study."""
    """R from a circular contact on the top face of an n^3 block to the
    whole bottom face, minus the 1-D bulk term.

    Returns (R_total, R_spread). The contact is every top-face cell
    whose centre lies within r0 of the axis; r0 <= dx/2 gives the
    single-node (point) attachment.
    """
    idx = np.arange(n**3).reshape(n, n, n)
    g = sigma*dx            # conductance between adjacent cell centres
    rows, cols, vals = [], [], []
    for ax in range(3):
        a = np.take(idx, np.arange(n - 1), axis=ax).ravel()
        b = np.take(idx, np.arange(1, n), axis=ax).ravel()
        rows += [a, b, a, b]
        cols += [a, b, b, a]
        vals += [np.full(a.size, g), np.full(a.size, g),
                 np.full(a.size, -g), np.full(a.size, -g)]
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n**3, n**3)).tocsr()
    c = (n - 1)/2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c,
                         indexing='ij')
    rr = np.sqrt(xx**2 + yy**2)*dx
    top = idx[:, :, n - 1]
    if r0 <= 0:                      # single-node (point) attachment
        contact = np.array([top.ravel()[np.argmin(rr)]])
    else:
        contact = top[rr <= r0]
        if contact.size == 0:
            contact = np.array([top.ravel()[np.argmin(rr)]])
    ground = idx[:, :, 0].ravel()
    I = 1.0
    b = np.zeros(n**3)
    b[contact] = I/contact.size
    # ground the bottom face by pinning it (Dirichlet), which is the
    # return electrode
    keep = np.setdiff1d(np.arange(n**3), ground)
    Ak = A[keep][:, keep].tocsr()
    if mode == 'equipotential':
        # MERGE the contact nodes into one. A real bond is a metal
        # contact: equipotential, with the current split SOLVED rather
        # than prescribed. This is the same distinction the port work
        # measured at 20% in R, and it is also what the classic
        # rho/(4 r0) assumes -- an equipotential disc, NOT uniform
        # current density (whose exact value is 8 rho/(3 pi^2 r0),
        # about 8% higher).
        pos = {int(v): i for i, v in enumerate(keep)}
        cset = set(int(c) for c in contact)
        col, nxt = np.empty(keep.size, dtype=int), 1
        for i, v in enumerate(keep):
            if int(v) in cset:
                col[i] = 0
            else:
                col[i] = nxt
                nxt += 1
        P = sp.coo_matrix((np.ones(keep.size),
                           (np.arange(keep.size), col)),
                          shape=(keep.size, nxt)).tocsr()
        A2 = (P.T @ Ak @ P).tocsr()
        b2 = np.zeros(nxt)
        b2[0] = I
        d2 = A2.diagonal()
        M2 = spla.LinearOperator(A2.shape, lambda x: x/d2)
        sol2, info2 = spla.cg(A2, b2, rtol=1e-12, maxiter=50000, M=M2)
        if info2 != 0:
            raise RuntimeError("cg (equipotential) info=%d" % info2)
        R_tot = float(sol2[0])
        R_bulk = ((n - 1)*dx)/(sigma*(n*dx)**2)
        return R_tot, R_tot - R_bulk, contact.size
    d_ = Ak.diagonal()
    M = spla.LinearOperator(Ak.shape, lambda x: x/d_)
    sol, info = spla.cg(Ak, b[keep], rtol=1e-12, maxiter=20000, M=M)
    if info != 0:
        raise RuntimeError("cg did not converge (info=%d)" % info)
    v = np.zeros(n**3)
    v[keep] = sol
    R_tot = float(v[contact].mean())
    # 1-D bulk term for a block of height (n-1)*dx and area (n*dx)^2
    R_bulk = ((n - 1)*dx)/(sigma*(n*dx)**2)
    return R_tot, R_tot - R_bulk, contact.size


def main():
    rho = 1.0/SIGMA
    r0 = 40e-6
    eq_exact = rho/(4*r0)                 # equipotential disc
    un_exact = 8*rho/(3*np.pi**2*r0)      # uniform current density
    print("constriction test, r0 = %.0f um, sigma %.3g S/m" % (1e6*r0, SIGMA))
    print("EXACT: equipotential disc rho/(4r0) = %.5e ; uniform-density "
          "8rho/(3 pi^2 r0) = %.5e (%.1f%% higher)"
          % (eq_exact, un_exact, 100*(un_exact/eq_exact - 1)), flush=True)

    print("\nA) FINITE-SIZE BIAS: fixed dx = 12.5 um, growing the block",
          flush=True)
    print("   %8s %7s %14s %10s %14s %10s"
          % ("L um", "L/r0", "R(uniform)", "vs exact", "R(equipot)",
             "vs exact"), flush=True)
    dx = 12.5e-6
    for nn in (32, 64, 96):
        L = nn*dx
        _, ru, _ = spreading_resistance(nn, dx, r0, mode='uniform')
        _, re, _ = spreading_resistance(nn, dx, r0, mode='equipotential')
        print("   %8.0f %7.1f %14.5e %9.2f%% %14.5e %9.2f%%"
              % (1e6*L, L/r0, ru, 100*(ru/un_exact - 1),
                 re, 100*(re/eq_exact - 1)), flush=True)

    print("\nB) DISCRETISATION: fixed 800 um block, refining dx",
          flush=True)
    print("   %8s %7s %14s %10s %14s %10s %9s"
          % ("dx um", "r0/dx", "R(uniform)", "vs exact", "R(equipot)",
             "vs exact", "equi/unif"), flush=True)
    L = 800e-6
    for nn in (16, 32, 48, 64):
        dxx = L/nn
        _, ru, _ = spreading_resistance(nn, dxx, r0, mode='uniform')
        _, re, _ = spreading_resistance(nn, dxx, r0, mode='equipotential')
        print("   %8.2f %7.1f %14.5e %9.2f%% %14.5e %9.2f%% %8.4f"
              % (1e6*dxx, r0/dxx, ru, 100*(ru/un_exact - 1),
                 re, 100*(re/eq_exact - 1), re/ru), flush=True)
    print("\nThe equipotential contact must sit BELOW the uniform one: "
          "at DC the unconstrained split minimises dissipation "
          "(Thomson), and prescribing a profile is a constraint.",
          flush=True)


if __name__ == '__main__':
    main()
