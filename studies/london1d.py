"""The 1-D FORMULATION BENCH: does the ~90% floor live in the basis?
# SPDX-License-Identifier: MIT

The microstrip's z-problem in one dimension: sheet currents j(z), the
strip film carrying +1 and the ground film -1 per unit width; energy

    E = (1/2) int B^2/mu0 dz + (mu0 lam^2/2) int j^2 dz,
    B(z) = mu0 * (current above z)

and minimising E over j at fixed totals IS the London problem (the
same variational principle the engine's L-minimisation embodies), so
L = 2E exactly, with the analytic answer mu0*(h + 2 lam coth(t/lam)).

Discretisation mirrors the engine EXACTLY where it matters: cells of
height dz, k piecewise-constant sub-bars per cell, per-cell basis =
{uniform aggregate} + {net-zero mode columns from the engine's OWN
conduction_weights (imported), kk=(1,k), p=1/lam}. Everything is
assembled by dense quadrature on a 40k-point z-grid (1-D integrals to
~1e-9), so there is NO truncation, NO Krylov, NO mask: whatever
deficit remains is the FORMULATION.

Variants:
  engine   : W sampled at sub-bar centroids (what conduction_weights does)
  cellavg  : W = exact exponential averages over each sub-bar
  contin   : continuum exponential shapes (span-exact within each cell)
"""
import sys
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from equiterminal import conduction_weights

MU0 = 4e-7*np.pi
T = 200e-9
H = 200e-9
NQ = 40000

def qr_netzero(cols):
    """Mean-subtract, normalise, prune -- same recipe as the engine."""
    from scipy.linalg import qr
    keep = []
    for c in cols:
        w = c - c.mean()
        n = np.linalg.norm(w)
        if n > 1e-12*max(np.linalg.norm(c), 1.0):
            keep.append(w/n)
    W = np.stack(keep, axis=1)
    _, R, piv = qr(W, mode='economic', pivoting=True)
    d = np.abs(np.diag(R))
    W = W[:, np.sort(piv[:d.size][d > 1e-7*d[0]])]
    return W - W.mean(axis=0, keepdims=True)

def solve(lam, nt, k, variant):
    dz = T/nt
    # z-grid over [0, 2T + H]: ground film [0, T], gap, strip [T+H, 2T+H]
    zt = 2*T + H
    z = (np.arange(NQ) + 0.5)*zt/NQ
    dq = zt/NQ
    films = [(0.0, T, -1.0), (T + H, zt, +1.0)]
    # basis columns as current DENSITIES on the quad grid
    cols = []
    meta = []          # (film index, 'agg'/'mode')
    for fi, (z0, z1, sgn) in enumerate(films):
        for c in range(nt):
            a = z0 + c*dz
            cell = (z >= a) & (z < a + dz)
            # aggregate: unit current, uniform
            u = np.zeros(NQ); u[cell] = 1.0/dz
            cols.append(u); meta.append((fi, 'agg'))
            # modes on k sub-bars
            if variant == 'engine':
                W = conduction_weights((1, k), (dz, dz), None, p=1.0/lam)
            else:
                # rebuild the same shape family with better sampling
                zz = None
                if variant == 'cellavg':
                    # exact exponential average per sub-bar
                    e = np.arange(k + 1)*dz/k
                    def avg(rate, flip):
                        x0, x1 = e[:-1], e[1:]
                        if flip: x0, x1 = dz - e[1:], dz - e[:-1]
                        return (np.exp(-rate*x0) - np.exp(-rate*x1))/(
                            rate*(e[1:] - e[:-1]))
                    raw = [avg(1/lam, False), avg(1/lam, True)]
                    W = qr_netzero(raw)
                elif variant == 'contin':
                    W = None
            if variant in ('engine', 'cellavg'):
                for mcol in range(W.shape[1]):
                    v = np.zeros(NQ)
                    sub = ((z[cell] - a)*k/dz).astype(int).clip(0, k - 1)
                    v[cell] = W[sub, mcol]/(dz/k)   # current -> density
                    cols.append(v); meta.append((fi, 'mode'))
            else:
                zz = z[cell] - a
                for shape in (np.exp(-zz/lam), np.exp(-(dz - zz)/lam)):
                    s0 = shape - shape.mean()
                    n = np.linalg.norm(s0)
                    if n > 1e-12:
                        v = np.zeros(NQ); v[cell] = s0/n/dq  # net-zero
                        cols.append(v); meta.append((fi, 'mode'))
    Phi = np.stack(cols, axis=1)                    # densities (NQ, nb)
    # B_a(z) = mu0 * integral_z^inf phi_a: cumulative from the top
    Ba = MU0*(np.cumsum(Phi[::-1], axis=0)[::-1] - Phi/2.0)*dq
    A = (Ba.T @ Ba)/MU0*dq                          # magnetic energy form
    lam_pen = MU0*lam*lam*(Phi.T @ Phi)*dq          # kinetic form
    Q = A + lam_pen
    # constraints: film totals +-1
    nb = Phi.shape[1]
    C = np.zeros((2, nb))
    for a2, (fi, kind) in enumerate(meta):
        if kind == 'agg':
            C[fi, a2] = 1.0
    b = np.array([-1.0, 1.0])
    # KKT
    Kk = np.block([[Q, C.T], [C, np.zeros((2, 2))]])
    rhs = np.concatenate([np.zeros(nb), b])
    sol = np.linalg.solve(Kk, rhs)
    c = sol[:nb]
    E = 0.5*c @ Q @ c
    return 2*E

def main():
    coth = lambda t, l: l/np.tanh(t/l)
    for lam in (9e-8,):
        Lex = MU0*(H + 2*coth(T, lam))
        Lex0 = MU0*(H + 2*coth(T, 1e-9))
        kex = Lex - Lex0
        print("lam=%g: exact L=%.6e, kinetic part %.6e (=2 mu0 lam coth)"
              % (lam, Lex, kex))
        print("%8s %4s %4s | %12s %10s" % ("variant", "nt", "k", "L", "recovered"))
        for variant in ('engine', 'cellavg', 'contin'):
            for nt, k in ((2, 7), (2, 12), (3, 12), (4, 12)):
                L = solve(lam, nt, k, variant)
                L0 = solve(1e-9, nt, k, variant)
                print("%8s %4d %4d | %12.6e %9.1f%%"
                      % (variant, nt, k, L, 100*(L - L0)/kex), flush=True)

if __name__ == '__main__':
    main()
