# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Which net-zero mode basis can actually represent skin effect?

SuperPEEC's skin-effect engine is exact in assembly and exonerated on
truncation, yet an untruncated 50 um wire still gives 201% of the needed
correction -- so the defect is the k x k PIECEWISE-CONSTANT mode basis
itself (see the Redistribution docstring in equiterminal.py). That is a
BASIS-SPAN question, and a span question can be settled in 2-D on a
single cross-section without any of SuperPEEC's machinery.

THE TESTBED. One isolated bar, per unit length, discretised into a fine
grid of parallel sub-conductors -- the same filament model SuperPEEC uses,
just in 2-D:

    (R + jw L) i = V 1,     Z = V / (1^T i)

with R diagonal (1/(sigma*area)) and L the 2-D log kernel. The reference
is the fine grid; a candidate basis is a Galerkin reduction of it,

    i = T c,   T = [aggregate | net-zero modes],
    (T^H (R + jwL) T) c = V T^H 1,

which is exactly what Redistribution does. So "how good is this basis"
becomes "how close is the reduced Z to the fine Z", measured the same way
SuperPEEC would measure it.

THE LOG-KERNEL CONSTANT IS HARMLESS HERE. Replacing L by L + c*11^T adds
jw*c*(1^T i)*1, i.e. it only rescales V. The current DISTRIBUTION and
hence Re(Z) are unchanged; only Im(Z) shifts by w*c. Since the 201% is an
R error, R is the quantity to compare and it needs no return path.

BASES COMPARED (all net-zero, all on the same k0 x k1 sub-bar grid):
  diff        k-1 differences of consecutive sub-bars   (SuperPEEC today)
  linear      2 columns, sub-bar centroid offsets       (VoxHenry/PWL-ish)
  conduction  exponential edge/corner modes from the interior Helmholtz
              equation, per Daniel/Sangiovanni-Vincentelli/White EPEP
              2000: edge p=(1+j)/delta q=0, corner p=q=(1+j)/(delta*sqrt2)

Conduction modes are complex; each is entered as TWO REAL columns (real
and imaginary part), which is both what SuperPEEC's real-kernel FFT requires
and a strictly larger span than one complex column.

RESULT 2026-08-05 -- THE GATE PASSED, 6.6x AT MATCHED km.
Square bar, NFINE=48. Entries are err in %% = the fraction of the
skin-effect CORRECTION the basis MISSES (0 exact, -100 captured none):

  basis            km   w/d=1   2      3      4.8    7      10
  diff  k=2         3  -100.0 -100.0 -100.0 -100.0 -100.0 -100.0
  diff  k=3         8   -51.6  -51.0  -48.8  -40.8  -39.6  -52.0
  diff  k=4        15   -30.4  -29.9  -28.0  -22.4  -26.5  -43.0
  linear k=8        2  -100.0 -100.0 -100.0 -100.0 -100.0 -100.0
  cond  k=8         2    -8.3   -8.0   -6.9   -4.6   -8.7  -18.0
  cond  k=8         8    -7.8   -7.6   -7.0   -5.5   -7.7  -13.6
  cond  k=12        8    -3.4   -3.3   -3.0   -2.4   -3.4   -6.1

At matched km=8 conduction beats diff 6.6x (-7.8 vs -51.6); conduction
at km=2 beats diff at km=15. Raising QUADRATURE k 8->12 at fixed km=8
goes -7.8 -> -3.4, confirming that k buys accuracy without adding solve
cost (km is what drives cost).

TWO SELF-CHECKS ON THE TESTBED: diff k=2 gives exactly -100%%,
reproducing the recorded k=2 blind spot, and linear does too -- both are
ODD bases and the skin profile on a symmetric isolated bar is EVEN, so
they are orthogonal to it.

WHAT THIS DOES **NOT** SHOW: it does not reproduce the 3-D 201%%
OVER-prediction. Every basis here UNDER-predicts. The recorded 3-D
behaviour is erratic (70%% on one wire, 201%% on another), consistent
with "wrong shape" but meaning the over-shoot comes from something this
testbed cannot see -- proximity from neighbouring filaments, or the
terminal coupling. Extending this to TWO COUPLED BARS is the cheap way
to test that before trusting the 3-D fix.

Usage:  PYTHONPATH=src python3 studies/modebasis2d.py
Env:    NFINE (default 48), KSUB (default 8), MU (relative), SIGMA
"""
import os
import numpy as np

MU0 = 4e-7*np.pi
SIGMA = float(os.environ.get('SIGMA', '5.8e7'))
NFINE = int(os.environ.get('NFINE', '48'))      # fine cells per side
KSUB = int(os.environ.get('KSUB', '8'))         # sub-bars per side
W_BAR = 1e-5                                    # bar side, m (square)


def geometry(n):
    """Cell centres and area for an n x n square cross-section."""
    a = W_BAR/n
    c = (np.arange(n) + 0.5)*a
    X, Y = np.meshgrid(c, c, indexing='ij')
    return X.ravel(), Y.ravel(), a


def impedance_matrix(n, freq):
    """(R + jwL) per unit length for the fine grid of parallel cells."""
    x, y, a = geometry(n)
    area = a*a
    d2 = ((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
    # 2-D log kernel; self term uses the square's self-GMD (0.44705*a).
    # The additive constant is arbitrary -- see the module docstring.
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.44705*a)
    L = -(MU0/(2*np.pi))*np.log(d/W_BAR)
    R = np.diag(np.full(x.size, 1.0/(SIGMA*area)))
    return R + 1j*2*np.pi*freq*L


def solve_Z(Z, T=None):
    """Z of the bar, either fully or reduced onto the columns of T."""
    n = Z.shape[0]
    one = np.ones(n, dtype=np.complex128)
    if T is None:
        i = np.linalg.solve(Z, one)
        return 1.0/(one @ i)
    Zr = T.conj().T @ Z @ T
    c = np.linalg.solve(Zr, T.conj().T @ one)
    return 1.0/(one @ (T @ c))


def subbar_map(nfine, k):
    """Fine-cell -> sub-bar index, and the sub-bar centroid offsets."""
    if nfine % k:
        raise ValueError("NFINE must be a multiple of KSUB")
    r = nfine//k
    ij = np.arange(nfine)//r
    I, J = np.meshgrid(ij, ij, indexing='ij')
    sub = (I*k + J).ravel()
    u = (np.arange(k) + 0.5)/k - 0.5
    U, V = np.meshgrid(u, u, indexing='ij')
    return sub, U.ravel(), V.ravel()


def expand(sub, Wsub, nfine):
    """Lift per-sub-bar weights to per-fine-cell columns."""
    return Wsub[sub, :]


def basis_diff(k):
    ns = k*k
    W = np.zeros((ns, ns - 1))
    for m in range(ns - 1):
        W[m, m] = +1.0
        W[m + 1, m] = -1.0
    return W


def basis_linear(U, V):
    W = np.stack([U, V], axis=1)
    return W - W.mean(axis=0, keepdims=True)


def basis_conduction(U, V, delta, km_max=None):
    """Exponential edge and corner modes, real and imaginary parts.

    U, V are centroid offsets in [-1/2, 1/2]; convert to physical
    distance from each of the four faces. Opposite-edge pairs are
    combined symmetrically and antisymmetrically: the SYMMETRIC pair is
    what pure skin effect needs (crowding at both faces at once), the
    ANTISYMMETRIC one is proximity.
    """
    p = (1 + 1j)/delta
    x = (U + 0.5)*W_BAR                      # distance from the x=0 face
    y = (V + 0.5)*W_BAR
    ex0, ex1 = np.exp(-p*x), np.exp(-p*(W_BAR - x))
    ey0, ey1 = np.exp(-p*y), np.exp(-p*(W_BAR - y))
    pc = (1 + 1j)/(delta*np.sqrt(2.0))
    corner = (np.exp(-pc*(x + y)) + np.exp(-pc*(x + (W_BAR - y)))
              + np.exp(-pc*((W_BAR - x) + y))
              + np.exp(-pc*((W_BAR - x) + (W_BAR - y))))
    cols = [ex0 + ex1 + ey0 + ey1,           # symmetric: skin
            ex0 - ex1,                       # proximity along x
            ey0 - ey1,                       # proximity along y
            corner]
    if km_max is not None:
        cols = cols[:km_max]
    out = []
    for c in cols:
        for part in (c.real, c.imag):
            part = part - part.mean()
            if np.abs(part).max() > 1e-30:
                out.append(part)
    W = np.stack(out, axis=1)
    return W - W.mean(axis=0, keepdims=True)


def build(name, ksub, nfine, delta, km_max=None):
    """(columns on the fine grid, label) for one candidate basis."""
    sub, U, V = subbar_map(nfine, ksub)
    if name == 'diff':
        Wsub = basis_diff(ksub)
    elif name == 'linear':
        Wsub = basis_linear(U, V)
    elif name == 'cond':
        Wsub = basis_conduction(U, V, delta, km_max=km_max)
    else:
        raise ValueError(name)
    return expand(sub, Wsub, nfine)


def main():
    print("2-D basis-span testbed: square bar, NFINE=%d, sigma=%.3g"
          % (NFINE, SIGMA))
    print("err = (R_basis - R_ref)/(R_ref - R_dc): the fraction of the "
          "skin-effect\nCORRECTION the basis misses. 0 = exact, "
          "-100 = captured none of it.\n")
    nf = NFINE*NFINE
    agg = np.ones((nf, 1))/nf

    # k is QUADRATURE (sub-bar resolution), km is UNKNOWNS. diff on a
    # k x k grid spans the WHOLE net-zero space there, so it is the best
    # any piecewise-constant basis can do at that resolution.
    cands = [('diff  k=2', 'diff', 2, None),
             ('diff  k=3', 'diff', 3, None),
             ('diff  k=4', 'diff', 4, None),
             ('linear k=8', 'linear', 8, None),
             ('cond  k=8 (1 mode)', 'cond', 8, 1),
             ('cond  k=8 (2 modes)', 'cond', 8, 2),
             ('cond  k=8 (4 modes)', 'cond', 8, 4),
             ('cond  k=12 (4 modes)', 'cond', 12, 4)]

    ratios = (1.0, 2.0, 3.0, 4.8, 7.0, 10.0)
    Zs, Rrefs = {}, {}
    for r in ratios:
        delta = W_BAR/r
        freq = 1.0/(np.pi*MU0*SIGMA*delta*delta)
        Zs[r] = impedance_matrix(NFINE, freq)
        Rrefs[r] = solve_Z(Zs[r]).real
    Rdc = 1.0/(SIGMA*W_BAR*W_BAR)

    hdr = "  %-22s %-4s" % ("basis", "km")
    hdr += "".join("  w/d=%-6.1f" % r for r in ratios)
    print(hdr)
    print("  %-22s %-4s" % ("R_ref (ohm/m)", "")
          + "".join("  %-10.4g" % Rrefs[r] for r in ratios))
    print("  " + "-"*(28 + 12*len(ratios)))
    for label, name, ksub, kmm in cands:
        delta0 = W_BAR/ratios[0]
        km = build(name, ksub, NFINE, delta0, kmm).shape[1]
        cells = []
        for r in ratios:
            delta = W_BAR/r
            Wf = build(name, ksub, NFINE, delta, kmm)
            T = np.concatenate([agg, Wf], axis=1).astype(np.complex128)
            Rb = solve_Z(Zs[r], T).real
            err = 100*(Rb - Rrefs[r])/(Rrefs[r] - Rdc)
            cells.append("  %-+10.1f" % err)
        print("  %-22s %-4d" % (label, km) + "".join(cells))
    print("\n  (entries are err in %; km is the number of net-zero unknowns"
          "\n   per filament, which is what drives solve cost)")


if __name__ == '__main__':
    main()
