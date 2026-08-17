# SPDX-License-Identifier: MIT
"""2-D CROSS-SECTION BASIS STUDY -- isolates the basis question from all of
SuperPEEC's machinery (no FMM, no truncation, no terminals, no loop basis).

Fix the GEOMETRY (the voxelised circle SuperPEEC actually solves), subdivide
each coarse cell into k x k sub-cells, build ONE dense per-unit-length
impedance matrix over the sub-cells, then compare bases as Galerkin
projections of that same matrix:

  full      every sub-cell a DOF          -> the CONVERGED answer
  coarse    one DOF per coarse cell       -> SuperPEEC k=1
  lin       coarse + 2 linear modes/cell  -> VoxHenry-style PWL
  lin_bnd   linear modes on BOUNDARY cells only
  exp_bnd   1 exponential (skin-profile) mode per exposed face
  stair     coarse + full k x k staircase == full (sanity)

Per-unit-length reduction: take LONG boxes (length LBAR >> cross-section)
and divide by LBAR. The additive constant in the 2-D log kernel is
harmless: it acts as c*1*1^T, i.e. a series inductance that shifts Im(Z)
without changing the current DISTRIBUTION, hence not Re(Z).

VALIDATION: 'full' on the 80-cell geometry should reproduce the 3-D
solver's converged 0.04032 ohm for the 50um wire. If it does, the whole
2-D reduction is trustworthy.
"""
import sys, time, numpy as np
from greens import box_pair_stencil_pairs as spair

MU, SIG = 4*np.pi*1e-7, 5.8e7
DX, RAD, FREQ, WIRELEN = 1e-6, 5e-6, 1e10, 50e-6
LBAR = 1e-2                      # long enough that 3-D -> 2-D per metre
W = 2*np.pi*FREQ
DELTA = np.sqrt(2.0/(W*MU*SIG))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def coarse_cells():
    nt = int(round(2*RAD/DX))
    c = (np.arange(nt) + 0.5 - nt/2.0)*DX
    yy, zz = np.meshgrid(c, c, indexing='ij')
    disc = (yy**2 + zz**2) < RAD**2
    idx = np.argwhere(disc)
    return disc, idx


disc, cidx = coarse_cells()
ncell = len(cidx)
nt = disc.shape[0]
# boundary = at least one of the 4 transverse neighbours empty
pad = np.pad(disc, 1)
interior = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
is_bnd = np.array([not interior[j, k] for j, k in cidx])
print("cross-section: %d coarse cells (%d boundary, %d interior), k=%d"
      % (ncell, is_bnd.sum(), (~is_bnd).sum(), K))
print("delta = %.4g m = %.3f cells;  sub-cell = %.3f cells"
      % (DELTA, DELTA/DX, 1.0/K))

# --- sub-cell boxes -------------------------------------------------------
h = DX/K
lo = np.zeros((ncell*K*K, 3)); hi = np.zeros_like(lo)
sub_of_cell = np.zeros((ncell, K*K), dtype=int)
uu = np.zeros(ncell*K*K); vv = np.zeros(ncell*K*K)
n = 0
for ci, (j, k) in enumerate(cidx):
    y0 = (j - nt/2.0)*DX; z0 = (k - nt/2.0)*DX
    for p in range(K):
        for q in range(K):
            lo[n] = [0.0, y0 + p*h, z0 + q*h]
            hi[n] = [LBAR, y0 + (p+1)*h, z0 + (q+1)*h]
            uu[n] = (p + 0.5)/K - 0.5          # centroid offset in [-.5,.5]
            vv[n] = (q + 0.5)/K - 0.5
            sub_of_cell[ci, p*K + q] = n
            n += 1
nsub = n
asub = h*h
print("sub-cells: %d   (dense Z = %.0f MB)" % (nsub, nsub*nsub*16/1e6))

t0 = time.perf_counter()
ia, ib = np.meshgrid(np.arange(nsub), np.arange(nsub), indexing='ij')
S = spair(lo[ia.ravel()], hi[ia.ravel()], lo[ib.ravel()], hi[ib.ravel()])
Lp = (S.reshape(nsub, nsub)/(asub*asub))/LBAR          # H per metre
Z = np.diag(np.full(nsub, 1.0/(SIG*asub))) + 1j*W*Lp   # per metre
print("Z built in %.1f s" % (time.perf_counter()-t0))


def solve(P):
    """Galerkin-project onto columns of P and return R of the wire."""
    A = P.conj().T @ Z @ P
    b = P.conj().T @ np.ones(nsub)
    x = np.linalg.solve(A, b)
    itot = np.ones(nsub) @ (P @ x)
    return (1.0/itot).real*WIRELEN


def basis(kind):
    cols = []
    for ci in range(ncell):
        s = sub_of_cell[ci]
        agg = np.zeros(nsub); agg[s] = 1.0/(K*K); cols.append(agg)
    if kind == 'coarse':
        return np.array(cols).T
    for ci in range(ncell):
        s = sub_of_cell[ci]
        if kind in ('lin_bnd', 'exp_bnd') and not is_bnd[ci]:
            continue
        if kind in ('lin', 'lin_bnd'):
            for w in (uu[s], vv[s]):
                m = np.zeros(nsub); m[s] = w - w.mean(); cols.append(m)
        elif kind == 'exp_bnd':
            # inward distance from each EXPOSED transverse face
            j, k = cidx[ci]
            for d, ax, sgn in ((0, uu, +1), (0, uu, -1),
                               (1, vv, +1), (1, vv, -1)):
                nb = (j + sgn, k) if d == 0 else (j, k + sgn)
                if 0 <= nb[0] < nt and 0 <= nb[1] < nt and disc[nb]:
                    continue                       # face not exposed
                xi = (0.5 - sgn*ax[s])*DX          # depth from that face
                w = np.exp(-xi/DELTA)
                m = np.zeros(nsub); m[s] = w - w.mean(); cols.append(m)
    return np.array(cols).T


full = solve(np.eye(nsub))
print("\n%-10s %-7s %-13s %-9s %s" % ("basis", "nDOF", "R [ohm]",
                                      "vs full", "of the gap"))
res = {}
for kind in ('coarse', 'lin', 'lin_bnd', 'exp_bnd'):
    P = basis(kind)
    R = solve(P)
    res[kind] = R
    print("%-10s %-7d %-13.7g %-+9.2f%% %s"
          % (kind, P.shape[1], R, 100*(R/full-1),
             "%.0f%%" % (100*(R-res['coarse'])/(full-res['coarse']))
             if kind != 'coarse' else "-"))
print("%-10s %-7d %-13.7g %-+9.2f%% %s" % ('full', nsub, full, 0.0, "100%"))
print("\n3-D solver on the SAME geometry: k=1 -> 0.03800266, "
      "converged -> 0.04032062")
