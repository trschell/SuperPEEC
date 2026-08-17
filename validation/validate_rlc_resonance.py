# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""RLC self-resonance validation of the mixed-row LpPR formulation.

Geometry: a folded parallel-plate line -- two conductor plates one cell
apart in z, shorted at the far x-end, driven across the gap at the open
x-end. Electrically this is a loop inductance L (down one plate, through
the short, back the other) in parallel with the plate capacitance C across
the driving port: a parallel RLC resonator.

The test is fully self-consistent within the code's own physics. Using only
the wired operators (Z = R + jw*Lp via traverseRL, A via connectA, P = n2n),
it assembles the small dense mixed-row MNA and:

  1. extracts R and L from the capacitance-free (LpR-equivalent) solve
     Z_LpR(w) = R + jw*L at low frequency;
  2. extracts the port capacitance C from the electrostatic solve
     q = P^{-1} v, holding each PLATE at a uniform potential (each plate is
     one conductor, so it is an equipotential -- energising a single node
     per plate instead grounds the rest of the grid and understates C by
     7x to 21x depending on scheme);
  3. predicts f_res = 1 / (2*pi*sqrt(L*C));
  4. LOCATES the resonances of the full LpPR input impedance Z(w) = V_port/I
     and checks that the ADOPTED (-jw) capacitive sign resonates near f_res
     while the flipped (+jw) sign does not resonate ANYWHERE in the band.

Two things make step 4 trustworthy, and it is worth being explicit about
them because the obvious implementations of both are wrong:

* THE ASSEMBLY MUST BE RESCALED. Written naturally, the node rows carry the
  coefficient of potential P (~1e14 F^-1) against a capacitive block jw
  (~1e11), and the assembled MNA has a condition number ~1e15 -- at the
  double-precision limit. It then returns noise whose residual still looks
  small, and the "resonance" is wherever the noise happens to peak. Left-
  multiplying the node rows by (PiE P + PiJ)^-1 is exact and drops that to
  ~1e4-1e7. See the comment at Wnode below.
* THE PEAK MUST NOT BE FOUND BY argmax. This resonance has Q ~ 1e7, so its
  |Z| pole is far narrower than any practical sweep and the height and
  position of the largest sample are sampling artefacts. Resonances are
  instead located by the + -> - sign change of Im Z (inductive below a
  parallel resonance, capacitive above), which is sampling independent, and
  then pinned down by bisection.

The (+jw) check is what pins the sign -- impossible at 10 Hz, where the
capacitive term is ~17 orders below the resistive one, but decisive here,
and sharper than a peak-height comparison: at the flipped sign the port
reactance stays inductive across the whole band, so there is no crossing at
all rather than merely a smaller peak.

Scope: this validates INTERNAL CONSISTENCY -- that LpPR reproduces the RLC
network formed by the code's own quasi-static L and C, resonating near
1/(2*pi*sqrt(L*C)) with the correct sign. The structure is genuinely
distributed, so it has a whole ladder of higher resonances (all reported)
and the single-pole lumped oracle is only expected to place the fundamental
to a few tens of percent -- observed 13% (cell) and 5% (edge). Any
self-resonance necessarily sits where the structure is a fraction of a
wavelength, at the edge of quasi-static (retardation-free) PEEC validity,
so the absolute resonant frequency of the real copper part would shift
somewhat with retardation and a finer mesh; that is a separate accuracy
question.

Run inside the toolbox:  python3 validate_rlc_resonance.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import multipole as mp
import stencils as st

mu0 = 4*np.pi*1e-7
c0 = 299792458.0
eps0 = 1/(mu0*c0**2)
conductivity = 5.81e7

# ---------------------------------------------------------------- geometry
NT = np.array([8, 4, 3])          # x-length, y-width, z (two plates + gap)
cell = 5e-4                        # 500 um cells -> resonance ~ 10s of GHz
LT = NT*cell
fullstruc = np.zeros(NT, dtype=np.int8)
fullstruc[:, :, 0] = 1            # bottom plate
fullstruc[:, :, -1] = 1           # top plate
fullstruc[-1, :, :] = 1           # shorting wall at far x-end
gap = cell                        # z-separation of the plates (1 cell)

# nleaf must cover the staggered filament grids (NT+1) for the single-group
# inductive p2p; node<->panel attachment is decoupled from nleaf (uses the
# true node grid, tree.py node2panel).
nleaf = st.single_level_nleaf(NT)
M = mp.Tree(fullstruc, nleaf, LT, 1, 1e0, None, capacitive=True)
M.e.r = M.e.l[1] / (M.e.l[0]*M.e.l[2]*conductivity)
M.f.r = M.f.l[0] / (M.f.l[1]*M.f.l[2]*conductivity)
M.g.r = M.g.l[2] / (M.g.l[0]*M.g.l[1]*conductivity)
M.lv[0].beta = 1.0
M.jomega = 1j
M.alpha = 0
M.RDFinit()

esize = np.size(M.e.struc)
fsize = np.size(M.f.struc)
gsize = np.size(M.g.struc)
efgsize = esize + fsize + gsize
nodesize = np.size(M.lv[0].struc)

whole = np.zeros((efgsize + nodesize,), dtype=np.complex128)
M.e.data = whole[:esize]
M.f.data = whole[esize:esize+fsize]
M.g.data = whole[esize+fsize:efgsize]
M.lv[0].data = whole[efgsize:]

print("folded parallel-plate line: %d filaments, %d nodes (%d external)"
      % (efgsize, nodesize, M.external.size))

# ------------------------------------------- dense operators from the code
# Lp: probe traverseRL at jw = 1j, then strip the diagonal resistance.
Rdiag = np.concatenate([np.full(esize, M.e.r), np.full(fsize, M.f.r),
                        np.full(gsize, M.g.r)]).astype(np.complex128)
M.jomega = 1j
Lp = np.zeros((efgsize, efgsize), dtype=np.complex128)
for k in range(efgsize):
    whole[:efgsize] = 0
    whole[k] = 1.0
    M.traverseRL()
    Lp[:, k] = whole[:efgsize]
Lp = (Lp - np.diag(Rdiag)) / 1j          # real symmetric partial inductance
Lp = 0.5*(Lp + Lp.T)

# A: incidence, probe connectA (node V -> filament voltage drops)
Amat = np.zeros((efgsize, nodesize), dtype=np.complex128)
for k in range(nodesize):
    M.lv[0].data[:] = 0
    M.lv[0].data[k] = 1.0
    (ae, af, ag) = M.connectA()
    Amat[:, k] = np.concatenate([ae, af, ag])
At = Amat.T.copy()

P = np.asarray(M.n2n, dtype=np.complex128)       # node-level coeff of potential
ext = M.external
extmask = np.zeros(nodesize, dtype=bool)
extmask[ext] = True
PiE = np.diag(extmask.astype(np.complex128))
PiJ = np.diag((~extmask).astype(np.complex128))

# Node rows in the natural (charge) form are (PiE P + PiJ) At against a
# capacitive block jw PiE. P is a coefficient of potential, ~1e14 F^-1 here,
# while jw ~ 1e11, so those blocks differ by orders of magnitude from the
# filament rows and from each other: the assembled MNA comes out with a
# condition number ~1e15, i.e. AT the double-precision limit, and the solve
# returns pure noise even though its residual looks small. Left-multiplying
# the node rows by (PiE P + PiJ)^-1 is an EXACT row operation -- same
# solution, no approximation -- and puts the system in the standard
# well-scaled form [[Z, A], [At, -jw P^-1]], dropping the condition number
# to ~1e4-1e7. Without it none of the checks below mean anything.
Wnode = np.linalg.inv(PiE @ P + PiJ)
Cnode = Wnode @ PiE                               # P^-1 on the external nodes

# ------------------------------------------------------ terminals / port
def node_index(x, y, z):
    mark = M.parsesource(np.array([x]), np.array([y]), np.array([z]),
                         np.array([1.0], dtype=np.complex128), 'node')
    nz = np.nonzero(mark)[0]
    if nz.size == 0:
        raise RuntimeError("terminal node (%d,%d,%d) not in structure"
                           % (x, y, z))
    return nz[0]

# One node per CELL: the grid is NT and spans z = 0..NT[2]-1, so the top
# plate's node is z = NT[2]-1. Asking for z = NT[2] does not raise -- it
# flattens to a linear index that lands on a completely different cell on
# the BOTTOM plate, turning the port into a short between two
# y-neighbours and making L, C and the whole sweep meaningless.
iA = node_index(0, 0, NT[2] - 1)   # top plate cell, open end (drive +I)
iB = node_index(0, 0, 0)           # bottom plate, open end (ground)
print("port: drive node", iA, " ground node", iB)

Iport = 1.0                        # 1 A test current


def solve_mna(omega, cap_sign):
    """Dense mixed-row LpPR solve. cap_sign=-1 is the adopted sign;
    cap_sign=0 drops capacitance (LpR-equivalent R+jwL reference)."""
    Z = np.diag(Rdiag) + 1j*omega*Lp
    n = efgsize + nodesize
    Amat_full = np.zeros((n, n), dtype=np.complex128)
    rhs = np.zeros((n,), dtype=np.complex128)
    Amat_full[:efgsize, :efgsize] = Z
    Amat_full[:efgsize, efgsize:] = Amat
    # node rows, pre-scaled by (PiE P + PiJ)^-1 as set up above: the current
    # operator is then just At and the RHS just the injected node current,
    # with the capacitance entering as P^-1 rather than the RHS as P.
    Amat_full[efgsize:, :efgsize] = At
    if cap_sign != 0:
        Amat_full[efgsize:, efgsize:] = cap_sign*1j*omega*Cnode
    sn = np.zeros(nodesize, dtype=np.complex128)
    sn[iA] = Iport
    sn[iB] = -Iport
    rhs[efgsize:] = sn
    # ground node iB: replace its node equation with V_iB = 0
    grow = efgsize + iB
    Amat_full[grow, :] = 0
    Amat_full[grow, grow] = 1.0
    rhs[grow] = 0.0
    # two-sided inf-norm equilibration
    rs = np.abs(Amat_full).max(1); rs[rs == 0] = 1
    A1 = Amat_full/rs[:, None]
    cs = np.abs(A1).max(0); cs[cs == 0] = 1
    x = np.linalg.solve(A1/cs[None, :], rhs/rs)/cs
    V = x[efgsize:]
    # Overall port polarity: the branch convention Z i + A V = 0 makes the
    # readout come out negated (A is the +gradient incidence), so flip to
    # get a passive driving-point impedance (Re, Im > 0 below resonance).
    return -(V[iA] - V[iB]) / Iport


# ------------------------------------------------------- extract R, L, C
w_lo = 2*np.pi*1e6
Zlo = solve_mna(w_lo, 0)
R_dc = Zlo.real
L_ext = Zlo.imag / w_lo
# Port capacitance: q = P^{-1} v. Each PLATE is one conductor, so the whole
# plate is an equipotential -- every top-plate node sits at +1/2 V and every
# bottom-plate node at -1/2 V, and C is the total charge gathered on the top
# plate. Energising a single node per plate and leaving the rest of the
# grid at 0 V instead grounds those nodes, which shorts out most of the
# field and understates C by 7x (cell) to 21x (edge); f_res then comes out
# high by the square root of that and no sweep can match it. The shorting
# wall at the far x-end joins the two plates, so its nodes belong to
# neither equipotential and are excluded.
cc = np.stack([M.lv[0].idx // (M.ntotal[1]*M.ntotal[2]),
               (M.lv[0].idx // M.ntotal[2]) % M.ntotal[1],
               M.lv[0].idx % M.ntotal[2]], 1)
top = (cc[:, 2] == NT[2] - 1)          # one node per cell: top plate layer
bot = (cc[:, 2] == 0)
wall = (cc[:, 0] >= NT[0] - 1)
top &= ~wall
bot &= ~wall
vport = np.zeros(nodesize, dtype=np.complex128)
vport[top] = 0.5
vport[bot] = -0.5
C_port = float(np.real(np.linalg.solve(P, vport)[top].sum()))
f_res = 1/(2*np.pi*np.sqrt(L_ext*C_port))

print("\nextracted lumped elements (from the code's own operators):")
print("  R  = %.4e ohm" % R_dc)
print("  L  = %.4e H" % L_ext)
print("  C  = %.4e F" % C_port)
print("  predicted parallel resonance f_res = %.4e Hz (%.2f GHz)"
      % (f_res, f_res/1e9))

# ------------------------------------------------------------- sweep
# The parallel resonance has Q ~ 1e7, so its |Z| pole is far narrower than
# any practical log-spaced sweep: which sample happens to land nearest the
# pole -- and therefore where |Z| peaks and how high -- is an artefact of
# the sampling, not a property of the physics. Locating the resonance by
# argmax of a coarse sweep is therefore meaningless. The REACTANCE sign is
# not: Z is inductive (Im Z > 0) below a parallel resonance and capacitive
# (Im Z < 0) above it, so each resonance shows up as a + -> - crossing of
# Im Z whose location is sampling independent. That is what is tested.
def Zport(f, cap_sign):
    return solve_mna(2*np.pi*f, cap_sign)


def crossings(cap_sign, band, npts=400):
    """Frequencies where Im Z goes + -> -, i.e. parallel resonances."""
    fgrid = np.logspace(np.log10(band[0]), np.log10(band[1]), npts)
    im = np.array([Zport(f, cap_sign).imag for f in fgrid])
    out = []
    for k in range(npts - 1):
        if im[k] > 0 and im[k+1] < 0:
            lo, hi = fgrid[k], fgrid[k+1]
            for _ in range(40):                      # bisect to pin it down
                mid = np.sqrt(lo*hi)
                if Zport(mid, cap_sign).imag > 0:
                    lo = mid
                else:
                    hi = mid
            out.append(np.sqrt(lo*hi))
    return out, fgrid, im


band = (f_res*10**-0.8, f_res*10**0.8)
res_minus, fgrid, im_minus = crossings(-1, band)
res_plus, _, im_plus = crossings(+1, band)

print("\nfrequency sweep of the driving-point impedance (coarse, for "
      "reading only):")
print("  %-11s %-13s %-13s %-13s" %
      ("f [GHz]", "|Z| (-jw)", "|Z| (+jw)", "|Z| RLC-model"))
fs = np.logspace(np.log10(band[0]), np.log10(band[1]), 15)
for f in fs:
    w = 2*np.pi*f
    zr = 1/(1/(R_dc + 1j*w*L_ext) + 1j*w*C_port)
    print("  %-11.3f %-13.4e %-13.4e %-13.4e"
          % (f/1e9, abs(Zport(f, -1)), abs(Zport(f, +1)), abs(zr)))

print("\nparallel resonances located by Im Z sign change over %.2f-%.2f GHz:"
      % (band[0]/1e9, band[1]/1e9))
print("  adopted (-jw): %s"
      % (", ".join("%.3f GHz" % (r/1e9) for r in res_minus) or "NONE"))
print("  flipped (+jw): %s"
      % (", ".join("%.3f GHz" % (r/1e9) for r in res_plus) or "NONE"))

# ------------------------------------------------------------- checks
FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name +
          ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


print()
check("adopted (-jw) sign produces a parallel resonance",
      len(res_minus) > 0,
      "%d found in band" % len(res_minus))

if res_minus:
    f1 = res_minus[0]
    # The lumped RLC oracle is a single-pole approximation of a structure
    # that is genuinely distributed (it has the whole ladder of higher
    # resonances listed above), so agreement to a few tens of percent is
    # the most that can be asked of it -- but the two must not be decades
    # apart, which is what a wrong C, a wrong port or a broken solve give.
    check("fundamental sits at the lumped-oracle f_res (within 30%)",
          abs(np.log(f1/f_res)) < np.log(1.30),
          "first resonance %.3f GHz vs oracle %.3f GHz (ratio %.3f)"
          % (f1/1e9, f_res/1e9, f1/f_res))
    # a genuine resonance must lift |Z| far above the off-resonance level
    zpk = abs(Zport(f1, -1))
    zoff = min(abs(Zport(band[0], -1)), abs(Zport(band[1], -1)))
    check("resonance is pronounced (|Z| near the pole >> off-resonance)",
          zpk/zoff > 10.0, "|Z| %.3e at %.3f GHz vs %.3e off-resonance"
          % (zpk, f1/1e9, zoff))

# THE SIGN DISCRIMINATOR. With the capacitive term entered at the flipped
# sign the port cannot resonate at all: the reactance stays inductive over
# the entire band, so there is no crossing anywhere -- not merely a smaller
# peak. This is a yes/no statement, independent of sampling.
check("flipped (+jw) sign does NOT resonate anywhere in the band",
      len(res_plus) == 0,
      "Im Z(+jw) stays inductive: min %.4e, max %.4e over %d samples"
      % (im_plus.min(), im_plus.max(), im_plus.size))

# below the fundamental the solve must track the lumped model closely --
# this is what catches a mis-scaled or mis-conditioned assembly
flo = band[0]
wlo = 2*np.pi*flo
zlo = Zport(flo, -1)
zrlo = 1/(1/(R_dc + 1j*wlo*L_ext) + 1j*wlo*C_port)
rel = abs(zlo - zrlo)/abs(zrlo)
check("well below resonance Z matches the lumped RLC model (< 15%)",
      rel < 0.15, "%.4e vs %.4e at %.2f GHz (rel %.3f)"
      % (abs(zlo), abs(zrlo), flo/1e9, rel))

print()
if res_minus:
    print("resonance summary: fundamental at %.3f GHz (oracle %.3f GHz); "
          "%d higher resonances in band; flipped sign has none"
          % (res_minus[0]/1e9, f_res/1e9, max(len(res_minus)-1, 0)))
if FAIL:
    print("FAILURES:", ", ".join(FAIL))
    sys.exit(1)
print("all checks passed -- LpPR captures RLC self-resonance; "
      "capacitive sign is -jw")
