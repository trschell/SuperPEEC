# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""C' and Z0 of SFQ5ee striplines vs Tolpygo et al. -- the LpPR chapter.

The companion to studies/sfq5ee_measured.py, on the axis inductance-only
tools cannot serve: the same M1aM0bM2 striplines solved through the
CAPACITIVE (LpPR) path, giving C' and hence Z0 = sqrt(L'/C') and the
phase velocity -- the quantities that matter for passive transmission
lines (the paper: Z ~ 97 * L/l [Ohm per pH/um] for w > 0.4 um; the
8.0-Ohm PTL design point sits at w = 2.82 um; eps_r = 4.6 for the
PECVD SiO2, from their resonance measurements).

METHOD. Same geometry as sfq5ee_measured (nominal stack, lambda =
90 nm), built directly as a VoxelModel with eps_bg = 4.6 (the uniform-
dielectric composition rule the LpPR path supports for superconductors;
see squid_washer.py). C': OPEN-ended stub, Im Z = -1/(w C), two lengths
differenced to cancel port and end fringing (the stub's quarter-wave
resonance is ~3 THz; at 10 GHz the electrostatic reading is exact to
~1e-5). L': same path with the far-end wall, Im Z = +w L, cross-checked
against the LpR value. TRUTH via TEM DUALITY: with a uniform
dielectric, C'_true = eps*mu / L'_geo(lambda->0), and L'_geo comes from
the same 2-D magnetostatic oracle used in sfq5ee_measured.

RESULTS (2026-09-03, dz = 100 nm, dx = 50 nm, nominal stack):

    w um    C' fF/um   L' pH/um   Z0 Ohm    v/c
    0.35     0.2775     0.4265    39.21    0.307
    1.00     0.5420     0.2028    19.35    0.318
    2.00     0.9492     0.1123    10.88    0.323
    2.82     1.2749     0.0822     8.03    0.326

AGAINST THE PAPER:
  * the 8.0-Ohm PTL design point (w = 2.82 um): Z0 = 8.03 -- 0.4%;
    L' = 0.0822 vs the implied 8/97 = 0.0825 pH/um -- 0.4%.
  * phase velocity ~0.32c for wide lines: 0.318-0.326 measured here.
  * the Z ~ 97 * L/l rule (stated for w > 0.4 um): within 2% at
    w >= 1 um; -5% at w = 0.35 um, below the rule's stated range.
CROSS-PATH: L' from this capacitive formulation agrees with the
inductive path (sfq5ee_measured) to 0.2-0.5% at every width -- two
disjoint discretisations of the same physics.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'src')]

import voxmodel                            # noqa: E402
from port_impedance import LpPRSolver      # noqa: E402

MU0 = 4e-7*np.pi
EPS0 = 8.8541878128e-12
T = 200e-9
G = 200e-9
LAM = 9e-8
EPSR = 4.6
ZMARG = 400e-9
LENS = (4e-6, 8e-6)


def build(w, ln, dx, dz, wall):
    wp = w + 12e-6
    nyw = int(round(w/dx))
    nyp = int(round(wp/dx))
    ny = nyp + 8
    j0p, j1p = 4, 4 + nyp
    j0s = (ny - nyw)//2
    j1s = j0s + nyw
    MX = max(4, int(round(0.5e-6/dx)))
    L = int(round(ln/dx))
    nx = 2*MX + L
    z0 = int(round(ZMARG/dz))
    tz = int(round(T/dz))
    gz = int(round(G/dz))
    zM0 = (z0, z0 + tz)
    zM1 = (zM0[1] + gz, zM0[1] + gz + tz)
    zM2 = (zM1[1] + gz, zM1[1] + gz + tz)
    nz = zM2[1] + z0
    lamg = np.zeros((nx, ny, nz))
    lamg[MX:MX + L, j0p:j1p, zM0[0]:zM0[1]] = LAM
    lamg[MX:MX + L, j0p:j1p, zM2[0]:zM2[1]] = LAM
    lamg[MX:MX + L, j0s:j1s, zM1[0]:zM1[1]] = LAM
    if wall:
        lamg[MX + L - 2:MX + L, j0s:j1s, zM0[0]:zM2[1]] = LAM
    m = voxmodel.VoxelModel('sfq5ee_lc')
    m.dims = (nx, ny, nz)
    m.d = np.array([dx, dx, dz])
    m.sigma = np.zeros((nx, ny, nz))
    m.lambdaL = lamg
    m.superconductor = True
    m.eps_bg = EPSR
    m.freq = np.array([1e10])
    p = voxmodel.Port('P')
    for j in range(j0s, j1s):
        for k in range(*zM1):
            p._add('P', (MX, j, k, 0, 1))
    for j in range(j0p, j1p):
        for zz in (zM0, zM2):
            for k in range(*zz):
                p._add('N', (MX, j, k, 0, 1))
    p._freeze()
    m.ports = [p]
    return m


def solve_Z(w, ln, dx, dz, wall, freq):
    m = build(w, ln, dx, dz, wall)
    leaf, lv = m.partition()
    # LEAN capacitive tree (validate_dielectric PART H): kernel-direct
    # band-W route -- the stored-n2n path tried a DENSE Pinv_ext over
    # 86k external nodes (111 GiB) on this grid
    M = m.build_tree(leaf, lv, capacitive=True, fftnear=True,
                     keep_n2n=False)
    m.prepare(M, freq)
    # ccap='band': the default 'diag' dense P_ext^-1 eye probe is
    # 111 GiB at this grid's 86k external nodes (same lesson as
    # studies/hrefine.py)
    S = LpPRSolver(m, M, ccap='band')
    z, _, _ = S.solve(freq)
    return complex(z)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=float, nargs='+',
                    default=[0.35e-6, 1.0e-6, 2.0e-6])
    ap.add_argument('--dz', type=float, default=100e-9)
    ap.add_argument('--freq', type=float, default=1e10)
    a = ap.parse_args(argv)
    om = 2*np.pi*a.freq
    print("M1aM0bM2 stripline C'/Z0 (LpPR, eps_bg = %.1f)" % EPSR)
    print("%7s | %10s %10s %8s %8s | %s"
          % ("w um", "C' fF/um", "L' pH/um", "Z0 Ohm", "v/c", "notes"))
    for w in a.w:
        dx = 50e-9
        t0 = time.time()
        Cs, Ls = [], []
        for ln in LENS:
            zo = solve_Z(w, ln, dx, a.dz, False, a.freq)
            Cs.append(-1.0/(om*zo.imag))
            zs = solve_Z(w, ln, dx, a.dz, True, a.freq)
            Ls.append(zs.imag/om)
        dl = LENS[1] - LENS[0]
        Cp = (Cs[1] - Cs[0])/dl
        Lp = (Ls[1] - Ls[0])/dl
        Z0 = np.sqrt(Lp/Cp)
        v = 1.0/np.sqrt(Lp*Cp)/2.99792458e8
        z97 = 97*Lp*1e6                       # the paper's rule, Ohm
        print("%7.2f | %10.4f %10.4f %8.2f %8.3f | 97-rule Z=%.1f  (%.0f s)"
              % (w*1e6, Cp*1e9, Lp*1e6, Z0, v, z97, time.time() - t0),
              flush=True)


if __name__ == '__main__':
    main()
