# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The dielectric-program capstone: a FILLED pdn_planes board solved
end-to-end through the production LpPR path, exported for ParaView.

Board: 48x48 cells, two copper planes, FR4 (eps_r 4.2) filling the
2-cell gap, a 3x3 field of power/ground vias with antipad clearance
holes, a routing slot across the top plane. Port at the chip site,
P on the power (top) plane and N on the ground (bottom) plane: the
return current is DISPLACEMENT current through the FR4 -- the pure
capacitive PDN configuration that only the LpPR path can solve (the
LpR solver correctly refuses it: no galvanic return exists).

Single-level tree (the multilevel W-rescale is blocked for
dielectrics; see the dielectric-program notes), production
LpPRSolver (SystemMat + diagschur + fgmres).

Outputs in results/:
  pdn_capstone.vti        J (re/im/mag), V (re/im), conductor mask
  pdn_capstone_t***.vti   16-phase animation of Re(J e^{jwt})
  pdn_capstone.pvd        open THIS in ParaView for the animation
  pdn_capstone_mat.vti    material ids (0 empty / 1 copper / 2 FR4)

Run: PYTHONPATH=src python3 studies/pdn_capstone.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import stencils as st
import vtkout
from port_impedance import LpPRSolver
from pdn_planes import build_pdn, stats

FREQ = float(os.environ.get('FREQ', '1e8'))
NPLANE = int(os.environ.get('NPLANE', '48'))


def main():
    m = build_pdn(nplane=NPLANE, via_pitch=16, antipad_r=3, slot=True,
                  eps_r=4.2)
    nc, nd = stats(m)
    print("pdn_capstone: %s  %d copper + %d FR4 = %d cells"
          % (m.dims, nc, nd, nc + nd), flush=True)
    t0 = time.perf_counter()
    M = m.build_tree(st.single_level_nleaf(m.dims), 1, capacitive=True)
    m.prepare(M, FREQ)
    print("  single-level capacitive tree: %.1f s, %d nodes, %d "
          "external" % (time.perf_counter() - t0,
                        int(np.size(M.lv[0].struc)), len(M.external)),
          flush=True)
    t0 = time.perf_counter()
    S = LpPRSolver(m, M)
    print("  solver setup %.1f s (wsolve=%s)"
          % (time.perf_counter() - t0, S.wsolve), flush=True)
    z, x, info = S.solve(FREQ, verbose=True)
    w = 2*np.pi*FREQ
    C = float(np.imag(1.0/z)/w)
    print("  Z11(%.3g Hz) = %.6g%+.6gj ohm   C_eff %.4g F"
          % (FREQ, z.real, z.imag, C), flush=True)
    # crude plate-pair sanity: eps0*eps_r*A_overlap/d
    eps0 = 8.8541878128e-12
    d = float(m.d[2])
    a_over = float(((m.sigma[:, :, 0] != 0)
                    & (m.sigma[:, :, m.dims[2]-1] != 0)).sum())*d*d
    print("  plate-formula ballpark %.4g F (overlap %.3g m^2; "
          "fringing/vias push the solved value around it)"
          % (eps0*4.2*a_over/(2*d), a_over), flush=True)

    i = np.asarray(x[:S.S.efgsize])
    phi = -np.asarray(x[S.S.efgsize:])      # physical potential
    os.makedirs('results', exist_ok=True)
    written = vtkout.export_currents(
        m, M, i, 'results/pdn_capstone.vti', potentials=phi,
        phases=16, freq=FREQ)
    # material grid: 0 empty / 1 copper / 2 FR4
    ids = m.material_struc()
    vtkout.write_vti('results/pdn_capstone_mat.vti', tuple(m.dims),
                     tuple(float(v) for v in m.d),
                     {'material': np.asarray(ids, dtype=np.float32)})
    written.append('results/pdn_capstone_mat.vti')
    print("  wrote %d files; open results/pdn_capstone.pvd in "
          "ParaView" % len(written), flush=True)


if __name__ == '__main__':
    main()
