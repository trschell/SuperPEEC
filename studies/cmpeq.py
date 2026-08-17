# SPDX-License-Identifier: MIT
"""Same VoxHenry comparison, but through the EQUIPOTENTIAL TERMINAL
(and optionally the skin-effect redistribution), instead of the
prescribed-current port + analytic series correction in port_impedance."""
import os, sys, time, numpy as np, scipy.io, vhr, equiterminal as eq

VH = 'VoxHenry'
FREQS = [1.0, 1e9, 2.5e9, 5e9, 1e10]
PAIRS = [('straight_cond1_len30.0u_wid10.0u_dist20.0u.vhr',
          'results_numex1_straight_conductor'),
         ('wire_len50.0u_dia10.0u.vhr', 'results_numex2_wire')]


def reference(refdir):
    d = scipy.io.loadmat(os.path.join(VH, refdir, 'data_R_jL_mat.mat'))
    return dict(zip(np.asarray(d['freq_all']).ravel(),
                    np.asarray(d['R_jL_mat']).ravel()))


sub = len(sys.argv) > 1 and sys.argv[1] == 'skin'
for name, refdir in PAIRS:
    m = vhr.read_vhr(os.path.join(VH, 'Input_files', name))
    ref = reference(refdir)
    M = m.build_tree()
    print("=" * 74)
    print("%s   subdivide=%s" % (name, sub))
    print("  freq        SuperPEEC R     VoxHenry R   Rratio | "
          "SuperPEEC L     VoxHenry L   Lratio")
    for f in FREQS:
        m.prepare(M, f)
        kw = {}
        if sub:
            k = eq.recommend_subdivision(m.dx, m.uniform_sigma(), f)
            kw = dict(subdivide=(k > 1), skin_freq=f)
        try:
            S = eq.EquiTerminalSolver(m, M, 0, **kw)
            t = time.perf_counter()
            Z, i, info = S.solve(f)
            R, L = Z.real, (Z.imag/(2*np.pi*f) if f > 0 else float('nan'))
            k = min(ref, key=lambda x: abs(x-f))
            Rv, Lv = ref[k].real, ref[k].imag   # R_jL_mat stores R + jL directly
            print("  %-11.3g %-12.6g %-12.6g %-6.4f | %-12.6g %-12.6g %-6.4f"
                  % (f, R, Rv, R/Rv, L, Lv, L/Lv), flush=True)
        except Exception as e:
            print("  %-11.3g FAILED %s: %s" % (f, type(e).__name__, str(e)[:70]))
    del M
