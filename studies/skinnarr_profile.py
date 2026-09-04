# SPDX-License-Identifier: MIT
"""WHERE did each code put the current? -- the radial/depth profile.

R tells you how much the two codes disagree; it does not tell you WHY.
This recovers the actual current distribution over the cross-section
from each code on the SAME coarse mesh, and from a refined mesh as the
reference, so the narrative can talk about current rather than ohms.

  SuperPEEC   filament currents -> vtkout.current_density -> Jx(x,y,z)
  VoxHenry    the saved solution vector x, split into its five per-voxel
              blocks exactly as post_obtain_curr_coefs_on_grid.m does;
              Jx is the uniform part of each voxel.

Profiles are taken at the MID-SECTION (half way along the bar) so the
port end-effects do not contaminate them, and are reported as |J|
normalised to the cross-section mean, which makes meshes of different
resolution directly comparable.

The scalar summary is the CROWDING FACTOR: mean |J| over the cells
touching the surface, divided by mean |J| over the whole section. 1.0 is
a DC-like uniform distribution; the converged high-frequency value is
what a code has to reproduce.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'studies'))
import vhr                                                    # noqa: E402
import vtkout                                                 # noqa: E402
import equiterminal as eq                                     # noqa: E402
from skinnarr import SCRATCH, VH                              # noqa: E402


def sp_profile(path, freq, engine=False):
    """Mid-section |Jx| map from SuperPEEC (occupied cells only)."""
    m = vhr.read_vhr(path)
    M = m.build_tree()
    m.prepare(M, freq)
    S = eq.EquiTerminalSolver(m, M, 0, enrich=(dict(k=engine, f_ref=freq)
                                             if engine else None))
    Z, i, _info = S.solve(freq)
    J = vtkout.current_density(M, np.asarray(i)[:S.efg], m.dims)
    occ = np.asarray(m.struc(), dtype=bool)
    mid = m.dims[0] // 2
    jx = np.abs(J[mid, :, :, 0])
    return jx, occ[mid], float(np.real(Z))


def vh_profile(name, freq_index=None):
    """Mid-section |Jx| map from VoxHenry's saved solution vector."""
    import scipy.io as sio
    cur = os.path.join(VH, 'Results', name + '-data_curr_plot.mat')
    if not os.path.exists(cur):
        return None, None
    c = sio.loadmat(cur)
    Mc = np.asarray(c['Mc'])
    x = np.asarray(c['x']).ravel()
    occ = np.abs(Mc) > 1e-12
    nb = int(occ.sum())
    Jx = x[:nb]
    L, M_, N = Mc.shape
    grid = np.zeros((L, M_, N), dtype=complex)
    # VoxHenry fills in mz, my, mx order (see its own post-processor).
    dum = 0
    idx = np.argwhere(np.transpose(occ, (2, 1, 0)))      # (mz,my,mx)
    for (mz, my, mx) in idx:
        grid[mx, my, mz] = Jx[dum]; dum += 1
    mid = L // 2
    return np.abs(grid[mid]), occ[mid]


def crowding(jx, mask):
    """mean |J| on surface-touching cells / mean |J| over the section."""
    if mask.sum() == 0:
        return float('nan')
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                         & mask[1:-1, :-2] & mask[1:-1, 2:])
    surf = mask & ~inner
    if surf.sum() == 0:
        return float('nan')
    return float(jx[surf].mean() / jx[mask].mean())


def radial(jx, mask):
    """|J| vs normalised distance from the section centre, mean-normalised."""
    ii, jj = np.nonzero(mask)
    c0, c1 = (mask.shape[0]-1)/2.0, (mask.shape[1]-1)/2.0
    r = np.hypot(ii - c0, jj - c1)
    r = r/r.max() if r.max() > 0 else r
    v = jx[mask]/jx[mask].mean()
    o = np.argsort(r)
    return r[o], v[o]


def main():
    freq = float(sys.argv[1]) if len(sys.argv) > 1 else 1e10
    out = {}
    for key, coarse, fine in (('wire10', 'wire10_m1.vhr', 'wire10_m3.vhr'),
                              ('bar4', 'bar4_m1.vhr', 'bar4_m4.vhr'),
                              ('bar2', 'bar2_m1.vhr', 'bar2_m6.vhr')):
        cp = os.path.join(SCRATCH, coarse)
        fp = os.path.join(SCRATCH, fine)
        if not (os.path.exists(cp) and os.path.exists(fp)):
            print('skip %s (rungs not built)' % key); continue
        rec = {}
        jx, mk, R = sp_profile(cp, freq, engine=False)
        rec['plain'] = dict(crowd=crowding(jx, mk), R=R)
        jx_e, mk_e, R_e = sp_profile(cp, freq, engine=True)
        rec['engine'] = dict(crowd=crowding(jx_e, mk_e), R=R_e)
        jxv, mkv = vh_profile(coarse)
        if jxv is not None:
            rec['voxhenry'] = dict(crowd=crowding(jxv, mkv))
        jxf, mkf, Rf = sp_profile(fp, freq, engine=False)
        rec['refined'] = dict(crowd=crowding(jxf, mkf), R=Rf)
        out[key] = rec
        print('%-8s f=%.3g   crowding factor (surface |J| / mean |J|)' % (key, freq))
        for arm in ('plain', 'engine', 'voxhenry', 'refined'):
            if arm in rec:
                print('   %-9s %.3f%s' % (arm, rec[arm]['crowd'],
                                          '   R=%.5g' % rec[arm]['R']
                                          if 'R' in rec[arm] else ''))
        print()
    with open(os.path.join(ROOT, 'studies',
                           'skinnarr_profiles.json'), 'w') as fh:
        json.dump({'freq': freq, 'models': out}, fh, indent=1)


if __name__ == '__main__':
    main()
