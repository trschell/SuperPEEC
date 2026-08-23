# SPDX-License-Identifier: MIT
"""How well does each code place the current, on a COARSE mesh, at high
frequency? -- the measurement behind the SuperPEEC/VoxHenry narrative.

THE ENGINEERING QUESTION. At DC the current fills the conductor and both
codes get R right trivially. As frequency rises the current crowds into
a surface layer of depth delta, and R rises above R_dc. A voxel code
whose cells are BIGGER than delta cannot resolve that layer with the
mesh, so whatever it gets right must come from its BASIS -- the extra
freedom it carries inside each cell. That is the whole ballgame at high
frequency, and the two codes answer it differently:

    VoxHenry    5 unknowns per voxel: Jx, Jy, Jz + J2d + J3d, i.e.
                three uniform currents plus two intra-voxel variation
                functions (verified in
                src_post_process/post_obtain_curr_coefs_on_grid.m,
                which splits x into exactly those five blocks).
    SuperPEEC   1 unknown per cell per direction by default (piecewise
                constant), OPTIONALLY enriched by conduction modes --
                exponential skin profiles anchored to the cell faces and
                corners (equiterminal.conduction_weights).

So the comparison is polynomial enrichment against exponential
enrichment, both on a mesh too coarse to resolve the physics.

THE METRIC. Relative error in R is what an engineer reads, but it hides
the thing being tested: at low frequency EVERY code looks good because
R ~ R_dc and there is no skin effect to get wrong. So we also report

    delivered = (R_code - R_dc) / (R_true - R_dc)

the FRACTION OF THE AC RESISTANCE RISE the code captures on its coarse
mesh. 100% means the basis did the whole job; 0% means it behaved like
a DC solver. This isolates the basis from the trivially-correct part.

R_true comes from mesh refinement in SuperPEEC's PLAIN basis (no
enrichment, so nothing being tested is assumed) extrapolated to h -> 0,
and is only quoted where the ladder demonstrates convergence. R_dc is
the exact DC resistance of the discretised prism where that exists, and
the converged DC solve otherwise.

WE ARE AFTER ENGINEERING ACCURACY. A few percent on R at 10 GHz on a
2-cell-across bar is an excellent result, not a failure; the report is
written in those terms and rounds accordingly.

Usage:
  PYTHONPATH=src:studies python3 studies/skinnarr.py [--only KEY] [--quick]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))
import vhr                                                    # noqa: E402
import equiterminal as eq                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, 'studies', '_narr')
SIGMA = 5.8e7
MU0 = 4e-7 * np.pi


def skin_depth(f, sigma=SIGMA):
    return np.sqrt(2.0 / (2 * np.pi * f * MU0 * sigma))


# ---------------------------------------------------------------- geometry
def bar(nx_cross, dx, length_cells):
    """Solid rectangular bar, nx_cross x nx_cross cells, along x."""
    struc = np.ones((length_cells, nx_cross, nx_cross), dtype=np.int8)
    ports = []
    for j in range(nx_cross):
        for k in range(nx_cross):
            ports.append(('p1', 'P', 0, j, k, '-x'))
            ports.append(('p1', 'N', length_cells - 1, j, k, '+x'))
    return struc, ports


def round_wire(diam_cells, dx, length_cells):
    """Staircase-discretised circular wire along x (VoxHenry's numex2 rule:
    a cell is metal when its CENTRE lies inside the circle)."""
    n = diam_cells
    c = (n - 1) / 2.0
    r = n / 2.0
    yy, zz = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    disc = ((yy - c) ** 2 + (zz - c) ** 2) <= r * r
    struc = np.zeros((length_cells, n, n), dtype=np.int8)
    struc[:, disc] = 1
    ports = []
    for j in range(n):
        for k in range(n):
            if disc[j, k]:
                ports.append(('p1', 'P', 0, j, k, '-x'))
                ports.append(('p1', 'N', length_cells - 1, j, k, '+x'))
    return struc, ports


def hairpin(across, dx, arm_cells):
    """U-bend: two parallel x-arms joined at the far end, so the current
    must TURN THROUGH TWO RIGHT ANGLES. Both ports sit on x-normal faces
    at the near end, which is what SuperPEEC's terminal model requires
    (both port faces must share one axis).

    This is the geometry that tests what VoxHenry's J2d/J3d basis
    functions are actually FOR. On a straight run they measure as idle
    (block norm ~1e-14 on the 2-cell bar) and VoxHenry's R is
    indistinguishable from a plain piecewise-constant solver; the
    hypothesis is that they exist to represent 2-D and 3-D flow patterns
    inside a voxel, i.e. current TURNING, which a straight bar never
    asks for.
    """
    n, a = across, arm_cells
    gap = n                                   # separation between the arms
    W = 2 * n + gap                           # total width in y
    L = a + n                                 # total length in x
    struc = np.zeros((L, W, n), dtype=np.int8)
    struc[:a, :n, :] = 1                      # arm A (y low)
    struc[:a, n + gap:, :] = 1                # arm B (y high)
    struc[a:, :, :] = 1                       # the link joining them
    ports = []
    for j in range(n):
        for k in range(n):
            ports.append(('p1', 'P', 0, j, k, '-x'))
            ports.append(('p1', 'N', 0, n + gap + j, k, '-x'))
    return struc, ports


MODELS = {
    # key: (builder, cells across at the COARSE rung, physical cross-section,
    #       physical length, refinement factors)
    'bar2': dict(kind='bar', across=2, cs=2e-6, length=10e-6,
                 rungs=(1, 2, 3, 4, 6),
                 blurb='2 cells across -- VoxHenry\'s own design point'),
    'bar4': dict(kind='bar', across=4, cs=4e-6, length=10e-6,
                 rungs=(1, 2, 3, 4),
                 blurb='4 cells across'),
    'wire10': dict(kind='wire', across=10, cs=10e-6, length=50e-6,
                   rungs=(1, 2, 3),
                   blurb='numex2 round wire, staircase boundary'),
    'numex1': dict(kind='bar', across=20, cs=10e-6, length=30e-6,
                   rungs=(1, 2),
                   blurb="VoxHenry's flagship numex1 straight conductor, "
                         "at its own shipped 0.5 um mesh"),
    'bend4': dict(kind='bend', across=4, cs=4e-6, length=16e-6,
                  rungs=(1, 2, 3),
                  blurb='hairpin -- current turns through two right angles'),
}


def build(key, mult, freqs):
    """Write the .vhr for one rung; return (path, dx, ncells)."""
    m = MODELS[key]
    across = m['across'] * mult
    dx = m['cs'] / across
    ncell_len = int(round(m['length'] / dx))
    if m['kind'] == 'bar':
        struc, ports = bar(across, dx, ncell_len)
    elif m['kind'] == 'bend':
        struc, ports = hairpin(across, dx, ncell_len)
    else:
        struc, ports = round_wire(across, dx, ncell_len)
    os.makedirs(SCRATCH, exist_ok=True)
    p = os.path.join(SCRATCH, '%s_m%d.vhr' % (key, mult))
    vhr.write_vhr(p, struc, dx, SIGMA, tuple(freqs), ports)
    return p, dx, int(struc.sum())


# ------------------------------------------------------------------ solves
def solve_sp(path, freqs, engine=False, k=None):
    """SuperPEEC at one mesh. engine=False is the plain piecewise-constant
    basis; engine=True switches on the conduction-mode (skin) engine."""
    m = vhr.read_vhr(path)
    out = {}
    for f in freqs:
        M = m.build_tree()
        m.prepare(M, f)
        kw = {}
        if k is not None:
            kw['k'] = k
        S = eq.EquiTerminalSolver(m, M, 0, subdivide=engine,
                                  skin_freq=(f if engine else None),
                                  mode_basis=('conduction' if engine
                                              else 'diff'), **kw)
        t0 = time.perf_counter()
        Z, _i, info = S.solve(f)
        out[f] = dict(R=float(np.real(Z)), L=float(np.imag(Z)/(2*np.pi*f)),
                      wall=time.perf_counter() - t0,
                      resid=float(info.get('residual', np.nan))
                      if isinstance(info, dict) else np.nan)
        del M, S
    return out


VH = os.path.expanduser('~/Documents/octree/VoxHenry')


def solve_vh(path, timeout=3600):
    """Run VoxHenry on the SAME .vhr, on its own terms (5 unknowns per
    voxel: Jx, Jy, Jz, J2d, J3d). Returns {freq: {R, L}} plus the raw
    solution vector split into those five blocks, which is what lets us
    say WHERE VoxHenry put the current rather than only what R it got."""
    import shutil
    import subprocess
    import scipy.io as sio
    name = os.path.basename(path)
    dst = os.path.join(VH, 'Input_files', name)
    shutil.copy(path, dst)
    res = os.path.join(VH, 'Results', name + '-data_R_jL_mat.mat')
    cur = os.path.join(VH, 'Results', name + '-data_curr_plot.mat')
    for f in (res, cur):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, VH_MODEL=name)
    t0 = time.time()
    p = subprocess.run(['octave', '--no-gui', '--quiet', 'vh_exec_auto.m'],
                       cwd=VH, env=env, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if not os.path.exists(res):
        tail = p.stdout.decode('utf8', 'replace').splitlines()[-4:]
        return {'ok': False, 'error': ' | '.join(tail)[:300], 'wall': wall}
    d = sio.loadmat(res)
    Z = np.asarray(d['R_jL_mat'])
    while Z.ndim < 3:
        Z = Z[..., None]
    fr = np.atleast_1d(np.asarray(d['freq_all']).ravel())
    out = {'ok': True, 'wall': wall, 'byfreq': {}}
    for i, f in enumerate(fr):
        out['byfreq'][str(float(f))] = dict(
            R=float(np.real(Z[0, 0, i])),
            L=float(np.imag(Z[0, 0, i]) / (2 * np.pi * float(f))))
    # HOW VOXHENRY SPENT ITS EXTRA UNKNOWNS. x is five stacked blocks of
    # one coefficient per occupied voxel; the ratio of the J2d/J3d energy
    # to the Jx energy says how much of the answer the intra-voxel
    # enrichment is carrying -- 0 would mean the extra basis is idle.
    try:
        c = sio.loadmat(cur) if os.path.exists(cur) else None
    except Exception as exc:
        # The current vector is a BONUS -- it says where VoxHenry put the
        # current. Never let it cost us the R/L the run is really for.
        out['blocks_error'] = repr(exc)[:160]
        c = None
    if c is not None:
        # x is FIVE per-voxel blocks followed by the node potentials, so
        # x.size//5 is WRONG -- it silently mixes nodes into J3d. The
        # voxel count comes from Mc, exactly as VoxHenry's own
        # post_obtain_curr_coefs_on_grid.m does it.
        x = np.asarray(c['x']).ravel()
        nb = int((np.abs(np.asarray(c['Mc'])) > 1e-12).sum())
        blk = [x[i*nb:(i+1)*nb] for i in range(5)]
        nrm = [float(np.linalg.norm(b)) for b in blk]
        out['blocks'] = dict(n_per_block=int(nb),
                             names=['Jx', 'Jy', 'Jz', 'J2d', 'J3d'],
                             norm=nrm,
                             enrich_ratio=(nrm[3] + nrm[4]) / nrm[0]
                             if nrm[0] else None,
                             freq_of_curr_plot=2.5e9)
    return out


def dc_exact(path):
    """Exact DC resistance of the discretised conductor (prismatic only)."""
    m = vhr.read_vhr(path)
    occ = np.asarray(m.struc(), dtype=bool)
    d = float(np.atleast_1d(np.asarray(m.d, dtype=float))[0])
    n_along = occ.sum(axis=(1, 2))
    nz = n_along[n_along > 0]
    if nz.size != occ.shape[0] or not np.all(nz == nz[0]):
        return None
    return occ.shape[0] * d / (SIGMA * int(nz[0]) * d * d)


def extrapolate(hs, qs):
    """Fit Q(h)=Q0+C h^p; return (Q0, p) or None. Also a Richardson check."""
    hs, qs = np.asarray(hs, float), np.asarray(qs, float)
    if hs.size < 3:
        return None
    best = None
    for p in np.arange(0.25, 3.01, 0.01):
        A = np.vstack([np.ones_like(hs), hs ** p]).T
        sol, *_ = np.linalg.lstsq(A, qs, rcond=None)
        r = float(np.sum((A @ sol - qs) ** 2))
        if best is None or r < best[2]:
            best = (float(sol[0]), float(p), r)
    d1, d2 = qs[-2] - qs[-3], qs[-1] - qs[-2]
    rich = None
    if abs(d2) > 0 and d1 * d2 > 0 and abs(d1 / d2) > 1.0000001:
        rich = float(qs[-1] + d2 / (d1 / d2 - 1.0))
    return dict(Q0=best[0], p=best[1], richardson=rich)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='studies/skinnarr_results.json')
    args = ap.parse_args()

    # Frequencies chosen to sweep dx/delta through 1 at the COARSE rung.
    freqs = [1.0, 1e8, 1e9, 2.5e9, 1e10, 2.5e10, 1e11]
    if args.quick:
        freqs = [1.0, 1e9, 1e10, 1e11]

    D = {}
    if os.path.exists(args.out):
        D = json.load(open(args.out))
    keys = [args.only] if args.only else list(MODELS)
    for key in keys:
        rec = D.setdefault(key, {'blurb': MODELS[key]['blurb'], 'rungs': {}})
        rec['freqs'] = freqs
        for mult in MODELS[key]['rungs']:
            if str(mult) in rec['rungs']:
                print('  %s m=%d cached' % (key, mult), flush=True)
                continue
            p, dx, nc = build(key, mult, freqs)
            print('== %s mult=%d  dx=%.4g um  cells=%d' % (key, mult,
                                                           dx*1e6, nc),
                  flush=True)
            t0 = time.time()
            plain = solve_sp(p, freqs, engine=False)
            row = dict(dx=dx, cells=nc, plain={str(f): v
                                               for f, v in plain.items()},
                       r_dc=dc_exact(p), wall=time.time() - t0)
            # the skin engine only at the COARSE rung -- that is where the
            # question lives; on a fine mesh nothing is being asked of it.
            if mult <= 2:
                t0 = time.time()
                try:
                    eng = solve_sp(p, freqs, engine=True)
                    row['engine'] = {str(f): v for f, v in eng.items()}
                except Exception as exc:
                    row['engine_error'] = repr(exc)[:200]
                    print('   engine FAILED: %r' % (exc,), flush=True)
                row['engine_wall'] = time.time() - t0
            if True:
                try:
                    row['voxhenry'] = solve_vh(p)
                    v = row['voxhenry']
                    print('    VoxHenry %s  %.1f s'
                          % ('ok' if v.get('ok') else 'FAILED: '
                             + str(v.get('error'))[:70], v.get('wall', 0)),
                          flush=True)
                except Exception as exc:
                    row['voxhenry'] = {'ok': False, 'error': repr(exc)[:200]}
                    print('    VoxHenry EXCEPTION %r' % (exc,), flush=True)
            rec['rungs'][str(mult)] = row
            for f in freqs:
                print('    f=%-9.3g dx/delta=%-6.2f R=%.6g'
                      % (f, dx/skin_depth(f), plain[f]['R']), flush=True)
            with open(args.out, 'w') as fh:
                json.dump(D, fh, indent=1)
    print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
