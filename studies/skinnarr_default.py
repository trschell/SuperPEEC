# SPDX-License-Identifier: MIT
"""Re-measure the skin engine THROUGH THE USER-FACING ENTRY POINT.

WHY THIS FILE EXISTS. `studies/skinnarr.py` hand-built an
`EquiTerminalSolver`, which silently overrode what a user actually gets:

    knob             TOML default (sppeec_input)   hand-built call
    skin.mode        'auto'                        forced on/off
    skin.basis       'conduction'                  conduction (matched)
    skin.k           auto min(12,max(7,2dx/delta)) subdivide=True
    skin.rc_uu/cross width-scaled _auto_rc         API fixed (3, 4)
    skin.boundary_only  TRUE                       FALSE  <- the big one

Measuring a hand-assembled configuration and reporting it as the code's
behaviour produced a false headline: "the engine overshoots +70.9% at
100 GHz". With the DEFAULT boundary_only the same case is -7.7%, and
cheaper. The pathology had been found in 2026-08-17 and already fixed by
making that the default; only `validate_input_lppr` caught the error.

So this drives the TOML path -- `sppeec_input.loads` -> model -> tree ->
sweeper -> solve -- which is what the CLI runs and what a user gets.

Usage: PYTHONPATH=src:studies python3 studies/skinnarr_default.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'studies'))
import sppeec_input                                           # noqa: E402
import vhr                                                    # noqa: E402
from skinnarr import SCRATCH                                  # noqa: E402

FREQS = [1e9, 2.5e9, 1e10, 2.5e10, 1e11]


def toml_for(path, freqs):
    """A minimal TOML pointing at the .vhr, with every skin knob left to
    its DEFAULT -- the documented user route.

    `[model] vhr = "..."` is a first-class part of the format (schema
    `'model': {'vhr'}`, mutually exclusive with `[grid]`), and is THE
    way to run a VoxHenry file through the doctrine path. It was broken
    until 2026-08-23 -- `model()` returned a bare `VoxelModel(path)`
    whose dims is None -- which is why studies hand-built solvers and
    silently got non-default settings.
    """
    m = vhr.read_vhr(path)
    occ = np.asarray(m.struc(), dtype=bool)
    nx = occ.shape[0]
    pf = [(y, z) for y in range(occ.shape[1]) for z in range(occ.shape[2])
          if occ[0, y, z]]
    nf = [(y, z) for y in range(occ.shape[1]) for z in range(occ.shape[2])
          if occ[nx-1, y, z]]
    return '\n'.join([
        '[model]', 'vhr = "%s"' % path,
        '[port]', 'equipotential = true',
        'p_faces = [%s]' % ', '.join('[0, %d, %d, "-x"]' % yz for yz in pf),
        'n_faces = [%s]' % ', '.join(
            '[%d, %d, %d, "+x"]' % (nx-1, y, z) for (y, z) in nf),
        '[solve]', 'freq = [%s]' % ', '.join(repr(f) for f in freqs)])


def main():
    out = {}
    for key in ('bar2', 'bar4', 'bend4', 'wire10', 'numex1'):
        path = os.path.join(SCRATCH, '%s_m1.vhr' % key)
        if not os.path.exists(path):
            print('skip %s' % key); continue
        rec = {}
        for f in FREQS:
            doc = toml_for(path, [f])
            try:
                pr = sppeec_input.loads(doc)
                m = pr.model()
                sw = pr.sweeper(m, pr.tree(m))
                t0 = time.perf_counter()
                Z, _info = sw.solve(f)
                rec[str(f)] = dict(R=float(np.real(Z)),
                                   wall=time.perf_counter() - t0,
                                   skin=dict(sw.S.enrich)
                                   if sw.S.enrich else None)
                sk = rec[str(f)]['skin'] or {}
                print('  %-8s f=%-9.3g R=%-11.6g  subdiv=%-5s bo=%-5s '
                      'rc=(%s,%s)' % (key, f, rec[str(f)]['R'],
                                      sk.get('k'),
                                      sk.get('reach'),
                                      sk.get('rc_uu'), sk.get('rc_cross')),
                      flush=True)
            except Exception as exc:
                rec[str(f)] = {'error': repr(exc)[:200]}
                print('  %-8s f=%-9.3g FAILED %r' % (key, f, exc), flush=True)
        out[key] = rec
    with open(os.path.join(ROOT, 'studies',
                           'skinnarr_default_results.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\nwrote studies/skinnarr_default_results.json')


if __name__ == '__main__':
    main()
