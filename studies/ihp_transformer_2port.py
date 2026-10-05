# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Transformer 2-port from TWO equipotential solves.

Each run drives one winding (its TOML's equipotential [port]) and reads
the OTHER winding's open-circuit voltage at its lead-end cells with
EquiTerminalSolver.probe_voltage -- the readout's work-conjugate
identity applied to a second current pattern, with the same Gram
correction -- so drive 1 gives Z11 and Z21, drive 2 gives Z22 and Z12.
Reciprocity Z12 = Z21 is the built-in check; k = Im M / sqrt(L1 L2).

  python3 studies/ihp_transformer_2port.py p1.toml p2.toml
      both drives in one process, then the table
  python3 studies/ihp_transformer_2port.py p1.toml p2.toml --only 1 --json d1.json
  python3 studies/ihp_transformer_2port.py p1.toml p2.toml --only 2 --json d2.json
  python3 studies/ihp_transformer_2port.py --combine d1.json d2.json
      one drive per process (the large rungs: each process then holds
      one model), combined afterwards

Run from the repository root (it imports src/).
"""
import os, sys, time
sys.path[:0] = ['src']
os.environ.setdefault('SPPEEC_BACKEND', 'opencl')
import numpy as np
import sppeec_input                                           # noqa: E402

import json
files = sys.argv[1:3]
only = int(sys.argv[sys.argv.index('--only') + 1]) if '--only' in sys.argv else None
outj = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
if '--combine' in sys.argv:
    # merge per-drive JSON results: --combine a.json b.json
    js = sys.argv[sys.argv.index('--combine') + 1:]
    Z, freqs = {}, None
    for jf in js:
        d = json.load(open(jf))
        freqs = d['freqs']
        for key, (re_, im_) in d['Z'].items():
            i, j, f = key.split(',')
            Z[(int(i), int(j), float(f))] = complex(re_, im_)
    probs = None
else:
    probs = [sppeec_input.load(f) for f in files]
    freqs = list(probs[0].freqs)
    cells = []
    for pr in probs:
        _, pf, nf = pr.ports_faces[0]
        cells.append(([c[:3] for c in pf], [c[:3] for c in nf]))
    Z = {}
for k, pr in enumerate(probs or []):
    if only is not None and k != only - 1:
        continue
    other = 1 - k
    m = pr.model(); M = pr.tree(m)
    t = time.time()
    sw = pr.sweeper(m, M)
    sw.S.keep_probe = True
    print("drive %d: setup %.0f s" % (k + 1, time.time() - t), flush=True)
    for f in pr.freqs:
        z, info = sw.solve(f)
        v = sw.S.probe_voltage(*cells[other])
        Z[(k, k, f)] = z
        Z[(other, k, f)] = v
        print("  f %.3g: Z%d%d = %s   Z%d%d = %s   (%d mv)"
              % (f, k + 1, k + 1, z, other + 1, k + 1, v, info['matvecs']),
              flush=True)
    del sw, M, m
if outj:
    json.dump(dict(freqs=freqs, Z={"%d,%d,%r" % key: [v.real, v.imag]
                                   for key, v in Z.items()}), open(outj, 'w'))
if only is not None:
    raise SystemExit
print("\n   f [Hz]    L1 [nH]   L2 [nH]   M21 [nH]  M12 [nH]  recip     k      R12 [ohm]")
for f in freqs:
    w = 2*np.pi*f
    z11, z22, z21, z12 = Z[(0, 0, f)], Z[(1, 1, f)], Z[(1, 0, f)], Z[(0, 1, f)]
    L1, L2 = z11.imag/w, z22.imag/w
    Mm = 0.5*(z21 + z12)
    print("%9.3g  %8.4f  %8.4f  %8.4f  %8.4f  %.1e  %.4f  %8.4f"
          % (f, L1*1e9, L2*1e9, z21.imag/w*1e9, z12.imag/w*1e9,
             abs(z21 - z12)/abs(Mm), (Mm.imag/w)/np.sqrt(L1*L2), Mm.real))
