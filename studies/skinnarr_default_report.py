# SPDX-License-Identifier: MIT
"""Corrected engine tables: DEFAULT settings only.

Reads skinnarr_default_results.json (the TOML/`[model] vhr` path, every
skin knob at its default) against the plain-basis refinement ladder in
skinnarr_results.json. Emits the markdown the narrative uses.
"""
import json
import os
import sys

import numpy as np

MU0 = 4e-7*np.pi
SIG = 5.8e7
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sd(f):
    return np.sqrt(2/(2*np.pi*f*MU0*SIG))


def main():
    C = json.load(open(os.path.join(ROOT, 'studies',
                                    'skinnarr_results.json')))
    D = json.load(open(os.path.join(ROOT, 'studies',
                                    'skinnarr_default_results.json')))
    out = []
    A = out.append
    A('| problem | f | dx/δ | R_true | plain | VoxHenry | '
      'SuperPEEC (defaults) |')
    A('|---|---:|---:|---:|---:|---:|---:|')
    for key in ('bar2', 'bar4', 'bend4', 'wire10', 'numex1'):
        if key not in D:
            continue
        rec = C[key]
        c = rec['rungs']['1']
        dx = c['dx']
        ks = sorted(rec['rungs'], key=int)
        vh = c.get('voxhenry', {})
        for f in (1e9, 2.5e9, 1e10, 2.5e10, 1e11):
            k = str(float(f))
            if k not in c['plain'] or str(f) not in D[key]:
                continue
            if 'R' not in D[key][str(f)]:
                continue
            Rt = rec['rungs'][ks[-1]]['plain'][k]['R']
            pl = c['plain'][k]['R']
            rv = (vh.get('byfreq', {}).get(k, {}).get('R')
                  if vh.get('ok') else None)
            dv = D[key][str(f)]['R']

            def e(R):
                return '%+.1f%%' % (100*(R - Rt)/Rt) if R is not None else '—'
            A('| %s | %.3g Hz | %.2f | %.5g | %s | %s | **%s** |'
              % (key, f, dx/sd(f), Rt, e(pl), e(rv), e(dv)))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
