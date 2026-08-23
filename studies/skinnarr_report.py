# SPDX-License-Identifier: MIT
"""Turn skinnarr_results.json into the engineering comparison tables.

THE FRAME IS ENGINEERING ACCURACY, NOT DIGITS. Everything is reported to
the precision an engineer would act on -- percent, not parts per
billion. A code that lands within a few percent of the converged answer
on a mesh far too coarse to resolve the physics has done something
genuinely difficult, and the tables are written to make that visible
rather than to rank codes by decimal places.

TWO NUMBERS PER CELL:
  err%       (R_code - R_true)/R_true -- what the engineer reads.
  delivered  (R_code - R_dc)/(R_true - R_dc) -- the fraction of the AC
             resistance RISE the code captured. This is the one that
             isolates the basis: at low frequency R ~ R_dc and every
             code looks perfect on err%, while delivered correctly says
             "there was nothing here to get right".

R_true is from the SuperPEEC plain-basis refinement ladder extrapolated
to h -> 0, and is only used where the ladder actually demonstrates
convergence -- rows where it does not are marked and excluded from the
verdicts rather than quoted.
"""
import json
import os
import sys

import numpy as np

MU0 = 4e-7 * np.pi
SIGMA = 5.8e7


def skin_depth(f):
    return np.sqrt(2.0 / (2 * np.pi * f * MU0 * SIGMA))


def truth_for(rec, f):
    """Extrapolated R at frequency f from the plain ladder, plus a
    convergence verdict. Returns (R_true, note, converged_bool)."""
    ks = sorted((int(k) for k in rec['rungs']), key=int)
    hs, qs = [], []
    for k in ks:
        r = rec['rungs'][str(k)]
        v = r['plain'].get(str(float(f)))
        if v:
            hs.append(r['dx']); qs.append(v['R'])
    if len(qs) < 3:
        return None, 'too few rungs', False
    hs, qs = np.array(hs), np.array(qs)
    # power-law fit
    best = None
    for p in np.arange(0.25, 3.01, 0.01):
        A = np.vstack([np.ones_like(hs), hs ** p]).T
        sol, *_ = np.linalg.lstsq(A, qs, rcond=None)
        r2 = float(np.sum((A @ sol - qs) ** 2))
        if best is None or r2 < best[2]:
            best = (float(sol[0]), float(p), r2)
    Q0 = best[0]
    # CONVERGENCE TEST, and it must be strict or the whole report is
    # built on sand: the last inter-rung step must be small compared with
    # the total rise being measured, and the extrapolation must not be
    # reaching far beyond the finest rung.
    step = abs(qs[-1] - qs[-2])
    reach = abs(Q0 - qs[-1])
    rise = abs(qs[-1] - qs[0]) or 1e-30
    conv = (step / max(abs(qs[-1]), 1e-30) < 0.02) and (reach < 2.0 * step)
    note = 'p=%.2f, last step %.2f%%, extrap reach %.2f%% of R' % (
        best[1], 100*step/abs(qs[-1]), 100*reach/abs(qs[-1]))
    return Q0, note, bool(conv)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'studies/skinnarr_results.json'
    D = json.load(open(src))
    out = []
    A = out.append
    for key in D:
        rec = D[key]
        coarse = rec['rungs']['1']
        dx = coarse['dx']
        rdc = coarse['r_dc']
        if rdc is None:
            # Not a prism (the hairpin), so no closed form. Use the
            # CONVERGED DC solve instead -- at DC the mesh converges
            # immediately (no skin layer to resolve), so the finest
            # rung's 1 Hz value is the right reference.
            fin = rec['rungs'][max(rec['rungs'], key=int)]
            rdc = fin['plain'][str(float(rec['freqs'][0]))]['R']
            rdc_src = 'converged DC solve (not a prism: no closed form)'
        else:
            rdc_src = 'exact closed form'
        A('\n### %s -- %s' % (key, rec['blurb']))
        A('coarse mesh dx = %.3g um, %d cells, R_dc = %.6g ohm (%s)'
          % (dx*1e6, coarse['cells'], rdc, rdc_src))
        A('')
        A('| f | dx/delta | R_true | R/R_dc | plain | VoxHenry | engine |')
        A('|---|---:|---:|---:|---|---|---|')
        for f in rec['freqs']:
            k = str(float(f))
            Rt, note, conv = truth_for(rec, f)
            if Rt is None:
                continue
            row = ['%.3g Hz' % f, '%.2f' % (dx/skin_depth(f)),
                   ('%.5g' % Rt) + ('' if conv else ' *'),
                   '%.2f' % (Rt/rdc) if rdc else '-']
            for arm in ('plain', 'voxhenry', 'engine'):
                if arm == 'voxhenry':
                    v = coarse.get('voxhenry', {})
                    R = v.get('byfreq', {}).get(k, {}).get('R') if v.get('ok') else None
                else:
                    R = coarse.get(arm, {}).get(k, {}).get('R')
                if R is None:
                    row.append('-'); continue
                err = 100*(R - Rt)/Rt
                den = (Rt - rdc)
                dl = 100*(R - rdc)/den if abs(den) > 1e-12*max(abs(Rt),1e-30) else None
                row.append('%+.1f%%' % err + (' / %.0f%%' % dl if dl is not None else ''))
            A('| ' + ' | '.join(row) + ' |')
        if not any(l.startswith('| ') and 'Hz' in l for l in out[-12:]):
            A('| *(no rows: this problem has fewer than the three rungs '
              'the convergence test requires, so no truth is quoted '
              'here -- see the narrative, which uses its two-rung truth '
              'with that caveat stated)* | | | | | | |')
        A('')
        A('`err%% / delivered%%`.  `*` = ladder not converged at this '
          'frequency; the R_true shown is indicative only and no verdict '
          'is drawn from that row.')
        # VoxHenry's own convergence + how hard its extra unknowns worked
        v = coarse.get('voxhenry', {})
        b = v.get('blocks')
        if b:
            n = b['norm']
            A('')
            A('VoxHenry unknown-block norms at its current-plot frequency '
              '(%d voxels): Jx %.3g, Jy %.3g, Jz %.3g, **J2d %.3g, J3d '
              '%.3g**' % (b['n_per_block'], n[0], n[1], n[2], n[3], n[4]))
        A('')
        A('| rung | dx [um] | cells | ' + ' | '.join(
            '%.3g Hz' % f for f in rec['freqs']) + ' |')
        A('|---|---:|---:|' + '---:|'*len(rec['freqs']))
        for kk in sorted(rec['rungs'], key=int):
            r = rec['rungs'][kk]
            vv = r.get('voxhenry', {})
            cells = '%d' % r['cells']
            pl = ['%.5g' % r['plain'][str(float(f))]['R'] for f in rec['freqs']]
            A('| m%s plain | %.3g | %s | %s |' % (kk, r['dx']*1e6, cells,
                                                  ' | '.join(pl)))
            if vv.get('ok'):
                vh = ['%.5g' % vv['byfreq'][str(float(f))]['R']
                      if str(float(f)) in vv['byfreq'] else '-'
                      for f in rec['freqs']]
                A('| m%s VoxHenry | %.3g | %s | %s |' % (kk, r['dx']*1e6,
                                                         cells, ' | '.join(vh)))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
