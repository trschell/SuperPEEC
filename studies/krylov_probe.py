# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Residual trajectory of one equipotential solve under a chosen Krylov
basis precision -- the single-vs-double probe.

Builds the doctrine TOML through the user-facing entry point, then
calls EquiTerminalSolver.solve directly with ``precision`` and
``maxiter`` overridden (everything else at the file's defaults), while
a thread samples the status file every few seconds and records every
(matvecs, residual) change. Prints the trajectory and the solve info.

  python3 studies/krylov_probe.py model.toml --precision double \
      --maxiter 8 --out traj_double.csv
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('toml')
    ap.add_argument('--precision', choices=('auto', 'single', 'double'),
                    default='auto')
    ap.add_argument('--maxiter', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    status_path = args.out + '.status.json'
    os.environ['SPPEEC_STATUS'] = status_path
    import sppeec_threads                                      # noqa
    import sppeec_input                                        # noqa
    os.chdir(ROOT)
    t0 = time.perf_counter()
    prob = sppeec_input.load(args.toml)
    m = prob.model()
    M = prob.tree(m)
    m.prepare(M, prob.freqs[0])
    sw = prob.sweeper(m, M)
    S = sw.S
    t_setup = time.perf_counter() - t0
    print('setup %.0f s  unknowns %d  precision %s  maxiter %d' %
          (t_setup, S.meshsize, args.precision, args.maxiter), flush=True)

    traj, stop = [], threading.Event()

    def sampler():
        last = None
        while not stop.is_set():
            try:
                s = json.load(open(status_path))
                d = s['task'].get('detail', {})
                key = (d.get('matvecs'), d.get('residual'))
                if key != last and key[1] is not None:
                    traj.append((time.perf_counter() - t1, key[0], key[1]))
                    last = key
            except Exception:
                pass
            stop.wait(3.0)
    t1 = time.perf_counter()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    Z, i, info = S.solve(float(prob.freqs[0]), current=prob.current,
                         rtol=prob.rtol, method=prob.method,
                         precision=args.precision, maxiter=args.maxiter)
    stop.set()
    th.join()
    with open(args.out, 'w') as f:
        f.write('t_s,matvecs,residual\n')
        for t, mv, r in traj:
            f.write('%.1f,%s,%.6e\n' % (t, mv, r))
    print('trajectory (matvecs:residual): ' + '  '.join(
        '%s:%.3g' % (mv, r) for _, mv, r in traj), flush=True)
    print('RESULT precision=%s Z=%r matvecs=%d flag=%s resid=%.3e '
          'solve %.0f s' % (args.precision, Z, info['matvecs'],
                            info['flag'], info['residual'], info['time']))


if __name__ == '__main__':
    main()
