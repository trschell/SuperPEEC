# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for sppeec_status (phase 1, 2026-08-23).

Three layers:

A. UNIT -- the status object itself: schema completeness, seq
   monotonicity, task-stack accounting, the matvec/budget percent,
   nesting-safe freq tasks, and ATOMICITY (a reader hammering the file
   while the writer publishes must never see a torn document).

B. END-TO-END -- a real one-point CLI run (module3wire, the same
   anchor problem validate_cli uses) with SPPEEC_STATUS set via the
   environment: the file must land in state 'done' with the model
   identity, resolved params, one result row and a nonzero matvec
   count.

C. NO-BEHAVIOUR-CHANGE -- the same run WITHOUT status enabled must
   print byte-identical results (same R to every printed digit, same
   matvec count). Instrumentation that moves the answer is a bug by
   the phase-1 design rule. (SPPEEC_GPU=0 pins the deterministic CPU
   path, same as the anchor policy.)
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

FAIL = []
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------- A. unit
def unit():
    os.environ.pop('SPPEEC_STATUS', None)
    import sppeec_status as st
    st.disable()
    path = os.path.join(tempfile.mkdtemp(prefix='sppeec_status_'),
                        'status.json')

    check('disabled hooks are no-ops', not st.enabled() and
          st.tick_matvec() is None and st.finish() is None)
    with st.task('nothing') as t:       # inert task when disabled
        t.tick()
        t.set(pct=50)

    st.enable(path=path, interval=0.0)
    d = json.load(open(path))
    need = ('schema', 'pid', 'seq', 'state', 'started_at',
            'updated_at', 'model', 'params', 'sweep', 'task',
            'overall', 'mem', 'counters')
    check('schema fields present on first write',
          all(k in d for k in need) and d['schema'] == 1
          and d['state'] == 'running' and d['pid'] == os.getpid())

    st.run_meta(model=dict(name='unit'), params=dict(rtol=1e-6))
    st.sweep_meta([1e6, 1e7])
    seqs, pcts = [], []
    with st.freq_task(1e6):
        with st.freq_task(1e6):         # nested: must not double-count
            pass
        with st.task('krylov', budget=10, matvecs=0):
            for _ in range(5):
                st.tick_matvec()
            d = json.load(open(path))
            seqs.append(d['seq'])
            pcts.append(d['overall']['pct'])
            check('krylov percent = matvecs/budget',
                  abs(d['task']['pct'] - 50.0) < 1e-9
                  and d['task']['detail']['matvecs'] == 5
                  and d['counters']['matvecs_total'] == 5)
            check('sweep state mid-point',
                  d['sweep']['current_freq'] == 1e6
                  and d['sweep']['index'] == 0
                  and d['sweep']['n'] == 2)
    st.record_result(1e6, R=0.5, matvecs=5, time_s=0.01)
    d = json.load(open(path))
    seqs.append(d['seq'])
    pcts.append(d['overall']['pct'])
    check('one point done: results row + index + overall in (0,100)',
          d['sweep']['results'] == [{'f': 1e6, 'R': 0.5, 'matvecs': 5.0,
                                     'time_s': 0.01}]
          and d['sweep']['index'] == 1
          and 0 < d['overall']['pct'] < 100)
    check('nested freq task did not double-count',
          d['sweep']['index'] == 1)
    with st.freq_task(1e7):
        pass
    d = json.load(open(path))
    seqs.append(d['seq'])
    pcts.append(d['overall']['pct'])
    check('sweep complete reads 100%',
          abs(d['overall']['pct'] - 100.0) < 1e-9
          and d['overall']['eta_s'] is not None)
    check('seq strictly increases', all(b > a for a, b in
                                        zip(seqs, seqs[1:])))
    check('overall never decreases', all(b >= a - 1e-9 for a, b in
                                         zip(pcts, pcts[1:])))

    # atomicity: hammer-read while force-publishing
    errs = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                json.load(open(path))
            except Exception as exc:
                errs.append(exc)

    th = threading.Thread(target=reader)
    th.start()
    for _ in range(300):
        st._S.publish(force=True) if hasattr(st, '_S') else None
    for _ in range(300):
        st.tick_matvec()
    stop.set()
    th.join()
    check('atomic writes: no torn reads under load', not errs,
          repr(errs[:1]) if errs else '')

    st.finish('done')
    d = st.read(path)
    check('finish() lands state=done, reader adds liveness',
          d['state'] == 'done' and d['_alive'] is True
          and d['_stale_s'] < 60)
    check('format_line renders', isinstance(st.format_line(d), str)
          and 'done' in st.format_line(d))
    st.disable()


# ------------------------------------- B/C. end-to-end + A/B identical
def run_cli(extra_env, *args):
    env = dict(os.environ, SPPEEC_GPU='0')
    env.pop('SPPEEC_STATUS', None)
    env.update(extra_env)
    return subprocess.run(
        [PY, os.path.join(ROOT, 'src', 'sppeec_cli.py'),
         'examples/module3wire.toml', '--freq', '1e5'] + list(args),
        capture_output=True, text=True, cwd=ROOT, env=env)


def endtoend():
    import sppeec_status as st
    out = tempfile.mkdtemp(prefix='sppeec_status_e2e_')
    spath = os.path.join(out, 'status.json')

    a = run_cli({})
    b = run_cli({'SPPEEC_STATUS': spath})
    check('plain run exits 0', a.returncode == 0,
          (a.stderr.strip().splitlines() or [''])[-1][:70])
    check('status run exits 0', b.returncode == 0,
          (b.stderr.strip().splitlines() or [''])[-1][:70])
    check('C: stdout byte-identical with status on vs off '
          '(same R every digit, same matvecs)', a.stdout == b.stdout)

    d = st.read(spath)
    check('B: final state done, schema 1',
          d.get('state') == 'done' and d.get('schema') == 1)
    check('B: model identity resolved',
          d['model'].get('name') == 'module3wire'
          and d['model'].get('formulation') == 'LpR'
          and d['model'].get('cells', 0) > 0)
    check('B: resolved params published',
          d['params'].get('method') in ('lgmres', 'bicgstab')
          and d['params'].get('rtol', 0) > 0)
    check('B: narrowed sweep + one result row with matvecs',
          d['sweep']['n'] == 1 and len(d['sweep']['results']) == 1
          and d['sweep']['results'][0].get('matvecs', 0) > 0
          and d['sweep']['results'][0].get('R', 0) > 0)
    check('B: overall 100%, matvec counter > 0, task stack empty',
          abs((d['overall']['pct'] or 0) - 100.0) < 1e-9
          and d['counters']['matvecs_total'] > 0
          and d['task']['stack'] == [])

    h = subprocess.run([PY, os.path.join(ROOT, 'src', 'sppeec_cli.py'),
                        '--help'], capture_output=True, text=True)
    check('--status/--status-file registered',
          '--status' in h.stdout and '--status-file' in h.stdout)


# ------------------------------------------- D. phase-2 setup tasks
def phase2():
    """Every phase-2 setup task must be observed on a real in-process
    equipotential solve (equibar: 384 cells, skin engine self-engages
    at its 1 GHz top frequency)."""
    import sppeec_status as st
    import sppeec_input
    st.disable()
    seen = set()
    st.enable(callback=lambda d: seen.update(d['task']['stack']),
              interval=0.0)
    pr = sppeec_input.load(os.path.join(ROOT, 'examples',
                                        'equibar.toml'))
    m = pr.model()
    M = pr.tree(m)
    sw = pr.sweeper(m, M)
    Z, info = sw.solve(1e9)
    st.finish('done')
    st.disable()
    check('D: tree/prepare/assemble tasks observed',
          {'build tree', 'prepare', 'assemble + preconditioner'}
          <= seen, repr(sorted(seen)))
    check('D: skin engine + mode-table ticker observed',
          any(s.startswith('skin engine k=') for s in seen)
          and 'mode tables' in seen, repr(sorted(seen)))
    check('D: freq task + krylov observed on the library path',
          any(s.startswith('solve f=') for s in seen)
          and 'krylov' in seen)
    check('D: solve still lands', info.get('matvecs', 0) > 0
          and float(abs(Z)) > 0)


def main():
    print('validate_status: A. unit')
    unit()
    print('validate_status: B/C. end-to-end + A/B')
    endtoend()
    print('validate_status: D. phase-2 setup tasks')
    phase2()
    if FAIL:
        print('FAIL: %d check(s): %s' % (len(FAIL), FAIL))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
