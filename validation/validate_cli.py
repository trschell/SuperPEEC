# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for sppeec_cli.py (2026-08-15): the doctrine-input runner.

Covers the CLI's contract, not the solver (the wire validators own
that): sanitizer failures exit 2 with a message and no traceback;
--freq NARROWS the declared sweep and refuses to extend it; a real
one-point run reproduces the module3wire anchor numbers and writes
the declared exports (.vti + quicklook + wires .vtp) with physically
consistent content (segment chain currents ~ shares of the drive).
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import os
import re
import subprocess
import sys
import tempfile

import numpy as np

FAIL = []
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


def run(*args):
    return subprocess.run([PY, os.path.join(HERE, '..', 'src', 'sppeec_cli.py')]
                          + list(args), capture_output=True, text=True,
                          cwd=os.path.join(HERE, '..'))


def main():
    r = run('does_not_exist.toml')
    check('missing input exits 2 with message, no traceback',
          r.returncode == 2 and 'not found' in r.stderr
          and 'Traceback' not in r.stderr)
    r = run('examples/module3wire.toml', '--freq', '7e7')
    check('--freq refuses to extend the declared sweep',
          r.returncode == 2 and 'never extends' in r.stderr)
    r = run('examples/module3wire.toml', '--quicklook', '-1')
    check('negative --quicklook rejected', r.returncode == 2)

    out = tempfile.mkdtemp(prefix='sppeec_cli_')
    r = run('examples/module3wire.toml', '--freq', '1e5',
            '--export-vti', '--export-wires', '--export-dir', out)
    check('one-point run exits 0', r.returncode == 0,
          (r.stderr.strip().splitlines() or [''])[-1][:70])
    row = [ln for ln in r.stdout.splitlines() if ln.strip()
           .startswith('100000')]
    check('result row printed', bool(row), r.stdout[-200:])
    if row:
        R = float(row[0].split()[1])
        # the module3wire anchor: R 0.000745967 at 1e5 (rtol 1e-4 band)
        check('R matches the module3wire anchor to 1%',
              abs(R - 0.000745967) < 0.01*0.000745967, '%g' % R)
    vti = os.path.join(out, 'module3wire_100000Hz.vti')
    ql = os.path.join(out, 'module3wire_100000Hz_quicklook.vti')
    vtp = os.path.join(out, 'module3wire_100000Hz_wires.vtp')
    check('.vti + quicklook + .vtp written',
          all(os.path.isfile(p) and os.path.getsize(p) > 1000
              for p in (vti, ql, vtp)))
    if os.path.isfile(vtp):
        txt = open(vtp).read()
        vals = [float(v) for v in re.search(
            r'Name="I_mag"[^>]*>\n([\d.eE+\- \n]+)</DataArray>',
            txt).group(1).split()]
        # 3 wires sharing a 1 A drive: every segment chain current in
        # a tight band around 1/3
        check('wire chain currents ~ shares of the 1 A drive',
              len(vals) > 0 and 0.2 < min(vals) and max(vals) < 0.5,
              '%d segs, %.3f..%.3f A' % (len(vals), min(vals),
                                         max(vals)))

    print('\n%d checks failed' % len(FAIL))
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
