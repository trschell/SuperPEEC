# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""RSFQ rung 1: the RSFQlib JTL against InductEx, as loop inductances.

Each configuration shorts one junction, opens the other and drives one
edge pin; the measured single-port loop inductance is an exact sum of
partial inductances from the InductEx back-annotated netlist
(THmitll_JTL_v3p0_extracted.cir):

    A  drive P1, J1 short, J2 open  ->  L1 + LP1
    B  drive P1, J2 short, J1 open  ->  L1 + L2 + L3 + LP2
    C  drive P2, J1 short, J2 open  ->  L4 + L3 + L2 + LP1
    D  drive P2, J2 short, J1 open  ->  L4 + LP2

Runs the doctrine entry point (sppeec_cli, defaults) per configuration
and pitch, and reports the ratio to InductEx. Each solve is a
subprocess so the memory of one rung never leaks into the next.

  python3 studies/rsfq_jtl.py --pitch 200e-9 100e-9 [--configs A B C D]
        [--basis overcomplete] [--out rsfq_jtl_results.json]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RSFQ = os.path.expanduser('~/Documents/RSFQlib/RSFQlib/mitll_JTL')
GDS = os.path.join(RSFQ, 'THmitll_JTL_v3p0.GDS')
CIR = os.path.join(RSFQ, 'THmitll_JTL_v3p0_extracted.cir')

CONFIGS = {
    'A': dict(drive='P1', short='J1', open='J2', terms=('L1', 'LP1')),
    'B': dict(drive='P1', short='J2', open='J1',
              terms=('L1', 'L2', 'L3', 'LP2')),
    'C': dict(drive='P2', short='J1', open='J2',
              terms=('L4', 'L3', 'L2', 'LP1')),
    'D': dict(drive='P2', short='J2', open='J1', terms=('L4', 'LP2')),
}


def inductex_partials(path=CIR):
    """{'L1': 2.07e-12, ...} from the back-annotated netlist."""
    out = {}
    for line in open(path):
        m = re.match(r'^(L\w+)\s+\S+\s+\S+\s+([0-9.]+E?[-+]?\d*)\s*$',
                     line.strip(), re.I)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def run(cfg, pitch, basis, workdir, python, freq):
    c = CONFIGS[cfg]
    toml = os.path.join(workdir, 'jtl_%s_%g.toml' % (cfg, pitch))
    cmd = [python, os.path.join(HERE, 'rsfq_gds2toml.py'), GDS,
           '--pitch', '%g' % pitch, '--out', toml, '--drive', c['drive'],
           '--short', c['short'], '--open', c['open'], '--freq', '%g' % freq]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    if basis:
        with open(toml, 'a') as f:
            f.write('basis = "%s"\n' % basis)
    log = toml[:-5] + '.log'
    t0 = time.time()
    with open(log, 'w') as f:
        subprocess.run([python, os.path.join(ROOT, 'src', 'sppeec_cli.py'),
                        toml, '-v'], stdout=f, stderr=subprocess.STDOUT,
                       check=True)
    txt = open(log).read()
    m = re.search(r'^\s*([0-9.e+]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s+\((\d+) mv',
                  txt, re.M)
    setup = re.search(r'setup ([0-9.]+) s', txt)
    cells = re.search(r'(\d+) occupied cells', open(toml).read())
    return dict(config=cfg, pitch=pitch, R=float(m.group(2)),
                L=float(m.group(3)), matvecs=int(m.group(4)),
                setup_s=float(setup.group(1)) if setup else None,
                wall_s=time.time() - t0, cells=int(cells.group(1)),
                mst_fallback='MST fundamental-cycle' in txt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pitch', type=float, nargs='+', default=[200e-9])
    ap.add_argument('--configs', nargs='+', default=list('ABCD'))
    ap.add_argument('--basis', default=None)
    ap.add_argument('--freq', type=float, default=1e10)
    ap.add_argument('--workdir', default=os.path.join(
        os.environ.get('TMPDIR', '/tmp'), 'rsfq_jtl'))
    ap.add_argument('--out', default=os.path.join(HERE, 'rsfq_jtl_results.json'))
    args = ap.parse_args(argv)
    os.makedirs(args.workdir, exist_ok=True)
    ref = inductex_partials()
    rows = []
    if os.path.exists(args.out):
        rows = json.load(open(args.out)).get('rows', [])
    print('InductEx partials (pH): %s' % ', '.join(
        '%s=%.3f' % (k, v*1e12) for k, v in sorted(ref.items())))
    print('%-3s %8s %9s %10s %10s %7s %6s %8s' % (
        'cfg', 'pitch', 'cells', 'L_pH', 'ref_pH', 'ratio', 'mv', 'setup_s'))
    for pitch in args.pitch:
        for cfg in args.configs:
            r = run(cfg, pitch, args.basis, args.workdir, sys.executable,
                    args.freq)
            r['ref'] = sum(ref[t] for t in CONFIGS[cfg]['terms'])
            r['ratio'] = r['L']/r['ref']
            r['basis'] = args.basis or 'auto'
            rows = [x for x in rows if not (x['config'] == cfg and
                                            x['pitch'] == pitch and
                                            x['basis'] == r['basis'])]
            rows.append(r)
            print('%-3s %8.0f %9d %10.4f %10.4f %7.3f %6d %8.0f%s' % (
                cfg, pitch*1e9, r['cells'], r['L']*1e12, r['ref']*1e12,
                r['ratio'], r['matvecs'], r['setup_s'] or 0,
                '  (MST fallback)' if r['mst_fallback'] else ''))
            json.dump(dict(reference=ref, rows=rows), open(args.out, 'w'),
                      indent=1)


if __name__ == '__main__':
    main()
