# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""RSFQ rung 2: the RSFQlib XNOR against InductEx, as loop inductances.

The XNOR is the most intricate cell in the library (19 junctions, 58
inductors, 2146 polygons in 4213 um^2 -- the maximum on every count).
Its four signal ports each reach ground through ONE junction, so each
gives a single-port loop inductance that is an exact two-term sum of
InductEx partials from THmitll_XNOR_v3p0_extracted.cir:

    A  drive P1 (a),   J1  short  ->  L1  + LP1   = 1.9326 pH
    B  drive P2 (b),   J4  short  ->  L4  + LP4   = 1.8642 pH
    C  drive P3 (clk), J9  short  ->  L8  + LP9   = 1.9634 pH
    D  drive P4 (q),   J19 short  ->  L20 + LP19  = 1.9436 pH

Every other junction is OPEN, so no other path to ground exists and
the measurement is the named loop alone.

THE PORT MAP IS DERIVED, NOT ASSUMED. The layout labels ports P1-P4
and the netlist calls them a/b/clk/q, and no file in the library
states the correspondence. It is fixed here by two independent facts
that agree: each netlist port reaches ground through exactly one
junction (a->B1, b->B4, clk->B9, q->B19), and each layout port's
NEAREST junction carries that same index -- P1(0.0,35.0)/J1(5.1,35.0),
P2(15.0,70.0)/J4(15.6,64.8), P3(15.0,0.0)/J9(14.3,6.0),
P4(60.0,35.0)/J19(54.4,35.0). A wrong map would compare against the
wrong reference and still look plausible, so it is stated here rather
than left implicit.

Runs the doctrine entry point (sppeec_cli, defaults) per configuration
and pitch, and reports the ratio to InductEx. Each solve is a
subprocess so the memory of one rung never leaks into the next, and
each writes a --status-file so a long rung can be watched rather than
guessed at.

  python3 studies/rsfq_xnor.py --pitch 200e-9 [--configs A B C D]
        [--pz 67.5e-9] [--out rsfq_xnor_results.json]
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
RSFQ = os.path.expanduser('~/Documents/RSFQlib/RSFQlib/mitll_XNOR')
GDS = os.path.join(RSFQ, 'THmitll_XNOR_v3p0.GDS')
CIR = os.path.join(RSFQ, 'THmitll_XNOR_v3p0_extracted.cir')

# Every junction not named `short` is OPEN (--junctions open), so the
# only conducting path to ground is the one named.
CONFIGS = {
    'A': dict(drive='P1', short='J1',  net='a',   terms=('L1', 'LP1')),
    'B': dict(drive='P2', short='J4',  net='b',   terms=('L4', 'LP4')),
    'C': dict(drive='P3', short='J9',  net='clk', terms=('L8', 'LP9')),
    'D': dict(drive='P4', short='J19', net='q',   terms=('L20', 'LP19')),
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


def run(cfg, pitch, basis, workdir, python, freq, pz=None,
        glb=False):
    c = CONFIGS[cfg]
    toml = os.path.join(workdir, 'xnor_%s_%g.toml' % (cfg, pitch))
    cmd = [python, os.path.join(HERE, 'rsfq_gds2toml.py'), GDS,
           '--pitch', '%g' % pitch, '--out', toml, '--drive', c['drive'],
           '--short', c['short'], '--junctions', 'open', '--freq', '%g' % freq]
    if pz:
        cmd += ['--pz', '%g' % pz]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    if basis:
        with open(toml, 'a') as f:
            f.write('basis = "%s"\n' % basis)
    log = toml[:-5] + '.log'
    status = toml[:-5] + '.status.json'
    t0 = time.time()
    solve = [python, os.path.join(ROOT, 'src', 'sppeec_cli.py'), toml, '-v',
             '--status-file', status]
    if glb:
        solve += ['--export-glb', '--export-dir', workdir]
    with open(log, 'w') as f:
        subprocess.run(solve, stdout=f, stderr=subprocess.STDOUT, check=True)
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
    ap.add_argument('--pz', type=float, default=67.5e-9,
                    help='z pitch; 67.5 nm makes M5 (135 nm) exact')
    ap.add_argument('--export-glb', action='store_true',
                    help='also write a Blender .glb per rung')
    ap.add_argument('--configs', nargs='+', default=list('ABCD'))
    ap.add_argument('--basis', default=None)
    ap.add_argument('--freq', type=float, default=1e10)
    ap.add_argument('--workdir', default=os.path.join(
        os.environ.get('TMPDIR', '/tmp'), 'rsfq_xnor'))
    ap.add_argument('--out', default=os.path.join(
        HERE, 'rsfq_xnor_results.json'))
    args = ap.parse_args(argv)
    os.makedirs(args.workdir, exist_ok=True)
    ref = inductex_partials()
    rows = []
    if os.path.exists(args.out):
        rows = json.load(open(args.out)).get('rows', [])
    print('InductEx partials (pH): %s' % ', '.join(
        '%s=%.3f' % (k, v*1e12) for k, v in sorted(ref.items())))
    print('%-3s %-4s %8s %9s %10s %10s %7s %6s %8s' % (
        'cfg', 'net', 'pitch', 'cells', 'L_pH', 'ref_pH', 'ratio', 'mv',
        'setup_s'))
    for pitch in args.pitch:
        for cfg in args.configs:
            r = run(cfg, pitch, args.basis, args.workdir, sys.executable,
                    args.freq, pz=args.pz, glb=args.export_glb)
            r['ref'] = sum(ref[t] for t in CONFIGS[cfg]['terms'])
            r['ratio'] = r['L']/r['ref']
            r['basis'] = args.basis or 'auto'
            rows = [x for x in rows if not (x['config'] == cfg and
                                            x['pitch'] == pitch and
                                            x['basis'] == r['basis'])]
            rows.append(r)
            print('%-3s %-4s %8.0f %9d %10.4f %10.4f %7.3f %6d %8.0f%s' % (
                cfg, CONFIGS[cfg]['net'], pitch*1e9, r['cells'],
                r['L']*1e12, r['ref']*1e12, r['ratio'], r['matvecs'],
                r['setup_s'] or 0,
                '  (MST fallback)' if r['mst_fallback'] else ''))
            json.dump(dict(reference=ref, rows=rows), open(args.out, 'w'),
                      indent=1)


if __name__ == '__main__':
    main()
