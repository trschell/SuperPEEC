# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Command-line runner for doctrine TOML inputs.

    python sppeec_cli.py input.toml [options]

DIVISION OF LABOUR (the design rule every future option must pass):
the TOML file is the complete, reproducible statement of the PROBLEM
-- geometry, materials, ports, frequencies, formulation. Anything
that changes converged numbers lives THERE. The command line holds
RUN POLICY only: what to export and where, which subset of the
declared sweep to run now, how chatty to be. If a proposed flag
would change the answer, that is the signal it is a missing TOML
key, not a CLI feature. (Narrowing is allowed -- ``--freq`` picks
points OF the declared sweep -- extending is not: a frequency the
file does not declare is an error, so a result never exists that
the file cannot reproduce.)

Exports pair in ParaView: the ``.vti`` volume current density under
the ``.vtp`` wire polylines (Tube filter on the wires, radius from
the ``radius`` cell array).
"""
import argparse
import os
import sys

# Thread defaults before numpy loads its BLAS -- see sppeec_threads.py.
# The CLI is a user's first contact with the solver, so it must not be
# the one entry point that silently runs 6x slow.
try:
    import sppeec_threads  # noqa: F401
except Exception:
    pass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MU0 = 4e-7*np.pi


class CliOptions:
    """Registry, parser and sanitizer for the command line.

    One class owns the whole surface: :meth:`_register` declares the
    options (grouped so future ones slot in), :meth:`parse` runs
    argparse and then the ``_sanitize_*`` methods -- every sanitizer
    failure goes through ``parser.error`` so the user gets usage +
    message + exit 2, never a traceback. Parsed, sanitized values
    land as attributes on the instance.

    Frequency narrowing (`--freq`) needs the file's declared sweep,
    which is not known at parse time -- so it is a two-stage
    sanitizer: syntax at parse(), matching at :meth:`select_freqs`.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='sppeec_cli.py',
            description='Solve a doctrine TOML input (Z(f), per-wire '
                        'sharing) and optionally export fields.',
            epilog='The TOML file defines the problem; these options '
                   'only control the run. See docs/input_doctrine.md.')
        self._register(self.parser)

    def _register(self, p):
        p.add_argument('input', help='doctrine TOML input file')
        run = p.add_argument_group('run scope')
        run.add_argument(
            '--freq', type=float, nargs='+', metavar='HZ', default=None,
            help='run only these points of the declared sweep '
                 '(each must match a [solve].freq entry -- the CLI '
                 'narrows the file, it never extends it)')
        run.add_argument('-v', '--verbose', action='store_true',
                         help='solver diagnostics (setup timings, '
                              'preconditioner engagement, residuals)')
        exp = p.add_argument_group(
            'export (run policy -- never changes the answer)')
        exp.add_argument(
            '--export-vti', action='store_true',
            help='write volume current density per frequency '
                 '(streaming .vti + binned quick-look companion)')
        exp.add_argument(
            '--export-wires', action='store_true',
            help='write bond wires per frequency (.vtp polylines '
                 'with solved chain currents)')
        exp.add_argument(
            '--export-dir', metavar='DIR', default='results',
            help='directory for exported files (default: results/)')
        exp.add_argument(
            '--quicklook', type=int, metavar='N', default=4,
            help='bin factor for the .vti quick-look companion '
                 '(default 4; 0 disables the companion)')
        mon = p.add_argument_group(
            'monitoring (run policy -- never changes the answer)')
        mon.add_argument(
            '--status', action='store_true',
            help='live progress line on stderr while solving')
        mon.add_argument(
            '--status-file', metavar='PATH', default=None,
            help='write machine-readable JSON progress to PATH '
                 '(atomic, throttled -- poll it from a notebook, GUI '
                 'or `python src/sppeec_status.py PATH`; '
                 'SPPEEC_STATUS=PATH is the env-var equivalent)')
        mon.add_argument(
            '--status-events', metavar='PATH', default=None,
            help='append one JSON line per state transition (task '
                 'start/end with durations, per-point results) to '
                 'PATH -- a timeline for GUIs and post-mortems; '
                 'SPPEEC_STATUS_EVENTS=PATH is the env-var '
                 'equivalent)')

    def parse(self, argv=None):
        ns = self.parser.parse_args(argv)
        for name in sorted(dir(self)):
            if name.startswith('_sanitize_'):
                getattr(self, name)(ns)
        vars(self).update(vars(ns))
        return self

    # -- sanitizers (alphabetical execution; keep them independent) --

    def _sanitize_export(self, ns):
        ns.exporting = ns.export_vti or ns.export_wires
        if ns.quicklook < 0:
            self.parser.error('--quicklook must be >= 0')
        if not ns.exporting:
            return
        try:
            os.makedirs(ns.export_dir, exist_ok=True)
        except OSError as exc:
            self.parser.error('cannot create --export-dir %r (%s)'
                              % (ns.export_dir, exc))

    def _sanitize_freq(self, ns):
        if ns.freq is not None and any(f <= 0 for f in ns.freq):
            self.parser.error('--freq values must be positive')

    def _sanitize_input(self, ns):
        if not os.path.isfile(ns.input):
            self.parser.error('input file %r not found' % ns.input)

    # -- post-load helpers -------------------------------------------

    def select_freqs(self, declared):
        """Match --freq against the file's sweep; error on strays."""
        if self.freq is None:
            return list(declared)
        picked = []
        for want in self.freq:
            hits = [f for f in declared
                    if abs(f - want) <= 1e-9*max(f, want)]
            if not hits:
                self.parser.error(
                    '--freq %g is not in the declared sweep %s -- the '
                    'CLI narrows the file, it never extends it (add '
                    'the point to [solve].freq instead)'
                    % (want, ['%g' % f for f in declared]))
            picked.extend(hits)
        return picked

    def export_paths(self, freq):
        """(vti_path, vtp_path) for one frequency, or Nones."""
        tag = ('%g' % freq).replace('+', '')
        stem = os.path.join(
            self.export_dir,
            os.path.splitext(os.path.basename(self.input))[0])
        return (('%s_%sHz.vti' % (stem, tag)) if self.export_vti
                else None,
                ('%s_%sHz_wires.vtp' % (stem, tag)) if self.export_wires
                else None)


def main(argv=None):
    opts = CliOptions().parse(argv)
    import sppeec_status as status
    if opts.status or opts.status_file or opts.status_events:
        status.enable(path=opts.status_file, tty=opts.status,
                      events=opts.status_events)
    try:
        rc = _run(opts, status)
    except SystemExit:
        raise                     # parser.error: message already out
    except BaseException as exc:
        status.finish('failed', error=repr(exc))
        raise
    status.finish('done')
    return rc


def _run(opts, status):
    import sppeec_input
    import vtkout

    prob = sppeec_input.load(opts.input)
    if opts.export_wires and not prob.wires(prob.freqs[0]
                                            if prob.freqs else 1.0):
        opts.parser.error('--export-wires: the input declares no '
                          '[[wire]] tables')
    freqs = opts.select_freqs(prob.freqs)
    if not freqs:
        opts.parser.error('nothing to solve: [solve].freq is empty')

    with status.task('setup', kind='setup'):
        m = prob.model()
        M = prob.tree(m)
        if opts.verbose:
            print('%s: %s, dims %s, %.1f%% fill'
                  % (os.path.basename(opts.input), prob.formulation,
                     tuple(m.dims), m.fill()), flush=True)
        sw = prob.sweeper(m, M, verbose=opts.verbose)
    # the sweeper published the file's declared sweep; narrow it to
    # what this run will actually solve (--freq)
    status.sweep_meta(freqs)

    written = []
    lppr = prob.formulation == 'LpPR'
    if getattr(sw, 'nports', 1) > 1:
        print('%12s  open-circuit Z matrix, ports %s'
              % ('freq [Hz]', [p.name for p in m.ports]))
    else:
        print('%12s %12s %12s  %s'
              % ('freq [Hz]', 'R [Ohm]',
                 'Im Z [Ohm]' if lppr else 'L [H]',
                 'C_eff/L_eff' if lppr else 'shares'))
    for f in freqs:
        with status.freq_task(f):
            Z, info = sw.solve(f)
        w = 2*np.pi*f
        if lppr and np.ndim(Z) == 2:
            names = [p.name for p in m.ports]
            print('%12g  Z matrix [Ohm] (%d mv, true resid %.1e):'
                  % (f, info['matvecs'], info['true_residual']),
                  flush=True)
            for i in range(Z.shape[0]):
                print('    %8s  %s' % (names[i], '  '.join(
                    '%12.6g%+12.6gj' % (Z[i, j].real, Z[i, j].imag)
                    for j in range(Z.shape[1]))), flush=True)
        elif lppr:
            # capacitive port below series resonance, inductive above
            # -- report the branch the sign picks (pdn_sweep's rule)
            eff = ('C %.4g F' % (np.imag(1.0/Z)/w)) if Z.imag < 0 \
                else ('L %.4g H' % (Z.imag/w))
            print('%12g %12.6g %12.6g  %s   (%d mv, true resid %.1e)'
                  % (f, Z.real, Z.imag, eff, info['matvecs'],
                     info['true_residual']), flush=True)
        else:
            # per-wire shares exist on the wire path only; the
            # equipotential path solves the split but has no wires
            sh = (np.round(np.real(info['share']), 4)
                  if 'share' in info else '')
            print('%12g %12.6g %12.6g  %s   (%d mv, resid %.1e)'
                  % (f, Z.real, abs(Z.imag)/w, sh,
                     info['matvecs'], info['residual']), flush=True)
        vti, vtp = opts.export_paths(f)
        if vti:
            written += vtkout.export_currents_streaming(
                m, M, info['i_f'], vti, quicklook=opts.quicklook)
        if vtp:
            vtkout.export_wires(sw.sol.wires, info['i_w'],
                                sw.sol.wc.seg0, sw.sol.wire_of_seg,
                                vtp)
            written.append(vtp)
    for path in written:
        print('wrote %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
