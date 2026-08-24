# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Progress/status API: what is SuperPEEC doing right now?

A single process-global status object that the solver reports into and
that external tools read out of -- a Jupyter notebook polling a JSON
file, a future GUI, ``watch cat``, or the CLI's own ``--status`` line.

DESIGN RULES (phase 1, agreed 2026-08-23):

* **Instrument the library, not the driver.** Hooks live in the solver
  modules, so hand-built solvers in studies report too. The CLI only
  contributes run metadata it happens to know.
* **Off by default, free when off.** Every hook is guarded by a single
  ``_S is None`` test; disabled instrumentation makes no syscalls,
  takes no locks and allocates nothing. Enabled instrumentation must
  not change converged answers (validate_status gates an A/B run).
* **Report what was RESOLVED, not what was requested** -- the same
  doctrine as [analyze-defaults-first]: ``params`` carries the skin
  kwargs, method, rtol the solver actually uses.
* **A status failure never kills a solve.** Sink errors disable the
  sink with one warning and the run continues.

ACTIVATION
    environment   SPPEEC_STATUS=/path/status.json   (any entry point)
    CLI           --status (stderr line), --status-file PATH (JSON)
    programmatic  sppeec_status.enable(path=..., callback=..., tty=...)

TRANSPORT. The JSON file is written atomically (tmp + os.replace, a
reader can never see a torn document) and throttled to ``interval``
seconds (default 0.25), with task boundaries forcing a write. Liveness:
readers check ``pid`` + ``updated_at``. In-process consumers subscribe
a callback instead; ``tty=True`` adds a self-overwriting progress line
on stderr.

SCHEMA (``"schema": 1``; consumers should ignore unknown keys)::

    {
      "schema": 1, "pid": 41232, "seq": 412,
      "state": "running" | "done" | "failed" | "exited",
      "started_at": 1755950000.1, "updated_at": 1755950096.3,
      "error": "...",                     # state == "failed" only
      "model":  { "name": ..., "input": ..., "formulation": ...,
                  "cells": ..., "dims": [...], "fill_pct": ...,
                  "nports": ..., "tree_levels": ... },
      "params": { "method": ..., "rtol": ..., "basis": ...,
                  "skin": {...}, ... },   # whatever was resolved
      "sweep":  { "freqs": [...], "n": 5, "index": 2,
                  "current_freq": 1e10,
                  "results": [ { "f":..., "R":..., "L":...,
                                 "matvecs":..., "time_s":... } ] },
      "task":   { "stack": ["solve f=1e10", "krylov"],
                  "current": "krylov", "pct": 34.0,
                  "detail": { "matvecs": 102, "budget": 300,
                              "rtol": 1e-6, ... } },
      "overall": { "pct": 46.0, "elapsed_s": 96.2, "eta_s": 118.0 },
      "mem":     { "rss_mb": 1240, "hwm_mb": 2470 },
      "counters": { "matvecs_total": 412 }
    }

HONESTY OF THE PERCENTAGES. ``task.pct`` inside an lgmres/bicgstab
solve is matvecs/budget -- the budget is a hard cap (maxiter*inner_m),
so this never overshoots, but a converging solve finishes "early". The
LpPR fgmres path has true residual norms available per iteration and
reports log-residual progress instead. ``overall`` starts from static
weights (setup 20% when a setup task was seen, sweep split evenly per
frequency) and RE-WEIGHTS adaptively once both the measured setup time
and at least one per-frequency time exist; a monotonic ratchet keeps
the re-weighting from ever reading as regress. The raw numbers
(matvecs, elapsed, per-frequency times in ``sweep.results``) are
always published so a smarter client can do better.

EVENT LOG (phase 3). ``enable(events=path)`` -- or
``SPPEEC_STATUS_EVENTS=path``, or the CLI's ``--status-events`` --
appends one JSON line per state transition (``start``, ``sweep``,
``task_start``, ``task_end`` with duration, ``result``, ``finish``),
giving a GUI a timeline and a post-mortem of where the time went
without parsing prints. Unlike the status file it is append-only
history, not current state.

Reading:  ``sppeec_status.read(path)`` -> dict with ``_stale_s`` and
``_alive`` added.  ``python src/sppeec_status.py PATH`` is a minimal
terminal watcher.
"""
import atexit
import contextlib
import json
import os
import sys
import threading
import time

SCHEMA = 1
_LOCK = threading.RLock()
_S = None                      # the singleton _Status, or None = off

SETUP_WEIGHT = 0.2             # of overall, when a setup task was seen


# --------------------------------------------------------------- utils
def _jdefault(o):
    """JSON fallback: numpy scalars -> float, everything else -> str."""
    try:
        return float(o)
    except Exception:
        return str(o)


def _mem_mb():
    """(rss_mb, hwm_mb) from /proc, or (None, None) off-Linux."""
    try:
        rss = hwm = None
        with open('/proc/self/status') as fh:
            for ln in fh:
                if ln.startswith('VmRSS:'):
                    rss = int(ln.split()[1]) // 1024
                elif ln.startswith('VmHWM:'):
                    hwm = int(ln.split()[1]) // 1024
        return rss, hwm
    except Exception:
        return None, None


class _Task:
    """One entry of the task stack. Created via :func:`task`."""

    def __init__(self, name, ticks=None, detail=None, kind=''):
        self.name = name
        self.ticks = ticks            # total ticks, or None
        self.done = 0
        self.pct = None               # explicit percent, if set
        self.detail = dict(detail or {})
        self.kind = kind              # '' | 'setup' | 'freq'
        self.t0 = time.time()

    def frac(self):
        if self.pct is not None:
            return min(1.0, max(0.0, self.pct / 100.0))
        if self.ticks:
            return min(1.0, self.done / float(self.ticks))
        return None

    # -- caller-facing -------------------------------------------------
    def tick(self, n=1):
        with _LOCK:
            self.done += n
            if _S is not None:
                _S.publish()

    def set(self, pct=None, **detail):
        with _LOCK:
            if pct is not None:
                self.pct = float(pct)
            self.detail.update(detail)
            if _S is not None:
                _S.publish()


class _Status:
    """The state behind the module API. Use the module functions."""

    def __init__(self, path=None, callback=None, tty=False,
                 interval=0.25, events=None):
        self.path = path
        self.events_path = events
        self.callbacks = [callback] if callback else []
        self.tty = bool(tty)
        self.interval = float(interval)
        self.t0 = time.time()
        self.seq = 0
        self.state = 'running'
        self.error = None
        self.model = {}
        self.params = {}
        self.freqs = None            # the declared/narrowed sweep
        self.current_freq = None
        self.freqs_done = 0
        self.freq_times = []         # wall seconds per completed point
        self.results = []
        self.stack = []              # list of _Task
        self.saw_setup = False
        self.setup_done = False
        self.setup_s = 0.0           # measured setup wall (adaptive w)
        self.matvecs_total = 0
        self._hwm = 0.0              # overall-percent ratchet
        self._last_write = 0.0
        self._last_tty = 0.0
        self._tty_len = 0

    def event(self, ev, **fields):
        """Append one JSONL line to the event log (no-op without one).
        Failures disable the log, never the run."""
        if not self.events_path:
            return
        try:
            rec = dict(fields, t=time.time(), ev=ev)
            with open(self.events_path, 'a') as fh:
                fh.write(json.dumps(rec, default=_jdefault) + '\n')
        except Exception as exc:
            sys.stderr.write('sppeec_status: disabling event log '
                             '(%r)\n' % (exc,))
            self.events_path = None

    # -- progress model ----------------------------------------------
    def _leaf_frac(self):
        """Innermost task fraction that is known, else 0."""
        for t in reversed(self.stack):
            f = t.frac()
            if f is not None:
                return f
        return 0.0

    def _setup_frac(self):
        if self.setup_done:
            return 1.0
        if any(t.kind == 'setup' for t in self.stack):
            return self._leaf_frac() if len(self.stack) > 1 else \
                (self.stack[0].frac() or 0.0)
        return 0.0

    def _sweep_frac(self):
        n = len(self.freqs) if self.freqs else 0
        if not n:
            return None
        cur = self._leaf_frac() if any(
            t.kind == 'freq' for t in self.stack) else 0.0
        return min(1.0, (self.freqs_done + cur) / float(n))

    def overall(self):
        sw = self._sweep_frac()
        if sw is None:
            return None
        # ADAPTIVE RE-WEIGHTING: once both the measured setup wall and
        # at least one per-frequency time exist, the setup:sweep split
        # comes from measurement instead of the static 20/80 prior.
        # The ratchet below keeps the switch from ever reading as
        # regress (a smaller measured setup share would otherwise drop
        # the number at the moment the first point completes).
        n = len(self.freqs)
        if self.freq_times and self.setup_s > 0.0 and n:
            per = sum(self.freq_times) / len(self.freq_times)
            tot = self.setup_s + per * n
            w = self.setup_s / tot if tot > 0 else 0.0
        else:
            w = SETUP_WEIGHT if self.saw_setup else 0.0
        val = 100.0 * (w * self._setup_frac() + (1.0 - w) * sw)
        self._hwm = max(self._hwm, val)
        return self._hwm

    def eta_s(self):
        sw = self._sweep_frac()
        if sw is None or not self.freq_times:
            return None
        n = len(self.freqs)
        per = sum(self.freq_times) / len(self.freq_times)
        return max(0.0, per * n * (1.0 - sw))

    # -- document -----------------------------------------------------
    def doc(self):
        with _LOCK:
            self.seq += 1
            leaf = self.stack[-1] if self.stack else None
            frac = self._leaf_frac() if leaf is not None else None
            rss, hwm = _mem_mb()
            d = {
                'schema': SCHEMA, 'pid': os.getpid(), 'seq': self.seq,
                'state': self.state,
                'started_at': self.t0, 'updated_at': time.time(),
                'model': dict(self.model), 'params': dict(self.params),
                'sweep': {
                    'freqs': self.freqs,
                    'n': len(self.freqs) if self.freqs else None,
                    'index': self.freqs_done,
                    'current_freq': self.current_freq,
                    'results': list(self.results)},
                'task': {
                    'stack': [t.name for t in self.stack],
                    'current': leaf.name if leaf else None,
                    'pct': (100.0 * frac) if frac is not None else None,
                    'detail': dict(leaf.detail) if leaf else {}},
                'overall': {
                    'pct': self.overall(),
                    'elapsed_s': time.time() - self.t0,
                    'eta_s': self.eta_s()},
                'mem': {'rss_mb': rss, 'hwm_mb': hwm},
                'counters': {'matvecs_total': self.matvecs_total},
            }
            if self.error is not None:
                d['error'] = self.error
            return d

    # -- sinks --------------------------------------------------------
    def publish(self, force=False):
        now = time.time()
        if not force and now - self._last_write < self.interval:
            return
        with _LOCK:
            if not force and now - self._last_write < self.interval:
                return
            self._last_write = now
            d = self.doc()
        if self.path:
            try:
                tmp = self.path + '.tmp'
                with open(tmp, 'w') as fh:
                    json.dump(d, fh, default=_jdefault)
                os.replace(tmp, self.path)
            except Exception as exc:
                sys.stderr.write('sppeec_status: disabling file sink '
                                 '(%r)\n' % (exc,))
                self.path = None
        for cb in self.callbacks:
            try:
                cb(d)
            except Exception as exc:
                sys.stderr.write('sppeec_status: dropping callback '
                                 '(%r)\n' % (exc,))
                self.callbacks.remove(cb)
        if self.tty and (force or now - self._last_tty >= 0.5):
            self._last_tty = now
            self._tty_line(d)

    def _tty_line(self, d):
        try:
            line = format_line(d)
            if sys.stderr.isatty():
                pad = max(0, self._tty_len - len(line))
                sys.stderr.write('\r' + line + ' ' * pad)
                self._tty_len = len(line)
                if d['state'] != 'running':
                    sys.stderr.write('\n')
                    self._tty_len = 0
            else:
                sys.stderr.write(line + '\n')
            sys.stderr.flush()
        except Exception:
            self.tty = False


# ------------------------------------------------------------- module API
def enabled():
    return _S is not None


def enable(path=None, callback=None, tty=False, interval=0.25,
           events=None):
    """Turn status reporting on (idempotent; sinks are merged)."""
    global _S
    with _LOCK:
        if _S is None:
            _S = _Status(path=path, callback=callback, tty=tty,
                         interval=interval, events=events)
            _S.event('start')
        else:
            if path:
                _S.path = path
            if callback:
                _S.callbacks.append(callback)
            if events:
                _S.events_path = events
            _S.tty = _S.tty or bool(tty)
        _S.publish(force=True)
        return _S


def disable():
    global _S
    with _LOCK:
        _S = None


def run_meta(model=None, params=None):
    """Merge run metadata. ``model`` is identity, ``params`` are the
    RESOLVED solver settings (post-default, post-auto)."""
    if _S is None:
        return
    with _LOCK:
        if model:
            _S.model.update(model)
        if params:
            _S.params.update(params)
        _S.publish(force=True)


def sweep_meta(freqs):
    """Declare the frequency sweep about to run (resets progress --
    the CLI calls this again after ``--freq`` narrowing)."""
    if _S is None:
        return
    with _LOCK:
        _S.freqs = [float(f) for f in freqs]
        _S.freqs_done = 0
        _S.event('sweep', n=len(_S.freqs))
        _S.publish(force=True)


@contextlib.contextmanager
def task(name, ticks=None, kind='', **detail):
    """Push a task for the duration of a ``with`` block. Yields the
    task; use ``.tick()`` / ``.set(pct=..., key=val)`` on it. Safe to
    call when disabled (yields an inert task)."""
    t = _Task(name, ticks=ticks, detail=detail, kind=kind)
    if _S is None:
        yield t
        return
    with _LOCK:
        if kind == 'setup':
            _S.saw_setup = True
        _S.stack.append(t)
        _S.event('task_start', task=name)
    _S.publish(force=True)
    try:
        yield t
    finally:
        with _LOCK:
            if t in _S.stack:
                _S.stack.remove(t)
            if kind == 'setup':
                _S.setup_done = True
                _S.setup_s += time.time() - t.t0
            _S.event('task_end', task=name,
                     dur_s=time.time() - t.t0)
        _S.publish(force=True)


@contextlib.contextmanager
def freq_task(freq):
    """The per-frequency task. Nesting-safe: if a freq task is already
    open (the CLI wraps the sweeper, which also wraps itself), the
    inner call is a no-op so the point is not double-counted."""
    if _S is None or any(t.kind == 'freq' for t in _S.stack):
        yield None
        return
    with _LOCK:
        _S.current_freq = float(freq)
        _S.setup_done = True         # first point => setup is over
    t0 = time.time()
    with task('solve f=%g' % freq, kind='freq') as t:
        yield t
    with _LOCK:
        _S.freqs_done += 1
        _S.freq_times.append(time.time() - t0)
        _S.current_freq = None
    _S.publish(force=True)


def record_result(freq, **fields):
    """Append one completed frequency point (partial results feed a
    live Z(f) plot). Pass whatever is known: R, L, imZ, matvecs,
    residual, time_s."""
    if _S is None:
        return
    row = {'f': float(freq)}
    for k, v in fields.items():
        if v is not None:
            try:
                row[k] = float(v)
            except Exception:
                row[k] = str(v)
    with _LOCK:
        _S.results.append(row)
        _S.event('result', **row)
    _S.publish(force=True)


def krylov_task(**detail):
    """Context manager for one outer Krylov solve; nullcontext-cheap
    when disabled. ``tick_matvec`` drives its percent off
    ``detail['budget']`` (the hard matvec cap)."""
    if _S is None:
        return contextlib.nullcontext(None)
    return task('krylov', **dict(detail, matvecs=0))


def tick_matvec(n=1):
    """One operator application. Cheap: an int, and a throttled
    publish."""
    if _S is None:
        return
    with _LOCK:
        _S.matvecs_total += n
        for t in reversed(_S.stack):
            if 'budget' in t.detail:
                t.detail['matvecs'] = t.detail.get('matvecs', 0) + n
                b = t.detail['budget']
                if b:
                    t.pct = min(99.0, 100.0 * t.detail['matvecs'] / b)
                break
    _S.publish()


def krylov_residual(rrel):
    """Attach a residual reading to the innermost Krylov task (the
    one carrying a matvec budget). The lgmres hook computes it FREE --
    by recognising the outer-cycle-opening matvec -- so this is just
    the delivery."""
    if _S is None:
        return
    with _LOCK:
        for t in reversed(_S.stack):
            if 'budget' in t.detail:
                t.detail['residual'] = float(rrel)
                break
    _S.publish()


def finish(state='done', error=None):
    """Terminal state; forces a final write."""
    if _S is None:
        return
    with _LOCK:
        _S.state = state
        _S.error = error
        _S.stack = []
        _S.event('finish', state=state)
    _S.publish(force=True)


@atexit.register
def _atexit():
    # A crash that bypasses finish() must not leave state 'running'
    # forever -- readers also check pid, but be honest on the way out.
    if _S is not None and _S.state == 'running':
        _S.state = 'exited'
        try:
            _S.publish(force=True)
        except Exception:
            pass


# ------------------------------------------------------------- consumers
def read(path):
    """Read a status file; adds ``_stale_s`` (age of the last write)
    and ``_alive`` (whether the writing pid still exists)."""
    with open(path) as fh:
        d = json.load(fh)
    d['_stale_s'] = time.time() - d.get('updated_at', 0)
    try:
        os.kill(int(d['pid']), 0)
        d['_alive'] = True
    except Exception:
        d['_alive'] = False
    return d


def format_line(d):
    """One-line human rendering of a status document."""
    parts = []
    ov = d.get('overall', {}).get('pct')
    parts.append('[%3.0f%%]' % ov if ov is not None else '[ -- ]')
    sw = d.get('sweep', {})
    if sw.get('current_freq') is not None:
        parts.append('f=%g (%d/%d)' % (sw['current_freq'],
                                       sw.get('index', 0) + 1,
                                       sw.get('n') or 0))
    t = d.get('task', {})
    if t.get('current'):
        s = t['current']
        if t.get('pct') is not None:
            s += ' %.0f%%' % t['pct']
        det = t.get('detail', {})
        if 'matvecs' in det and 'budget' in det:
            s += ' mv %d/%d' % (det['matvecs'], det['budget'])
        if det.get('residual') is not None:
            s += ' res %.1e' % det['residual']
        parts.append(s)
    mem = d.get('mem', {})
    if mem.get('rss_mb'):
        parts.append('rss %.1fG' % (mem['rss_mb'] / 1024.0))
    eta = d.get('overall', {}).get('eta_s')
    if eta is not None:
        parts.append('eta %ds' % int(eta))
    if d.get('state') != 'running':
        parts.append(d.get('state', ''))
    return ' '.join(parts)


def _watch(path, interval=1.0):
    while True:
        try:
            d = read(path)
        except FileNotFoundError:
            print('waiting for %s ...' % path)
            time.sleep(interval)
            continue
        extra = '' if d['_alive'] else '  [pid gone]'
        print(format_line(d) + extra, flush=True)
        if d.get('state') != 'running' or not d['_alive']:
            return 0 if d.get('state') == 'done' else 1
        time.sleep(interval)


# env activation: any entry point that imports the solver gets the
# file sink with no code change. SPPEEC_STATUS=0 / empty stays off.
_env = os.environ.get('SPPEEC_STATUS', '')
_env_ev = os.environ.get('SPPEEC_STATUS_EVENTS', '')
if (_env and _env != '0') or (_env_ev and _env_ev != '0'):
    enable(path=_env or None, events=_env_ev or None)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: sppeec_status.py STATUS.json '
                         '[interval_s]')
    raise SystemExit(_watch(sys.argv[1], float(sys.argv[2])
                            if len(sys.argv) > 2 else 1.0))
