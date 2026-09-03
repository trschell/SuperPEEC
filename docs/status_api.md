# The status API: watching a solve from outside

SuperPEEC can report what it is doing — the model, the parameters it
actually resolved, the current frequency, the current task, and two
percentages — to anything that wants to watch: a Jupyter notebook, a
GUI, `watch cat`, or the CLI's own progress line. This document
freezes **schema 1**. Consumers must ignore unknown keys; keys listed
here will not change meaning within schema 1.

A worked notebook lives at `examples/status_monitor.ipynb`.

## Turning it on

Status is **off by default and free when off**, and by design it can
never change a converged answer (`validation/validate_status.py` gates
a byte-identical A/B run every suite pass).

| Route | What you get |
|---|---|
| `SPPEEC_STATUS=/path/status.json` (env) | the JSON status file, from **any** entry point — the CLI, a study, your own script |
| `SPPEEC_STATUS_EVENTS=/path/events.jsonl` (env) | the append-only event log |
| `--status` (CLI) | a live self-overwriting progress line on stderr |
| `--status-file PATH` (CLI) | the JSON status file |
| `--status-events PATH` (CLI) | the event log |
| `sppeec_status.enable(path=…, events=…, callback=…, tty=…)` | programmatic; `callback` receives each status document as a dict, in process |

Sinks merge: enabling twice adds sinks rather than replacing them. A
sink failure disables that sink with one stderr warning and never the
solve.

## The status file

Written **atomically** (temp file + rename — a reader can never see a
torn document) and throttled to ~4 Hz, with task boundaries forcing a
write. Poll it; don't tail it.

```json
{ "schema": 1, "pid": 41232, "seq": 412,
  "state": "running",
  "started_at": 1755950000.1, "updated_at": 1755950096.3,
  "model":  { "name": "module3wire", "input": "examples/module3wire.toml",
              "formulation": "LpR", "dims": [65, 21, 3],
              "cells_occupied": 3780, "cells_lattice": 4095,
              "cells": 3780, "fill_pct": 92.3,
              "nports": 0, "tree_levels": 2 },
  "params": { "method": "lgmres", "rtol": 1e-4, "current": 1.0,
              "basis": "auto",
              "skin": { "subdivide": 7, "reach": 0,
                        "rc_uu": 12, "rc_cross": 16 } },
  "sweep":  { "freqs": [1e5, 1e6, 3e6, 1e7, 3e7], "n": 5, "index": 1,
              "current_freq": 1e6,
              "results": [ { "f": 1e5, "R": 7.46e-4, "imZ": 1.99e-3,
                             "L": 3.16e-9, "matvecs": 57,
                             "residual": 2.6e-5, "time_s": 1.19 } ] },
  "task":   { "stack": ["solve f=1e+06", "krylov"], "current": "krylov",
              "pct": 34.0,
              "detail": { "matvecs": 102, "budget": 300,
                          "rtol": 1e-4, "method": "lgmres" } },
  "overall": { "pct": 46.0, "elapsed_s": 96.2, "eta_s": 118.0 },
  "mem":     { "rss_mb": 1240, "hwm_mb": 2470 },
  "counters": { "matvecs_total": 412 } }
```

Field notes, in reading order:

* **`state`** — `running`, then exactly one of `done`, `failed`
  (with an `error` string), or `exited` (the process ended without
  reaching a terminal state; written from an atexit hook).
* **Liveness** — check `pid` still exists *and* `updated_at` is
  recent. `sppeec_status.read(path)` does both, returning the dict
  with `_alive` and `_stale_s` added.
* **`model`** — identity of the run. `name` is the input file's stem.
  Two cell counts, both true: **`cells_occupied`** is the conductor
  cells — what the solve's unknowns scale with (filaments and loops
  live only on metal) — and **`cells_lattice`** is the bounding grid
  (`dims` product), the count if the whole envelope were metal, which
  sizes the FFT/Toeplitz machinery and is what memory laws and the
  refinement-rung names refer to.
  `fill_pct = 100 · occupied/lattice`. `cells` is a legacy alias for
  `cells_occupied`. `nports` is 0 on the wire path (no port objects
  there).
* **`params`** — the **resolved** settings, post-default and
  post-auto: what the solver is actually using, not what was asked
  for. `skin` appears on the equipotential path with the engaged
  engine's kwargs.
* **`sweep`** — `freqs` is the run's actual plan (the CLI re-declares
  it after `--freq` narrowing). `index` counts completed points.
  `results` grows one row per completed frequency — poll it for a
  live Z(f) plot. `R`/`imZ`/`L` appear for scalar-impedance solves;
  multi-port matrix solves record only counters.
* **`task`** — the live task stack, innermost last. `pct` is the
  innermost known percent.
* **`overall.pct`** — an *estimate*: setup and sweep are weighted
  20/80 until the run has measured both a setup wall time and at
  least one per-frequency time, after which the weights come from
  measurement. A monotonic ratchet guarantees the number never
  decreases within a run. The raw materials (`elapsed_s`, per-point
  `time_s`, `matvecs`) are always present for a smarter client.
* **`mem`** — RSS and high-water mark from `/proc` (null off-Linux).

## Percent semantics inside a solve

Two kinds of task percent exist, and they are honest in different
ways:

* **`krylov`** (lgmres/bicgstab — every LpR-family outer solve):
  `pct = matvecs / budget` where the budget is the hard iteration cap
  (`maxiter × inner_m`). It never overshoots; a converging solve
  simply finishes early. On **lgmres** the detail also carries a true
  relative `residual`, refreshed once per outer cycle at zero extra
  matvecs: lgmres opens each cycle by applying A to the iterate it
  just reported, and the counting wrapper recognises that call and
  reads `‖rhs − A·x‖` off work already being paid for. bicgstab's
  matvecs never touch the iterate, so it reports budget percent only.
* **`fgmres`** (the LpPR path): true residual norms are available per
  iteration for free, so `pct` is **log-residual progress** — orders
  of magnitude travelled from the initial residual toward the
  stopping test `‖r‖ < tol·‖b‖` — with the current relative
  `residual` in the detail.

Setup tasks (`build tree`, `prepare`, `terminal coupler`,
`skin engine k=N`, `mode tables`, `assemble + preconditioner`,
`build wire solver`, `export fields`, `drive port j/n`) either tick a
known chunk count (`mode tables`, `export fields`) or are unmeasured
brackets that exist so the stack always says *what* is running.

## The event log

One JSON object per line, append-only — history, where the status
file is current state:

    {"t": 1755950000.2, "ev": "start"}
    {"t": 1755950003.1, "ev": "task_start", "task": "build tree"}
    {"t": 1755950004.9, "ev": "task_end", "task": "build tree", "dur_s": 1.8}
    {"t": 1755950031.0, "ev": "result", "f": 1e5, "R": 7.46e-4, ...}
    {"t": 1755950096.3, "ev": "finish", "state": "done"}

Event classes: `start`, `sweep` (`n` points declared), `task_start`,
`task_end` (with `dur_s`), `result` (the same row appended to
`sweep.results`), `finish`. Summing `task_end.dur_s` by task name is
a free post-mortem of where the time went.

## Reading patterns

Terminal:

    python src/sppeec_status.py /path/status.json      # minimal watcher

Python / Jupyter (any process, no imports from the solver needed
beyond this one module):

```python
import sppeec_status
d = sppeec_status.read('/path/status.json')
print(sppeec_status.format_line(d))
if not d['_alive'] and d['state'] == 'running':
    print('writer died without finishing')
```

Same process (a script driving the sweeper directly):

```python
import sppeec_status
sppeec_status.enable(callback=lambda d: my_widget.update(d))
```

## Guarantees, restated

1. Off by default; a single pointer test when off.
2. Enabled instrumentation does not change any converged number —
   gated by an A/B byte-identical run plus the suite's byte anchors.
3. The status file is always a complete, parseable JSON document.
4. `seq` strictly increases; `overall.pct` never decreases within a
   run.
5. A status/sink failure never kills or alters the solve.
