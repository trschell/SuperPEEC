# SPDX-License-Identifier: MIT
"""Time one thread configuration on a real model -- the A/B harness
for tuning SPPEEC's thread knobs on big nodes.

WHY. sppeec_threads pins OPENBLAS=1 / OMP=4 / FFTW_TOP=6 -- measured
optima for the SMALL models the studies use (10x at 2k cells) but
worth only 1.10x by 605k cells, and never measured beyond that. On a
multi-hundred-M-cell rung those pins cap the whole machine at a few
busy cores (4 of 32 vCPUs = the ~12% utilisation observed on the
first CPU-only R5 run). Every knob honours the caller's environment,
so tuning is an export away -- this harness measures one setting per
process (they are read when the libraries LOAD, so a sweep is several
processes, not a loop).

USAGE (run once per row on the target node; compare the rows):

    cd SuperPEEC
    for T in "1 4 6" "8 8 8" "16 16 16"; do set -- $T
      SPPEEC_BLAS_THREADS=$1 OPENBLAS_NUM_THREADS=$1 \
      OMP_NUM_THREADS=$2 FFTW_THREADS_TOP=$3 \
      PYTHONPATH=src python3 studies/threadtune.py \
          examples/dbc_halfbridge_r4.toml
    done

R4 (51M cells, ~29 GB) is the cheap proxy; graduate the winner to one
R5 run. The solve here uses a LOOSE tolerance -- this is a TIMING
harness, its impedances are not for use.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import sppeec_threads                                          # noqa: E402
import numpy as np                                             # noqa: E402
import sppeec_input                                            # noqa: E402

RTOL = 1e-2      # loose on purpose: enough Krylov cycles to time,
                 # done in minutes; NOT an extraction tolerance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('toml')
    ap.add_argument('--freq', type=float, default=None,
                    help='frequency to time (default: first declared)')
    args = ap.parse_args()

    knobs = dict(sppeec_threads.report(),
                 SPPEEC_BLAS_THREADS=os.environ.get(
                     'SPPEEC_BLAS_THREADS'))
    print('threads: %s' % json.dumps(knobs), flush=True)

    prob = sppeec_input.load(args.toml)
    prob.rtol = RTOL
    f = args.freq if args.freq is not None else prob.freqs[0]

    t0 = time.perf_counter()
    m = prob.model()
    M = prob.tree(m)
    t_tree = time.perf_counter() - t0
    t0 = time.perf_counter()
    sw = prob.sweeper(m, M)
    t_sweeper = time.perf_counter() - t0

    # two solves: the first carries one-time costs (wire-solver build,
    # FFT plans), the second is the repeatable per-point cost
    rows = []
    for tag in ('first', 'second'):
        t0 = time.perf_counter()
        Z, info = sw.solve(f)
        wall = time.perf_counter() - t0
        mv = int(info.get('matvecs', 0) or 0)
        rows.append((tag, wall, mv))
        print('  %-6s solve %8.1f s  %4d mv  %6.2f s/mv  '
              '(R=%.4g, rtol %g -- timing only)'
              % (tag, wall, mv, wall/mv if mv else float('nan'),
                 float(np.real(Z)) if np.ndim(Z) == 0 else
                 float('nan'), RTOL), flush=True)

    print('RESULT %s' % json.dumps(dict(
        toml=os.path.basename(args.toml), freq=f, rtol=RTOL,
        threads=knobs, tree_s=round(t_tree, 2),
        sweeper_s=round(t_sweeper, 2),
        first_solve_s=round(rows[0][1], 2), first_mv=rows[0][2],
        second_solve_s=round(rows[1][1], 2), second_mv=rows[1][2],
        s_per_mv=round(rows[1][1]/rows[1][2], 3)
        if rows[1][2] else None)), flush=True)


if __name__ == '__main__':
    main()
