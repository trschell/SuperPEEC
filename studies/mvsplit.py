# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Where does one Krylov iteration go? Build + operator + preconditioner
split for ANY doctrine TOML (wire-bond path or equipotential path).

Prints: build phases (model / tree+prepare / solver setup), the solve
prologue (equipotential: the two lsqr projections), the operator apply
and preconditioner apply timed twice each (first = warm-up), and a
cProfile of two operator+preconditioner rounds bucketed by subsystem:
FMM (levels, leaf_induct, toeplitz, mp_fortran), loop-basis sparse
products, GeoMG/AMG v-cycles, the macro-block Schur, wire coupler,
Python/other. Run on a QUIET box; report the second timing.

  python3 studies/mvsplit.py examples/dbc_halfbridge_r4.toml
  python3 studies/mvsplit.py /path/to/jtl.toml --freq 1e10
"""
import argparse
import cProfile
import os
import pstats
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import sppeec_threads                                          # noqa
import numpy as np                                             # noqa
import sppeec_input                                            # noqa

BUCKETS = [
    ('FMM (levels/p2p/toeplitz/fortran)',
     ('levels.py', 'leaf_induct.py', 'toeplitz.py', 'multipole.py',
      'mp_fortran', 'tree.py', 'traverse')),
    ('wire coupler', ('wirecoupler.py', 'wirekernel.py', 'wireassembly.py')),
    ('GeoMG / AMG v-cycles', ('loopmg.py', 'gpu_amg.py', 'pyamg')),
    ('Schur (lu_solve + macro B)', ('lu_solve', '_lu_solve')),
    ('sparse basis products (Y/B)', ('_sparsetools', 'csr_matvec',
                                     'csc_matvec', '_compressed.py',
                                     '_base.py', '_matmul')),
    ('Krylov/numpy algebra', ('numpy', 'lgmres', 'gmres', 'bicgstab',
                              '_isolve')),
]


def bucket_of(path):
    for name, keys in BUCKETS:
        if any(k in path for k in keys):
            return name
    return 'python / other'


def hwm_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('toml')
    ap.add_argument('--freq', type=float, default=None)
    ap.add_argument('--lsqr', action='store_true',
                    help='also time the former lsqr projections')
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    T = {}
    t0 = time.perf_counter()
    prob = sppeec_input.load(args.toml)
    freq = args.freq or prob.freqs[0]
    m = prob.model()
    T['model'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    M = prob.tree(m)
    m.prepare(M, freq)
    T['tree + prepare'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sw = prob.sweeper(m, M)
    T['solver setup'] = time.perf_counter() - t0
    print('%s  f=%g  formulation %s' % (args.toml, freq, prob.formulation))
    for k, v in T.items():
        print('  build  %-16s %8.1f s' % (k, v))
    print('  build HWM %.2f GB' % hwm_gb())

    # ---- which path? --------------------------------------------------
    S = getattr(sw, 'S', None) or getattr(sw, 'sol', None)
    if S is None and hasattr(sw, 'solver'):
        S = sw.solver
    if hasattr(S, '_mesh_matvec'):                 # equipotential
        S.model.prepare(S.M, freq)
        S.term.set_frequency(freq)
        n = S.meshsize
        op, pc = S._mesh_matvec, S._precond
        # the particular solution + readout: tree route since aa4ebbe;
        # --lsqr times the two former projections instead (hours at 18M
        # filaments -- that is the measurement, not a mistake)
        s = np.zeros(S.nnode + 2, dtype=np.complex128)
        s[S.pnode[+1]] = 1.0
        s[S.pnode[-1]] = -1.0
        t0 = time.perf_counter()
        ihat = S._tree_particular(1.0, s)
        T['tree ihat (prologue)'] = time.perf_counter() - t0
        print('  solve  %-22s %8.1f s' % ('tree ihat (prologue)',
                                          T['tree ihat (prologue)']))
        if args.lsqr:
            from scipy.sparse.linalg import lsqr
            t0 = time.perf_counter()
            BT = S.Baug.T.tocsc().astype(np.complex128)
            lsqr(BT, s, atol=1e-12, btol=1e-12)
            T['lsqr ihat (prologue)'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            zi = S.apply_Z(ihat)
            lsqr(S.Baug.astype(np.complex128), zi, atol=1e-12, btol=1e-12)
            T['lsqr phi (epilogue)'] = time.perf_counter() - t0
            for k in ('lsqr ihat (prologue)', 'lsqr phi (epilogue)'):
                print('  solve  %-22s %8.1f s' % (k, T[k]))
        print('  unknowns %d  (efg %d, terminals %d, macro %d)'
              % (n, S.efg, S.term.n,
                 getattr(S.chol, 'nmac', 0)))
    elif hasattr(S, '_matvec'):                    # wire-bond path
        S.prepare(freq) if hasattr(S, 'prepare') else None
        n = S.Bmat.shape[1]
        op, pc = S._matvec, S._precond
        print('  unknowns %d  (efg %d)' % (n, S.efg))
    else:
        sys.exit('unknown solver type %s' % type(S))

    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) + 1j*rng.standard_normal(n)
    for name, fn in (('operator apply', op), ('precond apply', pc)):
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            fn(x)
            ts.append(time.perf_counter() - t0)
        print('  %-16s %8.2f s (warm-up)   %8.2f s' % (name, ts[0], ts[1]))
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(2):
        w = op(x)
        pc(w)
    pr.disable()
    st = pstats.Stats(pr)
    tot = {}
    for (file, line, fn), (cc, nc, tt, ct, callers) in st.stats.items():
        tot[bucket_of(file + fn)] = tot.get(bucket_of(file + fn), 0.0) + tt
    total = sum(tot.values())
    print('  per iteration (operator + precond), tottime buckets over 2 rounds:')
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print('    %-36s %7.2f s  %5.1f%%' % (k, v/2, 100*v/total))
    print('    %-36s %7.2f s' % ('TOTAL', total/2))
    st.sort_stats('tottime').print_stats(18)


if __name__ == '__main__':
    main()
