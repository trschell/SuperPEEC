# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for the threaded sparse kernels (mp_fortran CSRMV*/JACOBI8*).

The claims these kernels ship under, each checked here:
  A. the fp CSR matvecs equal scipy's EXACTLY (same ordered
     accumulation, all four dtype/index variants);
  B. the int8-data twins equal the fp-stored result EXACTLY (the
     mixed int8*real product promotes to the same reals);
  C. the fused Jacobi sweep equals the numpy expression
     x + omega*di*(b - A@x) EXACTLY (same per-element op order);
  D. threaded == serial BIT-IDENTICALLY (row-parallel: each output
     element is one thread's ordered dot product) -- checked across
     OMP_NUM_THREADS subprocesses;
  E. the lossless-int8 guard refuses out-of-range / non-integer data.

loopmg's end-to-end behaviour (int8 hierarchy on a real solve, on vs
off byte-identical) is covered by validate_status's A/B run.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

FAIL = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


def make_csr(n, k, rdtype, idtype, ints=False, rng=None):
    rng = rng or np.random.default_rng(11)
    idx = np.sort(rng.integers(0, n, (n, k)), axis=1).astype(idtype)
    if ints:
        d = rng.choice([-1.0, 1.0, 4.0], (n, k)).astype(rdtype)
    else:
        d = rng.standard_normal((n, k)).astype(rdtype)
    A = sp.csr_matrix((d.ravel(), idx.ravel(),
                       np.arange(0, n*k + 1, k, dtype=idtype)),
                      shape=(n, n))
    # scipy may normalise index dtypes on construction; force ours
    A.indices = A.indices.astype(idtype)
    A.indptr = A.indptr.astype(idtype)
    return A


def main():
    import mp_fortran as mpf
    import loopmg

    n, k = 200_000, 7
    rng = np.random.default_rng(5)
    for rdtype, rtag in ((np.float32, 's'), (np.float64, 'd')):
        for idtype, itag in ((np.int32, ''), (np.int64, 'l')):
            x = rng.standard_normal(n).astype(rdtype)
            b = rng.standard_normal(n).astype(rdtype)
            # A. fp kernel == scipy
            A = make_csr(n, k, rdtype, idtype)
            f = getattr(mpf, 'csrmv_%s%s' % (rtag, itag))
            check('A: csrmv_%s%s == scipy' % (rtag, itag),
                  np.array_equal(A @ x,
                                 f(A.indptr, A.indices, A.data, x)))
            # B. int8 twin == fp result on {4,+-1} data
            G = make_csr(n, k, rdtype, idtype, ints=True)
            y_fp = G @ x
            G8 = G.copy()
            G8.data = G8.data.astype(np.int8)
            f8 = getattr(mpf, 'csrmv8_%s%s' % (rtag, itag))
            check('B: csrmv8_%s%s == fp-stored' % (rtag, itag),
                  np.array_equal(y_fp, f8(G8.indptr, G8.indices,
                                          G8.data, x)))
            # C. fused Jacobi == the numpy chain
            di = (1.0/np.maximum(np.abs(G.diagonal()), 1.0)
                  ).astype(rdtype)
            omega = 2.0/3.0
            wdi = (omega*di).astype(rdtype)
            ref = x + omega*di*(b - G @ x)
            j8 = getattr(mpf, 'jacobi8_%s%s' % (rtag, itag))
            check('C: jacobi8_%s%s == numpy sweep' % (rtag, itag),
                  np.array_equal(ref, j8(G8.indptr, G8.indices,
                                         G8.data, x, b, wdi)))
            # loopmg dispatch reaches the int8 kernel
            check('B: loopmg._spmv dispatches int8 (%s%s)'
                  % (rtag, itag),
                  np.array_equal(y_fp, loopmg._spmv(G8, x)))

    # D. threaded == serial, in fresh processes (OMP is per-process)
    probe = (
        "import sys, hashlib; sys.path.insert(0, %r)\n"
        "import numpy as np, scipy.sparse as sp, mp_fortran as mpf\n"
        "rng = np.random.default_rng(3)\n"
        "n, k = 400000, 7\n"
        "idx = np.sort(rng.integers(0, n, (n, k)), 1).astype(np.int32)\n"
        "d = rng.choice([-1, 1, 4], (n, k)).astype(np.int8)\n"
        "ip = np.arange(0, n*k + 1, k, dtype=np.int32)\n"
        "x = rng.standard_normal(n).astype(np.float32)\n"
        "b = rng.standard_normal(n).astype(np.float32)\n"
        "w = np.full(n, 0.1, np.float32)\n"
        "y = mpf.csrmv8_s(ip, idx.ravel(), d.ravel(), x)\n"
        "z = mpf.jacobi8_s(ip, idx.ravel(), d.ravel(), x, b, w)\n"
        "print(hashlib.sha256(y.tobytes()).hexdigest(),\n"
        "      hashlib.sha256(z.tobytes()).hexdigest())\n"
        % os.path.join(HERE, '..', 'src'))
    outs = []
    for t in ('1', '4'):
        r = subprocess.run([sys.executable, '-c', probe],
                           capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS=t))
        outs.append(r.stdout.strip().split()[-2:]
                    if r.returncode == 0 else ['rc%d' % r.returncode])
    check('D: threaded == serial (csrmv8 + jacobi8)',
          len(outs[0]) == 2 and outs[0] == outs[1], repr(outs))

    # E. the lossless guard
    check('E: int8 guard accepts {4,+-1}, rejects 200 and 0.5',
          loopmg._int8_ok(np.array([4.0, -1.0, 1.0], np.float32))
          and not loopmg._int8_ok(np.array([200.0], np.float32))
          and not loopmg._int8_ok(np.array([0.5], np.float32)))

    # F. tier 2: the tiled stencil engages on a real Gram and is
    # bit-identical to the csr path IT CERTIFIED AGAINST -- the
    # hierarchy's own level-0 construction. (An ad-hoc rebuilt Gram
    # is NOT a valid reference: scipy slicing can order columns
    # differently, changing the summation order at the ulp level --
    # the reference-artifact trap, again.)
    import sppeec_input
    from port_impedance import _PRECOND_DT
    doc = (open(os.path.join(HERE, '..', 'examples', 'equibar.toml'))
           .read() + '\nbasis = "overcomplete"\n')
    os.chdir(os.path.join(HERE, '..'))
    pr = sppeec_input.loads(doc)
    m = pr.model()
    M = pr.tree(m)
    m.prepare(M, 1e6)
    sw = pr.sweeper(m, M)
    S = sw.S
    mg = S.chol.mg
    check('F: stencil engaged on the equibar Gram',
          getattr(mg, '_sten0', None) is not None)
    if mg._sten0 is not None:
        YT32 = S.YT.astype(_PRECOND_DT)
        A = ((YT32 @ YT32.T).tocsr()[:S.nplaq][:, :S.nplaq]).tocsr()
        A8 = A.copy()
        A8.data = A8.data.astype(np.int8)
        rng = np.random.default_rng(23)
        ok = True
        for _ in range(3):
            x = rng.standard_normal(A.shape[0]).astype(np.float32)
            if not np.array_equal(mg._sten0.matvec(x),
                                  loopmg._spmv(A8, x)):
                ok = False
        check('F: stencil matvec bit-identical to its csr path', ok)
        wdi = mg._wdi[0]
        x = rng.standard_normal(A.shape[0]).astype(np.float32)
        b = rng.standard_normal(A.shape[0]).astype(np.float32)
        import mp_fortran as mpf
        ref = mpf.jacobi8_s(A8.indptr, A8.indices, A8.data, x, b, wdi)
        check('F: stencil fused sweep bit-identical to jacobi8',
              np.array_equal(
                  mg._sten0.jacobi(x, b, mg._wdi0_t, 1), ref))
        # threaded == serial for the tiled kernels, fresh processes
        st_probe = (
            "import os, sys, hashlib, numpy as np\n"
            "sys.path[:0] = [%r, %r]\n"
            "os.chdir(%r)\n"
            "import sppeec_input\n"
            "doc = open('examples/equibar.toml').read() + "
            "'\\nbasis = \"overcomplete\"\\n'\n"
            "pr = sppeec_input.loads(doc)\n"
            "m = pr.model(); M = pr.tree(m); m.prepare(M, 1e6)\n"
            "sw = pr.sweeper(m, M)\n"
            "st = sw.S.chol.mg._sten0\n"
            "x = np.random.default_rng(4).standard_normal("
            "st.n).astype(np.float32)\n"
            "print('HASH', hashlib.sha256("
            "st.matvec(x).tobytes()).hexdigest())\n"
            % (os.path.join(HERE, '..', 'src'), HERE,
               os.path.join(HERE, '..')))
        hs, errs = [], []
        for t in ('1', '4'):
            r = subprocess.run(
                [sys.executable, '-c', st_probe],
                capture_output=True, text=True,
                env=dict(os.environ, OMP_NUM_THREADS=t,
                         SPPEEC_GPU='0'))
            for ln in r.stdout.splitlines():
                if ln.startswith('HASH'):
                    hs.append(ln.split()[1])
            if r.returncode:
                errs.append(r.stderr.strip().splitlines()[-1]
                            if r.stderr.strip() else 'rc %d' % r.returncode)
        # a probe that DIES must read as a broken probe, not as a
        # kernel mismatch (2026-08-26: a renamed attribute produced
        # hs == [] and the check text blamed the threads)
        check('F: threaded/serial probes ran', not errs and len(hs) == 2,
              '; '.join(errs) or repr(hs))
        check('F: tiled kernels threaded == serial',
              len(hs) == 2 and hs[0] == hs[1], repr(hs))
        check('F: single-block Gram engages in exact mode',
              getattr(mg._sten0, 'mode', None) == 'exact')

    # G. multi-block geometry: no global slot order exists (measured
    # 30 precedence conflicts on the halfbridge), so the stencil
    # engages in 'reordered' mode -- same operator, summation
    # reordered at the ulp level, certified to tolerance. Sound for
    # a preconditioner: iteration counts may move, answers cannot.
    doc2 = (open('examples/module3wire.toml').read()
            + '\nbasis = "overcomplete"\n')
    pr2 = sppeec_input.loads(doc2)
    m2 = pr2.model()
    M2 = pr2.tree(m2)
    m2.prepare(M2, 1e5)
    sol2 = pr2.solver(M2, 1e5, model=m2)
    fn = getattr(sol2.chol.__call__, '__func__', None)
    cells = getattr(fn, '__closure__', None) or ()
    geo = [c.cell_contents for c in cells
           if type(c.cell_contents).__name__ == '_GeoMGFactor']
    check('G: wire-path GeoMG reachable', len(geo) == 1)
    if geo:
        mg2 = geo[0].mg
        st2 = getattr(mg2, '_sten0', None)
        check('G: multi-block Gram engages (reordered mode)',
              st2 is not None
              and getattr(st2, 'mode', None) == 'reordered')
        if st2 is not None:
            # tolerance certification: an ad-hoc Gram rebuild is a
            # VALID reference here (unlike the bitwise checks --
            # column order differs, but 1e-5 >> reorder rounding)
            idx_yt = np.r_[0:sol2.nplaq,
                           sol2.nplaq + sol2.nd:sol2.size]
            Byt = sol2.Bmat[:, idx_yt].T.tocsr().astype(np.float32)
            A2 = ((Byt @ Byt.T).tocsr()
                  [:sol2.nplaq][:, :sol2.nplaq]).tocsr()
            A2.data = A2.data.astype(np.int8)
            rng = np.random.default_rng(31)
            worst = 0.0
            for _ in range(3):
                x = rng.standard_normal(A2.shape[0]).astype(np.float32)
                ref = loopmg._spmv(A2, x)
                d = float(np.linalg.norm(st2.matvec(x) - ref)
                          / np.linalg.norm(ref))
                worst = max(worst, d)
            check('G: reordered matvec within reorder tolerance',
                  worst < 1e-5, 'worst rel %.2e' % worst)

    if FAIL:
        print('FAIL: %d check(s): %s' % (len(FAIL), FAIL))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
