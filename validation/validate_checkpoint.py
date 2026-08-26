# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for the Krylov checkpoint/resume (SPPEEC_CHECKPOINT).

Built 2026-08-26 after a 4.4 h R5 solve died one outer cycle short of
tolerance under a dropped ssh session. The claims checked:

  A. an interrupted solve (budget-truncated here, standing in for a
     kill) leaves a checkpoint file behind;
  B. a relaunched solve resumes from it -- converging in FEWER
     matvecs than a cold solve -- and lands the same answer within
     warm-start tolerance (the shipped sweep-warm-start precedent:
     identical physics, different iterate path);
  C. the file is removed on converged success, so stale checkpoints
     only ever describe interrupted solves;
  D. a checkpoint of a DIFFERENT system (wrong rhs) is ignored: the
     solve runs cold and lands the cold answer.

module3wire through the library path; maxiter is overridden to
manufacture the interruption, which is exactly the plumbing under
test (krylov_solve), not a solver default being second-guessed.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import os
import tempfile

import numpy as np

FAIL = []


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


def main():
    os.environ.pop('SPPEEC_CHECKPOINT', None)
    import sppeec_input
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..'))
    pr = sppeec_input.load('examples/module3wire.toml')
    m = pr.model()
    M = pr.tree(m)
    f = 1e5

    def solve(maxiter=None):
        m.prepare(M, f)
        sol = pr.solver(M, f, model=m)
        kw = {} if maxiter is None else {'maxiter': maxiter}
        Z, info = sol.solve(f, current=pr.current, rtol=pr.rtol, **kw)
        return complex(Z), info

    # cold reference
    Z0, i0 = solve()
    check('cold solve converges', i0['flag'] == 0,
          'mv %d' % i0['matvecs'])

    ck = os.path.join(tempfile.mkdtemp(prefix='sppeec_ck_'), 'ck.npz')
    os.environ['SPPEEC_CHECKPOINT'] = ck
    os.environ['SPPEEC_CHECKPOINT_S'] = '0'    # dump every outer
    try:
        # A. interrupted solve leaves a checkpoint
        Zt, it = solve(maxiter=2)
        check('A: truncated solve did not converge', it['flag'] != 0,
              'flag %s mv %d' % (it['flag'], it['matvecs']))
        check('A: checkpoint file left behind', os.path.exists(ck))

        # B + C. resume: fewer matvecs, same answer, file removed
        Z1, i1 = solve()
        rel = abs(Z1 - Z0)/abs(Z0)
        check('B: resumed solve converges in fewer matvecs',
              i1['flag'] == 0 and i1['matvecs'] < i0['matvecs'],
              'mv %d vs cold %d' % (i1['matvecs'], i0['matvecs']))
        # two equally-converged iterate paths agree to ~rtol, no
        # tighter -- the residual bounds the answer at that level
        # (measured: 9.8e-6 at rtol 1e-4; the sweep warm start's
        # "identical to 8.6e-13" was at rtol 1e-12)
        check('B: resumed answer within the solve tolerance of cold',
              rel < pr.rtol, 'rel %.2e (rtol %g)' % (rel, pr.rtol))
        check('C: checkpoint removed on success',
              not os.path.exists(ck))

        # D. a foreign checkpoint is ignored
        n = None
        # manufacture one with the right size but wrong rhs
        Zt2, it2 = solve(maxiter=2)          # to learn the size
        with np.load(ck) as z:
            n = z['x'].shape[0]
        rng = np.random.default_rng(1)
        np.savez(ck + '.tmp.npz',
                 x=np.zeros(n, np.complex64),
                 rhs=rng.standard_normal(n).astype(np.complex64))
        os.replace(ck + '.tmp.npz', ck)
        Z2, i2 = solve()
        rel2 = abs(Z2 - Z0)/abs(Z0)
        check('D: foreign checkpoint ignored (cold-equivalent solve)',
              i2['flag'] == 0 and rel2 < 1e-8
              and i2['matvecs'] >= i0['matvecs'] - 2,
              'mv %d vs cold %d, rel %.2e'
              % (i2['matvecs'], i0['matvecs'], rel2))
    finally:
        os.environ.pop('SPPEEC_CHECKPOINT', None)
        os.environ.pop('SPPEEC_CHECKPOINT_S', None)

    if FAIL:
        print('FAIL: %d check(s): %s' % (len(FAIL), FAIL))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
