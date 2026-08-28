# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Is the macro-block rank deficit PREDICTABLE from pure counting?

THE STALL (docket 2026-08-27): the macro Schur block is rank-deficient
(dependent hole generators), its LU inverts v-cycle noise, and every
cheap truncation gives WRONG answers.  The proposed structural fix is a
combinatorial dedup -- but a dedup needs a TARGET COUNT it can trust.

CLAIM UNDER TEST.  The plaquette block's rank is fixed by topology:

    rank(plaquettes) = nplaq - n_voxels - n_cavities

because the kernel of "2-cell -> its boundary 1-chain" is spanned by
closed surfaces, and those are exactly the individual voxel boundaries
(n_voxels of them, independent since there are no 3-cycles) plus one
per enclosed cavity.  If that holds, then the number of genuinely
independent hole generators is known WITHOUT any rank computation:

    K_true = (efg - nn + ncomp) - rank(plaquettes)

and the greedy tree-cotree collapse's overshoot becomes measurable at
any scale, for free, in integer arithmetic.

Measured here against dense SVD ground truth on a small perforated
board.  Cheap on purpose: this must never become another hero run.
"""
import os
import sys

sys.path[:0] = ['src', 'studies']

import numpy as np

import equiterminal as eq
from pdn_planes import build_pdn

NPLANE = int(os.environ.get('NPLANE', 32))
STITCH = int(os.environ.get('STITCH', 8))
FREQ = 1e8


def rank_dense(A, tol=None):
    """Numerical rank via dense SVD -- small matrices only."""
    D = np.asarray(A.todense(), dtype=np.float64)
    s = np.linalg.svd(D, compute_uv=False)
    if tol is None:
        tol = max(D.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 0)
    return int((s > tol).sum()), s


def main():
    print('building pdn nplane=%d stitch=%d' % (NPLANE, STITCH))
    m = build_pdn(nplane=NPLANE, eps_r=None, stitch=STITCH)
    sig = np.asarray(m.sigma)
    n_voxels = int((sig != 0).sum())
    leaf, levels = m.partition()
    M = m.build_tree(leaf, levels)
    m.prepare(M, FREQ)
    S = eq.EquiTerminalSolver(m, M, 0, basis='overcomplete')

    nplaq = int(S.nplaq)
    nholes = int(S.nholes)
    nportcyc = int(S.nportcyc)
    nu = int(S.nu)
    nmacro = nholes + nportcyc + nu
    efg = int(S.efg)
    meshsize = int(S.meshsize)

    print('\n--- counts ---')
    print('conductor voxels      %d' % n_voxels)
    print('filaments (efg)       %d' % efg)
    print('plaquettes            %d' % nplaq)
    print('hole generators       %d' % nholes)
    print('port cycles           %d' % nportcyc)
    print('redistribution modes  %d' % nu)
    print('macro columns         %d' % nmacro)
    print('meshsize (total cols) %d' % meshsize)

    Y = S.Y.tocsc()
    print('\ncomputing dense ranks (%d x %d) ...' % Y.shape)
    r_all, _ = rank_dense(Y)
    r_p, _ = rank_dense(Y[:, :nplaq])

    macro_indep = r_all - r_p
    macro_def = nmacro - macro_indep

    print('\n--- measured ranks ---')
    print('rank(Y)               %d   (meshsize %d, overcomplete by %d)'
          % (r_all, meshsize, meshsize - r_all))
    print('rank(plaquettes)      %d   (nplaq %d, deficient by %d)'
          % (r_p, nplaq, nplaq - r_p))
    print('macro independent     %d  of %d' % (macro_indep, nmacro))
    print('MACRO DEFICIENCY      %d   <-- the stall driver' % macro_def)

    print('\n--- the counting prediction ---')
    pred_rp_nocav = nplaq - n_voxels
    print('predicted rank(plaq)  %d   (nplaq - n_voxels, cavities=0)'
          % pred_rp_nocav)
    print('measured  rank(plaq)  %d' % r_p)
    gap = pred_rp_nocav - r_p
    print('gap                   %d   %s' % (
        gap, 'MATCH (cavities=0 confirmed)' if gap == 0
        else 'implies %d cavities / extra relations' % gap))

    cyc_graph = r_all  # rank(Y) IS the cycle space the solver targets
    k_true = cyc_graph - r_p
    print('\nindependent macro directions needed  %d' % k_true)
    print('greedy collapse produced             %d hole generators'
          % nholes)
    print('port cycles (expected independent)   %d' % nportcyc)
    print('=> hole-generator overshoot          %d'
          % (nmacro - k_true))


if __name__ == '__main__':
    main()
