# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
Created on Tue Jun  3 23:20:48 2014

@author: timmy
"""

import os as _os
import numpy as np
# import traceback
# import warnings
# import sys
# import toeplitz as tp
import multipole as mp
import meshgraph as mg
import reluctance as rel
import stencils as st
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr
from scipy.linalg import toeplitz
from scipy.linalg import lu_factor, lu_solve, cho_solve
# from pyamg.krylov import bicgstab
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import cg
from scipy.sparse.csgraph import reverse_cuthill_mckee
import pyamg
from pyamg.krylov import gmres
from pyamg.krylov import fgmres
from pyamg.util.linalg import norm
import sksparse.cholmod as cholmod
from cvxopt import spmatrix, amd
from cvxopt import matrix as cvxmatrix
from scipy.sparse import dia_matrix, coo_matrix, issparse
from scipy.sparse.linalg import splu
import time

# SystemMat and its helpers now live in systemmat.py. Re-exported here
# so the long-standing head-exec pattern (exec main.py up to
# 'conductivity = 5.81e7') keeps working for existing scripts -- but new
# code should 'from systemmat import SystemMat' instead.
from systemmat import (SystemMat, _dense_pext_fallback, print_res,
                       nfold, pscale, bscale, cscale)

filconst = 1
# alphap is retired: it was the internal-node continuity selector of the
# original (P^2 + alpha)-transformed LpPR node equations, superseded by the
# exact mixed-row matvecLpPR (see docs/alphap_whitepaper.pdf).
alphap = 1e-3
# Capacitance fold-in: number of defect-correction sweeps in the LpPR
# loop-reduction preconditioner (>=1). nfold=1 is the plain inductive saddle;
# nfold>1 folds the nodal capacitance -jw*C_cap into the preconditioner.
# Default 1 (off): on setup1 the high-frequency cost is dominated by the
# INDUCTIVE inner solve (the geometric Cholesky loop preconditioner degrading
# with omega), not the neglected capacitance -- setup1 sits far below its
# self-resonance, so the fold-in there gives limited benefit and is masked
# unless the inner saddle rtol is also tightened (measured: 58->41 outer iters
# at 10 GHz with a tight inner solve, no change with the loose default). Raise
# nfold for capacitance-dominated problems (near resonance). See
# docs/precond_whitepaper.pdf.

# LpPR preconditioner choice (peectype='LpPR'):
#   'reluctance' : FIXED sparse block-triangular preconditioner built from the
#                  reluctance K~ = Lp^{-1} (reluctance.py), solved by STANDARD
#                  GMRES. Conditioning IMPROVES with frequency -- the high-f fix
#                  (setup1: ~27 outer iters at 10 GHz vs 58-66 for 'loop'). It
#                  degrades below ~300 MHz (K~ ~ jw Z^{-1} is a poor Z^{-1} when
#                  Z is resistive-dominated), so prefer 'loop' at low frequency.
#   'loop'       : ITERATIVE loop-reduction (inductive-saddle) preconditioner,
#                  flexible FGMRES. Frequency-flat and cheap at low f; degrades
#                  toward 10 GHz. See SystemMat.loopprecinit / reluctanceprecinit
#                  and docs/precond_whitepaper.pdf.
#   'diagschur'  : FIXED sparse diagonal-admittance Schur preconditioner
#                  (PyPEEC's mathematical framework): Z -> D = diag(R + jw
#                  diag(Lp)), C_cap kept diagonal, and the sparsified MNA
#                  inverted EXACTLY via the nodal Schur complement
#                  S_d = -jw C_cap - A^T D^{-1} A (7-point pattern, cheapest
#                  per-frequency factor of the three). Run RIGHT-
#                  preconditioned (fgmres => tol on the TRUE residual):
#                  under an incomplete near-field W (multilevel/circulant)
#                  left preconditioning silently floors the true residual
#                  at 1e-6..1e-2 (profile_precond_creep.py). The low-f fix:
#                  excellent while wL/R <~ 1, degrades once the neglected
#                  mutual inductance dominates -- hand over to 'reluctance'
#                  there. See SystemMat.diagschurprecinit.
lpprsolver = 'reluctance'


rscale = 1e+0
lscale = 1e+0
# Auto-tune the leaf box size from occupancy (see recommend_leaf). OFF by
# default: it changes the near/far split, so results move at FMM
# truncation level and the byte-regression anchors no longer hold. Worth
# 14-24x on the capacitive matvec when the setup's leaf is far from the
# occupancy-appropriate one. Enable with SPPEEC_AUTOLEAF=1.
AUTOLEAF = _os.environ.get('SPPEEC_AUTOLEAF', '0') == '1'
fscale = 1e+0
iscale = 1e+0
vscale = 1e+0


# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     traceback.print_stack()
#     log = file if hasattr(file, 'write') else sys.stderr
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
# warnings.showwarning = warn_with_traceback
#
#
def recommend_leaf(fullstruc, N):
    """Leaf box size from occupancy, per the measured optimum.

    Leaf size sets the near/far split and is the dominant performance
    knob (14-24x at fixed accuracy). The optimum is NOT a constant --
    it moves strongly with fill, because in dense geometry the
    near-field pair count grows as leaf^3 per node (real near/far
    balance, small optimum) while in sparse geometry the neighbouring
    boxes are mostly empty, so the near field barely grows and larger
    boxes simply cut box count and per-box overhead.

    The two operators do NOT share an optimum, so this is a compromise,
    not either one's best. Measured interior minima:

      capacitive (traverseP3): solid 16^3 @99.98% -> leaf 4 (2.07 ms,
        vs 6.92 at leaf 2, 4.65 at leaf 8); 12^3 core @12.5% -> leaf 6;
        2x2 wire @0.17% -> leaf 16 (0.69 ms, vs 58.1 at leaf 2)
      inductive (traverseRL): solid 16^3 -> leaf 6 (11.37 ms); 2x2 wire
        -> leaf 16 or higher (8.88 ms, vs 830 at leaf 2)

    They agree at the sparse end (16) and disagree when dense (4 vs 6),
    so the dense bin uses 5: within 14% of the capacitive optimum and 6%
    of the inductive one. The curves are shallow near the floor, so
    landing within a factor of two captures most of the benefit; the
    large wins come from escaping a far-too-small leaf, not from hitting
    the minimum exactly.

    Clamped to keep >= 4 leaf boxes per axis. traverseP3 returns NaN at
    ng == 2 (pre-existing far-field defect) and ng == 3 is one step from
    it, which is too close for a production default.
    """
    fill = float(np.asarray(fullstruc).mean())
    if fill > 0.5:
        leaf = 5
    elif fill > 0.05:
        leaf = 8
    else:
        leaf = 16
    nt = np.asarray(fullstruc).shape
    leaf = min(leaf, max(2, min(nt)//3))
    return np.array([leaf]*3), fill


def build_tree(fullstruc, N, LT, numlevels, lscale, nmax, capacitive,
               autoleaf=False):
    """Tree construction, optionally with the leaf size auto-tuned.

    IMPORTANT: `ntotalfull` depends on the leaf size, so changing N
    while holding LT fixed silently rescales the CELL PITCH -- a
    different physical problem, which masquerades as discretisation
    error (the artefact is independent of nmax and equals leaf/NT).
    When autoleaf changes N this therefore rescales LT to preserve the
    pitch the caller's own N/LT would have produced. Costs two extra
    throwaway builds, which is negligible against a solve.

    The rescale is PER AXIS. `ntotalfull` pads each axis independently
    up to a whole number of leaf boxes, so on a non-cubic grid the
    padding ratio differs between axes -- 60x20x20 at leaf 5 pads to
    65x25x25, i.e. 1.083 on x but 1.25 on y and z. A single scalar
    factor (what this did until 2026-07-31) then leaves ANISOTROPIC
    cells, and that failure is quiet: it shows up only as the
    x-directed filament resistance M.f.r drifting away from M.e.r and
    M.g.r. Every setup here is cubic, so the scalar was correct for
    them, but anything read in from outside -- every VoxHenry .vhr
    model, for one -- is not. See vhr.VhrModel.build_tree, which does
    the same thing for externally supplied geometry.

    Auto-tuning changes the near/far split and hence FMM truncation, so
    results move at truncation level and byte anchors do not hold. It
    is off by default for exactly that reason.
    """
    if not autoleaf:
        return mp.Tree(fullstruc, N, LT, numlevels, lscale, nmax,
                       capacitive=capacitive)
    Nnew, fill = recommend_leaf(fullstruc, N)
    if np.array_equal(np.asarray(Nnew), np.asarray(N)):
        return mp.Tree(fullstruc, N, LT, numlevels, lscale, nmax,
                       capacitive=capacitive)
    ref = mp.Tree(fullstruc, N, LT, numlevels, lscale, nmax,
                  capacitive=capacitive)
    pitch = np.array(ref.e.l, dtype=float)
    del ref
    probe = mp.Tree(fullstruc, Nnew, LT, numlevels, lscale, nmax,
                    capacitive=capacitive)
    fac = pitch/np.asarray(probe.e.l, dtype=float)
    del probe
    print("autoleaf: fill %.3f%% -> leaf %s (was %s); LT x %s to pin "
          "cell pitch %s"
          % (100*fill, [int(v) for v in Nnew],
             [int(v) for v in np.asarray(N)],
             np.array2string(fac, precision=6),
             np.array2string(pitch, precision=4)))
    M = mp.Tree(fullstruc, Nnew, np.asarray(LT, float)*fac, numlevels,
                lscale, nmax, capacitive=capacitive)
    got = np.asarray(M.e.l, dtype=float)
    if np.any(np.abs(got/pitch - 1.0) > 1e-9):
        raise AssertionError(
            "autoleaf: cell pitch not preserved: wanted %s, got %s "
            "(ratios %s)" % (list(pitch), list(got), list(got/pitch)))
    return M














conductivity = 5.81e7  # conductivity of annealed copper, Siemens/meter
f = 1e1  # operating frequency
# peectype = 'LpPR2'
peectype = 'LpR'
# The capacitive/potential (P) machinery is only needed by the LpPR* PEEC
# formulations; LpR builds an inductance-only tree and skips it (that machinery
# is also unfinished for numlevels > 1). Passed to Tree(..., capacitive=...).
capacitive = peectype in ('LpPR', 'LpPR2', 'LpPRdense')
precondtype = ['neumann']
'''
precondtype is a list and can take the following options:
neumann : Neumann-series
rdf     : RDF (for saddle-point systems)
'''
# Geometry: 1 = solid cube (setup1 -- legacy benchmark; homogeneous, keep
# for byte regressions), 2 = ring (setup2), 3 = PCB plane+traces+vias
# (setup3 -- the DEFAULT since 2026-07-27: representative of the IC/PCB
# target application).
# Geometry now lives in setups.py; these wrappers keep the driver's
# (M, fullstruc) call shape and apply the discretisation choices that are
# genuinely the DRIVER's (lscale, capacitive, AUTOLEAF), not the geometry's.
from setups import GEOMETRIES


def _build_setup(k):
    g = GEOMETRIES[k]()
    M = build_tree(g.struc, g.N, g.LT, g.numlevels, lscale, g.nmax,
                   capacitive, autoleaf=AUTOLEAF)
    return M, g.struc


def setup1():
    return _build_setup(1)


def setup2():
    return _build_setup(2)


def setup3():
    return _build_setup(3)


sourcetype = 3
f /= fscale
structtimestart = time.time()
if sourcetype == 1:
    M, fullstruc = setup1()
elif sourcetype == 2:
    M, fullstruc = setup2()
elif sourcetype == 3:
    M, fullstruc = setup3()
print("Structure setup time:", time.time() - structtimestart)
jomega = 1j*2*np.pi*f
M.e.r = M.e.l[1] / (M.e.l[0]*M.e.l[2]*conductivity) / rscale
M.f.r = M.f.l[0] / (M.f.l[1]*M.f.l[2]*conductivity) / rscale
M.g.r = M.g.l[2] / (M.g.l[0]*M.g.l[1]*conductivity) / rscale
# M.lv[0].F = 1/M.lv[0].?
S = SystemMat(M, jomega)
S.M.lv[0].neumann = False
S.M.alpha = 0
M.RDFinit()
if peectype in ('LpPR', 'LpPRdense'):
    A = LinearOperator((S.wholesize, S.wholesize), matvec=S.matvecLpPR,
                       rmatvec=S.matvecLpPR, dtype=M.lv[0].data.dtype)
    Schur = LinearOperator((S.efgsize, S.efgsize), matvec=S.matvecLpPR3,
                           rmatvec=S.matvecLpPR3, dtype=M.lv[0].data.dtype)
elif peectype == 'LpR':
    timeadjstart = time.time()
    adjmat = S.M.adjmats()
    # Z = mg.get_mesh_incidence(adjmat, S.efgsize)
    Z = mg.getmesh_fortran(adjmat, S.esize, S.efsize, S.efgsize, S.nodesize)
    Z.data = np.float64(Z.data)
    ZT = Z.T.tocsc()
    S.Zmesh = Z          # matvecLpR reads these off the instance
    S.ZmeshT = ZT
    ZT.data = np.float64(ZT.data)
    meshsize = np.shape(Z)[1]
    A = LinearOperator((meshsize, meshsize), matvec=S.matvecLpR,
                       rmatvec=S.matvecLpR, dtype=M.lv[0].data.dtype)
    print("Mesh generation time:", time.time() - timeadjstart)
elif peectype == 'LpPR2':
    A = LinearOperator((S.efgsize, S.efgsize), matvec=S.matvecLpPR2,
                       rmatvec=S.matvecLpPR2, dtype=M.lv[0].data.dtype)
sourcetimestart = time.time()
Source = np.copy(S.wholedata)
Source[:S.esize] = np.zeros_like(M.e.struc, dtype=np.complex128)
Source[S.esize:S.efsize] = np.zeros_like(M.f.struc, dtype=np.complex128)
Source[S.efsize:S.efgsize] = np.zeros_like(M.g.struc, dtype=np.complex128)
if sourcetype == 1:
    snx = np.zeros((2,), dtype=int)
    sny = np.zeros_like(snx)
    snz = np.zeros_like(snx)
    val = np.zeros_like(snx, dtype=np.complex128)
    for sn in [snx, sny, snz]:
        sn[0] = 0
        sn[1] = 5
    val[0] = -1e-6
    # val[0] = 4
    val[1] = 1e-6
elif sourcetype == 2:
    # Ring injection ACROSS THE SLOT. The geometry cuts cells
    # {ntxmid-1, ntxmid} for y < ntymid; this drives the conductor cell
    # flanking the cut on one side and drains the one on the other, so
    # current has to go the long way round the annulus.
    #
    # Stated as faces on the flanking CELLS and mapped through
    # st.port_node, so the drive lands on conductor cells.
    ntxmid = int(fullstruc.shape[0]/2)
    dz = 4
    dh = 6  # y-thickness of ring
    faces = ([((ntxmid - 2, siy, siz), 0, +1, -1e-6)
              for siy in range(dh) for siz in range(dz)]
             + [((ntxmid + 1, siy, siz), 0, -1, +1e-6)
                for siy in range(dh) for siz in range(dz)])
    # A node exists only where the CELL is conductor, and the
    # annulus clips a few of the nominal drive cells (its inner
    # radius cuts y = dh-1). Keep the ones that exist and rescale so
    # each group still carries the same total current as before.
    faces = [f for f in faces if fullstruc[f[0]] != 0]
    tot = dh*dz*1e-6
    nlo = sum(1 for f in faces if f[3] < 0)
    nhi = len(faces) - nlo
    if not nlo or not nhi:
        raise RuntimeError("setup2 drive has no conductor cells on one "
                           "side of the slot (%d / %d)" % (nlo, nhi))
    nodes = [st.port_node(c, a, s) for c, a, s, _ in faces]
    snx = np.array([n[0] for n in nodes], dtype=int)
    sny = np.array([n[1] for n in nodes], dtype=int)
    snz = np.array([n[2] for n in nodes], dtype=int)
    val = np.array([(-tot/nlo if f[3] < 0 else tot/nhi) for f in faces],
                   dtype=np.complex128)
elif sourcetype == 3:
    # Port at mid-trace: inject on top of signal trace 1, return on the
    # bottom of the ground plane directly beneath -- drives current out
    # along the trace, down the corner vias, and back through the plane.
    # Described as voxel FACES and mapped to nodes through
    # st.port_node: the trace cell is z=3 and the plane cell z=0,
    # each driven at its own centre.
    _port3 = (((7, 4, 3), 2, +1, +1e-6),      # top of signal trace 1
              ((7, 4, 0), 2, -1, -1e-6))      # bottom of the plane below
    _nodes = [st.port_node(c, a, s) for c, a, s, _ in _port3]
    snx = np.array([n[0] for n in _nodes], dtype=int)
    sny = np.array([n[1] for n in _nodes], dtype=int)
    snz = np.array([n[2] for n in _nodes], dtype=int)
    val = np.array([v for _, _, _, v in _port3], dtype=np.complex128)
val /= iscale
Source[S.efgsize:] = S.M.parsesource(snx, sny, snz, val, 'node')
# Source[S.efgsize:] = np.zeros_like(M.lv[0].struc, dtype=np.complex128)
# Source[S.efgsize+108972]
# Source[S.efgsize+69504] = 1
# Source[S.efgsize+300] = -1
# Source[S.efgsize+57403] = 1
# Source[S.efgsize+299] = -1
# Source[69504] = 1
# Source[300] = -1

if peectype in ('LpPR', 'LpPRdense'):
    # Mixed-row RHS matching matvecLpPR: external node rows get P s_n,
    # internal rows keep the raw injection s_n (see
    # docs/alphap_whitepaper.pdf, eq. (14)).
    SourceOrig = Source.copy()
    S.M.lv[0].data[:] = Source[S.efgsize:]
    S.M.traverseP3()
    Source[S.efgsize:][S.extmask] = S.M.lv[0].data[S.extmask]
elif peectype == 'LpPR2':
    In = np.copy(Source[S.efgsize:])
    S.RHS2(Source)
    Source = np.copy(S.wholedata[:S.efgsize])

print("setup complete")
print("Source setup time:", time.time() - sourcetimestart)

P = None
if peectype == 'LpPR':
    # All LpPR preconditioners solve the SAME W-rescaled MNA operator K' =
    # rescaleLpPR (external node rows scaled by P_ext^{-1} -> standard PEEC MNA);
    # they differ in the preconditioner and hence the Krylov solver:
    #   'reluctance' : FIXED sparse block-triangular preconditioner from K~ =
    #                  Lp^{-1}, applied under STANDARD GMRES. The high-f solver
    #                  (conditioning improves with frequency).
    #   'diagschur'  : FIXED sparse diagonal-admittance Schur preconditioner
    #                  (PyPEEC framework), STANDARD GMRES. The low-f solver
    #                  (exact once the neglected mutual inductance is small).
    #   'loop'       : ITERATIVE inductive-saddle (loop-reduction) preconditioner
    #                  under flexible FGMRES; frequency-flat, best at low f.
    # All are matrix-free/sparse and scale. The dense block-triangular
    # alternative (precondinitLpPR) is validated separately by
    # validate_lppr_precond.py but does not scale.
    precondtimestart = time.time()
    Kprime = LinearOperator((S.wholesize, S.wholesize),
                            matvec=S.rescaleLpPR, dtype=np.complex128)
    if lpprsolver == 'reluctance':
        S.reluctanceprecinit()
        PLpPR = LinearOperator((S.wholesize, S.wholesize),
                               matvec=S.precondReluctance, dtype=np.complex128)
    elif lpprsolver == 'diagschur':
        S.diagschurprecinit()
        PLpPR = LinearOperator((S.wholesize, S.wholesize),
                               matvec=S.precondDiagSchur, dtype=np.complex128)
    else:
        S.loopprecinit()
        PLpPR = LinearOperator((S.wholesize, S.wholesize),
                               matvec=S.precondFoldin, dtype=np.complex128)
    print("LpPR preconditioner setup time:", time.time() - precondtimestart)
elif peectype == 'LpPRdense':   # retained dense-densify path (debug only)
    itervector = np.zeros((A.shape[0],), dtype=np.complex128)
    Afull = np.zeros((A.shape[0], A.shape[1]), dtype=np.complex128)
    for ii in range(A.shape[0]):
        itervector[:] = 0
        itervector[ii] = 1
        Afull[:, ii] = A*itervector
    AA = Afull[:S.efgsize, :S.efgsize]
    # AAinv = np.linalg.inv(AA)
    AB = Afull[:S.efgsize, S.efgsize:]
    AC = Afull[S.efgsize:, :S.efgsize]
    AD = Afull[S.efgsize:, S.efgsize:]
    # ADinv = np.linalg.inv(AD)
    # BDinvC = np.dot(np.dot(AB, ADinv), AC)
    # diagBDinvC = np.diag(BDinvC)
    # BDinvCapprox = np.zeros_like(BDinvC)
    #     BDinvCapprox[i, i] = diagBDinvC[i]
    # AinvA = np.linalg.inv(AA - BDinvC)
    # AinvB = -np.dot(np.dot(AinvA, AB), ADinv)
    # AinvC = -np.dot(np.dot(ADinv, AC), AinvA)
    # AinvD = ADinv - np.dot(np.dot(ADinv, AC), AinvB)
    # P = np.bmat([[AinvA, AinvB], [AinvC, AinvD]])
elif peectype == 'LpPR2':
    itervector = np.zeros((S.wholesize,), dtype=np.complex128)
    Afull = np.zeros((S.wholesize, S.wholesize), dtype=np.complex128)
    for ii in range(S.wholesize):
        itervector[:] = 0
        itervector[ii] = 1
        Afull[:, ii] = S.matvecLpPR(itervector)
    AA = Afull[:S.efgsize, :S.efgsize]
    AAinv = np.linalg.inv(AA)
    AB = Afull[:S.efgsize, S.efgsize:]
    AC = Afull[S.efgsize:, :S.efgsize]
    AD = Afull[S.efgsize:, S.efgsize:]
    ADinv = np.linalg.inv(AD)
    BDinvC = np.dot(np.dot(AB, ADinv), AC)
    CAinvB = np.dot(np.dot(AC, AAinv), AB)
    CDinv = -np.dot(BDinvC, AAinv)
    # P = np.dot(AAinv, np.eye(S.efgsize) + CDinv + np.dot(CDinv, CDinv))
    Schur = Afull
    SchurInv = np.linalg.inv(Afull)
    lowSchurInv = np.abs(SchurInv) < 0.27
    # SchurInv[lowSchurInv] = 0
    # P = SchurInv
elif peectype == 'LpR':
    pass
    itervector = np.zeros((meshsize,), dtype=np.complex128)
    Afull = np.zeros((meshsize, meshsize), dtype=np.complex128)
    for ii in range(meshsize):
        itervector[:] = 0
        itervector[ii] = 1
        Afull[:, ii] = S.matvecLpR(itervector)
        # Pfull[:, ii] = S.M.RDFapply2(itervector)
    #     Afull[ii, ii] += 1e-5
    # # Pfull = np.linalg.inv(Afull)
    # AA = Afull[:S.efgsize, :S.efgsize]
    # AB = Afull[:S.efgsize, S.efgsize:]
    # AC = Afull[S.efgsize:, :S.efgsize]
    # AD = Afull[S.efgsize:, S.efgsize:]

    precondtime = time.time()
    # ZTZcoo = ZT.dot(Z).tocoo()
    # ZTZcvx = spmatrix(ZTZcoo.data.tolist(), ZTZcoo.row.tolist(),
    #                   ZTZcoo.col.tolist())
    # ZTZamd = np.int32(amd.order(ZTZcvx)).flatten()
    # ZTZsparse = ZT.dot(Z)
    # ZTZsparse = ZTZsparse.T
    # ZTZsparse.data = np.float32(ZTZsparse.data)
    # ZTZsparse.indices = ZTZamd[ZTZsparse.indices]
    # ZTZsparse = ZTZsparse.tocsc()
    # ZTZsparse.indices = ZTZamd[ZTZsparse.indices]
    # ZTZchol = cholmod.cholesky(ZTZsparse, ordering_method='amd')
    ZT32 = ZT.copy()
    ZT32.data = np.float32(ZT32.data)
    Zchol = cholmod.cholesky_AAt(ZT32, mode='supernodal', ordering_method='amd')
    time1 = time.time()
    print("Time to setup Cholesky preconditioner:", time1 - precondtime)
    # ZTZfac = cholmod.symbolic(ZTZcvx)
    # cholmod.numeric(ZTZcvx, ZTZfac)
    # ZTZilu = spilu(ZTZsparse.T, drop_tol=1e-8)
    def PZTZ_to_32(vec):
        realvec = Zchol(np.float32(np.real(vec)))
        imagvec = Zchol(np.float32(np.imag(vec)))
        # cholmod.solve(ZTZfac, realvec)
        # cholmod.solve(ZTZfac, imagvec)
        return np.float64(realvec) + 1j*np.float64(imagvec)
    PZTZchol = LinearOperator((meshsize, meshsize), dtype=np.complex128,
                              matvec=PZTZ_to_32)
    # Mdiagdata = np.zeros(ZTZsparse.shape[0])
    #     rowstart = ZTZsparse.indptr[row]
    #     rowstop = ZTZsparse.indptr[row+1]
    #     Mdiagdata[row] = norm(ZTZsparse.data[rowstart:rowstop])**2
    # Mdiag = dia_matrix((Mdiagdata, 0), shape=ZTZsparse.shape)
    # def PZTZcg_to_32(vec):
    #     print("applying preconditioner ...")
    #     xcg = cg(ZTZsparse, np.complex64(vec), tol=1e-8, maxiter=1000)
    #     # if xcg[1] == 0:
    #     #     print("Successful return!")
    #     print("norm:", norm(np.abs(ZTZsparse.dot(xcg[0]) - np.complex64(vec))))
    #     return np.complex128(xcg[0])
    # PZTZcg = LinearOperator((meshsize, meshsize), dtype=np.complex128,
    #                         matvec=PZTZcg_to_32)
    # PAA = np.zeros_like(AA)
    # PAA[argsignif] = AA[argsignif]
    # PZTAAZ = np.dot(np.dot(ZT.toarray(), AA), Z.toarray())
    # from scipy.sparse.linalg import spilu
    # from scipy.sparse import csc_matrix
    # PILU = spilu(csc_matrix(PZTAAZ))
    # PLO = LinearOperator((meshsize, meshsize), matvec=PILU.solve,
    #                      dtype=M.lv[0].data.dtype)

    # SchurD = AA - np.dot(np.dot(AB, np.linalg.inv(AD)), AC)
    # SchurDinv = np.linalg.inv(SchurD)
    # Pfull = np.zeros((S.efgsize, S.efgsize), dtype=np.complex128)
    #     itervector[:] = 0
    #     itervector[ii] = 1
    #     Pfull[:, ii] = S.M.spaiapply2(itervector)
    # Prec = np.zeros((S.efgsize, S.efgsize), dtype=np.complex128)
    # PfullnnzX, PfullnnzY = np.nonzero(Pfull)
    # Prec[PfullnnzX, PfullnnzY] = SchurDinv[PfullnnzX, PfullnnzY]
    # S.M.SchurDinv = Prec
    # S.M.SchurD = SchurD
    # S.M.AB = AB
    # S.M.AC = AC
    # S.M.Dinv = np.linalg.inv(AD)
    #
    # Schur = np.dot(np.dot(AC, np.linalg.inv(AA)), AB)
    # SchurInv = np.linalg.inv(Schur)
    # AD = np.eye(S.nodesize)
    # AA[:S.esize, S.esize:S.efsize] += np.dot(AB[:S.esize, S.efgsize:],
    #                                          AC[S.efgsize:, S.esize:S.efsize])
    # AA[:S.esize, S.efsize:S.efgsize] += \
    #     np.dot(AB[:S.esize, S.efgsize:], AC[S.efgsize:, S.efsize:S.efgsize])
    # AA[S.esize:S.efsize, S.efsize:S.efgsize] += \
    #     np.dot(AB[S.esize:S.efsize, S.efgsize:],
    #            AC[S.efgsize:, S.efsize:S.efgsize])
    # Pinv = np.bmat([[AA, AB], [AC, AD]])
    # P = np.linalg.inv(Pinv)
    # Pfull[:S.efgsize, :S.efgsize] = np.linalg.inv(Afull)[:S.efgsize, :S.efgsize]
    # import scipy.sparse
    # eyemat[S.efgsize:, S.efgsize:] = 1e-3*np.eye(S.nodesize)
    # Acsc = scipy.sparse.csc_matrix(Afull + eyemat)
    # invA = scipy.sparse.linalg.spilu(Acsc, drop_tol=1e-3)
    # (w, v) = np.linalg.eig(Afull)
    # Ainv = np.dot(np.dot(v.T, winv), v)
    # P = LinearOperator((S.wholesize, S.wholesize), matvec=S.M.RDFapply3,
    #                    dtype=M.lv[0].data.dtype)


if peectype == 'LpR':
    B = LinearOperator((S.nodesize, S.efgsize), matvec=S.connectB,
                       rmatvec=S.connectBT, dtype=M.lv[0].data.dtype)
    BT = LinearOperator((S.efgsize, S.nodesize), matvec=S.connectBT,
                        rmatvec=S.connectB, dtype=M.lv[0].data.dtype)
    solve = lsqr(B, Source[S.efgsize:], atol=1e-12, btol=1e-12)
    xhat = solve[0]
    S.wholedata[:S.efgsize] = xhat
    S.M.traverseRL()
    Vf1 = Source[:S.efgsize] - S.wholedata[:S.efgsize]
    Vm = ZT.dot(Vf1)
    # Vf2 = np.dot(ZZ.T, SourceOrig[:S.efgsize] - np.dot(AA, xhat))
    # (v, flag) = gmres(A, Vm, maxiter=1, tol=1e-10, restrt=30)
    inititer = S.numiters
    (v, flag) = lgmres(A, Vm, M=PZTZchol, maxiter=10, rtol=1e-12, inner_m=30)
    # (v, flag) = fgmres(A, Vm, M=PZTZcg, maxiter=150, tol=1e-8)
    # (v, flag) = bicgstab(A, Vm, maxiter=20, tol=1e-12, callback=print_res)
    x = Z.dot(v) + xhat
    S.wholedata[:S.efgsize] = x
    S.M.traverseRL()
    sanity = S.wholedata[:S.efgsize].copy()
    S.wholedata[:S.efgsize] *= -1
    S.wholedata[:S.efgsize] += Source[:S.efgsize]
    solve = lsqr(BT, S.wholedata[:S.efgsize])
    y = solve[0]
    print("final iteration count:", S.numiters - inititer)
elif peectype == 'LpPR':
    # Mixed-row LpPR on the W-rescaled operator K' = rescaleLpPR.
    #   'reluctance' : STANDARD (left-preconditioned) GMRES -- a FIXED linear
    #                  operator. Best at high frequency.
    #   'diagschur'  : FIXED operator too, but run RIGHT-preconditioned
    #                  (fgmres) so the tolerance is on the TRUE residual --
    #                  under an incomplete near-field W it would otherwise
    #                  hide the (WP-I) rescale defect (see the solve branch
    #                  below and profile_precond_creep.py). Best at low f.
    #   'loop'       : flexible FGMRES -- the loop-reduction preconditioner is
    #                  itself iterative (inner lgmres on the loop-reduced
    #                  impedance), a varying operator, so FGMRES is required.
    # See SystemMat.reluctanceprecinit / diagschurprecinit / loopprecinit,
    # docs/precond_whitepaper.
    inititer = S.numiters
    Sourcep = S.rescaleRHS(Source)
    if lpprsolver == 'reluctance':
        # tol is on the PRECONDITIONED residual; S~^{-1} shrinks residual
        # components along the stiff nodal-Laplacian directions (scale
        # |A^T K~ A|/w), so at low frequency a loose tol under-resolves the
        # pure-KCL internal-node rows (measured at 1 GHz: true residual
        # 2.7e-3 at tol 1e-8 vs 5.9e-5 at 1e-10, for ~1.3x the matvecs).
        # Tighten toward 1e-10 below a few GHz, or check the printed true
        # residual; see validate_lppr_multilevel.py.
        (x, flag) = gmres(Kprime, Sourcep, M=PLpPR, tol=1e-8, maxiter=1,
                          restrt=300)
        print("LpPR reluctance gmres: %d outer matvecs, rel residual %.3e "
              "(rescaled K')"
              % (S.numiters - inititer, norm(Sourcep - Kprime*x) / norm(Sourcep)))
    elif lpprsolver == 'diagschur':
        # RIGHT-preconditioned (fgmres; M is a fixed operator, so this is
        # right-preconditioned GMRES -- pyamg's gmres offers only left), so
        # tol is on the TRUE residual. This is load-bearing under an
        # incomplete near-field W (multilevel / circulant trees): there the
        # rescale leaves a (W P - I) A^T i defect on the external node rows
        # that S_d^{-1} DAMPS, so left-preconditioned GMRES converges its
        # preconditioned residual while the true residual floors at
        # 1e-6..1e-2 (measured: 2.4e-2 at 1 MHz on the 15^3 multilevel tree
        # at a converged 1e-8, i.e. silently wrong currents; right-
        # preconditioned it reaches a true 1e-8 in 20-75 outers, creep
        # factor 1.6-4.2x vs complete W). See profile_precond_creep.py.
        (x, flag) = fgmres(Kprime, Sourcep, M=PLpPR, tol=1e-8, maxiter=1,
                           restrt=300)
        print("LpPR diagschur fgmres (right-prec): %d outer matvecs, rel "
              "residual %.3e (rescaled K')"
              % (S.numiters - inititer, norm(Sourcep - Kprime*x) / norm(Sourcep)))
    else:
        (x, flag) = fgmres(Kprime, Sourcep, M=PLpPR, tol=1e-8, maxiter=8,
                           restrt=40)
        print("LpPR loop fgmres: %d outer matvecs, rel residual %.3e "
              "(rescaled K')"
              % (S.numiters - inititer, norm(Sourcep - Kprime*x) / norm(Sourcep)))
elif peectype == 'LpPRdense':
    rowsc = np.abs(Afull).max(axis=1)
    rowsc[rowsc == 0] = 1.0
    Aeq = Afull / rowsc[:, None]
    colsc = np.abs(Aeq).max(axis=0)
    colsc[colsc == 0] = 1.0
    Aeq /= colsc[None, :]
    x = np.linalg.solve(Aeq, Source / rowsc) / colsc
    flag = 0
    print("equilibrated direct solve, rel residual:",
          norm(Source - Afull.dot(x)) / norm(Source))
else:
    # (x, flag) = gmres(A, Source, M=None, maxiter=100, tol=4.3e-2, restrt=25,
    # (x, flag) = gmres(A, Source, M=None, maxiter=100, tol=1e-5, restrt=25,
    (x, flag) = gmres(A, Source, maxiter=20, tol=1e-12, restrt=100,
                      callback=print_res)
    # (x, flag) = bicgstab(A, Source, maxiter=400, tol=1e-12,
    #                      callback=print_res)
    # (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var) = \
    #     lsqr(A, Source, atol=1e-8, btol=1e-8, show=True)
    print(str(S.numiters) + " matvec(s) completed")
    print("norm: " + str(norm(np.abs(Source - A*x))))
    x[S.efgsize:] *= S.M.lv[0].beta
if flag > 0:
    print("Convergence to tolerance not achieved!")
elif flag < 0:
    print("Numerical breakdown or illegal input!")
else:
    print("Success!")
# time1 is only assigned on the LpR branch; guard so the LpPR paths reach
# the solution-recovery block below instead of dying on a NameError.
if 'time1' in dir():
    print(time.time() - time1)
if peectype == 'LpR':
    print("norm: " + str(norm(np.abs(Vm - A*v))))


if peectype == 'LpPR2':
    S.wholedata[:S.efgsize] = x[:]
    S.M.lv[0].data[:] = -In
    S.M.lv[0].data[:] += S.M.connectAT()
    S.M.traverseP3()
    S.M.lv[0].data[:] /= S.M.jomega
elif peectype == 'LpR':
    S.wholedata[:S.efgsize] = x[:]
    S.M.lv[0].data[:] = y[:]
else:
    # LpPR: x is the full [currents; node potentials] vector
    S.wholedata[:] = np.asarray(x).ravel()
# slevels[ssize-1].getneighbors()
#     slevels[level].midm2linit()
# [e, f, g] = slevels[0].get_struc_efg()
# e.getslabidx()
# f.getslabidx()
# g.getslabidx()
