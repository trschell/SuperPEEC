# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Cross-level oracle validation of the traverseP3 multilevel FAR FIELD.

The capacitive matvec phi = P q splits at numlevels > 1 into the sparse
near-field n2n (panel pairs within the 27-neighbour leaf groups, validated by
validate_p2pinit3.py) plus the panel FMM far field (scatter node charges to
panels, combined px/py/pz upward pass, m2m/m2l/l2l through the levels, l2p,
gather) -- the far field being the piece this file validates for the first
time.

Oracle: at numlevels == 1 the assembled n2n is the COMPLETE dense operator
(every panel pair through the closed-form tables). Build the same 15^3-cell
cube once at a single level and again as multilevel trees whose leaf boxes are
small enough that non-neighbour (far-field) interactions are substantial:

  * numlevels=2, nleaf=4 (4^3 boxes)  -- exercises the top-level FFT M2L;
  * numlevels=3, nleaf=2 (8^3 leaves) -- exercises mid-level M2L, M2M, L2L.

Random complex node charges are pushed through traverseP3 on each tree and
compared after mapping the multilevel (group, local) node ordering to global
coordinates. The checks are (1) the far field is a substantial fraction of
the answer (the test is not vacuous), (2) the multilevel result matches the
dense oracle, (3) the error DECREASES from nmax 2 to 4 -- the signature that
it is truncation-limited there -- and (4) the CONVERGED error (nmax 8) sits
at the finite-panel floor, the level set by P2M/L2P collapsing each panel to
its centre.

Check (4) is the one with teeth, and it is here because check (3) alone was
not enough. A GEOMETRY error raises the floor while leaving the convergence
rate looking perfectly normal: the leafinit panel-offset sign error (see
validate_leafinit_geometry.py) put the 'cell' floor at 2.5e-2, and because
that still "converged in nmax" the project's usual truncation-vs-wiring
discriminator pointed away from geometry for a long time. Converging in nmax
rules out truncation and NOTHING ELSE.

This validation CAUGHT A LATENT AUTHOR BUG in the shared multipole
machinery: midinit built the M2M shift vectors with theta = arccos(z/r)
while the leaf expansions and every M2L transfer table live in the z-FLIPPED
frame (theta = pi - arccos(z/r)) -- so m2m MIRRORED the children's
z-offsets inside each parent. The error cancels nowhere and corrupts every
interaction routed through m2m, but it is latent unless a numlevels >= 3
tree has a non-empty top-level interaction list: setup2 (2^3 top boxes, all
mutual neighbours, empty top far field) never triggered it, and the LpR
residual regressions could not see it (a self-consistently wrong operator
still converges). The final check here therefore also validates the
INDUCTIVE far field (pure Lp: r=0, jw=1) through the same failing
configuration, guarding the fix for both kernels.

Run inside the toolbox:  python3 validate_traverseP3_farfield.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import multipole as mp
import stencils as st

FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def coords(idx, dims):
    return np.stack([idx // (dims[1]*dims[2]),
                     (idx // dims[2]) % dims[1], idx % dims[2]], 1)


def nodemap(M, key1):
    """Permutation mapping this tree's node ordering into the oracle's."""
    n = M.lv[0].n.astype(int)
    glob = []
    for g in range(np.size(M.lv[0].idx0) - 1):
        fidx = np.r_[M.lv[0].idx0[g]:M.lv[0].idx0[g+1]]
        c = coords(M.lv[0].idx[fidx], n)
        c[:, 0] += M.lv[0].xidx[g]*n[0]
        c[:, 1] += M.lv[0].yidx[g]*n[1]
        c[:, 2] += M.lv[0].zidx[g]*n[2]
        glob.append(c)
    glob = np.concatenate(glob)
    key = glob[:, 0]*100000 + glob[:, 1]*1000 + glob[:, 2]
    lookup = {k: i for i, k in enumerate(key1)}
    missing = [k for k in key if k not in lookup]
    if missing:
        raise RuntimeError(
            "%d of %d multilevel nodes have no counterpart in the oracle "
            "(first: key %d = cell %s). The two trees are being decoded "
            "against DIFFERENT node lattices -- ask each tree for its own "
            "dims (M.ntotal) rather than assuming NT or NT+1."
            % (len(missing), key.size, missing[0],
               (missing[0]//100000, (missing[0]//1000) % 100, missing[0] % 1000)))
    return np.array([lookup[k] for k in key])


CELL = 1e-5
NT = np.array([15, 15, 15])
fullstruc = np.ones(NT, dtype=np.int8)

# oracle: single level, n2n == complete dense P
M1 = mp.Tree(fullstruc, NT, NT*CELL, 1, 1e0, None, capacitive=True)
# Decode against the TREE'S OWN node dims, not an assumed lattice: the node
# grid is NT.
# NT+1, so under 'cell' the oracle's own indices decoded against a 16^3
# lattice while its nodes live on 15^3, and the multilevel side then found
# no match (KeyError on a valid cell). Same class as the decode already
# fixed in validate_p2pinit3.
c1 = coords(M1.lv[0].idx, np.asarray(M1.ntotal, dtype=int))
key1 = c1[:, 0]*100000 + c1[:, 1]*1000 + c1[:, 2]
nn = M1.lv[0].idx.size
rng = np.random.default_rng(7)
q1 = (rng.standard_normal(nn) + 1j*rng.standard_normal(nn)) * 1e-12
M1.lv[0].data = q1.copy()
M1.traverseP3()
phi1 = M1.lv[0].data.copy()
print("oracle: %d nodes, dense single-level P applied\n" % nn)

def _nl(nleaf):
    """Leaf size as a 3-vector; accepts a scalar or a per-axis array.

    Per-axis leaves are what reach an ``ng == 2`` level on a box that is
    not thin -- see the ng==2 configs below.
    """
    return np.broadcast_to(np.asarray(nleaf, dtype=int), (3,)).copy()


def pinned_LT(nleaf, numlevels, capacitive=True):
    """Domain size that makes the multilevel tree's cell pitch equal CELL.

    mp.Tree pads ntotal up to ntotalfull INTERNALLY (15 -> 16 at nleaf 4,
    but 15 -> 18 at nleaf 3 and 15 -> 20 at nleaf 5), so a hard-coded pad
    silently rescales the lattice against the single-level oracle and the
    comparison then measures the rescale, not the far field. This used to
    read `LTpad = [16,16,16]*CELL`, correct only for the two leaf sizes
    exercised here. Ask a probe tree for the pitch it actually chose and
    rescale, per axis.
    """
    probe = mp.Tree(fullstruc, _nl(nleaf), NT*CELL, numlevels, 1e0,
                    2, capacitive=capacitive)
    return NT*CELL*(CELL/np.asarray(probe.e.l, dtype=float))


def run_multilevel(nleaf, numlevels, nmax):
    M = mp.Tree(fullstruc, _nl(nleaf),
                pinned_LT(nleaf, numlevels), numlevels, 1e0, nmax,
                capacitive=True)
    assert np.abs(np.asarray(M.e.l) - CELL).max()/CELL < 1e-9, \
        "cell pitch not pinned to the oracle"
    perm = nodemap(M, key1)
    M.lv[0].data = q1[perm].copy()
    near = np.asarray(M.n2n.dot(M.lv[0].data)).ravel()
    M.traverseP3()
    phi = M.lv[0].data.copy()
    err = np.linalg.norm(phi - phi1[perm])/np.linalg.norm(phi1[perm])
    farfrac = (np.linalg.norm(phi1[perm] - near)
               / np.linalg.norm(phi1[perm]))
    return err, farfrac, phi


print("%-26s %-11s %-11s" % ("config", "rel_err", "far_frac"))
err2, far2, _ = run_multilevel(4, 2, 4)
print("%-26s %-11.2e %-11.2f" % ("numlevels=2 nleaf=4 nmax=4", err2, far2))
err3, far3, _ = run_multilevel(2, 3, 4)
print("%-26s %-11.2e %-11.2f" % ("numlevels=3 nleaf=2 nmax=4", err3, far3))
err2lo, _, _ = run_multilevel(4, 2, 2)
print("%-26s %-11.2e" % ("numlevels=2 nleaf=4 nmax=2", err2lo))
err2hi, _, _ = run_multilevel(4, 2, 8)
print("%-26s %-11.2e" % ("numlevels=2 nleaf=4 nmax=8", err2hi))

# ng == 2 REGRESSION. A per-axis leaf of [4,4,8] on the 15^3 box gives a
# top level of 4x4x2, i.e. exactly TWO boxes on one axis while the other
# two still carry a real far field. Until 2026-08-05 this returned NaN
# EVERYWHERE: topinit's mirror fills were guarded by `ng[c] > 2`, so at
# ng == 2 the single negative offset was never filled, absrjn stayed 0
# there, yjnkm/absrjn produced inf/nan, and the near-field zeroing that
# follows only clears the three-axis INTERSECTION -- so the nan reached
# t3multnonsym3's fftn and poisoned the whole transfer function. Both
# kernels were affected. The finiteness check is listed first because it
# is the one that fails loudly on a regression.
err_ng2, far_ng2, phi_ng2 = run_multilevel([4, 4, 8], 2, 4)
err_ng2lo, _, _ = run_multilevel([4, 4, 8], 2, 2)
print("%-26s %-11.2e %-11.2f" % ("ng==2 (leaf 4,4,8) nmax=4", err_ng2,
                                 far_ng2))

print()
check("ng==2 top level produces FINITE output (the NaN defect)",
      np.isfinite(phi_ng2).all(),
      "%d of %d entries non-finite" % ((~np.isfinite(phi_ng2)).sum(),
                                       phi_ng2.size))
# The tolerance here is looser than the nleaf=4 checks above and that is
# the LEAF, not the ng==2. Holding the leaf shape at [4,4,8] and varying
# only the box height so the z box count goes 2/3/4 gives inductive
# 2.99e-3 / 2.72e-3 / 3.67e-3 -- flat, so ng==2 is not special once the
# mirror guards are right; an 8-cell leaf simply collapses more sources
# to the box centre than a 4-cell one. Convergence in nmax is checked
# alongside because on its own an absolute bound would not distinguish
# truncation from a wiring error sitting at the same level.
check("ng==2 top level matches dense oracle", err_ng2 < 1e-2,
      "rel err = %.2e, far field %.0f%% of |phi|"
      % (err_ng2, 100*far_ng2))
check("ng==2 error shrinks with expansion order (truncation, not wiring)",
      err_ng2 < 0.6*err_ng2lo,
      "nmax 2 -> 4: %.2e -> %.2e" % (err_ng2lo, err_ng2))
check("far field is a substantial fraction of the answer",
      far2 > 0.05 and far3 > 0.05,
      "nl=2: %.2f, nl=3: %.2f of |phi|" % (far2, far3))
check("numlevels=2 (top-level FFT M2L) matches dense oracle",
      err2 < 2e-3, "rel err = %.2e" % err2)
check("numlevels=3 (mid-level M2M/M2L/L2L + top) matches dense oracle",
      err3 < 2e-3, "rel err = %.2e" % err3)
# Raising nmax only buys anything while TRUNCATION dominates. Both schemes
# then hit a floor set by the finite panel SIZE -- P2M/L2P collapse each
# panel to its centre, and no expansion order fixes that (nleaf=4, 15^3:
# cell 3.05e-3/3.39e-4/2.82e-4/2.778e-4/2.768e-4 and edge
# 2.48e-3/2.49e-4/9.16e-5/8.49e-5/8.44e-5 at nmax 2/4/6/8/10). So compare
# 2 -> 4, where truncation still dominates in both schemes; comparing
# 4 -> 6 measures the floor and fails under 'cell' for a legitimate
# reason. The floor itself is checked separately below -- that is the
# check with teeth, because a GEOMETRY error raises the floor without
# touching the convergence rate (the leafinit panel-offset sign error put
# it at 2.5e-2 while nmax convergence still looked perfectly normal).
check("error shrinks with expansion order (truncation, not wiring)",
      err2 < 0.5*err2lo, "nmax 2 -> 4: %.2e -> %.2e" % (err2lo, err2))
# cell panels are whole faces, edge panels are quartered, so cell's floor
# sits ~3x higher; both are far below anything a geometry error produces.
floor = 5e-4
check("converged error is at the finite-panel floor (panel geometry sound)",
      err2hi < floor, "nmax=8: %.2e (floor tol %.0e)" % (err2hi, floor))

# ---- inductive far field through the same m2m-sensitive configuration ----
# (pure Lp: r=0, jomega=1 -- with realistic r the resistive term buries the
# far field and the check is vacuous, which is how the m2m bug stayed
# latent). Guards the midinit z-frame fix for the inductive kernel.
Mi1 = mp.Tree(fullstruc, NT+1, NT*CELL, 1, 1e0, None, capacitive=False)
for lf in [Mi1.e, Mi1.f, Mi1.g]:
    lf.r = 0.0
Mi1.jomega = 1.0
c1f = coords(Mi1.e.idx, Mi1.e.n.astype(int))
key1f = c1f[:, 0]*100000 + c1f[:, 1]*1000 + c1f[:, 2]
qe = (rng.standard_normal(key1f.size) + 1j*rng.standard_normal(key1f.size))
Mi1.e.data = qe.copy()
Mi1.f.data = np.zeros(Mi1.f.struc.size, np.complex128)
Mi1.g.data = np.zeros(Mi1.g.struc.size, np.complex128)
Mi1.traverseRL()
ve1 = Mi1.e.data.copy()

def run_inductive(nleaf, numlevels, nmax):
    """Inductive pure-Lp matvec on a multilevel tree, vs the dense oracle."""
    Mi = mp.Tree(fullstruc, _nl(nleaf),
                 pinned_LT(nleaf, numlevels, capacitive=False), numlevels,
                 1e0, nmax, capacitive=False)
    assert np.abs(np.asarray(Mi.e.l) - CELL).max()/CELL < 1e-9, \
        "cell pitch not pinned to the inductive oracle"
    for lf in [Mi.e, Mi.f, Mi.g]:
        lf.r = 0.0
    Mi.jomega = 1.0
    n = Mi.e.n.astype(int)
    globf = []
    for g in range(np.size(Mi.e.idx0) - 1):
        fidx = np.r_[Mi.e.idx0[g]:Mi.e.idx0[g+1]]
        c = coords(Mi.e.idx[fidx], n)
        c[:, 0] += Mi.lv[0].xidx[g]*Mi.lv[0].n[0]
        c[:, 1] += Mi.lv[0].yidx[g]*Mi.lv[0].n[1]
        c[:, 2] += Mi.lv[0].zidx[g]*Mi.lv[0].n[2]
        globf.append(c)
    globf = np.concatenate(globf)
    keyf = globf[:, 0]*100000 + globf[:, 1]*1000 + globf[:, 2]
    lookupf = {k: i for i, k in enumerate(key1f)}
    permf = np.array([lookupf[k] for k in keyf])
    Mi.e.data = qe[permf].copy()
    Mi.f.data = np.zeros(Mi.f.struc.size, np.complex128)
    Mi.g.data = np.zeros(Mi.g.struc.size, np.complex128)
    Mi.traverseRL()
    out = Mi.e.data.copy()
    err = np.linalg.norm(out - ve1[permf])/np.linalg.norm(ve1[permf])
    return err, out


erri, _ = run_inductive(2, 3, 4)
print("\ninductive pure-Lp numlevels=3 nleaf=2 nmax=4: rel err %.2e" % erri)
erri2, outi2 = run_inductive([4, 4, 8], 2, 4)
print("inductive pure-Lp ng==2 (leaf 4,4,8) nmax=4: rel err %.2e" % erri2)
check("INDUCTIVE ng==2 top level produces FINITE output",
      np.isfinite(outi2).all(),
      "%d of %d entries non-finite" % ((~np.isfinite(outi2)).sum(),
                                       outi2.size))
# Same reasoning as the capacitive ng==2 tolerance above: the level is
# set by the 8-cell leaf, verified flat across ng = 2/3/4 at that leaf.
check("INDUCTIVE ng==2 top level matches dense oracle", erri2 < 1e-2,
      "rel err = %.2e" % erri2)
check("INDUCTIVE far field matches dense oracle (m2m z-frame fix)",
      erri < 2e-3, "rel err = %.2e (was 5.1e-2 with the mirrored m2m)" % erri)

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL))
    sys.exit(1)
print("all checks passed -- traverseP3 multilevel far field agrees with the "
      "single-level dense oracle")
