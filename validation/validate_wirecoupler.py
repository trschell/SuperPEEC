# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for wirecoupler.py -- the wire near field against a real tree.

THE PROPERTY UNDER TEST IS THE SPLIT, NOT THE KERNEL. wirekernel's
values are gated in validate_wirekernel; what can silently go wrong
here is the near/far BOUNDARY: a filament assigned to the wrong side
double counts or drops once the far field is switched on. So part A
re-derives every segment's near set from first principles (global cell
// n against the segment's centroid box, exactly the 27-box rule) with
no reference to the coupler's group walk, and demands set equality.

Model: a 12x12x12 copper cube, nleaf [4,4,4], 2 levels -- big enough
for a real multilevel group structure, small enough that brute-force
enumeration over every filament is instant.

Run: PYTHONPATH=src python3 validate_wirecoupler.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import time

import numpy as np

import voxmodel
import wirekernel as wk
from wirecoupler import WireNear, WireCoupler, _thinline_mutual

FAIL = []


def check(name, ok, detail=""):
    print("  %-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def build_model():
    """A 12x12x6 metal block with air above -- the PHYSICAL hybrid
    geometry: wires fly over the voxel bulk, they are never embedded in
    it (where the wire is, there is no voxel metal; the foot attaches
    through the constriction model, not through co-volumetric mutuals).
    The z axis is spanned by one box -- the 2-D FMM ladder -- which also
    exercises the coupler on a non-cubic box partition."""
    m = voxmodel.VoxelModel('wirecoupler_gate')
    m.dims = (12, 12, 6)
    m.d = 1e-6
    m.sigma = np.full(m.dims, 5.8e7)
    m.freq = np.array([1e6])
    return m, m.build_tree(nleaf=[4, 4, 7], numlevels=2)


def build_segments(d):
    """Three deliberately different segments, all in the air above the
    block (cross-sections DISJOINT from every voxel bar -- see part F
    for why overlap is out of scope).

    A: skewed, flying just above the surface -- exercises the skew path
       on all three orientations at once.
    B: y-aligned, GRAZING the surface (axis at z = 6.5 cells, radius
       0.35, so its bottom clears the top voxel faces by 0.15 cells) --
       exercises the exact-parallel path, the perpendicular skip, the
       touching-regime quadrature, and a NEAR wire-wire pair with A.
    C: skewed, far above the block -- its 27-box neighbourhood holds no
       filaments at all (empty near row) and it is a FAR wire-wire pair
       to both A and B (box distance > 1), so its Lww off-diagonal
       blocks must be STRUCTURALLY absent.
    """
    u = np.array([0.5, 0.7, 0.3]); u /= np.linalg.norm(u)
    A = wk.round_wire(np.array([3.2, 4.1, 7.2])*d, u, 2*d, 0.35*d,
                      nring=3, nsect=(4, 8, 12), delta=0.3*d)
    B = wk.round_wire(np.array([6.5, 6.5, 6.5])*d, [0, 1, 0], 2*d,
                      0.35*d, nring=3, nsect=(4, 8, 12))
    C = wk.round_wire(np.array([5.0, 5.0, 40.0])*d, [0.8, 0.1, 0.6],
                      2*d, 0.35*d, nring=2, nsect=(6, 12))
    return [A, B, C]


def part_a(wn):
    """Near-set equality against a from-first-principles enumeration."""
    print("\nPART A -- 27-box near sets, independently re-derived")
    n = wn.n
    filbox = wn.fil_cell // n[None, :]
    ok_all, detail = True, []
    for s, fs in enumerate(wn.segments):
        want = set()
        for k, (leaf, axis, off) in enumerate(wn.leaves):
            if abs(float(fs[0].u[axis])) < 1e-14:
                continue        # perpendicular: coupler skips, so do we
            sel = np.where(
                (wn.fil_axis == axis)
                & (np.abs(filbox - wn.wbox[s][None, :]).max(axis=1) <= 1)
            )[0]
            want |= set(sel.tolist())
        rows = wn.C[wn.seg0[s]:wn.seg0[s + 1]]
        got = set(np.unique(rows.tocoo().col).tolist())
        ok_all &= (got == want)
        detail.append("%d" % len(want))
    check("near columns == brute-force box rule, all segments",
          ok_all, "sizes " + "/".join(detail))
    # the tree's own group boxes agree with the cell//n decode --
    # this is what ties filament_cells to the split
    boxes_ok = True
    for leaf, axis, off in wn.leaves:
        i0 = np.asarray(leaf.idx0, dtype=np.int64)
        gb = np.stack([np.asarray(leaf.xidx, dtype=np.int64),
                       np.asarray(leaf.yidx, dtype=np.int64),
                       np.asarray(leaf.zidx, dtype=np.int64)], axis=1)
        sel = np.where(wn.fil_axis == axis)[0]
        fb = filbox[sel] - filbox[sel]  # placeholder shape
        loc = wn.fil_cell[sel] // wn.n[None, :]
        for g in range(i0.size - 1):
            sl = np.s_[i0[g]:i0[g + 1]]
            if not np.array_equal(loc[sl],
                                  np.broadcast_to(gb[g], loc[sl].shape)):
                boxes_ok = False
    check("group (xidx,yidx,zidx) == global cell // n", boxes_ok, "")


def part_b(wn):
    """Values: C entries against the scalar kernel, pair by pair."""
    print("\nPART B -- C values == scalar wirekernel.mutual")
    rng = np.random.default_rng(3)
    worst = 0.0
    nchk = 0
    Cd = wn.C.tocsr()
    for s, fs in enumerate(wn.segments):
        rows = Cd[wn.seg0[s]:wn.seg0[s + 1]]
        coo = rows.tocoo()
        if coo.nnz == 0:
            continue
        pick = rng.choice(coo.nnz, min(40, coo.nnz), replace=False)
        for p in pick:
            e, col, v = coo.row[p], coo.col[p], coo.data[p]
            axis = int(wn.fil_axis[col])
            ref = wk.mutual(fs[e], wk.voxel_bar(wn.fil_cell[col], wn.l,
                                                axis, nq=wn.nq),
                            ng=wn.ng)
            worst = max(worst, abs(v - ref)/abs(ref))
            nchk += 1
    # ENTRY-IDENTITY TOLERANCE IS STORAGE-DTYPE-BOUND (2026-08-13):
    # kernels are built fp64 but STORED in wirecoupler._KERN_DT (fp32
    # by default, the phase-3a memory decision), so assembled entries
    # differ from a fresh fp64 computation at fp32 rounding (~1e-7
    # rel; measured 2.9e-8 here). 1e-6 still catches any real
    # quadrature or assembly bug, which errs at 1e-3+. Under
    # SPPEEC_WIRE_FP64=1 the old 1e-12 identity is reproduced.
    import wirecoupler as _wc
    _tol = 1e-12 if _wc._KERN_DT is np.float64 else 1e-6
    check("sampled C entries match scalar kernel", worst < _tol,
          "max rel %.2e over %d pairs (tol %g)" % (worst, nchk, _tol))


def part_c(wn):
    """Wire-wire block: structure and values."""
    print("\nPART C -- wire<->wire near block")
    L = wn.Lww.tocsr()
    # near-machine, not exact: the same-segment parallel-path blocks
    # evaluate G at i/j-swapped arguments in a different float order
    sym = (L - L.T)
    rel = (np.abs(sym.data).max()/np.abs(L.data).max()) if sym.nnz else 0.0
    check("Lww symmetric to machine precision", rel < 1e-12,
          "%.2e" % rel)
    # near pair (A,B): equals mutual_block -- to STORAGE precision
    # (fp32 kernel caches since phase 3a; see part B's comment)
    import wirecoupler as _wc
    _tol = 1e-14 if _wc._KERN_DT is np.float64 else 1e-6
    BAB = wk.mutual_block(wn.segments[0], wn.segments[1], ng=wn.ng)
    got = L[wn.seg0[0]:wn.seg0[1], wn.seg0[1]:wn.seg0[2]].toarray()
    rel = np.abs(got - BAB).max()/np.abs(BAB).max()
    check("near pair block == mutual_block", rel < _tol,
          "max rel %.2e (tol %g)" % (rel, _tol))
    # far pair (A,C) and (B,C): structurally absent
    far1 = L[wn.seg0[0]:wn.seg0[1], wn.seg0[2]:wn.seg0[3]]
    far2 = L[wn.seg0[1]:wn.seg0[2], wn.seg0[2]:wn.seg0[3]]
    check("far pair blocks structurally absent (FMM's job)",
          far1.nnz == 0 and far2.nnz == 0,
          "nnz %d/%d" % (far1.nnz, far2.nnz))
    # self blocks: diagonal is the GMD self term
    fs = wn.segments[1]
    g0 = wk.gmd(fs[0].off, fs[0].area)
    ref = wk.mutual(fs[0], fs[0], self_gmd=g0)
    got = L[wn.seg0[1], wn.seg0[1]]
    check("self-block diagonal == GMD self term",
          abs(got/ref - 1) < _tol, "%.6e (rel %.2e, tol %g)"
          % (got, abs(got/ref - 1), _tol))
    # empty near row for the segment above the block
    rowsC = wn.C[wn.seg0[2]:wn.seg0[3]]
    check("segment outside the lattice has an empty near row",
          rowsC.nnz == 0, "nnz %d" % rowsC.nnz)


def part_d(M, segments, wn):
    """The refusals: long segments and (elsewhere) mismatched leaves."""
    print("\nPART D -- guards")
    d = float(wn.l[0])
    lng = wk.round_wire(np.array([2, 2, 2])*d, [1, 0, 0], 10*d, 0.3*d,
                        nring=1, nsect=4)
    try:
        WireNear(M, [lng])
        check("over-long segment refused", False, "no exception")
    except ValueError as e:
        check("over-long segment refused", "split the wire" in str(e), "")


def part_e(M, segments, wn):
    """Cost, and the cheap-quadrature trade measured on the real near
    block rather than guessed from single-pair estimators."""
    print("\nPART E -- cost and quadrature economy")
    t0 = time.perf_counter()
    full = WireNear(M, segments, nq=4, ng=16)
    t1 = time.perf_counter()
    cheap = WireNear(M, segments, nq=3, ng=8)
    t2 = time.perf_counter()
    us_full = 1e6*(t1 - t0)/max(full.near_pairs, 1)
    us_cheap = 1e6*(t2 - t1)/max(cheap.near_pairs, 1)
    # compare on the union of stored entries, relative to the largest
    # entry of each row block -- tiny entries are allowed to move more
    D = (full.C - cheap.C).tocoo()
    scale = np.abs(full.C.toarray()).max(axis=1)
    rel = (np.abs(D.data)/scale[D.row]).max() if D.nnz else 0.0
    Dw = (full.Lww - cheap.Lww).tocoo()
    sw = np.abs(full.Lww.toarray()).max(axis=1)
    relw = (np.abs(Dw.data)/sw[Dw.row]).max() if Dw.nnz else 0.0
    print("    full  (ng=16, nq=4): %6.1f us/pair, %d pairs"
          % (us_full, full.near_pairs))
    print("    cheap (ng=8,  nq=3): %6.1f us/pair  (%.1fx)"
          % (us_cheap, us_full/max(us_cheap, 1e-9)))
    print("    cheap-vs-full: C %.2e, Lww %.2e (rel to row max)"
          % (rel, relw))
    check("batched near build < 400 us/pair", us_full < 400,
          "%.1f us" % us_full)
    # MEASURED TRADE (2026-08-11, this geometry): the deviation lives
    # ENTIRELY on the grazing pairs -- 1.85e-3 on the bar 0.15 cells
    # under wire B, machine precision beyond touching range -- and the
    # full rule itself carries 2.9e-4 there (part F). Both are far
    # below the 1-4-8-12 cross-section model's 0.5% worst case, so
    # (ng=8, nq=3) at 3.4x cheaper is a legitimate economy setting;
    # the gate pins the measured level, it does not aspire past it.
    check("(ng=8, nq=3) within 5e-3 of full quadrature",
          rel < 5e-3 and relw < 5e-3, "C %.2e Lww %.2e" % (rel, relw))


def part_f():
    """THE RECORDED LIMIT: overlapping cross-sections do not converge.

    A wire element COAXIAL with a voxel bar (their transverse sections
    overlapping) puts the log singularity of the parallel kernel on a
    POSITIVE-MEASURE set of the 4-D transverse product space -- the
    same slowly-convergent integral as the disc GMD, but met by plain
    product quadrature. Measured on first build (2026-08-11): cheap vs
    full moved 5.0e-2 and even (ng=32, nq=6) vs (ng=16, nq=4) moved
    3.1e-2, on exactly the bars the wire overlapped -- while every
    disjoint pair sat at machine precision.

    CONSEQUENCE: wire volumes must be carved out of the voxel metal (in
    a physical hybrid they already are -- wires fly through air and the
    foot uses the constriction model). This part pins the finding: if
    the disagreement below ever drops to quadrature-noise levels,
    something changed (singularity subtraction?) and the docstrings
    should be updated to match.
    """
    print("\nPART F -- overlapping pairs: the recorded non-convergence")
    d = 1e-6
    fs = wk.round_wire(np.array([6.5, 6.5, 5.5])*d, [0, 1, 0], 2*d,
                       0.35*d, nring=2, nsect=(6, 12))
    cell = np.array([[6, 6, 5]])       # the bar the wire is coaxial with
    a = wk.mutual_voxels(fs, cell, d, 1, nq=4, ng=16)
    b = wk.mutual_voxels(fs, cell, d, 1, nq=6, ng=32)
    rel = np.abs(a - b).max()/np.abs(a).max()
    check("coaxial overlap is NOT converged (recorded limit)",
          rel > 1e-3, "(16,4) vs (32,6): %.2e" % rel)
    # the CONTRAST that makes the rule usable: the same wire lifted to
    # GRAZE the surface (sections disjoint by 0.15 cells) converges
    fg = wk.round_wire(np.array([6.5, 6.5, 6.5])*d, [0, 1, 0], 2*d,
                       0.35*d, nring=2, nsect=(6, 12))
    cg = np.array([[6, 7, 5]])
    a = wk.mutual_voxels(fg, cg, d, 1, nq=4, ng=16)
    b = wk.mutual_voxels(fg, cg, d, 1, nq=6, ng=32)
    relg = np.abs(a - b).max()/np.abs(a).max()
    check("grazing (disjoint) pair IS converged", relg < 1e-3,
          "(16,4) vs (32,6): %.2e" % relg)


def part_g():
    """The addition theorem in the repo's normalization -- the identity
    the whole far pathway stands on. With the FMM-normalized Y and the
    z-flipped polar angle, sum_nm rho^n Y_n^-m(y) Y_n^m(x)/r^(n+1)
    must converge to 1/|x-y| for rho < r."""
    print("\nPART G -- addition theorem, repo normalization + z-flip")
    from special import sph_harm_of_cos
    rng = np.random.default_rng(0)

    def ang(v):
        r = np.linalg.norm(v)
        return r, np.pi - np.arccos(v[2]/r), np.arctan2(v[1], v[0])
    worst = 0.0
    for _ in range(4):
        y = rng.normal(size=3); y *= 0.3/np.linalg.norm(y)
        x = rng.normal(size=3); x *= 2.0/np.linalg.norm(x)
        ry, ty, py = ang(y)
        rx, tx, px = ang(x)
        s = 0.0
        for n in range(17):
            for mm in range(-n, n + 1):
                s += (ry**n*sph_harm_of_cos(n, -mm, ty, py)
                      * sph_harm_of_cos(n, mm, tx, px)/rx**(n + 1))
        worst = max(worst, abs(s.real*np.linalg.norm(x - y) - 1))
    check("identity holds at nmax=16", worst < 1e-8, "%.2e" % worst)


def _drive(M, wc, i_f, i_w):
    """One traverseRL with the coupler, jomega already set to 1.0:
    returns (filament output buffer, out_w)."""
    sizes = [np.size(leaf.idx) for leaf, _, _ in wc.leaves]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    wc.i_f = i_f.astype(np.complex128)
    wc.i_w = i_w.astype(np.complex128)
    wc.out_w = np.zeros(wc.nwel, dtype=np.complex128)
    for (leaf, _, _), o, s in zip(wc.leaves, offs, sizes):
        leaf.data = i_f[o:o + s].astype(np.complex128)
    M.traverseRL(extra=wc)
    v = np.concatenate([leaf.data for leaf, _, _ in wc.leaves])
    return v, wc.out_w


def part_h(M, wc):
    """Adjointness: wire->voxel (P2L + near) and voxel->wire (M2P +
    near) must be EXACT transposes -- the m <-> -m permutation makes
    the far tables structurally adjoint, so any disagreement is a
    plumbing bug (wrong slice, dropped conjugate, frame slip), not an
    approximation."""
    print("\nPART H -- the two directions are exact transposes")
    rng = np.random.default_rng(5)
    x = rng.normal(size=wc.nfil)          # filament currents
    y = rng.normal(size=wc.nwel)          # wire element currents
    v_fil, _ = _drive(M, wc, np.zeros(wc.nfil), y)   # wire -> voxel
    _, out_w = _drive(M, wc, x, np.zeros(wc.nwel))   # voxel -> wire
    lhs = complex((v_fil*x).sum())
    rhs = complex((out_w*y).sum())
    rel = abs(lhs - rhs)/abs(lhs)
    check("<C i_w, x> == <C^T x, i_w>", rel < 1e-12,
          "%.6e vs %.6e (rel %.2e)" % (lhs.real, rhs.real, rel))


def part_i(M, wc):
    """Accuracy of the hybrid against the DENSE exact coupling, both
    directions, plus the thin-line far wire<->wire pairs."""
    print("\nPART I -- hybrid (near + FMM far) vs dense exact coupling")
    rng = np.random.default_rng(9)
    # dense reference: every (element, filament) pair, exact kernels
    Cd = np.zeros((wc.nwel, wc.nfil))
    for s, fs in enumerate(wc.segments):
        for leaf, axis, off in wc.leaves:
            if abs(float(fs[0].u[axis])) < 1e-14:
                continue
            size = np.size(leaf.idx)
            cells = wc.fil_cell[off:off + size]
            Cd[wc.seg0[s]:wc.seg0[s + 1], off:off + size] = \
                wk.mutual_voxels(fs, cells, wc.l, axis)
    # THE FAR MODEL'S GRANULARITY, measured then set aside: every
    # element of a segment gets the SAME far coupling (one centroid
    # point source per segment), so per-element entries deviate by
    # O(a/r) -- the cross-section offset seen from the far pair. That
    # term is odd over the cross-section and CANCELS in any aggregate
    # a physical solve produces; per-entry comparisons against dense
    # therefore overstate the model error. Measured here: ~1.5e-2
    # per-entry vs ~4e-3 aggregated. Gate the aggregate; print the
    # granularity so a regression in it is still visible.
    x = rng.normal(size=wc.nfil)
    y_ent = rng.normal(size=wc.nwel)
    yseg = rng.normal(size=len(wc.segments))
    y = wc.A.T.dot(yseg)                 # uniform within each segment
    v_ent, _ = _drive(M, wc, np.zeros(wc.nfil), y_ent)
    relg = (np.abs(v_ent.real - Cd.T @ y_ent).max()
            / np.abs(Cd.T @ y_ent).max())
    v_fil, _ = _drive(M, wc, np.zeros(wc.nfil), y)
    _, out_w = _drive(M, wc, x, np.zeros(wc.nwel))
    want_v = Cd.T @ y
    relv = (np.abs(v_fil.real - want_v).max()/np.abs(want_v).max())
    got_w = wc.A.dot(out_w.real)         # per-segment totals
    want_w = wc.A.dot(Cd @ x)
    relw = np.abs(got_w - want_w).max()/np.abs(want_w).max()
    print("    per-entry granularity %.2e ; uniform-drive wire->voxel "
          "%.2e ; aggregated voxel->wire %.2e" % (relg, relv, relw))
    check("wire->voxel (uniform drive) within 1% of dense",
          relv < 1e-2, "%.2e" % relv)
    check("voxel->wire (segment totals) within 1% of dense",
          relw < 1e-2, "%.2e" % relw)
    # thin-line far wire<->wire: the claim is SEGMENT-level (mean over
    # the element block, where the O(a/r) odd term cancels); expected
    # residual is O((a/r)^2) + skew quadrature, well under 1e-3
    s1, s2 = 0, 2                        # A and C: a genuinely far pair
    B = wk.mutual_block(wc.segments[s1], wc.segments[s2])
    thin = _thinline_mutual(wc.segments[s1][0], wc.segments[s2][0])
    rel = abs(B.mean()/thin - 1)
    check("thin-line == far block MEAN to 1e-3", rel < 1e-3,
          "%.2e (per-entry spread %.2e)"
          % (rel, np.abs(B - thin).max()/np.abs(B).max()))
    # and the Wff matvec agrees with exact blocks on segment totals --
    # under UNIFORM intra-segment drive, the model's claim (random
    # per-element drive re-measures the granularity, ~4e-3)
    got = wc.A.dot(wc.A.T.dot(wc.Wff.dot(wc.A.dot(y))))
    want = np.zeros(len(wc.segments))
    Bc = wk.mutual_block(wc.segments[1], wc.segments[s2])
    yA = y[wc.seg0[0]:wc.seg0[1]]
    yB = y[wc.seg0[1]:wc.seg0[2]]
    yC = y[wc.seg0[2]:wc.seg0[3]]
    want[0] = (B @ yC).sum()
    want[1] = (Bc @ yC).sum()
    want[2] = (B.T @ yA).sum() + (Bc.T @ yB).sum()
    rel = np.abs(got - want).max()/np.abs(want).max()
    check("Wff matvec matches exact far blocks (totals) to 1e-3",
          rel < 1e-3, "%.2e" % rel)


if __name__ == '__main__':
    m, M = build_model()
    segments = build_segments(float(M.e.l[0]))
    wn = WireNear(M, segments)
    print("tree: %s cells/box, %d levels, %d filaments; %d wire elements"
          % (list(wn.n), M.numlevels, wn.nfil, wn.nwel))
    print("near pairs: %d (%.1f%% of dense)"
          % (wn.near_pairs, 100*wn.near_frac))
    part_a(wn)
    part_b(wn)
    part_c(wn)
    part_d(M, segments, wn)
    part_e(M, segments, wn)
    part_f()
    part_g()
    m.prepare(M, float(m.freq[0]))
    M.jomega = 1.0
    wc = WireCoupler(M, segments)
    part_h(M, wc)
    part_i(M, wc)
    print("\n%d checks failed" % len(FAIL))
    raise SystemExit(1 if FAIL else 0)
