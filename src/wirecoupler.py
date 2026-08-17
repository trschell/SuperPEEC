# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""NEAR-FIELD coupling of bond-wire segments into the voxel lattice.

The bond-wire hybrid (wirekernel.py) replaces an arbitrarily oriented
wire, IN THE FAR FIELD, by three axis-aligned point sources at each
segment's centroid weighted by its direction cosines -- measured
O((l/r)^2) against the exact kernel, <0.2% beyond 3 leaf boxes, the same
order the FMM carries for ordinary filaments. This module supplies the
other half of that split: the DIRECT pairs the FMM's M2L ladder excludes.

THE BOUNDARY IS THE TREE'S, NOT A DISTANCE (the TerminalCoupler rule).
A wire segment is attached to the leaf box containing its centroid; its
near set is that box plus the 26 neighbours -- exactly what ``leaf.p2p``
covers -- in EVERY orientation leaf, because a skewed wire couples to
all three filament orientations at once (the e/f/g split exists because
perpendicular filaments have zero mutual, which is also why the
three-point-source far field is exact in principle). Anything other
than the 27-box rule double counts or silently drops pairs once the far
field is switched on.

WHAT IS BUILT:
  ``C``    sparse (n wire elements x e|f|g buffer): wire<->voxel
           mutual partial inductances over the 27-box neighbourhood,
           evaluated by the batched exact kernels
           (:func:`wirekernel.mutual_voxels` -- parallel pairs closed
           form, skewed pairs analytic-inner + Gauss-outer).
  ``Lww``  sparse (n wire elements x n wire elements), symmetric:
           wire<->wire near pairs -- segment pairs whose boxes are
           within Chebyshev distance 1 (same-segment diagonal blocks
           included, with GMD self terms). Far wire<->wire pairs are
           the FMM's job, through the same point sources.

FRAME CONVENTION. Wire geometry is given in the model frame with the
origin at the corner of voxel (0,0,0) and cell pitch ``leaf.l``: the
voxel bar at low cell c runs from the centre of c to the centre of c+1,
``p0 = (c + 0.5)*l`` -- the :func:`wirekernel.voxel_bar` convention,
which the exact-degenerate gate in validate_wirekernel ties to
``terminal.box_mutual_matrix``.

SEGMENT LENGTH RULE. The far-field point-source error is O((l_seg/r)^2)
with r >= 2 leaf boxes for the nearest far pair, so segments must stay
SHORT RELATIVE TO THE LEAF BOX. A segment longer than one box extent is
refused outright -- its far field would be wrong by construction, and
splitting a wire into more segments is always available to the caller.

NO OVERLAPPING METAL. Wire cross-sections must be DISJOINT from every
voxel bar's: a wire element whose section overlaps a bar's (e.g. a wire
coaxial with a filament column it is embedded in) puts the parallel
kernel's log singularity on a positive-measure set, and the quadrature
then converges as slowly as the disc GMD -- measured 3e-2 still moving
between (ng=16, nq=4) and (ng=32, nq=6), against machine precision for
every disjoint pair (validate_wirecoupler part F pins this). This is
not a modelling restriction in practice: where the wire is, there is no
voxel metal -- wires fly through air, and the bond foot couples through
the constriction model, not through co-volumetric mutuals. Grazing
(touching but disjoint) pairs are fine: the analytic axial integral
softens 1/r to a log and the touching-pair study measured sub-0.04%
at n=3 even at l/a = 0.2.
"""
import numpy as np
import scipy.sparse as sp

import wirekernel as wk

# fp32 campaign PHASE 3a (decided with the user 2026-08-13): the wire
# coupling caches in single precision. The R3 census ranked them the
# LARGEST solver-side store: _far 494 MB complex128 + _kern_ww 142 +
# _kern_wv 45 MB float64 -> ~340 MB saved at this rung alone. Safe by
# structure: kernels are BUILT in fp64 (the quadratures cancel), only
# STORED single; every consumer accumulates through numpy promotion
# into complex128 buffers (reweight products, p2m's (G*data).sum(),
# pre_l2p's scalar*G), so entry error is ~1e-7 relative -- far below
# every gate (hybrid-vs-dense 2e-3, dense Z oracle 2e-5). The adjoint
# identity survives EXACTLY: read and inject share one stored table,
# so its residual stays at fp64 rounding regardless of storage dtype.
# TIER-1 NOTE: the build-time C/Lww are assembled from the STORED
# (single) kernels, not the fresh fp64 ones, so reweight() remains
# bit-identical to the initial build. SPPEEC_WIRE_FP64=1 restores
# double-precision caches for A/B measurement.
import os as _os
_KERN_DT = (np.float64 if _os.environ.get('SPPEEC_WIRE_FP64') == '1'
            else np.float32)
_KERN_CDT = (np.complex128 if _KERN_DT is np.float64
             else np.complex64)
from equiterminal import _leaf_order, filament_cells

MU0 = 4e-7*np.pi


def _segment_centroid(fs):
    f0 = fs[0]
    return f0.p0 + 0.5*f0.length*f0.u


class WireNear:
    """Near-field blocks for a set of wire segments against a tree.

    Parameters
    ----------
    M : multipole.Tree
    segments : list of list of :class:`wirekernel.Filament`
        One entry per wire segment; the elements of a segment share
        ``p0``/``u``/``length`` (:func:`wirekernel.round_wire` output).
        Chain/wire bookkeeping stays with the caller -- this class only
        needs the flat segment list, and ``seg0`` gives each segment's
        row offset in the element numbering.
    nq, ng : int
        Voxel cross-section and outer axial quadrature orders for the
        skewed near pairs. The defaults are the gated full-accuracy
        ones; validate_wirecoupler part E measures the cheaper
        (ng=8, nq=3) setting so the trade is recorded, not guessed.
    """

    def __init__(self, M, segments, nq=4, ng=16, chunk=256):
        if not segments:
            raise ValueError("no wire segments given")
        self.M = M
        self.segments = segments
        self.nq, self.ng = int(nq), int(ng)
        self.leaves = _leaf_order(M)
        # one box partition for all three orientations -- the near/far
        # split must be identical per orientation component or a wire
        # pair could be near in x and far in y, double counting.
        l0 = np.asarray(self.leaves[0][0].l, dtype=float)
        n0 = np.asarray(self.leaves[0][0].n, dtype=np.int64)
        for leaf, _, _ in self.leaves[1:]:
            if not (np.allclose(leaf.l, l0) and
                    np.array_equal(np.asarray(leaf.n, dtype=np.int64), n0)):
                raise RuntimeError("e/f/g leaves disagree on box "
                                   "partition -- near/far split undefined")
        self.l = l0
        self.n = n0
        # per-filament axis and LOW cell, buffer order (e|f|g)
        self.fil_axis, self.fil_cell = filament_cells(M)
        self.nfil = self.fil_axis.size
        offs = [off for _, _, off in self.leaves]
        sizes = [np.size(leaf.idx) for leaf, _, _ in self.leaves]
        if [0, sizes[0], sizes[0] + sizes[1]] != offs:
            raise RuntimeError("leaf buffer offsets disagree with idx "
                               "sizes (%s vs %s)" % (offs, sizes))
        # groups per leaf: box coords -> filament slice
        self._groups = []
        for leaf, axis, off in self.leaves:
            if M.numlevels == 1:
                i0 = np.array([0, np.size(leaf.idx)], dtype=np.int64)
                gof = {(0, 0, 0): 0}
            else:
                i0 = np.asarray(leaf.idx0, dtype=np.int64)
                box = np.stack([np.asarray(leaf.xidx, dtype=np.int64),
                                np.asarray(leaf.yidx, dtype=np.int64),
                                np.asarray(leaf.zidx, dtype=np.int64)],
                               axis=1)
                gof = {tuple(b): g for g, b in enumerate(box)}
            self._groups.append((i0, gof, off, axis))
        # segment -> element row offsets, and attachment boxes
        counts = [len(fs) for fs in segments]
        self.seg0 = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        self.nwel = int(self.seg0[-1])
        box_ext = self.n*self.l
        self.wbox = np.zeros((len(segments), 3), dtype=np.int64)
        for s, fs in enumerate(segments):
            f0 = fs[0]
            if M.numlevels > 1 and f0.length > float(box_ext.min()):
                raise ValueError(
                    "segment %d is longer (%.3g m) than a leaf box "
                    "(%.3g m); its far-field point source would be wrong "
                    "by construction -- split the wire into shorter "
                    "segments" % (s, f0.length, float(box_ext.min())))
            c = np.floor(_segment_centroid(fs)/self.l).astype(np.int64)
            self.wbox[s] = 0 if M.numlevels == 1 else c // self.n
        self._build_wire_voxel(chunk)
        self._build_wire_wire()

    # -- wire <-> voxel --------------------------------------------------

    def near_columns(self, s):
        """Buffer-global filament indices in segment ``s``'s 27-box
        neighbourhood, one array per orientation leaf."""
        out = []
        b = self.wbox[s]
        for i0, gof, off, axis in self._groups:
            parts = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        g = gof.get((b[0] + dx, b[1] + dy, b[2] + dz))
                        if g is not None and i0[g + 1] > i0[g]:
                            parts.append(np.arange(i0[g], i0[g + 1]))
            cols = (np.concatenate(parts) + off if parts
                    else np.zeros(0, dtype=np.int64))
            out.append(cols)
        return out

    def _build_wire_voxel(self, chunk):
        self._kern_wv = []
        rows, cols, vals = [], [], []
        npairs = 0
        for s, fs in enumerate(self.segments):
            r0 = self.seg0[s]
            E = len(fs)
            for cset, (i0, gof, off, axis) in zip(self.near_columns(s),
                                                  self._groups):
                # a wire exactly perpendicular to this orientation
                # couples to none of its filaments (zero mutual) --
                # don't store a block of structural zeros
                if cset.size == 0 or abs(float(fs[0].u[axis])) < 1e-14:
                    continue
                cells = self.fil_cell[cset]
                # POINT-RESOLVED kernel cached (geometry only): a
                # frequency retune reweights it with the new shapes
                # instead of requadraturing -- the Tier-1 setup fix
                K, eid = wk.mutual_voxels(fs, cells, self.l, axis,
                                          nq=self.nq, ng=self.ng,
                                          chunk=chunk,
                                          return_kernel=True)
                K = np.asarray(K, dtype=_KERN_DT)
                self._kern_wv.append((s, K, eid, cset))
                B = wk.reweight_rows(K, eid, fs)
                rr, cc = np.meshgrid(r0 + np.arange(E), cset,
                                     indexing='ij')
                rows.append(rr.ravel())
                cols.append(cc.ravel())
                vals.append(B.ravel())
                npairs += E*cset.size
        if rows:
            self.C = sp.csr_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.nwel, self.nfil))
        else:
            self.C = sp.csr_matrix((self.nwel, self.nfil))
        self.near_pairs = npairs
        self.near_frac = (float(npairs)/(self.nwel*self.nfil)
                          if self.nfil else 0.0)

    def reweight(self):
        """Recompute C and Lww from the cached point-resolved kernels
        with the segments' CURRENT shapes (after Filament.set_shape).
        Geometry, near sets, sparsity and everything else persist --
        this is the whole per-frequency coupling cost."""
        rows, cols, vals = [], [], []
        for s, K, eid, cset in self._kern_wv:
            fs = self.segments[s]
            B = wk.reweight_rows(K, eid, fs)
            rr, cc = np.meshgrid(self.seg0[s] + np.arange(len(fs)),
                                 cset, indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(B.ravel())
        if rows:
            self.C = sp.csr_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.nwel, self.nfil))
        rows, cols, vals = [], [], []

        def _put(r0, c0, B):
            rr, cc = np.meshgrid(r0 + np.arange(B.shape[0]),
                                 c0 + np.arange(B.shape[1]),
                                 indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(B.ravel())

        for s1, s2, K, e1, e2 in self._kern_ww:
            B = wk.reweight_block(K, e1, e2, self.segments[s1],
                                  self.segments[s2])
            _put(self.seg0[s1], self.seg0[s2], B)
            if s2 != s1:
                _put(self.seg0[s2], self.seg0[s1], B.T)
        self.Lww = sp.csr_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nwel, self.nwel))

    # -- wire <-> wire ---------------------------------------------------

    def _build_wire_wire(self):
        S = len(self.segments)
        rows, cols, vals = [], [], []

        def _put(r0, c0, B):
            rr, cc = np.meshgrid(r0 + np.arange(B.shape[0]),
                                 c0 + np.arange(B.shape[1]),
                                 indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(B.ravel())

        self._kern_ww = []
        for s1 in range(S):
            Ks, eids = wk.segment_self_kernel(self.segments[s1])
            Ks = np.asarray(Ks, dtype=_KERN_DT)
            self._kern_ww.append((s1, s1, Ks, eids, eids))
            _put(self.seg0[s1], self.seg0[s1],
                 wk.reweight_block(Ks, eids, eids, self.segments[s1],
                                   self.segments[s1]))
            for s2 in range(s1 + 1, S):
                if np.abs(self.wbox[s1] - self.wbox[s2]).max() > 1:
                    continue          # a far pair -- see Wff below
                K, e1, e2 = wk.mutual_block(self.segments[s1],
                                            self.segments[s2],
                                            ng=self.ng,
                                            return_kernel=True)
                K = np.asarray(K, dtype=_KERN_DT)
                self._kern_ww.append((s1, s2, K, e1, e2))
                B = wk.reweight_block(K, e1, e2, self.segments[s1],
                                      self.segments[s2])
                _put(self.seg0[s1], self.seg0[s2], B)
                _put(self.seg0[s2], self.seg0[s1], B.T)
        self.Lww = sp.csr_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nwel, self.nwel))


def _thinline_mutual(f1, f2, ng=24, tol=1e-9):
    """Partial mutual (H) between two segment AXES as thin lines.

    For FAR segment pairs the cross-section correction is O((a/r)^2) --
    negligible at >= 2 boxes -- so the exact thin-line double integral
    is both cheaper and MORE accurate than a truncated multipole
    exchange. Collinear disjoint spans are safe: the exact parallel
    form's ln(d) terms cancel identically (their coefficients sum to
    zero), leaving only benign float cancellation.
    """
    dot = float(f1.u @ f2.u)
    if abs(dot) < tol:
        return 0.0
    if abs(abs(dot) - 1.0) < tol:
        sgn = 1.0 if dot > 0 else -1.0
        u = f1.u
        s1a = float(f1.p0 @ u)
        s2a = float(f2.p0 @ u)
        s2b = s2a + sgn*f2.length
        dp = f2.p0 - f1.p0
        dperp = dp - (dp @ u)*u
        d = max(float(np.linalg.norm(dperp)), 1e-30)
        return sgn*MU0/(4*np.pi)*wk.mutual_parallel(
            s1a, s1a + f1.length, min(s2a, s2b), max(s2a, s2b), d)
    return dot*MU0/(4*np.pi)*wk.mutual_skew_lines(
        f1.p0, f1.u, f1.length, f2.p0, f2.u, f2.length, ng=ng)


class WireCoupler(WireNear):
    """Near + FAR wire coupling: the ``extra`` object for traverseRL.

    The far field is ATTACHMENT-FREE -- the open question left by the
    TerminalCoupler pattern (whose sources must sit in a leaf box that
    owns a group of the right orientation; a bond wire arcing through
    free air routinely has no such box). Instead of attaching wire
    moments to somebody else's box, the coupler talks to the leaf
    expansions directly, in both directions:

      voxel -> wire  M2P: at the ``p2m`` hook ``leaf.above.data`` holds
                     every group's fresh multipole moments; the wire
                     potential is read off far groups' moments at the
                     segment centroid.
      wire -> voxel  P2L: at the ``pre_l2p`` hook ``leaf.above.data``
                     holds every group's local expansion; the wire
                     injects local coefficients into far groups and the
                     tree's own ``leaf.l2p()`` distributes them to
                     filaments with its m0 scaling.

    Both use the classic addition theorem in the repo's normalization
    (verified to machine precision in validate_wirecoupler part G):
    with moments M_nm = sum q rho^n Y_n^{-m}, the potential outside is
    sum M_nm Y_n^m(x)/r^{n+1}, and a source at y outside a box induces
    locals L_nm = Y_n^{-m}(y)/rho^{n+1} read back with r^n Y_n^m. The
    P2L table is therefore the M2P table with columns permuted
    m <-> -m, which makes the two directions EXACT TRANSPOSES by
    construction. Box centres are anchored EMPIRICALLY (world minus
    leafinit-frame position of one filament per leaf, asserted uniform)
    rather than derived from the stagger conventions -- the half-cell
    trap has been hit too many times in this repo to re-derive frames.

    A segment enters orientation d with weight w = u_d*l_seg/l_axis
    (the TerminalCoupler t_l/l_axis rule, per direction cosine): a
    point source, exact to O((l_seg/r)^2) -- measured <0.2% beyond 3
    leaf boxes, the FMM's own order. wire<->wire far pairs skip the
    tree entirely: exact thin-line segment mutuals (``Wff``, built
    once), better than a truncated multipole exchange and trivially
    cheap at bond-wire segment counts.

    MATVEC PROTOCOL (mirrors TerminalCoupler): before traverseRL set
      ``i_f``   the e|f|g filament input currents (a copy -- the sweep
                overwrites ``leaf.data``),
      ``i_w``   the wire element currents,
      ``out_w`` a zeroed complex accumulator, length ``nwel``.
    After the sweep ``leaf.data`` includes jomega*(wire->voxel
    coupling) via the injected locals and the near hook; ``out_w``
    holds the RAW voxel->wire couplings (near + far) -- the caller
    applies jomega and adds the wire's own R, Lww and Wff terms
    (:meth:`wire_matvec`). Scaling by jomega for out_w is left to the
    caller because the sweep's own jomega multiply happens after the
    hooks fire.

    SCALING NOTE. The far tables are flat over (segment, far group):
    build and matvec cost O(n_seg * n_groups * nnmax), storage one
    complex table of that size. For bond-wire counts (hundreds of
    segments) against the first target models this is trivial; a
    320^2-board-with-wires build would want the hierarchical upgrade
    (read parent-level moments for distant groups) -- documented, not
    built.
    """

    def __init__(self, M, segments, nq=4, ng=16, chunk=256):
        super().__init__(M, segments, nq=nq, ng=ng, chunk=chunk)
        S = len(segments)
        # per-leaf near-column slices, both directions, sliced once
        self._Cn = []
        self._CnT = []
        for leaf, axis, off in self.leaves:
            size = np.size(leaf.idx)
            blk = self.C[:, off:off + size].tocsr()
            self._Cn.append(blk)
            self._CnT.append(blk.T.tocsr())
        if M.numlevels > 1:
            self._build_far()
        self._build_wire_far()
        self.i_f = None
        self.i_w = None
        self.out_w = None

    def reweight(self):
        super().reweight()
        # refresh the per-leaf column slices the hooks use
        self._Cn = []
        self._CnT = []
        for leaf, axis, off in self.leaves:
            size = np.size(leaf.idx)
            blk = self.C[:, off:off + size].tocsr()
            self._Cn.append(blk)
            self._CnT.append(blk.T.tocsr())
        # far tables, Wff, A_foot: geometry-only, untouched

    # -- far tables ------------------------------------------------------

    def _anchor(self, leaf, axis, off):
        """World position of each group's expansion centre, EMPIRICAL.

        Reconstructs leafinit's box-frame position for every filament
        of this leaf (node family on the filament's own axis, centre
        family transversely -- the 'cell' scheme rule) and subtracts it
        from the filament's world position. The result must be constant
        within every group; asserted, because a half-cell slip here
        would silently misplace every far interaction.
        """
        n = self.n
        l = self.l
        node = [np.arange(-(n[i] - 1)/2 - 0.5, (n[i] - 1)/2 + 0.5)
                for i in range(3)]
        centre = [np.arange(-(n[i] - 1)/2, (n[i] - 1)/2 + 1)
                  for i in range(3)]
        fam = [node[i] if i == axis else centre[i] for i in range(3)]
        idx = np.asarray(leaf.idx, dtype=np.int64)
        ix = idx//(n[1]*n[2])
        iy = (idx//n[2]) % n[1]
        iz = idx % n[2]
        pbox = np.stack([fam[0][ix]*l[0], fam[1][iy]*l[1],
                         fam[2][iz]*l[2]], axis=1)
        sel = slice(off, off + idx.size)
        pw = (self.fil_cell[sel] + 0.5)*l[None, :]
        pw[:, axis] += 0.5*l[axis]
        anchors = pw - pbox
        i0 = np.asarray(leaf.idx0, dtype=np.int64)
        cg = np.zeros((i0.size - 1, 3))
        for g in range(i0.size - 1):
            if i0[g + 1] > i0[g]:
                a = anchors[i0[g]:i0[g + 1]]
                if np.abs(a - a[0]).max() > 1e-9*l.min():
                    raise RuntimeError(
                        "leaf %s group %d: expansion centre not uniform "
                        "-- frame decode broke" % (leaf.orientation, g))
                cg[g] = a[0]
        return cg

    def _build_far(self):
        from special import sph_harm_of_cos
        nmax = self.leaves[0][0].nmax
        nnmax = (nmax + 1)**2
        # m <-> -m column permutation: G with this permutation IS the
        # P2L table, so both directions share one array
        self._mflip = np.concatenate(
            [[n*n + n - m for m in range(-n, n + 1)]
             for n in range(nmax + 1)]).astype(np.int64)
        self._far = []          # per leaf: list over segments of
        #                         (group indices, G table) or None
        self._wsrc = np.zeros((len(self.leaves), len(self.segments)))
        for k, (leaf, axis, off) in enumerate(self.leaves):
            cg = self._anchor(leaf, axis, off)
            i0 = np.asarray(leaf.idx0, dtype=np.int64)
            gbox = np.stack([np.asarray(leaf.xidx, dtype=np.int64),
                             np.asarray(leaf.yidx, dtype=np.int64),
                             np.asarray(leaf.zidx, dtype=np.int64)],
                            axis=1)
            occupied = np.where(i0[1:] > i0[:-1])[0]
            per_leaf = []
            for s, fs in enumerate(self.segments):
                ud = float(fs[0].u[axis])
                if abs(ud) < 1e-14:
                    per_leaf.append(None)
                    continue
                self._wsrc[k, s] = ud*fs[0].length/float(self.l[axis])
                far = occupied[np.abs(gbox[occupied] - self.wbox[s]
                                      [None, :]).max(axis=1) >= 2]
                if far.size == 0:
                    per_leaf.append(None)
                    continue
                x = (fs[0].p0 + 0.5*fs[0].length*fs[0].u)[None, :] - cg[far]
                r = np.linalg.norm(x, axis=1)
                th = np.pi - np.arccos(x[:, 2]/r)      # z-FLIPPED frame
                ph = np.arctan2(x[:, 1], x[:, 0])
                G = np.zeros((far.size, nnmax), dtype=np.complex128)
                for nn in range(nmax + 1):
                    rn = r**(nn + 1)
                    for mm in range(-nn, nn + 1):
                        G[:, nn*nn + nn + mm] = sph_harm_of_cos(
                            nn, mm, th, ph)/rn
                # built fp64 (the harmonics/r^n cancellation), STORED
                # single -- consumers accumulate in complex128
                per_leaf.append((far, np.ascontiguousarray(
                    G.astype(_KERN_CDT))))
            self._far.append(per_leaf)

    def _build_wire_far(self):
        """Exact thin-line mutuals for FAR segment pairs (box Chebyshev
        distance >= 2) -- the pairs Lww excludes."""
        S = len(self.segments)
        W = np.zeros((S, S))
        for s1 in range(S):
            for s2 in range(s1 + 1, S):
                if np.abs(self.wbox[s1] - self.wbox[s2]).max() <= 1:
                    continue
                v = _thinline_mutual(self.segments[s1][0],
                                     self.segments[s2][0])
                W[s1, s2] = W[s2, s1] = v
        self.Wff = W
        # element <-> segment aggregation (elements are parallel
        # branches: far field couples to the segment TOTAL current and
        # feeds back the same voltage on every element)
        self.A = sp.csr_matrix(
            (np.ones(self.nwel),
             (np.concatenate([np.full(len(fs), s)
                              for s, fs in enumerate(self.segments)]),
              np.arange(self.nwel))),
            shape=(S, self.nwel))

    def wire_matvec(self, i_w):
        """The wire<->wire inductive coupling: exact near blocks plus
        thin-line far pairs. RAW partial inductances -- the caller
        applies jomega alongside out_w."""
        return self.Lww.dot(i_w) + self.A.T.dot(self.Wff.dot(
            self.A.dot(i_w)))

    # -- the traverseRL hooks --------------------------------------------

    def _leaf_index(self, leaf):
        for k, (lf, _, _) in enumerate(self.leaves):
            if lf is leaf:
                return k
        raise RuntimeError("unknown leaf in hook")

    def p2m(self, leaf):
        k = self._leaf_index(leaf)
        if self.M.numlevels == 1:
            return
        for s in range(len(self.segments)):
            ent = self._far[k][s]
            if ent is None:
                continue
            far, G = ent
            m0 = MU0/(4*np.pi)*float(self.l[
                self.leaves[k][1]])**2
            val = (G*leaf.above.data[far]).sum()
            self.out_w[self.seg0[s]:self.seg0[s + 1]] += \
                m0*self._wsrc[k, s]*val

    def p2p(self, leaf):
        k = self._leaf_index(leaf)
        off = self.leaves[k][2]
        size = self._Cn[k].shape[1]
        self.out_w += self._Cn[k].dot(self.i_f[off:off + size])
        leaf.data += self._CnT[k].dot(self.i_w)

    def pre_l2p(self, leaf):
        k = self._leaf_index(leaf)
        for s in range(len(self.segments)):
            ent = self._far[k][s]
            if ent is None:
                continue
            far, G = ent
            i_s = complex(self.i_w[self.seg0[s]:self.seg0[s + 1]].sum())
            if i_s == 0:
                continue
            leaf.above.data[far, :] += \
                (self._wsrc[k, s]*i_s)*G[:, self._mflip]

    def l2p(self, leaf):
        pass
