# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""LeafPoten leaf-level class: the capacitive near-field (P2P) operator.

Counterpart of :class:`leaf_induct.LeafInduct` for the *capacitive* side of
the PEEC discretization: instead of current filaments coupled by partial
inductances, this level holds charge *panels* coupled by coefficients of
potential (the ``P`` in ``LpPR``). The kernels come from :mod:`greens`
(:func:`gen2_p_parz` for parallel panels, :func:`gen_p_per` for perpendicular
ones), and the near-field product is again a Toeplitz/FFT convolution over
the 27-box neighborhood.

The extra complication relative to the inductive leaf is that panels come in
three orientations (``'x'``, ``'y'``, ``'z'``, by surface normal) that all
couple to each other: each LeafPoten instance is one *source* orientation and
produces fields on all three *target* orientations at once, on grids that are
mutually staggered by half a cell. The per-axis workspace sizes this implies
are computed by :meth:`coeffpoten_sizes`.

STATUS: the FFT path (:meth:`p2pinit`/:meth:`p2p`) is now FINISHED and
validated against the explicit assembly by ``validate_leaf_poten_p2p.py``
-- all nine source/target orientation pairs agree with :meth:`p2pinit3`
to ~1e-15 across cubic, anisotropic and large-leaf geometries. It is not
yet wired into ``tree.traverseP3``, which still applies the near field as
the assembled sparse ``n2n``; that assembly is still needed regardless,
because the W = P_ext^-1 rescaling and C_cap want matrix entries rather
than an operator (the same split ``circulant_poten`` uses).

The explicit sparse path :meth:`p2pinit3` is the validated oracle (see
``validate_p2pinit3.py``): it works at any ``numlevels``, with
same-orientation blocks exactly symmetric, cross blocks reciprocal, the
far field matching the point-charge law, and the multilevel assembly
agreeing with the single-level one to machine precision. Split out of
multipole.py.
"""
from multipole_common import *  # noqa: F401,F403  shared imports/guards
from special import *  # noqa: F401,F403
from greens import *  # noqa: F401,F403
import stencils as st
from levels import Level, LeafLevel, MidLevel, TopLevel


class LeafPoten(LeafLevel):
    """Leaf-level coefficient-of-potential operator for one panel orientation.

    Extends :class:`levels.LeafLevel` with the capacitive near-field
    machinery: precomputed 27-neighbor transfer arrays for all three target
    orientations (:meth:`p2pinit`, or the sparse-matrix variant
    :meth:`p2pinit3`) and the slab-sweep driver :meth:`p2p` that applies them.

    Parameters
    ----------
    n : ndarray of int
        Panels per box along x, y, z.
    ng : ndarray of int
        Number of leaf boxes along x, y, z.
    l : ndarray of float
        Physical panel/cell dimensions along x, y, z (metres).
    nmax : int or None
        Maximum expansion order (``None`` = single-group/direct mode in
        :meth:`p2pinit3`).
    orientation : {'x', 'y', 'z'}
        Surface-normal direction of this instance's source panels.

    Attributes
    ----------
    self_poten : float or None
        Self coefficient of potential of one panel (set by
        :meth:`p2pinitold`; the newer init paths leave it ``None``).

    Notes
    -----
    Unfinished --- see the module docstring. The constructor deliberately does
    not call any ``p2pinit`` variant.
    """
    def __init__(self, n, ng, l, nmax, orientation):
        LeafLevel.__init__(self, n, ng, l, nmax, orientation)
        self.self_poten = None

    def coeffpoten_sizes(self, px, py, pz):
        """Padded transform length per axis, per target orientation.

        The convolution for one (target group, source group) pair only has
        to represent the slot difference d = stride_t*i - stride_s*j with
        i over the target's panels and j over the source's -- the
        neighbour offset lives in the KERNEL, since every one of the 27
        neighbours carries its own transfer block. So the requirement per
        axis is just

            sz[t, c] = ss[c]*n_src[c] + st[c]*n_tgt[c]

        with ss/st the same half-unit lattice strides :meth:`p2p` uses
        (2 on the axis where the relevant grid is flat, else 1); the extra
        slot beyond the reachable d range is the circulant's zero wrap.

        The original sizing handed out 4m on transverse axes and 3m on the
        target normal, as though a single array had to span the whole
        three-box neighbourhood. That is 4x (parallel) to 6x (cross) more
        volume than the convolution can ever address, and every element of
        it was being transformed.
        """
        s = {'x': 0, 'y': 1, 'z': 2}[self.orientation]
        targets = (px, py, pz)
        sz = np.zeros((3, 3), np.int_)
        for t in range(3):
            ss, stt = self.lattice_strides(t)
            for c in range(3):
                sz[t, c] = ss[c]*int(self.n[c]) + stt[c]*int(targets[t].n[c])
        return sz

    def lattice_strides(self, t):
        """Per-axis (source, target) placement strides for block self -> t.

        Under the EDGE scheme a panel is a quarter face and a cross block
        couples grids staggered by half a cell, so the flat grid is
        upsampled onto a common half-unit lattice -- stride 2 on whichever
        grid is flat along the axis in question.

        Under the CELL scheme panels are WHOLE faces and every grid is a
        full-cell lattice, so both strides are 1 on every axis and a cross
        block is a plain convolution in the panel index. The half-unit
        machinery simply does not apply; see the cross-table branch in
        :meth:`p2pinit`.
        """
        return [1, 1, 1], [1, 1, 1]

    def p2pinit(self, px, py, pz):
        """Precompute the 27-neighbor potential transfer arrays (dense path).

        Builds ``self.p2p_trans_x/_y/_z`` --- one FFT-ready Toeplitz transfer
        block per neighbor offset (indexed ``9(dx+1)+3(dy+1)+(dz+1)``) and per
        *target* orientation --- from the closed-form kernels of
        :mod:`greens`: :func:`gen2_p_parz` couples this source orientation to
        same-oriented (parallel) panels, :func:`gen_p_per` to the two
        perpendicular orientations. The kernel axes are permuted so each
        table is in the leaf's x, y, z order, and the sizes come from
        :meth:`coeffpoten_sizes`.

        The inner helper ``pidx(c, t, n)`` generates the 1-D gather indices
        along axis ``c`` for target orientation ``t`` and neighbor offset
        ``n`` in ``{-1, 0, 1}``; its four cases encode the half-cell
        staggering between normal and transverse axes (the strided ``::2``
        ranges pick alternating half-step samples). Each gathered block is
        pre-transformed with :func:`tp.t3multnonsym4` for the FFT convolution
        in :meth:`p2p`; the raw (untransformed) blocks are kept in
        ``self.transx/y/z`` for inspection.

        Validated against the explicit :meth:`p2pinit3` assembly by
        ``validate_leaf_poten_p2p.py`` (all nine source/target orientation
        pairs agree to ~5e-16).
        """
        if self.orientation == 'x':
            s = 0  # source orientation
        elif self.orientation == 'y':
            s = 1
        elif self.orientation == 'z':
            s = 2
        else:
            raise RuntimeError('Incorrect orientation given!')
        n1 = np.zeros((3,), dtype=np.int_)
        n2 = np.zeros((3,), dtype=np.int_)
        n4 = np.zeros((3,), dtype=np.int_)
        m2 = np.zeros((3,), dtype=np.int_)
        m3 = np.zeros((3,), dtype=np.int_)
        m4 = np.zeros((3,), dtype=np.int_)
        l2 = np.zeros((3,), dtype=self.l.dtype)
        l = np.zeros((3,), dtype=self.l.dtype)
        for ii in range(3):
            n1[ii] = self.n[ii]
            n2[ii] = self.m[ii] + self.n[ii]
            n4[ii] = 3*self.m[ii] + self.n[ii]
            m2[ii] = 2*self.m[ii]
            m3[ii] = 2*self.m[ii]
            m4[ii] = 4*self.m[ii]
            l2[ii] = self.l[ii]/2
            l[ii] = self.l[ii]
        sz = self.coeffpoten_sizes(px, py, pz)
        p2p_trans_x = np.zeros((27, sz[0, 0] - 1, sz[0, 1] - 1, sz[0, 2] - 1),
                               dtype=np.complex128)
        self.p2p_trans_x = np.zeros((27, sz[0, 0], sz[0, 1], sz[0, 2]),
                                    dtype=np.complex128)
        p2p_trans_y = np.zeros((27, sz[1, 0] - 1, sz[1, 1] - 1, sz[1, 2] - 1),
                               dtype=np.complex128)
        self.p2p_trans_y = np.zeros((27, sz[1, 0], sz[1, 1], sz[1, 2]),
                                    dtype=np.complex128)
        p2p_trans_z = np.zeros((27, sz[2, 0] - 1, sz[2, 1] - 1, sz[2, 2] - 1),
                               dtype=np.complex128)
        self.p2p_trans_z = np.zeros((27, sz[2, 0], sz[2, 1], sz[2, 2]),
                                    dtype=np.complex128)
        # coeffpoten_x holds all possible partial coefficients of potential for
        # x-directed target panels for a 3x3 cluster of groups. The projection
        # runs from all points in the 3x3 cluster to all points in the center
        # group.
        if self.orientation == 'x':
            coeffpoten_x = gen2_p_parz(l[1], l[2], l[0], n4[1], n4[2], n2[0])
            coeffpoten_x = np.transpose(coeffpoten_x, (2, 0, 1))
            coeffpoten_y = gen_p_per(l[2], l2[0], l[1], 2*m4[2], 2*m4[0], 2*m4[1])
            coeffpoten_y = np.transpose(coeffpoten_y, (1, 2, 0))
            coeffpoten_z = gen_p_per(l[1], l2[0], l[2], 2*m4[1], 2*m4[0], 2*m4[2])
            coeffpoten_z = np.transpose(coeffpoten_z, (1, 0, 2))
        elif self.orientation == 'y':
            coeffpoten_x = gen_p_per(l[2], l2[1], l[0], 2*m4[2], 2*m4[1], 2*m4[0])
            coeffpoten_x = np.transpose(coeffpoten_x, (2, 1, 0))
            coeffpoten_y = gen2_p_parz(l[2], l[0], l[1], n4[2], n4[0], n2[1])
            coeffpoten_y = np.transpose(coeffpoten_y, (1, 2, 0))
            coeffpoten_z = gen_p_per(l[0], l2[1], l[2], 2*m4[0], 2*m4[1], 2*m4[2])
        elif self.orientation == 'z':
            coeffpoten_x = gen_p_per(l[1], l[0], l2[2], 2*m4[1], 2*m4[0], 2*m4[2])
            coeffpoten_x = np.transpose(coeffpoten_x, (1, 0, 2))
            coeffpoten_y = gen_p_per(l[0], l[1], l2[2], 2*m4[0], 2*m4[1], 2*m4[2])
            coeffpoten_z = gen2_p_parz(l[0], l[1], l[2], n4[0], n4[1], n2[2])

        def pidx(c, t, n):
            if (c == s and t != c) or (c != s and t == c):
                if n == -1:
                    return np.r_[:m2[c], m2[c]+1:m4[c]-1:2]
                if n == 1:
                    return np.r_[m4[c]-1:m2[c]-1:-1, m2[c]-2:0:-2]
            if c == s:
                if t == c:
                    if n == -1:
                        return np.r_[1:m2[c]]
                    if n == 0:
                        return np.r_[n1[c]-1:0:-1, :n1[c]]
                    if n == 1:
                        return np.r_[m2[c]-1:0:-1]
                else:
                    return np.r_[2*(n1[c]-1)-1:0:-2, :m2[c]]
            else:
                if t == c:
                    return np.r_[m2[c]-1:-1:-1, 1:m2[c]-1:2]
                if n == -1:
                    return np.r_[1:m4[c]]
                if n == 0:
                    return np.r_[m2[c]-1:0:-1, :m2[c]]
                if n == 1:
                    return np.r_[m4[c]-1:0:-1]

        # Circulant-embedding origin per axis, per target.
        # t3multnonsym4(T, o0, o1, o2) treats T[o_c - 1] as offset zero,
        # so o_c - 1 must cover the most negative reachable slot
        # difference, which is ss[c]*(n_src[c] - 1). With the right-sized
        # sz above this is exact rather than over-provisioned, and the
        # same expression now covers parallel and cross alike.
        targets = (px, py, pz)
        origs = []
        for t in range(3):
            ss, _stt = self.lattice_strides(t)
            origs.append([ss[c]*(int(self.n[c]) - 1) + 1 for c in range(3)])

        # Cell scheme: whole-face panels. The parallel tables above are
        # already right (same-orientation faces stay in registry, panel
        # size == stride, gen2_p_parz's own domain), but the cross tables
        # must be the whole-face perpendicular kernel at half-cell
        # offsets, which gen_p_per cannot express -- exactly the swap
        # p2pinit3 makes. Use the SAME p_per_table with the SAME offset so
        # the two paths read one table convention: off[c] absorbs the
        # signed separation, which a perpendicular coupling needs because
        # it is not symmetric under reflecting one axis the way the
        # parallel one is.
        cross = [coeffpoten_x, coeffpoten_y, coeffpoten_z]
        self._celloff = []
        for t in range(3):
            off = np.array([int(self.n[c]) + int(targets[t].n[c]) + 2
                            for c in range(3)], dtype=np.int_)
            self._celloff.append(off)
            if t != s:
                cross[t] = p_per_table(l, s, t, off)
        coeffpoten_x, coeffpoten_y, coeffpoten_z = cross

        def gather1d(c, t, neigh):
            """1-D kernel gather indices along axis `c`, target `t`,
            neighbour offset `neigh`.

            Slot k of the circulant carries offset d = k - (orig-1). The
            neighbour enters as the TARGET box size p.n[c] (p2pinit3's
            convention), with a sign flip relative to p2pinit3 because the
            FFT sweep's Fortran kernel has OUTPUT = GROUP and SOURCE =
            NEIGHGROUP -- the opposite role assignment.

            Registry axes use plain |separation|. On a cross block the two
            grids are staggered by half a cell (a panel is flat along its
            own normal), so source and target are placed on a COMMON
            HALF-UNIT lattice -- upsampling whichever grid is flat -- and
            gen_p_per's half-offset indexing applies: index m for m >= 0
            else -m-1 when the source is the flat one (c == s), and the
            mirrored m-1 / -m when the target is (c == t). These are
            exactly p2pinit3's validated per-axis cases.
            """
            pn = int(targets[t].n[c])
            d = np.arange(sz[t, c] - 1) - (origs[t][c] - 1)
            # both grids full-cell: parallel is symmetric so it takes
            # |sep| as ever, cross takes the SIGNED separation shifted
            # by the same off p2pinit3 uses. The relative sign of the
            # neighbour term against d is fixed by the parallel case
            # (which passes), so it carries over unchanged -- note the
            # absolute value there hides the sign of d itself, so it
            # is the shared `d - neigh*pn` form, not the sign of d,
            # that is being relied on here.
            if t == s:
                return np.abs(d - neigh*pn)
            return d - neigh*pn + self._celloff[t][c]

        for neighx in range(-1, 2):
            #     xidx1dx = np.r_[1:m2[0]]
            #     xidx1dyz = np.r_[:m2[0], m2[0]+1:m4[0]-1:2]
            # elif neighx == 0:
            #     xidx1dx = np.r_[n1[0]-1:0:-1, :n1[0]]
            #     xidx1dyz = np.r_[2*(n1[0]-1)-1:0:-2, :m2[0]]
            # elif neighx == 1:
            #     xidx1dx = np.r_[m2[0]-1:0:-1]
            #     xidx1dyz = np.r_[m4[0]-1:m2[0]-1:-1, m2[0]-2:0:-2]
            xidxx = np.repeat(gather1d(0, 0, neighx), (sz[0, 1]-1)*(sz[0, 2]-1))
            xidxy = np.repeat(gather1d(0, 1, neighx), (sz[1, 1]-1)*(sz[1, 2]-1))
            xidxz = np.repeat(gather1d(0, 2, neighx), (sz[2, 1]-1)*(sz[2, 2]-1))
            for neighy in range(-1, 2):
                #     yidx1dy = np.r_[:m2[1], m2[1]+1:m4[1]-1:2]
                #     yidx1dxz = np.r_[1:m4[1]]
                # elif neighy == 0:
                #     yidx1dy = np.r_[m2[1]-1:-1:-1, 1:n2[1]-1:2]
                #     yidx1dxz = np.r_[m2[1]-1:0:-1, :m2[1]]
                # elif neighy == 1:
                #     yidx1dy = np.r_[m4[1]-1:m2[1]-1:-1, m2[1]-2:0:-2]
                #     yidx1dxz = np.r_[m4[1]-1:0:-1]
                yidxx = np.tile(np.repeat(gather1d(1, 0, neighy),
                                sz[0, 2] - 1), sz[0, 0] - 1)
                yidxy = np.tile(np.repeat(gather1d(1, 1, neighy),
                                sz[1, 2] - 1), sz[1, 0] - 1)
                yidxz = np.tile(np.repeat(gather1d(1, 2, neighy),
                                sz[2, 2] - 1), sz[2, 0] - 1)
                for neighz in range(-1, 2):
                    #     zidx1dxy = np.r_[1:m4[2]]
                    #     zidx1dz = np.r_[:m2[2], m2[2]+1:m4[2]-1:2]
                    # elif neighz == 0:
                    #     zidx1dxy = np.r_[m2[2]-1:0:-1, :m2[2]]
                    #     zidx1dz = np.r_[m2[2]-1:-1:-1, 1:n2[2]-1:2]
                    # elif neighz == 1:
                    #     zidx1dxy = np.r_[m4[2]-1:0:-1]
                    #     zidx1dz = np.r_[m4[2]-1:m2[2]-1:-1, m2[2]-2:0:-2]
                    zidxx = np.tile(gather1d(2, 0, neighz),
                                    (sz[0, 0] - 1) * (sz[0, 1] - 1))
                    zidxy = np.tile(gather1d(2, 1, neighz),
                                    (sz[1, 0] - 1) * (sz[1, 1] - 1))
                    zidxz = np.tile(gather1d(2, 2, neighz),
                                    (sz[2, 0] - 1) * (sz[2, 1] - 1))
                    neigh = 9*(neighx+1) + 3*(neighy+1) + neighz + 1
                    p2p_trans_x[neigh, ...] = np.reshape(
                        coeffpoten_x[xidxx, yidxx, zidxx],
                        (sz[0, 0] - 1, sz[0, 1] - 1, sz[0, 2] - 1))
                        # (2*n1[0]-1, m4[1]-1, m4[2]-1))
                    self.transx = p2p_trans_x
                    self.p2p_trans_x[neigh, ...] = tp.t3multnonsym4(
                        p2p_trans_x[neigh, ...], origs[0][0], origs[0][1],
                        origs[0][2])
                    p2p_trans_y[neigh, ...] = np.reshape(
                        coeffpoten_y[xidxy, yidxy, zidxy],
                        (sz[1, 0] - 1, sz[1, 1] - 1, sz[1, 2] - 1))
                        # (n3[0]-1, n3[1]-1, m4[2]-1))
                    self.transy = p2p_trans_y
                    self.p2p_trans_y[neigh, ...] = tp.t3multnonsym4(
                        p2p_trans_y[neigh, ...], origs[1][0], origs[1][1],
                        origs[1][2])
                    p2p_trans_z[neigh, ...] = np.reshape(
                        coeffpoten_z[xidxz, yidxz, zidxz],
                        (sz[2, 0] - 1, sz[2, 1] - 1, sz[2, 2] - 1))
                        # (n3[0]-1, m4[1]-1, n3[2]-1))
                    self.p2p_trans_z[neigh, ...] = tp.t3multnonsym4(
                        p2p_trans_z[neigh, ...], origs[2][0], origs[2][1],
                        origs[2][2])
                    self.transz = p2p_trans_z

    def p2pinitold(self, orientation):
        """Superseded first attempt at the potential transfer arrays.

        Earlier version of :meth:`p2pinit` that sized every transfer block
        uniformly (``2n - 1`` per axis, transformed with
        :func:`tp.t3multnonsym3` like the inductive case) and encoded the
        half-cell staggering in per-orientation ``xidx1d``/``yidx1d``/
        ``zidx1d`` index tables instead of the ``pidx`` helper. Also sets
        ``self.self_poten`` (the panel self coefficient) in the ``'y'``/``'z'``
        branches. Kept for reference; superseded by :meth:`p2pinit` and its
        orientation-dependent sizing.
        """
        xidx1d = np.zeros((3, 4*self.n[0] - 1), dtype=int)
        yidx1d = np.zeros((3, 4*self.n[1] - 1), dtype=int)
        zidx1d = np.zeros((3, 4*self.n[2] - 1), dtype=int)
        for ii in range(3):
            xidx1d[ii, :] = np.r_[2*self.n[0]-1:0:-1, :2*self.n[0]]
            yidx1d[ii, :] = np.r_[2*self.n[1]-1:0:-1, :2*self.n[1]]
            zidx1d[ii, :] = np.r_[2*self.n[2]-1:0:-1, :2*self.n[2]]
        if orientation == 'x':
            coeffpoten_x = gen2_p_parz(self.l[1]/2, self.l[2]/2, self.l[0]/2,
                                       4*self.n[1], 4*self.n[2], 4*self.n[0])
            coeffpoten_x = np.transpose(coeffpoten_x, (2, 0, 1))
            coeffpoten_y = gen_p_per(self.l[2]/2, self.l[0]/2, self.l[1]/2,
                                     4*self.n[2], 4*self.n[0], 4*self.n[1])
            coeffpoten_y = np.transpose(coeffpoten_y, (1, 2, 0))
            coeffpoten_y_even = coeffpoten_y[::2, :, :]
            coeffpoten_y_odd = coeffpoten_y[1::2, :, :]
            coeffpoten_z = gen_p_per(self.l[1]/2, self.l[0]/2, self.l[2]/2,
                                     4*self.n[1], 4*self.n[0], 4*self.n[2])
            coeffpoten_z = np.transpose(coeffpoten_z, (1, 0, 2))
            coeffpoten_z_even = coeffpoten_z[:, :, ::2]
            coeffpoten_z_odd = coeffpoten_z[:, :, 1::2]
            xidx1d[0, :] = np.r_[self.n[0]-1:0:-1, :self.n[0]]
            xidx1d[1, :] = np.r_[2*self.n[0]-1:0:-1, 0, :2*self.n[0]]
            xidx1d[2, :] = np.r_[2*self.n[0]-1:0:-1, 0, :2*self.n[0]]
            # yidx1d[0, :] = np.r_[
            xidx1d[1, :self.n[0]-1] = np.r_[2*self.n[0]-2:0:-1, 0]
            xidx1d[2, :self.n[0]-1] = np.r_[2*self.n[0]-2:0:-1, 0]
            yidx1d[1, self.n[1]-1:] = np.r_[0, :2*self.n[1]-1]
            zidx1d[2, self.n[2]-1:] = np.r_[0, :2*self.n[2]-1]
        elif orientation == 'y':
            coeffpoten_x = gen_p_per(self.l[2], self.l[1], self.l[0],
                                     2*self.n[2], 2*self.n[1], 2*self.n[0])
            coeffpoten_x = np.transpose(coeffpoten_x, (2, 1, 0))
            coeffpoten_y = gen2_p_parz(self.l[2], self.l[0], self.l[1],
                                       2*self.n[2], 2*self.n[0], 2*self.n[1])
            coeffpoten_y = np.transpose(coeffpoten_y, (1, 2, 0))
            coeffpoten_z = gen_p_per(self.l[0], self.l[1], self.l[2],
                                     2*self.n[0], 2*self.n[1], 2*self.n[2])
            self.self_poten = coeffpoten_y[0, 0, 0]
            xidx1d[0, self.n[0]-1:] = np.r_[0, :self.n[0]-1]
            yidx1d[0, :self.n[1]-1] = np.r_[self.n[1]-2:0:-1, 0]
            yidx1d[2, :self.n[1]-1] = np.r_[self.n[1]-2:0:-1, 0]
            zidx1d[2, self.n[2]-1:] = np.r_[0, :self.n[2]-1]
        elif orientation == 'z':
            coeffpoten_x = gen_p_per(self.l[1], self.l[0], self.l[2],
                                     2*self.n[1], 2*self.n[0], 2*self.n[2])
            coeffpoten_x = np.transpose(coeffpoten_x, (1, 0, 2))
            coeffpoten_y = gen_p_per(self.l[0], self.l[1], self.l[2],
                                     2*self.n[2], 2*self.n[0], 2*self.n[1])
            coeffpoten_z = gen2_p_parz(self.l[0], self.l[1], self.l[2],
                                       2*self.n[0], 2*self.n[1], 2*self.n[2])
            self.self_poten = coeffpoten_z[0, 0, 0]
            xidx1d[0, self.n[0]-1:] = np.r_[0, :self.n[0]-1]
            yidx1d[1, self.n[1]-1:] = np.r_[0, :self.n[1]-1]
            zidx1d[0, :self.n[2]-1] = np.r_[self.n[2]-2:0:-1, 0]
            zidx1d[1, :self.n[2]-1] = np.r_[self.n[2]-2:0:-1, 0]
        else:
            raise RuntimeError('Incorrect orientation given!')
        p2p_trans_x = np.zeros((27, 2*self.n[0]-1, 2*self.n[1]-1, 2*self.n[2]-1),
                               dtype=np.complex128)
        self.p2p_trans_x = np.zeros((27, 2*self.n[0], 2*self.n[1], 2*self.n[2]),
                                    dtype=np.complex128)
        p2p_trans_y = np.zeros((27, 2*self.n[0]-1, 2*self.n[1]-1, 2*self.n[2]-1),
                               dtype=np.complex128)
        self.p2p_trans_y = np.zeros((27, 2*self.n[0], 2*self.n[1], 2*self.n[2]),
                                    dtype=np.complex128)
        p2p_trans_z = np.zeros((27, 2*self.n[0]-1, 2*self.n[1]-1, 2*self.n[2]-1),
                               dtype=np.complex128)
        self.p2p_trans_z = np.zeros((27, 2*self.n[0], 2*self.n[1], 2*self.n[2]),
                                    dtype=np.complex128)
        xidx = np.zeros((3, (2*self.n[0]-1)*(2*self.n[1]-1)*(2*self.n[2]-1)),
                        dtype=xidx1d.dtype)
        yidx = np.zeros((3, (2*self.n[0]-1)*(2*self.n[1]-1)*(2*self.n[2]-1)),
                        dtype=yidx1d.dtype)
        zidx = np.zeros((3, (2*self.n[0]-1)*(2*self.n[1]-1)*(2*self.n[2]-1)),
                        dtype=zidx1d.dtype)
        for coeffpoten, target_orientation, ito in \
            zip([coeffpoten_x, coeffpoten_y, coeffpoten_z],
                [p2p_trans_x, p2p_trans_y, p2p_trans_z],
                range(3)):
            for neighx in range(-1, 2):
                if neighx == -1:
                    ap = xidx1d[ito, self.n[0]]
                    xidxneigh = np.r_[ap:2*self.n[0]+ap-1]
                elif neighx == 0:
                    xidxneigh = xidx1d[ito, :]
                    if ito == 0:
                        if orientation == 'x':
                            xidxneigh = np.r_[self.n[0]-1:0:-1, :self.n[0]]
                        elif orientation == 'y':
                            xidxneigh = np.r_[2*self.n[0]-1:0:-1, 0, :2*self.n[0]]
                elif neighx == 1:
                    bp = xidx1d[ito, self.n[0]-2]
                    xidxneigh = np.r_[2*self.n[0]+bp-2:bp-1:-1]
                xidx[ito, :] = np.repeat(xidxneigh,
                                         (2*self.n[1]-1)*(2*self.n[2]-1))
                for neighy in range(-1, 2):
                    if neighy == -1:
                        ap = yidx1d[ito, self.n[1]]
                        yidxneigh = np.r_[ap:2*self.n[1]+ap-1]
                    elif neighy == 0:
                        yidxneigh = yidx1d[ito, :]
                    elif neighy == 1:
                        bp = yidx1d[ito, self.n[1]-2]
                        yidxneigh = np.r_[2*self.n[1]+bp-2:bp-1:-1]
                    yidxtemp = np.repeat(yidxneigh, 2*self.n[2] - 1)
                    yidx[ito, :] = np.tile(yidxtemp, 2*self.n[0] - 1)
                    for neighz in range(-1, 2):
                        if neighz == -1:
                            ap = zidx1d[ito, self.n[2]]
                            zidxneigh = np.r_[ap:2*self.n[2]+ap-1]
                        elif neighz == 0:
                            zidxneigh = zidx1d[ito, :]
                        elif neighz == 1:
                            bp = zidx1d[ito, self.n[2]-2]
                            zidxneigh = np.r_[2*self.n[2]+bp-2:bp-1:-1]
                        zidx[ito, :] = np.tile(zidxneigh, (2*self.n[0]-1) *
                                                          (2*self.n[1]-1))
                        neigh = 9*(neighx+1) + 3*(neighy+1) + neighz + 1
                        target_orientation[neigh, ...] = np.reshape(coeffpoten[
                            xidx[ito, :], yidx[ito, :], zidx[ito, :]],
                                (2*self.n[0]-1, 2*self.n[1]-1, 2*self.n[2]-1))
        for neigh in range(27):
            self.p2p_trans_x[neigh, ...] = \
                tp.t3multnonsym3(p2p_trans_x[neigh, ...],
                                 self.n[0], self.n[1], self.n[2])
            self.p2p_trans_y[neigh, ...] = \
                tp.t3multnonsym3(p2p_trans_y[neigh, ...],
                                 self.n[0], self.n[1], self.n[2])
            self.p2p_trans_z[neigh, ...] = \
                tp.t3multnonsym3(p2p_trans_z[neigh, ...],
                                 self.n[0], self.n[1], self.n[2])
        self.transx = p2p_trans_x[13, ...]
        self.transy = p2p_trans_y[13, ...]
        self.transz = p2p_trans_z[13, ...]
        #     print("trans:        {:.15e}".format(float(np.real(
        #         self.transx[self.n[0]-1,self.n[1]-1,self.n[2]-1]))))

    def p2p(self, px, py, pz, targetdatax, targetdatay, targetdataz):
        """Apply the capacitive near-field operator to all three target sets.

        Slab-sweep FFT convolution analogous to
        :meth:`leaf_induct.LeafInduct.p2pcpu`, but rectangular: the sources
        are this instance's panels (``self.data``) and the results are
        *accumulated* into the three target orientations at once.

        Source and target panel grids are staggered by half a cell on the
        normal axes, so each (source, target) pair is convolved on a COMMON
        HALF-UNIT LATTICE: whichever grid is flat along a given axis is
        upsampled with stride 2 when placed into / read out of the
        workspace. ``ss``/``st`` below are those per-axis strides; they are
        all 1 for the parallel blocks (t == s), which therefore keep the
        plain contiguous layout. The matching kernel index arithmetic lives
        in :meth:`p2pinit`'s ``gather1d``.

        Parameters
        ----------
        px, py, pz : LeafPoten
            The three target-orientation leaf objects.
        targetdatax, targetdatay, targetdataz : ndarray
            Output arrays (in the target objects' panel ordering) to which
            the potentials are added in place.

        Requires a prior :meth:`p2pinit` call to populate
        ``self.p2p_trans_*``.
        """
        s = {'x': 0, 'y': 1, 'z': 2}[self.orientation]
        sz = self.coeffpoten_sizes(px, py, pz)
        targets = (px, py, pz)
        outs = (targetdatax, targetdatay, targetdataz)
        trans = (self.p2p_trans_x, self.p2p_trans_y, self.p2p_trans_z)
        _strides = [self.lattice_strides(t) for t in range(3)]
        ss = [_strides[t][0] for t in range(3)]
        stt = [_strides[t][1] for t in range(3)]
        nin_s = [[ss[t][c]*int(self.n[c]) for c in range(3)]
                 for t in range(3)]
        nin_t = [[stt[t][c]*int(targets[t].n[c]) for c in range(3)]
                 for t in range(3)]
        n0, n1, n2 = (int(v) for v in self.n)

        # Workspace cache. The sweep needs a source and a target
        # ToeplitzCoeffPoten per (target orientation, slab size), and
        # building one is expensive: it allocates sz^3-class buffers AND
        # six FFTW_MEASURE plans. Rebuilding them every matvec made the
        # FFT near field get SLOWER with leaf size instead of amortising
        # (measured 18x/85x/192x/450x vs the sparse n2n at leaf 2/4/6/8),
        # which is backwards and was purely this allocation.
        #
        # behind/current/ahead must be three DISTINCT buffers at once, so
        # the source cache is keyed by a 3-slot rotation as well as by
        # (orientation, slab size) -- keying on size alone would alias
        # them together whenever consecutive slabs hold equal group
        # counts, which is the common case.
        cache = self.__dict__.setdefault('_p2pwsp', {})

        def wsp(role, t, ngroups, slot=0):
            key = (role, t, int(ngroups), slot)
            w = cache.get(key)
            if w is None:
                nin = nin_s[t] if role == 's' else nin_t[t]
                w = tp.ToeplitzCoeffPoten(nin, sz[t, :], int(ngroups))
                cache[key] = w
            return w

        def newslab(t, ngroups, slot):
            # the source side is written strided and only partially, so
            # the zero padding has to be restored on every reuse
            w = wsp('s', t, max(ngroups, 1), slot)
            w.a[...] = 0
            w.b[...] = 0
            w.c[...] = 0
            return w

        ahead = [newslab(t, 1, 2) for t in range(3)]   # zeroed placeholder
        current = [None, None, None]
        flat = np.empty((n0*n1*n2,), dtype=np.complex128)
        for countx in range(-1, int(self.ng[0])):
            behind = current
            current = ahead
            slot = (countx + 1) % 3
            if countx < self.ng[0] - 1:
                sizeslab = int(self.slabidx0[countx+2] -
                               self.slabidx0[countx+1])
                ahead = [newslab(t, sizeslab, slot) for t in range(3)]
                # FUSED SCATTER via cached index packs (2026-08-08,
                # the leaf_induct._slab_pack treatment ported): the
                # per-group python loop of strided assignments was
                # ~half the near-field cost once the lean tree put
                # this path on every LpPR iteration (545 ms of a
                # 1386 ms matvec at 320^2). The pack maps each source
                # element straight to its strided position in the
                # workspace: one fancy-index write per (slab, t),
                # bit-identical placement by construction.
                packs = self.__dict__.setdefault('_p2p_spacks', {})
                pk = packs.get(countx + 1)
                if pk is None:
                    pk = []
                    for t in range(3):
                        e = ss[t]
                        ash = None
                        fps = []
                        srcs = []
                        for countg in range(sizeslab):
                            group = self.slabidx[
                                countg + self.slabidx0[countx+1]]
                            fidx = np.arange(self.idx0[group],
                                             self.idx0[group+1])
                            ii = np.asarray(self.idx[fidx])
                            i0 = ii//(n1*n2)
                            i1 = (ii//n2) % n1
                            i2 = ii % n2
                            ash = ahead[t].a.shape
                            fp = np.ravel_multi_index(
                                (np.full(ii.size, countg),
                                 e[0]*i0, e[1]*i1, e[2]*i2), ash)
                            fps.append(fp)
                            srcs.append(fidx)
                        pk.append((np.concatenate(fps)
                                   if fps else np.zeros(0, np.int64),
                                   np.concatenate(srcs)
                                   if srcs else np.zeros(0, np.int64),
                                   ash))
                    packs[countx + 1] = pk
                for t in range(3):
                    fp, srcs, ash = pk[t]
                    if ash is not None and ahead[t].a.shape != ash:
                        raise RuntimeError("p2p pack shape drift")
                    ahead[t].a.ravel()[fp] = self.data[srcs]
                for t in range(3):
                    w = ahead[t]
                    w.fftab()
                    w.b[:, :, :nin_s[t][1], :] = w.a
                    w.fftbc()
                    w.c[:, :, :, :nin_s[t][2]] = w.b
                    w.fftcc()
            else:
                ahead = [newslab(t, 1, slot) for t in range(3)]
            if countx < 0 or current[0] is None or \
                    np.shape(current[0].c)[0] == 0:
                continue
            selfslabidx = self.slabidx[self.slabidx0[countx]:
                                       self.slabidx0[countx+1]]
            for t in range(3):
                p = targets[t]
                tslabidx = p.slabidx[p.slabidx0[countx]:
                                     p.slabidx0[countx+1]]
                if int(np.size(tslabidx)) == 0:
                    continue
                # target side needs no zeroing: c, then b, then a are
                # each fully overwritten on the way back out
                tw = wsp('t', t, int(np.size(tslabidx)))
                # OSLABIDX is never read by the kernel -- it only sizes
                # OUTMAT's group axis, which the kernel indexes as
                # OREVSLABIDX(group)+1, so it must be the TARGET slab's
                # group list (passing a scalar count corrupted the heap).
                tw.c[...] = mp_fortran.p2p(
                    behind[t].c.T, current[t].c.T, ahead[t].c.T,
                    selfslabidx, tslabidx, countx, self.neighbors.T,
                    self.xidx, trans[t].T, self.revslabidx,
                    p.revslabidx).T
                tw.ifftcc()
                tw.b[...] = tw.c[..., :nin_t[t][2]]
                tw.ifftcb()
                tw.a[...] = tw.b[..., :nin_t[t][1], :]
                tw.ifftba()
                # FUSED GATHER, mirror of the scatter pack above: map
                # each target element to its strided position in the
                # inverse-transformed workspace; one fancy-index read
                # + add per (slab, t). Each target element appears
                # exactly once, so += on slices is collision-free.
                tpacks = self.__dict__.setdefault('_p2p_tpacks', {})
                tp_ = tpacks.get((countx, t))
                if tp_ is None:
                    e = stt[t]
                    pn = [int(v) for v in p.n]
                    fps = []
                    dsts = []
                    for countg in range(int(np.size(tslabidx))):
                        group = tslabidx[countg]
                        fidx = np.arange(p.idx0[group],
                                         p.idx0[group+1])
                        ii = np.asarray(p.idx[fidx])
                        i0 = ii//(pn[1]*pn[2])
                        i1 = (ii//pn[2]) % pn[1]
                        i2 = ii % pn[2]
                        fp = np.ravel_multi_index(
                            (np.full(ii.size, countg),
                             e[0]*i0, e[1]*i1, e[2]*i2), tw.a.shape)
                        fps.append(fp)
                        dsts.append(fidx)
                    tp_ = (np.concatenate(fps)
                           if fps else np.zeros(0, np.int64),
                           np.concatenate(dsts)
                           if dsts else np.zeros(0, np.int64),
                           tw.a.shape)
                    tpacks[(countx, t)] = tp_
                fp, dsts, ash = tp_
                if tw.a.shape != ash:
                    raise RuntimeError("p2p target pack shape drift")
                outs[t][dsts] += tw.a.ravel()[fp]

    def p2pinit3(self, px, py, pz):
        """Build sparse panel-to-panel coupling matrices (explicit path).

        Alternative to the FFT approach of :meth:`p2pinit`/:meth:`p2p`:
        assembles the near-field coupling from this source-orientation panel
        set to each target set as three explicit sparse matrices
        (``scipy`` LIL, returned as CSR), suitable for direct matvec or for
        composition with the node-to-panel projections in
        ``Tree.__init__``. The closed-form kernel tables use the same
        :func:`gen2_p_parz`/:func:`gen_p_per` split as :meth:`p2pinit`.

        Entry ``[i, j]`` of each returned matrix is the coefficient from
        *source* panel ``i`` (this instance) to *target* panel ``j``, so the
        potential on the targets is ``P.T @ q_source``.

        Table index semantics: :func:`gen2_p_parz` (parallel panels, both
        grids in registry along every axis) is indexed by plain integer
        separations. :func:`gen_p_per` (perpendicular panels) is indexed by
        integer separations along the transverse axis but *half-offset*
        separations along the two normal axes: table index ``k`` on those
        axes corresponds to a centre distance of ``(k + 1/2)`` grid units,
        because a panel is flat along its own normal while the other panel
        extends half a cell to either side of the lattice point. The
        per-axis index arithmetic below converts the (plane-grid vs.
        half-cell-grid) panel indices to these table indices; the neighbour
        offset enters as ``+ delta * pn[c]`` on the target index (the
        ``neighbors`` convention is "group at +delta relative to self").

        Parameters
        ----------
        px, py, pz : LeafPoten
            Target-orientation leaf objects.

        Returns
        -------
        tuple of scipy.sparse.csr_matrix
            ``(P_sx, P_sy, P_sz)`` coupling this source orientation to the
            x-, y- and z-oriented target panels.

        Notes
        -----
        This is a pure-numpy reimplementation. The original Fortran kernels
        ``mp_fortran.coeffp``/``coeffpsingle`` are no longer used: they
        compared the 0-based ``so`` argument against 1-based orientation
        numbers (mis-staggering every block, so even the same-orientation
        blocks came out asymmetric) and computed cross-orientation distances
        by subtracting raw indices of grids with different per-axis units.
        Both defects are fixed here; the same loop serves the single-level
        (one group, self-neighbour only) and multilevel (27-neighbour)
        cases.
        """
        n1 = np.zeros((3,), dtype=np.int_)
        n2 = np.zeros((3,), dtype=np.int_)
        m2 = np.zeros((3,), dtype=np.int_)
        l2 = np.zeros((3,), dtype=self.l.dtype)
        l = np.zeros((3,), dtype=self.l.dtype)
        for ii in range(3):
            n1[ii] = self.n[ii]
            n2[ii] = self.m[ii] + self.n[ii]
            m2[ii] = 2*self.m[ii]
            l2[ii] = self.l[ii]/2
            l[ii] = self.l[ii]
        p2p_trans_x = lil_matrix((np.size(self.idx), np.size(px.idx)),
                                 dtype=np.float64)
        p2p_trans_y = lil_matrix((np.size(self.idx), np.size(py.idx)),
                                 dtype=np.float64)
        p2p_trans_z = lil_matrix((np.size(self.idx), np.size(pz.idx)),
                                 dtype=np.float64)
        # coeffpoten_x holds all possible partial coefficients of potential for
        # x-directed target panels for a 3x3 cluster of groups. The projection
        # runs from all points in the 3x3 cluster to all points in the center
        # group.
        if self.orientation == 'x':
            so = 0
            coeffpoten_x = gen2_p_parz(l[1], l[2], l[0], n2[1], n2[2], n2[0])
            coeffpoten_x = np.transpose(coeffpoten_x, (2, 0, 1))
            coeffpoten_y = gen_p_per(l[2], l2[0], l[1], m2[2], 2*m2[0], m2[1])
            coeffpoten_y = np.transpose(coeffpoten_y, (1, 2, 0))
            coeffpoten_z = gen_p_per(l[1], l2[0], l[2], m2[1], 2*m2[0], m2[2])
            coeffpoten_z = np.transpose(coeffpoten_z, (1, 0, 2))
        elif self.orientation == 'y':
            so = 1
            coeffpoten_x = gen_p_per(l[2], l2[1], l[0], m2[2], 2*m2[1], m2[0])
            coeffpoten_x = np.transpose(coeffpoten_x, (2, 1, 0))
            coeffpoten_y = gen2_p_parz(l[2], l[0], l[1], n2[2], n2[0], n2[1])
            coeffpoten_y = np.transpose(coeffpoten_y, (1, 2, 0))
            coeffpoten_z = gen_p_per(l[0], l2[1], l[2], m2[0], 2*m2[1], m2[2])
        elif self.orientation == 'z':
            so = 2
            coeffpoten_x = gen_p_per(l[1], l[0], l2[2], m2[1], m2[0], 2*m2[2])
            coeffpoten_x = np.transpose(coeffpoten_x, (1, 0, 2))
            coeffpoten_y = gen_p_per(l[0], l[1], l2[2], m2[0], m2[1], 2*m2[2])
            coeffpoten_z = gen2_p_parz(l[0], l[1], l[2], n2[0], n2[1], n2[2])
        # Cell scheme: panels are WHOLE FACES, so the parallel blocks are
        # unchanged (same-orientation panels stay in registry on every
        # axis, with panel size equal to stride -- exactly gen2_p_parz's
        # domain), but the cross blocks need whole faces at half-cell
        # offsets, which gen_p_per cannot express. p_per_table supplies
        # those; it is indexed by SIGNED separation because a
        # perpendicular coupling is not symmetric under reflecting one
        # axis, unlike the parallel one.
        tgts = (px, py, pz)
        celloff = []
        cross = list((coeffpoten_x, coeffpoten_y, coeffpoten_z))
        for tt in range(3):
            off = np.array([int(self.n[c]) + int(tgts[tt].n[c]) + 2
                            for c in range(3)], dtype=np.int_)
            celloff.append(off)
            if tt != so:
                cross[tt] = p_per_table(l, so, tt, off)
        coeffpoten_x, coeffpoten_y, coeffpoten_z = cross

        def boxcoords(idx, n):
            # compressed flat per-box index -> (x, y, z) grid indices
            return (idx // (n[1]*n[2]), (idx // n[2]) % n[1], idx % n[2])

        tables = (coeffpoten_x, coeffpoten_y, coeffpoten_z)
        targets = (px, py, pz)
        targetmats = (p2p_trans_x, p2p_trans_y, p2p_trans_z)
        scomp = boxcoords(self.idx, self.n)
        tcomp = [boxcoords(p.idx, p.n) for p in targets]
        for group in range(np.size(self.idx0) - 1):
            if self.idx0[group+1] <= self.idx0[group]:
                continue
            pidx = np.s_[self.idx0[group]:self.idx0[group+1]]
            scoord = [c[pidx][:, None] for c in scomp]
            for neigh in range(27):
                neighgroup = self.neighbors[neigh, group]
                if neighgroup < 0:
                    continue
                delta = (neigh//9 - 1, (neigh//3) % 3 - 1, neigh % 3 - 1)
                for to in range(3):
                    p = targets[to]
                    if p.idx0[neighgroup+1] <= p.idx0[neighgroup]:
                        continue
                    didx = np.s_[p.idx0[neighgroup]:p.idx0[neighgroup+1]]
                    tabidx = []
                    for c in range(3):
                        T = tcomp[to][c][didx][None, :] + \
                            delta[c]*int(p.n[c])
                        S = scoord[c]
                        # parallel: |sep|; cross: signed sep shifted
                        # into the table
                        if to == so:
                            tabidx.append(np.abs(T - S))
                        else:
                            tabidx.append(T - S + celloff[to][c])
                    targetmats[to][pidx, didx] = \
                        tables[to][tabidx[0], tabidx[1], tabidx[2]]
        return (p2p_trans_x.tocsr(), p2p_trans_y.tocsr(), p2p_trans_z.tocsr())


