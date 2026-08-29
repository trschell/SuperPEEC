# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Octree level classes for the fast multipole method (FMM).

Each level of the FMM tree is one of:

* :class:`Level`     -- shared base (geometry, expansion order, tree links);
* :class:`LeafLevel` -- the finest level, where the sources/filaments live;
  owns the particle<->expansion operators P2M (:meth:`~LeafLevel.p2m`) and L2P
  (:meth:`~LeafLevel.l2p`), the near-field P2P being supplied by subclasses;
* :class:`MidLevel`  -- intermediate levels; owns the translation operators
  M2M (up), L2L (down) and the M2L far-field step over the 189-box
  interaction list;
* :class:`TopLevel`  -- the coarsest level, where the M2L between all boxes is
  translation-invariant and is applied by FFT (Toeplitz).

Expansions are in spherical harmonics truncated at order ``nmax``, giving
``nnmax=(nmax+1)**2`` complex coefficients per box; the harmonics and the
A_n^m normalization come from :mod:`special`. Split out of multipole.py.
"""
from multipole_common import *  # noqa: F401,F403  shared imports/guards
import os as _os

# fp32 campaign PHASE 4 (2026-08-28): the LEAF GATHER BUFFERS in
# single precision. These are the group-contiguous copies of the P2M
# and L2P operators, built lazily on the FIRST MATVEC -- which is why
# docs/memory_census_r4.md missed them: it walks the BUILT solver,
# before any matvec exists. Measured at R4 they are the single largest
# store in the whole run, 10.09 GB in six complex128 arrays (e/f/g x
# _mfil_g/_ynmr_g, 25 harmonics per filament), taking the true peak
# from the census's 9.45 GB to 18.15.
#
# THE DYNAMIC-RANGE TRAP -- the same one validate_mid_fp32 documents
# for the mid tables, and the reason a plain .astype(complex64) here
# would be WRONG. These tables carry r**n in SI metres, and ynmr also
# carries m0 = mu0/(4 pi) * l**2. Headroom of the smallest non-zero
# ynmr entry above float32's smallest normal, computed per rung:
#
#     demo 0.5 mm   12.4 decades      R4 0.0625 mm   7.6 decades
#     R3   0.125 mm  8.8 decades      R6 0.02   mm   4.6 decades
#     ... and at a 1 um pitch, -3.2 decades: it UNDERFLOWS.
#
# Underflow here is silent, and would zero the high-order harmonics at
# fine pitch -- the class of bug that survives every residual check
# because the far field merely gets quietly less accurate. So the
# magnitude is factored into an fp64 scalar and only the NORMALISED
# table is stored single. The dot is linear in the table, so
# scale*dot(table/scale, x) is the same contraction with fp32 rounding
# on entries that now span 1.0 down to ~1e-15 at R4 -- scale-free by
# construction, at any pitch.
#
# SPPEEC_LEAF_FP64=1 restores fp64 storage for A/B measurement.
_LEAF_FP64 = _os.environ.get('SPPEEC_LEAF_FP64') == '1'


def _gather_single(src, idx, axis, chunk_elems=1 << 22):
    """Gather ``src`` along ``axis`` by ``idx``, stored complex64 with
    an fp64 scale. Returns ``(buffer, scale)``.

    Built CHUNKWISE and straight into the single-precision output. The
    source tables are tiny -- (nnmax, cells-per-leaf-box), a few tens
    of kB -- and it is the per-filament gather that is gigabytes, so
    materialising the fp64 gather first (and then a scaled copy of it)
    would spend 3x the final buffer in transients and hand most of the
    saving straight back. The scale is taken over the FULL source,
    which bounds the gathered subset by construction and needs no
    extra pass.

    Falls back to a plain fp64 gather (scale 1.0) under
    SPPEEC_LEAF_FP64, and for an all-zero or non-finite table, where
    there is no magnitude to factor out.
    """
    idx = np.asarray(idx)
    if _LEAF_FP64:
        return (np.ascontiguousarray(src[:, idx]) if axis == 1
                else np.ascontiguousarray(src[idx, :])), 1.0
    scale = float(np.abs(src).max())
    if not (scale > 0.0) or not np.isfinite(scale):
        return (np.ascontiguousarray(src[:, idx]) if axis == 1
                else np.ascontiguousarray(src[idx, :])), 1.0
    n = idx.size
    out = np.empty((src.shape[0], n) if axis == 1
                   else (n, src.shape[1]), dtype=np.complex64)
    # size the chunk by ELEMENTS, not rows: each gathered row carries
    # nnmax harmonics, so a chunk counted in rows silently builds an
    # fp64 temporary nnmax times bigger than intended (the first cut
    # here peaked at 3.8x the final buffer, caught by the gate)
    per = max(1, src.shape[0] if axis == 1 else src.shape[1])
    chunk = max(1, int(chunk_elems)//per)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        if axis == 1:
            out[:, a:b] = src[:, idx[a:b]]/scale
        else:
            out[a:b, :] = src[idx[a:b], :]/scale
    return out, scale


# GPU path for the top-level M2L (and nothing else yet): opt-in via
# SPPEEC_GPU=1, because not every machine has one and the CPU results
# are the bit-anchored reference. Falls back to the CPU path loudly on
# the first failure, then silently thereafter.
# DEFAULT-ON (user decision 2026-08-12): '0' opts out, unset/'auto'
# probes the hardware, '1' forces (loud on failure). Answer agreement
# measured to 8 digits at convergence -- swamped by engineering rtol.
def _gpu_probe():
    if _os.environ.get('SPPEEC_GPU', 'auto') == '0':
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return _os.environ.get('SPPEEC_GPU') == '1'


_GPU = _gpu_probe()
from special import *  # noqa: F401,F403
from greens import *  # noqa: F401,F403
from stencils import AXIS_OF, PANEL_ORIENTATIONS
import stencils as st


class Level:
    """Base class for a single level of the FMM octree.

    Holds the geometry and expansion parameters common to every level and the
    slots the subclasses populate: the expansion coefficients (`data`), the
    compressed group indexing (`idx`/`idx0`), the up/down tree links
    (`above`/`below`), and the neighbour lists. ``nnmax=(nmax+1)**2`` is the
    number of spherical-harmonic coefficients per expansion.
    """
    def __init__(self, n, l, nmax):
        """Initialize level geometry and expansion order.

        Parameters
        ----------
        n : ndarray of int
            Grid dimensions at this level (boxes, or filaments per box at the
            leaf) along x, y, z.
        l : sequence of float
            Physical cell size along x, y, z (metres).
        nmax : int or None
            Maximum multipole/local expansion order; sets
            ``nnmax=(nmax+1)**2``. ``None`` leaves the expansion unsized.
        """
        n = n.astype(int)
        self.n = n
        self.m = n  # for single-level, the size of the nodal structure
        # must be preserved for the P2P operations
        self.l = l
        self.nmax = nmax
        if self.nmax is not None:
            self.nnmax = (nmax + 1)**2
        else:
            self.nnmax = None
        self.data = None
        self.idx = None
        self.idx0 = None
        self.below = None
        self.above = None
        self.neighbors = None
        self.struc_e = None
        self.struc_f = None
        self.struc_g = None


class LeafLevel(Level):
    """The finest level of the tree, where the sources/filaments live.

    Adds the leaf-only structures to :class:`Level`: the number of boxes per
    axis (`ng`), the per-orientation occupancy, the slab/neighbour indexing
    used by the near-field sweep, and the P2M/L2P spherical-harmonic operators
    built by :meth:`leafinit`. The near-field P2P operator itself is provided
    by concrete subclasses (:class:`~leaf_induct.LeafInduct`,
    :class:`~leaf_poten.LeafPoten`).

    `orientation` selects the element type: the inductive filament directions
    ``'e'``, ``'f'``, ``'g'`` or the potential-panel directions ``'x'``,
    ``'y'``, ``'z'`` --- it fixes the intra-box positions and the source
    normalization used in :meth:`leafinit`.
    """
    def __init__(self, n, ng, l, nmax, orientation):
        """Initialize the leaf level and, if fully specified, its operators.

        Parameters
        ----------
        n : ndarray of int
            Filaments/panels per box along x, y, z.
        ng : ndarray of int
            Number of leaf boxes along x, y, z.
        l : sequence of float
            Physical cell size along x, y, z (metres).
        nmax : int or None
            Maximum expansion order.
        orientation : {'e','f','g','x','y','z'} or None
            Element type/direction. When both `orientation` and `nmax` are
            given, :meth:`leafinit` is called to build the P2M/L2P operators.
        """
        Level.__init__(self, n, l, nmax)
        self.ng = ng
        self.ng = self.ng.astype(int)
        self.struc = None
        self.slabidx = None
        self.slabidx0 = None
        self.revslabidx = None
        self.xidx = None
        self.yidx = None
        self.zidx = None
        self.orientation = orientation
        #     if member.ndim != 1:
        #         raise TypeError('Leaf-level data should be 1-D.')
        if orientation is not None and nmax is not None:
            self.leafinit()

    def getslabidx(self):
        """Group the leaf boxes into x-slabs for the near-field sweep.

        Builds the index arrays the P2P drivers use to walk the domain one
        x-slab at a time: ``slabidx`` (occupied group ids ordered by x-slab),
        ``slabidx0`` (CSR-style start offset of each x-slab), and
        ``revslabidx`` (group id -> position within its slab, ``-1`` if the
        group is empty). Only non-empty groups are retained.
        """
        gsize = np.size(self.idx0) - 1
        self.slabidx = np.empty((gsize,), dtype=int)
        self.revslabidx = np.empty((gsize,), dtype=int)
        self.revslabidx[:] = -1
        self.slabidx0 = np.zeros((int(self.ng[0])+1,), dtype=int)
        for countx in range(int(self.ng[0])):
            argx = np.argwhere(self.xidx == countx).T[0]
            sargx = np.size(argx)
            sarg = 0
            arg = np.ndarray((0,), dtype=int)
            for i in range(sargx):
                if self.idx0[argx[i]+1] > self.idx0[argx[i]]:
                    arg = np.append(arg, argx[i])
                    sarg += 1
            self.slabidx0[countx+1] = self.slabidx0[countx] + sargx
            self.slabidx[self.slabidx0[countx]:self.slabidx0[countx+1]] = argx
            self.revslabidx[argx] = np.r_[:sargx]
        self.slabidx = self.slabidx[:self.slabidx0[countx+1]]

    def leafinit(self):
        """Precompute the leaf P2M and L2P spherical-harmonic operators.

        Builds two coefficient tables from the fixed intra-box source
        positions:

        * ``self.mfil`` (``(nnmax, prod(n))``) --- the **P2M** operator:
          column *i* holds ``r_i**n * Y_n^{-m}(theta_i, phi_i)`` for each
          harmonic ``(n, m)``, so :meth:`p2m` forms a box's multipole moments
          by contracting ``mfil`` with the source amplitudes.
        * ``self.ynmr`` (``(prod(n), nnmax)`` after transpose) --- the **L2P**
          operator: ``m0 * r_i**n * Y_n^m``, so :meth:`l2p` evaluates the
          box's local expansion back at each source point.

        The prefactor ``m0`` is orientation-dependent: the inductive filaments
        (``'e'``/``'f'``/``'g'``) use ``mu0/(4*pi)*l_axis**2`` (magnetic), the
        potential panels (``'x'``/``'y'``/``'z'``) use ``1/(4*pi*eps0)``
        (electric).

        ELEMENT POSITIONS -- THE STAGGER
        --------------------------------
        Every axis carries two lattices of ``n[i]`` points, offset from one
        another by exactly half a cell (verified bitwise for all n; the
        absolute values alternate between integer and half-integer with the
        parity of ``n``, which is a red herring -- only the relative offset
        is physical):

        * ``NODE``   -- the cell-corner planes. Filaments lie ON these
          transversely (a filament runs along a cell EDGE); panels lie on
          them along their normal (a panel lies IN a cell-face plane).
        * ``CENTRE`` -- half a cell further along, the cell interiors.
          Filaments use this along their OWN axis (they span one cell
          there); panels use it transversely, and because the panel lattice
          is refined 2x transversely with ``l`` halved, that lands on the
          centre of each quarter-face -- the piece of surface belonging to
          one corner node.

        which collapses the six cases to one rule::

            use CENTRE on axis i  iff  (i is the element's own axis)
                                        XOR (this is a panel)

        Filaments and panels take OPPOSITE assignments, which is a direct
        consequence of corner nodes: charge sits on faces while current runs
        along edges, so the two are half a cell apart in every direction.

        Under the planned cell-centred rewrite (``docs/cell_centred_scope.md``)
        the XOR disappears: with nodes at cell centres, an x-filament spans
        cell centres i..i+1 and an x-normal panel is the face between the same
        two cells, so BOTH sit at (boundary in x, centre in y, centre in z).
        Filaments and same-normal panels become co-located, the 2x transverse
        panel refinement goes away, and this rule becomes simply "the offset
        family on the element's own axis, the centred family elsewhere".
        """
        mu0 = 4*np.pi*1e-7
        c0 = 299792458
        eps0 = 1/(mu0 * c0**2)
        if self.orientation not in AXIS_OF:
            print("incorrect orientation given!")
            return
        axis = AXIS_OF[self.orientation]
        panel = self.orientation in PANEL_ORIENTATIONS
        node = [np.arange(-(self.n[i]-1)/2-0.5, (self.n[i]-1)/2+0.5)
                for i in range(3)]
        centre = [np.arange(-(self.n[i]-1)/2, (self.n[i]-1)/2+1)
                  for i in range(3)]
        # Under 'cell' the XOR disappears: an x-filament spans cell
        # centres i-1..i and an x-normal panel is the face between those
        # same two cells, so both take the offset family on their own
        # axis and the centred family elsewhere -- they are co-located.
        # This USED TO SAY False, contradicting the sentence above and
        # putting every panel a half cell off ITS OWN axis and a half
        # cell off on the two transverse axes -- so px and py came out
        # displaced by (+1/2,-1/2,0) cells instead of (-1/2,+1/2,0), a
        # FULL CELL of error on every cross-orientation panel pair. The
        # inductive path hid it (orthogonal partial inductances vanish,
        # so e/f/g never couple and each sweep tolerates a constant
        # translation); the capacitive path runs px/py/pz through ONE
        # shared sweep, where the relative offset is real, and paid 6.3%
        # far-field error. p2pinit3, node2panel ("a cell-scheme panel at
        # local i is the face between cells i-1 and i") and
        # validate_p2pinit3's own centres() all use the offset family --
        # leafinit was the odd one out. See validate_leafinit_geometry.py.
        fam = [centre[i] if ((i == axis) != True) else node[i]
               for i in range(3)]
        grid = np.meshgrid(fam[0], fam[1], fam[2], indexing='ij')
        posfilx = self.l[0]*grid[0].ravel()
        posfily = self.l[1]*grid[1].ravel()
        posfilz = self.l[2]*grid[2].ravel()
        self.mfil = np.zeros((self.nnmax, np.prod(self.n)),
                             dtype=np.complex128)
        self.ynmr = np.zeros((self.nnmax, np.prod(self.n)),
                             dtype=np.complex128)
        if panel:
            m0 = 1/(4*np.pi * eps0)
        else:
            m0 = mu0/(4*np.pi)*self.l[axis]**2
        rfil = np.sqrt(posfilx**2 + posfily**2 + posfilz**2)
        # An element can sit EXACTLY at the box expansion centre --
        # possible only with MIXED leaf parities (both transverse axes
        # odd, own axis even), which cubic same-count leaves can never
        # produce; per-axis leaves (anisotropic cells, 2026-08-07) can:
        # (5,5,2) puts one z-filament at r = 0 and arccos(0/0) NaN'd
        # its whole leaf. The limit is well defined -- r^n Y_n^m -> 0
        # for n > 0 and Y_0^0 for n = 0 -- so pin theta/phi to 0 there;
        # the r = 0 factor below does the rest (0**0 == 1 in numpy).
        at0 = rfil == 0.0
        with np.errstate(invalid='ignore'):
            thetafil = np.pi - np.arccos(
                np.where(at0, 1.0, posfilz)/np.where(at0, 1.0, rfil))
        thetafil[at0] = 0.0
        phifil = np.arctan2(posfily, posfilx)
        for n in range(self.nmax+1):
            r = rfil**n
            for m in range(-n, n+1):
                idx = n**2 + n + m
                self.mfil[idx, :] = r*sph_harm_of_cos(
                    n, -m, thetafil, phifil)
                self.ynmr[idx, :] = m0*r*sph_harm_of_cos(
                    n, m, thetafil, phifil)
        self.ynmr = np.transpose(self.ynmr)

    def p2m(self, accumulate=False):
        """Particle-to-multipole: form each box's multipole expansion.

        For every non-empty leaf box, contracts the P2M operator ``mfil`` with
        the box's source amplitudes (``self.data``) to produce its multipole
        moments, written into the parent level's coefficient array
        (``self.above.data``). This is the first (upward) step of the FMM.

        With ``accumulate=True`` the parent array is NOT reallocated and the
        moments are added to whatever is already there. This is how the three
        panel leaves (px/py/pz), which share one parent level and one Coulomb
        kernel, combine into a single multipole sweep: the first leaf runs a
        plain ``p2m()`` (fresh zeroed parent), the other two accumulate.
        (The inductive leaves e/f/g must NOT be combined this way -- each runs
        its own full sweep in traverseRL.)
        """
        msize = np.size(self.idx0) - 1
        if not accumulate:
            self.above.data = np.zeros((msize, (self.nmax+1)**2),
                                       dtype=np.complex128)
        # The element ranges idx0[g]:idx0[g+1] are CONTIGUOUS, so the
        # old ``np.r_`` fancy index was an arange in disguise -- built
        # 24096 times per matvec on square_coil (~7% of the sweep). The
        # operator columns are gathered ONCE into a group-contiguous
        # copy so the loop body is a pure slice + gemv, bit-identical
        # to the old arithmetic (same per-group columns, same dot).
        if getattr(self, '_mfil_g', None) is None:
            self._mfil_g, self._mfil_s = _gather_single(
                self.mfil, self.idx, 1)
        i0 = self.idx0
        sc = self._mfil_s
        for group in range(msize):
            a, b = i0[group], i0[group+1]
            if b > a:
                self.above.data[group, :] += sc*np.dot(
                    self._mfil_g[:, a:b], self.data[a:b])

    def l2p(self):
        """Local-to-particle: evaluate each box's local expansion at sources.

        For every non-empty leaf box, contracts the L2P operator ``ynmr`` with
        the box's local-expansion coefficients (``self.above.data``) and adds
        the result to the source amplitudes ``self.data``. This is the final
        (downward) step of the FMM far-field contribution.
        """
        msize = np.size(self.idx0) - 1
        # Same slice + pre-gathered-operator rewrite as p2m -- see the
        # comment there. Bit-identical.
        if getattr(self, '_ynmr_g', None) is None:
            self._ynmr_g, self._ynmr_s = _gather_single(
                self.ynmr, self.idx, 0)
        i0 = self.idx0
        sc = self._ynmr_s
        for group in range(msize):
            a, b = i0[group], i0[group+1]
            if b > a:
                self.data[a:b] += sc*np.dot(
                    self._ynmr_g[a:b, :], self.above.data[group, :])

    def traverse(self):
        """Apply one full FMM matvec over the whole tree.

        Runs the complete fast-multipole sweep starting from this leaf level:
        the upward pass (:meth:`p2m` then :meth:`~MidLevel.m2m` up to the top),
        the multipole-to-local translations (:meth:`~TopLevel.m2l` at the top
        and :meth:`~MidLevel.m2l` on the way down), the downward
        :meth:`~MidLevel.l2l` pass, and finally the near-field
        :meth:`p2p` and :meth:`l2p`, leaving the interaction result in
        ``self.data``. (Note: the single-level path uses the P2P/M2L drivers
        directly rather than this method.)
        """
        self.p2m()
        current = self
        while not isinstance(current.above, TopLevel):
            current = current.above
            current.m2m()
        current.above.m2l()
        while not isinstance(current, LeafLevel):
            current.m2l()
            current.l2l()
            current = current.below
        self.p2p()
        self.l2p()


class MidLevel(Level):
    """An intermediate (non-leaf, non-root) level of the FMM tree.

    Owns the three translation operators that move expansions between this
    level and its neighbours: M2M (:meth:`m2m`, children -> parent), L2L
    (:meth:`l2l`, parent -> children), and the far-field M2L (:meth:`m2l`)
    over the interaction list (the parent-neighbour stencil at child
    resolution minus the 27 near boxes already handled at the parent
    level: 6x6x6 - 27 = 189 for the 2x2x2 split, 6x6x3 - 27 = 81 when
    one axis is unsplit, per-axis in general).
    The translation matrices are precomputed once in :meth:`midinit` and
    :meth:`midm2linit`.
    """
    def __init__(self, n, l, nmax):
        """Initialize the level and precompute its M2M/L2L operators.

        Calls :meth:`midinit` to build the translation matrices; the M2L
        interaction data (:meth:`midm2linit`) is set up separately once the
        neighbour lists are known.
        """
        Level.__init__(self, n, l, nmax)
        self.xidx = None
        self.yidx = None
        self.zidx = None
        self.midinit()

    def midm2linit(self):
        """Precompute the M2L interaction list and transfer matrices.

        PER-AXIS in the child count ``self.n`` (= ``nmidlev``): the
        historical 2x2x2 case gives the familiar 8 parities and the
        6x6x6 - 27 = 189-entry interaction list; an UNSPLIT axis
        (``nmidlev = 1``, a flat geometry's mid level) gives e.g.
        ``[2,2,1]`` -> 4 parities, 6x6x3 - 27 = 81 entries. The slot
        pitch on every axis is the CHILD box length -- on an unsplit
        axis the child length equals the parent's and there is one
        slot per parent, which is what makes the transfer distances
        right there (the old hard-coded code placed children at
        half-parent spacing on every axis and mis-computed the far
        field by ~4% the moment an axis was unsplit; measured
        2026-08-06 on circular_coil).

        PARITY CONVENTION: mixed radix, ``p = (x*n1 + y)*n2 + z`` --
        exactly the ``iterator`` value the depth-first scan stores in
        ``self.idx`` (tree.py). The old code read parities as
        ``4x + 2y + z``, identical for [2,2,2] and silently scrambled
        for anything else.

        Builds, for this level:

        * ``self.farneighbors`` (``(n_groups, nfar)``) --- for each box,
          the ids of the boxes in its interaction list (well-separated
          at this level but not at the parent's): the parent-neighbour
          stencil at child resolution minus the 27-box child near
          window, indexed by child parity.
        * ``self.transfer`` (``(npar, nfar, nnmax, nnmax)``) --- the M2L
          translation matrix for each child parity and relative offset,
          mapping a source box's multipole coefficients to this box's
          local coefficients. Uses the Greengard-Rokhlin M2L formula
          with the A_n^m coefficients (:func:`special.a_nm`) and the
          spherical harmonic ``Y_{j+n}^{k-m}`` divided by
          ``r**(j+n+1)``.

        Requires the neighbour lists (``self.neighbors``) to be set
        first.
        """
        nc = self.n.astype(int)              # children per axis
        npar = int(np.prod(nc))
        S = tuple(int(v) for v in 3*nc)      # child slots per axis
        nfar = int(np.prod(S)) - 27
        posS = np.empty(S, dtype=int)
        argfar = np.empty((npar, nfar), dtype=int)
        for x in range(nc[0]):
            for y in range(nc[1]):
                for z in range(nc[2]):
                    p = (x*nc[1] + y)*nc[2] + z
                    cx, cy, cz = nc[0] + x, nc[1] + y, nc[2] + z
                    posS[:] = 1
                    posS[cx-1:cx+2, cy-1:cy+2, cz-1:cz+2] = 0
                    argfar[p, :] = np.argwhere(posS.ravel()).T[0]
        neighS = np.zeros(S, dtype=int)
        # int32: group ids, and exactly what the Fortran kernel's
        # INTEGER argument wants -- int64 made f2py copy-convert the
        # whole array EVERY m2l call
        self.farneighbors = np.zeros((np.size(self.idx), nfar),
                                     dtype=np.int32)
        gp = np.empty((npar,), dtype=int)
        for abovegroup in range(np.size(self.above.idx)):
            neighS[:] = -1
            for aboveneigh in range(27):
                nx = aboveneigh//9
                ny = (aboveneigh//3) % 3
                nz = aboveneigh % 3
                gp[:] = -1
                neigh = self.neighbors[aboveneigh, abovegroup]
                if neigh >= 0:
                    nidx = np.r_[self.idx0[neigh]:self.idx0[neigh+1]]
                    gp[self.idx[nidx]] = nidx
                    neighS[nc[0]*nx:nc[0]*(nx+1), nc[1]*ny:nc[1]*(ny+1),
                           nc[2]*nz:nc[2]*(nz+1)] = \
                        np.reshape(gp, (nc[0], nc[1], nc[2]))
            for gidx in range(self.idx0[abovegroup],
                              self.idx0[abovegroup+1]):
                self.farneighbors[gidx, :] = \
                    neighS.ravel()[argfar[self.idx[gidx], :]]

        # Slot-difference geometry, in CHILD box lengths per axis. A
        # source slot s and a child at slot c = nc + p differ by
        # d = s - c in [-(2*nc - 1), 2*nc); the near mask marks
        # |d| <= 1 per axis, and each parity's window below maps window
        # index w to slot s = w directly (w = d + c).
        rng = [np.arange(-(2*int(nc[d]) - 1), 2*int(nc[d]))
               for d in range(3)]
        posx, posy, posz = np.meshgrid(*rng, indexing='ij')
        posx = self.l[0]*posx.astype(float)
        posy = self.l[1]*posy.astype(float)
        posz = self.l[2]*posz.astype(float)
        radiisource = np.sqrt(posx**2 + posy**2 + posz**2)
        radiisource[2*nc[0]-2:2*nc[0]+1, 2*nc[1]-2:2*nc[1]+1,
                    2*nc[2]-2:2*nc[2]+1] = -1
        thetasource = np.arccos(posz/radiisource)
        phisource = np.pi - np.arctan2(posy, posx)
        radii = np.empty((npar, nfar), dtype=float)
        theta = np.empty((npar, nfar), dtype=float)
        phi = np.empty((npar, nfar), dtype=float)
        for x in range(nc[0]):
            for y in range(nc[1]):
                for z in range(nc[2]):
                    p = (x*nc[1] + y)*nc[2] + z
                    args = np.s_[nc[0]-1-x:4*nc[0]-1-x,
                                 nc[1]-1-y:4*nc[1]-1-y,
                                 nc[2]-1-z:4*nc[2]-1-z]
                    radiigroup = radiisource[args]
                    thetagroup = thetasource[args]
                    phigroup = phisource[args]
                    arggroup = np.argwhere(radiigroup.ravel() >= 0).T[0]
                    radii[p, :] = radiigroup.ravel()[arggroup]
                    theta[p, :] = thetagroup.ravel()[arggroup]
                    phi[p, :] = phigroup.ravel()[arggroup]
        self.theta = theta
        self.phi = phi

        self.transfer = np.zeros((npar, nfar, self.nnmax, self.nnmax),
                                 dtype=np.complex128)
        for j in range(self.nmax+1):
            for k in range(-j, j+1):
                idxjk = j**2 + j + k
                for n in range(self.nmax+1):
                    for m in range(-n, n+1):
                        idxnm = n**2 + n + m
                        i_km = 1j**np.abs(k-m)/(1j**np.abs(k)*1j**np.abs(m))
                        a_jknm = a_nm(j, k)*a_nm(n, m)/a_nm(j+n, k-m)
                        c = i_km*a_jknm/(-1)**j
                        for pos in range(npar):
                            y_jknm = sph_harm_of_cos(j+n, k-m, theta[pos, :],
                                                     phi[pos, :])
                            self.transfer[pos, :, idxjk, idxnm] = \
                                c * y_jknm / radii[pos, :]**(j+n+1)

        # OPT-IN c64 storage of the transfer table (SPPEEC_MID_FP32=1).
        # NOT the default: the table is FIXED-size per level (15 MB at
        # nmax 4) and the CPU kernel is compute-bound, so c64 storage
        # measured 0.82x -- the mixed-precision inner loop pays a
        # conversion per element and the halved bandwidth buys nothing
        # (2026-08-15, median-of-10 at 4320 groups). Kept for a future
        # GPU mid-M2L port, where VRAM and bandwidth do pay.
        # The raw SI entries carry 1/r**(j+n+1) and overflow float32's
        # exponent range on fine pitches (measured 5.2e38 at a 1e-5 m
        # pitch), so the magnitude is factored out IN fp64 FIRST -- the
        # ftrans lesson. r0**(j+n+1) factorizes as u[jk]*v[nm] with
        # u = r0**(j+1), v = r0**n, so de-normalisation folds into
        # per-vector scalings around the kernel call (m2l) instead of a
        # second table. Compute precision stays double: the c64 table
        # feeds a DOUBLE COMPLEX accumulator (MID_M2L_C64).
        if _os.environ.get('SPPEEC_MID_FP32') != '1':
            self._mid_u = None
            self._mid_v = None
        else:
            r0 = float(np.exp(np.mean(np.log(radii))))
            deg = np.floor(np.sqrt(np.arange(self.nnmax))).astype(int)
            self._mid_u = r0**(deg + 1.0)         # per idxjk, fp64
            self._mid_v = r0**deg.astype(float)   # per idxnm, fp64
            self.transfer = (self.transfer
                             * self._mid_u[None, None, :, None]
                             * self._mid_v[None, None, None, :]
                             ).astype(np.complex64)

    def m2l(self):
        """Multipole-to-local: accumulate the far-field at this level.

        For every box, sums the M2L translation (``self.transfer``) of the
        multipole expansions of its 189 interaction-list boxes
        (``self.farneighbors``) into its own local expansion, overwriting
        ``self.data``. Delegated to the Fortran kernel ``mp_fortran.mid_m2l``
        (the commented Python loop below is the reference implementation).
        """
        if self._mid_u is None:                   # fp64 default
            self.data = mp_fortran.mid_m2l(self.transfer.T, self.data.T,
                                           self.idx,
                                           self.farneighbors.T).T
        else:
            # T = T_stored/(u[jk]*v[nm]): pre-divide the multipoles by
            # u, post-divide the locals by v -- both fp64, so the only
            # single-precision step is the stored table itself
            md = (self.data/self._mid_u[None, :]).T
            out = mp_fortran.mid_m2l_c64(self.transfer.T, md, self.idx,
                                         self.farneighbors.T)
            self.data = out.T/self._mid_v[None, :]
        #     for neigh in range(189):
        #         ngroup = self.farneighbors[gidx, neigh]
        #         if ngroup >= 0:
        #             pos = self.idx[gidx]
        #             l_jk[gidx, :] += np.dot(self.transfer[pos, neigh, :, :],
        #                                     self.data[ngroup, :])

    def midinit(self):
        """Precompute the M2M and L2L translation matrices for this level.

        Builds the two parent<->child translation operators from the box-centre
        offsets within a parent cell:

        * ``self.m2mtrans`` --- the **M2M** (multipole-to-multipole, upward)
          matrix used by :meth:`m2m` to shift children's multipole expansions
          to the parent centre.
        * ``self.l2ltrans`` --- the **L2L** (local-to-local, downward) matrix
          used by :meth:`l2l` to shift the parent's local expansion to the
          child centres (built with ``phi`` shifted by pi, i.e. the reversed
          offset).

        Both follow the Greengard-Rokhlin translation formulas with the
        ``A_n^m`` coefficients and ``z**|j|`` radial factors.
        """
        x0 = np.arange(-(self.n[0]-1)/2, (self.n[0]-1)/2+1)
        y0 = np.arange(-(self.n[1]-1)/2, (self.n[1]-1)/2+1)
        z0 = np.arange(-(self.n[2]-1)/2, (self.n[2]-1)/2+1)
        posgx, posgy, posgz = np.meshgrid(x0, y0, z0, indexing='ij')
        posgx = np.float64(posgx)*self.l[0]
        posgy = np.float64(posgy)*self.l[1]
        posgz = np.float64(posgz)*self.l[2]
        rgroup = np.sqrt(posgx**2 + posgy**2 + posgz**2)
        thetagroup = np.arccos(posgz/rgroup)
        phigroup = np.arctan2(posgy, posgx)
        rgroup = rgroup.ravel()
        self.rgroup = rgroup
        thetagroup = thetagroup.ravel()
        phigroup = phigroup.ravel()
        self.m2mtrans = np.ndarray((self.nmax+1, ), dtype=object)
        zj = np.zeros((self.nmax+1, np.prod(self.n)))
        zjyjk = np.zeros((self.nnmax, np.prod(self.n)), dtype=np.complex128)
        self.yjk = np.zeros((self.nnmax, np.prod(self.n)), dtype=np.complex128)
        czjyjk = np.zeros((64,), dtype=np.complex128)
        self.m2mtrans = np.zeros((self.nnmax, np.prod(self.n)*self.nnmax),
                                 dtype=np.complex128)
        # The leaf expansions (leafinit) and the M2L transfer tables
        # (midm2linit/topinit) all live in the z-FLIPPED harmonic frame
        # (theta = pi - arccos(z/r)); topinit encodes its transfer direction
        # as (arccos(z/r), phi+pi), which IS the flipped-frame reversed
        # vector. The M2M shift vectors below must use the same flipped
        # frame -- with the plain arccos(z/r) the children's z-offsets are
        # MIRRORED inside every parent, an error that cancels nowhere and
        # corrupts every interaction routed through m2m (latent until a
        # numlevels>=3 tree with a non-empty top-level interaction list;
        # found by validate_traverseP3_farfield.py, which showed the same
        # nmax-independent ~5% far-field error for the inductive path).
        # The L2L block below keeps (arccos(z/r), phi+pi): like topinit,
        # that combination is exactly the flipped-frame reversed offset.
        for j in range(self.nmax+1):
            zj[j, :] = np.abs(rgroup)**j
            for k in range(-j, j+1):
                self.yjk[j**2+j+k, :] = sph_harm_of_cos(j, k, thetagroup,
                                                        phigroup)
                zjyjk[j**2+j+k, :] = zj[j, :] * \
                    sph_harm_of_cos(j, -k, np.pi - thetagroup, phigroup)
        for n in range(self.nmax+1):
            for m in range(-n, n+1):
                anm = a_nm(n, m)
                im = 1j**np.abs(m)
                idxnm = n**2 + n + m
                for j in range(n+1):
                    lowbound = max(m+j-n, -j)
                    highbound = min(m+n-j+1, j+1)
                    for k in range(lowbound, highbound):
                        idxm = (n - j)**2 + n + m - j - k
                        idxjk = j**2 + j + k
                        a = a_nm(j, k)*a_nm(n-j, m-k)/anm
                        i = im/(1j**np.abs(k))/(1j**np.abs(m-k))
                        c = a*i
                        czjyjk = c*zjyjk[idxjk, :]
                        self.m2mtrans[idxnm, idxm::self.nnmax] = czjyjk
        phigroup += np.pi
        self.l2ltrans = np.zeros((np.prod(self.n)*self.nnmax, self.nnmax),
                                 dtype=np.complex128)
        for n in range(self.nmax+1):
            for m in range(-n, n+1):
                anm = a_nm(n, m)
                im = 1j**np.abs(m)
                idxnm = n**2 + n + m
                for j in range(n, self.nmax+1):
                    rho = np.abs(rgroup)**(j-n)
                    for k in range(m-j+n, m-n+j+1):
                        idxjk = j**2 + j + k
                        a = anm*a_nm(j-n, k-m)/a_nm(j, k)
                        i = 1j**np.abs(k)/(im*1j**np.abs(k-m))/(-1)**(j+n)
                        c = a*i
                        yjk = sph_harm_of_cos(j-n, k-m, thetagroup, phigroup)
                        self.l2ltrans[idxnm::self.nnmax, idxjk] = c*yjk*rho

    def m2m(self):
        """Multipole-to-multipole: translate expansions up to the parent.

        For each parent box, gathers its children's multipole coefficients and
        applies ``self.m2mtrans`` to shift them to the parent centre, writing
        the combined expansion into ``self.above.data``. The upward pass of the
        FMM at this level.
        """
        gsize = np.size(self.idx0) - 1
        self.above.data = np.empty((gsize, self.nnmax), dtype=np.complex128)
        m_temp = np.zeros((np.prod(self.n), self.nnmax), dtype=np.complex128)
        for group in range(gsize):
            m_temp[...] = 0
            fidx = np.r_[self.idx0[group]:self.idx0[group+1]]
            m_temp[self.idx[fidx], :] = self.data[fidx, :]
            self.above.data[group, :] = np.dot(self.m2mtrans, m_temp.ravel())

    def l2l(self):
        """Local-to-local: translate the parent's expansion down to children.

        For each box, applies ``self.l2ltrans`` to its parent's local
        expansion (``self.above.data``) to shift it to the child centres and
        adds the result to ``self.data``. The downward pass of the FMM at this
        level.
        """
        for group in range(np.size(self.idx0) - 1):
            fidx = np.r_[self.idx0[group]:self.idx0[group+1]]
            l_temp = np.dot(self.l2ltrans, self.above.data[group, :])
            l_temp = np.reshape(l_temp, (np.prod(self.n), self.nnmax))
            self.data[fidx, :] += l_temp[self.idx[fidx], :]


# fp32 campaign PHASE 3b (decided with the user 2026-08-14): the
# top-level M2L spectra STORED single precision. ftrans was the
# largest single array in the R3 census (382 MB complex128; scales
# with top-grid volume x 81 channels). It is OPERATOR DATA, built
# fp64 (the |r|^(j+n) / harmonic construction is where cancellation
# lives) and cast on store; storage error ~1e-7 relative sits far
# below the top far-field's own 2.8e-4 method accuracy. Consumers:
#   * GPU path: re-embeds its own device spectra FROM ftrans at init
#     (one per-channel upcast, init-time only) -- production boxes
#     never touch ftrans again after init;
#   * CPU fallback (Fortran FMMtop.m2l): needs complex128, so
#     m2lfortran RESTORES ftrans to fp64 IN PLACE on first CPU use --
#     a CPU-only box ends exactly where it was before this change,
#     a GPU box keeps the half-size copy;
#   * numpy m2l(): dtype-promotion safe as-is.
# SPPEEC_TOP_FP64=1 stores double from the start (A/B hatch).
_TOP_DT = (np.complex128 if _os.environ.get('SPPEEC_TOP_FP64') == '1'
           else np.complex64)


class TopLevel:
    """The coarsest (root) level of the FMM tree.

    At the top level every box interacts with every other, and because the
    M2L kernel depends only on the box-to-box offset the operator is
    (multilevel) Toeplitz --- so the full all-to-all M2L is applied by FFT
    rather than an explicit interaction list. :meth:`topinit` precomputes the
    FFT'd transfer function; :meth:`m2l` / :meth:`m2lfortran` apply it (NumPy
    and Fortran paths). :meth:`getneighbors` fills in the neighbour lists for
    every level below. Reuses :class:`Level`'s initializer for its geometry.
    """
    def __init__(self, n, l, nmax):
        """Initialize the top level and precompute its FFT M2L operator.

        Borrows :meth:`Level.__init__` for geometry/expansion sizing, then
        calls :meth:`topinit` to build the Toeplitz transfer function.
        """
        Level.__init__(self, n, l, nmax)
        self.xidx = None
        self.yidx = None
        self.zidx = None
        self.topinit()

    def topinit(self):
        """Precompute the translation-invariant (Toeplitz) top-level M2L.

        Evaluates the M2L transfer function for every box-to-box offset ---
        the spherical harmonic ``Y_{j+n}^{k-m}`` divided by ``|r|**(j+n+1)``
        --- over the full ``(2*ng-1)`` difference grid, exploiting the mirror
        symmetries of ``theta``/``phi`` to fill the negative offsets. The
        near-field region (offsets handled directly at lower levels) is zeroed,
        and each harmonic block is FFT-transformed via
        :func:`tp.t3multnonsym3` into ``self.ftrans`` so the all-to-all M2L
        becomes an elementwise multiply in Fourier space. Also precomputes the
        M2L coefficient array ``self.c`` (the ``A_n^m`` combination and
        imaginary phase factor).

        Note: the ``|r|**(j+n)`` division is singular at the self-offset; that
        entry is zeroed immediately afterward, so the (suppressed) divide
        warning there is benign.

        THE MIRROR GUARDS ARE ``ng[c] > 1``, NOT ``> 2`` (fixed 2026-08-05).
        The difference grid is ``2*ng[c]-1`` wide and the forward fill covers
        indices ``ng-1 : 2*ng-1``, i.e. offsets ``0 .. +(ng-1)``; every
        NEGATIVE offset comes from the mirror. At ``ng[c] == 2`` the grid is
        3 wide and offset ``-1`` (index 0) exists and needs mirroring, but
        the old ``> 2`` guard skipped it -- leaving ``absrjn`` ZERO there, so
        ``yjnkm/absrjn`` produced inf/nan, and the near-field zeroing that
        follows only clears the three-axis INTERSECTION, so on a grid like
        ``34x2x34`` the nan survived into ``t3multnonsym3``'s ``fftn`` and
        made the whole transfer function nan. That is the long-standing
        "``ng == 2`` gives NaN" defect, and it hit the inductive path as well
        as ``traverseP3``. At ``ng[c] == 2`` both sides of every mirror are
        length 1 (``[0::-1]`` from ``[2:3]``); at ``ng[c] == 1`` the source
        slice is empty, which is why the guard is still needed at all.
        """
        ng = self.n.astype(int)
        lg = self.l
        # the whole-domain top-level transform is the one FFT that
        # benefits from threads (see toeplitz._FFTW_THREADS_TOP).
        # LAZY since phase 3b: tm2l's a/b/c are pyfftw WORKING BUFFERS
        # (206 MB at R3, plus the FFTW plans' own internal storage --
        # the census's opaque '24x pyfftw.FFTW'). They serve ONLY the
        # CPU m2lfortran path, which a GPU box never runs, so they are
        # built on first CPU use instead of here. Cost of the move: on
        # CPU-only boxes the FFTW_MEASURE planning happens inside the
        # FIRST matvec rather than in setup -- same total, different
        # attribution; remember that when profiling first-call timings.
        self._tm2l = None
        self._tm2l_args = (ng, self.nnmax, tp._FFTW_THREADS_TOP)
        posgx, posgy, posgz = np.meshgrid(np.arange(ng[0]), np.arange(ng[1]),
                                          np.arange(ng[2]), indexing='ij')
        posgx = np.float64(posgx)*lg[0]
        posgy = np.float64(posgy)*lg[1]
        posgz = np.float64(posgz)*lg[2]
        rgroup = np.sqrt(posgx**2 + posgy**2 + posgz**2)
        rgroup[0, 0, 0] = rgroup[1, 0, 0]  # avoid divide-by-zero error
        thetagroup = np.zeros((2*ng[0]-1, 2*ng[1]-1, 2*ng[2]-1))
        phigroup = np.zeros((2*ng[0]-1, 2*ng[1]-1, 2*ng[2]-1))
        thetagroup[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1, ng[2]-1:2*ng[2]-1] \
            = np.arccos(posgz/rgroup)
        phigroup[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1, ng[2]-1:2*ng[2]-1] \
            = np.arctan2(posgy, posgx)
        if ng[2] > 1:
            thetagroup[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1, ng[2]-2::-1] \
                = np.arccos(-posgz[:, :, 1:ng[2]]/rgroup[:, :, 1:ng[2]])
            phigroup[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1, ng[2]-2::-1] \
                = phigroup[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1,
                           ng[2]:2*ng[2]-1]
        if ng[1] > 1:
            thetagroup[ng[0]-1:2*ng[0]-1, ng[1]-2::-1, :2*ng[2]-1] \
                = thetagroup[ng[0]-1:2*ng[0]-1, ng[1]:2*ng[1]-1, :]
            phigroup[ng[0]-1:2*ng[0]-1, ng[1]-2::-1, ng[2]-1:2*ng[2]-1] \
                = np.arctan2(-posgy[:, 1:ng[1], :], posgx[:, 1:ng[1], :])
            if ng[2] > 1:
                phigroup[ng[0]-1:2*ng[0]-1, ng[1]-2::-1, ng[2]-2::-1] \
                    = np.arctan2(-posgy[:, 1:ng[1], 1:ng[2]],
                                 posgx[:, 1:ng[1], 1:ng[2]])
        if ng[0] > 1:
            thetagroup[ng[0]-2::-1, :2*ng[1]-1, :2*ng[2]-1] \
                = thetagroup[ng[0]:2*ng[0]-1, :, :]
            phigroup[ng[0]-2::-1, ng[1]-1:2*ng[1]-1, ng[2]-1:2*ng[2]-1] \
                = np.arctan2(posgy[1:ng[0], :ng[1], :ng[2]],
                             -posgx[1:ng[0], :ng[1], :ng[2]])
            if ng[2] > 1:
                phigroup[ng[0]-2::-1, ng[1]-1:2*ng[1]-1, ng[2]-2::-1] \
                    = phigroup[ng[0]-2::-1, ng[1]-1:2*ng[1]-1, ng[2]:2*ng[2]-1]
            if ng[1] > 1:
                phigroup[ng[0]-2::-1, ng[1]-2::-1, ng[2]-1:2*ng[2]-1] \
                    = np.arctan2(-posgy[1:ng[0], 1:ng[1], :],
                                 -posgx[1:ng[0], 1:ng[1], :])
                if ng[2] > 1:
                    phigroup[ng[0]-2::-1, ng[1]-2::-1, ng[2]-2::-1] \
                        = phigroup[ng[0]-2::-1, ng[1]-2::-1, ng[2]:2*ng[2]-1]
        phigroup += np.pi
        nnmax2 = (2*self.nmax + 1)**2
        absrjn = np.zeros((2*ng[0]-1, 2*ng[1]-1, 2*ng[2]-1))
        yjnkm = np.zeros((nnmax2, 2*ng[0]-1, 2*ng[1]-1, 2*ng[2]-1),
                         dtype=np.complex128)
        trans = np.zeros((nnmax2, 2*ng[0]-1, 2*ng[1]-1, 2*ng[2]-1),
                         dtype=np.complex128)
        # STORED in _TOP_DT (single by default, phase 3b), PER-CHANNEL
        # MAX-NORMALISED. The normalisation is not cosmetic: channel
        # j+n carries 1/r^(j+n+1) in SI metres, up to ~1e27 at mm-scale
        # top boxes and PAST FLOAT32'S 1e38 EXPONENT RANGE on finer
        # geometries -- storing raw values NaN'd every validator with
        # a sub-mm pitch (found 2026-08-14). Scaled per channel the
        # within-channel dynamic range is only (rmax/rmin)^(j+n+1),
        # comfortably inside fp32; the fp64 scale vector restores
        # magnitudes at every consumer. Each channel is computed in
        # full fp64 and cast on assignment -- no accuracy loss vs
        # build-double-then-cast, no 2x transient.
        self.ftrans = np.zeros((nnmax2, 2*ng[0], 2*ng[1], 2*ng[2]),
                               dtype=_TOP_DT)
        self._ftrans_scale = np.ones(nnmax2)
        for jpn in range(2*self.nmax):
            absrjn[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1, ng[2]-1:2*ng[2]-1] = \
                np.abs(rgroup)**(jpn+1)
            if ng[2] > 1:
                absrjn[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1, ng[2]-2::-1] \
                    = absrjn[ng[0]-1:2*ng[0]-1, ng[1]-1:2*ng[1]-1,
                             ng[2]:2*ng[2]-1]
            if ng[1] > 1:
                absrjn[ng[0]-1:2*ng[0]-1, ng[1]-2::-1, :] \
                    = absrjn[ng[0]-1:2*ng[0]-1, ng[1]:2*ng[1]-1, :]
            if ng[0] > 1:
                absrjn[ng[0]-2::-1, :, :] = absrjn[ng[0]:2*ng[0]-1, :, :]
            for kmm in range(-jpn, jpn+1):
                idx = jpn**2 + jpn + kmm
                yjnkm[idx, :, :, :] = sph_harm_of_cos(jpn, kmm, thetagroup,
                                                      phigroup)
                # absrjn = |r|^(j+n) is zero at the self-position, so this
                # division yields inf/nan there. Those entries are the near-
                # field region that is zeroed out on the following lines, so the
                # warning is benign; suppress it.
                with np.errstate(divide='ignore', invalid='ignore'):
                    trans[idx, :, :, :] = \
                        np.reshape(yjnkm[idx, :, :, :],
                                   (2*ng[0]-1, 2*ng[1]-1, 2*ng[2]-1))/absrjn
                ngmin = np.int_(np.maximum(ng - 2, np.zeros((3,))))
                trans[idx, ngmin[0]:ng[0]+1, ngmin[1]:ng[1]+1,
                      ngmin[2]:ng[2]+1] = 0
                blk = tp.t3multnonsym3(np.reshape(
                    trans[idx, :, :, :], (2*ng[0]-1, 2*ng[1]-1, 2*ng[2]-1)),
                    ng[0], ng[1], ng[2])
                if self.ftrans.dtype != np.complex128:
                    # normalise IN FP64, before the cast can overflow
                    s = float(np.abs(blk).max())
                    if s > 0.0:
                        self._ftrans_scale[idx] = s
                        blk = blk/s
                self.ftrans[idx, :, :, :] = blk
        self.c = np.zeros(self.nnmax**2, dtype=np.complex128)
        for n in range(self.nmax+1):
            for m in range(-n, n+1):
                idxnm = n**2 + n + m
                anm = a_nm(n, m)
                for j in range(self.nmax+1):
                    for k in range(-j, j+1):
                        idxjk = j**2 + j + k
                        idxnmjk = self.nnmax*idxnm + idxjk
                        ajk = a_nm(j, k)
                        ajnkm = a_nm(j+n, k-m)
                        imagkm = 1j**(np.abs(m-k) - np.abs(k) - np.abs(m))
                        self.c[idxnmjk] = (-1)**(-j)*imagkm*ajk*anm/ajnkm

    def getneighbors(self):
        """Fill in each level's neighbour lists, top to bottom.

        Walks the tree downward from the top level and, at every level,
        computes the 27-box (3x3x3) neighbour list for each group from its
        integer coordinates (`xidx`/`yidx`/`zidx`) via the Fortran helper
        ``f_setup.f_getneighbors``, storing it in ``current.neighbors``. These
        lists drive the near-field P2P and the mid-level M2L interaction lists.
        """
        current = self.below
        # while not isinstance(current, LeafLevel):
        while current is not None:
            # if current.above is not None:
            # xidx += np.floor_divide(current.idx, current.n[1]*current.n[2])
            # yidx += np.mod(np.floor_divide(current.idx, current.n[2]),
            #                current.n[1])
            # zidx += np.mod(current.idx, current.n[2])
            current.neighbors = \
                f_setup.f_getneighbors(current.xidx, current.yidx,
                                       current.zidx).T
            current = current.below
            # current.xidx = xidx
            # current.yidx = yidx
            # current.zidx = zidx

    def m2l(self):
        """All-to-all top-level M2L via FFT (NumPy path).

        Applies the translation-invariant M2L over every pair of top-level
        boxes as an FFT convolution: forward-transforms the multipole
        coefficients (``tp.t3mult7``), multiplies by the precomputed transfer
        function ``self.ftrans`` weighted by ``self.c``, and inverse-FFTs to
        obtain the local expansions, written back to ``self.data``. See
        :meth:`m2lfortran` for the faster Fortran equivalent.
        """
        mg = np.zeros((self.nnmax, np.prod(self.n)), dtype=np.complex128)
        mg[:, self.idx] = self.data.T
        mg = np.reshape(mg, (self.nnmax, self.n[0], self.n[1], self.n[2]))
        fmg = tp.t3mult7(mg, self.nnmax, self.n)
        lnm = np.zeros((self.nnmax, 2*self.n[0], 2*self.n[1], 2*self.n[2]),
                       dtype=np.complex128)
        for n in range(self.nmax + 1):
            for m in range(-n, n+1):
                idxnm = n**2 + n + m
                nnmaxidxnm = self.nnmax*idxnm
                for j in range(self.nmax+1):
                    for k in range(-j, j+1):
                        idxjk = j**2 + j + k
                        idxnmjk = nnmaxidxnm + idxjk
                        ch = (j+n)**2 + j + n + k - m
                        # scale restores the fp32-normalised channel
                        # magnitude (ones under fp64 storage)
                        c = self.c[idxnmjk]*self._ftrans_scale[ch]
                        lnm[idxnm, :, :, :] += c*np.reshape(
                            self.ftrans[ch, :, :, :],
                            (2*self.n[0], 2*self.n[1], 2*self.n[2])) * \
                            np.reshape(fmg[j**2+j+k, :, :, :],
                                       (2*self.n[0], 2*self.n[1], 2*self.n[2]))
        lnm = np.fft.ifft(lnm, 2*self.n[2], 3)
        lnm = np.fft.ifft(lnm[:, :, :, :self.n[2]], 2*self.n[1], 2)
        lnm = np.fft.ifft(lnm[:, :, :self.n[1], :self.n[2]], 2*self.n[0], 1)
        lnm = lnm[:, :self.n[0], :self.n[1], :self.n[2]]
        lnm = np.reshape(lnm, (self.nnmax, np.prod(self.n)))
        self.data = lnm[:, self.idx].T

    def _m2l_gpu_init(self):
        """One-time GPU setup for the CHANNEL-TILED top-level M2L.

        v2 (2026-08-12): the v1 path materialised the translation
        tensor DENSELY as (grid, nnmax, nnmax) -- 13.76 GB at the
        51M-cell rung, OOM on a 12 GB card -- even though ``ftrans``
        has only (2*nmax+1)**2 = 81 distinct combined-harmonic
        channels (a 7.7x redundant expansion). v2 keeps the compact
        form: per channel t, a small dense (nnmax, nnmax) coupling
        C_t (the c-weights masked to T == t) and the channel's grid
        spectrum ftrans[t]. The apply contracts
            lnm += (C_t @ fmg) * ftrans[t]
        over the 81 channels -- pure bandwidth. ``ftrans`` stays
        device-resident when it fits; otherwise it STREAMS per
        channel from the host (the p2p slab lesson: the device
        working set is a few (nnmax, G) buffers regardless of
        operator size, so a modest GPU scales to 1e9-cell top grids).
        """
        import cupy as cp
        nn = self.nnmax
        nn2 = (2*self.nmax + 1)**2
        T = np.zeros((nn, nn), dtype=np.int64)
        C = np.zeros((nn, nn), dtype=np.complex128)
        for n in range(self.nmax + 1):
            for m in range(-n, n + 1):
                idxnm = n*n + n + m
                for j in range(self.nmax + 1):
                    for k in range(-j, j + 1):
                        idxjk = j*j + j + k
                        T[idxnm, idxjk] = (j + n)**2 + (j + n) + (k - m)
                        C[idxnm, idxjk] = self.c[nn*idxnm + idxjk]
        Ct = np.zeros((nn2, nn, nn), dtype=np.complex128)
        for t in range(nn2):
            Ct[t][T == t] = C[T == t]
        self._gpu_Ct = cp.asarray(Ct)              # 81 x 25 x 25: tiny
        self._gpu_t_used = [t for t in range(nn2) if Ct[t].any()]
        # flop-minimal MAC maps (each (nm, jk) pair has exactly ONE
        # active channel T[nm, jk]; dense per-channel gemms did ~100x
        # redundant fp64 work -- fatal on a consumer card's 1/64-rate
        # fp64 pipe, measured ~2.5 s/call at R4)
        self._gpu_T = cp.asarray(T)                # (nn, nn) int
        self._gpu_C = cp.asarray(C)                # (nn, nn) complex
        # 5-SMOOTH FFT SIZES (2026-08-12): the natural 2n padding can
        # carry large PRIME factors (R4's top grid [107,134,12] pads
        # to 2x107 / 4x67 -- both prime -> cuFFT Bluestein, ~10x off
        # radix, 2.5 s/call measured). A circular convolution of the
        # (2n-1)-offset kernel with n-support moments is exact at ANY
        # size >= 2n-1, so re-embed the kernel on the next 2/3/5-
        # smooth grid instead. The raw wrapped kernel is recovered
        # from the stored 2n-grid spectra by inverse FFT (one-time;
        # no topinit change), head [0:n] and tail (n-1) slots mapped
        # across, and transformed at the smooth size.
        def _smooth5(k):
            def ok(x):
                for p in (2, 3, 5):
                    while x % p == 0:
                        x //= p
                return x == 1
            while not ok(k):
                k += 1
            return k
        n0, n1, n2 = (int(v) for v in self.n)
        S = tuple(_smooth5(2*v - 1) for v in (n0, n1, n2))
        self._gpu_S = S
        G = int(np.prod(S))
        ft2 = np.empty((nn2, G), dtype=np.complex128)
        srcs, dsts = [], []
        for ni, Si in zip((n0, n1, n2), S):
            srcs.append(np.r_[0:ni, ni + 1:2*ni])
            dsts.append(np.r_[0:ni, Si - (ni - 1):Si])
        for t in range(nn2):
            # per-channel upcast + de-normalise (init-time only):
            # ftrans is STORED single and channel-scaled since phase
            # 3b; the re-embed ifft/fft run in fp64 so storage is the
            # only single-precision step
            k2n = np.fft.ifftn(
                (self.ftrans[t].astype(np.complex128)
                 * self._ftrans_scale[t]).reshape(2*n0, 2*n1, 2*n2))
            emb = np.zeros(S, dtype=np.complex128)
            emb[np.ix_(dsts[0], dsts[1], dsts[2])] = \
                k2n[np.ix_(srcs[0], srcs[1], srcs[2])]
            ft2[t] = np.fft.fftn(emb).ravel()
        free, _total = cp.cuda.runtime.memGetInfo()
        need_work = 4*nn*G*16
        if ft2.nbytes + need_work < 0.85*free:
            self._gpu_ft = cp.asarray(ft2)
            self._gpu_ft_host = None
        else:
            self._gpu_ft = None                    # stream per channel
            self._gpu_ft_host = ft2
        self._gpu_idx = cp.asarray(self.idx)
        self._gpu_ready = True

    def _m2l_gpu(self):
        """Top-level M2L on the GPU, channel-tiled (see _m2l_gpu_init).

        Not bitwise identical to the CPU path (cuFFT rounding, channel
        summation order); agreement is at the 1e-13 class and swamped
        by rtol, per the GPU defaults policy."""
        import cupy as cp
        if not getattr(self, '_gpu_ready', False):
            self._m2l_gpu_init()
        n0, n1, n2 = (int(v) for v in self.n)
        nn = self.nnmax
        nn2 = (2*self.nmax + 1)**2
        S = self._gpu_S
        G = int(np.prod(S))
        # channel-chunked forward FFTs (5-smooth sizes) bound the
        # pad workspace
        mg = cp.zeros((nn, n0*n1*n2), dtype=cp.complex128)
        mg[:, self._gpu_idx] = cp.asarray(
            np.ascontiguousarray(self.data.T))
        fmg = cp.empty((nn, G), dtype=cp.complex128)
        CH = 8
        for c0 in range(0, nn, CH):
            ch = min(CH, nn - c0)
            pad = cp.zeros((ch,) + S, dtype=cp.complex128)
            pad[:, :n0, :n1, :n2] = mg[c0:c0 + ch].reshape(
                ch, n0, n1, n2)
            fmg[c0:c0 + ch] = cp.fft.fftn(
                pad, axes=(1, 2, 3)).reshape(ch, G)
        del mg, pad
        lnm = cp.empty((nn, G), dtype=cp.complex128)
        if self._gpu_ft is not None:
            # flop-minimal path: per output harmonic i, gather that
            # row's 25 channel spectra, one elementwise multiply, one
            # (1 x nn) zgemv -- 625 G-vector ops total instead of the
            # per-channel gemms' ~50k slot products
            buf = cp.empty((nn, G), dtype=cp.complex128)
            for i in range(nn):
                cp.take(self._gpu_ft, self._gpu_T[i], axis=0, out=buf)
                cp.multiply(buf, fmg, out=buf)
                lnm[i] = self._gpu_C[i] @ buf
        else:
            # streamed operator (hero scale): channel loop, one host
            # fetch per channel
            lnm[:] = 0
            Y = cp.empty((nn, G), dtype=cp.complex128)
            for t in self._gpu_t_used:
                ft = cp.asarray(self._gpu_ft_host[t])
                cp.matmul(self._gpu_Ct[t], fmg, out=Y)
                cp.multiply(Y, ft[None, :], out=Y)
                cp.add(lnm, Y, out=lnm)
            del Y
        del fmg
        out = cp.empty((nn, n0*n1*n2), dtype=cp.complex128)
        for c0 in range(0, nn, CH):
            ch = min(CH, nn - c0)
            b = cp.fft.ifftn(lnm[c0:c0 + ch].reshape(
                (ch,) + S), axes=(1, 2, 3))
            out[c0:c0 + ch] = b[:, :n0, :n1, :n2].reshape(ch, -1)
        self.data = cp.asnumpy(out[:, self._gpu_idx].T)

    def m2lfortran(self):
        """All-to-all top-level M2L via FFT (Fortran path).

        Identical result to :meth:`m2l` but the transfer-multiply is done by
        the ``FMMtop.m2l`` Fortran kernel and the inverse FFT by
        ``self.tm2l.ifft``; this is the production path used in the multi-level
        traversal. Overwrites ``self.data`` with the local expansions.
        With ``SPPEEC_GPU=1`` the whole stage runs on the device instead
        (:meth:`_m2l_gpu`); any GPU failure falls back here permanently
        for the process, with one warning.
        """
        global _GPU
        if _GPU:
            try:
                self._m2l_gpu()
                return
            except Exception as exc:
                import warnings
                warnings.warn("SPPEEC_GPU=1 but the GPU m2l failed (%s: %s)"
                              " -- falling back to the CPU path for the "
                              "rest of the process"
                              % (type(exc).__name__, exc))
                _GPU = False
        # phase 3b: this CPU path needs the fp64 spectra (the Fortran
        # kernel is complex*16) and the planned FFT buffers -- both
        # deferred so a GPU box never pays for them. One-time each.
        if self.ftrans.dtype != np.complex128:
            self.ftrans = (self.ftrans.astype(np.complex128)
                           * self._ftrans_scale[:, None, None, None])
            self._ftrans_scale = np.ones_like(self._ftrans_scale)
        if self._tm2l is None:
            ng_, nn_, thr_ = self._tm2l_args
            self._tm2l = tp.ToeplitzM2L(ng_, nn_, threads=thr_)
        mg = np.zeros((self.nnmax, np.prod(self.n)), dtype=np.complex128)
        mg[:, self.idx] = self.data.T
        mg = np.reshape(mg, (self.nnmax, self.n[0], self.n[1], self.n[2]))
        # forward transform on tm2l's PLANNED (and, at the top level,
        # THREADED) buffers -- same staged padded pipeline t3mult7 ran
        # through numpy's pocketfft, which was 120 ms/matvec on
        # square_coil and unplanned every call
        fmg = self._tm2l.fft(mg)
        lnm = FMMtop.m2l(fmg.T, self.ftrans.T, self.c, self.nmax,
                         self.n[0], self.n[1], self.n[2])
        lnm = self._tm2l.ifft(lnm.T)
        lnm = np.reshape(lnm, (self.nnmax, np.prod(self.n)))
        self.data = lnm[:, self.idx].T


