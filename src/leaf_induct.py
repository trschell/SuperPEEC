# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""LeafInduct leaf-level class: the inductive near-field (P2P) operator.

Handles the near-field (direct) part of the partial-inductance interaction in
the FMM/PEEC solve: the coupling between filaments in a leaf box and those in
its 26 nearest-neighbour boxes, which the multipole expansion does not cover.

Because the partial inductance between two filaments depends only on their
relative offset, each leaf box's self-and-neighbour interaction is a
(block-)Toeplitz operator and is applied by zero-padded FFT convolution rather
than a dense matrix-vector product. The generating inductance table comes from
:func:`greens.genL3D`. The class also builds and applies a sparse-approximate-
inverse (SPAI / RDF) preconditioner for the iterative solver.

Two near-field drivers are provided: :meth:`p2pcpu` (the FFTW slab
pipeline, the default) and :meth:`p2pgpu2` (the CuPy port, bound when
a GPU is present and SPPEEC_GPU permits). Split out of multipole.py.
"""
from multipole_common import *  # noqa: F401,F403  shared imports/guards
from special import *  # noqa: F401,F403
from greens import *  # noqa: F401,F403
import os as _os

from levels import Level, LeafLevel, MidLevel, TopLevel


class LeafInduct(LeafLevel):
    """Leaf-level partial-inductance near-field operator for one orientation.

    Represents the inductive P2P interaction for one filament orientation
    within a leaf box. Extends :class:`levels.LeafLevel` with the near-field
    transfer arrays (built in :meth:`p2pinit`), the FFT-convolution drivers,
    and the SPAI/RDF preconditioner (:meth:`spaiinit` / :meth:`spaiapply`).

    Parameters
    ----------
    n : sequence of int
        Leaf grid dimensions (filaments per box along x, y, z).
    ng : sequence of int
        Number of leaf boxes (groups) along x, y, z.
    l : sequence of float
        Physical filament dimensions along x, y, z (metres).
    lscale : float
        Scale factor the raw partial inductances are divided by (non-
        dimensionalization / conditioning of the system matrix).
    nmax : int
        Maximum multipole order (passed through to :class:`LeafLevel`).
    orientation : {'e', 'f', 'g'}
        Filament current direction; selects which axis is the filament-length
        axis when the inductance table is generated.

    Attributes
    ----------
    selfind : float
        Self partial inductance of a single filament (the ``[0, 0, 0]`` entry
        of the scaled inductance table).
    p2p : callable
        The active near-field driver (bound to :meth:`p2pcpu`).
    """
    def __init__(self, n, ng, l, lscale, nmax, orientation):
        LeafLevel.__init__(self, n, ng, l, nmax, orientation)
        self.r = 0
        self.lscale = lscale
        self.jomega = None
        self.p2pinit(orientation)
        self.p2p = self.p2pcpu
        # GPU near field (cupy): DEFAULT-ON (user decision
        # 2026-08-12; measured 4.5e-16 vs CPU -- machine epsilon, so
        # even oracle runs keep their precision). SPPEEC_GPU_P2P or
        # SPPEEC_GPU set to '0' opts out; anchor-based suites pin
        # '0' explicitly, per the defaults-serve-users policy.
        if (_os.environ.get('SPPEEC_GPU_P2P', 'auto') != '0'
                and _os.environ.get('SPPEEC_GPU', 'auto') != '0'):
            try:
                import cupy
                if cupy.cuda.runtime.getDeviceCount() > 0:
                    self.p2p = self.p2pgpu2
            except Exception:
                pass                 # auto: CPU fallback is normal
        self.iternumber = 0

    def __del__(self):
        """Release the persistent FFTW plans held by this leaf."""
        mp_fortran.destroy_fft_plans(self.plans)

    def p2pinit(self, orientation):
        """Precompute the near-field partial-inductance transfer arrays.

        Allocates the FFT workspaces and persistent FFTW plans, generates the
        self-filament partial-inductance table with :func:`greens.genL3D`
        (permuting the axes so the filament-length axis matches `orientation`,
        then transposing the result back to the leaf's x, y, z layout), scales
        it by ``lscale``, and assembles the 27 neighbour-direction transfer
        blocks in ``self.p2p_transfer``. Each block is the slice of the
        inductance table for one relative neighbour offset (neighbour index
        ``9*(dx+1) + 3*(dy+1) + (dz+1)`` for offsets ``dx, dy, dz`` in
        ``{-1, 0, 1}``), pre-transformed via :func:`tp.t3multnonsym3` so it can
        be consumed directly by the FFT convolution in the p2p drivers.

        Parameters
        ----------
        orientation : {'e', 'f', 'g'}
            Filament current direction, selecting the axis permutation.

        Side Effects
        ------------
        Sets ``self.plans``, ``self.induct``, ``self.selfind``,
        ``self.p2p_transfer`` (and the scratch ``singlegroup_*`` buffers).
        """
        self.singlegroup = tp.ToeplitzP2P(self.m)
        self.singlegroup_a = np.zeros((2*self.m[0], self.m[1], self.m[2]),
                                      dtype=np.complex128)
        self.singlegroup_b = np.zeros((2*self.m[0], 2*self.m[1], self.m[2]),
                                      dtype=np.complex128)
        self.singlegroup_c = np.zeros((2*self.m[0], 2*self.m[1], 2*self.m[2]),
                                      dtype=np.complex128)
        self.plans = mp_fortran.get_fft_plans(self.m, self.singlegroup_a.T,
                                              self.singlegroup_b.T,
                                              self.singlegroup_c.T)
        # the p2p algorithm only requires induct to have a size of 2*n in
        # each dimension, but the preconditioner requires 2*n + 1
        if orientation == 'e':
            induct = genL3D(self.l[0], self.l[1], self.l[2],
                            2*self.n[0]+1, 2*self.n[1]+1, 2*self.n[2]+1)
        elif orientation == 'f':
            induct = genL3D(self.l[1], self.l[0], self.l[2],
                            2*self.n[1]+1, 2*self.n[0]+1, 2*self.n[2]+1)
            induct = np.transpose(induct, (1, 0, 2))
        elif orientation == 'g':
            induct = genL3D(self.l[0], self.l[2], self.l[1],
                            2*self.n[0]+1, 2*self.n[2]+1, 2*self.n[1]+1)
            induct = np.transpose(induct, (0, 2, 1))
        induct /= self.lscale
        self.selfind = induct[0, 0, 0]
        self.induct = induct
        p2p_transfer = np.zeros((27, 2*self.n[0]-1, 2*self.n[1]-1,
                                 2*self.n[2]-1), dtype=np.complex128)
        self.p2p_transfer = np.zeros((27, 2*self.n[0], 2*self.n[1],
                                      2*self.n[2]), dtype=np.complex128)
        for neighx in range(-1, 2):
            if neighx == -1:
                xidx = np.r_[1:2*self.n[0]]
            elif neighx == 0:
                xidx = np.r_[self.n[0]-1:0:-1, :self.n[0]]
            elif neighx == 1:
                xidx = np.r_[2*self.n[0]-1:0:-1]
            xidx = np.repeat(xidx, (2*self.n[1]-1)*(2*self.n[2]-1))
            for neighy in range(-1, 2):
                if neighy == -1:
                    yidx = np.r_[1:2*self.n[1]]
                elif neighy == 0:
                    yidx = np.r_[self.n[1]-1:0:-1, :self.n[1]]
                elif neighy == 1:
                    yidx = np.r_[2*self.n[1]-1:0:-1]
                yidx = np.repeat(yidx, 2*self.n[2] - 1)
                yidx = np.tile(yidx, 2*self.n[0] - 1)
                for neighz in range(-1, 2):
                    if neighz == -1:
                        zidx = np.r_[1:2*self.n[2]]
                    elif neighz == 0:
                        zidx = np.r_[self.n[2]-1:0:-1, :self.n[2]]
                    elif neighz == 1:
                        zidx = np.r_[2*self.n[2]-1:0:-1]
                    zidx = np.tile(zidx, (2*self.n[0]-1)*(2*self.n[1]-1))
                    neigh = 9*(neighx+1) + 3*(neighy+1) + neighz + 1
                    p2p_transfer[neigh, ...] = \
                        np.reshape(induct[xidx, yidx, zidx],
                                   (2*self.n[0]-1, 2*self.n[1]-1, 2*self.n[2]-1))
                    self.p2ptrans0 = p2p_transfer
                    self.p2p_transfer[neigh, ...] = \
                        tp.t3multnonsym3(p2p_transfer[neigh, ...],
                                         self.n[0], self.n[1], self.n[2])

    def p2pcpu(self):
        """Apply the near-field partial-inductance operator (active CPU path).

        In-place near-field matvec: computes the contribution of every leaf
        box's filaments and those of its 26 neighbours to ``self.data``,
        overwriting ``self.data`` with the result. The domain is swept one
        x-slab at a time, keeping a rolling ``behind``/``current``/``ahead``
        window of forward-FFT'd source slabs; the Fortran kernel
        ``mp_fortran.p2p`` convolves each target group against its neighbours
        using ``self.p2p_transfer``, and the result is inverse-FFT'd back and
        scattered to ``self.data``. Bound to ``self.p2p`` by the constructor.
        """
        self.iternumber += 1
        # Per-slab scatter/gather index packs, built once: the old
        # per-group loops (zero + fancy scatter + reshape per box, and
        # the mirrored gather) were ~300 ms/matvec of pure Python on
        # square_coil -- 42% of the p2p stage. flatpos addresses the
        # (sizeslab, prod(n)) slab buffer; srcidx addresses self.data.
        # Same values to the same slots: bit-identical.
        if getattr(self, '_slab_pack', None) is None:
            nflat = int(np.prod(self.n))
            packs = []
            for cx in range(int(self.ng[0])):
                gs = self.slabidx[self.slabidx0[cx]:self.slabidx0[cx+1]]
                rows, cols, src = [], [], []
                for cg, group in enumerate(gs):
                    a, b = int(self.idx0[group]), int(self.idx0[group+1])
                    cols.append(self.idx[a:b].astype(np.intp))
                    src.append(np.arange(a, b, dtype=np.intp))
                    rows.append(np.full(b - a, cg, dtype=np.intp))
                if len(rows):
                    flatpos = (np.concatenate(rows)*nflat
                               + np.concatenate(cols))
                    srcidx = np.concatenate(src)
                else:
                    flatpos = np.zeros(0, dtype=np.intp)
                    srcidx = np.zeros(0, dtype=np.intp)
                packs.append((flatpos, srcidx))
            self._slab_pack = packs
            self._nflat = nflat
        aheadslab = tp.ToeplitzM2L(self.m, 1)
        aheadslab.c[...] = 0
        #                        dtype=np.complex128)
        currentslab = None
        for countx in range(-1, int(self.ng[0])):
            behindslab = currentslab
            currentslab = aheadslab
            if countx < self.ng[0]-1:
                sizeslab = self.slabidx0[countx+2] - self.slabidx0[countx+1]
                aheadslab = tp.ToeplitzM2L(self.m, sizeslab)
                aheadslab.a[:] = 0
                aheadslab.b[:] = 0
                aheadslab.c[:] = 0
                # one scatter for the whole slab (see _slab_pack above)
                flatpos, srcidx = self._slab_pack[countx+1]
                buf = np.zeros((sizeslab, self._nflat),
                               dtype=np.complex128)
                buf.ravel()[flatpos] = self.data[srcidx]
                aheadslab.a[:, :self.n[0], :self.n[1], :self.n[2]] = \
                    buf.reshape((sizeslab, self.n[0], self.n[1],
                                 self.n[2]))
                aheadslab.a = aheadslab.fftab()
                aheadslab.b[:, :, :self.m[1], :] = aheadslab.a
                aheadslab.b = aheadslab.fftbc()
                aheadslab.c[:, :, :, :self.m[2]] = aheadslab.b
                aheadslab.c = aheadslab.fftcc()
                # aheadslab[:, :, :self.m[1], :self.m[2]] = \
                #     np.fft.fft(aheadslab[:, :, :self.m[1], :self.m[2]],
                #                2*self.m[0], 1)
                # aheadslab[:, :, :, :self.m[2]] = \
                #     np.fft.fft(aheadslab[:, :, :, :self.m[2]],
                #                2*self.m[1], 2)
            if countx >= 0 and np.shape(currentslab.c)[0] > 0:
                sizeslab = self.slabidx0[countx+1] - self.slabidx0[countx]
                selfslabidx = self.slabidx[self.slabidx0[countx]:
                                           self.slabidx0[countx+1]]
                targetslab = tp.ToeplitzM2L(self.m, sizeslab)
                targetslab.c[...] = \
                    mp_fortran.p2p(behindslab.c.T, currentslab.c.T,
                                   aheadslab.c.T, selfslabidx, selfslabidx,
                                   countx, self.neighbors.T, self.xidx,
                                   self.p2p_transfer.T, self.revslabidx,
                                   self.revslabidx).T
                # targetslab.c[...] = self.p2pinner(behindslab.c, currentslab.c,
                #                                   aheadslab.c, selfslabidx,
                #                                   countx)
                targetslab.c = targetslab.ifftcc()
                targetslab.b[...] = targetslab.c[..., :self.m[2]]
                targetslab.b = targetslab.ifftcb()
                targetslab.a[...] = targetslab.b[..., :self.m[1], :]
                targetslab.a = targetslab.ifftba()
                #     group = self.slabidx[countg + self.slabidx0[countx]]
                #     for countn in range(27):
                #         neighgroup = self.neighbors[countn, group]
                #         if neighgroup >= 0:
                #             x = self.xidx[neighgroup] - countx
                #             trans = self.p2p_transfer[countn, :, :, :]
                #             yzidx = self.revslabidx[neighgroup]
                #             if x == -1:
                #                 targetslab[countg, ...] += \
                #                     trans*behindslab[yzidx, :, :, :]
                #             elif x == 0:
                #                 targetslab[countg, ...] += \
                #                     trans*currentslab[yzidx, :, :, :]
                #             elif x == 1:
                #                 targetslab[countg, ...] += \
                #                     trans*aheadslab[yzidx, :, :, :]
                #             else:
                #                 print("invalid slab!")
                ####################
                # targetslab[..., :self.m[2]] = \
                #     np.fft.ifft(targetslab[..., :self.m[2]], 2*self.m[1], 2)
                # targetslab[..., :self.m[1], :self.m[2]] = \
                #     np.fft.ifft(targetslab[..., :self.m[1], :self.m[2]],
                #                 2*self.m[0], 1)
                # one gather for the whole slab (see _slab_pack above)
                flatpos, srcidx = self._slab_pack[countx]
                buf = np.ascontiguousarray(
                    targetslab.a[:, :self.n[0], :self.n[1], :self.n[2]])
                self.data[srcidx] = buf.reshape(
                    (sizeslab, self._nflat)).ravel()[flatpos]

    def p2pgpu2(self):
        """CuPy near-field driver: the p2pcpu pipeline on the device.

        Same algorithm, same rolling behind/current/ahead x-slab window
        (which is also the STREAMING structure for problems whose
        vectors outgrow VRAM: the device working set is three slabs of
        spectra regardless of total size). Differences from p2pcpu:
        the staged padded FFTs collapse to one batched
        ``cupy.fft.fftn`` (mathematically identical), and the Fortran
        27-neighbour multiply-accumulate becomes a gathered elementwise
        product with ``cupyx.scatter_add``. complex128 throughout --
        the kernel is bandwidth-bound, so fp64 costs bandwidth only,
        and the operator keeps oracle-grade precision (cuFFT vs FFTW
        rounding differs at ~1e-15; not bit-identical to the CPU
        path). Opt-in via SPPEEC_GPU_P2P=1.
        """
        import cupy as cp
        import cupyx
        self.iternumber += 1
        if getattr(self, '_gpu_pack', None) is None:
            nflat = int(np.prod(self.n))
            n0, n1, n2 = (int(v) for v in self.n)
            packs = []
            for cx in range(int(self.ng[0])):
                gs = self.slabidx[self.slabidx0[cx]:self.slabidx0[cx+1]]
                rows, cols, src = [], [], []
                for cg, group in enumerate(gs):
                    a, b = int(self.idx0[group]), int(self.idx0[group+1])
                    cols.append(self.idx[a:b].astype(np.intp))
                    src.append(np.arange(a, b, dtype=np.intp))
                    rows.append(np.full(b - a, cg, dtype=np.intp))
                flatpos = ((np.concatenate(rows)*nflat
                            + np.concatenate(cols)) if rows
                           else np.zeros(0, dtype=np.intp))
                srcidx = (np.concatenate(src) if src
                          else np.zeros(0, dtype=np.intp))
                # 27-neighbour MAC pairs, split by source-slab offset
                mac = {-1: ([], [], []), 0: ([], [], []),
                       1: ([], [], [])}
                for cg, group in enumerate(gs):
                    for cn in range(27):
                        ngr = int(self.neighbors[cn, group])
                        if ngr < 0:
                            continue
                        dx = int(self.xidx[ngr]) - cx
                        t, p, tr = mac[dx]
                        t.append(cg)
                        p.append(int(self.revslabidx[ngr]))
                        tr.append(cn)
                packs.append(dict(
                    flatpos=cp.asarray(flatpos),
                    srcidx=cp.asarray(srcidx),
                    size=len(gs),
                    mac={dx: (cp.asarray(np.asarray(v[0], np.intp)),
                              cp.asarray(np.asarray(v[1], np.intp)),
                              cp.asarray(np.asarray(v[2], np.intp)))
                         for dx, v in mac.items()}))
            # The FFT grid is the TRANSFER'S grid, not 2*self.n:
            # the kernel table can be built on a larger reference
            # lattice than this leaf's own (non-cubic single-level
            # trees; found 2026-08-14 when the full suite first ran
            # under the GPU-p2p default). Convolving data of extent n
            # on the transfer grid is exact as long as the grid holds
            # 2n-1 alias-free offsets, asserted here; the Fortran CPU
            # kernel indexes the two extents independently and never
            # had the problem.
            S = tuple(int(v) for v in self.p2p_transfer.shape[1:])
            if any(S[i] < 2*int(self.n[i]) - 1 for i in range(3)):
                raise RuntimeError(
                    "p2p transfer grid %s too small for leaf lattice "
                    "%s" % (S, tuple(int(v) for v in self.n)))
            self._gpu_pack = dict(
                packs=packs, nflat=nflat,
                transfer=cp.asarray(self.p2p_transfer),
                shape=S)
        gp = self._gpu_pack
        cp_ = cp
        n0, n1, n2 = (int(v) for v in self.n)
        S = gp['shape']
        dev = cp_.asarray(self.data)
        out = cp_.empty_like(dev)

        def spec(cx):
            pk = gp['packs'][cx]
            if pk['size'] == 0:
                return cp_.zeros((0,) + S, dtype=cp_.complex128)
            buf = cp_.zeros((pk['size'], gp['nflat']),
                            dtype=cp_.complex128)
            buf.ravel()[pk['flatpos']] = dev[pk['srcidx']]
            pad = cp_.zeros((pk['size'],) + S, dtype=cp_.complex128)
            pad[:, :n0, :n1, :n2] = buf.reshape(pk['size'], n0, n1, n2)
            return cp_.fft.fftn(pad, axes=(1, 2, 3))

        slabs = {-1: None, 0: None, 1: spec(0)}
        for cx in range(int(self.ng[0])):
            slabs[-1], slabs[0] = slabs[0], slabs[1]
            slabs[1] = (spec(cx + 1) if cx + 1 < int(self.ng[0])
                        else cp_.zeros((0,) + S, dtype=cp_.complex128))
            pk = gp['packs'][cx]
            if pk['size'] == 0:
                continue
            tgt = cp_.zeros((pk['size'],) + S, dtype=cp_.complex128)
            for dx in (-1, 0, 1):
                t, p, tr = pk['mac'][dx]
                if t.size == 0 or slabs[dx] is None                         or slabs[dx].shape[0] == 0:
                    continue
                contrib = gp['transfer'][tr]*slabs[dx][p]
                cupyx.scatter_add(tgt.real, t, contrib.real)
                cupyx.scatter_add(tgt.imag, t, contrib.imag)
            res = cp_.fft.ifftn(tgt, axes=(1, 2, 3))[:, :n0, :n1, :n2]
            out[pk['srcidx']] = res.reshape(
                pk['size'], gp['nflat']).ravel()[pk['flatpos']]
        self.data[:] = cp_.asnumpy(out)

    def spaiinit(self):
        """Build the SPAI / RDF preconditioner for the iterative solver.

        Constructs a sparse approximate inverse of the system operator, stored
        as a 7-diagonal stencil ``self.Zinv`` (one column per filament, seven
        rows for the self and +/-x, +/-y, +/-z neighbours). For each leaf group
        it assembles the local dense system matrix ``Z`` from the neighbour
        inductance stencil (``coeff`` is the inductance table mirrored over the
        eight sign octants), adds the resistance term and the orientation-
        dependent RDF (Reluctance/Divergence-Free) correction, then solves a
        small least-squares problem (:func:`lsqr`) per filament to obtain that
        filament's inverse stencil.

        Requires ``self.jomega``, ``self.r``, ``self.alpha`` and ``self.beta``
        to be set. Populates ``self.Zinv``, consumed by :meth:`spaiapply`.
        """
        induct = self.induct
        coeff = np.zeros((4*self.m[0]+1, 4*self.m[1]+1, 4*self.m[2]+1),
                         dtype=induct.dtype)
        ncx = 2*self.m[0]
        ncy = 2*self.m[1]
        ncz = 2*self.m[2]
        coeff[ncx:, ncy:, ncz:] = induct
        coeff[ncx:, ncy:, ncz::-1] = induct
        coeff[ncx:, ncy::-1, ncz:] = induct
        coeff[ncx:, ncy::-1, ncz::-1] = induct
        coeff[ncx::-1, ncy:, ncz:] = induct
        coeff[ncx::-1, ncy:, ncz::-1] = induct
        coeff[ncx::-1, ncy::-1, ncz:] = induct
        coeff[ncx::-1, ncy::-1, ncz::-1] = induct
        struc = np.zeros((3*self.m[0], 3*self.m[1], 3*self.m[2]),
                         dtype=induct.dtype)
        singlegroup = np.zeros((np.prod(self.n),), dtype=induct.dtype)
        Zinv = np.zeros((7, self.idx0[-1]), dtype=self.data.dtype)
        for group in range(np.size(self.idx0)-1):
            struc[...] = 0
            for neigh in range(27):
                neighgroup = self.neighbors[neigh, group]
                if neighgroup >= 0:
                    singlegroup[:] = 0
                    neighidx0 = self.idx0[neighgroup]
                    neighidx1 = self.idx0[neighgroup+1]
                    iidx = self.idx[neighidx0:neighidx1]
                    singlegroup[iidx] = self.struc[neighidx0:neighidx1]
                    neighx = neigh//9
                    neighy = (neigh//3) % 3
                    neighz = neigh % 3
                    startx = neighx*self.m[0]
                    starty = neighy*self.m[1]
                    startz = neighz*self.m[2]
                    stopx = startx + self.n[0]
                    stopy = starty + self.n[1]
                    stopz = startz + self.n[2]
                    struc[startx:stopx, starty:stopy, startz:stopz] = \
                        np.reshape(singlegroup, (self.n[0], self.n[1],
                                                 self.n[2]))
            totalsize = np.count_nonzero(struc)
            innerx = np.s_[self.m[0]-1:2*self.m[0]+1]
            innery = np.s_[self.m[1]-1:2*self.m[1]+1]
            innerz = np.s_[self.m[2]-1:2*self.m[2]+1]
            innerstruc = struc[innerx, innery, innerz]
            innersize = np.count_nonzero(innerstruc)
            Z = np.zeros((innersize, totalsize), dtype=self.data.dtype)
            totalnnz = np.flatnonzero(struc)
            strucidx = np.empty_like(struc, dtype=int)
            strucidx[...] = -1
            strucidx[np.nonzero(struc)] = np.r_[:totalsize]
            innerstrucidx = np.empty_like(innerstruc, dtype=int)
            innerstrucidx[...] = -1
            innerstrucidx[np.nonzero(innerstruc)] = np.r_[:innersize]
            inner = 0
            for filx in range(-1, self.m[0]+1):
                for fily in range(-1, self.m[1]+1):
                    for filz in range(-1, self.m[2]+1):
                        fil = struc[self.m[0]+filx, self.m[1]+fily,
                                    self.m[2]+filz]
                        if fil > 0:
                            xstart = self.m[0] - filx
                            ystart = self.m[1] - fily
                            zstart = self.m[2] - filz
                            xstop = xstart + 3*self.m[0]
                            ystop = ystart + 3*self.m[1]
                            zstop = zstart + 3*self.m[2]
                            xmid = 2*self.m[0] - xstart
                            ymid = 2*self.m[1] - ystart
                            zmid = 2*self.m[2] - zstart
                            Ztemp = struc*self.jomega * \
                                coeff[xstart:xstop, ystart:ystop, zstart:zstop]
                            Ztemp[xmid, ymid, zmid] += self.r
                            # Here we add dot(Bi, Ci) for i={e, f, g}
                            # as prescribed by the RDF preconditioner.
                            # beta is the scaling factor that is applied
                            # to the row with C and D (D=0) and the column
                            # B and D, so the product of B and C is scaled
                            # by beta**2.
                            scaling_factor = 1/self.alpha*self.beta**2
                            Ztemp[xmid, ymid, zmid] -= 2*scaling_factor
                            if self.orientation == 'e':
                                Ztemp[xmid, ymid-1, zmid] += scaling_factor
                                Ztemp[xmid, ymid+1, zmid] += scaling_factor
                            elif self.orientation == 'f':
                                Ztemp[xmid-1, ymid, zmid] += scaling_factor
                                Ztemp[xmid+1, ymid, zmid] += scaling_factor
                            elif self.orientation == 'g':
                                Ztemp[xmid, ymid, zmid-1] += scaling_factor
                                Ztemp[xmid, ymid, zmid+1] += scaling_factor
                            # end previous comment
                            Z[inner, :] = np.ravel(Ztemp)[totalnnz]
                            inner += 1
            gidx = self.idx0[group]
            e = np.zeros((totalsize,), dtype=np.complex128)
            inner = 0
            for filx in range(self.m[0]):
                fx = filx + 1
                fox = filx + self.m[0]
                for fily in range(self.m[1]):
                    fy = fily + 1
                    foy = fily + self.m[1]
                    for filz in range(self.m[2]):
                        fz = filz + 1
                        foz = filz + self.m[2]
                        if innerstruc[fx, fy, fz] > 0:
                            i7 = np.zeros((7,), dtype=int)
                            i7[0] = innerstrucidx[fx-1, fy, fz]
                            i7[1] = innerstrucidx[fx, fy-1, fz]
                            i7[2] = innerstrucidx[fx, fy, fz-1]
                            i7[3] = innerstrucidx[fx, fy, fz]
                            i7[4] = innerstrucidx[fx, fy, fz+1]
                            i7[5] = innerstrucidx[fx, fy+1, fz]
                            i7[6] = innerstrucidx[fx+1, fy, fz]
                            o3 = strucidx[fox, foy, foz]
                            i7nnz = np.nonzero(i7 + 1)[0]
                            Zview = Z[i7[i7nnz], :]
                            e[:] = 0
                            e[o3] = 1
                            ZinvCol = lsqr(Zview.T, e, iter_lim=3)
                            idx = gidx + inner
                            Zinv[i7nnz, idx] = ZinvCol[0]
                            inner += 1
        self.Zinv = Zinv

    def spaistruc(self, group, struc):
        """Reassemble one group's 3x3x3-neighbour occupancy block.

        Helper that scatters the occupancy of `group` and its 26 neighbours
        into the padded array `struc`, then builds the index map
        `innerstrucidx` from the flattened structure into the inner block ---
        the same setup :meth:`spaiinit` does inline. Returns ``(struc,
        innerstrucidx)``.

        Note: this method references a local ``singlegroup`` that it never
        defines, so it would raise :class:`NameError` if called; it is not on
        the active path (:meth:`spaiinit` performs the assembly itself).
        """
        struc[...] = 0
        for neigh in range(27):
            neighgroup = self.neighbors[neigh, group]
            if neighgroup >= 0:
                singlegroup[:] = 0
                neighidx0 = self.idx0[neighgroup]
                neighidx1 = self.idx0[neighgroup+1]
                iidx = self.idx[neighidx0:neighidx1]
                singlegroup[iidx] = self.struc[neighidx0:neighidx1]
                neighx = neigh//9
                neighy = (neigh//3) % 3
                neighz = neigh % 3
                startx = neighx*self.m[0]
                starty = neighy*self.m[1]
                startz = neighz*self.m[2]
                stopx = startx + self.n[0]
                stopy = starty + self.n[1]
                stopz = startz + self.n[2]
                struc[startx:stopx, starty:stopy, startz:stopz] = \
                    np.reshape(singlegroup, (self.n[0], self.n[1],
                                             self.n[2]))
        totalsize = np.count_nonzero(struc)
        innerx = np.s_[self.m[0]-1:2*self.m[0]+1]
        innery = np.s_[self.m[1]-1:2*self.m[1]+1]
        innerz = np.s_[self.m[2]-1:2*self.m[2]+1]
        innerstruc = struc[innerx, innery, innerz]
        innersize = np.count_nonzero(innerstruc)
        Z = np.zeros((innersize, totalsize), dtype=self.data.dtype)
        strucidx = np.empty_like(struc, dtype=int)
        strucidx[...] = -1
        strucidx[np.nonzero(struc)] = np.r_[:totalsize]
        innerstrucidx = np.empty_like(innerstruc, dtype=int)
        innerstrucidx[...] = -1
        innerstrucidx[np.nonzero(innerstruc)] = np.r_[:innersize]
        return struc, innerstrucidx

    def spaiapply(self, vec):
        """Apply the SPAI / RDF preconditioner to a vector.

        Multiplies `vec` by the sparse approximate inverse built in
        :meth:`spaiinit`. For each leaf group it scatters the group and its
        neighbours into a padded workspace, applies the 7-point ``self.Zinv``
        stencil (self and +/-x, +/-y, +/-z), and gathers the result back into
        the output. Returns the preconditioned vector (same shape as `vec`).

        Parameters
        ----------
        vec : ndarray
            Input vector in the filament ordering used by ``self.data``.

        Returns
        -------
        ndarray
            The preconditioned vector ``M^{-1} @ vec``.
        """
        n = self.n
        singlegroup = np.zeros((3*n[0], 3*n[1], 3*n[2]+2), dtype=vec.dtype)
        singleZinv = np.zeros((7, n[0], n[1], n[2]), dtype=vec.dtype)
        outmat = np.zeros_like(vec)
        for group in range(np.size(self.idx0) - 1):
            singlegroup[...] = 0
            for neigh in range(27):
                neighx = neigh // 9
                neighy = (neigh//3) % 3
                neighz = neigh % 3
                neighgroup = self.neighbors[neigh, group]
                nidx = np.s_[self.idx0[neighgroup]:self.idx0[neighgroup+1]]
                xidx = neighx*n[0] + self.idx[nidx]//(n[1]*n[2])
                yidx = neighy*n[1] + (self.idx[nidx]//n[2]) % n[1]
                zidx = neighz*n[2] + self.idx[nidx] % n[2]
                singlegroup[xidx, yidx, zidx] = vec[nidx]
            gidx = np.s_[self.idx0[group]:self.idx0[group+1]]
            fidx = self.idx[gidx]
            xidx = fidx // (n[1]*n[2])
            yidx = (fidx // n[2]) % n[1]
            zidx = fidx % n[2]
            singleZinv[...] = 0
            singleZinv[:, xidx, yidx, zidx] = self.Zinv[:, gidx]
            outfull = singlegroup[n[0]-1:2*n[0]-1, n[1]:2*n[1], n[2]:2*n[2]] \
                * singleZinv[0, ...]
            outmat[gidx] = np.ravel(outfull)[fidx]
            outfull = singlegroup[n[0]:2*n[0], n[1]-1:2*n[1]-1, n[2]:2*n[2]] \
                * singleZinv[1, ...]
            outmat[gidx] += np.ravel(outfull)[fidx]
            outfull = singlegroup[n[0]:2*n[0], n[1]:2*n[1], n[2]-1:2*n[2]-1] \
                * singleZinv[2, ...]
            outmat[gidx] += np.ravel(outfull)[fidx]
            outfull = singlegroup[n[0]:2*n[0], n[1]:2*n[1], n[2]:2*n[2]] \
                * singleZinv[3, ...]
            outmat[gidx] += np.ravel(outfull)[fidx]
            outfull = singlegroup[n[0]:2*n[0], n[1]:2*n[1], n[2]+1:2*n[2]+1] \
                * singleZinv[4, ...]
            outmat[gidx] += np.ravel(outfull)[fidx]
            outfull = singlegroup[n[0]:2*n[0], n[1]+1:2*n[1]+1, n[2]:2*n[2]] \
                * singleZinv[5, ...]
            outmat[gidx] += np.ravel(outfull)[fidx]
            outfull = singlegroup[n[0]+1:2*n[0]+1, n[1]:2*n[1], n[2]:2*n[2]] \
                * singleZinv[6, ...]
            outmat[gidx] += np.ravel(outfull)[fidx]
        return outmat


